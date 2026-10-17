"""
Corpus store: SQLite is the single source of truth.

FAISS and BM25 are derived, in-memory caches rebuilt from this store whenever
`corpus_version()` changes. There are no pickle files, so there is nothing to go
stale and nothing to deserialize unsafely. Embeddings are stored as float32
blobs, so rebuilding an index never re-runs the embedding model.
"""
import json
import logging
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from docmind.config import CONFIG
from docmind.db import connect
from docmind.models import Chunk

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    file_id      TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    source       TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    n_units      INTEGER NOT NULL,
    n_chunks     INTEGER NOT NULL,
    ingested_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    file_id      TEXT NOT NULL REFERENCES documents(file_id) ON DELETE CASCADE,
    doc_id       TEXT NOT NULL,
    source       TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    ord          INTEGER NOT NULL,
    content      TEXT NOT NULL,
    metadata     TEXT NOT NULL,
    embedding    BLOB NOT NULL,
    UNIQUE(file_id, ord)
);
CREATE INDEX IF NOT EXISTS idx_chunks_file_ord ON chunks(file_id, ord);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class EmbeddingModelMismatch(RuntimeError):
    pass


def _row_to_chunk(row) -> Chunk:
    return Chunk(
        content=row["content"],
        doc_id=row["doc_id"],
        source=row["source"],
        chunk_index=row["chunk_index"],
        metadata=json.loads(row["metadata"]),
        chunk_id=row["chunk_id"],
        file_id=row["file_id"],
        ord=row["ord"],
    )


class CorpusStore:
    def __init__(self, db_path=None):
        self.db_path = db_path
        with connect(self.db_path) as conn:
            conn.executescript(SCHEMA)

    # ---- meta / versioning -------------------------------------------------
    def _get_meta(self, conn, key: str, default: str = "") -> str:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def _set_meta(self, conn, key: str, value: str) -> None:
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    def _bump(self, conn) -> None:
        v = int(self._get_meta(conn, "corpus_version", "0")) + 1
        self._set_meta(conn, "corpus_version", str(v))

    def version(self) -> int:
        with connect(self.db_path) as conn:
            return int(self._get_meta(conn, "corpus_version", "0"))

    def _embedding_signature(self) -> str:
        e = CONFIG.embedding
        return f"{e.backend}:{e.model_name}"

    # ---- documents ---------------------------------------------------------
    def has_file(self, file_id: str) -> bool:
        with connect(self.db_path) as conn:
            return conn.execute("SELECT 1 FROM documents WHERE file_id=?", (file_id,)).fetchone() is not None

    def get_document(self, file_id: str) -> Optional[Dict]:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM documents WHERE file_id=?", (file_id,)).fetchone()
            return dict(row) if row else None

    def list_documents(self) -> List[Dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY ingested_at DESC").fetchall()
            return [dict(r) for r in rows]

    def add_document(self, doc: Dict, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        sig = self._embedding_signature()
        with connect(self.db_path) as conn:
            existing_sig = self._get_meta(conn, "embedding_signature", "")
            n_existing = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            if existing_sig and existing_sig != sig and n_existing > 0:
                raise EmbeddingModelMismatch(
                    f"Corpus was embedded with '{existing_sig}' but the active model is '{sig}'. "
                    "Delete indexes/docmind.db (or re-ingest everything) to switch embedding models."
                )
            self._set_meta(conn, "embedding_signature", sig)
            conn.execute(
                "INSERT OR REPLACE INTO documents(file_id, filename, source, sha256, n_units, n_chunks, ingested_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc["file_id"], doc["filename"], doc["source"], doc["sha256"],
                 doc["n_units"], len(chunks), time.time()),
            )
            conn.executemany(
                "INSERT INTO chunks(chunk_id, file_id, doc_id, source, chunk_index, ord, content, metadata, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (c.chunk_id, c.file_id, c.doc_id, c.source, c.chunk_index, c.ord, c.content,
                     json.dumps(c.metadata), np.asarray(e, dtype="float32").tobytes())
                    for c, e in zip(chunks, embeddings)
                ],
            )
            self._bump(conn)

    def delete_document(self, file_id: str) -> bool:
        with connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM documents WHERE file_id=?", (file_id,))
            if cur.rowcount:
                self._bump(conn)
            return cur.rowcount > 0

    # ---- reads -------------------------------------------------------------
    def load_corpus(self, file_ids: Optional[Iterable[str]] = None) -> Tuple[List[Chunk], np.ndarray]:
        """All chunks (ordered by file, position) and their embedding matrix."""
        sql = "SELECT * FROM chunks"
        params: Tuple = ()
        if file_ids:
            ids = list(file_ids)
            sql += f" WHERE file_id IN ({','.join('?' * len(ids))})"
            params = tuple(ids)
        sql += " ORDER BY file_id, ord"
        with connect(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        chunks = [_row_to_chunk(r) for r in rows]
        if not rows:
            return [], np.zeros((0, 0), dtype="float32")
        matrix = np.vstack([np.frombuffer(r["embedding"], dtype="float32") for r in rows])
        return chunks, np.ascontiguousarray(matrix)

    def get_chunk(self, chunk_id_or_prefix: str) -> Optional[Chunk]:
        key = chunk_id_or_prefix.strip()
        if not key:
            return None
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM chunks WHERE chunk_id=?", (key,)).fetchone()
            if row is None:
                safe = key.replace("%", "").replace("_", "")
                row = conn.execute(
                    "SELECT * FROM chunks WHERE chunk_id LIKE ? ORDER BY chunk_id LIMIT 1", (safe + "%",)
                ).fetchone()
            return _row_to_chunk(row) if row else None

    def neighbors(self, chunk: Chunk, window: int = 1) -> List[Chunk]:
        """Chunks around `chunk` in document order (same file), inclusive."""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE file_id=? AND ord BETWEEN ? AND ? ORDER BY ord",
                (chunk.file_id, chunk.ord - window, chunk.ord + window),
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]
