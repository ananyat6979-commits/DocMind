"""
Corpus store: SQLite is the single source of truth.

FAISS and BM25 are derived, in-memory caches rebuilt from this store whenever
`corpus_version()` changes. There are no pickle files, so there is nothing to go
stale and nothing to deserialize unsafely. Embeddings are stored as float32
blobs, so rebuilding an index never re-runs the embedding model.
"""
import json
import re
import logging
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from docmind.config import CONFIG
from docmind.db import connect
from docmind.embeddings import compatible_embedding_identity
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
    ingested_at  REAL NOT NULL,
    lineage      TEXT NOT NULL DEFAULT '{}'
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
            columns={row["name"] for row in conn.execute("PRAGMA table_info(documents)")}
            if "lineage" not in columns:
                conn.execute("ALTER TABLE documents ADD COLUMN lineage TEXT NOT NULL DEFAULT '{}'")

    # ---- meta / versioning -------------------------------------------------
    def _get_meta(self, conn, key: str, default: str = "") -> str:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def _set_meta(self, conn, key: str, value: str) -> None:
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    def _bump(self, conn) -> None:
        from docmind.postgres import Connection
        if isinstance(conn,Connection):
            conn.execute("INSERT INTO meta(key,value) VALUES ('corpus_version','1') ON CONFLICT(tenant_id,key) DO UPDATE SET value=(meta.value::bigint+1)::text")
            return
        v = int(self._get_meta(conn, "corpus_version", "0")) + 1
        self._set_meta(conn, "corpus_version", str(v))

    def version(self) -> int:
        with connect(self.db_path) as conn:
            return int(self._get_meta(conn, "corpus_version", "0"))

    def _embedding_signature(self) -> str:
        e = CONFIG.embedding
        from docmind.embeddings import embedding_identity
        return embedding_identity()

    def ensure_embedding_compatible(self):
        with connect(self.db_path) as conn:
            signature=self._get_meta(conn,"embedding_signature","")
            count=conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if count and not compatible_embedding_identity(signature, self._embedding_signature()):
                raise EmbeddingModelMismatch("The corpus embedding identity differs from the active configuration; rebuild an isolated corpus")

    def content_identity(self) -> str:
        """Hash the actual checkpoint inputs, including tool-visible chunk identities."""
        import hashlib
        identity=hashlib.sha256()
        from docmind.postgres import DatabaseScope
        if isinstance(self.db_path,DatabaseScope):identity.update(b"tenant\0"+self.db_path.tenant.encode()+b"\0")
        with connect(self.db_path) as conn:
            for row in conn.execute("SELECT file_id,sha256,lineage FROM documents ORDER BY file_id"):
                identity.update(json.dumps(list(row),ensure_ascii=True).encode())
            for row in conn.execute("SELECT chunk_id,file_id,content,metadata,embedding FROM chunks ORDER BY chunk_id"):
                identity.update(json.dumps(list(row)[:-1],ensure_ascii=True).encode())
                identity.update(row[-1])
        return identity.hexdigest()

    # ---- documents ---------------------------------------------------------
    def has_file(self, file_id: str) -> bool:
        with connect(self.db_path) as conn:
            return conn.execute("SELECT 1 FROM documents WHERE file_id=?", (file_id,)).fetchone() is not None

    def get_document(self, file_id: str) -> Optional[Dict]:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM documents WHERE file_id=?", (file_id,)).fetchone()
            return {**dict(row),"lineage":json.loads(row["lineage"])} if row else None

    def list_documents(self) -> List[Dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY ingested_at DESC").fetchall()
            return [{**dict(r),"lineage":json.loads(r["lineage"])} for r in rows]

    def document_context(self, file_ids=None) -> Dict:
        """Fetch file-level quality once, without repeating it in every chunk."""
        ids = tuple(sorted(set(file_ids))) if file_ids is not None else None
        if ids == ():return {"partial": False, "languages": [], "documents": []}
        version = self.version()
        key = (version, ids)
        cached = getattr(self, "_document_context_cache", {})
        if key in cached:return cached[key]
        sql = "SELECT file_id,filename,lineage FROM documents"
        params = ()
        if ids is not None:
            sql += " WHERE file_id IN (" + ",".join("?" for _ in ids) + ")"
            params = ids
        documents = []
        with connect(self.db_path) as connection:
            for row in connection.execute(sql, params):
                lineage = json.loads(row["lineage"])
                language = lineage.get("language_info", {}).get("language", "und")
                if language == "und" and lineage.get("schema_version") != 2:
                    weights: dict[str, int] = {}
                    for old in connection.execute("SELECT metadata FROM chunks WHERE file_id=?", (row["file_id"],)):
                        metadata = json.loads(old["metadata"])
                        hint = metadata.get("language_info", {}).get("language", "und")
                        weights[hint] = weights.get(hint, 0) + metadata.get("char_count", 1)
                    total = sum(weights.values())
                    best = max(weights, key=lambda k: weights[k]) if weights else "und"
                    if total and weights[best] / total >= .8:language = best
                missing = [record.get("page") for record in lineage.get("page_quality", [])
                           if record.get("quarantined") and record.get("page") is not None]
                documents.append({"file_id": row["file_id"], "filename": row["filename"],
                    "language": language, "partial": bool(lineage.get("partial_document")),
                    "quarantined_pages": sorted(set(missing))})
        result = {"partial": any(d["partial"] for d in documents),
                  "languages": [d["language"] for d in documents], "documents": documents}
        if len(cached) >= 8:cached = {}
        cached[key] = result
        self._document_context_cache = cached
        return result

    def add_document(self, doc: Dict, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        sig = self._embedding_signature()
        with connect(self.db_path) as conn:
            from docmind.postgres import Connection
            if isinstance(conn,Connection):
                conn.corpus_lock()
                if embeddings.ndim!=2 or embeddings.shape[1]!=384:raise ValueError("Production vector schema requires 384-dimensional embeddings")
                if not np.isfinite(embeddings).all():raise ValueError("Nonfinite embeddings")
                if np.any(np.abs(np.linalg.norm(embeddings,axis=1)-1)>.002):raise ValueError("Normalized embeddings required")
            existing_sig = self._get_meta(conn, "embedding_signature", "")
            n_existing = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            if n_existing > 0 and not compatible_embedding_identity(existing_sig, sig):
                raise EmbeddingModelMismatch(
                    f"Corpus was embedded with '{existing_sig}' but the active model is '{sig}'. "
                    "Build a separate corpus from retained uploads; keep the original database and review history."
                )
            dimension=int(embeddings.shape[1])
            prior_dimension=self._get_meta(conn,"embedding_dimension","")
            if n_existing and prior_dimension and int(prior_dimension)!=dimension:
                raise EmbeddingModelMismatch("Embedding dimensions changed; rebuild an isolated corpus")
            self._set_meta(conn,"embedding_dimension",str(dimension))
            if not n_existing:self._set_meta(conn, "embedding_signature", sig)
            conn.execute("DELETE FROM documents WHERE file_id=?", (doc["file_id"],))
            conn.execute(
                "INSERT OR REPLACE INTO documents(file_id, filename, source, sha256, n_units, n_chunks, ingested_at, lineage)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (doc["file_id"], doc["filename"], doc["source"], doc["sha256"],
                 doc["n_units"], len(chunks), time.time(),json.dumps(doc.get("lineage",{}))),
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
            from docmind.postgres import Connection
            if isinstance(conn,Connection):conn.corpus_lock()
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
            if row is None and re.fullmatch(r"[a-fA-F0-9-]{8,36}", key):
                rows = conn.execute(
                    "SELECT * FROM chunks WHERE chunk_id >= ? AND chunk_id < ? ORDER BY chunk_id LIMIT 2",
                    (key, key + "\uffff"),
                ).fetchall()
                row = rows[0] if len(rows) == 1 else None
            return _row_to_chunk(row) if row else None

    def neighbors(self, chunk: Chunk, window: int = 1) -> List[Chunk]:
        """Chunks around `chunk` in document order (same file), inclusive."""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE file_id=? AND ord BETWEEN ? AND ? ORDER BY ord",
                (chunk.file_id, chunk.ord - window, chunk.ord + window),
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]
