"""
End-to-end ingestion pipeline.

Orchestrates: load → chunk → embed → index → persist.
This is the entry point for adding documents to DocMind.
"""
import logging
import sqlite3
import pickle
import json
from pathlib import Path
from typing import List, Union
from tqdm import tqdm

from docmind.models import Document, Chunk
from docmind.ingestion.loader import load_document, load_directory
from docmind.ingestion.chunker import chunk_document
from docmind.config import CONFIG

logger = logging.getLogger(__name__)


def _init_db() -> sqlite3.Connection:
    """
    SQLite for document metadata — no Docker, no Postgres needed.
    We store chunks as JSON blobs so we can reconstruct them at query time
    without keeping everything in RAM.
    """
    conn = sqlite3.connect(str(CONFIG.db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    TEXT PRIMARY KEY,
            doc_id      TEXT NOT NULL,
            source      TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content     TEXT NOT NULL,
            metadata    TEXT NOT NULL   -- JSON
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks(doc_id)")
    conn.commit()
    return conn


def _save_chunks(conn: sqlite3.Connection, chunks: List[Chunk]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO chunks
            (chunk_id, doc_id, source, chunk_index, content, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (c.chunk_id, c.doc_id, c.source, c.chunk_index,
             c.content, json.dumps(c.metadata))
            for c in chunks
        ]
    )
    conn.commit()


def load_all_chunks(conn: sqlite3.Connection) -> List[Chunk]:
    """Reconstruct Chunk objects from the SQLite store."""
    rows = conn.execute(
        "SELECT chunk_id, doc_id, source, chunk_index, content, metadata FROM chunks"
    ).fetchall()
    return [
        Chunk(
            content=row[4],
            doc_id=row[1],
            source=row[2],
            chunk_index=row[3],
            chunk_id=row[0],
            metadata=json.loads(row[5]),
        )
        for row in rows
    ]


def ingest(
    path: Union[str, Path],
    rebuild_index: bool = False,
) -> List[Chunk]:
    """
    Full ingestion pipeline. Skips chunking if indexes already exist
    and rebuild_index=False — this makes repeated runs instant.
    """
    from docmind.retrieval.dense import FAISS_INDEX_PATH
    from docmind.retrieval.hybrid import HybridRetriever

    path = Path(path)

    # Fast path: indexes exist and we're not rebuilding
    if not rebuild_index and FAISS_INDEX_PATH.exists():
        logger.info("Indexes already exist. Skipping ingestion. Use --rebuild to re-ingest.")
        conn = _init_db()
        existing_chunks = load_all_chunks(conn)
        conn.close()
        return existing_chunks

    # Load
    logger.info(f"Loading documents from: {path}")
    documents = load_directory(path) if path.is_dir() else load_document(path)

    if not documents:
        logger.warning("No documents loaded.")
        return []

    # Chunk
    all_chunks: List[Chunk] = []
    for doc in tqdm(documents, desc="Chunking", unit="doc"):
        all_chunks.extend(chunk_document(doc))

    logger.info(f"Produced {len(all_chunks)} chunks from {len(documents)} documents")

    # Persist
    conn = _init_db()
    _save_chunks(conn, all_chunks)
    conn.close()

    # Build indexes
    retriever = HybridRetriever()
    retriever.build_index(all_chunks, force_rebuild=True)

    logger.info("Ingestion complete.")
    return all_chunks
