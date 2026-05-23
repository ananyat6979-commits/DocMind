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
from config import CONFIG

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
    Full ingestion pipeline for a file or directory.

    Steps:
      1. Load raw Documents.
      2. Chunk each Document semantically.
      3. Save chunks to SQLite.
      4. Build/update the FAISS and BM25 indexes (done by retrieval layer).

    Returns the list of all Chunk objects ingested in this run.
    """
    path = Path(path)

    # Load
    logger.info(f"Loading documents from: {path}")
    if path.is_dir():
        documents = load_directory(path)
    else:
        documents = load_document(path)

    if not documents:
        logger.warning("No documents loaded — check file format and content")
        return []

    # Chunk
    all_chunks: List[Chunk] = []
    for doc in tqdm(documents, desc="Chunking", unit="doc"):
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    logger.info(f"Produced {len(all_chunks)} chunks from {len(documents)} documents")

    # Persist chunk metadata to SQLite
    conn = _init_db()
    _save_chunks(conn, all_chunks)
    conn.close()

    # Build retrieval indexes
    # Import here to avoid circular imports
    from docmind.retrieval.hybrid import HybridRetriever
    retriever = HybridRetriever()
    retriever.build_index(all_chunks, force_rebuild=rebuild_index)

    logger.info("Ingestion complete.")
    return all_chunks