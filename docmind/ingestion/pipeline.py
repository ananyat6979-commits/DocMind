"""
End-to-end ingestion: load -> chunk -> embed -> store.

Design (fixes the old "second upload silently reuses the first file's index" bug):
  * Every file gets a stable file_id = sha256(bytes)[:16].
  * Same bytes again  -> no-op (reported as cached). Different bytes -> added to
    the corpus alongside existing documents. Nothing is ever skipped just
    because "an index exists".
  * rebuild=True deletes that file's rows and re-ingests it.
  * Indexes are derived from the store, so they cannot disagree with it.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from tqdm import tqdm

from docmind.config import CONFIG
from docmind.embeddings import embed_texts
from docmind.ingestion.chunker import chunk_document
from docmind.ingestion.loader import (
    EmptyDocumentError, LOADERS, file_sha256, iter_supported_files, load_document,
)
from docmind.models import Chunk
from docmind.store import CorpusStore

logger = logging.getLogger(__name__)


@dataclass
class FileReport:
    file_id: str
    filename: str
    status: str            # "ingested" | "cached" | "reingested" | "failed"
    n_chunks: int = 0
    error: str = ""


@dataclass
class IngestReport:
    files: List[FileReport] = field(default_factory=list)

    @property
    def file_ids(self) -> List[str]:
        return [f.file_id for f in self.files if f.status != "failed"]

    @property
    def chunks_added(self) -> int:
        return sum(f.n_chunks for f in self.files if f.status in ("ingested", "reingested"))

    @property
    def n_cached(self) -> int:
        return sum(1 for f in self.files if f.status == "cached")

    @property
    def failures(self) -> List[FileReport]:
        return [f for f in self.files if f.status == "failed"]


def _ingest_file(path: Path, store: CorpusStore, rebuild: bool) -> FileReport:
    sha = file_sha256(path)
    file_id = sha[:16]

    existed = store.has_file(file_id)
    if existed and not rebuild:
        doc = store.get_document(file_id)
        logger.info(f"'{path.name}' already ingested (file_id={file_id}); skipping")
        return FileReport(file_id, path.name, "cached", doc["n_chunks"] if doc else 0)

    units = load_document(path)
    if not units:
        raise EmptyDocumentError(
            f"No extractable text in '{path.name}'. If this is a scanned PDF, run OCR first "
            "(e.g. `ocrmypdf in.pdf out.pdf`)."
        )

    chunks: List[Chunk] = []
    for unit in tqdm(units, desc=f"Chunking {path.name}", unit="unit", leave=False):
        for c in chunk_document(unit):
            c.file_id = file_id
            c.ord = len(chunks)           # global position within the file
            c.metadata = {**c.metadata, "file_id": file_id}
            chunks.append(c)

    if not chunks:
        raise EmptyDocumentError(f"'{path.name}' produced zero chunks.")

    embeddings = embed_texts([c.content for c in chunks], show_progress=len(chunks) > 64)

    if existed:
        store.delete_document(file_id)
    store.add_document(
        {"file_id": file_id, "filename": path.name, "source": str(path.resolve()),
         "sha256": sha, "n_units": len(units)},
        chunks, embeddings,
    )
    logger.info(f"Ingested '{path.name}': {len(units)} units -> {len(chunks)} chunks (file_id={file_id})")
    return FileReport(file_id, path.name, "reingested" if existed else "ingested", len(chunks))


def ingest(
    path: Union[str, Path],
    rebuild_index: bool = False,
    store: Optional[CorpusStore] = None,
) -> IngestReport:
    """Ingest a file or a directory (recursively). Safe to call repeatedly."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    store = store or CorpusStore()
    report = IngestReport()

    files = list(iter_supported_files(path)) if path.is_dir() else [path]
    if path.is_file() and path.suffix.lower() not in LOADERS:
        raise ValueError(f"Unsupported extension: {path.suffix}. Supported: {sorted(LOADERS)}")

    for f in files:
        try:
            report.files.append(_ingest_file(f, store, rebuild_index))
        except Exception as exc:
            logger.warning(f"Failed to ingest {f.name}: {exc}")
            if path.is_file():
                raise
            report.files.append(FileReport("", f.name, "failed", error=str(exc)))
    return report
