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

import importlib.metadata
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

from docmind.config import CONFIG
from docmind.db import connect
from docmind.embeddings import embed_texts
from docmind.ingestion.chunker import chunk_document
from docmind.ingestion.loader import (
    LOADERS,
    EmptyDocumentError,
    file_sha256,
    iter_supported_files,
    load_document,
)
from docmind.models import Chunk
from docmind.store import CorpusStore

logger = logging.getLogger(__name__)


@dataclass
class FileReport:
    file_id: str
    filename: str
    status: str  # "ingested" | "cached" | "reingested" | "failed"
    n_chunks: int = 0
    error: str = ""
    stored_filename: str = ""
    quality: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestReport:
    files: List[FileReport] = field(default_factory=list)

    @property
    def file_ids(self) -> List[str]:
        return [f.file_id for f in self.files if f.status != "failed"]

    @property
    def chunks_added(self) -> int:
        return sum(
            f.n_chunks for f in self.files if f.status in ("ingested", "reingested")
        )

    @property
    def n_cached(self) -> int:
        return sum(1 for f in self.files if f.status == "cached")

    @property
    def failures(self) -> List[FileReport]:
        return [f for f in self.files if f.status == "failed"]


def quality_summary(lineage):
    records = lineage.get("page_quality", [])
    return {
        "partial": bool(lineage.get("partial_document")),
        "quarantined_pages": [
            r["page"]
            for r in records
            if r.get("quarantined") and r.get("page") is not None
        ],
        "readable_units": sum(not r.get("quarantined") for r in records),
        "total_units": len(records),
    }


def _ingest_file(
    path: Path,
    store: CorpusStore,
    rebuild: bool,
    chunker_config=None,
    expected_language=None,
    encoding=None,
) -> FileReport:
    from docmind.ingestion.quality import VERSION as quality_version
    from docmind.language import VERSION as language_version
    from docmind.text import VERSION as boundary_version

    preprocessing = {
        "boundaries": boundary_version,
        "extraction_quality": quality_version,
        "language": language_version,
        "expected_language": expected_language,
        "encoding": encoding,
    }
    if path.suffix.lower() == ".pdf":
        preprocessing["pdf_layout"] = "pdf-layout-v1"
    sha = file_sha256(path)
    file_id = sha[:16]

    existed = store.has_file(file_id)
    if chunker_config is None:
        import os

        from docmind.config import ChunkerConfig

        if existed:
            previous = store.get_document(file_id)
            recipe = (previous or {}).get("lineage", {}).get("chunker")
            if recipe:
                chunker_config = ChunkerConfig(
                    **{**asdict(CONFIG.chunker), "strategy": recipe["strategy"]}
                )
        elif not os.getenv("DOCMIND_CHUNKER"):
            with connect(store.db_path) as connection:
                rows = connection.execute("SELECT lineage FROM documents").fetchall()
            recipes = {
                json.dumps(json.loads(r["lineage"]).get("chunker"), sort_keys=True)
                for r in rows
            }
            recipes.discard("null")
            if len(recipes) == 1:
                chunker_config = ChunkerConfig(**json.loads(next(iter(recipes))))
            elif rows:
                raise ValueError(
                    "Existing corpus has mixed or missing chunker recipes; choose an explicit chunker for new ingestion"
                )
    if existed and not rebuild:
        store.ensure_embedding_compatible()
        doc = store.get_document(file_id)
        if doc is None:
            raise ValueError("Cached document disappeared; retry ingestion")
        prior = doc.get("lineage", {}).get("preprocessing", {})
        compatible = (
            prior.get("boundaries") == boundary_version
            and prior.get("language")
            in (language_version, "language-v1", "language-v2")
            and prior.get("extraction_quality")
            in (quality_version, "extraction-quality-v2")
        )
        if not compatible or any(
            prior.get(key) != preprocessing.get(key)
            for key in ("expected_language", "encoding", "pdf_layout")
        ):
            raise ValueError(
                "Cached preprocessing differs; explicitly rebuild this document"
            )
        if doc.get("lineage", {}).get("chunker") != asdict(
            chunker_config or CONFIG.chunker
        ):
            raise ValueError(
                "Cached document uses a different chunker configuration; rebuild it explicitly"
            )
        logger.info(f"'{path.name}' already ingested (file_id={file_id}); skipping")
        return FileReport(
            file_id,
            path.name,
            "cached",
            doc["n_chunks"],
            stored_filename=doc["filename"],
            quality=quality_summary(doc.get("lineage", {})),
        )

    units = load_document(path, expected_language=expected_language, encoding=encoding)
    if not units or not any(unit.content.strip() for unit in units):
        hint = (
            " Run OCR first if this is a scanned PDF."
            if path.suffix.lower() == ".pdf"
            else ""
        )
        raise EmptyDocumentError(f"No extractable text in '{path.name}'." + hint)

    lineage: Dict[str, Any] = {
        "preprocessing": preprocessing,
        "chunker": asdict(chunker_config or CONFIG.chunker),
        "embedding": {
            "backend": CONFIG.embedding.backend,
            "model": CONFIG.embedding.model_name,
            "query_instruction": CONFIG.embedding.query_instruction,
            "revision": CONFIG.embedding.model_revision,
            "inference_identity": json.loads(store._embedding_signature()),
        },
        "parsers": {
            name: importlib.metadata.version(name)
            for name in ("python-docx", "pypdfium2")
        },
    }
    from docmind.language import document_language

    lineage["language_info"] = document_language(units)
    lineage["schema_version"] = 2
    chunks: List[Chunk] = []
    lineage["page_quality"] = [
        {
            "page": u.metadata.get("page"),
            "section": u.metadata.get("section"),
            **u.metadata.get("extraction_quality", {}),
        }
        for u in units
    ]
    partial = any(
        u.metadata.get("extraction_quality", {}).get("quarantined") for u in units
    )
    lineage["partial_document"] = partial
    for unit in tqdm(units, desc=f"Chunking {path.name}", unit="unit", leave=False):
        if unit.metadata.get("extraction_quality", {}).get("quarantined"):
            continue
        from docmind.language import language_info

        unit.metadata["language_info"] = language_info(unit.content)
        unit.metadata["partial_document"] = partial
        for c in chunk_document(unit, config=chunker_config):
            c.metadata["language_info"] = language_info(c.content)
            c.file_id = file_id
            c.ord = len(chunks)  # global position within the file
            c.metadata = {
                **c.metadata,
                "file_id": file_id,
                "document_language_info": lineage["language_info"],
                "lineage": {
                    "schema_version": 2,
                    "document_ref": file_id,
                    "preprocessing": preprocessing,
                },
            }
            chunks.append(c)

    if not chunks:
        raise EmptyDocumentError(f"'{path.name}' produced zero chunks.")

    embeddings = embed_texts(
        [c.content for c in chunks], show_progress=len(chunks) > 64
    )

    lineage["embedding"]["dimension"] = int(embeddings.shape[1])
    store.add_document(
        {
            "file_id": file_id,
            "filename": path.name,
            "source": str(path.resolve()),
            "sha256": sha,
            "n_units": len(units),
            "lineage": lineage,
        },
        chunks,
        embeddings,
    )
    logger.info(
        f"Ingested '{path.name}': {len(units)} units -> {len(chunks)} chunks (file_id={file_id})"
    )
    return FileReport(
        file_id,
        path.name,
        "reingested" if existed else "ingested",
        len(chunks),
        stored_filename=path.name,
        quality=quality_summary(lineage),
    )


def ingest(
    path: Union[str, Path],
    rebuild_index: bool = False,
    store: Optional[CorpusStore] = None,
    chunker_config=None,
    expected_language: Optional[str] = None,
    encoding: Optional[str] = None,
) -> IngestReport:
    """Ingest a file or a directory (recursively). Safe to call repeatedly."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    store = store or CorpusStore()
    report = IngestReport()

    files = list(iter_supported_files(path)) if path.is_dir() else [path]
    if path.is_file() and path.suffix.lower() not in LOADERS:
        raise ValueError(
            f"Unsupported extension: {path.suffix}. Supported: {sorted(LOADERS)}"
        )

    for f in files:
        try:
            report.files.append(
                _ingest_file(
                    f, store, rebuild_index, chunker_config, expected_language, encoding
                )
            )
        except Exception as exc:
            logger.warning(f"Failed to ingest {f.name}: {exc}")
            if path.is_file():
                raise
            report.files.append(FileReport("", f.name, "failed", error=str(exc)))
    return report
