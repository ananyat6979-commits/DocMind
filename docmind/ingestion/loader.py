"""
Multi-format document loader.

Returns one Document per page (PDF) or section (DOCX) so page numbers and
headings survive into citations.
"""
import hashlib
import logging
from pathlib import Path
from typing import List, Union

from docmind.models import Document

logger = logging.getLogger(__name__)


class EmptyDocumentError(ValueError):
    """Raised when a file parses but yields no extractable text (e.g. scanned PDF)."""


def file_sha256(path: Union[str, Path], block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(block), b""):
            h.update(part)
    return h.hexdigest()


def load_pdf(path: Path) -> List[Document]:
    """Extract text page by page using pdfplumber."""
    import pdfplumber

    docs = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(Document(
                    content=text.strip(),
                    source=str(path.resolve()),
                    metadata={
                        "file_type": "pdf",
                        "page": page_num + 1,
                        "total_pages": total,
                        "filename": path.name,
                    },
                ))
    if total and not docs:
        logger.warning(f"PDF '{path.name}' has {total} pages but no text layer (scanned?). OCR is required.")
    logger.info(f"PDF '{path.name}': loaded {len(docs)}/{total} non-empty pages")
    return docs


def _iter_docx_blocks(docx):
    """Yield paragraphs and tables in true document order."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in docx.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, docx)
        elif child.tag.endswith("}tbl"):
            yield Table(child, docx)


def load_docx(path: Path) -> List[Document]:
    """Load DOCX preserving heading-based sections. Tables are flattened to
    'a | b | c' rows so their content is searchable."""
    from docx import Document as DocxDoc
    from docx.table import Table

    docx = DocxDoc(str(path))
    sections: List[tuple] = []
    current_heading = "Preamble"
    current_lines: List[str] = []

    for block in _iter_docx_blocks(docx):
        if isinstance(block, Table):
            for row in block.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    current_lines.append(" | ".join(cells))
            continue
        text = block.text.strip()
        if not text:
            continue
        style_name = (block.style.name if block.style is not None else "") or ""
        if style_name.startswith("Heading") or style_name == "Title":
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines)))
                current_lines = []
            current_heading = text
        else:
            current_lines.append(text)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines)))

    docs = [
        Document(
            content=content,
            source=str(path.resolve()),
            metadata={"file_type": "docx", "section": heading, "filename": path.name},
        )
        for heading, content in sections
        if content.strip()
    ]
    logger.info(f"DOCX '{path.name}': loaded {len(docs)} sections")
    return docs


def load_text(path: Path) -> List[Document]:
    """Load plain text or markdown as a single Document."""
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return []
    file_type = "markdown" if path.suffix.lower() in (".md", ".markdown") else "text"
    return [Document(
        content=content,
        source=str(path.resolve()),
        metadata={"file_type": file_type, "filename": path.name},
    )]


# NOTE: legacy .doc is intentionally unsupported. python-docx cannot read it,
# and the old mapping failed at runtime. Convert to .docx first.
LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_text,
    ".md": load_text,
    ".markdown": load_text,
}


def load_document(path: Union[str, Path]) -> List[Document]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    loader = LOADERS.get(path.suffix.lower())
    if not loader:
        raise ValueError(f"Unsupported extension: {path.suffix}. Supported: {sorted(LOADERS)}")
    return loader(path)


def iter_supported_files(directory: Union[str, Path]):
    for file_path in sorted(Path(directory).rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in LOADERS:
            yield file_path


def load_directory(directory: Union[str, Path]) -> List[Document]:
    docs: List[Document] = []
    for file_path in iter_supported_files(directory):
        try:
            docs.extend(load_document(file_path))
        except Exception as exc:
            logger.warning(f"Skipping {file_path.name}: {exc}")
    logger.info(f"Directory '{directory}': loaded {len(docs)} total documents")
    return docs
