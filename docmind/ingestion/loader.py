"""
Multi-format document loader.

Returns one Document per page (PDF) or section (DOCX) so page numbers and
headings survive into citations.
"""
import hashlib
import logging
import os
import zipfile
import json
import subprocess
import sys
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


def _pdf_x_tolerance() -> float:
    """Horizontal gap (in points) above which pdfplumber inserts a space. The default of 3 glues words
    together in tightly kerned PDFs ('WeevaluateAgentS4'), which breaks word-based search. Choose a value
    with scripts/check_pdf_text.py and set DOCMIND_PDF_X_TOLERANCE. Re-ingest after changing it."""
    try:
        return float(os.getenv("DOCMIND_PDF_X_TOLERANCE", "3"))
    except ValueError:
        return 3.0


def load_pdf(path: Path) -> List[Document]:
    """Extract bounded text with PDFium, closing each page immediately."""
    import pypdfium2 as pdfium

    docs = []
    total_chars = 0
    with pdfium.PdfDocument(str(path)) as pdf:
        total = len(pdf)
        if total > 500:
            raise ValueError("PDF exceeds 500-page safety limit")
        for page_num in range(total):
            page = pdf[page_num]
            textpage = None
            try:
                textpage = page.get_textpage()
                if textpage.count_chars() > 200000:
                    raise ValueError("PDF page exceeds text safety limit")
                text = textpage.get_text_range().strip()
                total_chars += len(text)
                if total_chars > 8000000:
                    raise ValueError("PDF exceeds extracted text safety limit")
                if text:
                    docs.append(Document(content=text, source=str(path.resolve()), metadata={
                        "file_type": "pdf", "page": page_num + 1, "total_pages": total,
                        "filename": path.name, "parser": "pypdfium2"}))
            finally:
                if textpage is not None:
                    textpage.close()
                page.close()
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

    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 2000 or sum(i.file_size for i in entries) > 32 * 1024 * 1024:
            raise ValueError("DOCX expanded content exceeds 32 MiB or 2000 entries")
        if any(i.file_size > 8 * 1024 * 1024 or i.file_size / max(i.compress_size, 1) > 200 for i in entries):
            raise ValueError("DOCX entry exceeds expansion safety limits")
    docx = DocxDoc(str(path))
    sections: List[tuple] = []
    current_heading = "Preamble"
    current_lines: List[str] = []

    seen_cells = set()
    for block in _iter_docx_blocks(docx):
        if isinstance(block, Table):
            for row in block.rows:
                cells = []
                for cell in row.cells:
                    if cell._tc in seen_cells:
                        continue
                    seen_cells.add(cell._tc)
                    if cell.text.strip():
                        cells.append(cell.text.strip())
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
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        content = raw.decode("utf-16").strip()
    else:
        try:
            content = raw.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("Text must be UTF-8 or BOM-marked UTF-16; convert its encoding first") from exc
    if "\x00" in content or any(ord(c) < 32 and c not in "\t\n\r" for c in content):
        raise ValueError("Binary content is not accepted as text")
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
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("File exceeds 50 MiB safety limit")
    with path.open("rb") as stream:
        magic = stream.read(8)
    if path.suffix.lower() == ".pdf" and not magic.startswith(b"%PDF-"):
        raise ValueError("Invalid PDF signature")
    if path.suffix.lower() == ".docx" and not magic.startswith(b"PK"):
        raise ValueError("Invalid DOCX signature")
    loader = LOADERS.get(path.suffix.lower())
    if not loader:
        raise ValueError(f"Unsupported extension: {path.suffix}. Supported: {sorted(LOADERS)}")
    if path.suffix.lower() in (".pdf", ".docx") and os.getenv("DOCMIND_PARSER_WORKER") != "1":
        environment = os.environ.copy()
        for key in ("GROQ_API_KEY","DOCMIND_API_KEY","DOCMIND_APP_PASSWORD","DOCMIND_REVIEWER_PIN", "KAGGLE_API_TOKEN","HF_TOKEN"):
            environment.pop(key,None)
        try:
            result=subprocess.run([sys.executable,"-m","docmind.ingestion.parse_worker",str(path.resolve())],
                capture_output=True,text=True,encoding="utf-8",timeout=60,env=environment,
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Document parsing exceeded the 60-second limit") from exc
        if result.returncode:
            detail = result.stderr[-500:] if result.stderr else "Parser exceeded resource limits or rejected the file"
            raise ValueError(detail)
        return [Document(**item) for item in json.loads(result.stdout)]
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
