"""
Multi-format document loader.

Returns one Document per page (PDF) or section (DOCX) so page numbers and
headings survive into citations.
"""

import hashlib
import json
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List, Union

from docmind.models import Document

logger = logging.getLogger(__name__)


class EmptyDocumentError(ValueError):
    """Raised when a file parses but yields no extractable text (e.g. scanned PDF)."""


class UnusablePdfTextError(EmptyDocumentError):
    """A parsed PDF cannot contribute evidence to the corpus."""

    def __init__(self, *, text_extracted: bool) -> None:
        if text_extracted:
            message = "The PDF text failed extraction quality checks."
            self.action = "Upload a Unicode export from the source document."
        else:
            message = "No text was extractable from the PDF."
            self.action = "If this PDF is scanned, run OCR before uploading."
        super().__init__(message)


def file_sha256(path: Union[str, Path], block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(block), b""):
            h.update(part)
    return h.hexdigest()


def pdf_layout_flags(textpage, page_width):
    """Bounded geometry warning for repeated within-line column jumps.

    The first 512 character boxes are inspected. This is neither full layout
    reconstruction nor a certificate for pages without an observed warning.
    """
    jumps = []
    previous = None
    for index in range(min(textpage.count_chars(), 512)):
        box = textpage.get_charbox(index)
        left, bottom, right, top = box
        if (
            right - left > page_width * 0.12
            and textpage.get_text_range(index, 1).isspace()
        ):
            jumps.append((left + right) / 2)
            previous = None
            continue
        if right <= left or top <= bottom:
            continue
        if previous is not None:
            pl, pb, pr, pt = previous
            same_line = abs((top + bottom) / 2 - (pt + pb) / 2) < max(
                3, (top - bottom) * 0.4
            )
            if same_line and left - pr > page_width * 0.12:
                jumps.append((pr + left) / 2)
        previous = box
    if any(
        sum(abs(x - center) < page_width * 0.06 for x in jumps) >= 3 for center in jumps
    ):
        return ["ambiguous_column_or_table_order"]
    return []


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
                from docmind.ingestion.quality import extraction_quality

                fonts = []
                try:
                    import ctypes

                    for index in range(0, min(textpage.count_chars(), 128), 16):
                        flags = ctypes.c_int()
                        length = pdfium.raw.FPDFText_GetFontInfo(
                            textpage, index, None, 0, ctypes.byref(flags)
                        )
                        if 0 < length < 4096:
                            buffer = ctypes.create_string_buffer(length)
                            pdfium.raw.FPDFText_GetFontInfo(
                                textpage, index, buffer, length, ctypes.byref(flags)
                            )
                            fonts.append(buffer.value.decode("utf-8", errors="replace"))
                except (AttributeError, TypeError, ValueError):
                    logger.debug("Font metadata unavailable")
                quality = extraction_quality(text, font_names=fonts)
                try:
                    layout_flags = pdf_layout_flags(textpage, page.get_width())
                except Exception:
                    logger.warning("PDF layout inspection unavailable")
                    layout_flags = ["layout_geometry_unavailable"]
                quality["layout_version"] = "pdf-layout-v1"
                quality["layout_sample_character_cap"] = 512
                if layout_flags:
                    quality["flags"].extend(layout_flags)
                    quality.update(status="uncertain_or_mixed", quarantined=True)
                docs.append(
                    Document(
                        content=text,
                        source=str(path.resolve()),
                        metadata={
                            "file_type": "pdf",
                            "page": page_num + 1,
                            "total_pages": total,
                            "filename": path.name,
                            "parser": "pypdfium2",
                            "extraction_quality": quality,
                        },
                    )
                )
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
        if any(
            i.file_size > 8 * 1024 * 1024 or i.file_size / max(i.compress_size, 1) > 200
            for i in entries
        ):
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


def load_text(path: Path, encoding: str | None = None) -> List[Document]:
    """Load plain text or markdown as a single Document."""
    raw = path.read_bytes()
    if encoding == "cp1252":
        content = raw.decode("cp1252")
    elif raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        content = raw.decode("utf-16")
        encoding = "utf-16"
    else:
        try:
            content = raw.decode("utf-8-sig")
            encoding = "utf-8-sig"
        except UnicodeDecodeError as exc:
            raise ValueError(
                "Text must be UTF-8 or BOM-marked UTF-16; convert its encoding first"
            ) from exc
    if len(content) > 8000000:
        raise ValueError("Text exceeds the extracted character safety limit")
    from docmind.ingestion.quality import normalize_text_input

    content, edits = normalize_text_input(content)
    source_offset = len(content) - len(content.lstrip())
    content = content.strip()
    if "\x00" in content or any(ord(c) < 32 and c not in "\t\n\r" for c in content):
        raise ValueError("Binary content is not accepted as text")
    if not content:
        return []
    file_type = "markdown" if path.suffix.lower() in (".md", ".markdown") else "text"
    return [
        Document(
            content=content,
            source=str(path.resolve()),
            metadata={
                "file_type": file_type,
                "filename": path.name,
                "encoding": encoding,
                "source_offset": source_offset,
                "normalization_edits": edits,
            },
        )
    ]


# NOTE: legacy .doc is intentionally unsupported. python-docx cannot read it,
# and the old mapping failed at runtime. Convert to .docx first.
LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_text,
    ".md": load_text,
    ".markdown": load_text,
}


def load_document(
    path: Union[str, Path],
    expected_language: str | None = None,
    encoding: str | None = None,
) -> List[Document]:
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
        raise ValueError(
            f"Unsupported extension: {path.suffix}. Supported: {sorted(LOADERS)}"
        )
    if (
        path.suffix.lower() in (".pdf", ".docx")
        and os.getenv("DOCMIND_PARSER_WORKER") != "1"
    ):
        mode = os.getenv(
            "DOCMIND_PARSER_MODE",
            "local" if os.getenv("DOCMIND_TEST_MODE") == "1" else "service",
        )
        if mode == "service":
            from docmind.ingestion.parser_client import parse_remote

            return parse_remote(path, expected_language)
        if mode != "local" or not (
            os.getenv("DOCMIND_ALLOW_UNSANDBOXED_PARSING") == "1"
            or os.getenv("DOCMIND_TEST_MODE") == "1"
        ):
            raise ValueError(
                "Native parsing requires local mode and explicit trusted-input opt-in"
            )
        environment = parser_environment()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "docmind.ingestion.parse_worker",
                    str(path.resolve()),
                    expected_language or "",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=60,
                env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Document parsing exceeded the 60-second limit") from exc
        if result.returncode:
            detail = (
                result.stderr[-500:]
                if result.stderr
                else "Parser exceeded resource limits or rejected the file"
            )
            raise ValueError(detail)
        return [Document(**item) for item in json.loads(result.stdout)]
    units = load_text(path, encoding) if loader is load_text else loader(path)
    from docmind.ingestion.quality import extraction_quality

    for unit in units:
        old = unit.metadata.get("extraction_quality", {})
        quality = extraction_quality(
            unit.content, expected_language, old.get("font_names", [])
        )
        if old.get("layout_version"):
            quality["layout_version"] = old["layout_version"]
            quality["layout_sample_character_cap"] = old["layout_sample_character_cap"]
            layout_flags = [
                flag
                for flag in old.get("flags", [])
                if flag
                in ("ambiguous_column_or_table_order", "layout_geometry_unavailable")
            ]
            if layout_flags:
                quality["flags"].extend(layout_flags)
                quality.update(status="uncertain_or_mixed", quarantined=True)
        unit.metadata["extraction_quality"] = quality
    return units


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


def parser_environment() -> dict[str, str]:
    """Start from no inherited application settings or credentials."""
    environment = {
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "DOCMIND_PARSER_WORKER": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "LANG": "C.UTF-8",
    }
    for name in ("SystemRoot", "WINDIR", "TEMP", "TMP", "SYSTEMDRIVE"):
        if name in os.environ:
            environment[name] = os.environ[name]
    environment["PATH"] = str(Path(sys.executable).parent)
    if os.getenv("DOCMIND_TEST_MODE") == "1":
        environment["DOCMIND_TEST_MODE"] = "1"
    return environment
