"""Tight-kerning PDFs (glued words), the tolerance diagnostic, and the prose filter for question drafting."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pdfplumber")
ROOT = Path(__file__).resolve().parents[1]
SENTENCE = "We evaluate the final selected repository on the full benchmark and report the results"


def build_tight_pdf(path: Path, gap: float = 0.8, lines: int = 12):
    """Single-page PDF in a monospaced font where words are separated by only `gap` points, like a tightly
    kerned paper. Characters inside a word touch (gap 0)."""
    content, y = [], 750
    for _ in range(lines):
        x = 50.0
        for w in SENTENCE.split():
            content.append(f"BT /F1 10 Tf {x:.2f} {y} Td ({w}) Tj ET")
            x += 6 * len(w) + gap
        y -= 14
    stream = "\n".join(content).encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)
    return path


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))
    spec.loader.exec_module(mod)
    return mod


def test_default_tolerance_glues_tightly_kerned_words_and_the_setting_fixes_it(tmp_path, monkeypatch):
    from docmind.ingestion.loader import load_pdf
    pdf = build_tight_pdf(tmp_path / "tight.pdf")
    monkeypatch.setenv("DOCMIND_PDF_X_TOLERANCE", "3")
    glued = load_pdf(pdf)[0].content
    assert "Weevaluatethe" in glued.replace(" ", "") and "We evaluate the" not in glued
    monkeypatch.setenv("DOCMIND_PDF_X_TOLERANCE", "0.5")
    fixed = load_pdf(pdf)[0].content
    assert "We evaluate the final selected repository" in fixed


def test_invalid_tolerance_falls_back_to_default(tmp_path, monkeypatch):
    from docmind.ingestion.loader import _pdf_x_tolerance
    monkeypatch.setenv("DOCMIND_PDF_X_TOLERANCE", "not-a-number")
    assert _pdf_x_tolerance() == 3.0
    monkeypatch.delenv("DOCMIND_PDF_X_TOLERANCE")
    assert _pdf_x_tolerance() == 3.0


def test_diagnostic_recommends_a_smaller_tolerance_for_glued_text(tmp_path):
    mod = _load_script("check_pdf_text")
    res = mod.evaluate(str(build_tight_pdf(tmp_path / "tight.pdf")))
    by = {r["tolerance"]: r for r in res["rows"]}
    assert res["best"] is not None and res["best"] < 3.0
    assert by[3.0]["long_ratio"] > 0.1 and by[res["best"]]["long_ratio"] == 0.0
    assert "We evaluate the" in res["previews"][res["best"]]


def test_diagnostic_keeps_default_for_normally_spaced_text(tmp_path):
    mod = _load_script("check_pdf_text")
    res = mod.evaluate(str(build_tight_pdf(tmp_path / "normal.pdf", gap=6.0)))     # a real space width
    assert res["best"] == 3.0


def test_diagnostic_cli(tmp_path):
    pdf = build_tight_pdf(tmp_path / "tight.pdf")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_pdf_text.py"), str(pdf)], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(ROOT)))
    assert r.returncode == 0 and "export DOCMIND_PDF_X_TOLERANCE=" in r.stdout and "<-- best" in r.stdout
    missing = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_pdf_text.py"), str(tmp_path / "nope.pdf")], capture_output=True, text=True)
    assert missing.returncode == 1 and "not found" in missing.stdout


def test_sample_pages_spread_across_the_document():
    mod = _load_script("check_pdf_text")
    pages = mod.sample_pages(64)
    assert len(pages) == 6 and pages == sorted(pages) and 0 < pages[0] and pages[-1] < 63
    assert mod.sample_pages(3) == [0, 1, 2]


PROSE = ("The meta-agent receives training trajectories while validation instructions and scores remain hidden from it. "
         "Validation performance guides parent selection during evolution. We evaluate the final selected repository on the full "
         "benchmark. The approach limits specialization to the tasks that motivated these changes. Descendants can add, revise or "
         "remove interfaces and knowledge, and the population retains alternative designs for further recombination.")
TABLE = "Seed 163.2 71.4 55.0 Agent-S3 161.9 70.2 54.1 Ours 190.4 80.3 61.7 " * 8
EXAMPLE = "Save the native project as output/orchard_loop.lml. Then open File 3 and press 4 5 6. " * 6


def test_prose_filter_keeps_prose_and_drops_tables_and_trivia():
    mod = _load_script("make_testset")
    assert mod.is_prose(PROSE)
    assert not mod.is_prose(TABLE)
    assert not mod.is_prose(EXAMPLE)                   # digits and path-like trivia
    assert not mod.is_prose("Short.")
    assert not mod.is_prose("word " * 200)             # long but no sentences
