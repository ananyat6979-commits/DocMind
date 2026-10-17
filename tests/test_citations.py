from docmind.agent.citations import verify_citations
from docmind.models import Chunk, SearchResult


def _src(filename="report.pdf", page=3, section=None):
    md = {"filename": filename}
    if page is not None:
        md["page"] = page
    if section:
        md["section"] = section
    return SearchResult(Chunk("x", "d", filename, 0, metadata=md), 1.0, "reranker")


def test_valid_citation_kept():
    a, rep, ids = verify_citations("Fact (Source: report.pdf, p.3).", [_src()])
    assert "(Source: report.pdf, p.3)" in a and rep.validity == 1.0 and len(ids) == 1


def test_wrong_page_removed():
    a, rep, _ = verify_citations("Fact (Source: report.pdf, p.4).", [_src()])
    assert "p.4" not in a and rep.invalid and rep.validity == 0.0


def test_whitespace_and_case_tolerant():
    _, rep, _ = verify_citations("Fact (source:  Report.PDF ,  p. 3 ).", [_src()])
    assert rep.validity == 1.0


def test_multi_item_citation_partially_valid():
    a, rep, _ = verify_citations("Fact (Source: report.pdf, p.3; other.pdf, p.1).", [_src()])
    assert "other.pdf" not in a and "report.pdf, p.3" in a
    assert len(rep.valid) == 1 and len(rep.invalid) == 1


def test_filename_with_parentheses():
    s = _src(filename="report (1).pdf", page=2)
    a, rep, _ = verify_citations("Fact (Source: report (1).pdf, p.2).", [s])
    assert rep.validity == 1.0 and "report (1).pdf" in a


def test_section_citation_for_docx():
    s = _src(filename="memo.docx", page=None, section="Risk; Summary")
    a, rep, _ = verify_citations("Fact (Source: memo.docx, §Risk Summary).", [s])
    assert rep.validity == 1.0


def test_no_citations_has_no_validity():
    _, rep, _ = verify_citations("Fact without citation.", [_src()])
    assert rep.validity is None and rep.cited == []
