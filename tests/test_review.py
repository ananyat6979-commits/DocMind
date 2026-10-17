import json
import pytest

from docmind.config import CONFIG
from docmind.models import AgentResponse, Chunk, SearchResult
from docmind.review.policy import review_reasons
from docmind.review.store import ReviewStore
from docmind.telemetry.events import EventLog


def _resp(**kw):
    c = Chunk("evidence text", "d", "a.pdf", 0, metadata={"filename": "a.pdf", "page": 1})
    base = dict(answer="A (Source: a.pdf, p.1)", sources=[SearchResult(c, 3.0, "reranker")],
                reasoning_trace=[], query="q?", latency_ms=1.0,
                cited_sources=[SearchResult(c, 3.0, "reranker")],
                citation_report={"cited": ["a.pdf,p.1"], "valid": ["a.pdf,p.1"], "invalid": [], "validity": 1.0})
    base.update(kw)
    return AgentResponse(**base)


def test_clean_answer_is_not_flagged():
    assert review_reasons(_resp()) == []


def test_each_trigger():
    assert "abstained" in review_reasons(_resp(abstained=True))
    assert "no_citations" in review_reasons(_resp(citation_report={"cited": [], "valid": [], "invalid": []}))
    assert "invalid_citation" in review_reasons(_resp(citation_report={"cited": ["x"], "valid": [], "invalid": ["x"]}))
    assert "no_clean_final_answer" in review_reasons(_resp(finish_reason="fallback"))
    CONFIG.review.min_top_score = 5.0
    assert "low_retrieval_confidence" in review_reasons(_resp())


def test_audit_sampling_is_deterministic_and_roughly_calibrated():
    CONFIG.review.audit_rate = 0.2
    hits = sum(bool(review_reasons(_resp(response_id=f"id{i}"))) for i in range(2000))
    assert 300 < hits < 500                     # ~20% of 2000
    assert review_reasons(_resp(response_id="id7")) == review_reasons(_resp(response_id="id7"))


def test_review_lifecycle_and_export(tmp_path):
    rs = ReviewStore()
    flagged = _resp(abstained=False, needs_review=True, review_reasons=["no_citations"], response_id="r1")
    clean = _resp(response_id="r2")
    assert rs.record(flagged) == "pending" and rs.record(clean) == "auto_approved"
    assert [p["response_id"] for p in rs.pending()] == ["r1"]
    with pytest.raises(ValueError):
        rs.decide("r1", "edited", "me")                       # edit needs text
    assert rs.decide("r1", "edited", "me", edited_answer="Corrected answer", note="fixed")
    assert not rs.decide("r1", "approved", "me")              # cannot decide twice
    assert rs.status_of("r1") == "edited"
    out = tmp_path / "ts.json"
    assert rs.export_testset(str(out)) == 1
    item = json.loads(out.read_text())[0]
    assert item["ground_truth"] == "Corrected answer" and item["origin"] == "human_reviewed"


def test_event_log_roundtrip():
    ev = EventLog()
    ev.log("query", user="u1", latency_ms=1200, abstained=False)
    rows = ev.rows("query")
    assert rows[0]["user"] == "u1" and rows[0]["payload"]["latency_ms"] == 1200
