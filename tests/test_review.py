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


def test_dense_scores_do_not_reuse_a_reranker_review_threshold():
    CONFIG.review.min_top_score=0.1
    response=_resp()
    response.sources[0].retriever="dense"
    assert "retrieval_score_policy_unavailable" in review_reasons(response)


def test_old_extraction_policy_cannot_release_automatically(monkeypatch):
    from docmind.ingestion.quality import VERSION
    monkeypatch.setenv("DOCMIND_TEST_MODE", "0")
    monkeypatch.setattr("docmind.language.language_info", lambda text: {"language": "en"})
    response = _resp()
    metadata = response.sources[0].chunk.metadata
    metadata["language_info"] = {"language": "en"}
    metadata["lineage"] = {"preprocessing": {"extraction_quality": "extraction-quality-v1"}}
    assert "stale_extraction_policy" in review_reasons(response)
    metadata["lineage"]["preprocessing"]["extraction_quality"] = VERSION
    assert "stale_extraction_policy" not in review_reasons(response)
    del metadata["lineage"]
    assert "stale_extraction_policy" in review_reasons(response)


def test_output_language_is_gated_independently_of_question_and_citation(monkeypatch):
    from docmind.ingestion.quality import VERSION
    monkeypatch.setenv("DOCMIND_TEST_MODE","0")
    body="सही उत्तर तीन सौ डिग्री है"
    response=_resp(answer=body+" (Source: long English filename explaining the coolant protocol.pdf, p.1)",query="What is the coolant temperature limit during startup?")
    response.sources[0].chunk.metadata.update(language_info={"language":"en"},lineage={"preprocessing":{"extraction_quality":VERSION}})
    inspected=[]
    def detect(text):
        inspected.append(text)
        return {"language":"hi" if text==body else "en"}
    monkeypatch.setattr("docmind.language.language_info",detect)
    assert "answer_language_unvalidated" in review_reasons(response)
    assert body in inspected
    assert not any("Source:" in text for text in inspected)


def test_numeric_only_answers_do_not_require_a_language_guess(monkeypatch):
    from docmind.ingestion.quality import VERSION
    monkeypatch.setenv("DOCMIND_TEST_MODE","0")
    response=_resp(answer="300 (Source: a.pdf, p.1)",query="What is the coolant temperature limit?")
    response.sources[0].chunk.metadata.update(language_info={"language":"en"},lineage={"preprocessing":{"extraction_quality":VERSION}})
    def detect(text):
        assert text!="300"
        return {"language":"en"}
    monkeypatch.setattr("docmind.language.language_info",detect)
    assert "answer_language_unvalidated" not in review_reasons(response)


def test_numeric_language_exception_does_not_admit_stray_marks_or_controls():
    from docmind.review.policy import _language_neutral_number
    for value in ("300","१२.५","₹1,000.50","50%","½"):
        assert _language_neutral_number(value)
    for value in ("300 degrees","३००ा","300\u202e","...",""):
        assert not _language_neutral_number(value)
