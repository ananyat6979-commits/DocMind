import pytest
from docmind.config import CONFIG
from docmind.models import AgentResponse, Chunk, SearchResult
from docmind.review.release import apply_policy


def response(answer="Two seconds.", language="en", partial=False):
    chunk = Chunk(
        "The valve closes within two seconds.",
        "d",
        "manual.pdf",
        0,
        metadata={
            "page": 1,
            "document_language_info": {"language": language},
            "language_info": {"language": language},
            "partial_document": partial,
        },
    )
    source = SearchResult(chunk, 1, "dense")
    return AgentResponse(
        answer,
        [source],
        [],
        "What is the valve closing time?",
        1,
        cited_sources=[source],
        citation_report={
            "cited": ["manual.pdf"],
            "valid": ["manual.pdf"],
            "invalid": [],
            "quote_validation": {"quotes_matched": True},
            "attribution": {"supported_fraction": 0, "error": "Unavailable"},
        },
        document_context={"partial": partial, "languages": [language]},
    )


def test_nli_neither_grants_nor_denies_draft(monkeypatch):
    monkeypatch.setattr(
        "docmind.language.language_info", lambda text: {"language": "en"}
    )
    result = apply_policy(response())
    assert result.release_state == "draft" and not result.needs_review
    assert not result.hold_reasons
    monkeypatch.setattr(CONFIG.review, "release_mode", "strict")
    assert apply_policy(response()).hold_reasons == ["strict_human_approval"]


def test_short_english_context_does_not_validate_french_file(monkeypatch):
    monkeypatch.setattr(
        "docmind.language.language_info",
        lambda text: {"language": "und", "flags": ["short_text"]},
    )
    monkeypatch.setattr(
        "docmind.language.short_language_hint",
        lambda text: "fr" if "Deux" in text else "en",
    )
    assert apply_policy(response()).release_state == "draft"
    assert apply_policy(response("Deux secondes.", "fr")).release_state == "held"
    assert (
        "answer_language_unvalidated"
        in apply_policy(response("Deux secondes.", "en")).hold_reasons
    )


def test_abstention_visibility_and_partial_page_warning(monkeypatch):
    monkeypatch.setattr(
        "docmind.language.language_info", lambda text: {"language": "en"}
    )
    complete = response()
    complete.abstained = True
    assert apply_policy(complete).release_state == "draft"
    partial = response(partial=True)
    assert (
        apply_policy(partial).release_state == "draft"
        and "document_incomplete" in partial.warnings
    )
    partial.abstained = True
    assert apply_policy(partial).release_state == "held"
    assert "Absence cannot be confirmed" in partial.answer


def test_missing_quote_and_invalid_mode_fail_closed(monkeypatch):
    monkeypatch.setattr(
        "docmind.language.language_info", lambda text: {"language": "en"}
    )
    value = response()
    value.citation_report["quote_validation"]["quotes_matched"] = False
    assert apply_policy(value).release_state == "held"
    monkeypatch.setattr(CONFIG.review, "release_mode", "unknown")
    with pytest.raises(ValueError):
        apply_policy(response())


def test_short_english_boolean_requires_english_file_context(monkeypatch):
    monkeypatch.setattr(
        "docmind.language.language_info",
        lambda text: {"language": "und", "flags": ["short_text"]},
    )
    for answer in ("Yes.", "No.", "True", "False"):
        assert apply_policy(response(answer)).release_state == "draft"
        assert apply_policy(response(answer, "fr")).release_state == "held"


def test_english_abstention_cannot_validate_foreign_or_unknown_source_scope(
    monkeypatch,
):
    monkeypatch.setattr(
        "docmind.language.language_info", lambda text: {"language": "en"}
    )
    for language in ("hi", "fr", "und"):
        item = response(language=language)
        item.abstained = True
        item.cited_sources = []
        assert "language_policy_unvalidated" in apply_policy(item).hold_reasons


def test_interrogatives_need_not_appear_in_evidence_and_names_use_document_context():
    for query in (
        "Who operates this valve?",
        "Which planet is closest to the sun?",
        "What is the coolant temperature limit for this machine?",
    ):
        value = response()
        value.query = query
        assert apply_policy(value).release_state == "draft"
    for name in ("Hyderabad", "General Data Protection Regulation"):
        value = response(name)
        value.sources[0].chunk.content = "The source explicitly names " + name + "."
        assert apply_policy(value).release_state == "draft"
    assert apply_policy(response("Deux secondes.")).release_state == "held"


@pytest.mark.parametrize(
    "answer",
    [
        "a significant increase",
        "security requirements",
        "General Data Protection Regulation",
        "Hyderabad",
    ],
)
def test_source_identical_short_english_fragments_inherit_document_language(answer):
    value = response(answer)
    value.sources[0].chunk.content = (
        "The English source explicitly states " + answer + "."
    )
    assert apply_policy(value).release_state == "draft"


@pytest.mark.parametrize(
    "answer",
    [
        "Deux secondes.",
        "Zwei Sekunden.",
        "Kaksi sekuntia.",
        "यह हिन्दी उत्तर है।",
        "kya hai",
    ],
)
def test_english_context_never_clears_foreign_prose_or_hinglish(answer):
    value = response(answer)
    value.sources[0].chunk.content = "A quotation appears here: " + answer
    assert apply_policy(value).release_state == "held"
