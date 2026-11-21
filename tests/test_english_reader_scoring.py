"""Unserved requests and malformed abstentions must never earn correctness."""

import pytest

from scripts.score_english_reader_cpu import pair_scores, score, summarize


def example(answerable=True):
    return {
        "id": "one",
        "context_sha256": "paragraph",
        "answerable": answerable,
        "answers": [{"text": "two seconds"}, {"text": "2 seconds"}]
        if answerable
        else [],
    }


@pytest.mark.parametrize("status", ["unserved", "interrupted", "error"])
def test_failure_is_wrong_even_on_unanswerable_example(status):
    result = score({"status": status, "abstained": True}, example(False))
    assert result["exact_match"] == result["token_f1"] == 0


def test_blank_or_format_failure_does_not_become_correct_abstention():
    for row in [
        {"status": "completed", "short_answer": ""},
        {"status": "completed", "short_answer": None},
        {"status": "completed", "abstained": True, "finish_reason": "format_failure"},
    ]:
        assert score(row, example(False))["exact_match"] == 0


def test_explicit_abstention_and_answerable_aliases():
    assert (
        score({"status": "completed", "abstained": True}, example(False))["exact_match"]
        == 1
    )
    assert (
        score({"status": "completed", "abstained": True}, example())["category"]
        == "false_abstention"
    )
    row = {"status": "completed", "short_answer": "2 seconds", "release_state": "held"}
    assert score(row, example())["exact_match"] == 1
    assert score(row, example(False))["category"] == "answered_unanswerable"


def test_normalization_and_token_overlap_follow_public_span_metric():
    assert pair_scores("The two seconds!", "two seconds") == (1, 1)
    assert pair_scores("two seconds", "seconds") == (0, pytest.approx(2 / 3))


def test_missing_rows_stay_in_requested_denominator():
    a, b = example(), example(False)
    b.update(id="two", context_sha256="other")
    result = summarize(
        [a, b],
        [
            {
                "id": "one",
                "context_sha256": "paragraph",
                "status": "completed",
                "short_answer": "two seconds",
            }
        ],
    )
    assert result["requested_cohort"]["exact_match"]["mean"] == 0.5
    assert result["completed_only"]["exact_match"]["mean"] == 1
    assert result["always_abstain_baseline"]["mean"] == 0.5
    assert result["categories"]["unserved"] == 1
    assert result["population_release_risk"] is None


def test_changed_or_duplicate_reference_identity_fails():
    row = {"id": "one", "status": "unserved", "context_sha256": "changed"}
    with pytest.raises(ValueError):
        summarize([example()], [row])
    with pytest.raises(ValueError):
        summarize([example()], [row, row])
