"""Metric denominators and uncertainty use synthetic support judgments only."""

from copy import deepcopy

import pytest

from scripts.analyze_pdf_claim_diagnostic import checked_scores, confusion, summarize


def fixture():
    units = [
        {"answer_id": "a", "claim_id": "c1", "input_sha256": "one"},
        {"answer_id": "b", "claim_id": "c2", "input_sha256": "two"},
        {"answer_id": "b", "claim_id": "c3", "input_sha256": "three"},
    ]
    design = {
        "scope": "synthetic",
        "models": [{"id": "legacy"}, {"id": "comparison"}],
        "input_manifest": {"units": units},
        "primary_threshold": 0.8,
        "descriptive_sensitivity_threshold": 0.5,
    }
    records = [
        {**unit, "model_index": i, "status": "completed", "score": score}
        for i, scores in enumerate(([0.9, 0.9, 0.1], [0.9, 0.1, 0.1]))
        for unit, score in zip(units, scores)
    ]
    refs = {
        "first": {
            ("a", "c1"): "supported",
            ("b", "c2"): "unsupported",
            ("b", "c3"): "unclear",
        },
        "second": {("a", "c1"): "supported", ("b", "c2"): "supported"},
        "concordant": {("a", "c1"): "supported"},
    }
    return records, design, refs


def test_separate_references_and_paired_differences():
    args = fixture()
    before = deepcopy(args)
    out = summarize(*args)
    assert args == before
    first = out["models"][0]["references"]["first"]
    assert first["unclear_labels"] == 1 and first["decisive_labels"] == 2
    assert first["thresholds"]["0.8"]["precision"] == 0.5
    assert (
        out["models"][1]["references"]["first"]["thresholds"]["0.8"]["precision"] == 1
    )
    assert (
        out["models"][1]["references"]["second"]["thresholds"]["0.8"]["recall"] == 0.5
    )
    paired = out["paired_common_input_differences"]["first"]
    assert paired["common_claims"] == 2
    assert paired["metrics"]["precision"]["difference"] == 0.5
    assert out["population_false_release_risk"] is None


def test_missing_predictions_stay_in_requested_cohort():
    records, design, refs = fixture()
    records[0].update(status="unserved", score=None)
    out = summarize(records, design, refs)
    assert out["requested_claims"] == 3
    legacy = out["models"][0]["references"]["first"]
    assert legacy["decisive_labels_without_prediction"] == 1
    assert legacy["thresholds"]["0.8"]["n"] == 1
    assert out["paired_common_input_differences"]["first"]["common_claims"] == 1


def test_undefined_ratios_are_not_zero_or_perfect():
    metrics = confusion([], 0.8)
    assert metrics["precision"] is None and metrics["recall"] is None
    records, design, _ = fixture()
    out = summarize(records, design, {"negative_only": {("b", "c3"): "unsupported"}})
    metric = out["models"][0]["references"]["negative_only"]["thresholds"]["0.8"]
    assert metric["cluster_bootstrap"]["recall"]["undefined_resamples"] == 2000


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "hash", "nan", "fake_status"]
)
def test_invalid_numeric_records_fail(fault):
    records, design, _ = fixture()
    if fault == "missing":
        records.pop()
    elif fault == "duplicate":
        records.append(records[0])
    elif fault == "hash":
        records[0]["input_sha256"] = "changed"
    elif fault == "nan":
        records[0]["score"] = float("nan")
    else:
        records[0]["status"] = "unknown"
    with pytest.raises(ValueError):
        checked_scores(records, design)
