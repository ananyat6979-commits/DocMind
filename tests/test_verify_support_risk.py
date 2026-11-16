"""Saved-result integrity checks use synthetic fixtures, never research labels."""

import hashlib
import json

import pytest

from scripts.support_risk_component import baseline_results, evaluate, fit, policies
from scripts.verify_support_risk import verify
from tests.test_support_risk_component import design, rows


def fixture(tmp_path):
    plan = design()
    records = rows() + [
        {**rows()[0], "id": "heldout-" + task, "split": "heldout", "task": task}
        for task in ("QA", "Summary", "Data2txt")
    ]
    for row in records:
        row.update(
            source_id=row["id"],
            context_sha256=row["id"],
            detected_refusal=False,
            numeric_baseline_ok=True,
        )
    plan.update(
        rows=[
            {
                key: r[key]
                for key in ("id", "source_id", "split", "task", "context_sha256")
            }
            for r in records
        ],
        calibration_responses=100,
        heldout_responses=3,
    )
    path = tmp_path / "design.json"
    path.write_text(json.dumps(plan))
    selected = {
        key: fit(records, plan, stratified=value)
        for key, value in (("pooled", False), ("stratified", True))
    }
    summary = {key + "_calibration_selection": value for key, value in selected.items()}
    summary.update(
        {
            key + "_heldout": evaluate(records, value["policy"], plan)
            for key, value in selected.items()
        }
    )
    summary.update(
        per_task={
            task: {
                key: evaluate(
                    [r for r in records if r["task"] == task], value["policy"], plan
                )
                for key, value in selected.items()
            }
            for task in ("QA", "Summary", "Data2txt")
        },
        baselines=baseline_results(records, plan),
        statuses={"0:completed": 103},
        new_reader_calls=0,
        gpu_seconds=0,
        hosted_calls=0,
    )
    payloads = {
        "numeric_records.json": records,
        "summary.json": summary,
        "calibration_selection.json": {
            **selected,
            "selection_uses_heldout_labels": False,
            "design_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "fixed_policy_curves.json": [
            {"policy": p, "heldout": evaluate(records, p, plan)} for p in policies(plan)
        ],
    }
    for name, value in payloads.items():
        (tmp_path / name).write_text(json.dumps(value))
    return path


def test_verifier_rejects_a_dropped_requested_row(tmp_path):
    path = fixture(tmp_path)
    saved = tmp_path / "numeric_records.json"
    data = json.loads(saved.read_text())
    saved.write_text(json.dumps(data[:-1]))
    with pytest.raises(AssertionError):
        verify(tmp_path, path)


def test_verifier_rejects_nonfinite_scores(tmp_path):
    path = fixture(tmp_path)
    saved = tmp_path / "numeric_records.json"
    data = json.loads(saved.read_text())
    data[0]["score_top1"] = float("nan")
    saved.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        verify(tmp_path, path)


def test_verifier_accepts_complete_consistent_saved_results(tmp_path):
    path = fixture(tmp_path)
    assert verify(tmp_path, path)["verified"] is True


def test_verifier_rejects_changed_published_counts(tmp_path):
    path = fixture(tmp_path)
    saved = tmp_path / "summary.json"
    data = json.loads(saved.read_text())
    data["baselines"]["all_responses"]["released"] += 1
    saved.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        verify(tmp_path, path)
