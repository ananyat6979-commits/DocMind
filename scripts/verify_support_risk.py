"""Verify saved public support-risk numbers without models or source text."""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.support_risk_component import baseline_results, evaluate, fit, policies


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(directory, design_path):
    design = read(design_path)
    rows = read(directory / "numeric_records.json")
    summary = read(directory / "summary.json")
    selection = read(directory / "calibration_selection.json")
    expected = {case["id"]: case for case in design["rows"]}
    assert len(expected) == len(design["rows"])
    assert len(rows) == len(expected) * len(design["models"])
    assert len({(row["model_index"], row["id"]) for row in rows}) == len(rows)
    for row in rows:
        assert row["model_index"] in range(len(design["models"]))
        case = expected[row["id"]]
        assert all(row[key] == case[key] for key in ("source_id", "split", "task"))
        assert type(row["unsupported"]) is bool
        assert type(row["quality_problem"]) is bool
        for key in ("score_top1", "score_top4"):
            score = row.get(key)
            assert score is None or (
                isinstance(score, (int, float))
                and math.isfinite(score)
                and 0 <= score <= 1
            )
    for split, count in (
        ("calibration", design["calibration_responses"]),
        ("heldout", design["heldout_responses"]),
    ):
        subset = [case for case in expected.values() if case["split"] == split]
        assert len(subset) == count
        assert len({case["source_id"] for case in subset}) == count
    source_sets = [
        {r["source_id"] for r in expected.values() if r["split"] == split}
        for split in ("calibration", "heldout")
    ]
    assert not source_sets[0] & source_sets[1]
    for case_id in expected:
        paired = [r for r in rows if r["id"] == case_id]
        assert len({(r["unsupported"], r["quality_problem"]) for r in paired}) == 1
    calibration = {
        r["context_sha256"] for r in expected.values() if r["split"] == "calibration"
    }
    heldout = {
        r["context_sha256"] for r in expected.values() if r["split"] == "heldout"
    }
    assert not calibration & heldout
    assert (
        selection["design_sha256"]
        == hashlib.sha256(design_path.read_bytes()).hexdigest()
    )
    assert selection["selection_uses_heldout_labels"] is False
    for label, stratified in (("pooled", False), ("stratified", True)):
        chosen = fit(rows, design, stratified=stratified)
        assert selection[label] == chosen
        assert summary[label + "_calibration_selection"] == chosen
        assert summary[label + "_heldout"] == evaluate(rows, chosen["policy"], design)
        for task in ("QA", "Summary", "Data2txt"):
            subset = [row for row in rows if row["task"] == task]
            assert summary["per_task"][task][label] == evaluate(
                subset, chosen["policy"], design
            )
    curves = [
        {"policy": p, "heldout": evaluate(rows, p, design)} for p in policies(design)
    ]
    assert read(directory / "fixed_policy_curves.json") == curves
    assert summary["baselines"] == baseline_results(rows, design)
    assert summary["statuses"] == dict(
        Counter(str(r["model_index"]) + ":" + r["status"] for r in rows)
    )
    assert all(
        summary[key] == 0 for key in ("new_reader_calls", "gpu_seconds", "hosted_calls")
    )
    return {
        "verified": True,
        "requested_per_model": len(expected),
        "heldout_responses": design["heldout_responses"],
        "fixed_policies": len(curves),
        "scope": "Numerical consistency and frozen split identity, not independent semantic validation",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--design",
        type=Path,
        default=ROOT / "reports/releases/r6/support_risk_design.json",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.design)))
