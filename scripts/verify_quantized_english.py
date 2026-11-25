"""Verify public English cohort binding, numeric aggregates and compute provenance."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from scripts.english_reader_batch_cpu import checked_decision
from scripts.english_reader_gguf import choose
from scripts.score_english_reader_cpu import article_interval, interval


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_numbers(recorded, recomputed):
    if type(recorded) is not type(recomputed):
        return False
    if isinstance(recorded, float):
        return math.isclose(recorded, recomputed, rel_tol=0, abs_tol=1e-12)
    if isinstance(recorded, dict):
        return recorded.keys() == recomputed.keys() and all(
            same_numbers(recorded[k], recomputed[k]) for k in recorded
        )
    if isinstance(recorded, list):
        return len(recorded) == len(recomputed) and all(
            same_numbers(a, b) for a, b in zip(recorded, recomputed)
        )
    return recorded == recomputed


def verify_hashes(root, name, expected_names):
    entries = load(root / name)["sha256"]
    require(set(entries) == expected_names, f"Unexpected files in {name}")
    for filename, digest in entries.items():
        require(sha(root / filename) == digest, f"Hash mismatch: {filename}")


def verify_scores(path, expected, phase):
    report = load(path)
    rows = report["numeric_records"]
    identity = ("id", "context_sha256", "answerable")
    require(
        [{k: row[k] for k in identity} for row in rows] == expected,
        "Numeric cohort changed",
    )
    require(
        report["requested_questions"] == len(expected) and report["phase"] == phase,
        "Incorrect cohort count or phase",
    )
    allowed = set(identity) | {
        "article_sha256",
        "status",
        "release_state",
        "exact_match",
        "token_f1",
        "category",
    }
    statuses = {
        "completed",
        "unserved",
        "interrupted",
        "assembly_error",
        "generation_error",
    }
    for row in rows:
        require(
            set(row) == allowed and row["status"] in statuses,
            "Unexpected public record fields or status",
        )
        require(
            row["exact_match"] in (0.0, 1.0) and 0 <= row["token_f1"] <= 1,
            "Invalid numeric score",
        )
        if row["status"] != "completed":
            require(
                row["exact_match"] == row["token_f1"] == 0,
                "Failed requests cannot earn credit",
            )
    completed = [row for row in rows if row["status"] == "completed"]
    require(
        report["status_counts"] == dict(Counter(r["status"] for r in rows)),
        "Status totals changed",
    )
    require(
        report["categories"] == dict(Counter(r["category"] for r in rows)),
        "Category totals changed",
    )
    require(
        report["completed_release_states"]
        == dict(Counter(r["release_state"] or "unknown" for r in completed)),
        "Release counts changed",
    )
    baseline = [float(not row["answerable"]) for row in rows]
    require(
        same_numbers(report["always_abstain_baseline"], interval(baseline)),
        "Baseline changed",
    )
    for metric in ("exact_match", "token_f1"):
        values = [row[metric] for row in rows]
        require(
            same_numbers(report["requested_cohort"][metric], interval(values)),
            f"Requested {metric} changed",
        )
        require(
            same_numbers(
                report["completed_only"][metric],
                interval([r[metric] for r in completed]),
            ),
            f"Completed {metric} changed",
        )
        require(
            same_numbers(
                report["article_cluster_sensitivity"][metric],
                article_interval(rows, values),
            ),
            f"Cluster interval changed: {metric}",
        )
        for value, label in ((True, "answerable"), (False, "unanswerable")):
            require(
                same_numbers(
                    report["strata"][label][metric],
                    interval([r[metric] for r in rows if r["answerable"] == value]),
                ),
                f"{label} stratum changed: {metric}",
            )
    delta = [r["exact_match"] - b for r, b in zip(rows, baseline)]
    paired = report["paired_difference_from_always_abstain"]
    require(
        same_numbers(paired["question_bootstrap"], interval(delta)),
        "Paired question interval changed",
    )
    require(
        same_numbers(
            paired["article_cluster_sensitivity"], article_interval(rows, delta)
        ),
        "Paired article interval changed",
    )
    return report


def verify_compute(record, submission, seconds):
    require(
        record["source_commit"] == submission["source_commit"], "Compute source changed"
    )
    require(
        record["wall_reservation_seconds"] == seconds, "Compute reservation changed"
    )
    require(
        record["gpu_seconds"] == record["hosted_calls"] == 0,
        "Unexpected GPU or hosted calls",
    )
    require(
        math.isfinite(record["elapsed_seconds"])
        and 0 < record["elapsed_seconds"] <= seconds,
        "Elapsed time is inconsistent with reservation",
    )


def verify(root):
    verify_hashes(
        root,
        "pilot_publication_manifest.json",
        {
            "PROTOCOL.md",
            "README.md",
            "design.json",
            "input_manifest.json",
            "cohort_decision.json",
            "pilot_compute.json",
            "pilot_timing.json",
            "pilot_diagnostics.json",
            "final_submission.json",
        },
    )
    design, manifest = load(root / "design.json"), load(root / "input_manifest.json")
    decision = load(root / "cohort_decision.json")
    timing_path = root / "pilot_timing.json"
    expected = choose(load(timing_path), manifest, design)
    expected["pilot_timing_sha256"] = sha(timing_path)
    require(expected == decision, "Cohort decision differs from frozen timing rule")
    requested = checked_decision(decision, manifest)
    submission = load(root / "final_submission.json")
    budget = design["cpu_budget"]
    require(
        submission["decision_sha256"] == sha(root / "cohort_decision.json"),
        "Final submission decision changed",
    )
    require(
        submission["requested_questions"] == requested
        and submission["private"]
        and not submission["gpu_enabled"],
        "Final job scope changed",
    )
    final_budget = design["final_reserved_seconds"]
    require(
        final_budget == submission["wall_reservation_seconds"]
        and budget["prior_reserved_seconds"]
        + budget["pilot_reserved_seconds"]
        + final_budget
        <= budget["cumulative_seconds"],
        "Total study budget exceeded",
    )
    verify_compute(
        load(root / "pilot_compute.json"), submission, budget["pilot_reserved_seconds"]
    )
    pilot = verify_scores(root / "pilot_diagnostics.json", manifest["pilot"], "pilot")
    require(
        pilot["numeric_provenance"]["cohort_decision"] == submission["decision_sha256"],
        "Pilot scorer used a different decision",
    )
    final = root / "final"
    if final.exists():
        verify_hashes(
            final,
            "publication_manifest.json",
            {"README.md", "compute.json", "timing.json", "diagnostics.json"},
        )
        compute = load(final / "compute.json")
        verify_compute(compute, submission, final_budget)
        require(
            compute["bundle_sha256"] == submission["bundle_sha256"],
            "Final compute bundle differs from submission",
        )
        report = verify_scores(final / "diagnostics.json", decision["rows"], "final")
        require(
            report["numeric_provenance"]["cohort_decision"]
            == submission["decision_sha256"],
            "Final scorer used a different decision",
        )
        return "Pilot and final numeric records verified"
    return "Pilot numeric records verified. Final results pending."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(verify(args.root))
