"""Score both T4 prompt arms on the frozen, requested English cohort."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from scripts.score_english_reader_cpu import interval, summarize

ARMS = ("previous", "revised")
TERMINAL = {
    "completed",
    "unserved",
    "interrupted",
    "assembly_error",
    "generation_error",
}


def score_pairs(design, cases, records):
    if [
        {k: row[k] for k in ("id", "context_sha256", "answerable")} for row in cases
    ] != design["rows"]:
        raise ValueError("Evaluation cohort changed")
    expected = {(row["id"], arm) for row in cases for arm in ARMS}
    by_key = {(row["id"], row["arm"]): row for row in records}
    if len(by_key) != len(records) or set(by_key) != expected:
        raise ValueError("Each frozen question requires one checkpoint per arm")
    if any(row["status"] not in TERMINAL for row in records):
        raise ValueError("Results must have terminal statuses")

    reports = {}
    for arm in ARMS:
        rows = [by_key[row["id"], arm] for row in cases]
        report = summarize(cases, rows)
        calls = [call for row in rows for call in row.get("call_records", [])]
        report.update(
            status_counts=dict(Counter(row["status"] for row in rows)),
            observed_calls=len(calls),
            observed_generated_tokens=sum(c.get("output_tokens") or 0 for c in calls),
            call_tokens_unknown=sum(c.get("output_tokens") is None for c in calls),
            observed_attempt_seconds=sum(row.get("elapsed_seconds", 0) for row in rows),
            interrupted_attempt_times_unknown=sum(
                row["status"] == "interrupted" and "elapsed_seconds" not in row
                for row in rows
            ),
            output_cap_reaches=sum(c.get("max_length_reached") is True for c in calls),
        )
        reports[arm] = report

    previous, revised = (reports[arm]["numeric_records"] for arm in ARMS)

    def paired(rows, event):
        before_after = [(bool(event(a)), bool(event(b))) for a, b in rows]
        return {
            "counts": dict(Counter(f"{int(a)}|{int(b)}" for a, b in before_after)),
            "previous_count": sum(a for a, _ in before_after),
            "revised_count": sum(b for _, b in before_after),
            "difference": interval([float(b) - float(a) for a, b in before_after]),
        }

    pairs = list(zip(previous, revised))
    return {
        "requested_per_arm": len(cases),
        "arms": reports,
        "paired": {
            "correct_abstention_unanswerable": paired(
                [(a, b) for a, b in pairs if not a["answerable"]],
                lambda row: row["category"] == "correct_abstention",
            ),
            "false_abstention_answerable": paired(
                [(a, b) for a, b in pairs if a["answerable"]],
                lambda row: row["category"] == "false_abstention",
            ),
            "exact_match_answerable": paired(
                [(a, b) for a, b in pairs if a["answerable"]],
                lambda row: row["exact_match"],
            ),
            "exact_match_all": paired(pairs, lambda row: row["exact_match"]),
        },
    }


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    design, cases = args.design.read_bytes(), args.cases.read_bytes()
    spec = json.loads(design)
    if sha(cases) != spec["private_cases_sha256"]:
        raise ValueError("Frozen questions, passages or references changed")
    submission = json.loads(args.submission.read_bytes())
    compute_raw = (args.terminal / "compute.json").read_bytes()
    identity_raw = (args.terminal / "identity.json").read_bytes()
    compute, identity = json.loads(compute_raw), json.loads(identity_raw)
    for key in (
        "source_commit",
        "bundle_sha256",
        "reserved_gpu_seconds",
        "platform_wall_cap_seconds",
    ):
        if compute[key] != submission[key]:
            raise ValueError(f"Compute identity changed: {key}")
    elapsed = compute["elapsed_seconds_including_setup"]
    reserved = compute["reserved_gpu_seconds"]
    charge = compute["charged_gpu_seconds"]
    if (
        compute["hosted_calls"] != 0
        or compute["paid_compute"] is not False
        or len(identity["allocated_devices"]) != 2
        or any("T4" not in x for x in identity["allocated_devices"])
        or (identity["model"], identity["revision"])
        != (spec["model"], spec["revision"])
        or compute["status"] not in {"completed", "failed_or_interrupted"}
        or not 0 < elapsed <= compute["platform_wall_cap_seconds"]
        or charge != 120 * math.ceil(elapsed / 60)
        or charge > reserved
        or submission["prior_attempt_charged_gpu_seconds"] + reserved
        != spec["reserved_gpu_seconds"]
    ):
        raise ValueError("Compute violates the frozen identity or accounting")
    checkpoints = [
        (args.terminal / f"private/device-{i}.json").read_bytes() for i in range(2)
    ]
    records = [row for checkpoint in checkpoints for row in json.loads(checkpoint)]
    result = score_pairs(spec, json.loads(cases), records)
    result["sha256"] = {
        "design": sha(design),
        "private_cases": sha(cases),
        "compute": sha(compute_raw),
        "identity": sha(identity_raw),
        "private_device_records": [sha(raw) for raw in checkpoints],
        "scorer": sha(Path(__file__).read_bytes()),
        "shared_scorer": sha(
            Path(__file__).with_name("score_english_reader_cpu.py").read_bytes()
        ),
    }
    with args.out.open("xb") as handle:
        handle.write((json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(result["paired"])


if __name__ == "__main__":
    main()
