"""Score the frozen paired CPU run from terminal private checkpoints."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.score_paired_abstention import score_pairs


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(release, cases_path, terminal, submission_path, out):
    design_raw = (release / "design.json").read_bytes()
    recovery = json.loads((release / "recovery.json").read_bytes())
    initial_raw = (release / "initial_compute.json").read_bytes()
    cases_raw = cases_path.read_bytes()
    submission_raw = submission_path.read_bytes()
    compute_raw = (terminal / "job_status.json").read_bytes()
    timing_raw = (terminal / "terminal_timing.json").read_bytes()
    records_raw = (terminal / "paired/private/records.json").read_bytes()
    design, submission, compute, timing = map(
        json.loads, (design_raw, submission_raw, compute_raw, timing_raw)
    )
    if (
        sha(design_raw) != submission["design_sha256"]
        or sha(cases_raw) != design["private_cases_sha256"]
        or sha(initial_raw) != recovery["previous_attempt_sha256"]
        or submission["version"] != 2
    ):
        raise ValueError("Submission or frozen study inputs changed")
    for key in ("source_commit", "bundle_sha256", "wall_reservation_seconds"):
        if compute[key] != submission[key]:
            raise ValueError(f"Compute identity differs: {key}")
    elapsed = compute["elapsed_seconds"]
    cap = recovery["retry_wall_reservation_seconds"]
    prior_debit = recovery["previous_cpu_debit_seconds"]
    ledger = design["cpu_ledger"]
    if (
        compute["status"] not in {"completed", "failed_or_interrupted"}
        or compute["gpu_seconds"] != 0
        or compute["hosted_calls"] != 0
        or compute["wall_reservation_seconds"] != cap
        or prior_debit != submission["previous_cpu_debit_seconds"]
        or prior_debit + cap != ledger["new_reservation_seconds"]
        or ledger["prior_total_seconds"] + prior_debit + cap > ledger["ceiling_seconds"]
        or not math.isfinite(elapsed)
        or not 0 < elapsed <= cap
    ):
        raise ValueError("Run violates the frozen compute budget")
    records = json.loads(records_raw)
    attempts = timing["attempts"]
    indices = [attempt["index"] for attempt in attempts]
    if (
        len(indices) != len(set(indices))
        or any(type(index) is not int for index in indices)
        or set(indices)
        != {i for i, row in enumerate(records) if row["status"] != "unserved"}
        or any(
            not math.isfinite(a["elapsed_seconds"]) or a["elapsed_seconds"] <= 0
            for a in attempts
        )
        or sum(a["elapsed_seconds"] for a in attempts) > elapsed + 1
    ):
        raise ValueError("Run times and checkpoint slots do not reconcile")
    for attempt in attempts:
        records[attempt["index"]]["elapsed_seconds"] = attempt["elapsed_seconds"]
    result = score_pairs(design, json.loads(cases_raw), records)
    charge = 60 * math.ceil(elapsed / 60)
    result["cpu_ledger"] = {
        "prior_debit_seconds": ledger["prior_total_seconds"],
        "first_attempt_debit_seconds": prior_debit,
        "retry_debit_seconds": charge,
        "total_cpu_debit_seconds": ledger["prior_total_seconds"] + prior_debit + charge,
        "ceiling_seconds": ledger["ceiling_seconds"],
        "gpu_seconds": 0,
        "hosted_calls": 0,
    }
    result["sha256"] = {
        "design": sha(design_raw),
        "private_cases": sha(cases_raw),
        "submission": sha(submission_raw),
        "compute": sha(compute_raw),
        "timing": sha(timing_raw),
        "private_records": sha(records_raw),
        "scorer": sha(Path(__file__).read_bytes()),
        "paired_scorer": sha(
            Path(__file__).with_name("score_paired_abstention.py").read_bytes()
        ),
        "shared_scorer": sha(
            Path(__file__).with_name("score_english_reader_cpu.py").read_bytes()
        ),
    }
    with out.open("xb") as handle:
        handle.write((json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "cases", "terminal", "submission", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.release, args.cases, args.terminal, args.submission, args.out)
    print(result["paired"])
