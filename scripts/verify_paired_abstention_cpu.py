"""Verify the published numeric paired results without private model outputs."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from scripts.score_english_reader_cpu import interval


def verify(release: Path) -> None:
    final = release / "final"
    manifest = json.loads((final / "manifest.json").read_bytes())["sha256"]
    names = {"results.json", "compute.json", "terminal_timing.json"}
    if set(manifest) != names:
        raise ValueError("Unexpected public artifact inventory")

    records = {}
    for name in sorted(names):
        raw = (final / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest[name]:
            raise ValueError(f"Artifact hash mismatch: {name}")
        records[name] = json.loads(raw)

    result = records["results.json"]
    compute = records["compute.json"]
    timing = records["terminal_timing.json"]
    raw_design = (release / "design.json").read_bytes()
    design = json.loads(raw_design)
    if (
        result["sha256"]["design"] != hashlib.sha256(raw_design).hexdigest()
        or result["sha256"]["compute"] != manifest["compute.json"]
        or result["sha256"]["timing"] != manifest["terminal_timing.json"]
    ):
        raise ValueError(
            "Result identity does not match the design or terminal artifacts"
        )

    fields = ("id", "context_sha256", "answerable")
    for name, arm in result["arms"].items():
        rows = arm["numeric_records"]
        if (
            [{field: row[field] for field in fields} for row in rows] != design["rows"]
            or Counter(row["status"] for row in rows) != arm["status_counts"]
            or Counter(row["category"] for row in rows) != arm["categories"]
            or Counter(
                row["release_state"] for row in rows if row["status"] == "completed"
            )
            != arm["completed_release_states"]
        ):
            raise ValueError(f"Cohort or row summaries differ for {name}")
        for metric in ("exact_match", "token_f1"):
            if (
                interval([row[metric] for row in rows])
                != arm["requested_cohort"][metric]
            ):
                raise ValueError(f"Metric differs for {name}: {metric}")

    previous = result["arms"]["previous"]["numeric_records"]
    revised = result["arms"]["revised"]["numeric_records"]
    endpoints = {
        "correct_abstention_unanswerable": (False, "category", "correct_abstention"),
        "false_abstention_answerable": (True, "category", "false_abstention"),
        "exact_match_answerable": (True, "exact_match", 1.0),
        "exact_match_all": (None, "exact_match", 1.0),
    }
    for name, (stratum, metric, target) in endpoints.items():
        outcomes = [
            (int(a[metric] == target), int(b[metric] == target))
            for a, b in zip(previous, revised, strict=True)
            if stratum is None or a["answerable"] == stratum
        ]
        expected = {
            "counts": dict(Counter(f"{a}|{b}" for a, b in outcomes)),
            "previous_count": sum(a for a, _ in outcomes),
            "revised_count": sum(b for _, b in outcomes),
            "difference": interval([float(b) - float(a) for a, b in outcomes]),
        }
        if result["paired"][name] != expected:
            raise ValueError(f"Paired result differs: {name}")

    ledger = result["cpu_ledger"]
    elapsed = compute["elapsed_seconds"]
    attempts = timing["attempts"]
    total_attempt_time = sum(a["elapsed_seconds"] for a in attempts)
    if (
        [a["index"] for a in attempts] != list(range(2 * result["requested_per_arm"]))
        or any(
            not a["generation_completed"] or a["elapsed_seconds"] <= 0 for a in attempts
        )
        or not math.isclose(
            total_attempt_time,
            sum(arm["observed_attempt_seconds"] for arm in result["arms"].values()),
        )
        or total_attempt_time > elapsed
        or compute["status"] != "completed"
        or not 0 < elapsed <= compute["wall_reservation_seconds"]
        or compute["gpu_seconds"] != 0
        or compute["hosted_calls"] != 0
        or ledger["retry_debit_seconds"] != 60 * math.ceil(elapsed / 60)
        or ledger["total_cpu_debit_seconds"]
        != ledger["prior_debit_seconds"]
        + ledger["first_attempt_debit_seconds"]
        + ledger["retry_debit_seconds"]
        or ledger["total_cpu_debit_seconds"] > ledger["ceiling_seconds"]
    ):
        raise ValueError("Published timing or compute ledger does not reconcile")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    verify(parser.parse_args().release)
    print("Paired metrics, artifact hashes and compute ledger verified.")
