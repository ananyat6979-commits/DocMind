"""Score a frozen offline batch cohort and publish numeric results only."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.english_reader_batch_cpu import checked_decision
from scripts.score_english_reader_cpu import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "final"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest_raw = (args.profile / "input_manifest.json").read_bytes()
    pool_raw = (args.profile / "private/pool.json").read_bytes()
    decision_raw = args.decision.read_bytes()
    records_raw = args.records.read_bytes()
    manifest, pool = json.loads(manifest_raw), json.loads(pool_raw)
    decision, records = json.loads(decision_raw), json.loads(records_raw)
    if hashlib.sha256(pool_raw).hexdigest() != manifest["private_pool_sha256"]:
        raise ValueError("Private input pool differs from frozen manifest")
    if args.phase == "pilot":
        examples = pool["pilot"]
        expected = manifest["pilot"]
    else:
        size = checked_decision(decision, manifest)
        examples = pool["final"][:size]
        expected = decision["rows"]
    identities = [
        {key: row[key] for key in ("id", "context_sha256", "answerable")}
        for row in examples
    ]
    if identities != expected:
        raise ValueError("Scored questions differ from frozen cohort")
    if {row["id"] for row in records} != {row["id"] for row in expected}:
        raise ValueError("Every requested sequence requires a checkpoint slot")
    if any(row["status"] == "in_progress" for row in records):
        raise ValueError("Finalize interrupted status before scoring")

    result = summarize(examples, records)
    calls = [call for row in records for call in row.get("call_records", [])]
    result.update(
        phase=args.phase,
        status_counts=dict(Counter(row["status"] for row in records)),
        hold_reasons=dict(
            Counter(reason for row in records for reason in row.get("hold_reasons", []))
        ),
        attempted_native_batches=len(
            {row["batch_index"] for row in records if "batch_index" in row}
        ),
        completed_generation_token_records=len(calls),
        observed_input_tokens=sum(call["input_tokens"] for call in calls),
        observed_output_tokens=sum(call["output_tokens"] for call in calls),
        observed_output_cap_reaches=sum(
            bool(call["max_length_reached"]) for call in calls
        ),
        numeric_provenance={
            name: hashlib.sha256(raw).hexdigest()
            for name, raw in (
                ("input_manifest", manifest_raw),
                ("private_pool", pool_raw),
                ("cohort_decision", decision_raw),
                ("private_records", records_raw),
                ("batch_scoring_source", Path(__file__).read_bytes()),
                (
                    "shared_scoring_source",
                    Path(__file__)
                    .with_name("score_english_reader_cpu.py")
                    .read_bytes(),
                ),
                (
                    "interval_source",
                    Path(__file__)
                    .with_name("analyze_independent_review.py")
                    .read_bytes(),
                ),
            )
        },
    )
    with args.out.open("xb") as handle:
        handle.write((json.dumps(result, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
