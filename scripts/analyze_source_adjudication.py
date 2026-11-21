"""Keep source adjudication separate from original independent human votes."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from scripts.analyze_independent_review import checked_reviews


def analyze(packet, links, labels, originals):
    if packet.get("review_role") != "source_adjudication":
        raise ValueError("Source-adjudication packet required")
    if labels.get("schema") != "joint-adjudication-v1":
        raise ValueError("Separate adjudication export required")
    mode = labels.get("participant_mode")
    if mode not in ("joint", "single"):
        raise ValueError("Participant declaration required")
    if packet.get("cap_seconds") != 600:
        raise ValueError("Adjudication allowance must remain ten minutes")
    if packet.get("original_second_packet_sha256") != originals.get("second", ({},))[
        0
    ].get("sha256"):
        raise ValueError("Original second-review packet mismatch")
    by_role = {}
    for role, source in originals.items():
        records, _ = checked_reviews(*source)
        by_role[role] = {key: (item, rating) for key, item, rating in records}
        if len(by_role[role]) != len(records):
            raise ValueError("Repeated original review inputs")
    if set(by_role) != {"first-original", "first-changed", "second"}:
        raise ValueError("All original review sources are required")
    first = dict(by_role["first-original"])
    if first.keys() & by_role["first-changed"].keys():
        raise ValueError("Ambiguous first-review inputs")
    first.update(by_role["first-changed"])
    disputed = {}
    for key, (item, right) in by_role["second"].items():
        if key not in first:
            continue
        _, left = first[key]
        if left["answer_correctness"] != right["answer_correctness"] or any(
            left["claims"][c["id"]] != right["claims"][c["id"]] for c in item["claims"]
        ):
            disputed[key] = (left, right)
    # Check every packet input, including unfinished cases, against the source votes.
    synthetic = dict(labels, records=[])
    checked_reviews(packet, links, synthetic)
    lookup = {link["id"]: link for link in links}
    seen = set()
    for item in packet["items"]:
        digest = hashlib.sha256(
            json.dumps(
                {"answer": item["answer"], "claims": item["claims"]}, sort_keys=True
            ).encode()
        ).hexdigest()
        link = lookup[item["id"]]
        key = (link["task_id"], digest, item["question"])
        if digest != link["input_sha256"] or key not in disputed or key in seen:
            raise ValueError("Adjudication inputs differ from the disputed originals")
        seen.add(key)
        left, right = disputed[key]
        if item.get("prior_answer") != {
            "first": left["answer_correctness"],
            "second": right["answer_correctness"],
        } or item.get("prior_claims") != {
            c["id"]: {
                "first": left["claims"][c["id"]],
                "second": right["claims"][c["id"]],
            }
            for c in item["claims"]
        }:
            raise ValueError("Displayed original votes changed")
    if seen != set(disputed):
        raise ValueError("Disputed cases were omitted")
    completed, summary = checked_reviews(packet, links, labels)
    for _, _, rating in completed:
        if len(rating.get("note", "").strip()) < 10:
            raise ValueError("A source-based rationale is required")
    if completed and summary["recorded_running_seconds"] <= 0:
        raise ValueError("Completed adjudication requires recorded time")
    return {
        "scope": "Joint source adjudication"
        if mode == "joint"
        else "Single-person source reconsideration",
        "participant_mode": mode,
        "packet_sha256": packet["sha256"],
        "requested_cases": len(packet["items"]),
        "finished_cases": len(completed),
        "unfinished_cases": len(packet["items"]) - len(completed),
        "recorded_seconds": summary["recorded_running_seconds"],
        "answer_labels": dict(
            Counter(r["answer_correctness"] for _, _, r in completed)
        ),
        "claim_labels": dict(
            Counter(v for _, _, r in completed for v in r["claims"].values())
        ),
        "original_votes_unchanged": True,
        "independent_new_review": False,
        "recomputed_independent_agreement": False,
        "population_false_release_risk": None,
        "limitations": [
            "Disagreement-selected cases, no population extrapolation",
            "Participant presence and active work are self-reported",
            "Unclear judgments and unfinished cases remain unresolved",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--originals",
        type=Path,
        required=True,
        help="Private JSON mapping roles to root and labels paths",
    )
    parser.add_argument("--budget-before", type=Path, required=True)
    parser.add_argument("--allowance-seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    packet = read(args.root / "actual-answer-packet.private.json")
    originals = {}
    for role, source in read(args.originals).items():
        label_path = Path(source["labels"])
        if hashlib.sha256(label_path.read_bytes()).hexdigest() != packet[
            "input_exports_sha256"
        ].get(role):
            raise ValueError("Original export hash mismatch")
        root = Path(source["root"])
        originals[role] = (
            read(root / "actual-answer-packet.private.json"),
            read(root / "actual-answer-linkage.private.json"),
            read(label_path),
        )
    result = analyze(
        packet,
        read(args.root / "actual-answer-linkage.private.json"),
        read(args.labels),
        originals,
    )
    before = read(args.budget_before)
    charged = before["charged_total_seconds"] + result["recorded_seconds"]
    if not math.isfinite(args.allowance_seconds) or not (
        before["total_pdf_review_allowance_seconds"] <= args.allowance_seconds
        and 0 <= charged <= args.allowance_seconds
    ):
        raise ValueError(
            "Adjudication exceeds the separately authorized total allowance"
        )
    result["budget"] = {
        "allowance_seconds": args.allowance_seconds,
        "charged_before_seconds": before["charged_total_seconds"],
        "charged_total_seconds": charged,
        "remaining_seconds": args.allowance_seconds - charged,
        "prior_budget_sha256": hashlib.sha256(
            args.budget_before.read_bytes()
        ).hexdigest(),
    }
    result["export_sha256"] = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    result["original_exports_sha256"] = packet["input_exports_sha256"]
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
