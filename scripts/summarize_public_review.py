"""Validate a private human export and publish only aggregate audit counts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from docmind.evaluation.release import digest


def summarize(export, key, source_rows):
    if export.get("schema") != "public-label-review-v1" or export.get("independent_review") is not True:
        raise ValueError("Independent human-review attestation required")
    if export.get("dataset") != "facebook/xnli" or export.get("language") != "hi":
        raise ValueError("Wrong review source")
    seconds = export.get("active_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 <= seconds <= 1500.01:
        raise ValueError("Review exceeds the main-session budget")
    expected = {r["review_id"]: r for r in key["tasks"]}
    rows = export["records"]
    if len(rows) != len(expected) or len({r["review_id"] for r in rows}) != len(rows) or {r["review_id"] for r in rows} != set(expected):
        raise ValueError("Review cohort mismatch")
    counts = dict(adjudicated=0, unclear=0, unfinished=0, disagreements=0)
    for row in rows:
        task = expected[row["review_id"]]
        source = source_rows[task["source_index"]]
        if row["input_sha256"] != digest([source["premise"], source["hypothesis"]]) or source["label"] != task["original_label"]:
            raise ValueError("Review input mismatch")
        label = row["label"]
        if label is None:
            if row.get("finished") is not False: raise ValueError("Invalid unfinished label")
            counts["unfinished"] += 1
        elif label == "unclear":
            if row.get("finished") is not True: raise ValueError("Invalid unclear label")
            counts["unclear"] += 1
        elif type(label) is int and label in (0, 1, 2) and row.get("finished") is True:
            counts["adjudicated"] += 1
            counts["disagreements"] += int(label != source["label"])
        else:
            raise ValueError("Invalid human label")
    return {"scope": "Single-reviewer Hindi XNLI test-label audit; not PDF attribution", "n_preselected": len(rows), **counts,
            "active_seconds": seconds, "disagreement_rate_among_decisive_labels": counts["disagreements"] / counts["adjudicated"] if counts["adjudicated"] else None,
            "population_label_error_rate": None, "consistency": "unmeasured", "inter_annotator_agreement": "not applicable",
            "limits": ["Small randomized subset of a 200-item diagnostic cohort", "Unclear labels are separate from confirmed disagreements", "No population extrapolation from the decisive subset", "One reviewer; focused-item timings are not precise annotation durations" ]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    key = json.loads(args.private_key.read_text(encoding="utf-8"))
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != key["source_sha256"]: raise ValueError("Source hash mismatch")
    result = summarize(json.loads(args.export.read_text(encoding="utf-8")), key, pq.read_table(args.source).to_pylist())
    result["private_export_sha256"] = hashlib.sha256(args.export.read_bytes()).hexdigest()
    result["source_revision"] = key["revision"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
