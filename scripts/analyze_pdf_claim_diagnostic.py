"""Describe fixed PDF support predictions under separate original human references."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import math
import random

from scripts.analyze_independent_review import checked_reviews, percentile

DECISIVE = {"supported", "unsupported"}


def references(packet, links, originals):
    """Match original ratings only to the exact second-packet displayed inputs."""
    target = {
        (link["task_id"], link["input_sha256"], item["question"]): item
        for item in packet["items"]
        for link in links
        if link["id"] == item["id"]
    }
    if len(target) != len(packet["items"]):
        raise ValueError("Invalid target packet identity")
    output = {"first": {}, "second": {}}
    for role, source in originals.items():
        reviews, _ = checked_reviews(*source)
        group = "second" if role == "second" else "first"
        for key, item, rating in reviews:
            if key not in target:
                continue
            target_item = target[key]
            for claim in item["claims"]:
                unit = (target_item["id"], claim["id"])
                if unit in output[group]:
                    raise ValueError("Repeated matching human reference")
                output[group][unit] = rating["claims"][claim["id"]]
    output["concordant"] = {
        key: value
        for key, value in output["first"].items()
        if value in DECISIVE and output["second"].get(key) == value
    }
    return output


def checked_scores(records, design):
    expected = {
        (index, row["answer_id"], row["claim_id"]): row["input_sha256"]
        for index in range(len(design["models"]))
        for row in design["input_manifest"]["units"]
    }
    seen = set()
    for row in records:
        key = (row["model_index"], row["answer_id"], row["claim_id"])
        if key in seen or key not in expected or row["input_sha256"] != expected[key]:
            raise ValueError("Changed or duplicated score input")
        seen.add(key)
        if row["status"] == "completed":
            score = row["score"]
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError("Invalid completed probability")
        elif (
            row["status"]
            not in {
                "unserved",
                "missing_evidence",
                "claim_budget",
                "model_load_error",
                "scoring_error",
            }
            or row.get("score") is not None
        ):
            raise ValueError("Invalid missing prediction status")
    if seen != set(expected):
        raise ValueError("Requested prediction rows must remain in the denominator")


def confusion(pairs, threshold):
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for score, support in pairs:
        predicted = score >= threshold
        counts[
            ("tp" if predicted else "fn") if support else ("fp" if predicted else "tn")
        ] += 1
    return {
        **counts,
        "n": len(pairs),
        "precision": counts["tp"] / (counts["tp"] + counts["fp"])
        if counts["tp"] + counts["fp"]
        else None,
        "recall": counts["tp"] / (counts["tp"] + counts["fn"])
        if counts["tp"] + counts["fn"]
        else None,
    }


def intervals(clusters, threshold):
    rng = random.Random(19)
    values = {"precision": [], "recall": []}
    for _ in range(2000):
        pairs = (
            [pair for _ in clusters for pair in clusters[rng.randrange(len(clusters))]]
            if clusters
            else []
        )
        metrics = confusion(pairs, threshold)
        for name in values:
            if metrics[name] is not None:
                values[name].append(metrics[name])
    return {
        name: {
            "ci95": [percentile(v, 0.025), percentile(v, 0.975)] if v else None,
            "defined_resamples": len(v),
            "undefined_resamples": 2000 - len(v),
        }
        for name, v in values.items()
    }


def summarize(records, design, refs):
    checked_scores(records, design)
    answer_ids = sorted({r["answer_id"] for r in design["input_manifest"]["units"]})
    requested = {
        (r["answer_id"], r["claim_id"]) for r in design["input_manifest"]["units"]
    }
    if any(not set(reference) <= requested for reference in refs.values()):
        raise ValueError("Human labels refer to different claim inputs")
    output = {
        "requested_answers": len(answer_ids),
        "requested_claims": len(requested),
        "scope": design["scope"],
        "population_false_release_risk": None,
        "models": [],
    }
    for index, model in enumerate(design["models"]):
        rows = [row for row in records if row["model_index"] == index]
        block = {
            "identity": model,
            "prediction_statuses": dict(Counter(r["status"] for r in rows)),
            "premise_clipped_completed": sum(
                bool(r.get("premise_clipped"))
                for r in rows
                if r["status"] == "completed"
            ),
            "references": {},
        }
        for name, reference in refs.items():
            clusters = {answer_id: [] for answer_id in answer_ids}
            completed_without_decisive_reference = 0
            labels_without_prediction = 0
            for row in rows:
                key = (row["answer_id"], row["claim_id"])
                label = reference.get(key)
                if row["status"] == "completed" and label in DECISIVE:
                    clusters[row["answer_id"]].append(
                        (row["score"], label == "supported")
                    )
                elif row["status"] == "completed":
                    completed_without_decisive_reference += 1
                elif label in DECISIVE:
                    labels_without_prediction += 1
            pairs = [pair for values in clusters.values() for pair in values]
            block["references"][name] = {
                "decisive_labels": sum(v in DECISIVE for v in reference.values()),
                "unclear_labels": sum(v == "unclear" for v in reference.values()),
                "missing_or_excluded_labels": len(requested) - len(reference),
                "completed_without_decisive_reference": completed_without_decisive_reference,
                "decisive_labels_without_prediction": labels_without_prediction,
                "answer_clusters_with_scored_labels": sum(
                    bool(v) for v in clusters.values()
                ),
                "thresholds": {
                    str(threshold): {
                        **confusion(pairs, threshold),
                        "cluster_bootstrap": intervals(
                            list(clusters.values()), threshold
                        ),
                    }
                    for threshold in (
                        design["primary_threshold"],
                        design["descriptive_sensitivity_threshold"],
                    )
                },
            }
        output["models"].append(block)
    output["paired_common_input_differences"] = {}
    scored = {
        (row["model_index"], row["answer_id"], row["claim_id"]): row["score"]
        for row in records
        if row["status"] == "completed"
    }
    for name, reference in refs.items():
        common = {answer_id: [] for answer_id in answer_ids}
        for (answer_id, claim_id), label in reference.items():
            a, b = (0, answer_id, claim_id), (1, answer_id, claim_id)
            if label in DECISIVE and a in scored and b in scored:
                common[answer_id].append((scored[a], scored[b], label == "supported"))
        threshold = design["primary_threshold"]

        def difference(pairs):
            left = confusion([(a, label) for a, _, label in pairs], threshold)
            right = confusion([(b, label) for _, b, label in pairs], threshold)
            return {
                metric: right[metric] - left[metric]
                if left[metric] is not None and right[metric] is not None
                else None
                for metric in ("precision", "recall")
            }

        point = difference([p for group in common.values() for p in group])
        rng = random.Random(19)
        values = {"precision": [], "recall": []}
        groups = list(common.values())
        for _ in range(2000):
            sample = (
                [p for _ in groups for p in groups[rng.randrange(len(groups))]]
                if groups
                else []
            )
            delta = difference(sample)
            for metric in values:
                if delta[metric] is not None:
                    values[metric].append(delta[metric])
        output["paired_common_input_differences"][name] = {
            "direction": "comparison minus legacy, primary threshold only",
            "threshold": threshold,
            "common_claims": sum(len(v) for v in groups),
            "metrics": {
                metric: {
                    "difference": point[metric],
                    "ci95": [percentile(v, 0.025), percentile(v, 0.975)] if v else None,
                    "defined_resamples": len(v),
                    "undefined_resamples": 2000 - len(v),
                }
                for metric, v in values.items()
            },
        }
    output["uncertainty"] = (
        "Nominal seed-19 answer-cluster bootstrap, 2000 resamples; undefined ratios remain explicit. Selected cohort, no population guarantee."
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--originals", type=Path, required=True)
    parser.add_argument("--reference-snapshot", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    snapshot = read(args.reference_snapshot)
    manifest = read(args.originals)
    if set(manifest) != {"first-original", "first-changed", "second"}:
        raise ValueError("All original reference sources required")
    originals, export_hashes, primary_hashes = {}, {}, set()
    for role, source in manifest.items():
        root, exported = Path(source["root"]), Path(source["labels"])
        export_hashes[role] = hashlib.sha256(exported.read_bytes()).hexdigest()
        original_packet = read(root / "actual-answer-packet.private.json")
        if role == "second":
            if (
                export_hashes[role] != snapshot["input_export_sha256"]["second"]
                or original_packet["sha256"] != snapshot["packet_sha256"]
            ):
                raise ValueError("Original second reference changed")
        else:
            primary_hashes.add(original_packet["sha256"])
        originals[role] = (
            original_packet,
            read(root / "actual-answer-linkage.private.json"),
            read(exported),
        )
    if primary_hashes != set(snapshot["primary_packet_sha256"]) or {
        export_hashes[role] for role in ("first-original", "first-changed")
    } != set(snapshot["input_export_sha256"]["primary"]):
        raise ValueError("Original first references changed")
    packet = read(args.packet_root / "actual-answer-packet.private.json")
    links = read(args.packet_root / "actual-answer-linkage.private.json")
    design = read(args.design)
    if packet["sha256"] != design["input_manifest"]["source_packet_sha256"]:
        raise ValueError("Scored source packet changed")
    identity = read(args.scores / "identity.json")
    from scripts.pdf_claim_diagnostic import digest

    if identity["payload_sha256"] != design["input_manifest"][
        "payload_sha256"
    ] or identity["design_sha256"] != digest(design):
        raise ValueError("Scoring identity mismatch")
    if identity["gpu_seconds"] != 0 or identity["hosted_calls"] != 0:
        raise ValueError("Unexpected compute allocation")
    for model in identity["models"]:
        expected = design["models"][model["model_index"]]
        if (
            model["model"] != expected["model"]
            or model["revision"] != expected["revision"]
            or model["dtype"] != "torch.float32"
        ):
            raise ValueError("Scoring model identity mismatch")
    result = summarize(
        read(args.scores / "numeric_records.json"),
        design,
        references(packet, links, originals),
    )
    result["input_exports_sha256"] = export_hashes
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["numeric_records_sha256"] = hashlib.sha256(
        (args.scores / "numeric_records.json").read_bytes()
    ).hexdigest()
    result["identity_sha256"] = hashlib.sha256(
        (args.scores / "identity.json").read_bytes()
    ).hexdigest()
    result["reference_snapshot_sha256"] = hashlib.sha256(
        args.reference_snapshot.read_bytes()
    ).hexdigest()
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
