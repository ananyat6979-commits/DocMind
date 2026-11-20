"""Compare private human ratings only for identical question/answer/evidence inputs."""

import argparse
import hashlib
import json
import platform
import random
from collections import Counter
from pathlib import Path

from scripts.analyze_own_answer_review import summarize

CONTEXT_FIELDS = ("different_person", "prior_involvement", "seen_prior_results")


def agreement(pairs):
    n = len(pairs)
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    observed = sum(a == b for a, b in pairs) / n if n else None
    expected = (
        sum(left[label] * right[label] for label in left.keys() | right.keys()) / n**2
        if n
        else None
    )
    kappa = (observed - expected) / (1 - expected) if n and expected < 1 else None
    return {
        "paired_labels": n,
        "observed_agreement": observed,
        "cohen_kappa": kappa,
        "confusion": dict(Counter(a + "|" + b for a, b in pairs)),
        "kappa_status": "defined, exploratory"
        if kappa is not None
        else "undefined: no pairs or degenerate marginals",
    }


def percentile(values, p):
    ordered = sorted(values)
    at = (len(ordered) - 1) * p
    lo = int(at)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def bootstrap(clusters, kind):
    if len(clusters) < 10:
        return {
            "kappa_interval95": None,
            "status": "insufficient paired answer clusters; minimum ten",
            "paired_answer_clusters": len(clusters),
            "resamples": 0,
        }
    rng = random.Random(19)
    values = []
    for _ in range(2000):
        picked = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        pairs = [pair for cluster in picked for pair in cluster[kind]]
        value = agreement(pairs)["cohen_kappa"]
        if value is not None:
            values.append(value)
    return {
        "kappa_interval95": [percentile(values, 0.025), percentile(values, 0.975)]
        if len(values) >= 2
        else None,
        "resamples": 2000,
        "defined_resamples": len(values),
        "undefined_resamples": 2000 - len(values),
        "paired_answer_clusters": len(clusters),
        "status": "Nominal answer-cluster bootstrap, conditional on defined kappa; descriptive only",
    }


def checked_reviews(packet, links, labels):
    content = {key: value for key, value in packet.items() if key != "sha256"}
    actual = hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()
    if actual != packet["sha256"]:
        raise ValueError("Review packet content changed")
    summary = summarize(packet, links, labels)
    items = {item["id"]: item for item in packet["items"]}
    lookup = {link["id"]: link for link in links}
    result = []
    for rating in labels["records"]:
        if not rating.get("finished") or not rating.get("human_adjudicated"):
            continue
        item, link = items[rating["id"]], lookup[rating["id"]]
        payload = {"answer": item["answer"], "claims": item["claims"]}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        if digest != link["input_sha256"]:
            raise ValueError("Changed displayed inputs cannot reuse a judgment")
        result.append(((link["task_id"], digest, item["question"]), item, rating))
    return result, summary


def compare(primary_sources, second_source):
    packet, links, labels = second_source
    if (
        packet.get("review_role") != "second"
        or labels.get("reviewer_code") != "R2"
        or labels.get("schema") != "second-review-v1"
    ):
        raise ValueError("A distinct second-reviewer export is required")
    context = labels.get("reviewer_attestation", {})
    if any(context.get(key) not in ("yes", "no", "unsure") for key in CONTEXT_FIELDS):
        raise ValueError("Missing reviewer-context declaration")
    second, summary = checked_reviews(packet, links, labels)
    primary = {}
    for source in primary_sources:
        records, _ = checked_reviews(*source)
        for key, item, rating in records:
            if key in primary:
                raise ValueError("Repeated reference judgments require adjudication")
            primary[key] = (item, rating)
    clusters, unmatched = [], 0
    for key, item, second_rating in second:
        if key not in primary:
            unmatched += 1
            continue
        original_item, first = primary[key]
        if original_item["claims"] != item["claims"]:
            raise ValueError("Claim units differ despite input identity")
        clusters.append(
            {
                "answers": [
                    (first["answer_correctness"], second_rating["answer_correctness"])
                ],
                "claims": [
                    (first["claims"][claim["id"]], second_rating["claims"][claim["id"]])
                    for claim in item["claims"]
                ],
            }
        )
    independent = (
        context["different_person"] == "yes"
        and context["prior_involvement"] == "no"
        and context["seen_prior_results"] == "no"
    )
    output = {
        "scope": "Same-input paired human judgments on a selected English diagnostic cohort; not independent study replication",
        "independence_self_report": {key: context[key] for key in CONTEXT_FIELDS},
        "independence_status": "Second-person independent judgments self-reported, not externally verified"
        if independent
        else "Independence not established; paired rating agreement only",
        "requested_second_review_answers": len(packet["items"]),
        "finished_second_review_answers": summary["finished_review_answers"],
        "paired_answers": len(clusters),
        "completed_without_matching_reference": unmatched,
        "unfinished_answers": summary["unfinished_or_missing"],
        "recorded_second_review_seconds": summary["recorded_running_seconds"],
        "active_work_independently_observed": False,
        "packet_sha256": packet["sha256"],
        "adjudication": "Not performed; neither reviewer overwrites the other",
        "population_false_release_risk": None,
        "limitations": [
            "Small selected cohort; incomplete time-based completion has no population weights",
            "Unclear labels remain a separate category rather than being discarded",
            "Claim intervals resample answer clusters, not individual claims",
            "No claim-extraction gold, new reader generation or Hindi validation",
            "Agreement is not correctness or calibrated release safety",
        ],
    }
    for kind in ("answers", "claims"):
        output[kind] = {
            **agreement([p for c in clusters for p in c[kind]]),
            **bootstrap(clusters, kind),
        }
    return output


def load(root, export):
    return (
        json.loads(
            (root / "actual-answer-packet.private.json").read_text(encoding="utf-8")
        ),
        json.loads(
            (root / "actual-answer-linkage.private.json").read_text(encoding="utf-8")
        ),
        json.loads(export.read_text(encoding="utf-8")),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--second-root", type=Path, required=True)
    parser.add_argument("--second-labels", type=Path, required=True)
    parser.add_argument("--primary-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--primary-labels", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.primary_roots) != len(args.primary_labels):
        parser.error("Each primary packet needs its own label export")
    if args.out.exists():
        raise FileExistsError("Agreement snapshots are immutable")
    result = compare(
        [
            load(root, labels)
            for root, labels in zip(
                args.primary_roots, args.primary_labels, strict=True
            )
        ],
        load(args.second_root, args.second_labels),
    )
    result["input_export_sha256"] = {
        "second": hashlib.sha256(args.second_labels.read_bytes()).hexdigest(),
        "primary": [
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.primary_labels
        ],
    }
    result["primary_packet_sha256"] = [
        json.loads(
            (root / "actual-answer-packet.private.json").read_text(encoding="utf-8")
        )["sha256"]
        for root in args.primary_roots
    ]
    result["python_version"] = platform.python_version()
    result["analysis_code_sha256"] = {
        name: hashlib.sha256(
            (Path(__file__).parent / name).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for name in ("analyze_independent_review.py", "analyze_own_answer_review.py")
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
