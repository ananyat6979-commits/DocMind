"""Score explicit reader decisions without rewarding failed or blank outputs."""

from collections import Counter
import random
import re
import string

from scripts.analyze_independent_review import percentile


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def pair_scores(prediction, reference):
    left, right = normalize(prediction), normalize(reference)
    exact = float(left == right)
    a, b = left.split(), right.split()
    if not a or not b:
        return exact, float(a == b)
    overlap = sum((Counter(a) & Counter(b)).values())
    return exact, 2 * overlap / (len(a) + len(b))


def score(row, example):
    if row["status"] != "completed":
        return {"exact_match": 0.0, "token_f1": 0.0, "category": row["status"]}
    if row.get("finish_reason") == "format_failure":
        return {"exact_match": 0.0, "token_f1": 0.0, "category": "format_failure"}
    if row.get("abstained") is True:
        value = float(not example["answerable"])
        return {
            "exact_match": value,
            "token_f1": value,
            "category": "false_abstention"
            if example["answerable"]
            else "correct_abstention",
        }
    short = row.get("short_answer")
    if not isinstance(short, str) or not short.strip():
        return {"exact_match": 0.0, "token_f1": 0.0, "category": "invalid_metric_field"}
    if not example["answerable"]:
        return {
            "exact_match": 0.0,
            "token_f1": 0.0,
            "category": "answered_unanswerable",
        }
    references = [answer["text"] for answer in example["answers"]]
    if not references:
        raise ValueError("Answerable public reference has no answer span")
    values = [pair_scores(short, reference) for reference in references]
    return {
        "exact_match": max(v[0] for v in values),
        "token_f1": max(v[1] for v in values),
        "category": "short_answer_scored",
    }


def interval(values):
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    rng = random.Random(19)
    means = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(2000)
    ]
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "ci95": [percentile(means, 0.025), percentile(means, 0.975)],
    }


def summarize(examples, records):
    lookup = {row["id"]: row for row in records}
    expected = {row["id"] for row in examples}
    if (
        len(lookup) != len(records)
        or len(expected) != len(examples)
        or not set(lookup) <= expected
    ):
        raise ValueError("Repeated or unexpected experiment records")
    numeric = []
    for example in examples:
        row = lookup.get(example["id"], {"id": example["id"], "status": "unserved"})
        if (
            row["id"] in lookup
            and row.get("context_sha256") != example["context_sha256"]
        ):
            raise ValueError("Changed paragraph cannot reuse a public reference")
        numeric.append(
            {
                "id": example["id"],
                "context_sha256": example["context_sha256"],
                "answerable": example["answerable"],
                "status": row["status"],
                "release_state": row.get("release_state"),
                **score(row, example),
            }
        )
    output = {
        "requested_questions": len(examples),
        "numeric_records": numeric,
        "scope": "Selected public-reference supplied-paragraph CPU profile; short-answer metrics are not human full-answer correctness",
        "categories": dict(Counter(r["category"] for r in numeric)),
        "requested_cohort": {},
        "completed_only": {},
        "strata": {},
        "always_abstain_baseline": interval(
            [float(not e["answerable"]) for e in examples]
        ),
        "population_release_risk": None,
        "completed_release_states": dict(
            Counter(
                r.get("release_state", "unknown")
                for r in records
                if r["status"] == "completed"
            )
        ),
    }
    for metric in ("exact_match", "token_f1"):
        output["requested_cohort"][metric] = interval([r[metric] for r in numeric])
        output["completed_only"][metric] = interval(
            [r[metric] for r in numeric if r["status"] == "completed"]
        )
    for answerable in (True, False):
        group = [r for r in numeric if r["answerable"] == answerable]
        output["strata"]["answerable" if answerable else "unanswerable"] = {
            metric: interval([r[metric] for r in group])
            for metric in ("exact_match", "token_f1")
        }
    return output
