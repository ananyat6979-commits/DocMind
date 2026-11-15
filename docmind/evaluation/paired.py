"""Frozen paired designs and requested-denominator metrics, without model calls."""
import hashlib
import json
import math
import random
import re
import string
from typing import Any


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def normalize_answer(value):
    text = str(value).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score(answer, reference):
    prediction, gold = normalize_answer(answer).split(), normalize_answer(reference).split()
    from collections import Counter
    overlap = sum((Counter(prediction) & Counter(gold)).values())
    f1 = 2 * overlap / (len(prediction) + len(gold)) if prediction or gold else 1.0
    return {"exact_match": int(prediction == gold), "token_f1": f1}


def balanced_pool(rows, excluded, count=120):
    """Deterministic 60/40 mix, with no inspected model outcomes."""
    groups: dict[str, list[dict]] = {kind: [] for kind in ("bridge", "comparison")}
    seen = set()
    for row in sorted(rows, key=lambda r: hashlib.sha256(("docmind-core-v1:"+r["id"]).encode()).hexdigest()):
        if row["id"] in excluded or row["id"] in seen:continue
        seen.add(row["id"])
        if row.get("type") in groups:groups[row["type"]].append(row)
    n = min(count // 5, len(groups["bridge"]) // 3, len(groups["comparison"]) // 2) * 5
    selected = groups["bridge"][:n*3//5] + groups["comparison"][:n*2//5]
    random.Random(7).shuffle(selected)
    return selected


def freeze_design(pool, pilot_pair_seconds, remaining_seconds, max_n=100):
    if len(pilot_pair_seconds) != 10 or any(not math.isfinite(t) or t <= 0 for t in pilot_pair_seconds):
        raise ValueError("Ten valid timing-pilot pairs are required")
    if len({row["id"] for row in pool}) != len(pool):raise ValueError("Duplicate cohort identities")
    worst = max(pilot_pair_seconds) * 1.25
    # Adaptive diagnostic is dropped first; main cohort uses the conservative envelope.
    n = min(max_n, len(pool), max(0, int(remaining_seconds*.8/worst))) // 5 * 5
    cohort = balanced_pool(pool, set(), n)
    records = [{"id":row["id"], "type":row["type"],
        "order":["single", "react"] if index % 2 == 0 else ["react", "single"]} for index,row in enumerate(cohort)]
    return {"version":"paired-core-v1", "seed":7, "rows":records, "cohort_sha256":digest(records),
        "pilot_worst_pair_seconds":max(pilot_pair_seconds), "timing_margin":1.25, "reserve_fraction":.2,
        "selection_uses_outcomes":False, "adaptive_first_to_drop":True,
        "scope":"English local-reader frozen-evidence cost-benefit control"}


def summarize(design, records):
    import numpy as np
    requested = design["rows"]
    lookup = {(r["id"],r["arm"]):r for r in records}
    if len(lookup) != len(records):raise ValueError("Duplicate experiment attempts")
    allowed = {(r["id"],arm) for r in requested for arm in ("single", "react")}
    if not set(lookup) <= allowed:raise ValueError("Record outside frozen design")
    summary: dict[str, Any] = {"scope":design["scope"], "cohort_sha256":design["cohort_sha256"], "arms":{}}
    vectors = {}
    for arm in ("single", "react"):
        rows = [lookup.get((r["id"], arm), {"status":"unserved", "exact_match":0}) for r in requested]
        completed = [r for r in rows if r["status"] == "completed"]
        correct = sum(r.get("exact_match",0) for r in rows)
        vectors[arm] = np.array([r.get("exact_match",0) for r in rows], dtype=float)
        def per_correct(key):
            values = [r.get(key) for r in rows if r["status"] != "unserved"]
            return sum(values)/correct if correct and all(v is not None for v in values) else None
        summary["arms"][arm] = {"requested":len(rows),"completed":len(completed),"correct":correct,
            "requested_exact_match":correct/len(rows) if rows else None,
            "completed_exact_match":sum(r.get("exact_match",0) for r in completed)/len(completed) if completed else None,
            "statuses":{status:sum(r["status"]==status for r in rows) for status in sorted({r["status"] for r in rows})},
            "format_failures":sum(bool(r.get("format_failure")) for r in rows),
            "truncations":sum(bool(r.get("truncated")) for r in rows),
            "calls_per_correct":per_correct("calls"),"tokens_per_correct":per_correct("tokens"),
            "seconds_per_correct":per_correct("elapsed_seconds"),
            "short_answer_length_strata":{name:{"n":sum(predicate(r) for r in completed),
                "correct":sum(r.get("exact_match",0) for r in completed if predicate(r))} for name,predicate in
                (("0_to_8_words",lambda r:r.get("short_answer_words",0)<=8),("over_8_words",lambda r:r.get("short_answer_words",0)>8))}}
    differences = vectors["react"]-vectors["single"]
    rng = np.random.default_rng(7)
    if len(differences):
        boot = [float(rng.choice(differences,len(differences),replace=True).mean()) for _ in range(2000)]
        summary["paired_requested_em_difference"] = {"react_minus_single":float(differences.mean()),
            "bootstrap95_nominal":np.percentile(boot,[2.5,97.5]).tolist(),"resamples":2000,"seed":7}
    summary["human_correctness"] = "unmeasured; public short-answer labels are not independently reviewed"
    return summary
