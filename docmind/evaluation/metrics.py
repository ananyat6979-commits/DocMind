"""
Metric helpers with honest uncertainty. Every headline number in the README
should come from here with a confidence interval, never a bare point estimate.
"""
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_relevant(chunk_text: str, gold_substrings: Sequence[str]) -> bool:
    """A chunk is relevant if it contains ANY gold answer-bearing substring.
    Substring labels survive re-chunking, so they also work for chunker ablations."""
    t = normalize(chunk_text)
    return any(normalize(g) in t for g in gold_substrings if g.strip())


def precision_at_k(rel: Sequence[bool], k: int) -> float:
    return sum(rel[:k]) / k


def hit_at_k(rel: Sequence[bool], k: int) -> float:
    return 1.0 if any(rel[:k]) else 0.0


def mrr(rel: Sequence[bool], k: int) -> float:
    for i, r in enumerate(rel[:k], 1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rel: Sequence[bool], k: int) -> float:
    gains = [1.0 / math.log2(i + 2) for i, r in enumerate(rel[:k]) if r]
    n_rel = min(sum(rel), k)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return sum(gains) / ideal if ideal else 0.0


def bootstrap_ci(values: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 7
                 ) -> Tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap over questions."""
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=float)
    if v.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_relative_improvement(baseline: Sequence[float], treatment: Sequence[float],
                                n: int = 2000, seed: int = 11) -> Dict[str, Optional[float]]:
    """Relative improvement of treatment over baseline, (mean_t / mean_b) - 1, with a
    paired bootstrap CI (same questions resampled together). This is the right test
    for a claim like 'reranking improves precision by X%'."""
    b, t = np.asarray(baseline, dtype=float), np.asarray(treatment, dtype=float)
    if b.size != t.size or b.size == 0:
        return {"relative": None, "lo": None, "hi": None, "n": int(b.size)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, b.size, size=(n, b.size))
    mb, mt = b[idx].mean(axis=1), t[idx].mean(axis=1)
    ok = mb > 0
    rel = (mt[ok] / mb[ok]) - 1.0
    point = (t.mean() / b.mean() - 1.0) if b.mean() > 0 else None
    if rel.size == 0:
        return {"relative": point, "lo": None, "hi": None, "n": int(b.size)}
    return {"relative": point, "lo": float(np.quantile(rel, 0.025)), "hi": float(np.quantile(rel, 0.975)), "n": int(b.size)}


def best_f1_threshold(scores: Sequence[float], labels: Sequence[bool]) -> Dict[str, float]:
    """Pick the score cut-off that best separates relevant from irrelevant passages.
    Used to calibrate DOCMIND_MIN_RERANK_SCORE from YOUR data."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    if s.size == 0 or y.sum() == 0:
        return {"threshold": float("nan"), "f1": 0.0}
    best = (0.0, float(s.min()))
    for thr in np.unique(s):
        pred = s >= thr
        tp = float((pred & y).sum()); fp = float((pred & ~y).sum()); fn = float((~pred & y).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best[0]:
            best = (f1, float(thr))
    return {"threshold": best[1], "f1": best[0]}


def _mean_timing(records, key):
    vals = [r.get("timings", {}).get(key) for r in records if r.get("timings", {}).get(key) is not None]
    return float(np.mean(vals)) if vals else None


def _total_tokens(records):
    vals = [(r.get("timings", {}).get("prompt_tokens") or 0) + (r.get("timings", {}).get("completion_tokens") or 0)
            for r in records if r.get("timings", {}).get("prompt_tokens") is not None]
    return float(sum(vals)) if vals else None


def _count_by(records, fn):
    out = {}
    for r in records:
        out[fn(r)] = out.get(fn(r), 0) + 1
    return out


def answer_level_stats(records: List[Dict]) -> Dict:
    """Citation validity, abstention behaviour and latency from generation records."""
    answered = [r for r in records if not r["abstained"]]
    cited = sum(len(r["citation_report"].get("cited", [])) for r in answered)
    valid = sum(len(r["citation_report"].get("valid", [])) for r in answered)
    with_invalid = sum(1 for r in answered if r["citation_report"].get("invalid"))
    uncited = sum(1 for r in answered if not r["citation_report"].get("cited"))
    answerable = [r for r in records if r.get("answerable", True)]
    unanswerable = [r for r in records if not r.get("answerable", True)]
    lat = sorted(r["latency_ms"] for r in records) or [float("nan")]

    def pct(p):
        return float(np.percentile(lat, p))

    return {
        "n_total": len(records),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "citations_total": cited,
        "citation_validity": (valid / cited) if cited else None,
        "answers_with_invalid_citation": with_invalid,
        "answers_without_any_citation": uncited,
        "false_abstention_rate": (sum(r["abstained"] for r in answerable) / len(answerable)) if answerable else None,
        "correct_abstention_rate": (sum(r["abstained"] for r in unanswerable) / len(unanswerable)) if unanswerable else None,
        "hallucinated_answer_rate_on_unanswerable": (
            sum(not r["abstained"] for r in unanswerable) / len(unanswerable)) if unanswerable else None,
        "latency_p50_ms": pct(50), "latency_p95_ms": pct(95),
        "mean_llm_calls": float(np.mean([r["llm_calls"] for r in records])) if records else None,
        "mean_prompt_tokens": _mean_timing(records, "prompt_tokens"),
        "mean_completion_tokens": _mean_timing(records, "completion_tokens"),
        "total_tokens": _total_tokens(records),
        "finish_reasons": _count_by(records, lambda r: r.get("finish_reason", "unknown")),
        "review_rate": (sum(1 for r in records if r["needs_review"]) / len(records)) if records else None,
    }
