"""
Retrieval ablation: dense vs sparse vs hybrid vs hybrid+rerank (and fixed vs
semantic chunking), scored against substring-labeled gold passages, with
bootstrap confidence intervals and a paired test for the rerank claim.

This is what turns "40-60% precision improvement" from a quote into a number.
No LLM is involved, so it is free, fast and deterministic.
"""
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from docmind.config import CONFIG
from docmind.evaluation.metrics import (
    best_f1_threshold, bootstrap_ci, hit_at_k, is_relevant, mrr, ndcg_at_k,
    paired_relative_improvement, precision_at_k,
)
from docmind.retrieval.hybrid import HybridRetriever, MODES
from docmind.store import CorpusStore

logger = logging.getLogger(__name__)


def load_testset(path: str, verified_only: bool = False) -> List[Dict]:
    """Load a testset. With verified_only=True, only items a human marked verified are used,
    so unreviewed LLM-drafted questions can never end up in a reported benchmark."""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    for i, it in enumerate(items):
        it.setdefault("id", f"q{i + 1:03d}")
        it.setdefault("answerable", True)
        it.setdefault("gold_substrings", [])
        it.setdefault("verified", False)
    if verified_only:
        kept = [it for it in items if it["verified"]]
        if not kept:
            raise ValueError(
                f"{path} has {len(items)} items but none are verified. Run scripts/make_testset.py "
                "(interactive) or scripts/verify_testset.py on a draft, or pass --include-unverified "
                "for a throwaway trial run.")
        return kept
    return items


def evaluate_modes(testset: List[Dict], retriever: HybridRetriever, k: int = 5,
                   modes: Sequence[str] = MODES, file_ids: Optional[List[str]] = None) -> Dict:
    if "hybrid_rerank" in modes and retriever.reranker is None:
        raise ValueError("The reranker is disabled (DOCMIND_USE_RERANKER=0). The hybrid_rerank mode would silently equal "
                         "plain hybrid and the 'reranking' numbers would be fiction. Unset it, or run the ablation in the cloud.")
    labeled = [t for t in testset if t["answerable"] and t["gold_substrings"]]
    if not labeled:
        raise ValueError("No answerable items with gold_substrings in the testset. "
                         "Run scripts/make_testset.py or add gold_substrings by hand.")
    saved = CONFIG.retriever.min_rerank_score
    CONFIG.retriever.min_rerank_score = None          # measure raw ranking, not the abstention gate
    per_mode: Dict[str, Dict[str, List[float]]] = {}
    calib_scores: List[float] = []
    calib_labels: List[bool] = []
    try:
        for mode in modes:
            cols = {"precision": [], "hit": [], "mrr": [], "ndcg": []}
            for item in labeled:
                results = retriever.search(item["question"], top_k=k, file_ids=file_ids, mode=mode)
                rel = [is_relevant(r.chunk.content, item["gold_substrings"]) for r in results]
                cols["precision"].append(precision_at_k(rel, k))
                cols["hit"].append(hit_at_k(rel, k))
                cols["mrr"].append(mrr(rel, k))
                cols["ndcg"].append(ndcg_at_k(rel, k))
                if mode == "hybrid_rerank":
                    calib_scores += [r.score for r in results]
                    calib_labels += rel
            per_mode[mode] = cols
    finally:
        CONFIG.retriever.min_rerank_score = saved

    summary = {}
    for mode, cols in per_mode.items():
        summary[mode] = {m: dict(zip(("mean", "lo", "hi"), bootstrap_ci(v, CONFIG.eval.bootstrap_resamples)))
                         for m, v in cols.items()}
    out: Dict = {"k": k, "n_questions": len(labeled), "modes": summary}
    if "dense" in per_mode and "hybrid_rerank" in per_mode:
        out["rerank_vs_dense_precision"] = paired_relative_improvement(
            per_mode["dense"]["precision"], per_mode["hybrid_rerank"]["precision"])
    if "hybrid" in per_mode and "hybrid_rerank" in per_mode:
        out["rerank_vs_hybrid_precision"] = paired_relative_improvement(
            per_mode["hybrid"]["precision"], per_mode["hybrid_rerank"]["precision"])
    if "dense" in per_mode and "hybrid" in per_mode:
        out["hybrid_vs_dense_precision"] = paired_relative_improvement(
            per_mode["dense"]["precision"], per_mode["hybrid"]["precision"])
    if calib_scores:
        out["suggested_min_rerank_score"] = best_f1_threshold(calib_scores, calib_labels)
    return out


def compare_chunkers(testset: List[Dict], docs: Sequence[str], k: int = 5) -> Dict:
    """Rebuild a throwaway corpus per chunking strategy and compare retrieval.
    Never touches your real database."""
    from docmind.ingestion.pipeline import ingest
    results = {}
    original = CONFIG.chunker.strategy
    try:
        for strategy in ("fixed", "semantic"):
            CONFIG.chunker.strategy = strategy
            with tempfile.TemporaryDirectory() as tmp:
                store = CorpusStore(db_path=Path(tmp) / "ablation.db")
                n_chunks = 0
                for d in docs:
                    rep = ingest(d, store=store)
                    n_chunks += rep.chunks_added
                r = evaluate_modes(testset, HybridRetriever(store), k=k, modes=("hybrid_rerank",))
                r["n_chunks"] = n_chunks
                results[strategy] = r
    finally:
        CONFIG.chunker.strategy = original
    return results


def to_markdown(result: Dict) -> str:
    k = result["k"]
    lines = [f"Retrieval ablation, n={result['n_questions']} questions, k={k} (mean [95% bootstrap CI])", "",
             f"| Mode | Precision@{k} | Hit@{k} | MRR | nDCG@{k} |", "|---|---|---|---|---|"]
    for mode, m in result["modes"].items():
        cells = [f"{m[x]['mean']:.3f} [{m[x]['lo']:.3f}, {m[x]['hi']:.3f}]" for x in ("precision", "hit", "mrr", "ndcg")]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    lines.append("")
    for key, label in (("rerank_vs_dense_precision", "Hybrid+rerank vs dense-only"),
                       ("rerank_vs_hybrid_precision", "Rerank vs hybrid-without-rerank"),
                       ("hybrid_vs_dense_precision", "Hybrid vs dense-only")):
        v = result.get(key)
        if v and v.get("relative") is not None:
            ci = f" (95% CI {v['lo'] * 100:+.0f}% to {v['hi'] * 100:+.0f}%)" if v.get("lo") is not None else ""
            lines.append(f"- {label}: precision@{k} {v['relative'] * 100:+.1f}%{ci}, paired bootstrap over {v['n']} questions")
    t = result.get("suggested_min_rerank_score")
    if t and t["threshold"] == t["threshold"]:
        lines.append(f"- Suggested DOCMIND_MIN_RERANK_SCORE = {t['threshold']:.2f} (best F1 {t['f1']:.2f} on your labels)")
    return "\n".join(lines)
