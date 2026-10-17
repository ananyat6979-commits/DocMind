"""
Cross-encoder reranker.

A bi-encoder embeds query and passage separately. A cross-encoder reads
[CLS] query [SEP] passage [SEP] jointly, which is slower (so we only run it on
the ~20 fused candidates) but a much stronger relevance signal.

The model is a process-wide singleton: the old code reloaded the 90MB model for
every HybridRetriever instance (e.g. on every upload in the UI).
"""
import logging
import re
import threading
from typing import List, Optional

from docmind.config import CONFIG
from docmind.models import SearchResult

logger = logging.getLogger(__name__)

_MODEL = None
_LOCK = threading.Lock()


def _lexical_scores(query: str, passages: List[str]) -> List[float]:
    """Offline stand-in used only when reranker_model == 'lexical' (tests/CI)."""
    q = set(re.findall(r"\w+", query.lower()))
    out = []
    for p in passages:
        toks = set(re.findall(r"\w+", p.lower()))
        out.append(10.0 * (len(q & toks) / max(len(q), 1)) - 5.0)
    return out


def _get_model():
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading cross-encoder '{CONFIG.retriever.reranker_model}' (one-time)")
                _MODEL = CrossEncoder(CONFIG.retriever.reranker_model, device=CONFIG.embedding.device)
    return _MODEL


class CrossEncoderReranker:
    def rerank(self, query: str, results: List[SearchResult], top_k: Optional[int] = None,
               min_score: Optional[float] = None) -> List[SearchResult]:
        """Score (query, passage) pairs, sort descending, keep top_k, and drop
        anything under min_score (the abstention gate)."""
        if not results:
            return []
        top_k = top_k or CONFIG.retriever.top_k_rerank
        passages = [r.chunk.content for r in results]

        if CONFIG.retriever.reranker_model == "lexical":
            scores = _lexical_scores(query, passages)
        else:
            scores = _get_model().predict([[query, p] for p in passages], batch_size=16)

        scored = sorted(zip(scores, results), key=lambda x: float(x[0]), reverse=True)
        out = [
            SearchResult(chunk=r.chunk, score=float(s), retriever="reranker")
            for s, r in scored[:top_k]
        ]
        if min_score is not None:
            out = [r for r in out if r.score >= min_score]
        return out
