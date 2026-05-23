"""
Cross-encoder reranker.

The key insight: a bi-encoder (used in dense retrieval) encodes query and
document SEPARATELY, then compares embeddings. This is fast but loses
interaction signals between query tokens and document tokens.

A cross-encoder sees both query and document concatenated as a single input:
  [CLS] query [SEP] document [SEP]

This lets attention heads cross-attend between query and document, capturing
"does this document specifically answer this question?" much more accurately.

Tradeoff: cross-encoders are ~50x slower than bi-encoders per pair.
Solution: only rerank the ~20 candidates from stage 1, not the full corpus.
"""
import logging
from typing import List

from docmind.models import SearchResult
from config import CONFIG

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder '{CONFIG.retriever.reranker_model}'")
            self._model = CrossEncoder(CONFIG.retriever.reranker_model)
        return self._model

    def rerank(self, query: str, results: List[SearchResult], top_k: int = None) -> List[SearchResult]:
        """
        Score each (query, document) pair with the cross-encoder and
        return top_k results sorted by reranker score.
        """
        if not results:
            return []

        top_k = top_k or CONFIG.retriever.top_k_rerank
        model = self._get_model()

        # Cross-encoder expects list of [query, document] pairs
        pairs = [[query, r.chunk.content] for r in results]
        scores = model.predict(pairs)

        # Zip scores back onto results and sort descending
        scored = sorted(
            zip(scores, results),
            key=lambda x: x[0],
            reverse=True,
        )

        reranked = []
        for score, result in scored[:top_k]:
            reranked.append(SearchResult(
                chunk=result.chunk,
                score=float(score),
                retriever="reranker",
            ))

        return reranked