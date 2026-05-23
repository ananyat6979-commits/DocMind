"""
Hybrid retrieval: FAISS + BM25 → Reciprocal Rank Fusion → Cross-encoder rerank.

Reciprocal Rank Fusion (RRF) is the fusion method from Cormack et al. (2009).
For a document d at rank r in ranker i, its RRF score contribution is:
    1 / (k + r)   where k is a smoothing constant (default 60)

The final score is the sum of RRF contributions across all rankers.

Why RRF over score-based fusion?
  - Dense and sparse scores live on different scales (cosine vs BM25).
  - Normalizing scores is tricky; rank-based fusion sidesteps the problem.
  - RRF is robust and rarely needs tuning — k=60 works well empirically.
"""
import logging
from collections import defaultdict
from typing import List, Dict, Optional

from docmind.models import Chunk, SearchResult
from docmind.retrieval.dense import DenseRetriever
from docmind.retrieval.sparse import SparseRetriever
from docmind.retrieval.reranker import CrossEncoderReranker
from docmind.config import CONFIG

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]],
    k: int = 60,
) -> List[SearchResult]:
    """
    Fuse multiple ranked lists into one using RRF.
    Documents appearing in multiple lists get higher scores.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    chunk_map: Dict[str, Chunk] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            cid = result.chunk.chunk_id
            rrf_scores[cid] += 1.0 / (k + rank)
            chunk_map[cid] = result.chunk

    # Sort by fused score, return as SearchResult objects
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        SearchResult(chunk=chunk_map[cid], score=score, retriever="hybrid")
        for cid, score in fused
    ]


class HybridRetriever:
    """
    Orchestrates the full retrieval pipeline:
    dense retrieval + sparse retrieval → RRF fusion → cross-encoder reranking.
    """
    def __init__(self):
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.reranker = CrossEncoderReranker() if CONFIG.retriever.use_reranker else None

    def build_index(self, chunks: List[Chunk], force_rebuild: bool = False) -> None:
        """Build both FAISS and BM25 indexes from a list of Chunks."""
        from docmind.retrieval.dense import FAISS_INDEX_PATH
        
        if not force_rebuild and FAISS_INDEX_PATH.exists():
            logger.info("Indexes already exist. Use force_rebuild=True to regenerate.")
            return

        logger.info("Building retrieval indexes...")
        self.dense.build(chunks)
        self.sparse.build(chunks)
        logger.info("Indexes built successfully.")

    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Full retrieval pipeline for a query.
        Returns top_k results after reranking.
        """
        cfg = CONFIG.retriever
        top_k = top_k or cfg.top_k_rerank

        # Stage 1: parallel retrieval from both indexes
        dense_results = self.dense.search(query, top_k=cfg.top_k_dense)
        sparse_results = self.sparse.search(query, top_k=cfg.top_k_sparse)

        logger.debug(
            f"Stage 1: dense={len(dense_results)} | sparse={len(sparse_results)}"
        )

        # Fuse via RRF
        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=cfg.rrf_k,
        )

        # Stage 2: rerank with cross-encoder (if enabled)
        if self.reranker and fused:
            results = self.reranker.rerank(query, fused, top_k=top_k)
            logger.debug(f"Stage 2: reranked to {len(results)} results")
        else:
            results = fused[:top_k]

        return results
