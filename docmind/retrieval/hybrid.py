"""
Hybrid retrieval: FAISS + BM25 -> Reciprocal Rank Fusion -> cross-encoder rerank.

RRF (Cormack et al., 2009): score(d) = sum over rankers of 1 / (k + rank_i(d)).
Rank-based, so cosine and BM25 scales never need normalizing.

The retriever is a view over CorpusStore. Each search checks the store's
corpus_version and rebuilds the in-memory indexes if another process (or the
upload button) changed the corpus, so it can never serve a stale index.
"""
import logging
import threading
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from docmind.config import CONFIG
from docmind.models import Chunk, SearchResult
from docmind.retrieval.dense import DenseRetriever
from docmind.retrieval.reranker import CrossEncoderReranker
from docmind.retrieval.sparse import SparseRetriever
from docmind.store import CorpusStore

logger = logging.getLogger(__name__)

MODES = ("dense", "sparse", "hybrid", "hybrid_rerank")


def reciprocal_rank_fusion(result_lists: List[List[SearchResult]], k: int = 60) -> List[SearchResult]:
    rrf_scores: Dict[str, float] = defaultdict(float)
    chunk_map: Dict[str, Chunk] = {}
    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            cid = result.chunk.chunk_id
            rrf_scores[cid] += 1.0 / (k + rank)
            chunk_map[cid] = result.chunk
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [SearchResult(chunk=chunk_map[cid], score=score, retriever="hybrid") for cid, score in fused]


class HybridRetriever:
    def __init__(self, store: Optional[CorpusStore] = None):
        self.store = store or CorpusStore()
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.reranker = CrossEncoderReranker() if CONFIG.retriever.use_reranker else None
        self._version = -1
        self._lock = threading.Lock()

    def refresh(self, force: bool = False) -> None:
        """(Re)build in-memory indexes if the store changed."""
        v = self.store.version()
        if not force and v == self._version:
            return
        with self._lock:
            if not force and v == self._version:
                return
            chunks, matrix = self.store.load_corpus()
            self.dense.fit(chunks, matrix)
            self.sparse.fit(chunks)
            self._version = v
            logger.info(f"Indexes synced to corpus version {v} ({len(chunks)} chunks)")

    def is_empty(self) -> bool:
        self.refresh()
        return not self.dense._chunks

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        file_ids: Optional[Iterable[str]] = None,
        mode: str = "hybrid_rerank",
    ) -> List[SearchResult]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.refresh()
        cfg = CONFIG.retriever
        top_k = top_k or cfg.top_k_rerank
        fids = list(file_ids) if file_ids else None

        if mode == "dense":
            return self.dense.search(query, top_k=top_k, file_ids=fids)
        if mode == "sparse":
            return self.sparse.search(query, top_k=top_k, file_ids=fids)

        dense_results = self.dense.search(query, top_k=cfg.top_k_dense, file_ids=fids)
        sparse_results = self.sparse.search(query, top_k=cfg.top_k_sparse, file_ids=fids)
        fused = reciprocal_rank_fusion([dense_results, sparse_results], k=cfg.rrf_k)

        if mode == "hybrid" or not (self.reranker and fused):
            return fused[:top_k]
        return self.reranker.rerank(query, fused, top_k=top_k, min_score=cfg.min_rerank_score)
