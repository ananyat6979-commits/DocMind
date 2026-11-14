"""
Hybrid retrieval: FAISS + BM25 -> Reciprocal Rank Fusion -> cross-encoder rerank.

RRF (Cormack et al., 2009): score(d) = sum over rankers of 1 / (k + rank_i(d)).
Rank-based, so cosine and BM25 scales never need normalizing.

The retriever is a view over CorpusStore. Each search checks the store's
corpus_version and rebuilds the in-memory indexes if another process (or the
upload button) changed the corpus. Searches capture coherent dense/sparse
snapshots; supported deployment still uses one API service process.
"""
import logging
import threading
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from docmind.config import CONFIG
from docmind.models import Chunk, SearchResult
from docmind.retrieval.dense import DenseRetriever
from docmind.retrieval.postgres import PostgresDenseRetriever
from docmind.retrieval.reranker import CrossEncoderReranker
from docmind.retrieval.sparse import SparseRetriever
from docmind.store import CorpusStore

logger = logging.getLogger(__name__)

_USE_CONFIG_THRESHOLD = object()

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
        self.dense: DenseRetriever | PostgresDenseRetriever = DenseRetriever()
        self.sparse = SparseRetriever()
        self.reranker = CrossEncoderReranker() if CONFIG.retriever.use_reranker else None
        self._version = -1
        self._fit_mode: str | None = None
        self._n_chunks = 0
        self._embedding_identity: str | None = None
        self._lock = threading.RLock()

    def refresh(self, force: bool = False, mode: Optional[str] = None) -> None:
        """Build only views required by this mode, publishing one coherent snapshot."""
        mode = mode or CONFIG.retriever.mode
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        v = self.store.version()
        identity=self.store._embedding_signature()
        if not force and v == self._version and identity == self._embedding_identity and mode == self._fit_mode:
            return
        with self._lock:
            if not force and v == self._version and identity == self._embedding_identity and mode == self._fit_mode:
                return
            from docmind.postgres import DatabaseScope
            if isinstance(self.store.db_path,DatabaseScope):
                if mode!='dense':raise ValueError("Persistent PostgreSQL profile currently requires dense retrieval")
                from docmind.retrieval.postgres import PostgresDenseRetriever
                from docmind.db import connect
                self.store.ensure_embedding_compatible()
                self.dense=PostgresDenseRetriever(self.store)
                with connect(self.store.db_path) as conn:self._n_chunks=conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                self._version=v;self._embedding_identity=identity;self._fit_mode=mode
                return
            self.store.ensure_embedding_compatible()
            chunks, matrix = self.store.load_corpus()
            dense, sparse = DenseRetriever(), SparseRetriever()
            if mode != "sparse":
                dense.fit(chunks, matrix)
            if mode != "dense":
                sparse.fit(chunks)
            self.dense, self.sparse = dense, sparse
            self._version = v
            self._embedding_identity = identity
            self._fit_mode = mode
            self._n_chunks = len(chunks)
            logger.info(f"Indexes synced to corpus version {v} ({len(chunks)} chunks)")

    def is_empty(self) -> bool:
        with self._lock:
            self.refresh()
            return not self._n_chunks

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        file_ids: Optional[Iterable[str]] = None,
        mode: Optional[str] = None,
        min_score=_USE_CONFIG_THRESHOLD,
    ) -> List[SearchResult]:
        mode = mode or CONFIG.retriever.mode
        threshold = CONFIG.retriever.min_rerank_score if min_score is _USE_CONFIG_THRESHOLD else min_score
        if mode != "hybrid_rerank" and threshold is not None:
            raise ValueError("Reranker thresholds require hybrid_rerank mode; clear stale thresholds before changing retrieval")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "hybrid_rerank" and self.reranker is None:
            raise ValueError("The requested reranker is disabled; enable it explicitly or select dense/hybrid mode")
        with self._lock:
            self.refresh(mode=mode)
            dense, sparse = self.dense, self.sparse
        cfg = CONFIG.retriever
        top_k = top_k or cfg.top_k_rerank
        fids = list(file_ids) if file_ids else None

        if mode == "dense":
            return dense.search(query, top_k=top_k, file_ids=fids)
        if mode == "sparse":
            return sparse.search(query, top_k=top_k, file_ids=fids)

        dense_results = dense.search(query, top_k=cfg.top_k_dense, file_ids=fids)
        sparse_results = sparse.search(query, top_k=cfg.top_k_sparse, file_ids=fids)
        fused = reciprocal_rank_fusion([dense_results, sparse_results], k=cfg.rrf_k)

        if mode == "hybrid" or not (self.reranker and fused):
            return fused[:top_k]
        return self.reranker.rerank(query, fused, top_k=top_k, min_score=cfg.min_rerank_score if min_score is _USE_CONFIG_THRESHOLD else min_score)
