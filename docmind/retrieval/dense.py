"""
Dense retrieval via FAISS IndexFlatIP (exact cosine on L2-normalized vectors).

The index is an in-memory cache built from the corpus store. Filtering to a
subset of files uses FAISS's IDSelector, so scoped search stays exact.
"""
import logging
import os
from typing import Iterable, List, Optional, Sequence

import numpy as np

from docmind.config import CONFIG
from docmind.embeddings import embed_query
from docmind.models import Chunk, SearchResult

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self):
        self._index = None
        self._chunks: List[Chunk] = []
        self._file_ids: Optional[np.ndarray] = None

    def fit(self, chunks: Sequence[Chunk], matrix: np.ndarray) -> None:
        import faiss
        self._chunks = list(chunks)
        if not self._chunks:
            self._index, self._file_ids = None, None
            return
        if os.getenv("DOCMIND_DENSE_INDEX", "flat") == "hnsw":
            self._index = faiss.IndexHNSWFlat(matrix.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
            self._index.hnsw.efConstruction = 80
            self._index.hnsw.efSearch = 64
        else:
            self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._matrix = np.ascontiguousarray(matrix, dtype="float32")
        self._index.add(np.ascontiguousarray(matrix, dtype="float32"))
        self._file_ids = np.array([c.file_id for c in self._chunks])
        logger.info(f"Dense index ready: {self._index.ntotal} vectors, dim={matrix.shape[1]}")

    def search(self, query: str, top_k: Optional[int] = None,
               file_ids: Optional[Iterable[str]] = None) -> List[SearchResult]:
        if self._index is None:
            return []
        import faiss
        top_k = top_k or CONFIG.retriever.top_k_dense
        k = min(top_k, self._index.ntotal)
        qv = embed_query(query)

        if file_ids:
            assert self._file_ids is not None
            allowed = np.nonzero(np.isin(self._file_ids, list(file_ids)))[0].astype("int64")
            if allowed.size == 0:
                return []
            # Exact scoped search keeps access filters independent of ANN candidate depth.
            values = self._matrix[allowed] @ qv[0]
            chosen = np.argsort(-values, kind="stable")[:min(k, allowed.size)]
            scores, idxs = values[chosen][None, :], allowed[chosen][None, :]
        else:
            scores, idxs = self._index.search(qv, k)

        return [
            SearchResult(chunk=self._chunks[i], score=float(s), retriever="dense")
            for s, i in zip(scores[0], idxs[0]) if i != -1
        ]
