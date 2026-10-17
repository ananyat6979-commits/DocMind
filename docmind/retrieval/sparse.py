"""
Sparse retrieval: BM25 implemented with an inverted index and numpy.

Why not rank-bm25:
  * BM25Okapi's IDF goes negative when a term appears in more than half of the
    chunks, so on small corpora (a single short document) every score can be
    <= 0 and the old `scores > 0` filter returned nothing at all.
  * get_scores() loops in Python over every document for every query term.
This version uses the Lucene IDF, log(1 + (N - df + 0.5) / (df + 0.5)), which is
always positive, and touches only the postings of the query terms.
"""
import logging
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from docmind.config import CONFIG
from docmind.models import Chunk, SearchResult

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "what which who whom how when where why do does did not no yes can could should would about into than then".split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumerics (so 'data,' matches 'data'), drop stopwords."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = 0
        self._postings: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._idf: Dict[str, float] = {}
        self._dl = np.zeros(0, dtype="float32")
        self._avgdl = 1.0

    def fit(self, tokenized_docs: Sequence[Sequence[str]]) -> None:
        self.n = len(tokenized_docs)
        self._dl = np.array([len(d) for d in tokenized_docs], dtype="float32")
        self._avgdl = float(self._dl.mean()) if self.n and self._dl.mean() > 0 else 1.0
        acc: Dict[str, Dict[int, int]] = {}
        for i, doc in enumerate(tokenized_docs):
            for tok in doc:
                acc.setdefault(tok, {})
                acc[tok][i] = acc[tok].get(i, 0) + 1
        self._postings, self._idf = {}, {}
        for tok, d in acc.items():
            idx = np.fromiter(d.keys(), dtype="int64", count=len(d))
            tf = np.fromiter(d.values(), dtype="float32", count=len(d))
            self._postings[tok] = (idx, tf)
            df = len(d)
            self._idf[tok] = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def get_scores(self, query_tokens: Iterable[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype="float32")
        for tok in set(query_tokens):
            post = self._postings.get(tok)
            if post is None:
                continue
            idx, tf = post
            denom = tf + self.k1 * (1.0 - self.b + self.b * self._dl[idx] / self._avgdl)
            scores[idx] += self._idf[tok] * tf * (self.k1 + 1.0) / denom
        return scores


class SparseRetriever:
    def __init__(self):
        self._bm25: Optional[BM25Index] = None
        self._chunks: List[Chunk] = []
        self._file_ids: Optional[np.ndarray] = None

    def fit(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        if not self._chunks:
            self._bm25, self._file_ids = None, None
            return
        self._bm25 = BM25Index()
        self._bm25.fit([tokenize(c.content) for c in self._chunks])
        self._file_ids = np.array([c.file_id for c in self._chunks])
        logger.info(f"BM25 index ready: {len(self._chunks)} chunks")

    def search(self, query: str, top_k: Optional[int] = None,
               file_ids: Optional[Iterable[str]] = None) -> List[SearchResult]:
        if self._bm25 is None:
            return []
        top_k = top_k or CONFIG.retriever.top_k_sparse
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        if file_ids:
            scores = np.where(np.isin(self._file_ids, list(file_ids)), scores, 0.0)
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        top = top[np.argsort(-scores[top])]
        return [
            SearchResult(chunk=self._chunks[i], score=float(scores[i]), retriever="sparse")
            for i in top if scores[i] > 0
        ]
