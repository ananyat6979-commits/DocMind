"""
Sparse retrieval via BM25 (Best Match 25).

BM25 is a probabilistic ranking function based on TF-IDF with length
normalization. It excels at exact keyword matching — a query like
"EBITDA margin Q3 2024" will find exact matches that dense retrieval
might miss if the embedding space doesn't distinguish financial jargon.

We use rank_bm25 which is a pure-Python BM25 implementation.
"""
import logging
import pickle
from pathlib import Path
from typing import List

from docmind.models import Chunk, SearchResult
from docmind.config import CONFIG

logger = logging.getLogger(__name__)

BM25_PATH = CONFIG.index_dir / "bm25.pkl"


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer.
    For production you'd add stemming/lemmatization, but this is solid for a portfolio."""
    return text.lower().split()


class SparseRetriever:
    def __init__(self):
        self._bm25 = None
        self._chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        """Tokenize all chunks and build BM25 corpus."""
        from rank_bm25 import BM25Okapi

        if not chunks:
            raise ValueError("Cannot build BM25 from empty chunk list")

        tokenized_corpus = [_tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._chunks = chunks

        with open(BM25_PATH, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunks": chunks}, f)

        logger.info(f"BM25 index built: {len(chunks)} documents")

    def load(self) -> bool:
        if not BM25_PATH.exists():
            return False
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunks = data["chunks"]
        logger.info(f"BM25 index loaded: {len(self._chunks)} documents")
        return True

    def search(self, query: str, top_k: int = None) -> List[SearchResult]:
        if self._bm25 is None:
            if not self.load():
                raise RuntimeError("BM25 index not found. Run ingestion first.")

        top_k = top_k or CONFIG.retriever.top_k_sparse
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(
                chunk=self._chunks[i],
                score=float(scores[i]),
                retriever="sparse",
            )
            for i in top_indices
            if scores[i] > 0  # don't return zero-score results
        ]
