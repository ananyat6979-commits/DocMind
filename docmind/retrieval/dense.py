"""
Dense retrieval via FAISS.

We use IndexFlatIP (inner product on L2-normalized vectors = cosine similarity).
For a corpus up to ~100k chunks, exact search is fast enough on a laptop CPU.
If you scale beyond that, swap to IndexIVFFlat with nprobe tuning on Kaggle.
"""
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple

from docmind.models import Chunk, SearchResult
from docmind.config import CONFIG

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = CONFIG.index_dir / "faiss.index"
CHUNK_MAP_PATH   = CONFIG.index_dir / "chunk_map.pkl"   # maps FAISS int id → Chunk


class DenseRetriever:
    def __init__(self):
        self._index = None
        self._chunks: List[Chunk] = []
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                CONFIG.embedding.model_name,
                device=CONFIG.embedding.device,
            )
        return self._model

    def build(self, chunks: List[Chunk]) -> None:
        """Embed all chunks and build a FAISS index. Persists to disk."""
        import faiss

        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        logger.info(f"Embedding {len(chunks)} chunks for FAISS index...")
        model = self._get_model()
        texts = [c.content for c in chunks]

        embeddings = model.encode(
            texts,
            batch_size=CONFIG.embedding.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        # IndexFlatIP: exact inner product search; correct for normalized vectors
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._chunks = chunks

        # Persist
        faiss.write_index(self._index, str(FAISS_INDEX_PATH))
        with open(CHUNK_MAP_PATH, "wb") as f:
            pickle.dump(chunks, f)

        logger.info(f"FAISS index built: {self._index.ntotal} vectors, dim={dim}")

    def load(self) -> bool:
        """Load an existing FAISS index from disk. Returns True if successful."""
        import faiss

        if not FAISS_INDEX_PATH.exists() or not CHUNK_MAP_PATH.exists():
            return False

        self._index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(CHUNK_MAP_PATH, "rb") as f:
            self._chunks = pickle.load(f)

        logger.info(f"FAISS index loaded: {self._index.ntotal} vectors")
        return True

    def search(self, query: str, top_k: int = None) -> List[SearchResult]:
        """Embed the query and return top-k nearest chunks by cosine similarity."""
        if self._index is None:
            if not self.load():
                raise RuntimeError("FAISS index not found. Run ingestion first.")

        top_k = top_k or CONFIG.retriever.top_k_dense
        model = self._get_model()

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = self._index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS returns -1 for padding
                continue
            results.append(SearchResult(
                chunk=self._chunks[idx],
                score=float(score),
                retriever="dense",
            ))

        return results
