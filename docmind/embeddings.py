"""
Shared embedding backend.

Two backends:
  - sentence-transformers (default): BAAI/bge-small-en-v1.5
  - hash: deterministic bag-of-hashed-tokens vectors. Used for offline tests and
    CI so the test suite never downloads a model. Never use it for real retrieval.
"""
import logging
import re
import zlib
import threading
from typing import List

import numpy as np

from docmind.config import CONFIG

logger = logging.getLogger(__name__)
_model = None
_lock = threading.Lock()


class HashEmbedder:
    dim = 384

    def encode(self, texts, batch_size=32, normalize_embeddings=True,
               show_progress_bar=False, convert_to_numpy=True):
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for tok in re.findall(r"\w+", text.lower()):
                out[i, zlib.crc32(tok.encode("utf-8")) % self.dim] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms
        return out


def get_embedding_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                if CONFIG.embedding.backend == "hash":
                    logger.info("Using HashEmbedder (test backend)")
                    _model = HashEmbedder()
                else:
                    from sentence_transformers import SentenceTransformer
                    logger.info(f"Loading embedding model '{CONFIG.embedding.model_name}' (one-time)")
                    _model = SentenceTransformer(CONFIG.embedding.model_name, device=CONFIG.embedding.device)
    return _model


def reset_embedding_model() -> None:
    """Used by tests when the backend is switched."""
    global _model
    _model = None


def embed_texts(texts: List[str], show_progress: bool = False) -> np.ndarray:
    """Embed passages (no instruction prefix). Returns float32 (n, dim)."""
    model = get_embedding_model()
    vecs = model.encode(
        texts,
        batch_size=CONFIG.embedding.batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return np.ascontiguousarray(vecs, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a search query. bge models are trained to see an instruction
    prefix on the query side only; omitting it costs retrieval quality."""
    prefix = ""
    if CONFIG.embedding.backend != "hash" and CONFIG.embedding.model_name.startswith("BAAI/bge"):
        prefix = CONFIG.embedding.query_instruction
    return embed_texts([prefix + query])
