"""
Shared embedding backend.

Two backends:
  - sentence-transformers (default): pinned multilingual E5-small
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
_model_key = None
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
    global _model,_model_key
    cfg=CONFIG.embedding
    key=(cfg.backend,cfg.model_name,cfg.model_revision,cfg.device)
    if _model is None or _model_key!=key:
        with _lock:
            if _model is None or _model_key!=key:
                if cfg.backend=="hash":
                    _model=HashEmbedder()
                else:
                    if not cfg.model_revision:
                        raise ValueError("Pin DOCMIND_EMBEDDING_REVISION before changing embedding models")
                    from sentence_transformers import SentenceTransformer
                    _model=SentenceTransformer(cfg.model_name,device=cfg.device,revision=cfg.model_revision)
                _model_key=key
    return _model


def reset_embedding_model() -> None:
    """Used by tests when the backend is switched."""
    global _model,_model_key
    _model = None
    _model_key = None


def embed_texts(texts: List[str], show_progress: bool = False) -> np.ndarray:
    """Embed passages with the model-specific prefix policy, as float32 (n, dim)."""
    return _encode(texts, show_progress, query=False)


def _encode(texts: List[str], show_progress: bool = False, query: bool = False) -> np.ndarray:
    model = get_embedding_model()
    cfg = CONFIG.embedding
    if cfg.backend != "hash" and cfg.model_name.startswith("intfloat/multilingual-e5"):
        texts = [("query: " if query else "passage: ") + text for text in texts]
    elif query and cfg.backend != "hash" and cfg.model_name.startswith("BAAI/bge"):
        texts = [cfg.query_instruction + text for text in texts]
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
    return _encode([query], query=True)


def embedding_identity() -> str:
    """Separate inference representations and dependency baselines."""
    import importlib.metadata
    import json
    e = CONFIG.embedding
    versions = {}
    for name in ("transformers", "sentence-transformers", "torch", "tokenizers"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = "absent"
    return json.dumps({"backend":e.backend,"model":e.model_name,"revision":e.model_revision,
        "normalized":e.normalize_embeddings,"instruction":e.query_instruction,
        "prefix_policy":"bge-query-e5-both-v1","versions":versions},sort_keys=True)
