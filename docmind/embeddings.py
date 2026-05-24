"""
Shared embedding model registry.
The model is expensive to load (~2s). This ensures it loads once
and is reused across the chunker and dense retriever.
"""
import logging
from sentence_transformers import SentenceTransformer
from docmind.config import CONFIG

logger = logging.getLogger(__name__)
_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model '{CONFIG.embedding.model_name}' (one-time)")
        _model = SentenceTransformer(
            CONFIG.embedding.model_name,
            device=CONFIG.embedding.device,
        )
    return _model