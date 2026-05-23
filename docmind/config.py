"""
Central configuration for DocMind.
All tunable parameters live here: nothing hardcoded in pipeline code.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    # bge-small-en-v1.5 outperforms all-MiniLM on MTEB at the same ~33M param count.
    # The key difference: bge is trained with instruction-aware contrastive learning.
    model_name: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"         # safe default; set to "cuda" on Kaggle
    batch_size: int = 32        # reduce to 16 if you get OOM on your Pavilion
    normalize_embeddings: bool = True   # required for bge; enables cosine via dot product


@dataclass
class ChunkerConfig:
    strategy: Literal["semantic", "fixed"] = "semantic"
    min_chunk_size: int = 150   # chars; below this, merge with neighbour
    max_chunk_size: int = 1200  # chars; above this, hard-split
    overlap_sentences: int = 1  # carry last N sentences into next chunk for boundary context
    breakpoint_percentile: float = 95.0  # top 5% of similarity drops become boundaries


@dataclass
class RetrieverConfig:
    top_k_dense: int = 10       # candidates from FAISS
    top_k_sparse: int = 10      # candidates from BM25
    top_k_rerank: int = 5       # final results after cross-encoder
    rrf_k: int = 60             # RRF constant from the original paper (Cormack 2009)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True


@dataclass
class LLMConfig:
    provider: Literal["groq", "ollama"] = "groq"
    groq_model: str = "llama-3.1-8b-instant"  # free tier, very fast
    ollama_model: str = "phi3:mini"             # local fallback
    temperature: float = 0.1
    max_tokens: int = 1024
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))


@dataclass
class DocMindConfig:
    data_dir: Path = Path("data")
    index_dir: Path = Path("indexes")
    db_path: Path = Path("indexes/docmind.db")
    mlflow_tracking_uri: str = "mlruns"

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def __post_init__(self):
        self.data_dir.mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)


CONFIG = DocMindConfig()
