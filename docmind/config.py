"""
Central configuration for DocMind.
All tunable parameters live here. Nothing is hardcoded in pipeline code.

Environment overrides (all optional):
  GROQ_API_KEY              generator + judge key
  DOCMIND_CHUNKER=fixed     skip sentence-level embedding when chunking (fast on very long documents)
  DOCMIND_RETRIEVAL_MODE    dense (default), sparse, hybrid or hybrid_rerank
  DOCMIND_USE_RERANKER=0    disable optional cross-encoder availability
  DOCMIND_MAX_ITERATIONS    ReAct steps per question (default 5; 3 saves tokens)
  DOCMIND_DEVICE            cpu | cuda (embedding + reranker device)
  DOCMIND_APP_PASSWORD      password gate for the Streamlit UI (set this before exposing it)
  DOCMIND_HOME              base directory for data/, indexes/, reports/ (default: cwd)
  DOCMIND_TEST_MODE=1       offline mode: hash embedder, lexical reranker, demo LLM
  DOCMIND_MIN_RERANK_SCORE  abstain when the cross-encoder score is below this
  DOCMIND_REVIEW_MIN_SCORE  send to human review when the best score is below this
  DOCMIND_REVIEW_AUDIT_RATE fraction of confident answers sampled for human audit
  DOCMIND_REVIEWER_PIN      PIN required to open the review queue (UI and API)
  DOCMIND_API_KEY           API key required by the FastAPI service
  DOCMIND_LLM_MODEL         generator model id on Groq (required; run scripts/list_models.py)
  DOCMIND_JUDGE_MODEL       RAGAS judge model id on Groq (required for evaluation)
  DOCMIND_REASONING_EFFORT  low | medium | high, only for reasoning models (generator)
  DOCMIND_JUDGE_REASONING_EFFORT  same, for the judge
  DOCMIND_MAX_TOKENS        completion budget per call (default 2048; reasoning models need room)
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Optional, cast
from dotenv import load_dotenv

# Offline test mode must never read a developer's real .env: tests would otherwise pick up a live
# API key and send traffic to the real service.
if os.getenv("DOCMIND_TEST_MODE", "") != "1":
    load_dotenv(os.getenv("DOCMIND_SECRETS_FILE") or None,encoding="utf-8-sig")


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


HOME = Path(os.getenv("DOCMIND_HOME", ".")).resolve()
TEST_MODE = os.getenv("DOCMIND_TEST_MODE", "") == "1"


@dataclass
class EmbeddingConfig:
    # E5-small covers the measured English/Hindi component; source prefixes are required.
    model_name: str = field(default_factory=lambda: os.getenv("DOCMIND_EMBEDDING_MODEL","intfloat/multilingual-e5-small"))
    model_revision: Optional[str] = field(default_factory=lambda: os.getenv("DOCMIND_EMBEDDING_REVISION") or {
        "intfloat/multilingual-e5-small":"614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "BAAI/bge-small-en-v1.5":"5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    }.get(os.getenv("DOCMIND_EMBEDDING_MODEL","intfloat/multilingual-e5-small")))
    backend: Literal["sentence-transformers", "hash"] = "hash" if TEST_MODE else "sentence-transformers"
    device: str = field(default_factory=lambda: os.getenv("DOCMIND_DEVICE", "cpu"))   # "cuda" on Kaggle/Colab GPU
    batch_size: int = 32
    normalize_embeddings: bool = True
    query_instruction: str = "Represent this sentence for searching relevant passages: "


@dataclass
class ChunkerConfig:
    # "fixed" skips per-sentence embedding: much faster on long documents (a novel can take many minutes
    # with "semantic"), at some cost in chunk coherence. Compare them with scripts/ablation.py --chunkers.
    strategy: Literal["semantic", "fixed", "recursive"] = field(default_factory=lambda: cast(Literal["semantic", "fixed", "recursive"], os.getenv("DOCMIND_CHUNKER", "recursive")))
    min_chunk_size: int = 150
    max_chunk_size: int = 1200
    overlap_sentences: int = 1
    breakpoint_percentile: float = 95.0


@dataclass
class RetrieverConfig:
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_rerank: int = 5
    rrf_k: int = 60
    mode: Literal["dense","sparse","hybrid","hybrid_rerank"] = field(default_factory=lambda: cast(Literal["dense","sparse","hybrid","hybrid_rerank"],os.getenv("DOCMIND_RETRIEVAL_MODE","dense")))
    reranker_model: str = field(default_factory=lambda: "lexical" if TEST_MODE else os.getenv("DOCMIND_RERANKER_MODEL","cross-encoder/ms-marco-MiniLM-L-6-v2"))
    reranker_revision: Optional[str] = field(default_factory=lambda: os.getenv("DOCMIND_RERANKER_REVISION") or
        ("233902d25c440f23af6f7d6e94d2946bac0bee0a" if os.getenv("DOCMIND_RERANKER_MODEL","cross-encoder/ms-marco-MiniLM-L-6-v2")=="cross-encoder/ms-marco-MiniLM-L-6-v2" else None))
    # Set DOCMIND_USE_RERANKER=0 on a low-RAM machine to skip the cross-encoder (saves ~0.5 GB).
    # Dense is the default. This switch controls availability of explicit reranking modes.
    use_reranker: bool = field(default_factory=lambda: os.getenv("DOCMIND_USE_RERANKER", "1") != "0")
    # Abstention gate. None = disabled. Calibrate with scripts/ablation.py,
    # which prints a recommended value from your labeled testset.
    min_rerank_score: Optional[float] = field(default_factory=lambda: _env_float("DOCMIND_MIN_RERANK_SCORE", None))


@dataclass
class LLMConfig:
    provider: Literal["groq", "demo"] = "demo" if TEST_MODE else "groq"   # test mode never calls the live API
    # No default model id on purpose: hosted model ids get retired (the original defaults were shut down
    # on 2026-08-16) and a silent default rots. Set DOCMIND_LLM_MODEL; scripts/list_models.py shows choices.
    groq_model: str = field(default_factory=lambda: os.getenv("DOCMIND_LLM_MODEL", ""))
    temperature: float = 0.1
    max_tokens: int = field(default_factory=lambda: int(os.getenv("DOCMIND_MAX_TOKENS", "2048")))
    reasoning_effort: Optional[str] = field(default_factory=lambda: os.getenv("DOCMIND_REASONING_EFFORT") or None)
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    max_retries: int = 6
    max_sleep_seconds: float = 120.0   # if Groq asks us to wait longer (daily cap), fail fast


@dataclass
class EvalConfig:
    # A different, stronger model than the generator avoids self-preference bias.
    judge_model: str = field(default_factory=lambda: os.getenv("DOCMIND_JUDGE_MODEL", ""))
    judge_reasoning_effort: Optional[str] = field(default_factory=lambda: os.getenv("DOCMIND_JUDGE_REASONING_EFFORT") or None)
    judge_max_workers: int = 2          # Groq free tier: keep low
    judge_timeout: int = 180
    judge_max_retries: int = 10
    score_batch_size: int = 5           # samples scored per RAGAS call (checkpointed)
    bootstrap_resamples: int = 2000


@dataclass
class ReviewConfig:
    release_mode: str = field(default_factory=lambda: os.getenv("DOCMIND_RELEASE_MODE", "draft"))
    enabled: bool = True
    min_top_score: Optional[float] = field(default_factory=lambda: _env_float("DOCMIND_REVIEW_MIN_SCORE", None))
    audit_rate: float = field(default_factory=lambda: _env_float("DOCMIND_REVIEW_AUDIT_RATE", 0.10) or 0.0)
    hold_until_approved: bool = True   # True: UI hides flagged answers until a reviewer decides
    reviewer_pin: str = field(default_factory=lambda: os.getenv("DOCMIND_REVIEWER_PIN", ""))


@dataclass
class DocMindConfig:
    data_dir: Path = HOME / "data"
    index_dir: Path = HOME / "indexes"
    reports_dir: Path = HOME / "reports"
    db_path: Path = HOME / "indexes" / "docmind.db"
    mlflow_tracking_uri: str = str(HOME / "mlruns")
    max_agent_iterations: int = field(default_factory=lambda: int(os.getenv("DOCMIND_MAX_ITERATIONS", "5")))   # lower = fewer Groq tokens
    max_upload_mb: int = 50

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.index_dir, self.reports_dir, Path(self.db_path).parent):
            Path(d).mkdir(parents=True, exist_ok=True)

    def __post_init__(self):
        self.ensure_dirs()


CONFIG = DocMindConfig()
