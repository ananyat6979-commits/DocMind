"""
Central configuration for DocMind.
All tunable parameters live here. Nothing is hardcoded in pipeline code.

Environment overrides (all optional):
  GROQ_API_KEY              generator + judge key
  DOCMIND_USE_RERANKER=0    skip the cross-encoder (low-RAM laptops)
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
  DOCMIND_JUDGE_MODEL       RAGAS judge model on Groq
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Optional
from dotenv import load_dotenv

load_dotenv()


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


HOME = Path(os.getenv("DOCMIND_HOME", ".")).resolve()
TEST_MODE = os.getenv("DOCMIND_TEST_MODE", "") == "1"


@dataclass
class EmbeddingConfig:
    # bge-small-en-v1.5: ~33M params, 384 dims. Retrieval queries should be
    # prefixed with an instruction (see embeddings.embed_query).
    model_name: str = "BAAI/bge-small-en-v1.5"
    backend: Literal["sentence-transformers", "hash"] = "hash" if TEST_MODE else "sentence-transformers"
    device: str = field(default_factory=lambda: os.getenv("DOCMIND_DEVICE", "cpu"))   # "cuda" on Kaggle/Colab GPU
    batch_size: int = 32
    normalize_embeddings: bool = True
    query_instruction: str = "Represent this sentence for searching relevant passages: "


@dataclass
class ChunkerConfig:
    strategy: Literal["semantic", "fixed"] = "semantic"
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
    reranker_model: str = "lexical" if TEST_MODE else "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Set DOCMIND_USE_RERANKER=0 on a low-RAM machine to skip the cross-encoder (saves ~0.5 GB).
    # Retrieval is then hybrid-only and the abstention gate cannot apply. Never evaluate in this mode.
    use_reranker: bool = field(default_factory=lambda: os.getenv("DOCMIND_USE_RERANKER", "1") != "0")
    # Abstention gate. None = disabled. Calibrate with scripts/ablation.py,
    # which prints a recommended value from your labeled testset.
    min_rerank_score: Optional[float] = field(default_factory=lambda: _env_float("DOCMIND_MIN_RERANK_SCORE", None))


@dataclass
class LLMConfig:
    provider: Literal["groq", "demo"] = "demo" if (TEST_MODE and not os.getenv("GROQ_API_KEY")) else "groq"
    groq_model: str = "llama-3.1-8b-instant"
    temperature: float = 0.1
    max_tokens: int = 1024
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    max_retries: int = 6
    max_sleep_seconds: float = 120.0   # if Groq asks us to wait longer (daily cap), fail fast


@dataclass
class EvalConfig:
    # A different, stronger model than the generator avoids self-preference bias.
    judge_model: str = field(default_factory=lambda: os.getenv("DOCMIND_JUDGE_MODEL", "llama-3.3-70b-versatile"))
    judge_max_workers: int = 2          # Groq free tier: keep low
    judge_timeout: int = 180
    judge_max_retries: int = 10
    score_batch_size: int = 5           # samples scored per RAGAS call (checkpointed)
    bootstrap_resamples: int = 2000


@dataclass
class ReviewConfig:
    enabled: bool = True
    min_top_score: Optional[float] = field(default_factory=lambda: _env_float("DOCMIND_REVIEW_MIN_SCORE", None))
    audit_rate: float = field(default_factory=lambda: _env_float("DOCMIND_REVIEW_AUDIT_RATE", 0.10) or 0.0)
    hold_until_approved: bool = False   # True: UI hides flagged answers until a reviewer decides
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
