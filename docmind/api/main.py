"""
FastAPI application.

Three endpoints:
  POST /ingest   : add documents to the corpus
  POST /query    : ask a question, get an agent response
  GET  /health   : liveness check + index stats
"""
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from docmind.agent.react import ReActAgent
from docmind.retrieval.hybrid import HybridRetriever
from docmind.tracking.tracker import DocMindTracker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="DocMind",
    description="Intelligent document intelligence: hybrid retrieval + ReAct agent",
    version="1.0.0",
)

# Module-level singletons (initialized on first request)
_retriever: Optional[HybridRetriever] = None
_agent: Optional[ReActAgent] = None
_tracker = DocMindTracker()


def get_agent() -> ReActAgent:
    global _retriever, _agent
    if _agent is None:
        _retriever = HybridRetriever()
        _agent = ReActAgent(retriever=_retriever)
    return _agent


# ── Request/Response models ────────────────────────────────────────────────

class IngestRequest(BaseModel):
    path: str           # local file or directory path
    rebuild: bool = False

class IngestResponse(BaseModel):
    message: str
    chunks_ingested: int

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class SourceInfo(BaseModel):
    filename: str
    page: Optional[int] = None
    section: Optional[str] = None
    content_preview: str
    score: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceInfo]
    latency_ms: float
    reasoning_steps: int


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check. Returns index stats if available."""
    from docmind.retrieval.dense import FAISS_INDEX_PATH
    return {
        "status": "ok",
        "index_exists": FAISS_INDEX_PATH.exists(),
        "timestamp": time.time(),
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents(req: IngestRequest):
    """Ingest a file or directory into the DocMind corpus."""
    from docmind.ingestion.pipeline import ingest
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")

    try:
        chunks = ingest(path, rebuild_index=req.rebuild)
        return IngestResponse(
            message=f"Successfully ingested {req.path}",
            chunks_ingested=len(chunks),
        )
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Run the ReAct agent on a natural language question."""
    agent = get_agent()

    try:
        response = agent.run(req.question)
    except Exception as exc:
        logger.exception("Agent query failed")
        raise HTTPException(status_code=500, detail=str(exc))

    # Log to MLflow
    _tracker.log_query(
        question=req.question,
        answer=response.answer,
        latency_ms=response.latency_ms,
        num_sources=len(response.sources),
        num_reasoning_steps=len(response.reasoning_trace),
    )

    sources = [
        SourceInfo(
            filename=s.chunk.metadata.get("filename", s.chunk.source),
            page=s.chunk.metadata.get("page"),
            section=s.chunk.metadata.get("section"),
            content_preview=s.chunk.content[:200] + "..." if len(s.chunk.content) > 200 else s.chunk.content,
            score=round(s.score, 4),
        )
        for s in response.sources
    ]

    return QueryResponse(
        question=req.question,
        answer=response.answer,
        sources=sources,
        latency_ms=round(response.latency_ms, 1),
        reasoning_steps=len(response.reasoning_trace),
    )