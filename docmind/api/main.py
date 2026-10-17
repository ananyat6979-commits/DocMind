"""
FastAPI service.

  GET  /health              liveness + corpus stats
  GET  /documents           list ingested documents
  POST /upload              upload and ingest a file (multipart)
  POST /ingest              ingest a file already under the data directory
  POST /query               ask a question (optionally scoped to file_ids)
  POST /feedback            thumbs up/down on an answer
  GET  /review/pending      human review queue          (reviewer PIN if configured)
  POST /review/{id}         approve | edited | rejected (reviewer PIN if configured)

Security notes:
  * /ingest only accepts paths inside CONFIG.data_dir. The old version accepted
    any server path, which let a caller ingest and then query /etc/passwd.
  * Set DOCMIND_API_KEY to require an X-API-Key header on every route except /health.
  * Set DOCMIND_REVIEWER_PIN to require X-Reviewer-Pin on review routes.
"""
import hmac
import hashlib
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from docmind.agent.react import ReActAgent
from docmind.config import CONFIG
from docmind.ingestion.loader import LOADERS
from docmind.ingestion.pipeline import ingest
from docmind.retrieval.hybrid import HybridRetriever
from docmind.review.store import ReviewStore, DECISIONS
from docmind.store import CorpusStore
from docmind.telemetry.events import EventLog

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="DocMind", description="Hybrid retrieval + ReAct agent with human review", version="2.0.0")

_agent: Optional[ReActAgent] = None
_lock = threading.Lock()


def get_agent() -> ReActAgent:
    global _agent
    if _agent is None:
        with _lock:
            if _agent is None:
                _agent = ReActAgent(retriever=HybridRetriever())
    return _agent


def _eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("DOCMIND_API_KEY", "")
    if expected and not (x_api_key and _eq(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def require_reviewer(x_reviewer_pin: Optional[str] = Header(default=None)) -> str:
    pin = CONFIG.review.reviewer_pin
    if pin and not (x_reviewer_pin and _eq(x_reviewer_pin, pin)):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Reviewer-Pin")
    return "reviewer"


class IngestRequest(BaseModel):
    path: str
    rebuild: bool = False


class IngestResponse(BaseModel):
    message: str
    file_ids: List[str]
    chunks_added: int
    cached: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    file_ids: Optional[List[str]] = None
    user: str = "anonymous"


class SourceInfo(BaseModel):
    chunk_id: str
    filename: str
    page: Optional[int] = None
    section: Optional[str] = None
    content_preview: str
    score: float
    cited: bool


class QueryResponse(BaseModel):
    response_id: str
    question: str
    answer: str
    abstained: bool
    review_status: str
    review_reasons: List[str]
    citation_validity: Optional[float]
    sources: List[SourceInfo]
    latency_ms: float
    llm_calls: int


class FeedbackRequest(BaseModel):
    response_id: str
    helpful: bool
    user: str = "anonymous"
    comment: str = ""


class ReviewDecision(BaseModel):
    decision: str
    reviewer: str = "reviewer"
    edited_answer: Optional[str] = None
    note: str = ""


@app.get("/health")
def health():
    docs = CorpusStore().list_documents()
    return {"status": "ok", "documents": len(docs), "chunks": sum(d["n_chunks"] for d in docs),
            "llm_provider": CONFIG.llm.provider, "timestamp": time.time()}


@app.get("/documents", dependencies=[Depends(require_api_key)])
def documents():
    return CorpusStore().list_documents()


def _ingest_response(report, label: str) -> IngestResponse:
    if report.failures and not report.file_ids:
        raise HTTPException(status_code=422, detail="; ".join(f.error for f in report.failures))
    return IngestResponse(message=f"Ingested {label}", file_ids=report.file_ids,
                          chunks_added=report.chunks_added, cached=report.n_cached)


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def ingest_documents(req: IngestRequest):
    root = CONFIG.data_dir.resolve()
    path = Path(req.path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if root != path and root not in path.parents:
        raise HTTPException(status_code=400, detail=f"Path must be inside {root}")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")
    try:
        return _ingest_response(ingest(path, rebuild_index=req.rebuild), req.path)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/upload", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "upload").name
    if Path(name).suffix.lower() not in LOADERS:
        raise HTTPException(status_code=415, detail=f"Unsupported type. Supported: {sorted(LOADERS)}")
    dest_dir = CONFIG.data_dir / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    limit = CONFIG.max_upload_mb * 1024 * 1024
    written = 0
    digest = hashlib.sha256()
    temp = dest_dir / f".{uuid.uuid4().hex}.upload"
    with open(temp, "wb") as out:
        while chunk := file.file.read(1 << 20):
            written += len(chunk)
            if written > limit:
                out.close(); temp.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"File exceeds {CONFIG.max_upload_mb} MB")
            digest.update(chunk)
            out.write(chunk)
    dest = dest_dir / digest.hexdigest() / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        temp.unlink()
    else:
        temp.replace(dest)
    try:
        return _ingest_response(ingest(dest), name)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(req: QueryRequest):
    agent = get_agent()
    if agent.retriever.is_empty():
        raise HTTPException(status_code=409, detail="No documents ingested yet. POST /upload first.")
    try:
        resp = agent.run(req.question, file_ids=req.file_ids)
    except Exception as exc:
        logger.exception("Agent query failed")
        raise HTTPException(status_code=500, detail=str(exc))

    status = ReviewStore().record(resp, user=req.user)
    EventLog().log("query", user=req.user, response_id=resp.response_id, latency_ms=round(resp.latency_ms),
                   llm_calls=resp.llm_calls, abstained=resp.abstained, needs_review=resp.needs_review,
                   n_sources=len(resp.sources), finish=resp.finish_reason,
                   citation_validity=resp.citation_report.get("validity"))
    cited_ids = {s.chunk.chunk_id for s in resp.cited_sources}
    return QueryResponse(
        response_id=resp.response_id, question=req.question, answer=resp.answer, abstained=resp.abstained,
        review_status=status, review_reasons=resp.review_reasons,
        citation_validity=resp.citation_report.get("validity"),
        sources=[SourceInfo(
            chunk_id=s.chunk.chunk_id, filename=s.chunk.filename(),
            page=s.chunk.metadata.get("page"), section=s.chunk.metadata.get("section"),
            content_preview=(s.chunk.content[:200] + "...") if len(s.chunk.content) > 200 else s.chunk.content,
            score=round(s.score, 4), cited=s.chunk.chunk_id in cited_ids) for s in resp.sources],
        latency_ms=round(resp.latency_ms, 1), llm_calls=resp.llm_calls,
    )


@app.post("/feedback", dependencies=[Depends(require_api_key)])
def feedback(req: FeedbackRequest):
    if ReviewStore().get(req.response_id) is None:
        raise HTTPException(status_code=404, detail="Unknown response_id")
    EventLog().log("feedback", user=req.user, response_id=req.response_id, helpful=req.helpful, comment=req.comment)
    return {"ok": True}


@app.get("/review/pending", dependencies=[Depends(require_api_key)])
def review_pending(_: str = Depends(require_reviewer), limit: int = 50):
    return ReviewStore().pending(limit=limit)


@app.post("/review/{response_id}", dependencies=[Depends(require_api_key)])
def review_decide(response_id: str, body: ReviewDecision, _: str = Depends(require_reviewer)):
    if body.decision not in DECISIONS:
        raise HTTPException(status_code=400, detail=f"decision must be one of {sorted(DECISIONS)}")
    try:
        ok = ReviewStore().decide(response_id, body.decision, body.reviewer, body.edited_answer, body.note)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not ok:
        raise HTTPException(status_code=404, detail="Not found or already decided")
    EventLog().log("review_decision", user=body.reviewer, response_id=response_id, decision=body.decision)
    return {"ok": True}
