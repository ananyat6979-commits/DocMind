"""
FastAPI service.

  GET  /health              liveness only
  GET  /documents           list ingested documents
  POST /upload              upload and ingest a file (multipart)
  POST /ingest              ingest a file already under the data directory
  POST /query               ask a question (optionally scoped to file_ids)
  POST /feedback            thumbs up/down on an answer
  GET  /review/pending      human review queue          (reviewer PIN required)
  POST /review/{id}         approve | edited | rejected (reviewer PIN required)

Security notes:
  * /ingest only accepts paths inside CONFIG.data_dir. The old version accepted
    any server path, which let a caller ingest and then query /etc/passwd.
  * Configure DOCMIND_API_KEY for the required X-API-Key header on every route except /health.
  * Configure DOCMIND_REVIEWER_PIN for the required X-Reviewer-Pin on review routes.
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
from typing import List, Optional, Literal

from contextlib import asynccontextmanager
from functools import lru_cache
from fastapi import Request, Query, Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from docmind.agent.react import ReActAgent
from docmind.config import CONFIG
from docmind.ingestion.loader import LOADERS
from docmind.ingestion.pipeline import ingest
from docmind.retrieval.hybrid import HybridRetriever
from docmind.review.store import ReviewStore, DECISIONS
from docmind.store import CorpusStore, EmbeddingModelMismatch
from docmind.telemetry.events import EventLog

from docmind.tenancy import database_scope,data_root

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app):
    from docmind.jobs import IngestionQueue
    if os.getenv('DOCMIND_DATABASE_URL'):
        import json
        from docmind.postgres import DatabaseScope
        records=json.loads(Path(os.environ['DOCMIND_CREDENTIALS_FILE']).read_text(encoding='utf-8-sig'))
        tenants={r['tenant_id'] for r in records}
        if not 1<=len(tenants)<=16:raise RuntimeError('Production profile supports one to sixteen provisioned tenants')
        app.state.tenant_queues={tenant:IngestionQueue(DatabaseScope(os.environ['DOCMIND_DATABASE_URL'],tenant)) for tenant in tenants}
        for queue in app.state.tenant_queues.values():queue.start()
        try:yield
        finally:
            for queue in app.state.tenant_queues.values():queue.close()
        return
    queue = IngestionQueue()
    app.state.ingestion_queue = queue
    from docmind.uploads import sweep
    sweep()
    queue.start()
    yield
    queue.close()


app = FastAPI(lifespan=lifespan, title="DocMind", description="Hybrid retrieval + ReAct agent with human review", version="2.0.0")

from docmind.observability import RequestMonitor
_monitor = RequestMonitor(limit_per_minute=int(os.getenv("DOCMIND_REQUESTS_PER_MINUTE", "60")),limit_per_ip=int(os.getenv("DOCMIND_REQUESTS_PER_IP_PER_MINUTE","300")))
app.middleware("http")(_monitor)


_agent: Optional[ReActAgent] = None
_lock = threading.Lock()


@lru_cache(maxsize=32)
def _persistent_stores(db_path):
    return CorpusStore(db_path), ReviewStore(db_path), EventLog(db_path)


def get_store():
    return _persistent_stores(database_scope())[0]


def get_reviews():
    return _persistent_stores(database_scope())[1]


def get_events():
    return _persistent_stores(database_scope())[2]


_tenant_agents: dict = {}


def get_queue(request):
    if os.getenv('DOCMIND_DATABASE_URL'):
        return request.app.state.tenant_queues[request.state.principal.tenant_id]
    return request.app.state.ingestion_queue


def get_agent() -> ReActAgent:
    if os.getenv('DOCMIND_DATABASE_URL'):
        scope=database_scope()
        with _lock:
            if scope not in _tenant_agents:
                _tenant_agents[scope]=ReActAgent(retriever=HybridRetriever(get_store()))
            return _tenant_agents[scope]
    global _agent
    if _agent is None:
        with _lock:
            if _agent is None:
                _agent = ReActAgent(retriever=HybridRetriever(get_store()))
    return _agent


def _eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def require_api_key(request: Request) -> None:
    if not getattr(request.state,"principal",None):
        raise HTTPException(status_code=401,detail="Authenticated principal required")
    if "reader" not in request.state.principal.roles:
        raise HTTPException(status_code=403,detail="Reader permission required")


def require_ingestor(request: Request) -> None:
    if "ingestor" not in request.state.principal.roles:
        raise HTTPException(status_code=403,detail="Ingestor permission required")


def require_reviewer(request: Request, x_reviewer_pin: Optional[str] = Header(default=None)) -> str:
    if os.getenv("DOCMIND_CREDENTIALS_FILE"):
        if "reviewer" not in request.state.principal.roles:
            raise HTTPException(status_code=403,detail="Reviewer permission required")
        return request.state.principal.actor_id
    pin = CONFIG.review.reviewer_pin
    if not pin: raise HTTPException(status_code=503,detail="Configure DOCMIND_REVIEWER_PIN before reviewing")
    if not x_reviewer_pin or not _eq(x_reviewer_pin,pin):
        raise HTTPException(status_code=401,detail="Invalid reviewer credential")
    return "shared-reviewer"


def require_admin(request: Request, x_admin_pin: Optional[str] = Header(default=None)) -> None:
    if os.getenv("DOCMIND_CREDENTIALS_FILE"):
        if "admin" not in request.state.principal.roles:
            raise HTTPException(status_code=403,detail="Administrator permission required")
        return
    expected=os.getenv("DOCMIND_ADMIN_PIN", "")
    if not expected: raise HTTPException(status_code=503,detail="Configure DOCMIND_ADMIN_PIN before library mutations")
    if not x_admin_pin or not _eq(x_admin_pin,expected): raise HTTPException(status_code=401,detail="Invalid administrator credential")


class IngestRequest(BaseModel):
    path: str = Field(min_length=1,max_length=1024)
    rebuild: bool = False
    expected_language: Optional[Literal["en","hi","fr","de","fi"]] = None
    encoding: Optional[Literal["cp1252"]] = None


class IngestResponse(BaseModel):
    message: str
    file_ids: List[str]
    chunks_added: int
    cached: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    file_ids: Optional[List[str]] = Field(default=None, max_length=100)
    user: str = Field(default="anonymous", max_length=100)


class SourceInfo(BaseModel):
    chunk_id: str
    filename: str
    file_id: str = ""
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
    attribution: Optional[dict] = None
    attribution_status: str = "not_checked"
    sources: List[SourceInfo]
    latency_ms: float
    llm_calls: int


class FeedbackRequest(BaseModel):
    response_id: str
    helpful: bool
    user: str = Field(default="anonymous", max_length=100)
    comment: str = Field(default="", max_length=4000)


class ReviewDecision(BaseModel):
    decision: str
    reviewer: str = Field(default="reviewer", max_length=100)
    edited_answer: Optional[str] = Field(default=None, max_length=20000)
    note: str = Field(default="", max_length=4000)


@app.get("/health")
def health():
    return {"status":"ok", "timestamp":time.time()}


@app.get("/documents", dependencies=[Depends(require_api_key)])
def documents():
    return get_store().list_documents()


def _ingest_response(report, label: str) -> IngestResponse:
    if report.failures and not report.file_ids:
        raise HTTPException(status_code=422, detail="; ".join(f.error for f in report.failures))
    return IngestResponse(message=f"Ingested {label}", file_ids=report.file_ids,
                          chunks_added=report.chunks_added, cached=report.n_cached)


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key), Depends(require_ingestor)])
def ingest_documents(req: IngestRequest):
    root = data_root().resolve()
    path = Path(req.path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if root != path and root not in path.parents:
        raise HTTPException(status_code=400, detail=f"Path must be inside {root}")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")
    try:
        return _ingest_response(ingest(path, store=get_store(),rebuild_index=req.rebuild,expected_language=req.expected_language,encoding=req.encoding), req.path)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=422, detail=str(exc))


def _persist_upload(file):
    name = Path(file.filename or "upload").name
    if Path(name).suffix.lower() not in LOADERS:
        raise HTTPException(status_code=415, detail=f"Unsupported type. Supported: {sorted(LOADERS)}")
    dest_dir = data_root() / "uploads"
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
    from docmind.uploads import acquire
    acquire(dest)
    return dest, name


@app.post("/upload", response_model=IngestResponse, dependencies=[Depends(require_api_key), Depends(require_ingestor)])
def upload(file: UploadFile = File(...), expected_language: Optional[Literal["en","hi","fr","de","fi"]] = None, encoding: Optional[Literal["cp1252"]] = None):
    dest, name = _persist_upload(file)
    try:
        return _ingest_response(ingest(dest,store=get_store(),expected_language=expected_language,encoding=encoding), name)
    except Exception as exc:
        from docmind.ingestion.parser_client import ParserUnavailable
        raise HTTPException(status_code=503 if isinstance(exc, ParserUnavailable) else 422, detail=str(exc))
    finally:
        from docmind.uploads import cleanup
        cleanup(dest,get_store().db_path, release=True)


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(req: QueryRequest, request: Request):
    try:
        agent = get_agent()
        if agent.retriever.is_empty():
            raise HTTPException(status_code=409, detail="No documents ingested yet. POST /upload first.")
        resp = agent.run(req.question, file_ids=req.file_ids)
    except EmbeddingModelMismatch:
        raise HTTPException(status_code=503, detail="Corpus embeddings differ from the active configuration; rebuild an isolated corpus from retained uploads")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Agent query failed")
        raise HTTPException(status_code=500, detail="Query failed; consult server logs")

    status = get_reviews().record(resp, user=request.state.principal.actor_id)
    get_events().log("query", user=request.state.principal.actor_id, response_id=resp.response_id, latency_ms=round(resp.latency_ms),
                   llm_calls=resp.llm_calls, abstained=resp.abstained, needs_review=resp.needs_review,
                   n_sources=len(resp.sources), finish=resp.finish_reason,
                   request_id=getattr(request.state,"request_id",None), timings=resp.timings,
                   citation_validity=resp.citation_report.get("validity"))
    held = status == "pending" and CONFIG.review.hold_until_approved
    cited_ids = {s.chunk.chunk_id for s in resp.cited_sources}
    return QueryResponse(
        response_id=resp.response_id, question=req.question,
        answer="This answer is awaiting human review." if status=="pending" and CONFIG.review.hold_until_approved else resp.answer,
        abstained=resp.abstained,
        review_status=status, review_reasons=resp.review_reasons,
        citation_validity=None if held else resp.citation_report.get("validity"),
        attribution=None if held else resp.citation_report.get("attribution"),
        attribution_status="withheld" if held else "model_judgment",
        sources=[SourceInfo(
            chunk_id=s.chunk.chunk_id, filename=s.chunk.filename(),file_id=s.chunk.file_id,
            page=s.chunk.metadata.get("page"), section=s.chunk.metadata.get("section"),
            content_preview=(s.chunk.content[:200] + "...") if len(s.chunk.content) > 200 else s.chunk.content,
            score=round(s.score, 4), cited=s.chunk.chunk_id in cited_ids) for s in ([] if held else resp.sources)],
        latency_ms=round(resp.latency_ms, 1), llm_calls=resp.llm_calls,
    )


@app.post("/feedback", dependencies=[Depends(require_api_key)])
def feedback(req: FeedbackRequest, request: Request):
    item = get_reviews().get(req.response_id)
    if item is None or item["user"] != request.state.principal.actor_id:
        raise HTTPException(status_code=404, detail="Unknown response_id")
    get_events().log("feedback", user=request.state.principal.actor_id, response_id=req.response_id, helpful=req.helpful, comment=req.comment)
    return {"ok": True}


@app.get("/review/pending", dependencies=[Depends(require_api_key)])
def review_pending(_: str = Depends(require_reviewer), limit: int = Query(default=50, ge=1, le=200)):
    return get_reviews().pending(limit=limit)


@app.post("/review/{response_id}", dependencies=[Depends(require_api_key)])
def review_decide(response_id: str, body: ReviewDecision, reviewer: str = Depends(require_reviewer)):
    if body.decision not in DECISIONS:
        raise HTTPException(status_code=400, detail=f"decision must be one of {sorted(DECISIONS)}")
    try:
        ok = get_reviews().decide(response_id, body.decision, reviewer, body.edited_answer, body.note)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not ok:
        raise HTTPException(status_code=404, detail="Not found or already decided")
    get_events().log("review_decision", user=reviewer, response_id=response_id, decision=body.decision)
    return {"ok": True}


@app.delete("/documents/{file_id}", dependencies=[Depends(require_api_key), Depends(require_admin)])
def delete_document(file_id: str):
    from docmind.services import erase_document
    if not erase_document(file_id,store=get_store()):
        raise HTTPException(status_code=404, detail="Unknown document")
    return {"ok": True}


@app.post("/ingest/jobs", status_code=202, dependencies=[Depends(require_api_key), Depends(require_ingestor)])
def enqueue_ingestion(req: IngestRequest, request: Request):
    root = data_root().resolve()
    path = (root / req.path).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=400, detail="Choose a file within the data directory")
    if req.rebuild:
        raise HTTPException(status_code=400, detail="Queued rebuild is not supported")
    try:
        return {"job_id": get_queue(request).enqueue(path, owner=request.state.principal.actor_id,expected_language=req.expected_language,encoding=req.encoding), "status": "queued"}
    except ValueError as exc:
        raise HTTPException(status_code=429, detail=str(exc))


@app.get("/ingest/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def ingestion_status(job_id: str, request: Request):
    row = get_queue(request).get(job_id)
    if not row or row["owner"] != request.state.principal.actor_id:
        raise HTTPException(status_code=404, detail="Unknown ingestion job")
    return row


@app.delete("/ingest/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def cancel_ingestion(job_id: str, request: Request):
    row=get_queue(request).get(job_id)
    if not row or row["owner"] != request.state.principal.actor_id:
        raise HTTPException(status_code=404,detail="Unknown ingestion job")
    if not get_queue(request).cancel(job_id):
        raise HTTPException(status_code=409, detail="Only queued jobs can be cancelled")
    return {"ok": True}


@app.get("/ready", dependencies=[Depends(require_api_key)])
def readiness():
    if CONFIG.llm.provider != "demo" and (not CONFIG.llm.groq_api_key or not CONFIG.llm.groq_model):
        raise HTTPException(status_code=503, detail="Generation provider is not configured")
    store = get_store()
    try:
        store.ensure_embedding_compatible()
    except EmbeddingModelMismatch:
        raise HTTPException(status_code=503, detail="Corpus embeddings differ from the active configuration; rebuild an isolated corpus from retained uploads")
    return {"status": "ready", "corpus_version": store.version()}


@app.post("/uploads/jobs", status_code=202, dependencies=[Depends(require_api_key), Depends(require_ingestor)])
def enqueue_upload(request: Request, file: UploadFile = File(...), expected_language: Optional[Literal["en","hi","fr","de","fi"]] = None, encoding: Optional[Literal["cp1252"]] = None):
    dest, name = _persist_upload(file)
    try:
        return {"job_id": get_queue(request).enqueue(dest, owner=request.state.principal.actor_id,expected_language=expected_language,encoding=encoding), "status":"queued", "filename":name}
    except ValueError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    finally:
        from docmind.uploads import cleanup
        cleanup(dest,get_store().db_path, release=True)


@app.get("/metrics", dependencies=[Depends(require_api_key)])
def metrics(request: Request):
    if os.getenv('DOCMIND_DATABASE_URL'):
        with _monitor.lock:
            return {'requests_by_status':dict(_monitor.tenant_counts.get(request.state.principal.tenant_id,{})),
                'quota_per_minute':_monitor.limit,'scope':'This API worker and authenticated tenant; shared quota enforcement is database-wide'}
    with _monitor.lock:
        return {"requests_by_status":dict(_monitor.counts), "quota_per_minute":_monitor.limit}


@app.get("/responses/{response_id}", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def response_result(response_id: str, request: Request):
    item = get_reviews().get(response_id)
    if item is None: raise HTTPException(status_code=404, detail="Response no longer available")
    if item["user"] != request.state.principal.actor_id:
        raise HTTPException(status_code=404, detail="Response not available to this principal")
    status = item["status"]
    visible = status in ("auto_approved", "approved", "edited")
    info = item.get("response_info", {})
    edited = status == "edited"
    report = info.get("citation_report", {})
    return QueryResponse(response_id=response_id,question=item["question"],
        answer=(item["edited_answer"] or item["answer"]) if visible else "This response is not released.",
        abstained=False if edited else info.get("abstained",False),review_status=status,review_reasons=item["reasons"],
        citation_validity=report.get("validity") if visible and not edited else None,
        attribution=report.get("attribution") if visible and not edited else None,
        attribution_status="human_edited_not_rechecked" if edited else ("model_judgment" if visible else "withheld"),
        sources=[SourceInfo(chunk_id=s["chunk_id"],filename=s["filename"],file_id=s.get("file_id",""),
            page=s.get("page"),section=s.get("section"),content_preview=s["content"][:200],
            score=s["score"],cited=bool(s.get("cited")) and not edited) for s in item["sources"]] if visible else [],
        latency_ms=info.get("latency_ms",0),llm_calls=info.get("llm_calls",0))


@app.get("/whoami", dependencies=[Depends(require_api_key)])
def whoami(request: Request):
    principal=request.state.principal
    return {"actor_id":principal.actor_id,"tenant_id":principal.tenant_id,"roles":sorted(principal.roles),
            "identity_mode":"individual" if os.getenv("DOCMIND_CREDENTIALS_FILE") else "shared credential"}
