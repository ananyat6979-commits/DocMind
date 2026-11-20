import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from docmind.agent.reader import CitedReader
from docmind.config import CONFIG
from docmind.llm.demo import DemoLLM
from docmind.llm.execution import DeadlineExceeded
from docmind.models import Chunk, SearchResult
from docmind.review.store import ReviewStore


def source():
    return SearchResult(
        Chunk(
            "The shutdown valve closes within two seconds.",
            "d",
            "manual.pdf",
            0,
            metadata={
                "page": 1,
                "language_info": {"language": "en"},
                "document_language_info": {"language": "en"},
            },
            file_id="manual",
            chunk_id="exact-id",
        ),
        1,
        "dense",
    )


class Retriever:
    def search(self, question, file_ids=None):
        return [source()]

    def is_empty(self):
        return False


class AsyncReader:
    def __init__(self):
        self.active = 0
        self.peak = 0

    async def acomplete(self, messages, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.05)
            return json.dumps(
                {
                    "claims": [
                        {
                            "text": "Two seconds.",
                            "evidence": [
                                {
                                    "chunk_id": "exact-id",
                                    "quote": source().chunk.content,
                                }
                            ],
                        }
                    ]
                }
            )
        finally:
            self.active -= 1


def test_concurrent_requests_are_independent_and_unblocked(monkeypatch):
    monkeypatch.setattr(
        "docmind.language.language_info", lambda text: {"language": "en"}
    )
    backend = AsyncReader()
    reader = CitedReader(Retriever(), backend)

    async def exercise():
        return await asyncio.gather(
            *(reader.arun(f"Closing time question {i}") for i in range(4))
        )

    results = asyncio.run(exercise())
    assert backend.peak == 4
    assert len({r.response_id for r in results}) == 4
    assert all(r.llm_calls == 1 and r.release_state == "draft" for r in results)
    assert all(r.evidence[0]["text"] == source().chunk.content for r in results)
    reviews = ReviewStore()
    for result in results:
        assert reviews.record(result) == "draft"
        saved = reviews.get(result.response_id)
        assert saved["response_info"]["evidence"] == result.evidence
    assert reviews.pending() == []
    assert reviews.decide(results[0].response_id, "edited", "human", "Three seconds.")


def test_real_short_language_veto():
    from docmind.review.release import apply_policy
    from tests.test_draft_release import response

    assert apply_policy(response()).release_state == "draft"
    for text in ("Deux secondes.", "Zwei Sekunden.", "Kaksi sekuntia."):
        assert (
            "answer_language_unvalidated" in apply_policy(response(text)).hold_reasons
        )


def test_deadline_cancels_async_reader():
    class Slow:
        async def acomplete(self, messages, **kwargs):
            await asyncio.sleep(10)

    with pytest.raises((DeadlineExceeded, TimeoutError)):
        CitedReader(Retriever(), Slow(), deadline_seconds=0.02).run("How long?")


def test_current_api_persists_quotes_and_hides_held_evidence(monkeypatch):
    import docmind.api.main as m

    monkeypatch.setenv("DOCMIND_API_KEY", "test-key")
    monkeypatch.setattr(
        "docmind.language.language_info", lambda text: {"language": "en"}
    )
    monkeypatch.setattr(m, "_agent", CitedReader(Retriever(), DemoLLM()))
    import numpy as np

    from docmind.store import CorpusStore

    chunk = source().chunk
    CorpusStore().add_document(
        {
            "file_id": chunk.file_id,
            "filename": "manual.pdf",
            "source": "manual.pdf",
            "sha256": "fixture",
            "n_units": 1,
        },
        [chunk],
        np.ones((1, 384), dtype="float32") / np.sqrt(384),
    )
    client = TestClient(m.app, headers={"X-API-Key": "test-key"})
    reply = client.post("/query", json={"question": "What is the closing time?"})
    assert reply.status_code == 200, reply.text
    item = reply.json()
    assert item["review_status"] == "draft" and item["evidence"]
    fetched = client.get("/responses/" + item["response_id"]).json()
    assert fetched["evidence"] == item["evidence"]
    assert fetched["sources"][0]["content"] == source().chunk.content
    assert not fetched["sources"][0]["content_truncated"]
    assert fetched["attribution_status"] == "citation_found_support_not_verified"
    monkeypatch.setattr(CONFIG.review, "release_mode", "strict")
    monkeypatch.setattr(CONFIG.review, "hold_until_approved", False)
    held = client.post("/query", json={"question": "What is the closing time?"}).json()
    assert (
        held["review_status"] == "pending"
        and not held["evidence"]
        and not held["sources"]
    )
    assert "two seconds" not in held["answer"].lower()


def test_frozen_and_adaptive_loops_have_separate_retrieval_contracts(
    monkeypatch, scripted
):
    from docmind.agent.reader import CitedReActReader

    monkeypatch.setattr(
        "docmind.language.language_info", lambda text: {"language": "en"}
    )

    class Counting(Retriever):
        def __init__(self):
            self.queries = []

        def search(self, question, file_ids=None):
            self.queries.append(question)
            return super().search(question, file_ids)

    final = json.dumps(
        {
            "claims": [
                {
                    "text": "Two seconds.",
                    "evidence": [
                        {"chunk_id": "exact-id", "quote": source().chunk.content}
                    ],
                }
            ]
        }
    )
    for adaptive in (False, True):
        retrieval = Counting()
        backend = scripted(
            [
                'Thought: Search again.\nAction: search_documents("closing time")',
                "Final Answer: " + final,
            ]
        )
        result = CitedReActReader(retrieval, backend, adaptive=adaptive).run(
            "How long?"
        )
        assert result.llm_calls == 2 and result.release_state == "draft"
        assert len(retrieval.queries) == (2 if adaptive else 1)


def test_api_cancels_stalled_upstream_and_returns_504(monkeypatch):
    import docmind.api.main as m

    cancelled = []

    class Stalled:
        async def acomplete(self, messages, **kwargs):
            try:
                await asyncio.sleep(20)
            finally:
                cancelled.append(True)

    monkeypatch.setenv("DOCMIND_API_KEY", "test-key")
    monkeypatch.setenv("DOCMIND_QUERY_TIMEOUT_SECONDS", ".05")
    monkeypatch.setattr(m, "_agent", CitedReader(Retriever(), Stalled()))
    client = TestClient(m.app, headers={"X-API-Key": "test-key"})
    started = time.monotonic()
    response = client.post("/query", json={"question": "What is the closing time?"})
    assert response.status_code == 504 and time.monotonic() - started < 2
    assert cancelled == [True]


def test_sqlite_rejects_deleted_evidence_and_removes_abstention_context(make_file):
    from docmind.ingestion.pipeline import ingest
    from docmind.models import AgentResponse
    from docmind.services import erase_document
    from docmind.store import CorpusStore
    from tests.conftest import ALPHA

    store = CorpusStore()
    file_id = ingest(make_file("manual.txt", ALPHA), store=store).file_ids[0]
    chunk = store.load_corpus()[0][0]
    result = SearchResult(chunk, 1, "dense")
    reviews = ReviewStore()
    abstention = AgentResponse(
        "Not found.",
        [],
        [],
        "An absent fact?",
        1,
        abstained=True,
        document_context={"documents": [{"file_id": file_id}]},
    )
    reviews.record(abstention)
    assert erase_document(file_id, store)
    assert reviews.get(abstention.response_id) is None
    stale = AgentResponse(
        "A stale answer.", [result], [], "Question", 1, cited_sources=[result]
    )
    with pytest.raises(ValueError, match="Corpus changed"):
        reviews.record(stale)
    assert reviews.get(stale.response_id) is None


def test_cancellation_closes_a_real_stalled_http_connection():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import httpx

    from docmind.llm.execution import CompletionSession

    closed = threading.Event()
    received = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received.set()
            self.connection.settimeout(10)
            try:
                if self.connection.recv(1) == b"":
                    closed.set()
            except (ConnectionResetError, ConnectionAbortedError):
                closed.set()
            except OSError:
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def exercise():
        async with httpx.AsyncClient(
            timeout=None, trust_env=False, verify=False
        ) as client:

            class StalledHTTP:
                async def acomplete(self, messages, **kwargs):
                    await client.post(
                        "http://127.0.0.1:" + str(server.server_port), json=messages
                    )

            session = CompletionSession(StalledHTTP(), time.monotonic() + 30)
            task = asyncio.create_task(
                session.complete(
                    [{"role": "user", "content": "Synthetic local cancellation check"}]
                )
            )
            assert await asyncio.to_thread(received.wait, 10), (
                "The local upstream did not receive the request"
            )
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            assert await asyncio.to_thread(closed.wait, 2), (
                "The cancelled HTTP connection remained open"
            )
            assert session.calls == 1 and session.usage()["usage_unknown_calls"] == 1

    try:
        asyncio.run(exercise())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_react_protocol_markers_inside_json_quotes_are_data(monkeypatch, scripted):
    from dataclasses import replace

    from docmind.agent.reader import CitedReActReader

    monkeypatch.setattr(
        "docmind.language.language_info", lambda text: {"language": "en"}
    )
    text = "The document explains that Final Answer: is a protocol label."
    item = replace(source(), chunk=replace(source().chunk, content=text))

    class Quoted(Retriever):
        def search(self, question, file_ids=None):
            return [item]

    payload = json.dumps(
        {
            "short_answer": "A protocol label.",
            "claims": [
                {
                    "text": "A protocol label.",
                    "evidence": [{"chunk_id": item.chunk.chunk_id, "quote": text}],
                }
            ],
        }
    )
    for raw in (payload, "```json\n" + payload + "\n```"):
        result = CitedReActReader(Quoted(), scripted([raw])).run(
            "What does the document explain?"
        )
        assert (
            result.finish_reason == "final_answer" and result.release_state == "draft"
        )


def test_complete_abstention_does_not_claim_a_found_citation(monkeypatch):
    import docmind.api.main as m

    monkeypatch.setenv("DOCMIND_API_KEY", "test-key")
    monkeypatch.setattr(
        "docmind.language.language_info", lambda text: {"language": "en"}
    )

    class EnglishCorpus:
        def document_context(self, file_ids=None):
            return {"partial": False, "languages": ["en"], "documents": []}

    class Empty(Retriever):
        store = EnglishCorpus()

        def search(self, question, file_ids=None):
            return []

    monkeypatch.setattr(m, "_agent", CitedReader(Empty(), DemoLLM()))
    client = TestClient(m.app, headers={"X-API-Key": "test-key"})
    result = client.post("/query", json={"question": "What is the absent fact?"}).json()
    assert result["abstained"] and result["review_status"] == "draft"
    assert (
        result["attribution_status"] == "absence_not_verified"
        and not result["evidence"]
    )
    saved = client.get("/responses/" + result["response_id"]).json()
    assert saved["attribution_status"] == "absence_not_verified"


def test_unknown_selected_document_does_not_become_a_not_found_answer(monkeypatch):
    import docmind.api.main as m

    monkeypatch.setenv("DOCMIND_API_KEY", "test-key")
    monkeypatch.setattr(m, "_agent", CitedReader(Retriever(), DemoLLM()))
    client = TestClient(m.app, headers={"X-API-Key": "test-key"})
    result = client.post(
        "/query",
        json={"question": "What is the closing time?", "file_ids": ["missing-file"]},
    )
    assert result.status_code == 404 and "refresh" in result.json()["detail"]


def test_complete_reader_prompt_schema_is_valid_json_with_root_short_answer():
    import json

    from docmind.agent.reader import SYSTEM

    example = SYSTEM.split("complete example: ", 1)[1].split("\n", 1)[0].rstrip(".")
    payload = json.loads(example)
    assert isinstance(payload["claims"], list) and len(payload["claims"]) == 1
    assert payload["short_answer"] in payload["claims"][0]["text"]
    assert "short_answer" not in payload["claims"][0]


def test_invalid_auxiliary_summary_does_not_release_unmatched_evidence(monkeypatch):
    monkeypatch.setenv("DOCMIND_RELEASE_MODE", "draft")
    monkeypatch.setenv("DOCMIND_ATTRIBUTION", "0")
    session = SimpleNamespace(calls=1, usage=lambda: {})
    payload = {
        "short_answer": "unlinked",
        "claims": [
            {
                "text": "Two seconds.",
                "evidence": [{"chunk_id": "exact-id", "quote": source().chunk.content}],
            }
        ],
    }
    engine = CitedReader(Retriever())
    response = engine.assemble(
        "How long does the shutdown valve take to close?",
        [source()],
        json.dumps(payload),
        time.monotonic(),
        session,
    )
    assert response.release_state == "draft" and response.short_answer is None
    assert (
        response.answer == "Two seconds." and response.finish_reason == "final_answer"
    )
    payload["claims"][0]["evidence"][0]["quote"] = "invented evidence"
    response = engine.assemble(
        "How long does the shutdown valve take to close?",
        [source()],
        json.dumps(payload),
        time.monotonic(),
        session,
    )
    assert (
        response.release_state == "held"
        and "invalid_or_missing_quote" in response.hold_reasons
    )


def test_numeric_conflict_is_held_separately_from_valid_quote_location(monkeypatch):
    monkeypatch.setenv("DOCMIND_ATTRIBUTION", "0")
    payload = {
        "short_answer": "900 seconds.",
        "claims": [
            {
                "text": "900 seconds.",
                "evidence": [{"chunk_id": "exact-id", "quote": source().chunk.content}],
            }
        ],
    }
    session = SimpleNamespace(calls=1, usage=lambda: {})
    result = CitedReader(Retriever()).assemble(
        "How long does the shutdown valve take to close?",
        [source()],
        json.dumps(payload),
        time.monotonic(),
        session,
    )
    assert result.citation_report["quote_validation"]["quotes_matched"]
    assert result.citation_report["invalid"] == []
    assert "numeric_evidence_conflict" in result.hold_reasons
    assert result.release_state == "held"
