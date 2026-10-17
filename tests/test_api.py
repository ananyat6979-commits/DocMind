import io
import pytest
from fastapi.testclient import TestClient

from docmind.agent.react import ReActAgent
from docmind.config import CONFIG
from docmind.llm.demo import DemoLLM
from docmind.retrieval.hybrid import HybridRetriever
from tests.conftest import ALPHA, BETA


@pytest.fixture
def client(monkeypatch):
    import docmind.api.main as m
    monkeypatch.setattr(m, "_agent", ReActAgent(retriever=HybridRetriever(), llm=DemoLLM()))
    monkeypatch.delenv("DOCMIND_API_KEY", raising=False)
    CONFIG.review.reviewer_pin = ""
    return TestClient(m.app)


def _upload(client, name, text):
    return client.post("/upload", files={"file": (name, io.BytesIO(text.encode()), "text/plain")})


def test_upload_two_files_and_query_each(client):
    assert _upload(client, "alpha.txt", ALPHA).status_code == 200
    assert _upload(client, "beta.txt", BETA).status_code == 200
    assert client.get("/health").json()["documents"] == 2
    r = client.post("/query", json={"question": "vegetarian lentil curry"})
    assert r.status_code == 200 and "lentil" in r.json()["answer"].lower()



def test_upload_same_name_preserves_each_content_version(client):
    assert _upload(client, "report.txt", ALPHA).status_code == 200
    assert _upload(client, "report.txt", BETA).status_code == 200
    docs = client.get("/documents").json()
    assert len(docs) == 2
    assert {d["filename"] for d in docs} == {"report.txt"}
    assert len({d["source"] for d in docs}) == 2

def test_query_before_ingest_is_409(client):
    assert client.post("/query", json={"question": "hi"}).status_code == 409


def test_ingest_rejects_paths_outside_data_dir(client):
    r = client.post("/ingest", json={"path": "/etc/passwd"})
    assert r.status_code == 400


def test_ingest_inside_data_dir_ok(client):
    (CONFIG.data_dir / "a.txt").write_text(ALPHA)
    assert client.post("/ingest", json={"path": "a.txt"}).status_code == 200


def test_upload_rejects_unsupported_type(client):
    assert client.post("/upload", files={"file": ("x.exe", io.BytesIO(b"MZ"), "application/octet-stream")}).status_code == 415


def test_api_key_enforced(client, monkeypatch):
    monkeypatch.setenv("DOCMIND_API_KEY", "secret")
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200


def test_review_flow_over_api(client):
    CONFIG.review.audit_rate = 1.0            # every answer is routed to the queue
    _upload(client, "alpha.txt", ALPHA)
    q = client.post("/query", json={"question": "coolant temperature startup", "user": "ananya"}).json()
    assert q["review_status"] == "pending"
    pending = client.get("/review/pending").json()
    assert any(p["response_id"] == q["response_id"] for p in pending)
    r = client.post(f"/review/{q['response_id']}", json={"decision": "approved", "reviewer": "ananya"})
    assert r.status_code == 200
    assert client.post(f"/review/{q['response_id']}", json={"decision": "approved"}).status_code == 404
    assert client.post("/feedback", json={"response_id": q["response_id"], "helpful": True}).status_code == 200
    assert client.post("/feedback", json={"response_id": "nope", "helpful": True}).status_code == 404


def test_reviewer_pin_enforced(client):
    CONFIG.review.reviewer_pin = "1234"
    assert client.get("/review/pending").status_code == 401
    assert client.get("/review/pending", headers={"X-Reviewer-Pin": "1234"}).status_code == 200
    CONFIG.review.reviewer_pin = ""
