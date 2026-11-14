"""Individual identity and isolated parser transport boundaries."""
import hashlib
import json
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from docmind.config import CONFIG
from docmind.models import AgentResponse


def credentials(tmp_path,monkeypatch):
    path=tmp_path/"credentials.json"
    path.write_text(json.dumps([{"actor_id":name,"key_sha256":hashlib.sha256(token.encode()).hexdigest(),"roles":roles}
        for name,token,roles in [("alice","alice-test",["reader"]),("bob","bob-test",["reader","reviewer"]),
                                 ("admin","admin-test",["reader","admin","ingestor"])]]))
    monkeypatch.setenv("DOCMIND_CREDENTIALS_FILE",str(path))
    return path


def test_individual_principals_own_responses_and_review_identity(tmp_path,monkeypatch):
    credentials(tmp_path,monkeypatch)
    import docmind.api.main as api
    response=AgentResponse("Draft answer",[],[],"Q",1,needs_review=True)
    monkeypatch.setattr(api,"get_agent",lambda:SimpleNamespace(run=lambda *a,**kw:response,retriever=SimpleNamespace(is_empty=lambda:False)))
    client=TestClient(api.app)
    alice={"X-API-Key":"alice-test"};bob={"X-API-Key":"bob-test"}
    result=client.post("/query",json={"question":"Q","user":"spoofed-admin"},headers=alice)
    assert result.status_code==200
    rid=result.json()["response_id"]
    assert api.get_reviews().get(rid)["user"]=="alice"
    assert client.get("/responses/"+rid,headers=bob).status_code==404
    assert client.post("/review/"+rid,json={"decision":"approved"},headers=alice).status_code==403
    assert client.post("/review/"+rid,json={"decision":"approved","reviewer":"spoofed-admin"},headers=bob).status_code==200
    assert api.get_reviews().get(rid)["reviewer"]=="bob"
    assert client.post("/upload",files={"file":("a.txt",b"example")},headers=alice).status_code==403
    assert client.delete("/documents/no-such",headers=bob).status_code==403


def test_revocation_and_malformed_credentials_fail_closed(tmp_path,monkeypatch):
    from docmind.auth import authenticate
    path=credentials(tmp_path,monkeypatch)
    assert authenticate("alice-test")[0].actor_id=="alice"
    path.write_text("[]")
    assert authenticate("alice-test")== (None,503)
    path.write_text('{"unexpected":"shape"}')
    assert authenticate("alice-test")== (None,503)


def test_missing_parser_is_a_service_failure_not_native_fallback(tmp_path,monkeypatch):
    from docmind.ingestion.loader import load_document
    from docmind.ingestion.parser_client import ParserUnavailable
    monkeypatch.setenv("DOCMIND_PARSER_MODE","service")
    monkeypatch.setenv("DOCMIND_PARSER_SOCKET",str(tmp_path/"missing.sock"))
    path=tmp_path/"test.pdf";path.write_bytes(b"%PDF-1.4 missing parser")
    with pytest.raises(ParserUnavailable):load_document(path)


def test_parser_header_length_is_bounded():
    from docmind.ingestion.parser_service import handle
    import struct
    class Stream:
        def recv(self,size):return struct.pack("!I",5000)
    with pytest.raises(ValueError,match="header"):handle(Stream())


def test_text_encoding_is_explicit_at_api_boundary(monkeypatch):
    import docmind.api.main as api
    monkeypatch.setenv("DOCMIND_API_KEY","encoding-test")
    client=TestClient(api.app,headers={"X-API-Key":"encoding-test"})
    payload="Caf\u00e9 operators check the coolant limit before startup.".encode("cp1252")
    assert client.post("/upload",files={"file":("legacy.txt",payload)}).status_code==422
    assert client.post("/upload?encoding=cp1252",files={"file":("legacy.txt",payload)}).status_code==200
