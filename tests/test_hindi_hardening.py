"""Regression cases from the evidence and service review."""
import asyncio
import json
from types import SimpleNamespace
import unicodedata
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import JSONResponse

from docmind.attribution import attribute, evidence_spans
from docmind.config import CONFIG
from docmind.models import AgentResponse, Chunk, Document, SearchResult
from docmind.retrieval.sparse import tokenize
from docmind.text import sentence_spans, search_normalize


@pytest.mark.parametrize("text",[
    "Operators must keep the loop temperature\nbelow 12.5 degrees during startup.",
    "Set section 2.3.1 pressure to 12.5 bar.",
    "See Fig. 2 for the 12.5 bar limit.",
    "A wrapped line\nwithout punctuation",
])
def test_evidence_retains_wrapped_and_numeric_claims(text):
    spans=list(evidence_spans(text))
    assert any(start==0 and end==len(text) and quote==text for start,end,quote in spans)
    assert all(text[start:end]==quote for start,end,quote in spans)


@pytest.mark.parametrize("text",[
    "First sentence. Second sentence! Last fragment",
    "यह पहला वाक्य है। यह दूसरा वाक्य है।",
    "第一句。第二句！",
    "Unpunctuated heading\n\nBody sentence.\nTail",
])
def test_sentence_boundaries_cover_all_nonwhitespace(text):
    spans=sentence_spans(text)
    assert "".join(c for span in spans for c in span.text if not c.isspace()) == "".join(c for c in text if not c.isspace())
    assert all(text[s.start:s.end]==s.text for s in spans)


def test_hindi_danda_and_combining_marks():
    assert len(sentence_spans("यह पहला वाक्य है। यह दूसरा वाक्य है।"))==2
    assert tokenize("कंप्यूटर प्रणाली")==["कंप्यूटर","प्रणाली"]
    assert tokenize("क़ीमत १२.५")==tokenize("क़ीमत 12.5")
    assert search_normalize("क्‍ष")==search_normalize("क्ष")
    assert tokenize("café naïve")==["cafe","naive"]
    assert any(t.startswith("~g:") for t in tokenize("प्रणाली",hindi_ngrams=True))


def test_post_sentence_citation_attaches_without_placeholder_collision():
    source=SearchResult(Chunk("The limit is 300 degrees.","d","f.txt",0),1,"dense")
    for answer in ("The limit is 300 degrees. (Source: f.txt)","CITEREF7TOKEN is literal (Source: f.txt)."):
        report=attribute(answer,[source],lambda pairs:[.99]*len(pairs))
        assert report["n_claims"]==1 and report["supported_fraction"]==1


def test_multiturn_observations_always_include_file_identity():
    from docmind.agent.prompts import format_observation
    from docmind.agent.citations import verify_citations
    sources=[SearchResult(Chunk("Evidence.","d","same.txt",0,file_id=value),1,"dense") for value in ("aaa","bbb")]
    for source in sources:
        assert source.chunk.citation(include_id=True) in format_observation([source])
    _,report,ids=verify_citations("Fact "+sources[1].chunk.citation(include_id=True),sources)
    assert report.validity==1 and ids=={sources[1].chunk.chunk_id}


def test_numeric_tail_is_not_discarded():
    from docmind.ingestion.chunker import _build_chunks
    from dataclasses import replace
    cfg=replace(CONFIG.chunker,min_chunk_size=150,max_chunk_size=1200)
    text="A"*1200+" 300"
    chunks=_build_chunks(["A"*1200,"300"],[0],Document(text,"x"),config=cfg)
    assert any("300" in c.content for c in chunks)


def test_text_controls_and_explicit_legacy_encoding(tmp_path):
    from docmind.ingestion.loader import load_text
    path=tmp_path/"a.txt";path.write_bytes(b"First\fSecond \x1b[31mred\x1b[0m")
    unit=load_text(path)[0]
    assert unit.content=="First\nSecond red" and unit.metadata["normalization_edits"]
    path.write_bytes("Price €100".encode("cp1252"))
    with pytest.raises(ValueError):load_text(path)
    assert "€" in load_text(path,encoding="cp1252")[0].content
    path.write_bytes(b"bad\x00binary")
    with pytest.raises(ValueError):load_text(path)


def test_parser_environment_is_allowlisted(monkeypatch):
    from docmind.ingestion.loader import parser_environment
    for name in ("DOCMIND_ADMIN_PIN","DOCMIND_SECRETS_FILE","GROQ_API_KEY","UNRELATED_SECRET"):
        monkeypatch.setenv(name,"private-placeholder")
    env=parser_environment()
    assert not any(name in env for name in ("DOCMIND_ADMIN_PIN","DOCMIND_SECRETS_FILE","GROQ_API_KEY","UNRELATED_SECRET"))


def test_legacy_font_is_quarantined_and_script_mismatch_is_uncertain():
    from docmind.ingestion.quality import extraction_quality
    assert extraction_quality("legible-looking legacy bytes",font_names=["KrutiDev010"])["quarantined"]
    report=extraction_quality("This entire page has only Latin characters and technical values.",expected_language="hi")
    assert report["status"]=="uncertain_or_mixed" and report["flags"]


def test_possible_hinglish_is_unmeasured():
    from docmind.language import language_info
    result=language_info("Is machine ka temperature kya hai aur kitna hona chahiye?")
    assert result["language"]=="und" and "possible_romanized_hindi" in result["flags"]


def test_bogus_keys_do_not_allocate_quota_and_rejections_are_traced(monkeypatch):
    from docmind.observability import RequestMonitor
    monkeypatch.setenv("DOCMIND_API_KEY","valid-placeholder")
    async def exercise():
        monitor=RequestMonitor()
        async def downstream(request):return JSONResponse({"ok":True})
        for i in range(1000):
            request=Request({"type":"http","method":"GET","path":"/documents","headers":[(b"x-api-key",f"bad-{i}".encode())],"query_string":b"","client":("127.0.0.1",1)})
            response=await monitor(request,downstream)
            assert response.status_code==401 and "x-request-id" in response.headers
        assert not monitor.buckets and not monitor.ip_buckets
        request=Request({"type":"http","method":"GET","path":"/documents","headers":[(b"x-api-key",b"valid-placeholder")],"query_string":b"","client":("127.0.0.1",1)})
        assert (await monitor(request,downstream)).status_code==200
    asyncio.run(exercise())


def _client(monkeypatch):
    monkeypatch.setenv("DOCMIND_API_KEY","test-key")
    import docmind.api.main as api
    return api,TestClient(api.app,headers={"X-API-Key":"test-key"})


def test_pending_payload_and_final_response_lifecycle(monkeypatch):
    api,client=_client(monkeypatch)
    resp=AgentResponse("Unapproved 900 degrees",[],[],"Limit?",1,needs_review=True,review_reasons=["unsupported_claim"],citation_report={"attribution":{"claims":[{"claim":"Unapproved 900 degrees"}]}})
    monkeypatch.setattr(api,"get_agent",lambda:SimpleNamespace(run=lambda *a,**k:resp,retriever=SimpleNamespace(is_empty=lambda:False)))
    data=client.post("/query",json={"question":"Limit?"}).json()
    assert data["attribution"] is None and not data["sources"] and "900" not in data["answer"]
    rid=data["response_id"]
    assert client.get("/responses/"+rid).json()["attribution"] is None
    CONFIG.review.reviewer_pin="review-pin"
    assert client.post("/review/"+rid,json={"decision":"edited","edited_answer":"300 degrees"},headers={"X-Reviewer-Pin":"review-pin"}).status_code==200
    final=client.get("/responses/"+rid).json()
    assert final["answer"]=="300 degrees" and final["attribution"] is None
    assert final["attribution_status"]=="human_edited_not_rechecked"


def test_rejected_response_never_releases_answer(monkeypatch):
    api,client=_client(monkeypatch)
    resp=AgentResponse("Rejected draft",[],[],"q",1,needs_review=True)
    api.get_reviews().record(resp,user="shared-api")
    api.get_reviews().decide(resp.response_id,"rejected","reviewer")
    data=client.get("/responses/"+resp.response_id).json()
    assert data["review_status"]=="rejected" and "Rejected draft" not in json.dumps(data)


def test_failed_upload_is_removed(monkeypatch):
    api,client=_client(monkeypatch)
    monkeypatch.setattr(api,"ingest",lambda *a,**k:(_ for _ in ()).throw(ValueError("Corrupt PDF")))
    assert client.post("/upload",files={"file":("bad.pdf",b"%PDF-1.4 broken","application/pdf")}).status_code==422
    assert not list((CONFIG.data_dir/"uploads").rglob("*.pdf"))


def test_review_export_keeps_correction_answerable_and_verified(tmp_path):
    from docmind.review.store import ReviewStore
    from docmind.evaluation.retrieval_eval import load_testset
    store=ReviewStore()
    resp=AgentResponse("NOT_FOUND",[],[],"What city?",1,abstained=True,needs_review=True,review_reasons=["abstained"])
    store.record(resp);store.decide(resp.response_id,"edited","reviewer","Paris")
    path=tmp_path/"reviewed.json";store.export_testset(str(path))
    rows=load_testset(str(path),verified_only=True)
    assert len(rows)==1 and rows[0]["answerable"] and rows[0]["references"]==["Paris"]


def test_runtime_environment_changes_checkpoint_identity(tmp_path,monkeypatch):
    from docmind.evaluation.provenance import fingerprint
    path=tmp_path/"q.json";path.write_text("[]")
    monkeypatch.setenv("DOCMIND_TOOL_PROTOCOL","react");first=fingerprint(path)["run_id"]
    monkeypatch.setenv("DOCMIND_TOOL_PROTOCOL","native");assert fingerprint(path)["run_id"]!=first


def test_admin_pin_is_independent_of_reviewer(monkeypatch):
    api,client=_client(monkeypatch)
    CONFIG.review.reviewer_pin="reviewer-pin";monkeypatch.setenv("DOCMIND_ADMIN_PIN","admin-pin")
    assert client.delete("/documents/unknown",headers={"X-Reviewer-Pin":"reviewer-pin"}).status_code==401
    assert client.delete("/documents/unknown",headers={"X-Admin-Pin":"admin-pin"}).status_code==404


def test_hinglish_query_cannot_bypass_language_gate_via_english_sources(monkeypatch):
    from docmind.review.policy import review_reasons
    from docmind.models import Chunk,SearchResult
    monkeypatch.setenv("DOCMIND_TEST_MODE","0")
    monkeypatch.setattr(CONFIG.review,"audit_rate",0)
    source=SearchResult(Chunk("The limit is 300 degrees.","d","s",0,metadata={"language_info":{"language":"en"}}),1,"dense")
    response=AgentResponse("300 degrees",[source],[],"iska limit kya hai",1,citation_report={"cited":["c"],"attribution":{"supported_fraction":1.0}})
    assert "query_language_unvalidated" in review_reasons(response)


def test_scoring_normalizes_devanagari_digits_without_collapsing_decimals():
    from docmind.evaluation.metrics import answer_em
    assert answer_em("१२.५ किलोग्राम","12.5 किलोग्राम")==1
    assert answer_em("12.5","125")==0
    assert answer_em("क़दम","क़दम")==1


def test_default_attribution_cache_tracks_model_revision(monkeypatch):
    import docmind.attribution as module
    calls=[]
    class Judge:
        def __init__(self,**kwargs):self.identity=kwargs;calls.append(kwargs)
    monkeypatch.setattr(module,"NLIJudge",Judge)
    monkeypatch.setattr(module,"_JUDGE",None);monkeypatch.setattr(module,"_JUDGE_KEY",None)
    monkeypatch.setenv("DOCMIND_NLI_MODEL","first");monkeypatch.setenv("DOCMIND_NLI_REVISION","one")
    first=module.default_judge();assert module.default_judge() is first
    monkeypatch.setenv("DOCMIND_NLI_REVISION","two")
    assert module.default_judge() is not first and len(calls)==2


def test_orphan_sweeps_progress_past_referenced_files():
    from docmind.uploads import sweep
    from docmind.db import connect
    from docmind.store import CorpusStore
    CorpusStore()
    root=CONFIG.data_dir/"uploads"
    reference=root/("a"*64)/"one.txt";reference.parent.mkdir(parents=True)
    reference.write_text("referenced");(reference.parent/"two.txt").write_text("same referenced bytes")
    orphan=root/("b"*64)/"orphan.txt";orphan.parent.mkdir(parents=True);orphan.write_text("orphan")
    with connect() as conn:
        conn.execute("INSERT INTO documents(file_id,filename,source,sha256,n_units,n_chunks,ingested_at) VALUES (?,?,?,?,1,0,0)",("ref","one.txt",str(reference.resolve()),"a"*64))
    removed=sum(sweep(min_age_seconds=0,limit=1) for _ in range(6))
    assert removed==1 and not orphan.exists() and reference.exists()


def test_numeric_watermark_page_is_quarantined_without_discarding_values():
    from docmind.ingestion.quality import extraction_quality
    watermark = ("12345678901234567\n" * 40) + "Reprint 2025-26"
    report = extraction_quality(watermark, expected_language="hi")
    assert report["status"] == "uncertain_or_mixed"
    assert "repeated_numeric_artifacts" in report["flags"]
    assert report["quarantined"]
    for text in ("12345678901234567", "12.5 300 2025 2026", "१२.५ ३०० २०२५"):
        assert not extraction_quality(text)["quarantined"]
    assert not extraction_quality("\n".join(str(1000000000000000+i) for i in range(40)))["quarantined"]
