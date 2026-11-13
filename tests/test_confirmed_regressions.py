import io
import json
import unicodedata
import zipfile
import pytest
from fastapi.testclient import TestClient
from docx import Document as Docx
from docmind.config import CONFIG
from docmind.models import Document, Chunk, SearchResult
from docmind.ingestion.chunker import _build_chunks, _find_breakpoints, _split_oversized
from docmind.ingestion.loader import load_docx, load_text
from docmind.agent.citations import verify_citations
from docmind.evaluation.metrics import ndcg_at_k, answer_level_stats
from docmind.evaluation.harness import generate
from docmind.evaluation.selective import calibrate, risk_curve
from docmind.attribution import attribute, attribution_metrics
from docmind.ingestion.pipeline import ingest
from docmind.store import CorpusStore
from tests.conftest import ALPHA


def test_tiny_forward_merge_does_not_duplicate_overlap(monkeypatch):
    monkeypatch.setattr(CONFIG.chunker, "min_chunk_size", 45)
    sentences = ["Alpha starts here.", "Beta follows it.", "Gamma describes the remaining operating instructions."]
    chunks = _build_chunks(sentences, [1], Document(" ".join(sentences), "x"))
    assert chunks[0].content.count("Beta follows it.") == 1


def test_repetitive_text_has_no_forced_semantic_breaks():
    assert _find_breakpoints(["Identical sentence."] * 400) == []


def test_cjk_and_long_tokens_respect_maximum():
    assert max(map(len, _split_oversized("你好" * 1000, 120))) <= 120


def test_merged_cells_read_once(tmp_path):
    doc = Docx()
    table = doc.add_table(rows=2, cols=3)
    table.cell(0,0).merge(table.cell(0,2)).text = "Quarterly Revenue Summary"
    table.cell(1,0).text = "Same"
    table.cell(1,1).text = "Same"
    path = tmp_path / "table.docx"; doc.save(path)
    text = " ".join(d.content for d in load_docx(path))
    assert text.count("Quarterly Revenue Summary") == 1
    assert text.count("Same") == 2


def test_docx_expansion_rejected_before_parser(tmp_path):
    path = tmp_path / "bomb.docx"
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "x" * 9000000)
    with pytest.raises(ValueError, match="safety limits"):
        load_docx(path)


def test_text_decoding_and_binary_rejection(tmp_path):
    path = tmp_path / "text.txt"
    path.write_bytes("Température".encode("utf-16"))
    assert load_text(path)[0].content == "Température"
    path.write_bytes(b"abc\x00def")
    with pytest.raises(ValueError, match="Binary"):
        load_text(path)
    path.write_bytes(b"Temp\xe9rature")
    with pytest.raises(ValueError, match="encoding"):
        load_text(path)


def source(name="report.pdf", file_id="one", text="The coolant limit is 300 degrees."):
    return SearchResult(Chunk(text,"d",name,0,metadata={"filename":name,"page":3},file_id=file_id),1.0,"dense")


def test_unicode_citation_and_same_name_ambiguity():
    src = source("café.pdf")
    answer = "Fact " + unicodedata.normalize("NFD", src.chunk.citation())
    assert verify_citations(answer,[src])[1].validity == 1
    a, report, ids = verify_citations("Fact (Source: report.pdf, p.3)", [source(),source(file_id="two")])
    assert report.validity == 0 and ids == set()


def test_shared_page_credits_all_candidates():
    src = [source(),source(text="Second supporting span.")]
    assert len(verify_citations("Fact (Source: report.pdf, p.3)",src)[2]) == 2


def test_ndcg_penalizes_missing_relevant_passages():
    assert ndcg_at_k([True,False],2,total_relevant=2) == pytest.approx(1/(1+1/1.584962500721156))


def test_failed_generations_remain_in_denominator_and_resume_is_config_keyed(tmp_path,monkeypatch):
    class Runner:
        def run(self,*args,**kwargs):
            raise RuntimeError("deliberate")
    dataset=tmp_path / "test.json"
    dataset.write_text(json.dumps([{"id":"q1","question":"What limit?","answerable":True,"verified":True}]))
    output=tmp_path / "gen.jsonl"
    rows=generate(str(dataset),agent=Runner(),out_path=str(output))
    assert len(rows)==1 and rows[0]["error"]=="RuntimeError"
    assert answer_level_stats(rows)["failure_rate"]==1
    monkeypatch.setattr(CONFIG.chunker,"max_chunk_size",900)
    with pytest.raises(ValueError,match="provenance mismatch"):
        generate(str(dataset),agent=Runner(),out_path=str(output))


def test_auth_fails_closed_without_secrets(monkeypatch):
    import docmind.api.main as api
    monkeypatch.delenv("DOCMIND_API_KEY",raising=False)
    client=TestClient(api.app)
    assert client.get("/documents").status_code==503
    assert client.post("/query",json={"question":"hello"}).status_code==503
    monkeypatch.setenv("DOCMIND_API_KEY","secret")
    assert client.get("/documents").status_code==401
    monkeypatch.setattr(CONFIG.review,"reviewer_pin","")
    assert client.get("/review/pending",headers={"X-API-Key":"secret"}).status_code==503


def test_real_attribution_can_disagree_with_label_existence():
    src=[source()]
    answer="The coolant limit is 900 degrees (Source: report.pdf, p.3)."
    assert verify_citations(answer,src)[1].validity==1
    report=attribute(answer,src,lambda pairs:[0.01]*len(pairs))
    assert report["supported_fraction"]==0
    claim=report["claims"][0]
    assert claim["quote"]==src[0].chunk.content[claim["start"]:claim["end"]]
    assert attribution_metrics([True,False],[True,True])["recall"]==0.5


def test_uncited_claim_is_counted_and_citations_do_not_split_pages():
    report=attribute("A fabricated claim. The limit is 300 degrees (Source: report.pdf, p.3).",[source()],lambda pairs:[1]*len(pairs))
    assert report["n_claims"]==2
    assert report["supported_fraction"]==0.5


def test_calibration_never_fits_on_test_and_reports_review_cost():
    assert calibrate([1,2,3],[False,False,False])["threshold"] is None
    curve=risk_curve([1,2],[True,False],review_seconds=120)
    assert curve[0]["coverage"]==0 and curve[0]["review_minutes"]==4
    assert curve[-1]["risk"]==0.5


def test_ambiguous_chunk_prefix_fails_closed(make_file):
    store=CorpusStore();ingest(make_file("a.txt",ALPHA),store=store)
    from docmind.db import connect
    with connect() as conn:
        conn.execute("UPDATE chunks SET chunk_id='deadbeef00000000' WHERE ord=0")
    assert store.get_chunk("deadbeef").chunk_id=="deadbeef00000000"
    with connect() as conn:
        row=conn.execute("SELECT * FROM chunks LIMIT 1").fetchone()
        conn.execute("INSERT INTO chunks SELECT 'deadbeef11111111',file_id,doc_id,source,99,99,content,metadata,embedding FROM chunks LIMIT 1")
    assert store.get_chunk("deadbeef") is None


def test_erasure_removes_owned_upload_and_review(tmp_path):
    from docmind.services import erase_document
    from docmind.review.store import ReviewStore
    from docmind.models import AgentResponse
    path=CONFIG.data_dir / "uploads" / "hash" / "a.txt"
    path.parent.mkdir(parents=True);path.write_text(ALPHA)
    store=CorpusStore();fid=ingest(path,store=store).file_ids[0]
    chunks,_=store.load_corpus()
    response=AgentResponse("answer",[SearchResult(chunks[0],1,"dense")],[],"question",1)
    reviews=ReviewStore();reviews.record(response)
    assert erase_document(fid,store)
    assert not path.exists() and not store.list_documents()
    assert reviews.get(response.response_id) is None


def test_ingestion_queue_is_durable_and_cancellable(make_file,tmp_path):
    from docmind.jobs import IngestionQueue
    path=make_file("queue.txt",ALPHA)
    queue=IngestionQueue(tmp_path / "jobs.db")
    job=queue.enqueue(path)
    assert IngestionQueue(tmp_path / "jobs.db").get(job)["status"]=="queued"
    assert queue.run_one()
    assert queue.get(job)["status"]=="completed"
    other=queue.enqueue(path)
    assert queue.cancel(other)
    assert queue.get(other)["status"]=="cancelled"


def test_hnsw_search_respects_file_filter(make_file,monkeypatch):
    from docmind.retrieval.hybrid import HybridRetriever
    monkeypatch.setenv("DOCMIND_DENSE_INDEX","hnsw")
    store=CorpusStore();fid=ingest(make_file("one.txt",ALPHA),store=store).file_ids[0]
    results=HybridRetriever(store).search("coolant",mode="dense",file_ids=[fid])
    assert results and all(r.chunk.file_id==fid for r in results)
