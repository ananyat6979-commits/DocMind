"""Release safeguards: cohort sizing, incomplete samples and cache identities."""
import pytest
from docmind.evaluation.release import freeze_cohort,cohort_summary,blind_review_sample,reviewed_summary,template_pool


def test_timing_only_balanced_selection_costs_both_conditions():
    pool=[{"id":lang+str(i),"language":lang} for lang in ("en","hi") for i in range(200)]
    pilot=[{"id":"pilot-"+lang+str(i),"language":lang,"condition":condition,"elapsed_seconds":60}
           for lang in ("en","hi") for condition in ("baseline_capture","fresh") for i in range(3)]
    frozen=freeze_cohort(pool,pilot,consumed_seconds=1800,overhead_seconds=300)
    assert frozen["n_per_language"]==51
    changed=[dict(r,correct=False,answer="bad") for r in pilot]
    assert frozen["ids"]==freeze_cohort(pool,changed,1800,300)["ids"]
    assert freeze_cohort(pool,pilot,21600)["n_per_language"]==0
    with pytest.raises(ValueError): freeze_cohort(pool,pilot[:3],0)


def test_unserved_rows_stay_in_requested_denominator():
    result=cohort_summary({"ids":["one","two","three"]},[{"id":"one","correct":True},{"id":"two","error":"timeout"}])
    assert result["requested_cohort_correctness"]==1/3
    assert result["completed_only_correctness"]==1
    assert result["unserved_ids"]==["three"] and result["failed"]==1


def test_blinding_incomplete_review_and_release_floor():
    rows=[{"id":str(i),"question":"Q","answer":"A","evidence":"E","routing_stratum":"candidate_release","judge_score":.9} for i in range(30)]
    tasks,key=blind_review_sample(rows,{"candidate_release":25})
    assert all("routing_stratum" not in r and "judge_score" not in r for r in tasks)
    for task in tasks[:24]:task.update(finished=True,answer_correctness=True,duration_seconds=2)
    summary=reviewed_summary(tasks,key)
    assert summary["false_release_risk"]=="insufficient evidence"
    assert not summary["population_extrapolation_allowed"]
    tasks[24].update(finished=True,answer_correctness=False,duration_seconds=3)
    assert reviewed_summary(tasks,key)["false_release_risk"]==1/25


def test_synthetic_location_check_never_marks_human_gold():
    text="A meter means a unit of length. The limit is 12.5 kg."
    cache={"source":{"language":"en","url":"https://example.org/book.pdf"},"pdf_sha256":"abc",
        "pages":[{"page":1,"page_id":"p","text":text,"metadata":{"extraction_quality":{"quarantined":False}}}]}
    rows=template_pool([cache])
    assert rows and all(r["location_exact_match"] and not r["verified"] and not r["location_check_is_correctness"] for r in rows)
    assert any(r["proposed_answer"]=="12.5 kg" for r in rows)


def test_e5_prefixes_and_bge_query_are_separate(monkeypatch):
    from docmind import embeddings
    from docmind.config import CONFIG
    import numpy as np
    seen=[]
    class Encoder:
        def encode(self,texts,**kwargs):seen.extend(texts);return np.ones((len(texts),3))
    monkeypatch.setattr(embeddings,"get_embedding_model",lambda:Encoder())
    monkeypatch.setattr(CONFIG.embedding,"backend","sentence-transformers")
    monkeypatch.setattr(CONFIG.embedding,"model_name","intfloat/multilingual-e5-small")
    embeddings.embed_texts(["स्रोत"]);embeddings.embed_query("प्रश्न")
    assert seen==["passage: स्रोत","query: प्रश्न"]


def test_hindi_control_corruption_is_quarantined():
    from docmind.ingestion.quality import extraction_quality
    report=extraction_quality("यह क\x07म्प्यूटर प्रणाली है "*10,"hi")
    assert report["quarantined"] and "unexpected_control_characters" in report["flags"]


def test_corpus_identity_detects_equal_version_different_content(make_file,tmp_path):
    from docmind.ingestion.pipeline import ingest
    from docmind.store import CorpusStore
    first=CorpusStore(tmp_path/"first.db");second=CorpusStore(tmp_path/"second.db")
    ingest(make_file("one.txt","The coolant limit is 300 degrees."),store=first)
    ingest(make_file("two.txt","The pressure limit is 40 bars."),store=second)
    assert first.version()==second.version()
    assert first.content_identity()!=second.content_identity()


def test_checkpoint_rejects_rows_from_another_run(tmp_path):
    import json
    from types import SimpleNamespace
    from docmind.models import AgentResponse
    from docmind.evaluation.harness import generate
    question=tmp_path/"questions.json"
    question.write_text(json.dumps([{"id":"q","question":"limit?","verified":True,"ground_truth":"300"}]))
    agent=SimpleNamespace(run=lambda *a,**kw:AgentResponse("300",[],[],"limit?",1))
    output=tmp_path/"answers.jsonl"
    records=generate(str(question),agent,str(output))
    records[0]["run_id"]="another-run"
    output.write_text(json.dumps(records[0])+"\n")
    with pytest.raises(ValueError,match="Checkpoint records"):
        generate(str(question),agent,str(output))
