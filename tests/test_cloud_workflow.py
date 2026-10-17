"""The cloud workflow: draft -> verify -> cloud_run -> bundle, plus the safety nets around it."""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from docmind.config import CONFIG
from docmind.evaluation.retrieval_eval import load_testset
from tests.conftest import ALPHA, BETA

ROOT = Path(__file__).resolve().parents[1]


def _env(home):
    e = dict(os.environ, DOCMIND_TEST_MODE="1", DOCMIND_HOME=str(home), PYTHONPATH=str(ROOT))
    e.pop("GROQ_API_KEY", None)
    return e


def test_verified_only_filters_and_errors_clearly(tmp_path):
    f = tmp_path / "t.json"
    f.write_text(json.dumps([{"question": "a", "verified": True}, {"question": "b"}]))
    assert [i["question"] for i in load_testset(str(f), verified_only=True)] == ["a"]
    assert len(load_testset(str(f))) == 2
    f.write_text(json.dumps([{"question": "b"}]))
    with pytest.raises(ValueError, match="none are verified"):
        load_testset(str(f), verified_only=True)


def test_verify_testset_cli_accepts_edits_and_deletes(tmp_path):
    drafts = [
        {"id": "q001", "question": "Q accept", "ground_truth": "A1", "answerable": True, "gold_substrings": ["span1"], "context_preview": "p", "verified": False},
        {"id": "q002", "question": "Q edit", "ground_truth": "wrong", "answerable": True, "gold_substrings": ["span2"], "context_preview": "p", "verified": False},
        {"id": "q003", "question": "Q junk", "ground_truth": "x", "answerable": True, "gold_substrings": ["span3"], "context_preview": "p", "verified": False},
        {"id": "q004", "question": "Q unanswerable", "ground_truth": "", "answerable": False, "gold_substrings": [], "context_preview": "p", "verified": False},
    ]
    src, out = tmp_path / "d.json", tmp_path / "v.json"
    src.write_text(json.dumps(drafts))
    keys = "a\ne\ncorrected\nd\ny\n"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/verify_testset.py"), "--in", str(src), "--out", str(out)],
                       input=keys, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    done = json.loads(out.read_text())
    assert [d["question"] for d in done] == ["Q accept", "Q edit", "Q unanswerable"]
    assert done[1]["ground_truth"] == "corrected" and all(d["verified"] for d in done)
    # resumable: a second run has nothing left to review
    r2 = subprocess.run([sys.executable, str(ROOT / "scripts/verify_testset.py"), "--in", str(src), "--out", str(out)],
                        input="", capture_output=True, text=True)
    assert "0 drafts to review" in r2.stdout


def test_cloud_run_end_to_end_and_bundle(tmp_path):
    doc = tmp_path / "doc.txt"; doc.write_text(ALPHA + " " + BETA)
    ts = tmp_path / "ts.json"
    ts.write_text(json.dumps([
        {"question": "coolant temperature startup", "answerable": True, "gold_substrings": ["below 300 degrees"], "verified": True},
        {"question": "vegetarian lentil", "answerable": True, "gold_substrings": ["lentil curry"], "verified": True},
        {"question": "unverified draft", "answerable": True, "gold_substrings": ["zzz"], "verified": False},
    ]))
    home, out = tmp_path / "home", tmp_path / "res.zip"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/cloud_run.py"), "--docs", str(doc), "--testset", str(ts),
                        "--skip-ragas", "--out", str(out)], env=_env(home), capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "2 verified items" in r.stderr                       # the unverified draft was excluded
    names = zipfile.ZipFile(out).namelist()
    assert "reports/retrieval_latest.md" in names and "indexes/docmind.db" in names


def test_cloud_run_requires_key_unless_skip_ragas(tmp_path):
    doc = tmp_path / "d.txt"; doc.write_text(ALPHA)
    ts = tmp_path / "t.json"
    ts.write_text(json.dumps([{"question": "q", "answerable": True, "gold_substrings": ["x"], "verified": True}]))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/cloud_run.py"), "--docs", str(doc), "--testset", str(ts)],
                       env=_env(tmp_path / "h"), capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 1 and "GROQ_API_KEY" in r.stderr


def test_db_from_cloud_is_usable_elsewhere(tmp_path, make_file):
    """Ingest in 'the cloud', copy only the .db to 'the laptop': search works with no re-embedding."""
    import shutil
    from docmind.ingestion.pipeline import ingest
    from docmind.retrieval.hybrid import HybridRetriever
    from docmind.store import CorpusStore
    ingest(make_file("alpha.txt", ALPHA), store=CorpusStore())
    laptop_db = tmp_path / "laptop" / "docmind.db"; laptop_db.parent.mkdir()
    shutil.copy(CONFIG.db_path, laptop_db)
    hits = HybridRetriever(CorpusStore(db_path=laptop_db)).search("coolant temperature startup")
    assert hits and hits[0].chunk.filename() == "alpha.txt"


def test_exhausted_daily_cap_stops_generation_and_keeps_checkpoint(tmp_path, scripted):
    from docmind.agent.react import ReActAgent
    from docmind.evaluation.harness import generate
    from docmind.ingestion.pipeline import ingest
    from docmind.llm.groq_client import RateLimitExhausted
    from docmind.retrieval.hybrid import HybridRetriever
    from docmind.store import CorpusStore

    store = CorpusStore(); ingest(tmp_path / "a.txt" if False else (lambda p: (p.write_text(ALPHA), p)[1])(tmp_path / "a.txt"), store=store)
    ts = tmp_path / "t.json"
    ts.write_text(json.dumps([{"id": f"q{i}", "question": "coolant", "answerable": True, "ground_truth": "x",
                               "gold_substrings": ["x"], "verified": True} for i in range(3)]))

    class Boom:
        n = 0
        def complete(self, messages, stop=None, **kw):
            Boom.n += 1
            if Boom.n <= 2:        # q0: search then answer
                return ['Action: search_documents("coolant")', "Final Answer: Below 300 degrees (Source: a.txt)."][Boom.n - 1]
            raise RateLimitExhausted("daily cap")

    out = tmp_path / "gen.jsonl"
    with pytest.raises(RateLimitExhausted):
        generate(str(ts), ReActAgent(retriever=HybridRetriever(store), llm=Boom()), str(out))
    assert len(out.read_text().strip().splitlines()) == 1        # q0 saved, q1 stopped the run


def test_app_password_gate(monkeypatch, make_file):
    pytest.importorskip("streamlit")
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("DOCMIND_APP_PASSWORD", "s3cret")
    st.cache_resource.clear()
    app = str(ROOT / "app.py")
    at = AppTest.from_file(app, default_timeout=30).run()
    assert not at.exception and at.text_input[0].label == "Password"
    assert not any(t.label == "Ask a question" for t in at.text_input)      # nothing behind the gate
    at.text_input[0].set_value("nope").run()
    assert any("Wrong password" in e.value for e in at.error)
    at.text_input[0].set_value("s3cret").run()
    assert not at.exception and any("Add a document" in i.value for i in at.info)
