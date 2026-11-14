"""Drives the real Streamlit script headlessly (offline demo LLM)."""
import pytest
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from docmind.config import CONFIG
from docmind.ingestion.pipeline import ingest
from docmind.store import CorpusStore
from tests.conftest import ALPHA, BETA


APP = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py")


def _app():
    import streamlit as st
    st.cache_resource.clear()
    return AppTest.from_file(APP, default_timeout=30)


def test_app_loads_empty_and_shows_hint():
    at = _app().run()
    assert not at.exception
    assert any("Add a document" in i.value for i in at.info)


def test_app_answers_for_both_files_and_queues_review(make_file):
    CONFIG.review.audit_rate = 1.0
    store = CorpusStore()
    ingest(make_file("alpha.txt", ALPHA), store=store)
    ingest(make_file("beta.txt", BETA), store=store)     # second file: the old bug
    at = _app().run()
    assert not at.exception
    at.text_input(key="user_name").set_value("tester")
    form_input = [t for t in at.text_input if t.label == "Ask a question"][0]
    form_input.set_value("vegetarian lentil curry")
    [b for b in at.button if b.label == "Ask"][0].click().run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    from docmind.review.store import ReviewStore
    assert "lentil" in ReviewStore().pending()[0]["answer"].lower()
    assert any("hidden until a reviewer decides" in info.value for info in at.info)
    assert ReviewStore().pending()[0]["answer"] not in md


def test_deleted_review_cannot_release_cached_answer(make_file):
    from docmind.review.store import ReviewStore
    from docmind.services import erase_document
    CONFIG.review.audit_rate=1.0
    store=CorpusStore()
    fid=ingest(make_file("alpha.txt",ALPHA),store=store).file_ids[0]
    ingest(make_file("beta.txt",BETA),store=store)
    at=_app().run()
    [t for t in at.text_input if t.label=="Ask a question"][0].set_value("coolant temperature startup")
    [b for b in at.button if b.label=="Ask"][0].click().run()
    draft=ReviewStore().pending()[0]["answer"]
    assert erase_document(fid,store)
    at.run()
    assert not at.exception
    assert not any(draft in markdown.value for markdown in at.markdown)
    assert any("no longer available" in info.value for info in at.info)
