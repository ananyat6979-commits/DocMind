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
    assert "lentil" in md.lower()
    assert any("Awaiting human review" in w.value for w in at.warning)
