import importlib
import pytest


def test_env_switches_are_honoured(monkeypatch):
    import docmind.config as c
    monkeypatch.setenv("DOCMIND_USE_RERANKER", "0")
    monkeypatch.setenv("DOCMIND_MAX_ITERATIONS", "3")
    cfg = c.DocMindConfig()
    assert cfg.retriever.use_reranker is False and cfg.max_agent_iterations == 3
    monkeypatch.delenv("DOCMIND_USE_RERANKER"); monkeypatch.delenv("DOCMIND_MAX_ITERATIONS")
    cfg = c.DocMindConfig()
    assert cfg.retriever.use_reranker is True and cfg.max_agent_iterations == 5


def test_ablation_refuses_to_run_without_reranker(make_file):
    from docmind.config import CONFIG
    from docmind.evaluation.retrieval_eval import evaluate_modes
    from docmind.ingestion.pipeline import ingest
    from docmind.retrieval.hybrid import HybridRetriever
    from docmind.store import CorpusStore
    from tests.conftest import ALPHA
    store = CorpusStore(); ingest(make_file("a.txt", ALPHA), store=store)
    CONFIG.retriever.use_reranker = False
    try:
        r = HybridRetriever(store)
        assert r.reranker is None
        with pytest.raises(ValueError, match="reranker is disabled"):
            evaluate_modes([{"question": "q", "answerable": True, "gold_substrings": ["x"]}], r)
    finally:
        CONFIG.retriever.use_reranker = True


def test_search_still_works_without_reranker(make_file):
    from docmind.config import CONFIG
    from docmind.ingestion.pipeline import ingest
    from docmind.retrieval.hybrid import HybridRetriever
    from docmind.store import CorpusStore
    from tests.conftest import ALPHA
    store = CorpusStore(); ingest(make_file("a.txt", ALPHA), store=store)
    CONFIG.retriever.use_reranker = False
    try:
        hits = HybridRetriever(store).search("coolant temperature startup")
        assert hits and hits[0].retriever == "hybrid"
    finally:
        CONFIG.retriever.use_reranker = True
