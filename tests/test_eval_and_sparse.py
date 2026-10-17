import pytest
from docx import Document as DocxDoc

from docmind.evaluation.metrics import (
    bootstrap_ci, best_f1_threshold, is_relevant, mrr, ndcg_at_k, paired_relative_improvement, precision_at_k,
)
from docmind.evaluation.retrieval_eval import compare_chunkers, evaluate_modes, to_markdown
from docmind.ingestion.loader import load_docx, load_text, EmptyDocumentError
from docmind.ingestion.pipeline import ingest
from docmind.models import Document
from docmind.ingestion.chunker import chunk_document
from docmind.retrieval.hybrid import HybridRetriever
from docmind.retrieval.sparse import BM25Index, SparseRetriever, tokenize
from docmind.store import CorpusStore
from tests.conftest import ALPHA, BETA


def test_tokenizer_strips_punctuation():
    assert tokenize("Coolant, pump; (valve).") == ["coolant", "pump", "valve"]


def test_bm25_ranks_exact_match_first_and_works_on_tiny_corpus():
    idx = BM25Index(); idx.fit([tokenize("one single chunk about coolant")])
    assert idx.get_scores(["coolant"])[0] > 0       # rank-bm25 returned <= 0 here
    idx.fit([tokenize("soup menu"), tokenize("coolant pump valve"), tokenize("coolant")])
    s = idx.get_scores(["valve"])
    assert s.argmax() == 1 and s[0] == 0


def test_sparse_retriever_empty_query_and_no_match(make_file):
    store = CorpusStore(); ingest(make_file("a.txt", ALPHA), store=store)
    r = HybridRetriever(store)
    assert r.sparse.search("the of and") == []
    assert r.sparse.search("zzzqqq") == []


def test_metrics_math():
    assert precision_at_k([True, False, True], 5) == pytest.approx(0.4)
    assert mrr([False, True], 5) == 0.5
    assert ndcg_at_k([True, True, False], 3) == pytest.approx(1.0)
    assert is_relevant("The Coolant  must\nstay low", ["coolant must stay"])


def test_paired_improvement_detects_real_gain():
    base = [0.2] * 30 + [0.4] * 30
    treat = [x * 1.5 for x in base]
    r = paired_relative_improvement(base, treat)
    assert r["relative"] == pytest.approx(0.5) and r["lo"] > 0.3 and r["hi"] < 0.7


def test_bootstrap_ci_contains_mean():
    m, lo, hi = bootstrap_ci([0, 1, 1, 0, 1, 1, 1, 0])
    assert lo <= m <= hi


def test_threshold_calibration_separates_classes():
    r = best_f1_threshold([5, 4, 3, -2, -3, -4], [True, True, True, False, False, False])
    assert r["f1"] == 1.0 and -2 < r["threshold"] <= 3


def _labeled(make_file):
    store = CorpusStore()
    ingest(make_file("alpha.txt", ALPHA), store=store); ingest(make_file("beta.txt", BETA), store=store)
    testset = [
        {"id": "1", "question": "coolant temperature during startup", "answerable": True, "gold_substrings": ["below 300 degrees"]},
        {"id": "2", "question": "vegetarian lentil curry", "answerable": True, "gold_substrings": ["lentil curry"]},
        {"id": "3", "question": "shutdown valve closes", "answerable": True, "gold_substrings": ["within two seconds"]},
    ]
    return store, testset


def test_evaluate_modes_runs_all_modes(make_file):
    store, ts = _labeled(make_file)
    res = evaluate_modes(ts, HybridRetriever(store), k=3)
    assert set(res["modes"]) == {"dense", "sparse", "hybrid", "hybrid_rerank"}
    assert res["modes"]["hybrid"]["hit"]["mean"] == 1.0
    assert "rerank_vs_dense_precision" in res and "Precision@3" in to_markdown(res)


def test_compare_chunkers_uses_throwaway_db(make_file):
    store, ts = _labeled(make_file)
    docs = [make_file("alpha2.txt", ALPHA), make_file("beta2.txt", BETA)]
    res = compare_chunkers(ts, docs, k=3)
    assert set(res) == {"fixed", "semantic"}
    assert len(store.list_documents()) == 2          # real db untouched


def test_docx_tables_and_headings_are_loaded(tmp_path):
    d = DocxDoc(); d.add_heading("Safety", 1); d.add_paragraph("Wear goggles.")
    t = d.add_table(rows=1, cols=2); t.rows[0].cells[0].text = "Limit"; t.rows[0].cells[1].text = "300 degrees"
    p = tmp_path / "m.docx"; d.save(str(p))
    docs = load_docx(p)
    assert docs[0].metadata["section"] == "Safety"
    assert "Limit | 300 degrees" in docs[0].content


def test_empty_text_file_yields_no_docs_and_ingest_explains(tmp_path):
    p = tmp_path / "e.txt"; p.write_text("   ")
    assert load_text(p) == []
    with pytest.raises(EmptyDocumentError):
        ingest(p, store=CorpusStore())


def test_chunker_does_not_emit_sub_minimum_fragment_before_a_big_group():
    # Regression: a tiny leading group used to be emitted alone, below min_chunk_size.
    from docmind.config import CONFIG
    text = "Hi there. " + ("This sentence is reasonably long and talks about coolant pumps. " * 8)
    chunks = chunk_document(Document(content=text, source="t", metadata={}))
    assert all(len(c.content) >= CONFIG.chunker.min_chunk_size for c in chunks[:-1])


def test_giant_single_sentence_is_split():
    from docmind.config import CONFIG
    chunks = chunk_document(Document(content="word " * 1000, source="t", metadata={}))
    assert all(len(c.content) <= CONFIG.chunker.max_chunk_size for c in chunks) and len(chunks) > 1


def test_trailing_page_number_is_not_a_chunk():
    # Regression: real PDFs end each page with a footer like "14"; old code made it a 1-char chunk.
    body = "Coolant pumps are inspected every shift by trained operators on duty. " * 6
    chunks = chunk_document(Document(content=body + "14", source="t", metadata={}))
    assert all(len(c.content) >= 20 for c in chunks)
    assert chunks[-1].content.endswith("14")           # attached to the previous chunk, not lost


def test_pure_noise_tail_is_dropped():
    chunks = chunk_document(Document(content="Coolant pumps are inspected every shift by trained operators. " * 5 + "\n\n14", source="t", metadata={}))
    assert all(len(c.content) >= 20 for c in chunks)
