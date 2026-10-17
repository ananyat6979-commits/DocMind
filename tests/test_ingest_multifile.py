"""Item 7: uploading a second file must NOT silently reuse the first file's index."""
from docmind.ingestion.pipeline import ingest
from docmind.retrieval.hybrid import HybridRetriever
from docmind.store import CorpusStore
from tests.conftest import ALPHA, BETA


def test_second_file_is_indexed_and_searchable(make_file):
    store = CorpusStore()
    a = make_file("alpha.txt", ALPHA)
    b = make_file("beta.txt", BETA)

    r1 = ingest(a, store=store)
    assert r1.files[0].status == "ingested"

    retriever = HybridRetriever(store)
    assert retriever.search("coolant temperature startup")        # first doc works

    r2 = ingest(b, store=store)                                    # the bug: this used to be skipped
    assert r2.files[0].status == "ingested"
    assert r2.chunks_added > 0

    hits = retriever.search("vegetarian lentil curry", mode="hybrid")
    assert hits, "second file must be retrievable without restarting anything"
    assert hits[0].chunk.filename() == "beta.txt"
    assert len(store.list_documents()) == 2


def test_same_bytes_is_cached_not_duplicated(make_file):
    store = CorpusStore()
    f = make_file("alpha.txt", ALPHA)
    ingest(f, store=store)
    n_before = len(store.load_corpus()[0])
    r = ingest(f, store=store)
    assert r.files[0].status == "cached"
    assert len(store.load_corpus()[0]) == n_before


def test_rebuild_replaces_only_that_file(make_file):
    store = CorpusStore()
    a, b = make_file("alpha.txt", ALPHA), make_file("beta.txt", BETA)
    ingest(a, store=store); ingest(b, store=store)
    r = ingest(a, rebuild_index=True, store=store)
    assert r.files[0].status == "reingested"
    chunks, _ = store.load_corpus()
    assert {c.filename() for c in chunks} == {"alpha.txt", "beta.txt"}
    assert len(store.list_documents()) == 2


def test_scoped_search_only_returns_selected_file(make_file):
    store = CorpusStore()
    ra = ingest(make_file("alpha.txt", ALPHA), store=store)
    ingest(make_file("beta.txt", BETA), store=store)
    retriever = HybridRetriever(store)
    for mode in ("dense", "sparse", "hybrid", "hybrid_rerank"):
        hits = retriever.search("soup menu coolant", file_ids=ra.file_ids, mode=mode)
        assert all(h.chunk.file_id == ra.file_ids[0] for h in hits), mode


def test_retriever_picks_up_changes_from_another_handle(make_file):
    """A long-lived retriever (API singleton) must see documents ingested later."""
    store = CorpusStore()
    retriever = HybridRetriever(store)
    assert retriever.is_empty()
    ingest(make_file("alpha.txt", ALPHA), store=CorpusStore())
    assert not retriever.is_empty()


def test_directory_ingest_reports_failures(tmp_path, make_file):
    d = tmp_path / "docs"; d.mkdir()
    (d / "ok.txt").write_text(ALPHA)
    (d / "empty.txt").write_text("   ")
    r = ingest(d, store=CorpusStore())
    assert len(r.file_ids) == 1
    assert len(r.failures) == 0 or r.failures[0].filename == "empty.txt"
