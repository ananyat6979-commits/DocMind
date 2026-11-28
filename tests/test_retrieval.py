"""Tests for retrieval pipeline."""
import pytest
from docmind.retrieval.hybrid import reciprocal_rank_fusion
from docmind.models import Chunk, SearchResult


def _make_result(chunk_id: str, score: float, retriever: str = "dense") -> SearchResult:
    chunk = Chunk(content="test content", doc_id="d1", source="test.pdf", chunk_index=0)
    chunk.chunk_id = chunk_id
    return SearchResult(chunk=chunk, score=score, retriever=retriever)


def test_rrf_deduplication():
    """Chunk appearing in both lists should rank first."""
    shared_id = "chunk_abc"
    dense = [
        _make_result(shared_id, 0.9, "dense"),
        _make_result("chunk_x", 0.8, "dense"),
    ]
    sparse = [
        _make_result("chunk_y", 0.7, "sparse"),
        _make_result(shared_id, 0.6, "sparse"),
    ]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    assert fused[0].chunk.chunk_id == shared_id


def test_rrf_empty_lists():
    result = reciprocal_rank_fusion([[], []])
    assert result == []


def test_rrf_single_list():
    results = [_make_result(f"chunk_{i}", 1.0 - i * 0.1) for i in range(3)]
    fused = reciprocal_rank_fusion([results], k=60)
    assert len(fused) == 3
    # Original ranking should be preserved
    assert fused[0].chunk.chunk_id == "chunk_0"


def test_rrf_scores_are_positive():
    results = [_make_result(f"chunk_{i}", float(i)) for i in range(5)]
    fused = reciprocal_rank_fusion([results], k=60)
    assert all(r.score > 0 for r in fused)


def test_rrf_k_parameter():
    """Higher k should compress score differences."""
    results = [_make_result(f"chunk_{i}", 1.0) for i in range(3)]
    fused_low_k  = reciprocal_rank_fusion([results], k=1)
    fused_high_k = reciprocal_rank_fusion([results], k=1000)
    scores_low  = [r.score for r in fused_low_k]
    scores_high = [r.score for r in fused_high_k]
    # Low k = larger differences between ranks
    assert (scores_low[0] - scores_low[-1]) > (scores_high[0] - scores_high[-1])

def test_default_dense_does_not_run_sparse_or_reranker(make_file,monkeypatch):
    from docmind.config import CONFIG,DocMindConfig
    from docmind.ingestion.pipeline import ingest
    from docmind.retrieval.hybrid import HybridRetriever
    from docmind.store import CorpusStore
    from tests.conftest import ALPHA
    cfg=DocMindConfig()
    assert cfg.embedding.model_name=="intfloat/multilingual-e5-small" and cfg.embedding.model_revision
    assert cfg.retriever.mode=="dense"
    store=CorpusStore();ingest(make_file("a.txt",ALPHA),store=store)
    retriever=HybridRetriever(store);retriever.refresh()
    def unused(*args,**kwargs):raise AssertionError("Default dense path invoked an optional scorer")
    monkeypatch.setattr(retriever.sparse,"search",unused)
    monkeypatch.setattr(retriever.reranker,"rerank",unused)
    hits=retriever.search("coolant temperature startup")
    assert hits and all(r.retriever=="dense" for r in hits)
    monkeypatch.setattr(CONFIG.retriever,"min_rerank_score",0.1)
    with pytest.raises(ValueError,match="clear stale thresholds"):
        retriever.search("coolant temperature startup")


def test_dense_default_skips_sparse_index_construction(make_file,monkeypatch):
    from docmind.ingestion.pipeline import ingest
    from docmind.retrieval.hybrid import HybridRetriever
    from docmind.retrieval.sparse import SparseRetriever
    from docmind.store import CorpusStore
    from tests.conftest import ALPHA
    store=CorpusStore();ingest(make_file("source.txt",ALPHA),store=store)
    def unused(*args,**kwargs):raise AssertionError("Unrequested sparse index was built")
    monkeypatch.setattr(SparseRetriever,"fit",unused)
    assert HybridRetriever(store).search("coolant",mode="dense")


def test_sparse_mode_skips_dense_build_and_remains_nonempty(make_file,monkeypatch):
    from docmind.config import CONFIG
    from docmind.ingestion.pipeline import ingest
    from docmind.retrieval.hybrid import HybridRetriever
    from docmind.retrieval.dense import DenseRetriever
    from docmind.store import CorpusStore
    from tests.conftest import ALPHA
    store=CorpusStore();ingest(make_file("source.txt",ALPHA),store=store)
    def unused(*args,**kwargs):raise AssertionError("Unrequested dense index was built")
    monkeypatch.setattr(DenseRetriever,"fit",unused)
    monkeypatch.setattr(CONFIG.retriever,"mode","sparse")
    retriever=HybridRetriever(store)
    assert retriever.search("coolant") and not retriever.is_empty()


def test_concurrent_explicit_modes_keep_their_required_snapshot(make_file):
    from concurrent.futures import ThreadPoolExecutor
    from docmind.ingestion.pipeline import ingest
    from docmind.retrieval.hybrid import HybridRetriever
    from docmind.store import CorpusStore
    from tests.conftest import ALPHA
    store=CorpusStore();ingest(make_file("source.txt",ALPHA),store=store)
    retriever=HybridRetriever(store)
    modes=["dense","sparse","hybrid"]*12
    def ask(mode):
        hits=retriever.search("coolant",mode=mode)
        assert hits and all(hit.retriever==mode for hit in hits)
        return len(hits)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(ask,modes))


def test_empty_file_selection_returns_no_chunks(make_file):
    from docmind.agent.tools import ToolBox
    from docmind.ingestion.pipeline import ingest
    from docmind.retrieval.hybrid import MODES, HybridRetriever
    from docmind.store import CorpusStore
    from tests.conftest import ALPHA, BETA

    store = CorpusStore()
    alpha_id, = ingest(make_file("alpha.txt", ALPHA), store=store).file_ids
    beta_id, = ingest(make_file("beta.txt", BETA), store=store).file_ids
    retriever = HybridRetriever(store)
    for mode in MODES:
        assert retriever.search("coolant temperature", mode=mode)
        scoped = retriever.search("coolant temperature", mode=mode, file_ids=[alpha_id])
        assert scoped and {item.chunk.file_id for item in scoped} == {alpha_id}
        assert retriever.search("coolant temperature", mode=mode, file_ids=[]) == []

    retriever.refresh(mode="hybrid")
    assert retriever.dense.search("coolant temperature", file_ids=[]) == []
    assert retriever.sparse.search("coolant temperature", file_ids=[]) == []
    alpha = retriever.search("coolant temperature", file_ids=[alpha_id])[0].chunk
    beta = retriever.search("lentil", file_ids=[beta_id])[0].chunk
    empty = ToolBox(retriever, file_ids=[])
    assert empty.search_documents("coolant temperature")[0] == []
    assert empty.get_chunk_context(alpha.chunk_id)[0] == []
    scoped_tools = ToolBox(retriever, file_ids=[alpha_id])
    assert scoped_tools.get_chunk_context(alpha.chunk_id)[0]
    assert scoped_tools.get_chunk_context(beta.chunk_id)[0] == []
