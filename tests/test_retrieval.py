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