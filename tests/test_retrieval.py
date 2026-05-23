"""Smoke tests for the retrieval pipeline."""
import pytest
from docmind.retrieval.hybrid import reciprocal_rank_fusion
from docmind.models import Chunk, SearchResult


def _make_result(chunk_id: str, score: float, retriever: str = "dense") -> SearchResult:
    chunk = Chunk(content="test", doc_id="d1", source="test.pdf", chunk_index=0)
    chunk.chunk_id = chunk_id
    return SearchResult(chunk=chunk, score=score, retriever=retriever)


def test_rrf_deduplication():
    """A chunk appearing in both dense and sparse results should be ranked higher."""
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

    # The chunk appearing in both lists should rank first
    assert fused[0].chunk.chunk_id == shared_id


def test_rrf_empty_lists():
    result = reciprocal_rank_fusion([[], []])
    assert result == []