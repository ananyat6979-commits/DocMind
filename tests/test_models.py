"""Tests for core data models."""
from docmind.models import Document, Chunk, SearchResult, AgentResponse


def test_document_auto_id():
    d1 = Document(content="a", source="x", metadata={})
    d2 = Document(content="b", source="y", metadata={})
    assert d1.doc_id != d2.doc_id


def test_chunk_auto_id():
    c1 = Chunk(content="a", doc_id="d1", source="x", chunk_index=0)
    c2 = Chunk(content="b", doc_id="d1", source="x", chunk_index=1)
    assert c1.chunk_id != c2.chunk_id


def test_agent_response_fields():
    chunk = Chunk(content="test", doc_id="d1", source="test.pdf", chunk_index=0)
    source = SearchResult(chunk=chunk, score=0.9, retriever="dense")
    resp = AgentResponse(
        answer="The answer is 42.",
        sources=[source],
        reasoning_trace=["Thought: search", "Final Answer: 42"],
        query="What is the answer?",
        latency_ms=1234.5,
    )
    assert resp.answer == "The answer is 42."
    assert len(resp.sources) == 1
    assert resp.latency_ms == 1234.5