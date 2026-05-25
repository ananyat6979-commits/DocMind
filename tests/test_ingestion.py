"""Tests for ingestion pipeline."""
import pytest
from pathlib import Path
import tempfile

from docmind.ingestion.loader import load_text
from docmind.ingestion.chunker import chunk_document
from docmind.models import Document


def test_load_text_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is a test document. It has multiple sentences. Here is a third one.")
        tmp_path = Path(f.name)
    docs = load_text(tmp_path)
    assert len(docs) == 1
    assert "test document" in docs[0].content
    tmp_path.unlink()


def test_chunk_document_returns_chunks():
    doc = Document(
        content=(
            "The transformer architecture revolutionized natural language processing. "
            "Self-attention allows the model to weigh the importance of different words. "
            "BERT and GPT are both based on the transformer. "
            "These models are pre-trained on large corpora and fine-tuned on downstream tasks. "
            "Financial applications include sentiment analysis and document classification. "
            "The accuracy of these models depends heavily on the quality of training data."
        ),
        source="test",
        metadata={"filename": "test.txt"}
    )
    chunks = chunk_document(doc)
    assert len(chunks) >= 1
    assert all(c.content for c in chunks)
    assert all(c.doc_id == doc.doc_id for c in chunks)


def test_chunk_preserves_content():
    content = "Sentence one. Sentence two. Sentence three. " * 10
    doc = Document(content=content, source="test", metadata={})
    chunks = chunk_document(doc)
    recovered = " ".join(c.content for c in chunks)
    assert "Sentence one" in recovered


def test_document_has_required_fields():
    doc = Document(content="hello world", source="test.txt", metadata={})
    assert doc.doc_id is not None
    assert doc.content == "hello world"


def test_chunk_has_required_fields():
    doc = Document(content="First sentence. Second sentence. Third sentence.", source="test.txt", metadata={})
    chunks = chunk_document(doc)
    for chunk in chunks:
        assert chunk.chunk_id is not None
        assert chunk.doc_id == doc.doc_id
        assert chunk.source == "test.txt"
        assert len(chunk.content) > 0