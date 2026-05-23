"""
Core data models that flow through the entire DocMind pipeline.
Using dataclasses gives us clean, typed structures without ORM overhead.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import uuid


@dataclass
class Document:
    """A raw document after loading, before chunking.
    One Document per page (PDF) or section (DOCX) for metadata granularity."""
    content: str
    source: str                                  # absolute file path
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Chunk:
    """A semantic unit derived from a Document.
    This is the atom of retrieval — what gets embedded and indexed."""
    content: str
    doc_id: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """A retrieved chunk with its relevance score and which retriever produced it."""
    chunk: Chunk
    score: float
    retriever: str  # "dense" | "sparse" | "hybrid" | "reranker"


@dataclass
class AgentResponse:
    """Final output from the ReAct agent."""
    answer: str
    sources: List[SearchResult]
    reasoning_trace: List[str]  # each Thought/Action/Observation step, for debugging
    query: str
    latency_ms: float
