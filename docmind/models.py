"""
Core data models that flow through the entire DocMind pipeline.
Dataclasses give clean, typed structures without ORM overhead.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import re
import uuid


@dataclass
class Document:
    """A raw document unit after loading, before chunking.
    One Document per page (PDF) or section (DOCX). NOTE: doc_id identifies the
    page/section unit, not the file. Use Chunk.file_id for the whole file."""
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Chunk:
    """A semantic unit derived from a Document. The atom of retrieval."""
    content: str
    doc_id: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = None
    file_id: str = ""     # stable id of the whole file (sha256 prefix)
    ord: int = 0          # position of this chunk within the whole file

    def filename(self) -> str:
        return str(self.metadata.get("filename") or Path(self.source).name)

    def citation(self) -> str:
        """The exact citation string the agent must copy verbatim.
        Verified later against what was actually retrieved."""
        fn = self.filename()
        if "page" in self.metadata:
            return f"(Source: {fn}, p.{self.metadata['page']})"
        if "section" in self.metadata:
            sec = re.sub(r"[;\n]", " ", str(self.metadata["section"])).strip()
            return f"(Source: {fn}, §{sec})"
        return f"(Source: {fn})"


@dataclass
class SearchResult:
    """A retrieved chunk with its relevance score and which stage produced it."""
    chunk: Chunk
    score: float
    retriever: str  # "dense" | "sparse" | "hybrid" | "reranker" | "context"


@dataclass
class AgentResponse:
    """Final output from the ReAct agent."""
    answer: str
    sources: List[SearchResult]
    reasoning_trace: List[str]
    query: str
    latency_ms: float
    # Added fields (all defaulted so older call sites keep working)
    cited_sources: List[SearchResult] = field(default_factory=list)
    citation_report: Dict[str, Any] = field(default_factory=dict)
    abstained: bool = False
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    finish_reason: str = "final_answer"   # final_answer | synthesis | fallback | abstained
    timings: Dict[str, float] = field(default_factory=dict)
    llm_calls: int = 0
    response_id: str = field(default_factory=lambda: uuid.uuid4().hex)
