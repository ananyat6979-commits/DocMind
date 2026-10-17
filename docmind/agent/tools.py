"""
Agent tools. Both are backed by real retrieval; neither fabricates content.

  search_documents(query)      hybrid retrieval + rerank
  get_chunk_context(chunk_id)  the chunk plus its neighbours in document order
                               (accepts the 8-char id shown in observations)
"""
import logging
from typing import Dict, Iterable, List, Optional, Set, Tuple

from docmind.agent.prompts import format_observation
from docmind.models import SearchResult
from docmind.retrieval.hybrid import HybridRetriever
from docmind.store import CorpusStore

logger = logging.getLogger(__name__)

TOOL_NAMES = ("search_documents", "get_chunk_context")


class ToolBox:
    def __init__(self, retriever: HybridRetriever, store: Optional[CorpusStore] = None,
                 file_ids: Optional[Iterable[str]] = None, context_window: int = 1):
        self.retriever = retriever
        self.store = store or retriever.store
        self.file_ids: Optional[Set[str]] = set(file_ids) if file_ids else None
        self.context_window = context_window
        self.seen: Set[str] = set()   # chunk ids already shown to the model

    def search_documents(self, query: str) -> Tuple[List[SearchResult], str]:
        results = self.retriever.search(query, file_ids=self.file_ids)
        obs = format_observation(results, seen=self.seen)
        self.seen.update(r.chunk.chunk_id for r in results)
        return results, obs

    def get_chunk_context(self, chunk_id: str) -> Tuple[List[SearchResult], str]:
        chunk = self.store.get_chunk(chunk_id.strip().strip("'\""))
        if chunk is None or (self.file_ids and chunk.file_id not in self.file_ids):
            return [], f"Unknown chunk_id '{chunk_id}'. Use an id from a previous search_documents observation."
        neighbours = self.store.neighbors(chunk, window=self.context_window)
        results = [SearchResult(chunk=c, score=0.0, retriever="context") for c in neighbours]
        obs = format_observation(results, seen=self.seen, header="Surrounding context in document order:")
        self.seen.update(c.chunk_id for c in neighbours)
        return results, obs

    def execute(self, name: str, argument: str) -> Tuple[List[SearchResult], str]:
        if name == "search_documents":
            return self.search_documents(argument)
        if name == "get_chunk_context":
            return self.get_chunk_context(argument)
        return [], f"Unknown tool '{name}'. Available tools: {', '.join(TOOL_NAMES)}."
