"""
ReAct agent loop: reasoning + acting with no framework dependency.

Control flow summary (the old version overwrote every successful answer with
the failure message because of a dangling `else`; this version has exactly one
place where the answer is decided, `_decide_answer`, and one place where the
response is assembled):

  loop (max N steps)
    LLM -> Action?  execute tool, append Observation, continue
        -> Final Answer? accept only if evidence was retrieved
        -> neither: nudge
  no clean final answer but evidence exists -> one synthesis call
  verify citations against retrieved evidence, strip invalid ones
  decide abstention, review routing, assemble AgentResponse
"""
import logging
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple

from docmind.agent.citations import verify_citations
from docmind.agent.prompts import NOT_FOUND_TOKEN, REACT_SYSTEM_PROMPT, build_user_message
from docmind.agent.tools import TOOL_NAMES, ToolBox
from docmind.config import CONFIG
from docmind.llm.base import BaseLLM
from docmind.models import AgentResponse, SearchResult
from docmind.retrieval.hybrid import HybridRetriever
from docmind.review.policy import review_reasons

logger = logging.getLogger(__name__)

ACTION_RE = re.compile(
    r"^\s*Action:\s*(" + "|".join(TOOL_NAMES) + r")\s*\((.*)\)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL | re.IGNORECASE)
ABSTAIN_RE = re.compile(rf"^\s*{NOT_FOUND_TOKEN}\b[:\-\s]*", re.IGNORECASE)
SOFT_ABSTAIN_RE = re.compile(
    r"\b(?:could not|couldn't|cannot|can't|unable to|does not|doesn't|do not|don't)\b[^.\n]{0,60}"
    r"\b(?:find|contain|mention|provide|include|state|say|address|cover)\b",
    re.IGNORECASE,
)
LEAK_PREFIXES = ("Thought:", "Action:", "Observation:")
NO_ANSWER_MESSAGE = "I could not find this in the selected documents."
STOP = ["Observation:"]


def parse_action(text: str) -> Optional[Tuple[str, str]]:
    m = ACTION_RE.search(text)
    if not m:
        return None
    name = m.group(1).lower()
    arg = m.group(2).strip()
    arg = re.sub(r"^\w+\s*=\s*", "", arg)          # tolerate query="..." / chunk_id="..."
    arg = arg.strip().strip("\"'").strip()
    return (name, arg) if arg else None


def strip_leaks(text: str) -> str:
    kept = [ln for ln in text.split("\n") if not ln.strip().startswith(LEAK_PREFIXES)]
    return "\n".join(kept).strip()


class ReActAgent:
    def __init__(self, retriever: Optional[HybridRetriever] = None, llm: Optional[BaseLLM] = None,
                 max_iterations: Optional[int] = None):
        if llm is None:
            from docmind.llm.groq_client import get_llm
            llm = get_llm()
        self.llm = llm
        self.retriever = retriever or HybridRetriever()
        self.max_iterations = max_iterations or CONFIG.max_agent_iterations

    # ------------------------------------------------------------------ helpers
    def _complete(self, messages, timings, counter, stop=None) -> str:
        t = time.perf_counter()
        text = self.llm.complete(messages, stop=stop) if stop else self.llm.complete(messages)
        timings["llm_ms"] += (time.perf_counter() - t) * 1000
        counter[0] += 1
        return text or ""

    @staticmethod
    def _decide_answer(raw: str, has_valid_citation: bool) -> Tuple[str, bool]:
        """Return (final_text, abstained). Single source of truth for abstention."""
        text = strip_leaks(raw) or raw.strip()
        m = ABSTAIN_RE.match(text)
        if m:
            detail = text[m.end():].strip()
            return (f"{NO_ANSWER_MESSAGE} {detail}".strip(), True)
        if not has_valid_citation and SOFT_ABSTAIN_RE.search(text):
            return (text, True)
        return (text, False)

    # --------------------------------------------------------------------- run
    def run(self, question: str, file_ids: Optional[Iterable[str]] = None) -> AgentResponse:
        t0 = time.perf_counter()
        tools = ToolBox(self.retriever, file_ids=file_ids)
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question)},
        ]
        trace: List[str] = []
        sources: Dict[str, SearchResult] = {}
        timings = {"llm_ms": 0.0, "retrieval_ms": 0.0}
        calls = [0]
        searched = False
        raw_final: Optional[str] = None
        finish = "final_answer"

        for step in range(self.max_iterations):
            text = self._complete(messages, timings, calls, stop=STOP)
            trace.append(text)

            action = parse_action(text)
            if action:
                name, arg = action
                t = time.perf_counter()
                results, observation = tools.execute(name, arg)
                timings["retrieval_ms"] += (time.perf_counter() - t) * 1000
                searched = searched or name == "search_documents"
                for r in results:
                    sources.setdefault(r.chunk.chunk_id, r)
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}\n\nContinue. If you have enough evidence, give your Final Answer.",
                })
                continue

            final = FINAL_RE.search(text)
            if final:
                if sources:
                    raw_final = final.group(1).strip()
                    break
                if searched:
                    # Searched, nothing relevant came back: never trust an ungrounded answer.
                    raw_final = f"{NOT_FOUND_TOKEN}: the searches returned no relevant passages."
                    break
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user",
                                 "content": "You must call search_documents before a Final Answer. Search first."})
                continue

            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user",
                             "content": "Reply with either an Action (search_documents or get_chunk_context) or a Final Answer."})

        # Loop ended without a clean Final Answer.
        if raw_final is None:
            if sources:
                messages.append({"role": "user", "content": (
                    "Using only the Observations above, write your Final Answer now. "
                    "Start with 'Final Answer:' and copy citations exactly from cite_as.")})
                synth = self._complete(messages, timings, calls)       # no stop sequence
                trace.append(synth)
                m = FINAL_RE.search(synth)
                if m:
                    raw_final, finish = m.group(1).strip(), "synthesis"
                else:
                    raw_final, finish = (strip_leaks(synth) or synth), "fallback"
            else:
                raw_final, finish = f"{NOT_FOUND_TOKEN}: no relevant passages were retrieved.", "abstained"

        # Citation verification, then abstention decision (one place each).
        ordered = list(sources.values())
        cleaned, report, cited_ids = verify_citations(raw_final, ordered)
        answer, abstained = self._decide_answer(cleaned, has_valid_citation=bool(report.valid))
        if abstained:
            finish = "abstained"

        cited = [s for s in ordered if s.chunk.chunk_id in cited_ids]
        rest = [s for s in ordered if s.chunk.chunk_id not in cited_ids]
        latency_ms = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = latency_ms

        resp = AgentResponse(
            answer=answer,
            sources=cited + rest,
            reasoning_trace=trace,
            query=question,
            latency_ms=latency_ms,
            cited_sources=cited,
            citation_report=report.to_dict(),
            abstained=abstained,
            finish_reason=finish,
            timings=timings,
            llm_calls=calls[0],
        )
        if CONFIG.review.enabled:
            resp.review_reasons = review_reasons(resp)
            resp.needs_review = bool(resp.review_reasons)
        logger.info(f"ReAct done in {latency_ms:.0f}ms, {calls[0]} LLM calls, finish={finish}, "
                    f"abstained={abstained}, review={resp.review_reasons}")
        return resp
