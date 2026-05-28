"""
ReAct agent loop — Reasoning + Acting without any framework dependency.
"""
import re
import time
import logging
from typing import List, Optional

from docmind.models import AgentResponse, SearchResult
from docmind.agent.prompts import (
    REACT_SYSTEM_PROMPT,
    build_user_message,
    format_observation,
)
from docmind.llm.groq_client import get_llm
from docmind.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

ACTION_PATTERN = re.compile(
    r'Action:\s*search_documents\s*\(\s*["\']?([^"\')\r\n]+?)["\']?\s*\)',
    re.IGNORECASE
)
FINAL_ANSWER_PATTERN = re.compile(r'Final Answer:\s*(.*)', re.DOTALL | re.IGNORECASE)


class ReActAgent:
    def __init__(self, retriever: Optional[HybridRetriever] = None):
        self.llm = get_llm()
        self.retriever = retriever or HybridRetriever()
        self.max_iterations = 5

    def _execute_action(self, argument: str):
        results = self.retriever.search(argument)
        
        return results, format_observation(results)

    def run(self, question: str) -> AgentResponse:
        start_time = time.time()

        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question)},
        ]

        reasoning_trace: List[str] = []
        all_sources: List[SearchResult] = []
        final_answer = None
        iteration = 0

        for iteration in range(self.max_iterations):
            logger.debug(f"ReAct iteration {iteration + 1}/{self.max_iterations}")

            response_text = self.llm.complete(
                messages,
                stop=["\nObservation:", "Observation:"]
            )
            
            reasoning_trace.append(response_text)

            # Priority 1: check for action
            action_match = ACTION_PATTERN.search(response_text)
            if action_match:
                argument = action_match.group(1).strip().strip('"\'')
                
                sources, observation_text = self._execute_action(argument)
                all_sources.extend(sources)
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation_text}\n\nContinue your reasoning. If you have enough information, provide your Final Answer."
                })
                continue

            # Priority 2: check for final answer
            final_match = FINAL_ANSWER_PATTERN.search(response_text)
            if final_match:
                if all_sources or iteration >= self.max_iterations - 1:
                    final_answer = final_match.group(1).strip()
                    break
                else:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({
                        "role": "user",
                        "content": "You must call search_documents before providing a Final Answer. Search the document corpus first."
                    })
                    continue

            # Priority 3: neither action nor final answer — nudge
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": "Use the search_documents tool to find relevant information, then provide your Final Answer."
            })
        # If we have sources but exhausted iterations without Final Answer,
        # make one explicit synthesis call without a stop sequence
        if final_answer is None and all_sources:
            synthesis_prompt = (
                "Based on your research so far, provide your Final Answer now. "
                "You must begin your response with 'Final Answer:' and cite sources."
            )
            messages.append({"role": "user", "content": synthesis_prompt})
            synthesis_response = self.llm.complete(messages)  # no stop sequence
            reasoning_trace.append(synthesis_response)
            final_match = FINAL_ANSWER_PATTERN.search(synthesis_response)
            if final_match:
                final_answer = final_match.group(1).strip()
            else:
                # Strip reasoning lines and use whatever the model said
                final_answer = '\n'.join(
                    line for line in synthesis_response.split('\n')
                    if not line.strip().startswith(('Thought:', 'Action:', 'Observation:'))
                ).strip() or synthesis_response
        if final_answer is None:
            # Try to extract Final Answer from the last reasoning step
            if reasoning_trace:
                last = reasoning_trace[-1]
                last_match = FINAL_ANSWER_PATTERN.search(last)
                if last_match:
                    final_answer = last_match.group(1).strip()
            elif all_sources:
                # We have sources but no clean Final Answer — synthesize from last thought
                # Strip out Action: lines, keep only Thought content
                thought_lines = [
                    line for line in last.split('\n')
                    if not line.strip().startswith('Action:')
                    and not line.strip().startswith('Thought:')
                    and line.strip()
                ]
                final_answer = ' '.join(thought_lines).strip() or last
            else:
                final_answer = "I was unable to find a satisfactory answer in the document corpus."
        else:
            final_answer = "I was unable to find a satisfactory answer in the document corpus."

        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"ReAct completed in {latency_ms:.0f}ms, {iteration + 1} iterations")

        seen = set()
        unique_sources = []
        for s in all_sources:
            if s.chunk.chunk_id not in seen:
                seen.add(s.chunk.chunk_id)
                unique_sources.append(s)

        return AgentResponse(
            answer=final_answer,
            sources=unique_sources,
            reasoning_trace=reasoning_trace,
            query=question,
            latency_ms=latency_ms,
        )