"""
ReAct agent loop.

This implements the Reasoning + Acting pattern without any framework dependency.

The loop:
  1. Send current conversation to the LLM.
  2. Parse the response for Action: calls.
  3. Execute the tool.
  4. Inject the Observation back into the conversation.
  5. Repeat until "Final Answer:" appears or max_iterations is reached.

Building this from scratch is intentional: it shows you understand what
LangChain is actually doing under the hood, which matters in MLE interviews.
"""
from pyexpat.errors import messages
import re
import time
import logging
from typing import List, Tuple, Optional

from docmind.models import AgentResponse, SearchResult
from docmind.agent.prompts import (
    REACT_SYSTEM_PROMPT,
    build_user_message,
    format_observation,
)
from docmind.llm.groq_client import get_llm
from docmind.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

# Regex patterns for parsing LLM output
ACTION_PATTERN = re.compile(r'Action:\s*(\w+)\("([^"]+)"\)', re.IGNORECASE)
FINAL_ANSWER_PATTERN = re.compile(r'Final Answer:\s*(.*)', re.DOTALL | re.IGNORECASE)


class ReActAgent:
    def __init__(self, retriever: Optional[HybridRetriever] = None):
        self.llm = get_llm()
        self.retriever = retriever or HybridRetriever()
        self.max_iterations = 5

    def _execute_action(self, action_name: str, argument: str) -> Tuple[List[SearchResult], str]:
        """Dispatch an Action to the corresponding tool and return (results, observation_text)."""
        if action_name.lower() == "search_documents":
            results = self.retriever.search(argument)
            observation = format_observation(results)
            return results, observation
        else:
            return [], f"Unknown tool: {action_name}. Available tools: search_documents"

    def run(self, question: str) -> AgentResponse:
        """
        Execute the ReAct loop for a user question.
        Returns a structured AgentResponse with answer, sources, and reasoning trace.
        """
        start_time = time.time()

        # Conversation history: we append to this each iteration
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question)},
        ]

        reasoning_trace: List[str] = []
        all_sources: List[SearchResult] = []
        final_answer = None

        for iteration in range(self.max_iterations):
            logger.debug(f"ReAct iteration {iteration + 1}/{self.max_iterations}")

            # Get next step from LLM
            response_text = self.llm.complete(messages)
            reasoning_trace.append(response_text)

            logger.debug(f"LLM response:\n{response_text}")

            # Check if we have a Final Answer, but only allow it if we've retrieved something
            final_match = FINAL_ANSWER_PATTERN.search(response_text)
            if final_match:
                if all_sources or iteration >= self.max_iterations - 1:
                    final_answer = final_match.group(1).strip()
                    break
            else:
                # Model tried to answer without searching, force a retrieval
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": "You must call search_documents before providing a Final Answer. Search the document corpus first."
                })
            continue

            # Check if we have an Action to execute
            action_match = ACTION_PATTERN.search(response_text)
            if action_match:
                action_name = action_match.group(1)
                argument = action_match.group(2)

                logger.debug(f"Executing: {action_name}('{argument}')")
                sources, observation_text = self._execute_action(action_name, argument)
                all_sources.extend(sources)

                # Inject the observation back into conversation
                # We append the assistant's reasoning + the observation as a user turn
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation_text}\n\nContinue your reasoning."
                })
            else:
                # LLM gave us neither an action nor a final answer
                # Nudge it to continue in the correct format
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": "Continue. Either search for more information or provide your Final Answer."
                })

        if final_answer is None:
            # Fallback: extract whatever the last response contained
            final_answer = (
                reasoning_trace[-1] if reasoning_trace
                else "I was unable to find a satisfactory answer in the document corpus."
            )

        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"ReAct completed in {latency_ms:.0f}ms, {iteration + 1} iterations")

        # Deduplicate sources by chunk_id
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
