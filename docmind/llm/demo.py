"""
DemoLLM: a tiny deterministic, extractive stand-in for the real model.

It exists so the whole app (UI, API, review queue, tests) runs offline with no
API key. It is NOT a quality baseline and must never be used for evaluation.
Behaviour: step 1 searches the question; step 2 quotes the best passage with its
cite_as string, or answers NOT_FOUND if the observation had nothing.
"""
import re
from typing import Dict, List

from docmind.llm.base import BaseLLM
from docmind.agent.prompts import NOT_FOUND_TOKEN


class DemoLLM(BaseLLM):
    is_demo = True

    def complete(self, messages: List[Dict[str, str]], stop=None, **kwargs) -> str:
        observations = [m["content"] for m in messages if m["role"] == "user" and m["content"].startswith("Observation:")]
        if not observations:
            q = re.search(r"Question:\s*(.*?)\n", messages[-1]["content"] + "\n", re.S)
            question = (q.group(1) if q else "").strip().replace('"', "'")
            return f'Thought: I should search the documents.\nAction: search_documents("{question}")'
        obs = observations[-1]
        if "No relevant documents found" in obs:
            return f"Thought: Nothing relevant came back.\nFinal Answer: {NOT_FOUND_TOKEN}: no matching passage was retrieved."
        m = re.search(r"cite_as: (\(Source:[^\n]*?\)) \| chunk_id: [0-9a-f]+\n(.*?)(?:\n\n|\Z)", obs, re.S)
        if not m:
            return f"Thought: I could not parse a passage.\nFinal Answer: {NOT_FOUND_TOKEN}: no usable passage."
        cite, text = m.group(1), m.group(2).strip().replace("\n", " ")
        return f"Thought: The top passage looks relevant.\nFinal Answer: {text[:400]} {cite}"
