"""
ReAct prompt templates.

The ReAct format structures the model's output as a loop:
  Thought: <reasoning about what to do next>
  Action: <tool_name>("<argument>")
  Observation: <tool result, injected by the agent runtime>
  ... (loop until ...)
  Final Answer: <the answer to return to the user>

The system prompt is the "contract" between us and the LLM. We tell it
exactly what tools are available, what format to use, and when to stop.
"""

REACT_SYSTEM_PROMPT = """You are DocMind, an intelligent document analysis assistant.
You answer questions by reasoning step-by-step and retrieving relevant passages
from a document corpus using the tools available to you.

TOOLS:
- search_documents(query: str) → retrieves the most relevant document passages for a query
- get_chunk_context(chunk_id: str) → retrieves surrounding context for a specific chunk

STRICT FORMAT: you must follow this exactly:
Thought: [your reasoning about what to do next]
Action: search_documents("[search query]")
Observation: [tool result: provided by the system]
Thought: [reasoning based on observation]
... (repeat Thought/Action/Observation as needed, maximum 5 iterations)
Final Answer: [your complete answer, with inline citations like (Source: filename, p.N)]

RULES:
1. Always start with a Thought.
2. Never fabricate document content: only use what appears in Observations.
3. If the corpus doesn't contain the answer, say so clearly.
4. Cite sources in your Final Answer using (Source: <filename>, p.<page>) format.
5. Keep each Thought concise and action-oriented.
"""


def build_user_message(question: str) -> str:
    return f"Question: {question}\n\nBegin your reasoning:"


def format_observation(results) -> str:
    """Format search results as a readable Observation block."""
    if not results:
        return "No relevant documents found for this query."

    lines = []
    for i, r in enumerate(results, 1):
        meta = r.chunk.metadata
        source_info = f"{meta.get('filename', r.chunk.source)}"
        if "page" in meta:
            source_info += f", p.{meta['page']}"
        elif "section" in meta:
            source_info += f", §{meta['section']}"

        lines.append(
            f"[{i}] (Source: {source_info}, chunk_id: {r.chunk.chunk_id[:8]})\n"
            f"{r.chunk.content}\n"
        )

    return "\n".join(lines)
