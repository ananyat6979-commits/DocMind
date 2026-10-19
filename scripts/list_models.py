"""
List the chat models your Groq key can use right now, and show the current DocMind model settings.

  python scripts/list_models.py

Hosted model ids are retired from time to time. Run this whenever a call fails with "model not found",
then set DOCMIND_LLM_MODEL (generator) and DOCMIND_JUDGE_MODEL (evaluation judge, ideally larger).
Reasoning models also want DOCMIND_REASONING_EFFORT=low and DOCMIND_JUDGE_REASONING_EFFORT=low.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docmind.config import CONFIG  # noqa: E402


def main() -> int:
    if not CONFIG.llm.groq_api_key:
        print("GROQ_API_KEY is not set.")
        return 1
    from groq import Groq
    from docmind.llm.groq_client import list_chat_models
    ids = list_chat_models(Groq(api_key=CONFIG.llm.groq_api_key, max_retries=0))
    print("Chat models available to this key:")
    for i in ids:
        print("  ", i)
    print("\nCurrent settings:")
    print("   DOCMIND_LLM_MODEL   =", CONFIG.llm.groq_model or "(not set)")
    print("   DOCMIND_JUDGE_MODEL =", CONFIG.eval.judge_model or "(not set)")
    for label, val in (("generator", CONFIG.llm.groq_model), ("judge", CONFIG.eval.judge_model)):
        if val and val not in ids:
            print(f"   WARNING: the {label} model '{val}' is not in the list above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
