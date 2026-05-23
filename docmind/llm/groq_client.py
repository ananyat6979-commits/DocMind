"""
Groq LLM client.
Free tier: 100k tokens/minute on llama-3.1-8b-instant: more than enough.
Get your key at https://console.groq.com (free, no credit card needed).
"""
import logging
from typing import List, Dict
from groq import Groq
from docmind.llm.base import BaseLLM
from docmind.config import CONFIG

logger = logging.getLogger(__name__)


class GroqLLM(BaseLLM):
    def __init__(self):
        if not CONFIG.llm.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Add it to your .env file.\n"
                "Get a free key at https://console.groq.com"
            )
        self._client = Groq(api_key=CONFIG.llm.groq_api_key)
        self._model = CONFIG.llm.groq_model

    def complete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        temperature = kwargs.get("temperature", CONFIG.llm.temperature)
        max_tokens = kwargs.get("max_tokens", CONFIG.llm.max_tokens)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


def get_llm() -> BaseLLM:
    """Factory: returns the configured LLM implementation."""
    if CONFIG.llm.provider == "groq":
        return GroqLLM()
    # Extend here: add OllamaLLM for local fallback
    raise ValueError(f"Unknown LLM provider: {CONFIG.llm.provider}")
