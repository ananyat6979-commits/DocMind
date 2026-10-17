"""
Groq LLM client with rate-limit-aware retries.

Groq free-tier limits (console.groq.com/docs/rate-limits, checked Oct 2026):
  llama-3.1-8b-instant     30 RPM | 6K TPM  | 500K tokens/day
  llama-3.3-70b-versatile  30 RPM | 12K TPM | 100K tokens/day
A multi-step ReAct run can burn 5-8K tokens, so 429s are normal, not an error.
This client honours the Retry-After header, backs off with jitter, and fails
fast (instead of sleeping for hours) when a daily cap is hit.
"""
import logging
import random
import time
from typing import Dict, List, Optional

from docmind.config import CONFIG
from docmind.llm.base import BaseLLM

logger = logging.getLogger(__name__)

_RETRYABLE = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}


def _retry_after(exc: Exception) -> Optional[float]:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers:
        for key in ("retry-after", "Retry-After"):
            val = headers.get(key)
            if val:
                try:
                    return float(val)
                except ValueError:
                    pass
    return None


class RateLimitExhausted(RuntimeError):
    """Groq asked us to wait longer than max_sleep_seconds (usually the daily token cap)."""


class GroqLLM(BaseLLM):
    def __init__(self, model: Optional[str] = None):
        if not CONFIG.llm.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Add it to .env or the environment.\n"
                "Get a free key at https://console.groq.com"
            )
        from groq import Groq
        self._client = Groq(api_key=CONFIG.llm.groq_api_key, max_retries=0)
        self._model = model or CONFIG.llm.groq_model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def complete(self, messages: List[Dict[str, str]], stop=None, **kwargs) -> str:
        temperature = kwargs.get("temperature", CONFIG.llm.temperature)
        max_tokens = kwargs.get("max_tokens", CONFIG.llm.max_tokens)
        last_exc: Optional[Exception] = None

        for attempt in range(CONFIG.llm.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, stop=stop,
                )
                self.calls += 1
                usage = getattr(resp, "usage", None)
                if usage:
                    self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                if type(exc).__name__ not in _RETRYABLE or attempt == CONFIG.llm.max_retries:
                    raise
                wait = _retry_after(exc)
                if wait is None:
                    wait = min(2 ** attempt, 60) + random.uniform(0, 1)
                if wait > CONFIG.llm.max_sleep_seconds:
                    raise RateLimitExhausted(
                        f"Groq asked us to wait {wait:.0f}s (likely the daily token cap). "
                        "Try again later, or switch model / upgrade the plan."
                    ) from exc
                logger.warning(f"Groq {type(exc).__name__}; retry {attempt + 1} in {wait:.1f}s")
                time.sleep(wait)
        raise last_exc  # pragma: no cover


def get_llm(model: Optional[str] = None) -> BaseLLM:
    """Factory for the configured LLM."""
    if CONFIG.llm.provider == "demo":
        from docmind.llm.demo import DemoLLM
        return DemoLLM()
    if CONFIG.llm.provider == "groq":
        return GroqLLM(model=model)
    raise ValueError(f"Unknown LLM provider: {CONFIG.llm.provider}")
