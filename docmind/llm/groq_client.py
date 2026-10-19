"""
Groq LLM client: rate-limit-aware retries, reasoning-model support, and actionable errors.

Groq's free-tier limits are small and change, and hosted model ids are retired from time to time. So:
  * the model id is configuration (DOCMIND_LLM_MODEL), never a hidden default;
  * a "model not found" error lists the models your key can actually use;
  * 429s honour Retry-After, back off with jitter, and fail fast when a daily cap is hit
    (check console.groq.com/docs/rate-limits for the current limits of your model).
Reasoning models spend completion tokens on hidden reasoning. If the budget is too small the visible
answer comes back empty, which is reported as EmptyCompletion with the fix, instead of failing later
in a confusing way.
"""
import logging
import random
import re
import time
from typing import Dict, List, Optional

from docmind.config import CONFIG
from docmind.llm.base import BaseLLM

logger = logging.getLogger(__name__)

_RETRYABLE = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}
_NOT_CHAT = ("whisper", "tts", "playai", "guard", "safeguard", "embed", "orpheus")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class RateLimitExhausted(RuntimeError):
    """Groq asked us to wait longer than max_sleep_seconds (usually the daily token cap)."""


class ModelUnavailable(RuntimeError):
    """The configured model id does not exist for this key (retired, misspelled, or no access)."""


class EmptyCompletion(RuntimeError):
    """The model produced no visible text, typically because reasoning used the whole token budget."""


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


def clean_completion(text: str) -> str:
    """Remove inline <think>...</think> blocks some models emit. An unterminated block means the
    budget ran out mid-thought, so there is no usable answer."""
    text = _THINK.sub("", text or "")
    if "<think>" in text.lower():
        return ""
    return text.strip()


def list_chat_models(client) -> List[str]:
    ids = sorted(m.id for m in client.models.list().data)
    return [i for i in ids if not any(k in i.lower() for k in _NOT_CHAT)]


class GroqLLM(BaseLLM):
    def __init__(self, model: Optional[str] = None, reasoning_effort: Optional[str] = None):
        if not CONFIG.llm.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Add it to .env or the environment.\n"
                "Get a free key at https://console.groq.com"
            )
        from groq import Groq
        self._client = Groq(api_key=CONFIG.llm.groq_api_key, max_retries=0)
        self._model = model or CONFIG.llm.groq_model
        self._reasoning_effort = reasoning_effort if reasoning_effort is not None else CONFIG.llm.reasoning_effort
        if not self._model:
            raise ValueError(self._missing_model_message())
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    # ------------------------------------------------------------------ helpers
    def _available(self) -> str:
        try:
            return ", ".join(list_chat_models(self._client)) or "(none returned)"
        except Exception as exc:
            return f"(could not list models: {type(exc).__name__})"

    def _missing_model_message(self) -> str:
        return ("No model configured. Set DOCMIND_LLM_MODEL (generator) and DOCMIND_JUDGE_MODEL (evaluation judge) "
                "to ids your key can use.\nModels available to this key: " + self._available() +
                "\nExample (bash): export DOCMIND_LLM_MODEL=<id>   PowerShell: $env:DOCMIND_LLM_MODEL=\"<id>\"")

    # ---------------------------------------------------------------------- API
    def complete(self, messages: List[Dict[str, str]], stop=None, **kwargs) -> str:
        temperature = kwargs.get("temperature", CONFIG.llm.temperature)
        max_tokens = kwargs.get("max_tokens", CONFIG.llm.max_tokens)
        request = dict(model=self._model, messages=messages, temperature=temperature,
                       max_tokens=max_tokens, stop=stop)
        if self._reasoning_effort:
            request["reasoning_effort"] = self._reasoning_effort
        last_exc: Optional[Exception] = None

        for attempt in range(CONFIG.llm.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**request)
                self.calls += 1
                usage = getattr(resp, "usage", None)
                if usage:
                    self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                choice = resp.choices[0]
                text = clean_completion(choice.message.content or "")
                if not text and getattr(choice, "finish_reason", "") == "length":
                    raise EmptyCompletion(
                        f"{self._model} used its whole {max_tokens}-token budget before writing an answer "
                        "(hidden reasoning). Raise DOCMIND_MAX_TOKENS (try 4096) or lower reasoning with "
                        "DOCMIND_REASONING_EFFORT=low.")
                return text
            except Exception as exc:
                last_exc = exc
                name = type(exc).__name__
                if name == "NotFoundError" or getattr(exc, "status_code", None) == 404:
                    raise ModelUnavailable(
                        f"Model '{self._model}' does not exist for this key (retired, misspelled, or no access).\n"
                        f"Models available to this key: {self._available()}\n"
                        "Set DOCMIND_LLM_MODEL / DOCMIND_JUDGE_MODEL to one of them.") from exc
                if name not in _RETRYABLE or attempt == CONFIG.llm.max_retries:
                    raise
                wait = _retry_after(exc)
                if wait is None:
                    wait = min(2 ** attempt, 60) + random.uniform(0, 1)
                if wait > CONFIG.llm.max_sleep_seconds:
                    raise RateLimitExhausted(
                        f"Groq asked us to wait {wait:.0f}s (likely the daily token cap). "
                        "Try again later, switch model, or upgrade the plan."
                    ) from exc
                logger.warning(f"Groq {name}; retry {attempt + 1} in {wait:.1f}s")
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
