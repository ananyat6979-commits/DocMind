"""Provider behavior is simulated; these tests never use a key or network."""
import asyncio
import sys
import time
from types import SimpleNamespace
import pytest
from docmind.llm.groq_client import GroqLLM, RateLimitExhausted
from docmind.llm.execution import CompletionSession, DeadlineExceeded


def backend(monkeypatch, outcomes):
    state = SimpleNamespace(calls=0, active=0, peak=0, options=[])
    class Client:
        def __init__(self, **options):
            assert options['max_retries'] == 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        async def create(self, **options):
            state.options.append(options);state.calls += 1
            value = outcomes.pop(0)
            if isinstance(value, Exception):raise value
            state.active += 1;state.peak = max(state.peak, state.active)
            try:
                await asyncio.sleep(.02)
                return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                    choices=[SimpleNamespace(message=SimpleNamespace(content=value))])
            finally:state.active -= 1
    monkeypatch.setitem(sys.modules, 'groq', SimpleNamespace(AsyncGroq=Client))
    instance = GroqLLM.__new__(GroqLLM)
    instance._model='fake';instance._reasoning_effort=None
    instance.calls=instance.prompt_tokens=instance.completion_tokens=0;instance.call_records=[]
    return instance, state


def test_retry_attempts_and_unknown_usage_stay_in_request_ledger(monkeypatch):
    transient = type('APIConnectionError', (Exception,), {})('connection')
    instance, state = backend(monkeypatch, [transient, 'answer'])
    session=CompletionSession(instance, time.monotonic()+2)
    assert asyncio.run(session.complete([])) == 'answer'
    assert session.calls == state.calls == 2
    assert session.usage() == {'prompt_tokens':10, 'completion_tokens':5, 'usage_unknown_calls':1}
    assert session.records[0]['status'] == 'error'


def test_quota_fails_without_retry(monkeypatch):
    quota=type('RateLimitError', (Exception,), {'status_code':429})('limit')
    instance,state=backend(monkeypatch,[quota])
    with pytest.raises(RateLimitExhausted):asyncio.run(instance.acomplete([]))
    assert state.calls == 1 and instance.call_records[0]['input_tokens'] is None


def test_evaluation_zero_retry_and_capacity_bound(monkeypatch):
    instance,state=backend(monkeypatch,['answer']*8)
    async def run():
        return await asyncio.gather(*(instance.acomplete([], retries=0) for _ in range(8)))
    assert asyncio.run(run()) == ['answer']*8
    assert state.peak == 4 and state.calls == 8
    assert all(option['timeout'] <= 30 for option in state.options)


def test_default_provider_timeout_caps_monotonic_rounding(monkeypatch):
    import docmind.llm.groq_client as module

    now = 5.999999999999997
    assert (now + 30.0) - now > 30.0
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now))
    instance, state = backend(monkeypatch, ["default", "explicit"])
    assert asyncio.run(instance.acomplete([], retries=0)) == "default"
    assert asyncio.run(instance.acomplete([], deadline=now + 45, retries=0)) == "explicit"
    assert [option["timeout"] for option in state.options] == [30.0, 45.0]


def test_expired_capacity_wait_releases_slot_without_provider_attempt(monkeypatch):
    import docmind.llm.execution as execution
    import docmind.llm.groq_client as module

    clock = SimpleNamespace(now=0.0, released=0)

    class Slot:
        async def acquire(self):
            clock.now = 2.0

        def release(self):
            clock.released += 1

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(execution, "limit", lambda *args: Slot())
    instance, state = backend(monkeypatch, [])
    with pytest.raises(DeadlineExceeded):
        asyncio.run(instance.acomplete([], deadline=1.0, retries=0))
    assert clock.released == 1 and state.calls == instance.calls == 0
    assert instance.call_records == []
