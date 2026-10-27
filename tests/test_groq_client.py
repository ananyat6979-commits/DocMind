"""GroqLLM behaviour against a fake client: retired models, reasoning models, empty completions."""
import sys
import types

import pytest

from docmind.config import CONFIG
from docmind.llm import groq_client
from docmind.llm.groq_client import EmptyCompletion, GroqLLM, ModelUnavailable, clean_completion


class NotFoundError(Exception):          # matched by class name, like the real SDK's 404 error
    status_code = 404


def _resp(content, finish="stop", prompt=10, completion=5):
    msg = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason=finish)],
                                 usage=types.SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion))


class FakeGroq:
    behaviour = None          # callable(request) -> response | raises
    models = ["whisper-large", "model-a", "model-b", "tts-voice", "some-guard"]
    last_request = None

    def __init__(self, api_key=None, max_retries=0):
        outer = self
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: outer._create(kw)))
        self.models = types.SimpleNamespace(list=lambda: types.SimpleNamespace(
            data=[types.SimpleNamespace(id=i) for i in FakeGroq.models]))

    def _create(self, kw):
        FakeGroq.last_request = kw
        return FakeGroq.behaviour(kw)


@pytest.fixture
def groq(monkeypatch):
    # Inject a fake module so the tests never need the real SDK (CI installs only light test dependencies).
    fake_module = types.ModuleType("groq")
    fake_module.Groq = FakeGroq
    monkeypatch.setitem(sys.modules, "groq", fake_module)
    monkeypatch.setattr(CONFIG.llm, "groq_api_key", "k")
    monkeypatch.setattr(CONFIG.llm, "groq_model", "model-a")
    monkeypatch.setattr(CONFIG.llm, "reasoning_effort", None)
    monkeypatch.setattr(CONFIG.llm, "max_tokens", 2048)
    FakeGroq.behaviour = lambda kw: _resp("OK")
    return FakeGroq


def test_missing_model_error_lists_what_the_key_can_use(groq, monkeypatch):
    monkeypatch.setattr(CONFIG.llm, "groq_model", "")
    with pytest.raises(ValueError) as e:
        GroqLLM()
    msg = str(e.value)
    assert "DOCMIND_LLM_MODEL" in msg and "model-a" in msg and "model-b" in msg
    assert "whisper" not in msg and "tts" not in msg and "guard" not in msg      # non-chat models filtered


def test_retired_model_gives_actionable_error_not_a_raw_404(groq):
    def boom(kw):
        raise NotFoundError("model_not_found")
    groq.behaviour = boom
    with pytest.raises(ModelUnavailable) as e:
        GroqLLM().complete([{"role": "user", "content": "hi"}])
    assert "model-a" in str(e.value) and "model-b" in str(e.value) and "DOCMIND_LLM_MODEL" in str(e.value)


def test_reasoning_effort_and_token_budget_are_sent_only_when_configured(groq, monkeypatch):
    GroqLLM().complete([{"role": "user", "content": "hi"}])
    assert "reasoning_effort" not in groq.last_request and groq.last_request["max_tokens"] == 2048
    monkeypatch.setattr(CONFIG.llm, "reasoning_effort", "low")
    monkeypatch.setattr(CONFIG.llm, "max_tokens", 4096)
    GroqLLM().complete([{"role": "user", "content": "hi"}])
    assert groq.last_request["reasoning_effort"] == "low" and groq.last_request["max_tokens"] == 4096
    GroqLLM(reasoning_effort="high").complete([{"role": "user", "content": "hi"}])
    assert groq.last_request["reasoning_effort"] == "high"


def test_empty_answer_from_exhausted_budget_is_reported_with_the_fix(groq):
    groq.behaviour = lambda kw: _resp("", finish="length")
    with pytest.raises(EmptyCompletion) as e:
        GroqLLM().complete([{"role": "user", "content": "hi"}])
    assert "DOCMIND_MAX_TOKENS" in str(e.value) and "DOCMIND_REASONING_EFFORT" in str(e.value)


def test_empty_answer_with_normal_stop_is_returned_as_empty_string(groq):
    groq.behaviour = lambda kw: _resp("", finish="stop")
    assert GroqLLM().complete([{"role": "user", "content": "hi"}]) == ""


def test_inline_reasoning_blocks_are_stripped():
    assert clean_completion("<think>plan the answer</think>\nFinal Answer: x") == "Final Answer: x"
    assert clean_completion("<think>never finished") == ""
    assert clean_completion(None) == "" and clean_completion("  plain  ") == "plain"


def test_token_usage_accumulates(groq):
    llm = GroqLLM()
    llm.complete([{"role": "user", "content": "a"}])
    llm.complete([{"role": "user", "content": "b"}])
    assert (llm.prompt_tokens, llm.completion_tokens, llm.calls) == (20, 10, 2)


def test_judge_requires_a_configured_model(monkeypatch):
    pytest.importorskip("ragas")
    pytest.importorskip("langchain_groq")
    from docmind.evaluation import harness
    monkeypatch.setattr(CONFIG.llm, "groq_api_key", "k")
    monkeypatch.setattr(CONFIG.eval, "judge_model", "")
    with pytest.raises(ValueError, match="DOCMIND_JUDGE_MODEL"):
        harness.build_judge()


def test_stack_check_fails_clearly_when_models_unset(tmp_path):
    import os, subprocess, sys, json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, DOCMIND_HOME=str(tmp_path), PYTHONPATH=str(root), GROQ_API_KEY="k", DOCMIND_LLM_MODEL="", DOCMIND_JUDGE_MODEL="")
    env.pop("DOCMIND_TEST_MODE", None)
    # offline stand-ins would hide the check, so only inspect the mode logic via the real config
    code = ("import sys; sys.argv=['x','--allow-stand-ins']; sys.path.insert(0,'scripts'); import stack_check as s, argparse;"
            "a=argparse.Namespace(skip_llm=False, allow_stand_ins=False); print(s.c_mode(a)[1])")
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=root, capture_output=True, text=True).stdout
    assert "DOCMIND_LLM_MODEL is not set" in out and "DOCMIND_JUDGE_MODEL is not set" in out


# ------------------------------------------------------------ native tool-call recovery
class BadRequestError(Exception):
    status_code = 400

    def __init__(self, generation):
        super().__init__("tool_use_failed")
        self.body = {"error": {"code": "tool_use_failed", "message": "Tool choice is none, but model called a tool",
                               "failed_generation": generation}}


def test_tool_call_names_match_the_agent_tools():
    from docmind.agent.tools import TOOL_NAMES
    assert tuple(TOOL_NAMES) == groq_client._KNOWN_TOOLS


@pytest.mark.parametrize("raw, expected", [
    ('{"name": "search_documents", "arguments": {"query":"coolant temperature limit"}}', 'Action: search_documents("coolant temperature limit")'),
    ('{"name": "search_documents", "arguments": "{\\"query\\": \\"x y\\"}"}', 'Action: search_documents("x y")'),
    ('{"name": "get_chunk_context", "arguments": {"chunk_id": "ab12cd34"}}', 'Action: get_chunk_context("ab12cd34")'),
    ('[{"name": "search_documents", "arguments": {"query": "listed"}}]', 'Action: search_documents("listed")'),
    ('<function=search_documents>{"query": "tagged"}</function>', 'Action: search_documents("tagged")'),
    ('{"name": "search_documents", "arguments": {"query": "say \\"hi\\""}}', "Action: search_documents(\"say 'hi'\")"),
])
def test_tool_call_is_converted_to_react_text(raw, expected):
    out = groq_client.tool_call_to_react(raw)
    assert out is not None and expected in out and out.startswith("Thought:")


@pytest.mark.parametrize("raw", [
    '{"name": "delete_everything", "arguments": {"query": "x"}}',      # not one of our tools
    '{"name": "search_documents", "arguments": {}}',                    # no usable argument
    "not json at all", "", "[]",
])
def test_unrecognised_tool_calls_are_not_invented(raw):
    assert groq_client.tool_call_to_react(raw) is None


def test_rejected_native_tool_call_is_recovered_by_the_client(groq):
    def reject(kw):
        raise BadRequestError('{"name": "search_documents", "arguments": {"query":"coolant temperature limit during startup"}}')
    groq.behaviour = reject
    llm = GroqLLM()
    out = llm.complete([{"role": "user", "content": "q"}])
    assert 'Action: search_documents("coolant temperature limit during startup")' in out and llm.calls == 1


def test_other_bad_requests_still_raise(groq):
    class Other(BadRequestError):
        def __init__(self):
            super().__init__("x")
            self.body = {"error": {"code": "invalid_request_error", "message": "nope"}}
    groq.behaviour = lambda kw: (_ for _ in ()).throw(Other())
    with pytest.raises(Other):
        GroqLLM().complete([{"role": "user", "content": "q"}])


def test_agent_completes_when_the_model_uses_native_tool_calls(groq, make_file):
    """End to end: native call rejected by the API -> recovered -> search runs -> grounded, cited answer."""
    from docmind.agent.react import ReActAgent
    from docmind.ingestion.pipeline import ingest
    from docmind.retrieval.hybrid import HybridRetriever
    from docmind.store import CorpusStore
    from tests.conftest import ALPHA

    store = CorpusStore(); ingest(make_file("a.txt", ALPHA), store=store)
    calls = []

    def behaviour(kw):
        calls.append(1)
        if len(calls) == 1:
            raise BadRequestError('{"name": "search_documents", "arguments": {"query":"coolant temperature startup"}}')
        return _resp("Thought: done.\nFinal Answer: Coolant must stay below 300 degrees (Source: a.txt).")
    groq.behaviour = behaviour
    r = ReActAgent(retriever=HybridRetriever(store), llm=GroqLLM()).run("What is the coolant limit?")
    assert "300 degrees" in r.answer and r.citation_report["validity"] == 1.0 and not r.abstained
    assert r.llm_calls == 2 and r.finish_reason == "final_answer"


def test_chunker_env_switch(monkeypatch):
    import docmind.config as c
    monkeypatch.setenv("DOCMIND_CHUNKER", "fixed")
    assert c.DocMindConfig().chunker.strategy == "fixed"
    monkeypatch.delenv("DOCMIND_CHUNKER")
    assert c.DocMindConfig().chunker.strategy == "semantic"
