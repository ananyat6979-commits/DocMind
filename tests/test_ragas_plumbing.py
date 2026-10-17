"""
Offline test of the RAGAS wiring using a FAKE judge. It proves schema, metric names,
to_pandas parsing, abstention exclusion and checkpoint/resume. It says nothing about
judge quality: real scores need the real Groq judge (scripts/evaluate.py).
Skipped automatically if ragas / langchain-groq are not installed.
"""
import json
import pytest

pytest.importorskip("ragas")
pytest.importorskip("langchain_groq")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

from docmind.config import CONFIG  # noqa: E402
from docmind.evaluation import harness  # noqa: E402


class FakeJudge(BaseChatModel):
    @property
    def _llm_type(self):
        return "fake-judge"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = " ".join(str(m.content) for m in messages)
        if "judge the faithfulness" in text:
            out = {"statements": [{"statement": "s", "reason": "r", "verdict": 1}]}
        elif "analyze the complexity of each sentence" in text:
            out = {"statements": ["Coolant stays below 300 degrees."]}
        elif "noncommittal" in text:
            out = {"question": "What is the coolant limit?", "noncommittal": 0}
        elif "useful in arriving" in text:
            out = {"reason": "relevant", "verdict": 1}
        elif "attributed to the given context" in text:
            out = {"classifications": [{"statement": "s", "reason": "r", "attributed": 1}]}
        else:
            out = {"verdict": 1, "reason": "r", "statements": []}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(out)),
                                                      generation_info={"finish_reason": "stop"})])


def _records():
    base = dict(contexts=["Coolant must stay below 300 degrees."], ground_truth="Below 300 degrees.",
                citation_report={"cited": ["a"], "valid": ["a"], "invalid": []}, needs_review=False,
                review_reasons=[], latency_ms=100.0, llm_calls=2, timings={}, finish_reason="final_answer")
    return [
        dict(base, id="q1", question="coolant limit?", answerable=True, answer="Below 300 degrees (Source: a).", abstained=False),
        dict(base, id="q2", question="coolant again?", answerable=True, answer="Below 300 degrees (Source: a).", abstained=False),
        dict(base, id="q3", question="revenue?", answerable=False, answer="I could not find this.", abstained=True),
    ]


@pytest.fixture
def judge():
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    from docmind.embeddings import embed_texts

    class E(Embeddings):
        def embed_documents(self, t): return embed_texts(list(t)).tolist()
        def embed_query(self, t): return embed_texts([t]).tolist()[0]

    rc = RunConfig(timeout=30, max_retries=1, max_workers=2)
    return LangchainLLMWrapper(FakeJudge(), run_config=rc, bypass_n=True), LangchainEmbeddingsWrapper(E()), rc


def test_score_excludes_abstentions_and_returns_all_four_metrics(judge, tmp_path):
    out = tmp_path / "scores.jsonl"
    scores = harness.score(_records(), out_path=str(out), judge=judge, batch_size=1)
    assert {s["id"] for s in scores} == {"q1", "q2"}          # abstained q3 excluded
    for s in scores:
        for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            assert s[m] is not None and 0.0 <= s[m] <= 1.0, (s, m)


def test_score_resumes_from_checkpoint(judge, tmp_path):
    out = tmp_path / "scores.jsonl"
    harness.score(_records()[:1], out_path=str(out), judge=judge)
    again = harness.score(_records(), out_path=str(out), judge=judge)
    assert [s["id"] for s in again] == ["q1", "q2"]            # q1 not re-scored, q2 added


def test_summary_has_cis_and_answer_level_stats(judge, tmp_path):
    recs = _records()
    scores = harness.score(recs, out_path=str(tmp_path / "s.jsonl"), judge=judge)
    s = harness.summarize(recs, scores)
    assert s["ragas"]["faithfulness"]["n"] == 2
    assert s["answer_level"]["correct_abstention_rate"] == 1.0
    assert s["answer_level"]["citation_validity"] == 1.0


class BrokenJudge(FakeJudge):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("429 simulated rate limit")


def test_failed_rows_are_not_checkpointed_so_they_retry(tmp_path):
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    from docmind.embeddings import embed_texts

    class E(Embeddings):
        def embed_documents(self, t): return embed_texts(list(t)).tolist()
        def embed_query(self, t): return embed_texts([t]).tolist()[0]

    rc = RunConfig(timeout=10, max_retries=1, max_workers=1)
    broken = (LangchainLLMWrapper(BrokenJudge(), run_config=rc, bypass_n=True), LangchainEmbeddingsWrapper(E()), rc)
    out = tmp_path / "s.jsonl"
    scores = harness.score(_records()[:1], out_path=str(out), judge=broken, metrics=["faithfulness"])
    assert scores == [] and (not out.exists() or out.read_text().strip() == "")
