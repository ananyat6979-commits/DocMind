"""Real-stack verification, token accounting and the consolidated metrics report."""
import json
import os
import subprocess
import sys
from pathlib import Path

from docmind.evaluation.metrics import answer_level_stats
from tests.conftest import ALPHA

ROOT = Path(__file__).resolve().parents[1]


def _run(script, args, home, test_mode=True):
    env = dict(os.environ, DOCMIND_HOME=str(home), PYTHONPATH=str(ROOT))
    env.pop("GROQ_API_KEY", None)
    if test_mode:
        env["DOCMIND_TEST_MODE"] = "1"
    else:
        env.pop("DOCMIND_TEST_MODE", None)
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], env=env,
                          capture_output=True, text=True, cwd=home, timeout=180)


def test_stack_check_refuses_to_certify_offline_stand_ins(tmp_path):
    r = _run("stack_check.py", ["--skip-llm"], tmp_path)
    rep = json.loads((tmp_path / "reports" / "stack_check.json").read_text())
    assert r.returncode == 1 and rep["real_stack"] is False
    mode = next(c for c in rep["checks"] if c["name"] == "mode")
    assert mode["status"] == "fail" and "stand-in" in mode["detail"]
    assert "NOT VERIFIED" in r.stdout


def test_stack_check_plumbing_passes_with_stand_ins_but_is_never_real(tmp_path):
    r = _run("stack_check.py", ["--skip-llm", "--allow-stand-ins"], tmp_path)
    rep = json.loads((tmp_path / "reports" / "stack_check.json").read_text())
    by = {c["name"]: c for c in rep["checks"]}
    assert by["mode"]["status"] == "warn"
    assert by["embedder"]["status"] == "pass" and by["retrieval_smoke"]["status"] == "pass"
    assert len(by["embedder"]["metrics"]["sentences_per_second_runs"]) == 3 and "median" in by["embedder"]["detail"]
    assert rep["git_dirty"] in (True, False)
    assert by["llm"]["status"] == "skip" and rep["real_stack"] is False
    assert rep["scope"].startswith("partial") and by["retrieval_smoke"]["metrics"]["hit_at_1"] == 1.0


def test_token_deltas_recorded_per_answer(make_file):
    from docmind.agent.react import ReActAgent
    from docmind.ingestion.pipeline import ingest
    from docmind.retrieval.hybrid import HybridRetriever
    from docmind.store import CorpusStore

    class TokenLLM:
        prompt_tokens, completion_tokens = 1000, 50          # pre-existing totals must be subtracted
        replies = ['Action: search_documents("coolant temperature startup")', "Final Answer: Below 300 degrees (Source: a.txt)."]

        def complete(self, messages, stop=None, **kw):
            self.prompt_tokens += 700
            self.completion_tokens += 30
            return self.replies.pop(0)

    store = CorpusStore(); ingest(make_file("a.txt", ALPHA), store=store)
    r = ReActAgent(retriever=HybridRetriever(store), llm=TokenLLM()).run("coolant?")
    assert r.timings["prompt_tokens"] == 1400.0 and r.timings["completion_tokens"] == 60.0


def test_answer_stats_include_token_means_and_finish_reasons():
    rec = lambda p, c, fr: {"abstained": False, "citation_report": {"cited": ["a"], "valid": ["a"], "invalid": []},
                            "answerable": True, "latency_ms": 100.0, "llm_calls": 2, "needs_review": False,
                            "finish_reason": fr, "timings": {"prompt_tokens": p, "completion_tokens": c}}
    s = answer_level_stats([rec(1000, 50, "final_answer"), rec(2000, 150, "synthesis")])
    assert s["mean_prompt_tokens"] == 1500.0 and s["mean_completion_tokens"] == 100.0 and s["total_tokens"] == 3200.0
    assert s["finish_reasons"] == {"final_answer": 1, "synthesis": 1}
    assert answer_level_stats([{**rec(0, 0, "x"), "timings": {}}])["mean_prompt_tokens"] is None


def test_record_metrics_never_invents_numbers(tmp_path):
    r = _run("record_metrics.py", [], tmp_path)
    assert r.returncode == 0, r.stderr[-800:]
    md = (tmp_path / "reports" / "METRICS.md").read_text()
    assert md.count("NOT RUN") >= 6                          # nothing measured yet: every section says so
    assert "0." not in md.split("## 3.")[1].split("## 4.")[0]


def test_record_metrics_reports_what_exists(tmp_path):
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "ragas_latest.json").write_text(json.dumps({
        "generator_model": "g", "judge_model": "j", "generated_at": "t", "n_scored": 2,
        "ragas": {"faithfulness": {"mean": 0.9, "ci95": [0.8, 1.0], "n": 2, "n_failed": 0}}, "answer_level": {}}))
    (tmp_path / "reports" / "generations.jsonl").write_text(json.dumps({
        "id": "q1", "abstained": False, "answerable": True, "latency_ms": 2000.0, "llm_calls": 3, "needs_review": False,
        "finish_reason": "final_answer", "citation_report": {"cited": ["a"], "valid": ["a"], "invalid": []},
        "timings": {"prompt_tokens": 1200.0, "completion_tokens": 80.0}}) + "\n")
    _run("record_metrics.py", [], tmp_path)
    md = (tmp_path / "reports" / "METRICS.md").read_text()
    assert "faithfulness | 0.900 [0.800, 1.000]" in md
    assert "Tokens per answer: prompt 1200, completion 80" in md and "Citation validity: 1.000" in md


def test_update_readme_fills_stack_block(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    ur = importlib.import_module("update_readme")
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "stack_check.json").write_text(json.dumps({
        "real_stack": True, "scope": "full", "git_commit": "abc1234", "generated_at": "t", "python": "3.11",
        "versions": {"torch": "2.4", "sentence-transformers": "3.0", "faiss-cpu": "1.8", "ragas": "0.3.9"},
        "checks": [{"name": "embedder", "status": "pass", "detail": "dim=384"}]}))
    monkeypatch.setattr(ur, "ROOT", tmp_path)
    assert "Real stack verified: **True**" in ur.stack_block() and "embedder | pass" in ur.stack_block()
    (tmp_path / "reports" / "stack_check.json").unlink()
    assert ur.stack_block().startswith("Not yet run")


def test_untracked_report_does_not_mark_the_tree_dirty(tmp_path):
    """git_dirty must reflect modified tracked files only; the report itself is untracked output."""
    import subprocess as sp
    repo = tmp_path / "r"; repo.mkdir()
    for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@t"], ["git", "config", "user.name", "t"]):
        sp.run(cmd, cwd=repo, check=True)
    (repo / "a.txt").write_text("x"); sp.run(["git", "add", "a.txt"], cwd=repo, check=True)
    sp.run(["git", "commit", "-q", "-m", "m"], cwd=repo, check=True)
    (repo / "report.json").write_text("{}")                                  # untracked
    clean = sp.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True, text=True).stdout
    (repo / "a.txt").write_text("changed")                                   # tracked and modified
    dirty = sp.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True, text=True).stdout
    assert clean == "" and dirty != ""
