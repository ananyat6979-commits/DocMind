"""The interactive testset builder: keys, re-prompting, quitting, duplicates and evidence for unanswerable drafts."""
import builtins
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from docmind.config import CONFIG
from docmind.models import Chunk

ROOT = Path(__file__).resolve().parents[1]
WORDS = ("amber basalt cobalt dahlia ember fjord garnet harbor indigo juniper kestrel lantern meadow nectar obsidian pewter "
         "quartz russet saffron thistle umber velvet willow xenon yarrow zephyr").split()


def _script():
    spec = importlib.util.spec_from_file_location("make_testset_under_test", ROOT / "scripts" / "make_testset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeLLM:
    same_unanswerable = False
    calls = 0

    def __init__(self, *a, **k):
        pass

    def complete(self, messages, **kw):
        FakeLLM.calls += 1
        n = FakeLLM.calls
        text = messages[0]["content"]
        if "Passage:" in text:
            passage = text.split("Passage:\n", 1)[1]
            q = " ".join(WORDS[(n * 3 + j) % len(WORDS)] for j in range(4))
            return json.dumps({"question": f"{q}?", "ground_truth": "an answer", "gold_span": passage[:50]})
        if FakeLLM.same_unanswerable:
            return json.dumps({"question": "Which harbor lantern was never discussed anywhere?"})
        q = " ".join(WORDS[(n * 5 + j * 7) % len(WORDS)] for j in range(4))
        return json.dumps({"question": f"{q} unknown?"})


def _docs(tmp_path, count=6):
    paths = []
    for i in range(count):
        a, b, c = WORDS[i], WORDS[i + 7], WORDS[i + 13]
        text = (f"The {a} procedure requires careful calibration of the {b} before every shift begins. "
                f"Operators record the {c} reading on the logbook and compare it with the previous entry. "
                f"If the {a} value drifts the supervisor must be informed before work continues. "
                f"A trained technician then adjusts the {b} assembly and verifies the result with a second reading. "
                f"Every adjustment is documented so that the {c} history of the equipment stays complete and auditable. "
                f"New staff study the {a} manual during their first week and shadow an experienced colleague. ") * 2
        p = tmp_path / f"doc{i}.txt"
        p.write_text(text)
        paths.append(str(p))
    return paths


@pytest.fixture
def run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(CONFIG.chunker, "strategy", "fixed")
    FakeLLM.calls, FakeLLM.same_unanswerable = 0, False
    docs = _docs(tmp_path)
    out = tmp_path / "testset.json"

    def _run(keys, extra=(), llm=FakeLLM):
        it = iter(keys)

        def fake_input(prompt=""):
            try:
                v = next(it)
            except StopIteration:
                raise AssertionError(f"script asked for more input than scripted; last prompt: {prompt!r}")
            if isinstance(v, BaseException):
                raise v
            return v
        monkeypatch.setattr(builtins, "input", fake_input)
        mod = _script()
        mod.GroqLLM = llm
        monkeypatch.setattr(sys, "argv", ["make_testset.py", "--docs", *docs, "--out", str(out), *extra])
        mod.main()
        return json.loads(out.read_text()) if out.exists() else [], capsys.readouterr().out
    return _run


def test_accept_edit_skip_then_quit_saves_exactly_the_accepted_items(run):
    items, out = run(["a", "e", "A better standalone question?", "better answer", "s", "q"], ["--n", "10", "--unanswerable", "0"])
    assert len(items) == 2 and all(i["verified"] and i["answerable"] and i["gold_substrings"] for i in items)
    assert items[1]["ground_truth"] == "better answer" and items[1]["question"] == "A better standalone question?"
    assert items[0]["ground_truth"] == "an answer"
    assert "Progress so far is saved" in out and "2 verified answerable" in out


def test_stray_text_re_asks_instead_of_skipping_a_draft(run):
    items, out = run(["zzz", "What is the refrigeration problem?", "a", "q"], ["--n", "10", "--unanswerable", "0"])
    assert len(items) == 1                              # the first draft was accepted, not silently skipped
    assert out.count("Type one of") == 2


def test_ctrl_c_saves_and_exits_without_a_traceback(run):
    items, out = run(["a", KeyboardInterrupt()], ["--n", "10", "--unanswerable", "0"])
    assert len(items) == 1 and "Progress so far is saved" in out


def test_unanswerable_stage_shows_evidence_and_only_saves_confirmed(run):
    items, out = run(["y", "n", "q"], ["--n", "0", "--unanswerable", "3"])
    assert [i["answerable"] for i in items] == [False] and items[0]["verified"] is True
    assert "Closest passages in the whole corpus" in out and out.count("UNANSWERABLE?") == 3


def test_duplicate_unanswerable_drafts_are_not_asked_twice(run):
    FakeLLM.same_unanswerable = True
    items, out = run(["y"], ["--n", "0", "--unanswerable", "4"])
    assert len(items) == 1 and out.count("UNANSWERABLE?") == 1


def test_resuming_appends_and_never_repeats_a_question(run):
    first, _ = run(["a", "q"], ["--n", "10", "--unanswerable", "0"])
    second, _ = run(["a", "q"], ["--n", "10", "--unanswerable", "0", "--seed", "2"])
    qs = [i["question"] for i in second]
    assert len(second) == 2 and len(set(qs)) == 2 and second[0]["id"] == "q001" and second[1]["id"] == "q002"


PROSE = ("The meta-agent receives training trajectories while validation instructions and scores remain hidden from it. "
         "Validation performance guides parent selection during evolution. We evaluate the final selected repository on the full "
         "benchmark. The approach limits specialization to the tasks that motivated these changes. Descendants can add, revise or "
         "remove interfaces and knowledge, and the population retains alternative designs for further recombination.")


def test_select_chunks_excludes_appendix_pages_and_tables():
    mod = _script()
    mk = lambda text, page: Chunk(text, "d", "s.pdf", 0, metadata={"page": page} if page else {})
    chunks = [mk(PROSE, 5), mk(PROSE, 41), mk(PROSE, None), mk("Seed 163.2 71.4 55.0 " * 20, 6)]
    assert len(mod.select_chunks(chunks)) == 3                       # the table chunk is dropped
    kept = mod.select_chunks(chunks, max_page=30)
    assert [c.metadata.get("page") for c in kept] == [5, None]       # appendix page dropped, unpaged kept
    assert len(mod.select_chunks(chunks, include_tables=True)) == 4


def test_too_similar_flags_rephrasings_but_not_different_questions():
    mod = _script()
    seen = ["What specific criteria determine the priority order for handling interruption episodes?"]
    assert mod.too_similar("What specific criteria are used to determine the priority of interruption episodes?", seen)
    assert not mod.too_similar("Who designed the calibration procedure for the cooling loop?", seen)


def test_edit_keeps_question_and_answer_when_enter_is_pressed(run):
    items, _ = run(["e", "", "", "q"], ["--n", "10", "--unanswerable", "0"])
    assert len(items) == 1 and items[0]["ground_truth"] == "an answer" and items[0]["question"].endswith("?")


@pytest.mark.parametrize("q, expected", [
    ("According to the passage, what three components are included in the revised function?", True),
    ("What does this paragraph say about latency?", True),
    ("In the following excerpt, who is the reviewer?", True),
    ("According to the text, which loop is stronger?", True),
    ("What three components does the revised attractor function include?", False),
    ("How does the meta-agent avoid specializing to the benchmark tasks?", False),
    ("What passage of time separates the two measurements?", False),
])
def test_questions_that_refer_to_their_source_passage_are_rejected(q, expected):
    assert _script().refers_to_passage(q) is expected


def test_drafts_referring_to_the_passage_are_never_shown(run):
    class Leaky(FakeLLM):
        def complete(self, messages, **kw):
            text = messages[0]["content"]
            if "Passage:" in text:
                passage = text.split("Passage:\n", 1)[1]
                FakeLLM.calls += 1
                n = FakeLLM.calls
                q = f"According to the passage, what is {WORDS[n % 20]} {WORDS[(n + 3) % 20]}?" if n % 2 else \
                    f"What is {WORDS[n % 20]} {WORDS[(n + 3) % 20]} {WORDS[(n + 9) % 20]}?"
                return json.dumps({"question": q, "ground_truth": "x", "gold_span": passage[:50]})
            return super().complete(messages, **kw)
    items, out = run(["a", "a", "q"], ["--n", "12", "--unanswerable", "0"], llm=Leaky)
    assert items and not any("the passage" in i["question"].lower() for i in items)
    assert "According to the passage" not in out
