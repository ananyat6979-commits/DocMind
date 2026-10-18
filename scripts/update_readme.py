"""
Write real measured numbers into README.md between marker comments.
Nothing is invented: if a report file is missing, the block says 'not yet run'.

  python scripts/update_readme.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _fmt(m):
    if m is None or m.get("mean") != m.get("mean"):
        return "n/a"
    return f"{m['mean']:.3f} (95% CI {m['ci95'][0]:.3f} to {m['ci95'][1]:.3f}, n={m['n']})"


def ragas_block():
    p = ROOT / "reports" / "ragas_latest.json"
    if not p.exists():
        return "Not yet run. Execute `python scripts/evaluate.py --testset data/testset.json`."
    s = json.loads(p.read_text())
    a = s["answer_level"]
    rows = [f"Generator `{s['generator_model']}`, judge `{s['judge_model']}`, run {s['generated_at']}, "
            f"{a['n_total']} questions ({a['n_answerable']} answerable, {a['n_unanswerable']} unanswerable).", "",
            "| Metric | Result |", "|---|---|"]
    for k, v in s["ragas"].items():
        rows.append(f"| RAGAS {k} | {_fmt(v)} |")
    cv = a.get("citation_validity")
    rows += [f"| Citation validity (valid / cited) | {'n/a' if cv is None else f'{cv:.3f}'} over {a['citations_total']} citations |",
             f"| Answers with at least one invalid citation | {a['answers_with_invalid_citation']} |",
             f"| Correct abstention on unanswerable | {a['correct_abstention_rate']} |",
             f"| False abstention on answerable | {a['false_abstention_rate']} |",
             f"| Latency p50 / p95 | {a['latency_p50_ms'] / 1000:.1f}s / {a['latency_p95_ms'] / 1000:.1f}s |"]
    return "\n".join(rows)


def stack_block():
    p = ROOT / "reports" / "stack_check.json"
    if not p.exists():
        return "Not yet run. Execute `python scripts/stack_check.py`."
    s = json.loads(p.read_text())
    v = s["versions"]
    rows = [f"Real stack verified: **{s['real_stack']}** ({s['scope']}), commit `{s['git_commit']}`, {s['generated_at']}.", "",
            "| Component | Result |", "|---|---|"]
    rows += [f"| {c['name']} | {c['status']}: {c['detail'][:100]} |" for c in s["checks"]]
    rows += ["", f"torch {v.get('torch')}, sentence-transformers {v.get('sentence-transformers')}, faiss-cpu {v.get('faiss-cpu')}, "
             f"ragas {v.get('ragas')}, Python {s['python']}. Full report: `reports/METRICS.md`."]
    return "\n".join(rows)


def retrieval_block():
    p = ROOT / "reports" / "retrieval_latest.md"
    return p.read_text() if p.exists() else "Not yet run. Execute `python scripts/ablation.py --testset data/testset.json --docs <your docs>`."


def replace(text, tag, body):
    a, b = f"<!-- {tag}:START -->", f"<!-- {tag}:END -->"
    if a not in text or b not in text:
        sys.exit(f"Marker {tag} missing in README.md")
    head, rest = text.split(a, 1)
    _, tail = rest.split(b, 1)
    return f"{head}{a}\n{body}\n{b}{tail}"


def main():
    t = README.read_text(encoding="utf-8")
    if "<!-- STACK:START -->" in t:
        t = replace(t, "STACK", stack_block())
    t = replace(t, "RAGAS", ragas_block())
    t = replace(t, "RETRIEVAL", retrieval_block())
    README.write_text(t, encoding="utf-8")
    print("README.md updated from reports/")


if __name__ == "__main__":
    main()
