"""
Consolidate every measurement into one report: reports/metrics_latest.json and reports/METRICS.md.

  python scripts/record_metrics.py                       # collect what exists
  python scripts/record_metrics.py --invited 3           # also compute pilot adoption
  python scripts/record_metrics.py --run-tests           # also record the offline test result

Nothing is invented: a section whose evidence file is missing says NOT RUN and names the command
that produces it. Inputs: reports/stack_check.json, retrieval_latest.json, ragas_latest.json,
generations.jsonl, the corpus database, and the usage-events table.
"""
import argparse
import json
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docmind.config import CONFIG  # noqa: E402
from docmind.evaluation.metrics import answer_level_stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = CONFIG.reports_dir


def _load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _jsonl(name):
    p = R / name
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def corpus():
    try:
        from docmind.store import CorpusStore
        store = CorpusStore()
        docs = store.list_documents()
        chunks, matrix = store.load_corpus()
    except Exception as exc:
        return {"error": str(exc)}
    if not chunks:
        return None
    L = sorted(len(c.content) for c in chunks)
    n = len(L)
    return {"documents": len(docs), "chunks": n, "embedding_dim": int(matrix.shape[1]),
            "chunk_chars": {"min": L[0], "p10": L[n // 10], "median": L[n // 2], "p90": L[(9 * n) // 10], "max": L[-1]},
            "under_min_chunk_size": sum(1 for x in L if x < CONFIG.chunker.min_chunk_size),
            "per_file": {d["filename"]: d["n_chunks"] for d in docs}}


def pilot(invited):
    try:
        from docmind.review.store import ReviewStore
        from docmind.telemetry.events import EventLog
        ev = EventLog()
        q, fb = ev.rows("query"), ev.rows("feedback")
        reviews = ReviewStore().stats()
    except Exception as exc:
        return {"error": str(exc)}
    if not q:
        return None
    users = {r["user"] for r in q if r["user"] != "anonymous"}
    days = {}
    for r in q:
        days.setdefault(r["user"], set()).add(int(r["ts"] // 86400))
    lat = sorted(r["payload"].get("latency_ms", 0) for r in q)
    out = {"queries": len(q), "named_users": len(users),
           "repeat_users": sum(1 for u, d in days.items() if u != "anonymous" and len(d) >= 2),
           "abstained": sum(1 for r in q if r["payload"].get("abstained")),
           "sent_to_review": sum(1 for r in q if r["payload"].get("needs_review")),
           "latency_ms_p50": lat[len(lat) // 2], "latency_ms_p95": lat[max(int(len(lat) * 0.95) - 1, 0)],
           "feedback": len(fb), "helpful": sum(1 for f in fb if f["payload"].get("helpful")), "review_queue": reviews}
    if invited:
        out["invited"] = invited
        out["adoption_rate"] = len(users) / invited
    return out


def tests():
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--ignore=tests/test_ragas_plumbing.py", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    m = re.search(r"(\d+) passed", tail)
    f = re.search(r"(\d+) failed", tail)
    return {"summary": tail, "passed": int(m.group(1)) if m else 0, "failed": int(f.group(1)) if f else 0,
            "seconds": round(time.time() - t0, 1), "note": "offline stand-ins; says nothing about the real models"}


def build(args):
    gens = _jsonl("generations.jsonl")
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "stack": _load("stack_check.json"),
        "corpus": corpus(),
        "retrieval": _load("retrieval_latest.json"),
        "ragas": _load("ragas_latest.json"),
        "answers": answer_level_stats(gens) if gens else None,
        "pilot": pilot(args.invited),
        "tests": tests() if args.run_tests else None,
    }


def _nr(cmd):
    return f"NOT RUN. Command: `{cmd}`"


def _ci(m):
    return "n/a" if not m or m.get("mean") != m.get("mean") else f"{m['mean']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}] n={m['n']}"


def render(d):
    L = [f"# DocMind metrics", "", f"Generated {d['generated_at']}. Every number below comes from a file in `reports/`, the database, or the events table. Sections without evidence say NOT RUN.", ""]
    s = d["stack"]
    L += ["## 1. Stack and environment", ""]
    if not s:
        L += [_nr("python scripts/stack_check.py"), ""]
    else:
        L += [f"- Real stack verified: **{s['real_stack']}** (scope: {s['scope']}); failed checks: {s['failed'] or 'none'}",
              f"- Commit {s['git_commit']} (dirty: {s['git_dirty']}), Python {s['python']}, {s['platform']}, {s['cpu_count']} CPUs",
              f"- Versions: " + ", ".join(f"{k} {v}" for k, v in s["versions"].items() if v and k != "torch_cuda_available"),
              f"- CUDA available: {s['versions'].get('torch_cuda_available')}", "", "| Check | Status | Detail | Seconds |", "|---|---|---|---|"]
        L += [f"| {c['name']} | {c['status']} | {c['detail'][:110]} | {c['seconds']} |" for c in s["checks"]]
        L.append("")
    L += ["## 2. Corpus", ""]
    c = d["corpus"]
    if not c:
        L += [_nr("python scripts/demo.py --doc <file>  (or ingest through the app)"), ""]
    elif "error" in c:
        L += [f"Unavailable: {c['error']}", ""]
    else:
        cc = c["chunk_chars"]
        L += [f"- {c['documents']} documents, {c['chunks']} chunks, embedding dim {c['embedding_dim']}",
              f"- Chunk length (chars): min {cc['min']}, p10 {cc['p10']}, median {cc['median']}, p90 {cc['p90']}, max {cc['max']}",
              f"- Chunks under the configured minimum: {c['under_min_chunk_size']}", ""]
    L += ["## 3. Retrieval ablation", ""]
    r = d["retrieval"]
    if not r:
        L += [_nr("python scripts/ablation.py --testset data/testset.json --docs <files> --chunkers"), ""]
    else:
        k = r["k"]
        L += [f"n={r['n_questions']} questions, k={k}. Mean [95% bootstrap CI].", "", f"| Mode | Precision@{k} | Hit@{k} | MRR | nDCG@{k} |", "|---|---|---|---|---|"]
        for mode, m in r["modes"].items():
            L.append(f"| {mode} | " + " | ".join(f"{m[x]['mean']:.3f} [{m[x]['lo']:.3f}, {m[x]['hi']:.3f}]" for x in ("precision", "hit", "mrr", "ndcg")) + " |")
        for key, label in (("rerank_vs_dense_precision", "Rerank vs dense"), ("rerank_vs_hybrid_precision", "Rerank vs hybrid"), ("hybrid_vs_dense_precision", "Hybrid vs dense")):
            v = r.get(key)
            if v and v.get("relative") is not None:
                ci = f", 95% CI {v['lo'] * 100:+.0f}% to {v['hi'] * 100:+.0f}%" if v.get("lo") is not None else ""
                L.append(f"\n- {label}: precision {v['relative'] * 100:+.1f}%{ci} (paired bootstrap, n={v['n']})")
        t = r.get("suggested_min_rerank_score")
        if t and t["threshold"] == t["threshold"]:
            L.append(f"- Suggested DOCMIND_MIN_RERANK_SCORE = {t['threshold']:.2f} (best F1 {t['f1']:.2f})")
        L.append("")
    L += ["## 4. Answer quality (RAGAS, Groq judge)", ""]
    g = d["ragas"]
    if not g:
        L += [_nr("python scripts/evaluate.py --testset data/testset.json --docs <files>"), ""]
    else:
        L += [f"Generator `{g['generator_model']}`, judge `{g['judge_model']}`, run {g['generated_at']}.", "", "| Metric | Mean [95% CI] n |", "|---|---|"]
        L += [f"| {k} | {_ci(v)} (failed: {v['n_failed']}) |" for k, v in g["ragas"].items()]
        L.append("")
    L += ["## 5. Answer behaviour, latency and token use", ""]
    a = d["answers"]
    if not a:
        L += [_nr("python scripts/evaluate.py ...  (writes reports/generations.jsonl)"), ""]
    else:
        f = lambda x, p=3: "n/a" if x is None else (f"{x:.{p}f}")
        L += [f"- Questions: {a['n_total']} ({a['n_answerable']} answerable, {a['n_unanswerable']} unanswerable)",
              f"- Citation validity: {f(a['citation_validity'])} over {a['citations_total']} citations; answers with an invalid citation: {a['answers_with_invalid_citation']}; with no citation: {a['answers_without_any_citation']}",
              f"- Correct abstention on unanswerable: {f(a['correct_abstention_rate'])}; unsupported answers on unanswerable: {f(a['hallucinated_answer_rate_on_unanswerable'])}; false abstention on answerable: {f(a['false_abstention_rate'])}",
              f"- Latency p50 / p95: {a['latency_p50_ms'] / 1000:.1f}s / {a['latency_p95_ms'] / 1000:.1f}s; mean LLM calls {f(a['mean_llm_calls'], 2)}",
              f"- Tokens per answer: prompt {f(a.get('mean_prompt_tokens'), 0)}, completion {f(a.get('mean_completion_tokens'), 0)}; total {f(a.get('total_tokens'), 0)}",
              f"- Finish reasons: {a.get('finish_reasons')}; share sent to human review: {f(a['review_rate'])}", ""]
    L += ["## 6. Pilot and human review", ""]
    p = d["pilot"]
    if not p:
        L += [_nr("use the app with a name entered in the sidebar, then re-run with --invited N"), ""]
    elif "error" in p:
        L += [f"Unavailable: {p['error']}", ""]
    else:
        L += [f"- {p['queries']} queries from {p['named_users']} named users ({p['repeat_users']} on 2+ days)"
              + (f"; adoption {p['adoption_rate']:.0%} of {p['invited']} invited" if "invited" in p else ""),
              f"- Abstained {p['abstained']}, sent to review {p['sent_to_review']}; latency p50 {p['latency_ms_p50'] / 1000:.1f}s, p95 {p['latency_ms_p95'] / 1000:.1f}s",
              f"- Feedback: {p['helpful']} helpful of {p['feedback']}; review queue: {p['review_queue']}", "",
              "Hours saved come from the stopwatch baseline: `python scripts/pilot_report.py`.", ""]
    L += ["## 7. Offline tests", ""]
    t = d["tests"]
    L += [_nr("python scripts/record_metrics.py --run-tests"), ""] if not t else \
         [f"- {t['summary']} ({t['seconds']} s). {t['note']}", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--invited", type=int)
    ap.add_argument("--run-tests", action="store_true")
    args = ap.parse_args()
    d = build(args)
    R.mkdir(parents=True, exist_ok=True)
    (R / "metrics_latest.json").write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
    (R / "METRICS.md").write_text(render(d), encoding="utf-8")
    print((R / "METRICS.md").read_text(encoding="utf-8"))
    print(f"\nWritten: {R / 'METRICS.md'} and {R / 'metrics_latest.json'}")


if __name__ == "__main__":
    main()
