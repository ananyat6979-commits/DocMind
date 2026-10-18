"""
Verify that DocMind runs on the REAL stack, and record measurements.

The offline test suite deliberately uses stand-ins (hash embedder, lexical reranker, demo LLM) so it
needs no downloads or keys. That means a green test run says nothing about the real models. This
script proves each real component loads and behaves, then writes reports/stack_check.json.

  python scripts/stack_check.py               full check (needs GROQ_API_KEY; models download once)
  python scripts/stack_check.py --skip-llm    embeddings, reranker and retrieval only (no key needed)
  python scripts/stack_check.py --allow-stand-ins   plumbing check with offline stand-ins (never "real")

Exit code 0 only if no check failed. "real_stack": true in the JSON requires every check to pass on
the real components with nothing skipped (except the LLM checks under --skip-llm, which marks the
result as partial).
"""
import argparse
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from docmind.config import CONFIG  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["torch", "sentence-transformers", "faiss-cpu", "numpy", "ragas", "langchain-groq", "groq",
            "streamlit", "fastapi", "pdfplumber", "python-docx", "nltk"]

SAMPLE_DOC = (
    "The Alpha Protocol governs thermal regulation in the reactor. Coolant temperature must stay below 300 "
    "degrees during startup. Operators verify the coolant pump every shift. Emergency shutdown is triggered "
    "when the pressure exceeds 40 bar. The shutdown valve closes within two seconds of the trigger. "
    "The cafeteria serves soup on Mondays and Thursdays in the main hall. Vegetarian options include lentil "
    "curry and mushroom risotto. Allergen labels are printed on every tray card."
)
SENTENCES = [f"Sentence {i}: the coolant pump pressure and temperature are checked every shift." for i in range(64)]


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def check(results, name, fn, required=True):
    t0 = time.perf_counter()
    try:
        status, detail, metrics = fn()
    except Exception as exc:
        status, detail, metrics = "fail", f"{type(exc).__name__}: {exc}", {"traceback": traceback.format_exc()[-800:]}
    results.append({"name": name, "status": status, "detail": detail, "metrics": metrics,
                    "seconds": round(time.perf_counter() - t0, 2), "required": required})
    print(f"[{status.upper():4}] {name}: {detail}")


def c_mode(args):
    e, r, l = CONFIG.embedding, CONFIG.retriever, CONFIG.llm
    problems = []
    if e.backend != "sentence-transformers":
        problems.append(f"embedding backend is '{e.backend}' (offline stand-in). Unset DOCMIND_TEST_MODE")
    if r.reranker_model == "lexical":
        problems.append("reranker is the lexical stand-in. Unset DOCMIND_TEST_MODE")
    if not r.use_reranker:
        problems.append("reranker disabled by DOCMIND_USE_RERANKER=0")
    if not args.skip_llm and l.provider != "groq":
        problems.append(f"LLM provider is '{l.provider}' (demo stand-in). Set GROQ_API_KEY and unset DOCMIND_TEST_MODE")
    cfg = {"embedding_model": e.model_name, "embedding_backend": e.backend, "reranker_model": r.reranker_model,
           "use_reranker": r.use_reranker, "llm_provider": l.provider, "llm_model": l.groq_model,
           "judge_model": CONFIG.eval.judge_model, "chunker": CONFIG.chunker.strategy,
           "top_k_rerank": r.top_k_rerank, "device_setting": e.device, "test_mode": os.getenv("DOCMIND_TEST_MODE") == "1"}
    if problems:
        return ("warn" if args.allow_stand_ins else "fail"), "; ".join(problems), cfg
    return "pass", "real embedder, real reranker" + ("" if args.skip_llm else ", Groq LLM"), cfg


def c_versions(_):
    v = {}
    for p in PACKAGES:
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = None
    try:
        import torch
        v["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        v["torch_cuda_available"] = None
    missing = [k for k in ("sentence-transformers", "faiss-cpu", "torch") if not v.get(k)]
    return ("fail" if missing else "pass"), (f"missing: {missing}" if missing else "core packages present"), v


def c_embedder(_):
    from docmind.embeddings import embed_texts, get_embedding_model
    t0 = time.perf_counter(); get_embedding_model(); load_s = time.perf_counter() - t0
    t0 = time.perf_counter(); a = embed_texts(SENTENCES); dt = time.perf_counter() - t0
    b = embed_texts(SENTENCES)
    norms = np.linalg.norm(a, axis=1)
    pair = embed_texts(["Coolant must stay below 300 degrees.", "The coolant temperature should not exceed 300 degrees.",
                        "The cafeteria serves soup on Mondays."])
    sim_rel, sim_irr = float(pair[0] @ pair[1]), float(pair[0] @ pair[2])
    m = {"dim": int(a.shape[1]), "load_seconds": round(load_s, 2), "sentences_per_second": round(len(SENTENCES) / dt, 1),
         "norm_min": float(norms.min()), "norm_max": float(norms.max()), "sim_related": round(sim_rel, 3),
         "sim_unrelated": round(sim_irr, 3), "device": CONFIG.embedding.device}
    errs = []
    if "bge-small" in CONFIG.embedding.model_name and a.shape[1] != 384:
        errs.append(f"dim {a.shape[1]} != 384")
    if not (abs(norms.min() - 1) < 1e-3 and abs(norms.max() - 1) < 1e-3):
        errs.append("vectors not L2-normalised")
    if not np.allclose(a, b, atol=1e-5):
        errs.append("non-deterministic encodings")
    if not sim_rel > sim_irr:
        errs.append("related pair not closer than unrelated pair")
    return ("fail" if errs else "pass"), ("; ".join(errs) or f"dim={a.shape[1]}, {m['sentences_per_second']} sentences/s"), m


def c_reranker(_):
    if not CONFIG.retriever.use_reranker:
        return "skip", "reranker disabled", {}
    from docmind.models import Chunk, SearchResult
    from docmind.retrieval.reranker import CrossEncoderReranker
    q = "What is the coolant temperature limit during startup?"
    good = SearchResult(Chunk("Coolant temperature must stay below 300 degrees during startup.", "d", "s", 0), 0.0, "hybrid")
    bad = SearchResult(Chunk("The cafeteria serves soup on Mondays and Thursdays.", "d", "s", 1), 0.0, "hybrid")
    rr = CrossEncoderReranker()
    rr.rerank(q, [good, bad], top_k=2)                      # warm-up (model load)
    t0 = time.perf_counter(); out = rr.rerank(q, [good, bad], top_k=2); dt = time.perf_counter() - t0
    scores = {r.chunk.content[:20]: round(r.score, 3) for r in out}
    ok = out[0].chunk is good.chunk and out[0].score > out[1].score
    return ("pass" if ok else "fail"), (f"relevant passage ranked first ({out[0].score:.2f} vs {out[1].score:.2f})" if ok
                                         else "relevant passage not ranked first"), {"scores": scores, "latency_ms_2_pairs": round(dt * 1000, 1)}


def _build_store(tmp):
    from docmind.ingestion.pipeline import ingest
    from docmind.store import CorpusStore
    doc = Path(tmp) / "sample.txt"; doc.write_text(SAMPLE_DOC)
    store = CorpusStore(db_path=Path(tmp) / "check.db")
    rep = ingest(doc, store=store)
    return store, rep


def c_retrieval(_):
    from docmind.retrieval.hybrid import HybridRetriever
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter(); store, rep = _build_store(tmp); ingest_s = time.perf_counter() - t0
        chunks, matrix = store.load_corpus()
        r = HybridRetriever(store)
        cases = [("coolant temperature during startup", "300 degrees"), ("when does emergency shutdown trigger", "40 bar"),
                 ("vegetarian options", "lentil")]
        hits = []
        for q, needle in cases:
            top = r.search(q, top_k=3)
            hits.append(bool(top) and needle in top[0].chunk.content)
        m = {"chunks": len(chunks), "embedding_dim": int(matrix.shape[1]), "ingest_seconds": round(ingest_s, 2),
             "hit_at_1": sum(hits) / len(hits)}
    return ("pass" if all(hits) else "fail"), f"ingest ok, {len(chunks)} chunks, top-1 correct for {sum(hits)}/{len(hits)} queries", m


def c_llm(args):
    if args.skip_llm:
        return "skip", "--skip-llm", {}
    if CONFIG.llm.provider != "groq":
        return "skip", f"provider is '{CONFIG.llm.provider}'", {}
    from docmind.llm.groq_client import get_llm
    llm = get_llm()
    t0 = time.perf_counter(); out = llm.complete([{"role": "user", "content": "Reply with exactly the word: OK"}], max_tokens=8)
    dt = time.perf_counter() - t0
    m = {"model": CONFIG.llm.groq_model, "latency_ms": round(dt * 1000), "reply": out.strip()[:40],
         "prompt_tokens": getattr(llm, "prompt_tokens", None), "completion_tokens": getattr(llm, "completion_tokens", None)}
    return ("pass" if "ok" in out.lower() else "fail"), f"{CONFIG.llm.groq_model} replied in {dt * 1000:.0f} ms", m


def c_judge(args):
    if args.skip_llm:
        return "skip", "--skip-llm", {}
    if not CONFIG.llm.groq_api_key:
        return "skip", "no GROQ_API_KEY", {}
    try:
        import langchain_groq, ragas  # noqa: F401
    except ImportError as exc:
        return "fail", f"{exc}. Run: pip install -r requirements-eval.txt", {}
    from docmind.llm.groq_client import GroqLLM
    j = GroqLLM(model=CONFIG.eval.judge_model)
    t0 = time.perf_counter(); out = j.complete([{"role": "user", "content": "Reply with exactly the word: OK"}], max_tokens=8)
    dt = time.perf_counter() - t0
    return ("pass" if "ok" in out.lower() else "fail"), f"judge {CONFIG.eval.judge_model} replied in {dt * 1000:.0f} ms", \
        {"model": CONFIG.eval.judge_model, "latency_ms": round(dt * 1000)}


def c_e2e(args):
    if args.skip_llm:
        return "skip", "--skip-llm", {}
    from docmind.agent.react import ReActAgent
    from docmind.llm.groq_client import get_llm
    from docmind.retrieval.hybrid import HybridRetriever
    with tempfile.TemporaryDirectory() as tmp:
        store, _ = _build_store(tmp)
        resp = ReActAgent(retriever=HybridRetriever(store), llm=get_llm()).run("What is the coolant temperature limit during startup?")
    rep = resp.citation_report or {}
    ok = (not resp.abstained) and bool(rep.get("valid")) and not rep.get("invalid") and "300" in resp.answer
    m = {"latency_ms": round(resp.latency_ms), "llm_calls": resp.llm_calls, "finish_reason": resp.finish_reason,
         "citation_validity": rep.get("validity"), "timings": {k: round(v, 1) for k, v in resp.timings.items()},
         "answer_preview": resp.answer[:160]}
    return ("pass" if ok else "fail"), (f"answered with {len(rep.get('valid', []))} verified citation(s) in {resp.latency_ms:.0f} ms"
                                         if ok else f"answer not grounded/verified: {resp.answer[:120]!r}"), m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--allow-stand-ins", action="store_true")
    ap.add_argument("--out", default=str(CONFIG.reports_dir / "stack_check.json"))
    args = ap.parse_args()

    results = []
    check(results, "mode", lambda: c_mode(args))
    check(results, "versions", lambda: c_versions(args))
    check(results, "embedder", lambda: c_embedder(args))
    check(results, "reranker", lambda: c_reranker(args))
    check(results, "retrieval_smoke", lambda: c_retrieval(args))
    check(results, "llm", lambda: c_llm(args), required=not args.skip_llm)
    check(results, "judge", lambda: c_judge(args), required=not args.skip_llm)
    check(results, "end_to_end", lambda: c_e2e(args), required=not args.skip_llm)

    failed = [r["name"] for r in results if r["status"] == "fail"]
    mode_ok = next(r for r in results if r["name"] == "mode")["status"] == "pass"
    required_pass = all(r["status"] == "pass" for r in results if r["required"])
    versions = next(r for r in results if r["name"] == "versions")["metrics"]
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "git_commit": _git("rev-parse", "--short", "HEAD") or None,
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "versions": versions,
        "scope": "partial (no LLM checks)" if args.skip_llm else "full",
        "real_stack": bool(mode_ok and required_pass and not failed),
        "failed": failed, "checks": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    verdict = "REAL STACK VERIFIED" if report["real_stack"] and not args.skip_llm else \
              "REAL EMBEDDER/RERANKER VERIFIED (LLM not checked)" if report["real_stack"] else "NOT VERIFIED AS REAL STACK"
    print(f"\n{verdict}  | failed checks: {failed or 'none'} | report: {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
