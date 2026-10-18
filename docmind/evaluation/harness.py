"""
RAGAS evaluation harness, configured to use Groq as the judge.

Why this file looks the way it does:
  * RAGAS falls back to its own default judge provider, which needs a different API key.
    We pass a Groq judge (ChatGroq) and a local embedding model explicitly, otherwise
    evaluate() fails on that missing key.
  * Judge model is NOT the generator model (self-preference bias). Default judge:
    llama-3.3-70b-versatile. Generator: llama-3.1-8b-instant.
  * Groq rejects n>1, which RAGAS answer-relevancy requests. We set bypass_n=True
    and strictness=1.
  * Groq's free tier is 12K TPM and 100K tokens/day for the 70B judge. A full run
    can exceed the daily cap, so BOTH phases checkpoint to JSONL and resume:
        generate  -> reports/generations.jsonl   (agent answers; resumable)
        score     -> reports/ragas_scores.jsonl  (per-sample RAGAS scores; resumable)
  * Abstained answers are excluded from RAGAS (faithfulness of "I could not find
    this" is meaningless) and reported separately as abstention behaviour.
  * result.to_pandas() is used instead of result["metric"], whose meaning changed
    between RAGAS versions.
"""
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

from docmind.config import CONFIG
from docmind.evaluation.metrics import answer_level_stats, bootstrap_ci
from docmind.evaluation.retrieval_eval import load_testset
from docmind.llm.groq_client import RateLimitExhausted

logger = logging.getLogger(__name__)

METRIC_COLUMNS = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "context_precision": "llm_context_precision_with_reference",
    "context_recall": "context_recall",
}


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl(path: Path, rows: List[Dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------- phase 1
def generate(testset_path: str, agent=None, out_path: Optional[str] = None,
             file_ids: Optional[List[str]] = None, limit: Optional[int] = None,
             verified_only: bool = True) -> List[Dict]:
    """Run the agent over the testset. Resumable: finished ids are skipped."""
    from docmind.agent.react import ReActAgent
    agent = agent or ReActAgent()
    out = Path(out_path or CONFIG.reports_dir / "generations.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    testset = load_testset(testset_path, verified_only=verified_only)[: limit or None]
    done = {r["id"] for r in _read_jsonl(out)}
    for item in testset:
        if item["id"] in done:
            continue
        try:
            resp = agent.run(item["question"], file_ids=file_ids)
        except RateLimitExhausted:
            raise            # daily cap: stop now; everything so far is already checkpointed
        except Exception as exc:
            logger.warning(f"{item['id']} failed: {exc}")
            continue
        _append_jsonl(out, [{
            "id": item["id"], "question": item["question"],
            "ground_truth": item.get("ground_truth", ""),
            "answerable": item["answerable"],
            "answer": resp.answer,
            "contexts": [s.chunk.content for s in resp.sources],
            "abstained": resp.abstained, "finish_reason": resp.finish_reason,
            "citation_report": resp.citation_report, "needs_review": resp.needs_review,
            "review_reasons": resp.review_reasons, "latency_ms": resp.latency_ms,
            "llm_calls": resp.llm_calls, "timings": resp.timings,
        }])
        logger.info(f"generated {item['id']} ({len(done) + 1}/{len(testset)})")
        done.add(item["id"])
    return _read_jsonl(out)


# --------------------------------------------------------------------- phase 2
def build_judge():
    """Groq judge LLM + local embeddings, wrapped for RAGAS."""
    from langchain_core.embeddings import Embeddings
    from langchain_groq import ChatGroq
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig

    from docmind.embeddings import embed_texts

    if not CONFIG.llm.groq_api_key:
        raise ValueError("GROQ_API_KEY not set; it is needed for the RAGAS judge.")

    class _LocalEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return embed_texts(list(texts)).tolist()

        def embed_query(self, text):
            return embed_texts([text]).tolist()[0]

    run_config = RunConfig(timeout=CONFIG.eval.judge_timeout, max_retries=CONFIG.eval.judge_max_retries,
                           max_workers=CONFIG.eval.judge_max_workers)
    chat = ChatGroq(model=CONFIG.eval.judge_model, temperature=0.0,
                    api_key=CONFIG.llm.groq_api_key, max_retries=6)
    return (LangchainLLMWrapper(chat, run_config=run_config, bypass_n=True),
            LangchainEmbeddingsWrapper(_LocalEmbeddings()), run_config)


def score(records: List[Dict], out_path: Optional[str] = None, judge=None,
          metrics: Optional[List[str]] = None, batch_size: Optional[int] = None) -> List[Dict]:
    """Score answered, answerable records with RAGAS. Checkpoints after every batch."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    llm, emb, run_config = judge or build_judge()
    wanted = set(metrics or METRIC_COLUMNS)
    objs = []
    if "faithfulness" in wanted:
        objs.append(Faithfulness())
    if "answer_relevancy" in wanted:
        objs.append(ResponseRelevancy(strictness=1))
    if "context_precision" in wanted:
        objs.append(LLMContextPrecisionWithReference())
    if "context_recall" in wanted:
        objs.append(LLMContextRecall())

    out = Path(out_path or CONFIG.reports_dir / "ragas_scores.jsonl")
    done = {r["id"] for r in _read_jsonl(out)}
    todo = [r for r in records
            if r["id"] not in done and r["answerable"] and not r["abstained"] and r["contexts"] and r["ground_truth"]]
    bs = batch_size or CONFIG.eval.score_batch_size
    logger.info(f"RAGAS scoring {len(todo)} samples (already done: {len(done)}), judge={CONFIG.eval.judge_model}")

    for i in range(0, len(todo), bs):
        batch = todo[i:i + bs]
        ds = EvaluationDataset.from_list([{
            "user_input": r["question"], "response": r["answer"],
            "retrieved_contexts": r["contexts"], "reference": r["ground_truth"],
        } for r in batch])
        t = time.time()
        result = evaluate(ds, metrics=objs, llm=llm, embeddings=emb, run_config=run_config,
                          raise_exceptions=False, show_progress=False)
        df = result.to_pandas()
        rows = []
        for r, (_, row) in zip(batch, df.iterrows()):
            entry = {"id": r["id"]}
            for short, col in METRIC_COLUMNS.items():
                if col in df.columns:
                    v = row[col]
                    entry[short] = None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
            if all(entry.get(k) is None for k in METRIC_COLUMNS):
                # Every metric failed (typically 429s). Do NOT checkpoint, so the next run retries it.
                logger.warning(f"{r['id']}: all metrics failed (rate limit?); will retry on next run")
                continue
            rows.append(entry)
        _append_jsonl(out, rows)
        logger.info(f"scored batch {i // bs + 1}/{math.ceil(len(todo) / bs)} in {time.time() - t:.0f}s")
    return _read_jsonl(out)


# -------------------------------------------------------------------- summary
def summarize(records: List[Dict], scores: List[Dict]) -> Dict:
    out: Dict = {"judge_model": CONFIG.eval.judge_model, "generator_model": CONFIG.llm.groq_model,
                 "answer_level": answer_level_stats(records), "ragas": {}}
    for short in METRIC_COLUMNS:
        vals = [s.get(short) for s in scores if s.get(short) is not None]
        mean, lo, hi = bootstrap_ci(vals, CONFIG.eval.bootstrap_resamples)
        out["ragas"][short] = {"mean": mean, "ci95": [lo, hi], "n": len(vals),
                               "n_failed": len([s for s in scores if s.get(short) is None])}
    out["n_scored"] = len(scores)
    out["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return out


def run_evaluation(testset_path: str, agent=None, output_path: Optional[str] = None,
                   skip_generate: bool = False, skip_score: bool = False, limit: Optional[int] = None,
                   file_ids: Optional[List[str]] = None, metrics: Optional[List[str]] = None,
                   verified_only: bool = True) -> Dict:
    """Both phases, then summary JSON and an MLflow run."""
    reports = CONFIG.reports_dir
    gen_path = reports / "generations.jsonl"
    records = _read_jsonl(gen_path) if skip_generate else generate(testset_path, agent, str(gen_path), file_ids, limit, verified_only)
    scores = _read_jsonl(reports / "ragas_scores.jsonl") if skip_score else score(records, metrics=metrics)
    summary = summarize(records, scores)
    out = Path(output_path or reports / "ragas_latest.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        from docmind.tracking.tracker import DocMindTracker
        flat = {f"ragas_{k}": v["mean"] for k, v in summary["ragas"].items()}
        al = summary["answer_level"]
        for k in ("citation_validity", "false_abstention_rate", "correct_abstention_rate",
                  "latency_p50_ms", "latency_p95_ms", "review_rate"):
            if al.get(k) is not None:
                flat[k] = al[k]
        DocMindTracker().log_evaluation(
            flat, params={"judge": CONFIG.eval.judge_model, "generator": CONFIG.llm.groq_model,
                          "chunker": CONFIG.chunker.strategy, "top_k": CONFIG.retriever.top_k_rerank,
                          "n_questions": len(records)},
            artifact_paths=[out])
    except Exception as exc:  # tracking must never fail the eval
        logger.warning(f"MLflow skipped: {exc}")
    return summary
