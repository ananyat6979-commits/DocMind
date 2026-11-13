"""Free local HotpotQA distractor retrieval evaluation, isolated from the live corpus.

This is paragraph retrieval within ten supplied contexts, not a full-wiki or QA leaderboard score.
Human supporting-fact annotations come from HotpotQA. No labels are inferred from our chunker.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DOCMIND_HOME", str(ROOT / "reports" / "benchmark_workspace"))
import numpy as np
from docmind.evaluation.metrics import bootstrap_ci, ndcg_at_k, mrr
from docmind.evaluation.selective import calibrate, risk_curve
from docmind.retrieval.sparse import BM25Index, tokenize

DATASET = "hotpotqa/hotpot_qa"


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def acquire(directory):
    import pyarrow.parquet as pq
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "source.json"
    if lock.exists():
        manifest = json.loads(lock.read_text())
    else:
        metadata = get_json("https://huggingface.co/api/datasets/" + DATASET)
        revision = metadata["sha"]
        tree = get_json("https://huggingface.co/api/datasets/" + DATASET + "/tree/" + revision + "/distractor")
        path = next(item["path"] for item in tree if "validation" in item["path"] and item["path"].endswith(".parquet"))
        manifest = {"dataset": DATASET, "revision": revision, "path": path,
                    "license": "CC-BY-SA-4.0", "source": "https://github.com/hotpotqa/hotpot"}
    target = directory / "validation.parquet"
    if not target.exists():
        url = "https://huggingface.co/datasets/" + DATASET + "/resolve/" + manifest["revision"] + "/" + manifest["path"]
        urllib.request.urlretrieve(url, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if manifest.get("sha256") and manifest["sha256"] != digest:
        raise ValueError("Dataset hash mismatch")
    manifest["sha256"] = digest
    lock.write_text(json.dumps(manifest, indent=2))
    return pq.read_table(target).to_pylist(), manifest


def source_provenance():
    snapshot=ROOT/"source_manifest.json"
    if snapshot.exists():return json.loads(snapshot.read_text())
    import subprocess
    return {"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "dirty":bool(subprocess.check_output(["git","status","--porcelain"]))}


def ranks(scores):
    return np.argsort(-np.asarray(scores), kind="stable").tolist()


def rrf(a, b, depth):
    scores = np.zeros(len(a))
    for order in (a[:depth], b[:depth]):
        for rank, index in enumerate(order, 1):
            scores[index] += 1/(60+rank)
    return ranks(scores)


def summarize(rows):
    return {metric: dict(zip(("mean", "lo", "hi"), bootstrap_ci([r[metric] for r in rows])))
            for metric in ("recall2", "recall5", "hit5", "mrr", "ndcg5", "all_support2")}


def main():
    print("Starting CPU benchmark with bounded batches",flush=True)
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--batch-size",type=int,default=2)
    parser.add_argument("--allow-download",action="store_true")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--calibration", type=int, default=200)
    parser.add_argument("--output", default=str(ROOT / "reports" / "hotpot"))
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    data, source = acquire(out / "source")
    # Hash ordering makes sampling independent of question text, answer and labels.
    data.sort(key=lambda x: hashlib.sha256(("docmind-2026-v1:"+x["id"]).encode()).hexdigest())
    selected = data[:args.calibration+args.n]
    (out / "split_ids.json").write_text(json.dumps({"calibration": [x["id"] for x in selected[:args.calibration]],
        "heldout": [x["id"] for x in selected[args.calibration:]]}, indent=2))
    manifest = {"source": source, "n_calibration": args.calibration, "n_heldout": args.n,
        "scope": "HotpotQA validation distractor paragraph retrieval; ten contexts per question",
        "seed": "docmind-2026-v1", "python": platform.python_version(),
        "versions": {m: importlib.metadata.version(m) for m in ("torch", "sentence-transformers", "numpy")},
        "review_seconds_assumption": 60, "max_sequence_length":256, "batch_size":args.batch_size,"device":args.device, "answer_generation": "not measured"}
    records = []
    rerank_cache = {}
    cache_dir = out / "cache"
    cache_dir.mkdir(exist_ok=True)
    old_scores = out / "reranker_cache.jsonl"
    if old_scores.exists():
        for line in old_scores.read_text().splitlines():
            entry=json.loads(line)
            rerank_cache[(entry["id"],entry["headers"])]=entry["scores"]
    manifest["benchmark_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["source_functions_sha256"] = {
        str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/"docmind"/"evaluation"/"metrics.py",ROOT/"docmind"/"retrieval"/"sparse.py")}
    import subprocess
    manifest["source_snapshot"] = source_provenance()
    cache_manifest=cache_dir/"manifest.json"
    cache_key={"source":source,"split_ids":[r["id"] for r in selected],"max_seq_length":256}
    if cache_manifest.exists() and json.loads(cache_manifest.read_text())!=cache_key:
        raise ValueError("Cache provenance mismatch. Choose a fresh output directory.")
    cache_manifest.write_text(json.dumps(cache_key,indent=2))
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=args.device, local_files_only=not args.allow_download)
    reranker.max_length = 256
    manifest["reranker"]={"model":"cross-encoder/ms-marco-MiniLM-L-6-v2",
                          "revision":getattr(reranker.model.config,"_commit_hash",None)}
    for model_name in ("BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"):
        model = SentenceTransformer(model_name, device=args.device, local_files_only=not args.allow_download)
        model.max_seq_length = 256
        manifest.setdefault("models", []).append({"name": model_name,
            "revision": getattr(model[0].auto_model.config, "_commit_hash", None)})
        passages, groups = [], []
        for row in selected:
            texts = [" ".join(sentences) for sentences in row["context"]["sentences"]]
            groups.append((len(passages), len(passages)+len(texts)))
            passages.extend(texts)
        for header in (False, True):
            texts = []
            for row, (a, b) in zip(selected, groups):
                texts.extend([title+"\n"+text if header else text for title, text in zip(row["context"]["title"], passages[a:b])])
            print(f"Embedding {model_name} header={header}: {len(texts)} paragraphs", flush=True)
            cache_tag=model_name.replace("/","_")+"_headers"+str(header)
            matrix_path=cache_dir/(cache_tag+"_passages.npy")
            # Batch checkpoints keep completed embedding work after interruption.
            matrix_file=cache_dir/(cache_tag+"_partial.npy")
            progress_file=cache_dir/(cache_tag+"_progress.json")
            if matrix_path.exists():
                matrix=np.load(matrix_path)
            else:
                matrix=np.load(matrix_file) if matrix_file.exists() else None
                offset=json.loads(progress_file.read_text())["completed"] if progress_file.exists() else 0
                for block in range(offset,len(texts),64):
                    encoded=model.encode(texts[block:block+64],batch_size=args.batch_size,normalize_embeddings=True)
                    if matrix is None:
                        matrix=np.zeros((len(texts),encoded.shape[1]),dtype="float32")
                    matrix[block:block+len(encoded)]=encoded
                    np.save(matrix_file,matrix)
                    progress_file.write_text(json.dumps({"completed":block+len(encoded)}))
                    print(f"Embedded {block+len(encoded)}/{len(texts)}",flush=True)
                np.save(matrix_path,matrix)
            prefix = "Represent this sentence for searching relevant passages: " if model_name.startswith("BAAI/") else ""
            query_path=cache_dir/(cache_tag+"_queries.npy")
            if query_path.exists():
                queries=np.load(query_path)
            else:
                queries=model.encode([prefix+r["question"] for r in selected],batch_size=args.batch_size,normalize_embeddings=True)
                np.save(query_path,queries)
            for qi, (row, (a, b)) in enumerate(zip(selected, groups)):
                t0 = time.perf_counter()
                titles = row["context"]["title"]
                gold = set(row["supporting_facts"]["title"])
                rel = [title in gold for title in titles]
                bm = BM25Index(); bm.fit([tokenize(t) for t in texts[a:b]])
                sparse = ranks(bm.get_scores(tokenize(row["question"])))
                dense_scores = matrix[a:b] @ queries[qi]
                dense = ranks(dense_scores)
                variants = {"sparse": sparse, "dense": dense, "hybrid": rrf(dense,sparse,10)}
                cache_key = (row["id"],header)
                if cache_key not in rerank_cache:
                    rerank_cache[cache_key] = reranker.predict([[row["question"],t] for t in texts[a:b]], batch_size=args.batch_size).tolist()
                    with old_scores.open("a") as checkpoint:
                        checkpoint.write(json.dumps({"id":row["id"],"headers":header,"scores":rerank_cache[cache_key]})+"\n")
                rerank_scores = rerank_cache[cache_key]
                for depth in (5,10):
                    candidates = list(dict.fromkeys(dense[:depth]+sparse[:depth]))
                    variants["hybrid_rerank_depth"+str(depth)] = sorted(candidates, key=lambda i: -float(rerank_scores[i]))
                split = "calibration" if qi < args.calibration else "heldout"
                for mode, order in variants.items():
                    relevance = [rel[i] for i in order]
                    top = order[:2]
                    score = min(float(rerank_scores[i]) for i in top) if top else -100.0
                    record = {"id": row["id"], "split": split, "embedder": model_name,
                        "headers": header, "mode": mode,
                        "recall2": sum(relevance[:2])/len(gold), "recall5": sum(relevance[:5])/len(gold),
                        "hit5": float(any(relevance[:5])), "mrr": mrr(relevance,5),
                        "ndcg5": ndcg_at_k(relevance,5,len(gold)),
                        "all_support2": float(sum(relevance[:2]) == len(gold)), "score": score,
                        "ranked_titles": [titles[i] for i in order], "gold_titles": sorted(gold)}
                    # Controlled negatives remove both supporting paragraphs from the candidate context.
                    negative_order=[i for i in order if not rel[i]][:2]
                    record["negative_score"]=min(float(rerank_scores[i]) for i in negative_order) if negative_order else -100.0
                    record["selective_negative"]=(qi%2==1)
                    records.append(record)
                if qi % 25 == 0:
                    print(f"{model_name} header={header} {qi}/{len(selected)}", flush=True)
        del model
    (out / "records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    summaries = {}
    for model in sorted({r["embedder"] for r in records}):
        for header in (False,True):
            for mode in sorted({r["mode"] for r in records}):
                rows = [r for r in records if r["embedder"]==model and r["headers"]==header and r["mode"]==mode and r["split"]=="heldout"]
                summaries[f"{model}|headers={header}|{mode}"] = summarize(rows)
    calibration = [r for r in records if r["embedder"].startswith("BAAI/") and r["headers"] and r["mode"]=="hybrid_rerank_depth10" and r["split"]=="calibration"]
    test = [r for r in records if r["embedder"].startswith("BAAI/") and r["headers"] and r["mode"]=="hybrid_rerank_depth10" and r["split"]=="heldout"]
    def selective_score(r): return r["negative_score"] if r["selective_negative"] else r["score"]
    def selective_correct(r): return bool(r["all_support2"]) and not r["selective_negative"]
    fitted = calibrate([selective_score(r) for r in calibration], [selective_correct(r) for r in calibration])
    curve = risk_curve([selective_score(r) for r in test], [selective_correct(r) for r in test])
    selected_test = [r for r in test if fitted["threshold"] is not None and selective_score(r) >= fitted["threshold"]]
    selective = {"target_risk": 0.1, "correctness": "all supporting paragraphs in top two; negatives cannot be answered",
        "controlled_negatives_per_split":100,"negative_protocol":"Remove all supporting paragraphs on alternate question IDs, without duplicating questions",
        "calibration": fitted, "test_coverage": len(selected_test)/len(test),
        "test_risk": float(np.mean([1-selective_correct(r) for r in selected_test])) if selected_test else None,
        "curve": curve, "warning": "Retrieval completeness risk, not answer correctness; review time is an assumption"}
    report = {"provenance": manifest, "results": summaries, "selective": selective}
    (out / "summary.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    lines = ["# Measured HotpotQA retrieval results", "", manifest["scope"]+".",
        f"{args.n} held-out questions, disjoint from {args.calibration} calibration questions. 95% bootstrap CIs over questions.",
        "", "| Model / variant | Recall@5 | MRR@5 | nDCG@5 | All support@2 |", "|---|---|---|---|---|"]
    for key, metrics in summaries.items():
        cells = [f"{metrics[k]['mean']:.3f} [{metrics[k]['lo']:.3f}, {metrics[k]['hi']:.3f}]" for k in ("recall5","mrr","ndcg5","all_support2")]
        lines.append("| "+key+" | "+" | ".join(cells)+" |")
    lines += ["", "This is a bounded-context retrieval benchmark. It does not measure generated-answer accuracy, full-wiki retrieval, human review time, or entailment agreement.",
        "", "HotpotQA annotations and context are CC BY-SA 4.0. Source: https://github.com/hotpotqa/hotpot"]
    (out / "RESULTS.md").write_text("\n".join(lines))
    print("Saved",out / "summary.json",flush=True)


if __name__ == "__main__":
    main()
