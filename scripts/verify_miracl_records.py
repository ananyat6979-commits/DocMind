"""Recompute published component metrics from numeric rankings and pinned qrels.

This checks scoring and cohort accounting, not the neural ranking computation.
No source passage text is needed or published.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_miracl_component import QUERIES_REV, get, priority
from docmind.evaluation.metrics import recall_at_k, ndcg_at_k, mrr, bootstrap_ci


def verify_row(row, qrels):
    predicted = row["predicted_ids"]
    if len(predicted) > 5 or len(set(predicted)) != len(predicted):
        raise ValueError("Invalid or duplicate ranking")
    if row.get("error") and predicted:
        raise ValueError("Failed rows must not carry a successful ranking")
    if "elapsed_ms" in row and (not math.isfinite(row["elapsed_ms"]) or row["elapsed_ms"]<0):
        raise ValueError("Invalid latency")
    rel = [qrels.get(item, 0) > 0 for item in predicted]
    total = sum(v > 0 for v in qrels.values())
    expected = {"recall5": recall_at_k(rel, 5, total), "ndcg5": ndcg_at_k(rel, 5, total), "mrr5": mrr(rel, 5)}
    if any(not math.isclose(row[key], value, abs_tol=1e-12) for key, value in expected.items()):
        raise ValueError("Per-query metric mismatch")


def verify_selection(rows, selection, manifest):
    for key, entry in selection["contrasts"].items():
        language=key.split("_",1)[0]
        a=[r for r in rows if r["language"]==language and r["mode"]==entry["baseline"]]
        b={r["id"]:r for r in rows if r["language"]==language and r["mode"]==entry["treatment"]}
        if len(a)!=entry["n"] or len(b)!=entry["n"] or entry["n"]!=manifest["n_per_language"][language] or {r["id"] for r in a}!=set(b):
            raise ValueError("Selection comparison cohort mismatch")
        if any(r.get("error") for r in [*a,*b.values()]):raise ValueError("Selection comparison contains failed rows")
        for metric, actual in entry["metrics"].items():
            expected=bootstrap_ci([b[r["id"]][metric]-r[metric] for r in a],2000)
            if any(not math.isclose(x,y,abs_tol=1e-12) for x,y in zip(expected,(actual["difference"],*actual["ci95"]))):
                raise ValueError("Default selection interval mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text(encoding="utf-8"))
    run_id = manifest.pop("run_id")
    if hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() != run_id:
        raise ValueError("Manifest identity mismatch")
    summary = json.loads((args.directory / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (args.directory / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = [(r["language"], r["mode"], r["id"]) for r in rows]
    if len(keys) != len(set(keys)) or any(r["run_id"] != run_id for r in rows):
        raise ValueError("Duplicate or foreign checkpoint rows")
    for language in ("en", "hi"):
        prefix = f"https://huggingface.co/datasets/miracl/miracl/resolve/{QUERIES_REV}/miracl-v1.0-{language}/"
        with get(prefix + f"topics/topics.miracl-v1.0-{language}-dev.tsv") as response:
            topics = dict(line.split("\t", 1) for line in response.read().decode("utf-8-sig").splitlines() if line.strip())
        with get(prefix + f"qrels/qrels.miracl-v1.0-{language}-dev.tsv") as response:
            raw = response.read().decode("utf-8-sig")
        judgments = {}
        for line in raw.splitlines():
            qid, _, docid, grade = line.split()
            judgments.setdefault(qid, {})[docid] = int(grade)
        eligible = [qid for qid in topics if any(v > 0 for v in judgments.get(qid, {}).values())]
        ordered = sorted(eligible, key=lambda q: priority(language, q))[:manifest["n_per_language"][language]]
        cohort = set(ordered)
        for row in rows:
            if row["language"] != language: continue
            if row["id"] not in cohort: raise ValueError("Query outside frozen cohort")
            if language + "_" + row["mode"] not in summary["results"]: raise ValueError("Undeclared condition")
            verify_row(row, judgments[row["id"]])
        for key, result in summary["results"].items():
            if not key.startswith(language + "_"): continue
            mode = key[len(language) + 1:]
            available = [r for r in rows if r["language"] == language and r["mode"] == mode]
            served = [r for r in available if not r.get("error")]
            requested = len(cohort)
            if (result["n_requested"], result["n_completed"], result["n_failed"], result["n_unserved"]) != (requested, len(served), len(available)-len(served), requested-len(available)):
                raise ValueError("Cohort denominator mismatch")
            for metric in ("recall5", "ndcg5", "mrr5"):
                mean, lo, hi = bootstrap_ci([r[metric] for r in available] + [0.0] * (requested-len(available)), 2000)
                actual = result["metrics"][metric]
                if any(not math.isclose(a, b, abs_tol=1e-12) for a,b in zip((mean,lo,hi),(actual["requested_mean"],*actual["requested_ci95"]))):
                    raise ValueError("Aggregate interval mismatch")
                if served:
                    complete = bootstrap_ci([r[metric] for r in served], 2000)
                    if any(not math.isclose(a, b, abs_tol=1e-12) for a,b in zip(complete,(actual["completed_only_mean"],*actual["completed_only_ci95"]))):
                        raise ValueError("Completed-only interval mismatch")
        for key, entry in summary["paired_differences"].items():
            if not key.startswith(language + "_"): continue
            treatment, baseline = key[len(language)+1:].split("_minus_", 1)
            a = {r["id"]:r for r in rows if r["language"]==language and r["mode"]==baseline and not r.get("error")}
            b = {r["id"]:r for r in rows if r["language"]==language and r["mode"]==treatment and not r.get("error")}
            paired = [qid for qid in ordered if qid in a and qid in b]
            if len(paired)!=entry["n_completed_pairs"]: raise ValueError("Paired denominator mismatch")
            for metric, actual in entry["metrics"].items():
                expected = bootstrap_ci([b[qid][metric]-a[qid][metric] for qid in paired],2000)
                if any(not math.isclose(x,y,abs_tol=1e-12) for x,y in zip(expected,(actual["difference"],*actual["ci95"]))):
                    raise ValueError("Paired interval mismatch")
    if args.selection:
        selection=json.loads(args.selection.read_text(encoding="utf-8"))
        normalized=(args.directory/"records.jsonl").read_bytes().replace(b"\r\n",b"\n")
        if hashlib.sha256(normalized).hexdigest()!=selection["source_records_sha256"]:raise ValueError("Selection input identity mismatch")
        verify_selection(rows,selection,manifest)
    print(f"Verified {len(rows)} numeric rankings against pinned public judgments and requested denominators.")


if __name__ == "__main__":
    main()
