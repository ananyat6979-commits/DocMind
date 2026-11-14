"""Verify numeric NLI and ANN reports without repeating neural inference."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    parser.add_argument("--xnli-cache",type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import pyarrow.parquet as pq
    from release_measure import binary_stats
    from docmind.evaluation.metrics import bootstrap_ci
    summary=json.loads((args.directory/"summary.json").read_text(encoding="utf-8"))
    predictions=json.loads((args.directory/"nli-predictions.json").read_text(encoding="utf-8"))
    if len(predictions)!=800 or len({(r["model"],r["id"]) for r in predictions})!=800:raise ValueError("NLI cohort mismatch")
    for language in ("en","hi"):
        source=pq.read_table(args.xnli_cache/("xnli-"+language+".parquet")).to_pylist()
        order=list(range(len(source)));random.Random(7).shuffle(order)
        expected_ids={language+"-"+str(index) for index in order[:200]}
        for model in ("english_nli","multilingual_nli"):
            records=[dict(r) for r in predictions if r["model"]==model and r["id"].startswith(language+"-")]
            if len(records)!=200 or {r["id"] for r in records}!=expected_ids:raise ValueError("NLI stratum mismatch")
            for row in records:
                index=int(row["id"].split("-")[1]);original=source[index]
                if not math.isfinite(row["score"]) or not 0<=row["score"]<=1:raise ValueError("Invalid NLI score")
                if hashlib.sha256(original["premise"].encode()).hexdigest()!=row["premise_hash"]:raise ValueError("NLI source mismatch")
                row["label"]=original["label"]
            actual=binary_stats(records);expected=summary["results"]["xnli"]["results"][model+"_"+language]
            for field in ("tp","fp","fn","n","precision","recall","precision_ci95","recall_ci95"):
                if actual[field]!=expected[field]:raise ValueError("NLI aggregate mismatch: "+field)
    rows=json.loads((args.directory/"ann-query-records.json").read_text(encoding="utf-8"))
    if len(rows)!=400 or len({(r["kind"],r["scoped"],r["query_id"]) for r in rows})!=400:raise ValueError("ANN cohort mismatch")
    for row in rows:
        for field in ("predicted_indices","exact_indices"):
            values=row[field]
            if len(values)!=10 or len(set(values))!=10 or any(type(i)is not int or not 0<=i<summary["results"]["ann"]["vectors"] for i in values):raise ValueError("Invalid ANN ranking")
        actual=len(set(row["predicted_indices"])&set(row["exact_indices"]))/10
        if not math.isclose(actual,row["recall"],abs_tol=1e-12):raise ValueError("ANN recall mismatch")
    for kind in ("flat","hnsw"):
        for scoped in (False,True):
            values=[r for r in rows if r["kind"]==kind and r["scoped"]==scoped]
            if len(values)!=100:raise ValueError("ANN stratum mismatch")
            mean,lo,hi=bootstrap_ci([r["recall"] for r in values],2000)
            expected=summary["results"]["ann"]["results"][kind+"_"+("scoped_exact_fallback" if scoped else "unscoped")]
            if [mean,lo,hi]!=[expected["recall10_against_exact"]["mean"],*expected["recall10_against_exact"]["ci95"]]:raise ValueError("ANN interval mismatch")
            if not math.isclose(float(np.median([r["latency_ms"] for r in values])),expected["p50_ms"],abs_tol=1e-12):raise ValueError("ANN timing mismatch")
    print("Verified 800 NLI predictions and 400 ANN numeric records; no neural inference repeated.")


if __name__=="__main__":main()
