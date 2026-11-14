"""Bounded lexical-only MIRACL control, without loading neural models."""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--inputs",type=Path,required=True);parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):raise ValueError("Output already exists; choose a fresh directory to preserve prior measurements")
    os.environ.update(DOCMIND_TEST_MODE="0",DOCMIND_SECRETS_FILE=str(args.out/"no-provider.env"));os.environ.pop("GROQ_API_KEY",None)
    import numpy as np
    from miracl_component import prepare_pool,legacy_tokenize
    from docmind.retrieval.sparse import BM25Index,tokenize
    from docmind.evaluation.metrics import recall_at_k,ndcg_at_k,mrr,bootstrap_ci
    args.out.mkdir(parents=True,exist_ok=True)
    corpora={};identities={}
    for language in ("en","hi"):
        payload=json.loads((args.inputs/(language+"-pool.json")).read_text(encoding="utf-8"));queries,documents=prepare_pool(payload,2000)
        corpora[language]=(queries,documents)
        identities[language]={"cohort_sha256":hashlib.sha256(json.dumps(queries,sort_keys=True).encode()).hexdigest(),"corpus_sha256":hashlib.sha256(json.dumps(documents,sort_keys=True).encode()).hexdigest(),**{k:payload["manifest"][k] for k in ("query_revision","corpus_revision","topics_sha256","qrels_sha256","selected_ids_sha256")}}
    manifest={"scope":"MIRACL dev-derived judged-candidate lexical component, not full-corpus retrieval","platform":platform.platform(),"python":platform.python_version(),"versions":{k:md.version(k) for k in ("numpy","regex")},"source_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"n_per_language":{k:len(v[0]) for k,v in corpora.items()},"documents_per_language":{k:len(v[1]) for k,v in corpora.items()},"semantic_sources":identities,"distractors":2000,"public_labels":"not independently reviewed"}
    run_id=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest();manifest["run_id"]=run_id
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    records=[];fits={}
    with (args.out/"records.jsonl").open("w",encoding="utf-8") as stream:
        for language,(queries,documents) in corpora.items():
            for representation in (["legacy_regex","word","word_grams"] if language=="hi" else ["legacy_regex","word"]):
                mode="sparse_"+representation;token_fn=legacy_tokenize if representation=="legacy_regex" else lambda text:tokenize(text,representation=="word_grams")
                started=time.perf_counter();tokens=[token_fn(d["text"]) for d in documents];index=BM25Index();index.fit(tokens)
                fits[language+"_"+mode]={"seconds":time.perf_counter()-started,"tokens":sum(map(len,tokens)),"terms":len(index._postings)};del tokens
                for query in queries:
                    started=time.perf_counter();scores=index.get_scores(token_fn(query["question"]));order=np.argsort(-scores,kind="stable")[:5];order=order[scores[order]>0]
                    ids=[documents[int(i)]["docid"] for i in order];rel=[query["qrels"].get(i,0)>0 for i in ids];total=sum(v>0 for v in query["qrels"].values())
                    row={"language":language,"mode":mode,"id":query["id"],"run_id":run_id,"predicted_ids":ids,"recall5":recall_at_k(rel,5,total),"ndcg5":ndcg_at_k(rel,5,total),"mrr5":mrr(rel,5),"elapsed_ms":(time.perf_counter()-started)*1000,"error":None}
                    records.append(row);stream.write(json.dumps(row)+"\n")
                print("Finished lexical",language,representation,flush=True)
    summary={"manifest":manifest,"index_fits":fits,"results":{},"paired_differences":{},"limits":["Pooled judged candidates with all gold positives deliberately included", "Distinct language topic sets", "Original public labels without independent review", "Single-run local timings; not neural inference or API load", "Exploratory nominal 95 percent intervals without family-wise correction"]}
    for key in fits:
        language,mode=key.split("_",1);rows=[r for r in records if r["language"]==language and r["mode"]==mode]
        result={"status":"complete","n_requested":len(rows),"n_completed":len(rows),"n_failed":0,"n_unserved":0,"warm_p50_ms":float(np.median([r["elapsed_ms"] for r in rows])),"warm_p95_ms":float(np.percentile([r["elapsed_ms"] for r in rows],95)),"metrics":{}}
        for metric in ("recall5","ndcg5","mrr5"):
            mean,lo,hi=bootstrap_ci([r[metric] for r in rows],2000);result["metrics"][metric]={"requested_mean":mean,"requested_ci95":[lo,hi],"completed_only_mean":mean,"completed_only_ci95":[lo,hi]}
        summary["results"][key]=result
    for language in corpora:
        for baseline,treatment in [("sparse_legacy_regex","sparse_word")]+([("sparse_word","sparse_word_grams")] if language=="hi" else []):
            a={r["id"]:r for r in records if r["language"]==language and r["mode"]==baseline};b={r["id"]:r for r in records if r["language"]==language and r["mode"]==treatment}
            entry={"n_completed_pairs":len(a),"scope":"completed frozen query cohort","metrics":{}}
            for metric in ("recall5","ndcg5","mrr5"):
                mean,lo,hi=bootstrap_ci([b[q["id"]][metric]-a[q["id"]][metric] for q in corpora[language][0]],2000);entry["metrics"][metric]={"difference":mean,"ci95":[lo,hi]}
            summary["paired_differences"][language+"_"+treatment+"_minus_"+baseline]=entry
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")


if __name__=="__main__":main()
