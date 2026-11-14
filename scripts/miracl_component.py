"""Paired English/Hindi retrieval in a declared MIRACL judged-candidate pool.

This is not full-corpus MIRACL or an end-to-end PDF answer benchmark.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import re
import unicodedata
from pathlib import Path
import platform
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))


_LEGACY_TOKEN=re.compile(r"[\u3400-\u9fff]|[^\W_]+(?:'[^\W_]+)?")


def legacy_tokenize(text):
    from docmind.retrieval.sparse import _STOP
    return [t for t in _LEGACY_TOKEN.findall(unicodedata.normalize("NFC",text).casefold()) if t not in _STOP]


def prepare_pool(payload,distractors):
    from prepare_miracl_component import priority
    manifest=payload["manifest"];language=manifest["language"]
    if not manifest["complete"]:raise ValueError("Incomplete source preparation")
    queries=payload["queries"]
    if len(queries)!=manifest["n_requested"] or len({q["id"] for q in queries})!=len(queries):raise ValueError("Invalid query cohort")
    documents={d["docid"]:d for d in payload["documents"]}
    if len(documents)!=len(payload["documents"]):raise ValueError("Duplicate corpus IDs")
    required=set()
    for query in queries:
        required.update(k for k,v in query["qrels"].items() if v>0)
        required.update(sorted((k for k,v in query["qrels"].items() if v==0),key=lambda k:priority(language,k))[:10])
    if not required<=documents.keys():raise ValueError("Missing judged passages")
    selected=required|set(sorted(documents,key=lambda k:priority(language,k))[:distractors])
    return queries,[documents[k] for k in sorted(selected)]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--inputs",type=Path,required=True);parser.add_argument("--out",type=Path,required=True);parser.add_argument("--device",default="cpu");parser.add_argument("--distractors",type=int,default=2000);parser.add_argument("--max-seconds",type=int,default=5400);args=parser.parse_args()
    if not 1<=args.distractors<=20000 or not 1<=args.max_seconds<=6600:raise ValueError("Budget exceeded")
    args.out.mkdir(parents=True,exist_ok=True)
    os.environ.update(DOCMIND_HOME=str(args.out/"runtime"),DOCMIND_TEST_MODE="0",DOCMIND_SECRETS_FILE=str(args.out/"no-provider-secrets.env"),DOCMIND_DEVICE=args.device)
    os.environ.pop("GROQ_API_KEY",None)
    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder
    from docmind.config import CONFIG
    from docmind.embeddings import embed_texts,embed_query,reset_embedding_model
    from docmind.retrieval.sparse import BM25Index,tokenize
    from docmind.evaluation.metrics import recall_at_k,ndcg_at_k,mrr,bootstrap_ci
    from release_measure import MODELS
    torch.set_num_threads(2)
    deadline=time.monotonic()+args.max_seconds
    corpora={};source_hashes={}
    for language in ("en","hi"):
        path=args.inputs/(language+"-pool.json");raw=path.read_bytes();source_hashes[language]=hashlib.sha256(raw).hexdigest()
        queries,documents=prepare_pool(json.loads(raw),args.distractors)
        corpora[language]={"queries":queries,"documents":documents,"texts":[d["text"] for d in documents],"ids":[d["docid"] for d in documents]}
    versions={p:md.version(p) for p in ("torch","transformers","sentence-transformers","tokenizers","numpy","faiss-cpu")}
    manifest={"scope":"MIRACL dev-derived judged-candidate component subset, not full-corpus MIRACL","sources":source_hashes,
        "device":args.device,"platform":platform.platform(),"processor":platform.processor(),"versions":versions,"python":platform.python_version(),"distractors":args.distractors,
        "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"headers":False,"fusion_depth_per_retriever":10,
        "reranker_candidate_caps":[5,20],"models":MODELS,"n_per_language":{k:len(v["queries"]) for k,v in corpora.items()},
        "documents_per_language":{k:len(v["documents"]) for k,v in corpora.items()},"public_labels":"not independently reviewed"}
    run_id=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest();manifest["run_id"]=run_id
    checkpoint=args.out/"records.jsonl";manifest_path=args.out/"manifest.json"
    if checkpoint.exists() and checkpoint.stat().st_size:
        if not manifest_path.exists() or json.loads(manifest_path.read_text(encoding="utf-8"))["run_id"]!=run_id:raise ValueError("Checkpoint mismatch; choose a fresh output directory")
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    records=[json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()] if checkpoint.exists() else []
    keys=[(r["language"],r["mode"],r["id"]) for r in records]
    if len(keys)!=len(set(keys)) or any(r.get("run_id")!=run_id for r in records):raise ValueError("Invalid checkpoint rows")
    done=set(keys)
    rankings={};lexical={};loads={};timings={(r["language"],r["mode"],r["id"]):r["elapsed_ms"] for r in records}
    plans={language:["sparse_legacy_regex","sparse_word","bge_dense","e5_dense","e5_hybrid_word"]+
           (["sparse_word_grams","e5_hybrid_word_grams"] if language=="hi" else [])+
           ["e5_hybrid_word_"+judge+str(depth) for judge in ("english_rerank","multilingual_rerank") for depth in (5,20)]+
           (["e5_hybrid_word_grams_multilingual_rerank20"] if language=="hi" else []) for language in corpora}
    def emit(language,mode,query,indices,elapsed,error=None):
        if mode not in plans[language]:raise ValueError("Undeclared evaluation condition")
        key=(language,mode,query["id"])
        if key in done:return
        identifiers=corpora[language]["ids"];rel=[query["qrels"].get(identifiers[int(i)],0)>0 for i in indices]
        total=sum(v>0 for v in query["qrels"].values())
        row={"language":language,"mode":mode,"id":query["id"],"run_id":run_id,"recall5":recall_at_k(rel,5,total),"ndcg5":ndcg_at_k(rel,5,total),"mrr5":mrr(rel,5),"elapsed_ms":elapsed,"error":error,"predicted_ids":[identifiers[int(i)] for i in indices[:5]]}
        with checkpoint.open("a",encoding="utf-8") as stream:stream.write(json.dumps(row)+"\n")
        records.append(row);done.add(key);timings[key]=elapsed
    try:
        for language,corpus in corpora.items():
            for representation in (["legacy_regex","word","word_grams"] if language=="hi" else ["legacy_regex","word"]):
                if time.monotonic()>=deadline:raise TimeoutError("Compute budget exhausted")
                token_fn=legacy_tokenize if representation=="legacy_regex" else lambda text:tokenize(text,representation=="word_grams")
                index=BM25Index();index.fit([token_fn(text) for text in corpus["texts"]])
                lexical[(language,representation)]=index
                for query in corpus["queries"]:
                    start=time.monotonic();scores=index.get_scores(token_fn(query["question"]));order=np.argsort(-scores,kind="stable")[:10]
                    order=order[scores[order]>0];rankings[(language,"sparse_"+representation,query["id"])]=order.tolist()
                    emit(language,"sparse_"+representation,query,order,(time.monotonic()-start)*1000)
        for embedding in ("bge","e5"):
            CONFIG.embedding.model_name,CONFIG.embedding.model_revision=MODELS[embedding]
            CONFIG.embedding.backend="sentence-transformers";CONFIG.embedding.device=args.device
            load_start=time.monotonic()
            for language,corpus in corpora.items():
                batches=[]
                for offset in range(0,len(corpus["texts"]),128):
                    if time.monotonic()>=deadline:raise TimeoutError("Compute budget exhausted")
                    batches.append(embed_texts(corpus["texts"][offset:offset+128]))
                matrix=np.concatenate(batches)
                if not np.isfinite(matrix).all():raise FloatingPointError("Nonfinite embeddings")
                for query in corpus["queries"]:
                    if time.monotonic()>=deadline:raise TimeoutError("Compute budget exhausted")
                    start=time.monotonic();vector=embed_query(query["question"])[0];scores=matrix@vector;order=np.argsort(-scores,kind="stable")[:10]
                    rankings[(language,embedding+"_dense",query["id"])]=order.tolist()
                    dense_ms=(time.monotonic()-start)*1000
                    emit(language,embedding+"_dense",query,order,dense_ms)
                    if embedding=="e5":
                        for representation in (["word","word_grams"] if language=="hi" else ["word"]):
                            fusion_start=time.monotonic()
                            sparse=rankings[(language,"sparse_"+representation,query["id"])];fused={}
                            for ranked in (order.tolist(),sparse):
                                for rank,index in enumerate(ranked,1):fused[index]=fused.get(index,0)+1/(60+rank)
                            fused_order=sorted(fused,key=lambda i:(-fused[i],i));name="e5_hybrid_"+representation
                            rankings[(language,name,query["id"])]=fused_order
                            emit(language,name,query,fused_order,dense_ms+timings[(language,"sparse_"+representation,query["id"])]+(time.monotonic()-fusion_start)*1000)
                del matrix,batches
            loads[embedding+"_encoding_seconds"]=time.monotonic()-load_start
            reset_embedding_model()
        for judge_name,model,revision in (("english_rerank","cross-encoder/ms-marco-MiniLM-L-6-v2","233902d25c440f23af6f7d6e94d2946bac0bee0a"),("multilingual_rerank",*MODELS["reranker"])):
            if time.monotonic()>=deadline:raise TimeoutError("Compute budget exhausted")
            start=time.monotonic();judge=CrossEncoder(model,revision=revision,device=args.device,max_length=512);judge.predict([["query","passage"]]);loads[judge_name+"_load_seconds"]=time.monotonic()-start
            for language,corpus in corpora.items():
                variants=[("word",5),("word",20)]+([("word_grams",20)] if language=="hi" and judge_name=="multilingual_rerank" else [])
                for representation,depth in variants:
                    mode="e5_hybrid_"+representation+"_"+judge_name+str(depth)
                    for query in corpus["queries"]:
                        if (language,mode,query["id"]) in done:continue
                        if time.monotonic()>=deadline:raise TimeoutError("Compute budget exhausted")
                        start=time.monotonic();candidates=rankings[(language,"e5_hybrid_"+representation,query["id"])][:depth]
                        try:
                            scores=judge.predict([[query["question"],corpus["texts"][i]] for i in candidates],batch_size=16)
                            if not np.isfinite(scores).all():raise FloatingPointError("Nonfinite reranker scores")
                            order=[candidates[int(i)] for i in np.argsort(-np.asarray(scores),kind="stable")]
                            emit(language,mode,query,order,timings[(language,"e5_hybrid_"+representation,query["id"])]+(time.monotonic()-start)*1000)
                        except Exception as exc:emit(language,mode,query,[],(time.monotonic()-start)*1000,type(exc).__name__)
            del judge
    except Exception as exc:
        manifest["stopped_with"]=type(exc).__name__
        print("Stopped component profile:",type(exc).__name__,flush=True)
    finally:
        summary={"manifest":manifest,"loads":loads,"results":{},"limits":["Pooled judged candidates, not full Wikipedia retrieval","Distinct topic sets per language","No independent public-label review","Unserved rows stay in requested-cohort statistics","Timing excludes corpus preparation and is not API load","Hybrid and reranked latency sums separately measured components; single run only"]}
        for language,modes in plans.items():
            for mode in modes:
                available=[r for r in records if r["language"]==language and r["mode"]==mode]
                served=[r for r in available if not r.get("error")];requested=len(corpora[language]["queries"])
                result={"warm_p50_ms":float(np.median([r["elapsed_ms"] for r in served])) if served else None,"warm_p95_ms":float(np.percentile([r["elapsed_ms"] for r in served],95)) if served else None,"status":"complete" if len(served)==requested else ("partial" if served else ("failed" if available else "unmeasured")),"n_requested":requested,"n_completed":len(served),"n_failed":len(available)-len(served),"n_unserved":requested-len(available),"metrics":{}}
                for metric in ("recall5","ndcg5","mrr5"):
                    values=[r[metric] for r in available]+[0.0]*(requested-len(available))
                    mean,lo,hi=bootstrap_ci(values,2000)
                    complete_mean,complete_lo,complete_hi=bootstrap_ci([r[metric] for r in served],2000)
                    result["metrics"][metric]={"requested_mean":mean,"requested_ci95":[lo,hi],"completed_only_mean":complete_mean if served else None,"completed_only_ci95":[complete_lo,complete_hi] if served else None}
                summary["results"][language+"_"+mode]=result
        summary["paired_differences"]={}
        comparisons=[("sparse_legacy_regex","sparse_word"),("sparse_word","sparse_word_grams"),("bge_dense","e5_dense"),("e5_dense","e5_hybrid_word"),("e5_hybrid_word","e5_hybrid_word_multilingual_rerank20"),("e5_hybrid_word_english_rerank20","e5_hybrid_word_multilingual_rerank20"),("e5_hybrid_word_multilingual_rerank5","e5_hybrid_word_multilingual_rerank20"),("e5_hybrid_word","e5_hybrid_word_grams")]
        for language in corpora:
            for baseline,treatment in comparisons:
                if baseline not in plans[language] or treatment not in plans[language]:continue
                a={r["id"]:r for r in records if r["language"]==language and r["mode"]==baseline}
                b={r["id"]:r for r in records if r["language"]==language and r["mode"]==treatment}
                paired=[q["id"] for q in corpora[language]["queries"] if q["id"] in a and q["id"] in b and not a[q["id"]].get("error") and not b[q["id"]].get("error")]
                entry={"n_completed_pairs":len(paired),"scope":"completed paired subset; do not extrapolate incomplete profiles","metrics":{}}
                for metric in ("recall5","ndcg5","mrr5"):
                    if paired:
                        mean,lo,hi=bootstrap_ci([b[q][metric]-a[q][metric] for q in paired],2000)
                        entry["metrics"][metric]={"difference":mean,"ci95":[lo,hi]}
                summary["paired_differences"][language+"_"+treatment+"_minus_"+baseline]=entry
        (args.out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
        print("Saved component summary",flush=True)
    if manifest.get("stopped_with") not in (None,"TimeoutError"):raise SystemExit(1)

if __name__=="__main__":main()
