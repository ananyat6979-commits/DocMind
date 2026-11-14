"""Free T4 component measurements and a bounded English PDF diagnostic pilot.

This is not the hosted default, an official leaderboard run or Hindi PDF validation.
Restricted text and per-example labels stay in the private output directory.
"""
import argparse
import gc
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.pop("GROQ_API_KEY",None)
os.environ.setdefault("DOCMIND_HOME",str(ROOT/"reports"/"release_workspace"/"runtime"))
from docmind.evaluation.release import digest
from docmind.evaluation.metrics import bootstrap_ci

XNLI_REV="b8dd5d7af51114dbda02c0e3f6133f332186418e"
MODELS={
 "bge":("BAAI/bge-small-en-v1.5","5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"),
 "e5":("intfloat/multilingual-e5-small","614241f622f53c4eeff9890bdc4f31cfecc418b3"),
 "reranker":("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1","1427fd652930e4ba29e8149678df786c240d8825"),
 "english_nli":("typeform/distilbert-base-uncased-mnli","cfa538a0fddbbd978fefe8966c1aeff7ad409c90"),
 "multilingual_nli":("MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli","0a71e92a985b6e1ad1828cf67ce9c459639c1dca")}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=True,default=str),encoding="utf-8")


def acquire_xnli(directory):
    import pyarrow.parquet as pq
    result={}
    for language in ("en","hi"):
        path=directory/("xnli-"+language+".parquet")
        if not path.exists():
            urllib.request.urlretrieve("https://huggingface.co/datasets/facebook/xnli/resolve/"+XNLI_REV+"/"+language+"/test-00000-of-00001.parquet",path)
        data=pq.read_table(path).to_pylist()
        order=list(range(len(data)));random.Random(7).shuffle(order)
        result[language]=[{**data[i],"id":language+"-"+str(i),"source_index":i} for i in order[:200]]
    return result


def binary_stats(rows):
    import numpy as np
    from docmind.attribution import attribution_metrics
    point=attribution_metrics([r["score"]>=.8 for r in rows],[r["label"]==0 for r in rows])
    groups={k:[r for r in rows if r["premise_hash"]==k] for k in sorted({r["premise_hash"] for r in rows})}
    counts=np.array([[sum(r["score"]>=.8 and r["label"]==0 for r in group),
                      sum(r["score"]>=.8 and r["label"]!=0 for r in group),
                      sum(r["score"]<.8 and r["label"]==0 for r in group)] for group in groups.values()])
    rng=np.random.default_rng(7);samples=counts[rng.integers(0,len(counts),(2000,len(counts)))].sum(axis=1)
    for name,column in (("precision",1),("recall",2)):
        denominator=samples[:,0]+samples[:,column];valid=denominator>0
        point[name+"_ci95"]=np.quantile(samples[valid,0]/denominator[valid],[.025,.975]).tolist() if valid.any() else [None,None]
    positive_groups=[group for group in groups.values() if any(r["score"]>=.8 for r in group)]
    if point["fp"]==0 and point["tp"]>0 and all(sum(r["score"]>=.8 for r in group)==1 for group in positive_groups):
        point["precision_bootstrap_ci95"]=point["precision_ci95"]
        point["precision_ci95"]=[.025**(1/len(positive_groups)),1.0]
        point["precision_ci_method"]="Exact binomial boundary interval: one positive decision per independent premise cluster"
    else:point["precision_ci_method"]="Premise-cluster percentile bootstrap"
    point.update(threshold=.8,independent_premise_clusters=len(groups),n_failed=sum(bool(r.get("error")) for r in rows),failure_decision="no entailment; retained in requested denominator",human_labels="original public labels, not independently reviewed")
    return point


def nli_components(data,output,device="cuda:0"):
    from docmind.attribution import NLIJudge
    import torch
    summaries={};public_predictions=[]
    for name in ("english_nli","multilingual_nli"):
        start=time.perf_counter();model,revision=MODELS[name]
        judge=NLIJudge(model,revision=revision,local_files_only=False,device=device,batch_size=16)
        rows=[]
        for language,examples in data.items():
            predictions=[];errors=[]
            for offset in range(0,len(examples),16):
                batch=examples[offset:offset+16]
                try:
                    values=judge([(r["premise"],r["hypothesis"]) for r in batch]);failures=[None]*len(batch)
                except Exception:
                    values=[];failures=[]
                    for row in batch:
                        try:values.append(judge([(row["premise"],row["hypothesis"])])[0]);failures.append(None)
                        except Exception as exc:values.append(0.0);failures.append(type(exc).__name__)
                for index,value in enumerate(values):
                    if not __import__("math").isfinite(value):values[index]=0.0;failures[index]="NonfiniteScore"
                predictions.extend(values);errors.extend(failures)
            records=[{"id":r["id"],"label":r["label"],"score":float(p),"error":error,"premise_hash":hashlib.sha256(r["premise"].encode()).hexdigest()} for r,p,error in zip(examples,predictions,errors)]
            rows.extend(records);summaries[name+"_"+language]=binary_stats(records)
            public_predictions.extend({"model":name,**{k:r[k] for k in ("id","score","premise_hash","error")}} for r in records)
        write(output/"private"/(name+".json"),rows)
        summaries[name+"_load_and_score_seconds"]=time.perf_counter()-start
        del judge;gc.collect();torch.cuda.empty_cache()
    write(output/"nli-predictions.json",public_predictions)
    return {"scope":"XNLI test-derived binary entailment component subset; not PDF attribution",
        "development_split_excluded":True,"source_revision":XNLI_REV,"models":MODELS,"results":summaries,
        "license":"XNLI noncommercial terms; raw examples and per-example labels are private"}


def ann_component(data,output,device="cuda:0"):
    import numpy as np
    import faiss
    import torch
    import docmind.retrieval.dense as module
    from docmind.retrieval.dense import DenseRetriever
    from docmind.config import CONFIG
    from docmind.models import Chunk
    from docmind.embeddings import embed_texts,embed_query,reset_embedding_model
    from sentence_transformers import CrossEncoder
    CONFIG.embedding.backend="sentence-transformers";CONFIG.embedding.device=device
    CONFIG.embedding.model_name,CONFIG.embedding.model_revision=MODELS["e5"]
    import pyarrow.parquet as pq
    corpus=list(dict.fromkeys((lang,r["premise"]) for lang in ("en","hi") for r in pq.read_table(output/("xnli-"+lang+".parquet")).to_pylist()))
    premises=[text for lang,text in corpus]
    matrix=embed_texts(premises)
    assert np.isfinite(matrix).all() and np.allclose(np.linalg.norm(matrix,axis=1),1,atol=.001)
    chunks=[Chunk(text,"d","component",i,file_id=corpus[i][0]) for i,text in enumerate(premises)]
    question_records=[(lang,r) for lang,examples in data.items() for r in examples[:50]]
    questions=[r["hypothesis"] for lang,r in question_records]
    query_language={r["hypothesis"]:lang for lang,examples in data.items() for r in examples[:50]}
    vectors={question:embed_query(question) for question in questions}
    original=module.embed_query;module.embed_query=lambda q:vectors[q]
    results={};truth={};query_records=[];lookup={chunk.chunk_id:i for i,chunk in enumerate(chunks)};faiss.omp_set_num_threads(2)
    try:
        for kind in ("flat","hnsw"):
            os.environ["DOCMIND_DENSE_INDEX"]=kind
            retriever=DenseRetriever();start=time.perf_counter();retriever.fit(chunks,matrix)
            build=time.perf_counter()-start
            for scoped in (False,True):
                rows=[]
                for query_index,question in enumerate(questions):
                    language,question_record=question_records[query_index]
                    start=time.perf_counter();hits=retriever.search(question,10,file_ids=[language] if scoped else None)
                    elapsed=(time.perf_counter()-start)*1000
                    ids=[r.chunk.chunk_id for r in hits]
                    if kind=="flat":truth[(question,scoped)]=ids
                    ref=truth[(question,scoped)]
                    row={"recall":len(set(ids)&set(ref))/len(ref) if ref else 0,"latency_ms":elapsed}
                    rows.append(row)
                    query_records.append({"kind":kind,"scoped":scoped,"query_id":question_record["id"],"language":language,**row,"predicted_indices":[lookup[i] for i in ids],"exact_indices":[lookup[i] for i in ref]})
                mean,lo,hi=bootstrap_ci([r["recall"] for r in rows],2000)
                results[kind+"_"+("scoped_exact_fallback" if scoped else "unscoped")]= {
                    "recall10_against_exact":{"mean":mean,"ci95":[lo,hi]},"build_seconds":build,
                    "p50_ms":float(np.percentile([r["latency_ms"] for r in rows],50)),
                    "p95_ms":float(np.percentile([r["latency_ms"] for r in rows],95))}
    finally:module.embed_query=original
    reset_embedding_model();gc.collect();torch.cuda.empty_cache()
    model,revision=MODELS["reranker"]
    reranker=CrossEncoder(model,revision=revision,device=device)
    values=reranker.predict([(examples[0]["hypothesis"],examples[0]["premise"]) for examples in data.values()])
    assert np.isfinite(values).all()
    del reranker;gc.collect();torch.cuda.empty_cache()
    write(output/"ann-query-records.json",query_records)
    return {"faiss_version":faiss.__version__,"numpy_version":np.__version__,"vector_sha256":hashlib.sha256(matrix.tobytes()).hexdigest(),"hnsw":{"M":32,"efConstruction":80,"efSearch":64,"threads":2},"scope":"Application ANN fidelity on real E5 embeddings from XNLI text, not semantic retrieval accuracy or a scale claim",
        "vectors":len(chunks),"queries":len(questions),"model":MODELS["e5"],"results":results,
        "reranker_finite_smoke":True,"limits":["Small constructed-sentence corpus","Scoped application path deliberately uses exact filtering","Embedding time excluded from search timings"]}


def legacy_evidence(text):
    matches=list(re.finditer(r"[^\n.!?]+(?:[.!?]+|$)",text))
    for i,match in enumerate(matches):
        yield match.start(),match.end(),text[match.start():match.end()]
        if i+1<len(matches):
            end=matches[i+1].end();yield match.start(),end,text[match.start():end]


def pilot_accounting(records,n):
    if n<1 or len(records)>n:raise ValueError("Invalid requested pilot cohort")
    completed=[r for r in records if not r.get("error")]
    held=sum(bool(r.get("needs_review")) for r in completed)
    failed=len(records)-len(completed);unserved=n-len(records)
    return {"n_requested":n,"n_completed":len(completed),"failed":failed,"n_unserved":unserved,
            "completed_only_routing_withhold_rate":held/len(completed) if completed else None,
            "requested_not_released_rate":(held+failed+unserved)/n}


def english_pdf_pilot(inputs,output,n=3):
    device="cuda:0"
    import torch
    from docmind.config import CONFIG
    from docmind.models import Document
    from docmind.ingestion.chunker import chunk_document
    from docmind.embeddings import embed_texts,reset_embedding_model
    from docmind.store import CorpusStore
    from docmind.retrieval.hybrid import HybridRetriever
    from docmind.agent.react import ReActAgent
    from docmind.llm.local_reader import LocalReader
    import docmind.attribution as attribution
    from docmind.attribution import NLIJudge,attribute
    from docmind.text import sentence_spans
    os.environ["DOCMIND_DENSE_INDEX"]="flat"
    cache=json.loads((inputs/"cache-en.json").read_text(encoding="utf-8"))
    pool=json.loads((inputs/"pool-en.json").read_text(encoding="utf-8"))
    CONFIG.embedding.model_name,CONFIG.embedding.model_revision=MODELS["bge"]
    CONFIG.embedding.backend="sentence-transformers";CONFIG.embedding.device=device
    CONFIG.retriever.mode="hybrid_rerank"
    CONFIG.retriever.reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
    CONFIG.retriever.reranker_revision="233902d25c440f23af6f7d6e94d2946bac0bee0a"
    CONFIG.chunker.strategy="recursive";CONFIG.llm.temperature=0;CONFIG.llm.max_tokens=256
    chunks=[]
    for page in cache["pages"]:
        if page["metadata"]["extraction_quality"]["quarantined"]:continue
        doc=Document(page["text"],"english.pdf",metadata={**page["metadata"],"filename":"english.pdf",
            "partial_document":True,"language_info":{"language":"en","validation":"unmeasured"}})
        for chunk in chunk_document(doc):chunk.file_id=cache["pdf_sha256"][:16];chunk.ord=len(chunks);chunks.append(chunk)
    matrix=embed_texts([c.content for c in chunks]);store=CorpusStore(CONFIG.db_path)
    store.add_document({"file_id":cache["pdf_sha256"][:16],"filename":"english.pdf","source":"english.pdf",
        "sha256":cache["pdf_sha256"],"n_units":len(cache["pages"])},chunks,matrix)
    reset_embedding_model();gc.collect();torch.cuda.empty_cache()
    model,revision=MODELS["english_nli"]
    judge=NLIJudge(model,revision,local_files_only=False,device=device,batch_size=16)
    reader=LocalReader(max_input_tokens=8192);runner=ReActAgent(retriever=HybridRetriever(store),llm=reader,max_iterations=5,attribution_judge=judge)
    records=[];replays=[]
    fixed=attribution.evidence_spans
    for item in pool[:n]:
        start=time.perf_counter();prior=len(reader.call_records)
        try:
            response=runner.run(item["question"])
            record={"id":item["id"],"language":"en","condition":"baseline_capture","elapsed_seconds":time.perf_counter()-start,
                "answer":response.answer,"sources":[{"chunk":__import__("dataclasses").asdict(s.chunk),"score":s.score,"retriever":s.retriever} for s in response.sources],
                "calls":reader.call_records[prior:],"finish_reason":response.finish_reason,"needs_review":response.needs_review,"review_reasons":response.review_reasons,
                "citation_validity":response.citation_report.get("validity"),"attribution":response.citation_report.get("attribution")}
            claims=[s.text for s in sentence_spans(response.answer)]
            attribution.evidence_spans=legacy_evidence
            try:old=attribute(response.answer,response.sources,judge,claim_units=claims)
            finally:attribution.evidence_spans=fixed
            new=attribute(response.answer,response.sources,judge,claim_units=claims)
            replays.append({"id":item["id"],"legacy":old,"repaired":new,"input_sha256":digest([response.answer,record["sources"]])})
        except Exception as exc:
            record={"id":item["id"],"language":"en","condition":"baseline_capture","elapsed_seconds":time.perf_counter()-start,"error":type(exc).__name__}
        records.append(record);write(output/"private"/"english-pilot.json",records);write(output/"private"/"replay.json",replays)
    write(output/"reader-identity.json",reader.identity)
    result={"scope":"English local-reader T4 diagnostic pilot; synthetic drafts, no human correctness labels",
        "retrieval_profile":{"embedding":MODELS["bge"],"mode":"hybrid_rerank","reranker":[CONFIG.retriever.reranker_model,CONFIG.retriever.reranker_revision],"chunker":"recursive"},
        **pilot_accounting(records,n),
        "partial_document_policy":"All answers require review because the source contains quarantined pages",
        "attribution_replays":[{"id":r["id"],"legacy_supported_fraction":r["legacy"]["supported_fraction"],"repaired_supported_fraction":r["repaired"]["supported_fraction"]} for r in replays],
        "call_measurements":[{"id":r["id"],"elapsed_seconds":r["elapsed_seconds"],"calls":r.get("calls",[]),"error":r.get("error")} for r in records],
        "reader":reader.identity,"hindi_pdf_profile":"unmeasured: no eligible pages in the current inspected corpus",
        "balanced_cohort":"not frozen: missing Hindi PDF pilot and fresh-condition timings"}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--inputs",type=Path,required=True);parser.add_argument("--out",type=Path,required=True);parser.add_argument("--pilot-n",type=int,default=3)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if not 1<=args.pilot_n<=6:raise ValueError("Exploratory English pilot cap is six")
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError("This profile requires the free GPU runtime")
    result={"scope":"English/Hindi components and English local-reader T4 pilot, hosted default unmeasured",
        "versions":{k:md.version(k) for k in ("torch","transformers","sentence-transformers","tokenizers")},
        "python":platform.python_version(),"gpu":torch.cuda.get_device_name(0),"allocated_gpu_count":torch.cuda.device_count(),
        "human_review_count":0,"human_validation":"pending user annotation","results":{}}
    if result["versions"]["transformers"]!="5.23.0" or result["versions"]["sentence-transformers"]!="5.7.0":raise RuntimeError("Dependency baseline mismatch")
    started=time.perf_counter();data=acquire_xnli(args.out)
    for name,operation in (("xnli",lambda:nli_components(data,args.out)),("ann",lambda:ann_component(data,args.out)),("pdf_pilot",lambda:english_pdf_pilot(args.inputs,args.out,args.pilot_n))):
        try:result["results"][name]=operation()
        except Exception as exc:
            import traceback;traceback.print_exc();result["results"][name]={"error":type(exc).__name__,"detail":str(exc)[:500]}
        result["elapsed_seconds"]=time.perf_counter()-started
        result["peak_allocated_cuda_bytes"]=torch.cuda.max_memory_allocated()
        write(args.out/"summary.json",result)
        print("Completed",name,flush=True)
    print("Measurement complete",flush=True)

if __name__=="__main__":main()
