"""Measure NLI attribution against existing human CLIMATE-FEVER evidence labels.

Only source IDs, hashes, labels and scores are redistributed. The source dataset's
license is unspecified on its dataset card; this runner downloads it for local evaluation.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("DOCMIND_HOME",str(ROOT/"reports"/"attribution_workspace"))
sys.path.insert(0,str(ROOT/"scripts"))
from public_benchmark import source_provenance
import numpy as np
from docmind.attribution import NLIJudge,attribution_metrics,attribute
from docmind.agent.citations import verify_citations
from docmind.models import Chunk,SearchResult


def get(url):
    with urllib.request.urlopen(url,timeout=120) as response:return json.load(response)


def cluster_ci(rows,threshold):
    groups={key:[r for r in rows if r["claim_id"]==key] for key in sorted({r["claim_id"] for r in rows})}
    keys=list(groups)
    counts=np.array([[sum(r["score"]>=threshold and r["label"] for r in groups[k]),
        sum(r["score"]>=threshold and not r["label"] for r in groups[k]),
        sum(r["score"]<threshold and r["label"] for r in groups[k])] for k in keys],dtype=float)
    rng=np.random.default_rng(17)
    indices=rng.integers(0,len(keys),size=(2000,len(keys)))
    totals=counts[indices].sum(axis=1)
    out={}
    for metric,column in (("precision",1),("recall",2)):
        denominator=totals[:,0]+totals[:,column]
        values=totals[denominator>0,0]/denominator[denominator>0]
        out[metric]=list(np.quantile(values,[0.025,0.975])) if len(values) else [None,None]
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--batch-size",type=int,default=2)
    args=parser.parse_args()
    out=ROOT/"reports"/"attribution";out.mkdir(parents=True,exist_ok=True)
    source=out/"source";source.mkdir(exist_ok=True)
    lock=source/"source.json";parquet=source/"climate.parquet"
    if lock.exists():
        metadata=json.loads(lock.read_text())
    else:
        info=get("https://huggingface.co/api/datasets/tdiggelm/climate_fever")
        path=next(s["rfilename"] for s in info["siblings"] if s["rfilename"].endswith(".parquet"))
        metadata={"dataset":"tdiggelm/climate_fever","revision":info["sha"],"path":path,
                  "license":"unspecified; raw data is not redistributed"}
    if not parquet.exists():
        urllib.request.urlretrieve("https://huggingface.co/datasets/tdiggelm/climate_fever/resolve/"+
            metadata["revision"]+"/"+metadata["path"],parquet)
    digest=hashlib.sha256(parquet.read_bytes()).hexdigest()
    if metadata.get("sha256") and metadata["sha256"]!=digest:raise ValueError("Dataset hash mismatch")
    metadata["sha256"]=digest;lock.write_text(json.dumps(metadata,indent=2))
    import pyarrow.parquet as pq
    data=pq.read_table(parquet).to_pylist()
    data.sort(key=lambda x:hashlib.sha256(("attribution-v1:"+x["claim_id"]).encode()).hexdigest())
    data=data[:300]
    model="typeform/distilbert-base-uncased-mnli"
    revision=get("https://huggingface.co/api/models/"+model)["sha"]
    print("Loading NLI model",model,revision,flush=True)
    import torch
    torch.set_num_threads(2)
    judge=NLIJudge(model,revision=revision,local_files_only=False,device=args.device,batch_size=args.batch_size)
    rows=[]
    for qi,item in enumerate(data):
        pairs=[(e["evidence"],item["claim"]) for e in item["evidences"]]
        scores=judge(pairs)
        for e,score in zip(item["evidences"],scores):
            row={"claim_id":item["claim_id"],"evidence_id":e["evidence_id"],
                "split":"calibration" if qi<100 else "heldout",
                "score":score,"label":e["evidence_label"]==0,
                "human_label":int(e["evidence_label"]),
                "evidence_sha256":hashlib.sha256(e["evidence"].encode()).hexdigest(),
                "start":0,"end":len(e["evidence"])}
            rows.append(row)
        with (out/"scores.jsonl").open("a") as checkpoint:
            checkpoint.write("".join(json.dumps(r)+"\n" for r in rows[-len(scores):]))
        if qi%10==0:print(qi,"/",len(data),flush=True)
    # Fixed threshold is declared before scoring. Calibration split is descriptive only.
    threshold=0.8
    held=[r for r in rows if r["split"]=="heldout"]
    metrics=attribution_metrics([r["score"]>=threshold for r in held],[r["label"] for r in held])
    metrics["ci95"]=cluster_ci(held,threshold)
    always=attribution_metrics([True]*len(held),[r["label"] for r in held])
    src=[SearchResult(Chunk("The coolant limit is 300 degrees. The CEO joined in 2020.","d","manual.txt",0,
                           metadata={"filename":"manual.txt"}),1.0,"dense")]
    examples=[]
    for claim in ("The coolant limit is 300 degrees","The coolant limit is 900 degrees","The CEO resigned in 2019"):
        answer=claim+" (Source: manual.txt)."
        examples.append({"answer":answer,"citation_existence":verify_citations(answer,src)[1].validity,
                         "attribution":attribute(answer,src,judge,threshold)})
    report={"provenance":{"source":metadata,"source_snapshot":source_provenance(),"model":judge.identity,"python":platform.python_version(),
        "versions":{k:importlib.metadata.version(k) for k in ("torch","transformers")},
        "split_seed":"attribution-v1","n_calibration_claims":100,"n_heldout_claims":200,
        "threshold":threshold,"max_sequence_length":256,"device":args.device,"batch_size":args.batch_size,"bootstrap_unit":"claim, preserving five evidence pairs",
        "label_source":"Existing independent human evidence labels, not new human review of generated answers"},
        "nli":metrics,"citation_existence_baseline":always,"examples":examples,
        "limitations":["Single-domain transfer evaluation","Source annotations include disagreement",
            "Possible pretraining contamination is unmeasured","Does not validate generated DocMind answers"]}
    (out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    (out/"scores.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=="__main__":main()
