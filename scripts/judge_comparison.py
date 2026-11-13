"""Compare cited-span attribution judges on a fresh human-labeled claim split."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
os.environ.setdefault("DOCMIND_HOME",str(ROOT/"reports"/"judge_workspace"))
from public_benchmark import get_json,source_provenance
from attribution_benchmark import cluster_ci
from docmind.attribution import NLIJudge,attribute,attribution_metrics
from docmind.models import Chunk,SearchResult


def acquire(out):
    import pyarrow.parquet as pq
    metadata=get_json("https://huggingface.co/api/datasets/tdiggelm/climate_fever")
    path=next(s["rfilename"] for s in metadata["siblings"] if s["rfilename"].endswith(".parquet"))
    source=out/"source";source.mkdir(exist_ok=True)
    parquet=source/"climate.parquet"
    urllib.request.urlretrieve("https://huggingface.co/datasets/tdiggelm/climate_fever/resolve/"+metadata["sha"]+"/"+path,parquet)
    data=pq.read_table(parquet).to_pylist()
    data.sort(key=lambda x:hashlib.sha256(("attribution-v1:"+x["claim_id"]).encode()).hexdigest())
    return {"calibration":data[:100],"heldout":data[300:500]}, {"dataset":"tdiggelm/climate_fever",
        "revision":metadata["sha"],"path":path,"sha256":hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "license":"Unspecified; no source sentences are redistributed"}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--device",default="cuda");args=parser.parse_args()
    out=ROOT/"reports"/"judges";out.mkdir(parents=True,exist_ok=True)
    data,source=acquire(out)
    results={};identities={};all_rows=[]
    for model in ("typeform/distilbert-base-uncased-mnli","facebook/bart-large-mnli"):
        revision=get_json("https://huggingface.co/api/models/"+model)["sha"]
        judge=NLIJudge(model,revision=revision,local_files_only=False,device=args.device,batch_size=16)
        identities[model]=judge.identity
        rows=[]
        for split,items in data.items():
            for qi,item in enumerate(items):
                for ei,evidence in enumerate(item["evidences"]):
                    source_result=SearchResult(Chunk(evidence["evidence"],"d","evidence.txt",0,
                        metadata={"filename":"evidence.txt"}),1.0,"dense")
                    answer=item["claim"]+" (Source: evidence.txt)"
                    report=attribute(answer,[source_result],judge,.8,claim_units=[answer])
                    claim=report["claims"][0]
                    row={"claim_id":item["claim_id"],"evidence_index":ei,"split":split,"model":model,
                        "label":evidence["evidence_label"]==0,"score":claim["entailment"],
                        "supported":claim["supported"],"start":claim["start"],"end":claim["end"],
                        "quote_sha256":hashlib.sha256((claim["quote"] or "").encode()).hexdigest(),
                        "quote_roundtrip":claim["quote"]==evidence["evidence"][claim["start"]:claim["end"]] if claim["quote"] else False}
                    rows.append(row)
                if qi%20==0:print(model,split,qi,"/",len(items),flush=True)
        results[model]={}
        for split in ("calibration","heldout"):
            subset=[r for r in rows if r["split"]==split]
            metrics=attribution_metrics([r["supported"] for r in subset],[r["label"] for r in subset])
            metrics["ci95_claim_bootstrap"]=cluster_ci(subset,.8)
            metrics["quote_roundtrip_rate"]=sum(r["quote_roundtrip"] for r in subset)/len(subset)
            results[model][split]=metrics
        all_rows.extend(rows)
        del judge;gc.collect()
        import torch;torch.cuda.empty_cache()
    def calibration_f1(model):
        precision=results[model]["calibration"]["precision"] or 0
        recall=results[model]["calibration"]["recall"] or 0
        return 2*precision*recall/(precision+recall) if precision+recall else 0
    selected=max(results,key=calibration_f1)
    summary={"provenance":{"source":source,"source_snapshot":source_provenance(),"models":identities,
        "claim_split":"100 calibration claims at hash positions 0:100; fresh 200 held-out claims at positions 300:500",
        "threshold":.8,"scope":"Human-defined claims, actual cited-span selection and NLI; no automatic claim-segmentation validation",
        "model_selection":"Calibration F1; fixed threshold; held-out labels excluded from selection",
        "bootstrap_unit":"Claim, preserving five evidence pairs","n_heldout_pairs":1000},
        "results":results,"selected_on_calibration":selected,
        "limitation":"Evidence labels are human sentence-level support, not human minimal-span annotations"}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    (out/"records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in all_rows))
    print(json.dumps(results,indent=2),flush=True)

if __name__=="__main__":main()
