"""Fresh CPU component baselines. No hosted requests or reader generation."""
import argparse
import importlib.metadata as md
import os
from pathlib import Path
import platform
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,default=ROOT/"reports/release_workspace/cpu-components");args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    os.environ.update(DOCMIND_TEST_MODE="0",DOCMIND_DEVICE="cpu",DOCMIND_HOME=str(args.out/"runtime"),DOCMIND_SECRETS_FILE=str(args.out/"no-provider-secrets.env"))
    os.environ.pop("GROQ_API_KEY",None)
    import torch
    from release_measure import acquire_xnli,nli_components,ann_component,write
    torch.set_num_threads(2)
    result={"scope":"English/Hindi CPU component baseline, not the T4 reader or hosted default",
        "device":"cpu","platform":platform.platform(),"python":platform.python_version(),
        "versions":{k:md.version(k) for k in ("torch","transformers","sentence-transformers","tokenizers")},
        "human_verification":"not independently reviewed","results":{}}
    started=time.perf_counter();data=acquire_xnli(args.out)
    for name,operation in (("xnli",lambda:nli_components(data,args.out,"cpu")),("ann",lambda:ann_component(data,args.out,"cpu"))):
        try:result["results"][name]=operation()
        except Exception as exc:
            import traceback;traceback.print_exc();result["results"][name]={"error":type(exc).__name__,"detail":str(exc)[:500]}
        result["elapsed_seconds"]=time.perf_counter()-started;write(args.out/"summary.json",result)
        print("Finished CPU",name,flush=True)
    if any("error" in item for item in result["results"].values()):raise SystemExit(1)

if __name__=="__main__":main()
