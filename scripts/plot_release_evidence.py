"""Render verified retrieval component results as standalone research figures."""
import argparse
import hashlib
import importlib.metadata as md
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--report",type=Path,required=True);parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):raise ValueError("Choose a fresh figure directory")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    data=json.loads(args.report.read_text(encoding="utf-8"));args.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"svg.hashsalt":"docmind-r1","axes.spines.top":False,"axes.spines.right":False})
    modes=[("bge_dense","BGE dense"),("e5_dense","E5 dense"),("e5_hybrid_word","E5 + word hybrid"),("e5_hybrid_word_english_rerank20","Hybrid + English rerank 20"),("e5_hybrid_word_multilingual_rerank20","Hybrid + multilingual rerank 20")]
    fig,axes=plt.subplots(2,2,figsize=(11.5,8.5),sharey=True)
    positions=np.arange(len(modes));colors=["#536574","#157a66","#536574","#536574","#536574"]
    for column,(language,name) in enumerate((("en","English"),("hi","Hindi"))):
        rows=[data["results"][language+"_"+mode] for mode,_ in modes]
        if any(r["status"]!="complete" or r["n_completed"]!=200 for r in rows):raise ValueError("Figure requires the complete declared cohort")
        points=np.array([r["metrics"]["recall5"]["requested_mean"]*100 for r in rows]);intervals=np.array([r["metrics"]["recall5"]["requested_ci95"] for r in rows])*100
        axis=axes[0,column];axis.barh(positions,points,color=colors,height=.65)
        axis.errorbar(points,positions,xerr=np.array([points-intervals[:,0],intervals[:,1]-points]),fmt="none",ecolor="#17222d",capsize=3,linewidth=1)
        axis.set_xlim(0,100);axis.set_title(name+" relevance");axis.set_xlabel("Recall@5 (%) with 95% intervals")
        for i,point in enumerate(points):axis.text(intervals[i,1]+1,positions[i],f"{point:.1f}",va="center",fontsize=9)
        latency=np.array([r["warm_p50_ms"] for r in rows]);axis=axes[1,column]
        axis.barh(positions,latency,color=colors,height=.65);axis.set_xscale("log");axis.set_xlim(1,3000);axis.set_title(name+" warm retrieval cost");axis.set_xlabel("Median milliseconds, logarithmic scale")
        for i,value in enumerate(latency):axis.text(value*1.08,positions[i],f"{value:.1f}",va="center",fontsize=9)
    for axis in axes.flat:
        axis.set_yticks(positions,[label for _,label in modes]);axis.grid(axis="x",alpha=.18);axis.set_axisbelow(True)
    axes[0,0].invert_yaxis()
    fig.suptitle("English/Hindi retrieval components",fontsize=16,y=.98)
    fig.text(.02,.035,'200 public dev questions per language; pooled judged candidates with all positives retained, not full-corpus MIRACL.\nIntervals are nominal percentile bootstrap. Cost excludes model loading, corpus preparation and generation; hybrid latency sums measured components.\nGreen identifies the engineering default. English noninferiority and Hindi PDF attribution are not established.',fontsize=8,va="bottom")
    fig.tight_layout(rect=(.01,.12,.99,.95))
    fig.savefig(args.out/"retrieval_components.png",dpi=300,metadata={"Software":"DocMind"})
    fig.savefig(args.out/"retrieval_components.svg",metadata={"Date":None,"Creator":"DocMind"})
    plt.close(fig)
    manifest={"source_report_canonical_sha256":hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"matplotlib":md.version("matplotlib"),"numpy":md.version("numpy"),"files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()}}
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__=="__main__":main()
