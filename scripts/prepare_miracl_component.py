"""Freeze a bounded judged-candidate MIRACL task without retaining full corpus shards.

The pool deliberately includes all selected-query positives and up to ten judged
negatives per query, plus a deterministic corpus sample. It is not full-corpus MIRACL.
"""
import argparse
import gzip
import hashlib
import heapq
import json
from pathlib import Path
import re
import time
import urllib.request

QUERIES_REV="5be20db9509754dadad47689368639fcec739c00"
CORPUS_REV="d921ec7e349ce0d28daf30b2da9da5ee698bef0d"
_DOCID=re.compile(rb'"docid"\s*:\s*"([^"\\]+)"')


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"DocMind-public-evaluation"}),timeout=90)


def write(path,value):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=True),encoding="utf-8");temp.replace(path)


def priority(language,value):
    return int.from_bytes(hashlib.sha256(("miracl-pool-v1:7:"+language+":"+value).encode()).digest()[:8],"big")


class CandidateSampler:
    def __init__(self,language,required,capacity):
        self.language=language;self.required=set(required);self.capacity=capacity
        self.sample={};self.found={};self.heap=[];self.seen=0

    def offer(self,document_id,load):
        self.seen+=1
        score=priority(self.language,document_id)
        selected=len(self.heap)<self.capacity or score < -self.heap[0][0]
        if not selected and document_id not in self.required:return
        record=load()
        if record["docid"]!=document_id:raise ValueError("Corpus identity mismatch")
        if document_id in self.required:self.found[document_id]=record
        if selected:
            if document_id in self.sample:return
            if len(self.heap)>=self.capacity:
                _,old=heapq.heappop(self.heap);self.sample.pop(old,None)
            heapq.heappush(self.heap,(-score,document_id));self.sample[document_id]=record

    def documents(self):
        return sorted({**self.sample,**self.found}.values(),key=lambda row:row["docid"])


class HashedStream:
    def __init__(self,stream):self.stream=stream;self.hash=hashlib.sha256();self.count=0
    def read(self,size=-1):
        part=self.stream.read(size);self.hash.update(part);self.count+=len(part);return part


def prepare(language,directory,metadata,n=200,distractors=20000):
    prefix=f"https://huggingface.co/datasets/miracl/miracl/resolve/{QUERIES_REV}/miracl-v1.0-{language}/"
    with get(prefix+f"topics/topics.miracl-v1.0-{language}-dev.tsv") as response:topics=response.read()
    with get(prefix+f"qrels/qrels.miracl-v1.0-{language}-dev.tsv") as response:labels=response.read()
    queries=dict(line.split("\t",1) for line in topics.decode("utf-8-sig").splitlines() if line.strip())
    judgments={}
    for line in labels.decode("utf-8-sig").splitlines():
        qid,_,docid,grade=line.split();judgments.setdefault(qid,{})[docid]=int(grade)
    eligible=[qid for qid in queries if any(v>0 for v in judgments.get(qid,{}).values())]
    selected=sorted(eligible,key=lambda qid:priority(language,qid))[:n]
    if len(selected)!=n:raise ValueError("Insufficient public queries")
    required=set()
    for qid in selected:
        required.update(docid for docid,grade in judgments[qid].items() if grade>0)
        required.update(sorted((docid for docid,grade in judgments[qid].items() if grade==0),key=lambda docid:priority(language,docid))[:10])
    sampler=CandidateSampler(language,required,distractors)
    rows=[{"id":qid,"question":queries[qid],"qrels":judgments[qid]} for qid in selected]
    manifest={"language":language,"n_requested":n,"query_revision":QUERIES_REV,"corpus_revision":CORPUS_REV,
        "topics_sha256":hashlib.sha256(topics).hexdigest(),"qrels_sha256":hashlib.sha256(labels).hexdigest(),
        "selected_ids_sha256":hashlib.sha256(json.dumps(selected).encode()).hexdigest(),
        "pool_scope":"All selected-query positives, up to ten judged negatives per query, plus hash-selected distractors; not full-corpus MIRACL",
        "distractor_cap":distractors,"human_labels":"Original public judgments, not independently reviewed",
        "licenses":["MIRACL card declares Apache-2.0","Underlying Wikipedia text has its own attribution/share-alike terms; raw text stays private"],"shards":[]}
    shards=[r for r in metadata["siblings"] if r["rfilename"].startswith(f"miracl-corpus-v1.0-{language}/") and r["rfilename"].endswith(".jsonl.gz")]
    shards.sort(key=lambda r:int(r["rfilename"].rsplit("/",1)[-1].split("-")[1].split(".")[0]))
    for number,shard in enumerate(shards):
        expected=shard.get("lfs",{}).get("sha256")
        if not expected:raise ValueError("Missing immutable corpus hash")
        url=f"https://huggingface.co/datasets/miracl/miracl-corpus/resolve/{CORPUS_REV}/"+shard["rfilename"]
        start=time.perf_counter()
        with get(url) as response:
            stream=HashedStream(response)
            with gzip.GzipFile(fileobj=stream) as compressed:
                for line in compressed:
                    if not line.strip():continue
                    match=_DOCID.search(line[:256])
                    if match:
                        document_id=match.group(1).decode("utf-8")
                        sampler.offer(document_id,lambda line=line:json.loads(line))
                    else:
                        record=json.loads(line);sampler.offer(record["docid"],lambda record=record:record)
            # Include any trailing compressed bytes in the immutable source check.
            while stream.read(1<<20):pass
        actual=stream.hash.hexdigest()
        if actual!=expected:raise ValueError("Corpus shard hash mismatch")
        manifest["shards"].append({"file":shard["rfilename"],"sha256":actual,"bytes":stream.count,"seconds":time.perf_counter()-start})
        manifest.update(documents_seen=sampler.seen,required=len(required),required_found=len(sampler.found),complete=False)
        write(directory/(language+"-pool.json"),{"manifest":manifest,"queries":rows,"documents":sampler.documents()})
        print(language,number+1,"/",len(shards),"required",len(sampler.found),"/",len(required),flush=True)
    manifest["complete"]=set(sampler.found)==required
    if not manifest["complete"]:manifest["missing_ids"]=sorted(required-set(sampler.found))
    write(directory/(language+"-pool.json"),{"manifest":manifest,"queries":rows,"documents":sampler.documents()})
    return manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True);parser.add_argument("--queries",type=int,default=200);parser.add_argument("--distractors",type=int,default=20000);args=parser.parse_args()
    if not 1<=args.queries<=200 or not 1<=args.distractors<=20000:raise ValueError("Bounded pool limits exceeded")
    args.out.mkdir(parents=True,exist_ok=True)
    with get(f"https://huggingface.co/api/datasets/miracl/miracl-corpus/revision/{CORPUS_REV}?blobs=true") as response:metadata=json.load(response)
    manifests={}
    for language in ("en","hi"):
        manifests[language]=prepare(language,args.out,metadata,args.queries,args.distractors)
        write(args.out/"summary.json",manifests)

if __name__=="__main__":main()
