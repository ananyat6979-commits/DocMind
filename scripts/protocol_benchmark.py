"""Token-capped provider-native versus ReAct comparison with frozen retrieval."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--n",type=int,default=20)
    parser.add_argument("--token-budget",type=int,default=80000)
    parser.add_argument("--secrets-file",default=str(Path.home()/".config"/"docmind"/"groq.env"))
    args=parser.parse_args()
    os.environ["DOCMIND_SECRETS_FILE"]=args.secrets_file
    os.environ["DOCMIND_ATTRIBUTION"]="0"
    os.environ["DOCMIND_HOME"]=str(ROOT/"reports"/"protocol_workspace")
    from dotenv import load_dotenv
    load_dotenv(args.secrets_file,encoding="utf-8-sig")
    from docmind.config import CONFIG
    from docmind.llm.groq_client import GroqLLM
    from docmind.agent.react import ReActAgent
    from docmind.agent.citations import CITATION_RE
    from docmind.models import Chunk,SearchResult
    from docmind.evaluation.metrics import bootstrap_ci
    from public_benchmark import acquire,source_provenance
    from qa_and_robustness import scores
    CONFIG.llm.temperature=0.0;CONFIG.llm.max_tokens=512
    CONFIG.llm.max_retries=1;CONFIG.llm.max_sleep_seconds=60
    CONFIG.review.enabled=False
    out=ROOT/"reports"/"protocols";out.mkdir(parents=True,exist_ok=True)
    data,source=acquire(ROOT/"reports"/"hotpot"/"source")
    questions={r["id"]:r for r in data}
    path=ROOT/"reports"/"benchmarks"/"hotpot"/"records.jsonl.gz"
    with gzip.open(path,"rt",encoding="utf-8") as stream:ranking=[json.loads(line) for line in stream]
    selected=[r for r in ranking if r["split"]=="heldout" and r["embedder"]=="BAAI/bge-small-en-v1.5" and r["headers"] and r["mode"]=="dense"][:args.n]
    class BudgetExceeded(RuntimeError):pass
    class CappedClient(GroqLLM):
        def __init__(self):
            super().__init__(reasoning_effort="none")
            self.accounted=0;self.unknown_usage_reserve=0
        def reserve(self,messages,tools=None):
            # UTF-8 bytes conservatively bound BPE tokens, with schema/message overhead.
            bound=sum(len(str(m.get("content","")).encode())+32 for m in messages)+CONFIG.llm.max_tokens+512
            if tools:bound+=len(json.dumps(tools).encode())
            if self.accounted+bound>args.token_budget:raise BudgetExceeded("Experimental token cap reached")
            return bound,self.prompt_tokens+self.completion_tokens
        def settle(self,reservation,before):
            actual=self.prompt_tokens+self.completion_tokens-before
            self.accounted+=actual if actual else reservation
            if not actual:self.unknown_usage_reserve+=reservation
        def complete(self,messages,stop=None,**kwargs):
            bound,before=self.reserve(messages)
            try:return super().complete(messages,stop=stop,**kwargs)
            finally:self.settle(bound,before)
        def complete_tools(self,messages,tools):
            bound,before=self.reserve(messages,tools)
            try:return super().complete_tools(messages,tools)
            finally:self.settle(bound,before)
    client=CappedClient()
    manifest={"model":client._model,"provider":"Groq","reasoning_effort":"none","temperature":0,
        "max_output_tokens":512,"maximum_llm_calls":4,"n_requested_per_protocol":len(selected),
        "token_cap":args.token_budget,"source":source,"source_snapshot":source_provenance(),
        "question_ids":[r["id"] for r in selected],
        "retrieval":"Frozen dense top-five replay; each paragraph capped at 600 characters; same evidence for both protocols",
        "scope":"Protocol and prompt comparison; NLI and review routing disabled; hosted model revision is opaque"}
    checkpoint=out/"records.jsonl";manifest_path=out/"manifest.json"
    if checkpoint.exists() and checkpoint.stat().st_size:
        raise ValueError("Use a fresh output folder; this bounded comparison does not silently reuse generations")
    manifest_path.write_text(json.dumps(manifest,indent=2))
    class Store:
        def __init__(self,chunks):self.chunks=chunks
        def get_chunk(self,key):
            matches=[c for c in self.chunks if c.chunk_id.startswith(key)]
            return matches[0] if len(matches)==1 else None
        def neighbors(self,chunk,window=1):return [chunk]
    class FrozenRetriever:
        def __init__(self,chunks):self.chunks=chunks;self.store=Store(chunks)
        def search(self,*unused,**kwargs):return [SearchResult(c,1.0,"frozen_dense") for c in self.chunks[:5]]
    records=[];stopped=None
    for index,ranking_row in enumerate(selected):
        item=questions[ranking_row["id"]]
        contexts={hashlib.sha256(title.encode()).hexdigest():(title," ".join(sentences))
            for title,sentences in zip(item["context"]["title"],item["context"]["sentences"])}
        chunks=[]
        for i,title_hash in enumerate(ranking_row["ranked_title_sha256"][:5]):
            title,text=contexts[title_hash]
            chunks.append(Chunk((title+"\n"+text)[:600],str(i),f"doc-{i}.txt",0,
                metadata={"filename":f"doc-{i}.txt"},chunk_id=hashlib.sha256((item["id"]+str(i)).encode()).hexdigest(),file_id=str(i)))
        order=("react","native") if index%2==0 else ("native","react")
        for protocol in order:
            os.environ["DOCMIND_TOOL_PROTOCOL"]=protocol
            before=(client.prompt_tokens,client.completion_tokens,client.calls,client.accounted)
            started=time.perf_counter();error=None;answer="";attempted=not bool(stopped)
            response=None
            if not stopped:
                try:
                    runner=ReActAgent(FrozenRetriever(chunks),client,max_iterations=3 if protocol=="react" else 4)
                    response=runner.run(item["question"]+" Return only the short answer followed by its citation.")
                    answer="" if response.abstained else CITATION_RE.sub("",response.answer).strip()
                except Exception as exc:
                    error=type(exc).__name__
                    if error in {"BudgetExceeded","RateLimitExhausted","RateLimitError"}:stopped=error
            else:error=stopped
            em,f1=scores(answer,item["answer"]) if not error else (0.0,0.0)
            record={"id":item["id"],"protocol":protocol,"answer":answer,"exact_match":em,"token_f1":f1,
                "error":error,"attempted":attempted,"latency_ms":(time.perf_counter()-started)*1000 if attempted else None,
                "prompt_tokens":client.prompt_tokens-before[0],"completion_tokens":client.completion_tokens-before[1],
                "provider_calls":client.calls-before[2],"budget_debited":client.accounted-before[3],
                "abstained":response.abstained if response else None}
            records.append(record)
            with checkpoint.open("a") as stream:stream.write(json.dumps(record,ensure_ascii=True)+"\n")
            print(index+1,protocol,"error="+str(error),"accounted_tokens="+str(client.accounted),flush=True)
    report={"provenance":manifest,"total_prompt_tokens":client.prompt_tokens,"total_completion_tokens":client.completion_tokens,
        "unknown_usage_budget_reserve":client.unknown_usage_reserve,"budget_accounted":client.accounted,
        "stop_reason":stopped,"protocols":{}}
    import numpy as np
    for protocol in ("react","native"):
        rows=[r for r in records if r["protocol"]==protocol];attempts=[r for r in rows if r["attempted"]]
        report["protocols"][protocol]={"n_requested":len(rows),"n_attempted":len(attempts),
            "n_failed":sum(bool(r["error"]) for r in rows),
            "failure_rate":sum(bool(r["error"]) for r in rows)/len(rows),
            "exact_match":dict(zip(("mean","lo","hi"),bootstrap_ci([r["exact_match"] for r in rows]))),
            "token_f1":dict(zip(("mean","lo","hi"),bootstrap_ci([r["token_f1"] for r in rows]))),
            "latency_p50_ms":float(np.percentile([r["latency_ms"] for r in attempts],50)) if attempts else None,
            "latency_p95_ms":float(np.percentile([r["latency_ms"] for r in attempts],95)) if attempts else None,
            "mean_provider_calls_per_attempt":sum(r["provider_calls"] for r in rows)/len(attempts) if attempts else None,
            "total_tokens":sum(r["prompt_tokens"]+r["completion_tokens"] for r in rows)}
    (out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report["protocols"],indent=2),flush=True)

if __name__=="__main__":main()
