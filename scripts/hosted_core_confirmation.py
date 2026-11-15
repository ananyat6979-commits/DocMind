"""Disabled-by-default exploratory hosted confirmation. Dry runs never import the provider.

The current release does not authorize new hosted evaluation calls. An operator
must explicitly enable both gates for a later, confirmed free-tier-only run.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
MAX_CALLS=500
MAX_CHARGED_TOKENS=1000000
MAX_SECONDS=7200


def permitted(flag,environment):
    return bool(flag) and environment.get('DOCMIND_ALLOW_HOSTED_EVALUATION')=='1'


def committed(path):
    relative=path.resolve().relative_to(ROOT)
    subprocess.run(['git','ls-files','--error-unmatch',str(relative)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--quiet','HEAD','--',str(relative)],cwd=ROOT,check=True)


def load_design(args):
    from docmind.evaluation.paired import digest
    for path in (args.local_summary,args.local_design,args.local_numeric):committed(path)
    summary=json.loads(args.local_summary.read_text(encoding='utf-8'))
    design=json.loads(args.local_design.read_text(encoding='utf-8'))
    if not all(summary.get('arms',{}).get(arm,{}).get('completed',0)>0 for arm in ('single','react')):
        raise ValueError('Committed completed local results are required before hosted confirmation')
    if summary.get('cohort_sha256')!=design.get('cohort_sha256'):raise ValueError('Local cohort identities differ')
    numeric=json.loads(args.local_numeric.read_text(encoding='utf-8'))
    expected={row['id']:row['evidence_sha256'] for row in numeric if row['arm']=='single'}
    inputs=json.loads(args.frozen_inputs.read_text(encoding='utf-8'))
    lookup={row['id']:row for row in inputs}
    if len(lookup)!=len(inputs):raise ValueError('Duplicate frozen inputs')
    rows=design['rows'][:100]
    for row in rows:
        item=lookup[row['id']]
        actual=digest([(source['chunk_id'],source['content']) for source in item['sources']])
        if actual!=expected.get(row['id']):raise ValueError('Evidence differs from captured local inputs')
        if not isinstance(item.get('reference'),str) or not isinstance(item.get('question'),str):raise ValueError('Missing public question/reference')
    plan={'scope':'Exploratory free-tier provider confirmation under frozen evidence, not a hosted default validation',
        'rows':rows,'cohort_sha256':digest(rows),'limits':{'attempted_calls':MAX_CALLS,'conservatively_charged_tokens':MAX_CHARGED_TOKENS,'wall_seconds':MAX_SECONDS},
        'frozen_input_sha256':hashlib.sha256(args.frozen_inputs.read_bytes()).hexdigest(),
        'worst_case_calls':len(rows)*5,'execution_enabled':False,'retries':0,'paid_fallback':False,
        'charging':'Before each attempt: serialized message UTF-8 bytes plus 1024 tokens per message plus output cap. This conservative protocol charge is not a billing record.',
        'token_ceiling_may_prevent_serving_entire_cohort':True}
    return plan,lookup


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=True));temporary.replace(path)


async def execute(plan,lookup,out):
    # No provider import, key read, client construction or model listing above this line.
    from docmind.llm.groq_client import GroqLLM,RateLimitExhausted
    from docmind.agent.reader import CitedReader,CitedReActReader
    from docmind.evaluation.paired import score,summarize
    from docmind.models import Chunk,SearchResult
    os.environ['DOCMIND_ATTRIBUTION']='0'
    end=time.monotonic()+MAX_SECONDS;charged=0;attempts=0;records=[];stop=False
    class Ceiling(RuntimeError):pass
    class Bounded(GroqLLM):
        async def acomplete(self,messages,**kwargs):
            nonlocal charged,attempts
            cost=len(json.dumps(messages,ensure_ascii=False).encode())+1024*(len(messages)+1)+int(kwargs.get('max_tokens',512))
            if attempts>=MAX_CALLS or charged+cost>MAX_CHARGED_TOKENS or time.monotonic()>=end:raise Ceiling('Exploratory run budget exhausted')
            charged+=cost;attempts+=1
            kwargs['retries']=0
            kwargs['deadline']=min(kwargs.get('deadline',end),end)
            return await super().acomplete(messages,**kwargs)
    class Frozen:
        def __init__(self,sources):self.sources=sources
        def search(self,query,file_ids=None):return self.sources
    for row in plan['rows']:
        for arm in row['order']:
            if stop or time.monotonic()>=end:break
            item=lookup[row['id']];sources=[]
            for value in item['sources']:
                chunk=Chunk(value['content'],'frozen','public-article',0,chunk_id=value['chunk_id'],file_id=value['file_id'],metadata=value['metadata'])
                sources.append(SearchResult(chunk,0,'frozen'))
            started=time.monotonic();before_calls=attempts;backend=None
            record={'id':row['id'],'arm':arm,'status':'error','exact_match':0,'evidence_sha256':hashlib.sha256(json.dumps(item['sources'],sort_keys=True).encode()).hexdigest()}
            try:
                backend=Bounded()
                engine=(CitedReader if arm=='single' else CitedReActReader)(Frozen(sources),backend,deadline_seconds=min(180,max(.01,end-time.monotonic())))
                response=await engine.arun(item['question'])
                quote=response.citation_report.get('quote_validation',{})
                failure=response.finish_reason=='format_failure' or not quote.get('short_answer_explicit') or not quote.get('short_answer_valid', False)
                record.update(status='format_failure' if failure else 'completed',format_failure=failure,
                    short_answer_words=len((response.short_answer or '').split()),answer=response.answer,short_answer=response.short_answer,
                    release_state=response.release_state,hold_reasons=response.hold_reasons)
                if not failure:record.update(score('' if response.abstained else response.short_answer or '',item['reference']))
            except (Ceiling,RateLimitExhausted) as exc:
                record['error']=type(exc).__name__;stop=True
            except Exception as exc:record['error']=type(exc).__name__
            finally:
                call_records=getattr(backend,'call_records',[]) if backend is not None else []
                known=all(c.get('input_tokens') is not None and c.get('output_tokens') is not None for c in call_records)
                record.update(calls=attempts-before_calls,tokens=sum(c['input_tokens']+c['output_tokens'] for c in call_records) if known else None,
                    usage_unknown_calls=sum(c.get('input_tokens') is None or c.get('output_tokens') is None for c in call_records),
                    call_records=call_records,elapsed_seconds=time.monotonic()-started)
                if backend is not None:await backend.aclose()
            records.append(record);write(out/'records.private.json',records)
            write(out/'summary.json',summarize({'rows':plan['rows'],'cohort_sha256':plan['cohort_sha256'],'scope':plan['scope']},records))
            write(out/'usage.json',{'attempted_calls':attempts,'conservatively_charged_tokens':charged,'remaining_seconds':max(0,end-time.monotonic()),'paid_fallback':False})
        if stop:break


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('local-summary','local-design','local-numeric','frozen-inputs','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--enable-hosted',action='store_true');parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    if args.enable_hosted and not permitted(args.enable_hosted,os.environ):
        parser.error('Hosted execution disabled: both --enable-hosted and DOCMIND_ALLOW_HOSTED_EVALUATION=1 are required')
    if args.enable_hosted and args.dry_run:parser.error('Choose a dry run or explicitly enabled execution')
    out=args.out.resolve();private_root=(ROOT/'reports/release_workspace').resolve()
    if private_root not in out.parents:parser.error('Keep all provider outputs under the ignored release workspace')
    if out.exists():raise FileExistsError('Existing attempts cannot be rerun or overwritten')
    plan,lookup=load_design(args);write(out/'plan.json',plan)
    print(json.dumps({'requested_pairs':len(plan['rows']),'order_sha256':plan['cohort_sha256'],'worst_case_calls':plan['worst_case_calls'],'limits':plan['limits'],'execution_enabled':permitted(args.enable_hosted,os.environ)}))
    if permitted(args.enable_hosted,os.environ):
        plan['execution_enabled']=True;write(out/'plan.json',plan);asyncio.run(execute(plan,lookup,out))


if __name__=='__main__':main()
