"""Bounded private own-document local-reader smoke test. No hosted calls or label synthesis."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True).encode()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=True),encoding='utf-8');temp.replace(path)


def prepare(root,original,confirmation,out):
    if out.exists():raise FileExistsError('Never overwrite a frozen cohort')
    packet=json.loads((root/'confirmation-packet.private.json').read_text(encoding='utf-8'))
    labels=json.loads(confirmation.read_text(encoding='utf-8'))
    if labels.get('packet_sha256')!=packet['sha256']:raise ValueError('Confirmation packet mismatch')
    original_bytes=original.read_bytes()
    if hashlib.sha256(original_bytes).hexdigest()!=packet['original_export_sha256']:raise ValueError('Original export changed')
    records=labels['records'];lookup={r['id']:r for r in records}
    if len(lookup)!=len(records):raise ValueError('Repeated human decisions')
    proposals={r['id']:r for r in packet['items']}
    if not set(lookup)<=set(proposals):raise ValueError('Unknown reviewed item')
    gold=[]
    # One unchanged, genuinely human-authored record from the first export.
    first=json.loads(original_bytes)
    for r in first['records']:
        if r.get('id')!='nist-33' or not r.get('finished') or r.get('question_validity')!='valid' or r.get('answerable')!='yes':continue
        gold.append({'id':r['id'],'question':r['question'],'answer':r['answer'],'aliases':r.get('aliases',''),'support_quote':r['support_quote'],'answerable':True,'group':'nist','source_page':33,'origin':'human_authored','source_filename':'nist-sp800-63b-4.pdf'})
    for identifier,r in lookup.items():
        if not r.get('finished') or not r.get('human_adjudicated'):continue
        if (r.get('question_validity'),r.get('answer_correctness'),r.get('evidence_support'))!=('valid','correct','supported'):continue
        p=proposals[identifier]
        if not r.get('question','').strip() or not r.get('answer','').strip():raise ValueError('Empty accepted field')
        if p['answerable'] and not r.get('support_quote','').strip():raise ValueError('Missing evidence')
        if not p['answerable'] and not r.get('absence_note','').strip():raise ValueError('Missing absence rationale')
        gold.append({**{k:p[k] for k in ('id','group','source_page','source_passage','origin','answerable')},**{k:r.get(k,'') for k in ('question','answer','aliases','support_quote','absence_note')},'source_filename':'docmind-operations-guide.pdf' if p['group']=='project' else 'nist-sp800-63b-4.pdf'})
    if not gold:raise ValueError('No eligible human-confirmed questions')
    import random
    random.Random(7).shuffle(gold)
    tasks=[]
    for r in gold:
        tasks.append({**r,'task_id':r['id']+':pdf','format':'pdf'})
    # Same underlying questions are a paired format diagnostic, not independent N.
    for r in gold:
        if r['group']=='project':tasks.append({**r,'task_id':r['id']+':docx','format':'docx','source_filename':'docmind-operations-guide.docx'})
    cache=json.loads((root/'parsed-cache.private.json').read_text(encoding='utf-8'))
    for d in cache['documents']:
        if digest(d['units'])!=d['units_sha256']:raise ValueError('Parser cache changed')
        if hashlib.sha256((root/'corpus'/d['filename']).read_bytes()).hexdigest()!=d['sha256']:raise ValueError('Source PDF/DOCX changed')
    # Human changes are retained. Location failures are diagnostics, never semantic relabeling.
    import re,unicodedata
    def normal(t):return ' '.join(unicodedata.normalize('NFC',t).split())
    docs={d['filename']:d for d in cache['documents']}
    for t in tasks:
        units=docs[t['source_filename']]['units']
        t['gold_quote_location_found']=bool(t['support_quote']) and any(normal(t['support_quote']) in normal(u['content']) for u in units)
    design={'scope':'English own-document cached-parsing local-reader smoke test, same reviewer confirmed sources; not independent annotation or hosted product validation','seed':7,'pdf_questions':len(gold),'requested_tasks':len(tasks),'cohort_sha256':digest(tasks),'confirmation_packet_sha256':packet['sha256'],'confirmation_export_sha256':hashlib.sha256(confirmation.read_bytes()).hexdigest(),'original_export_sha256':hashlib.sha256(original_bytes).hexdigest(),'parser_cache_sha256':hashlib.sha256((root/'parsed-cache.private.json').read_bytes()).hexdigest(),'labels_checked':len(records),'labels_finished':sum(bool(r.get('finished')) for r in records),'recorded_confirmation_seconds':labels['elapsed_ms']/1000,'selection_uses_model_outcomes':False,'wall_cap_seconds':1800,'reserved_gpu_seconds':3600,'output_token_cap':1024,'retries':0,'paired_docx_is_additional_independent_sample':False,'docx_layout_visually_verified':False,'source_license_notes':{'guide':'Original project fixture','nist':'NIST public US-government publication; attribution retained; private source copy'}}
    write(out/'private/gold.json',tasks);write(out/'design.json',design)
    return design


def summarize(design,records):
    lookup={r['task_id']:r for r in records}
    if len(lookup)!=len(records):raise ValueError('Repeated task attempts')
    groups={}
    for kind in ('pdf','docx'):
        rows=[r for r in records if r['format']==kind];n=design['pdf_questions'] if kind=='pdf' else design['requested_tasks']-design['pdf_questions']
        complete=[r for r in rows if r['status']=='completed'];correct=sum(r.get('exact_match',0) for r in rows)
        groups[kind]={'requested':n,'attempted':len(rows),'format_completed':len(complete),'unserved':n-len(rows),'format_failures':sum(r['status']=='format_failure' for r in rows),'runtime_errors':sum(r['status']=='error' for r in rows),'candidate_em_correct':correct,'requested_candidate_em':correct/n if n else None,'drafts':sum(r.get('release_state')=='draft' for r in rows),'held':sum(r.get('release_state')=='held' for r in rows),'numerical_failures':sum(r.get('numerical_failure',False) for r in rows),'truncations':sum(r.get('truncated',False) for r in rows),'human_answer_correctness':'pending separate actual-answer review','false_release_risk':'insufficient evidence'}
    return {'scope':design['scope'],'cohort_sha256':design['cohort_sha256'],'groups':groups,'paid_compute':False,'hosted_calls':0}


def finalize(out):
    design=json.loads((out/'design.json').read_text());records=json.loads((out/'private/records.json').read_text()) if (out/'private/records.json').exists() else []
    write(out/'summary.json',summarize(design,records));write(out/'numeric_records.json',[{k:v for k,v in r.items() if k not in ('question','answer','short_answer','evidence','sources','raw_outputs','document_context')} for r in records])


def run(root,out):
    os.environ.pop('GROQ_API_KEY',None)
    os.environ.update(DOCMIND_TEST_MODE='0',DOCMIND_HOME=str(out/'runtime'),DOCMIND_SECRETS_FILE=str(out/'no-provider.env'),DOCMIND_ATTRIBUTION='0',DOCMIND_DEVICE='cuda',DOCMIND_USE_RERANKER='0',DOCMIND_CHUNKER='recursive',DOCMIND_RETRIEVAL_MODE='dense')
    deadline=float(os.environ['DOCMIND_EVAL_DEADLINE_UNIX'])
    if not 0<deadline-time.time()<=1800:raise ValueError('Own-document profile exceeds reservation')
    import torch
    torch.set_num_threads(2)
    from docmind.models import Document
    from docmind.ingestion import pipeline
    from docmind.store import CorpusStore
    from docmind.retrieval.hybrid import HybridRetriever
    from docmind.agent.reader import CitedReader
    from docmind.llm.local_reader import LocalReader
    from docmind.evaluation.paired import score
    design=json.loads((out/'design.json').read_text());tasks=json.loads((out/'private/gold.json').read_text())
    if digest(tasks)!=design['cohort_sha256']:raise ValueError('Frozen cohort changed')
    cache_bytes=(root/'parsed-cache.private.json').read_bytes()
    if hashlib.sha256(cache_bytes).hexdigest()!=design['parser_cache_sha256']:raise ValueError('Parser cache changed')
    cache=json.loads(cache_bytes);docs={d['filename']:d for d in cache['documents']}
    def replay(path,expected_language=None,encoding=None):
        d=docs[path.name]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=d['sha256'] or digest(d['units'])!=d['units_sha256']:raise ValueError('Parsed source hash mismatch')
        return [Document(**json.loads(json.dumps(u))) for u in d['units']]
    pipeline.load_document=replay
    store=CorpusStore();files={};ingestion=[]
    for name in sorted({t['source_filename'] for t in tasks}):
        started=time.monotonic();report=pipeline.ingest(root/'corpus'/name,store=store,expected_language='en');files[name]=report.file_ids[0]
        ingestion.append({'filename':name,'elapsed_seconds':time.monotonic()-started,'report':asdict(report)})
    write(out/'ingestion.json',ingestion)
    raw=[]
    class Recorded(LocalReader):
        def complete(self,messages,stop=None,**kwargs):
            result=super().complete(messages,stop=stop,**kwargs);raw.append(result);return result
    llm=Recorded(max_input_tokens=8192,max_new_tokens=1024);retriever=HybridRetriever(store=store)
    write(out/'reader_identity.json',{**llm.identity,'actual_dtype':str(next(llm.model.parameters()).dtype),'gpu':torch.cuda.get_device_name(0)})
    records=[]
    for task in tasks:
        if time.time()>deadline-20:break
        started=time.monotonic();before=len(llm.call_records);raw_before=len(raw)
        row={'task_id':task['task_id'],'format':task['format'],'status':'error','exact_match':0,'answerable':task['answerable'],'gold_quote_location_found':task['gold_quote_location_found']}
        try:
            engine=CitedReader(retriever,llm,deadline_seconds=min(180,max(.01,deadline-time.time()-15)))
            engine.output_tokens=1024
            response=engine.run(task['question'],file_ids=[files[task['source_filename']]])
            failure=response.finish_reason=='format_failure'
            row.update(status='format_failure' if failure else 'completed',question=task['question'],answer=response.answer,short_answer=response.short_answer,abstained=response.abstained,release_state=response.release_state,hold_reasons=response.hold_reasons,evidence=response.evidence,document_context=response.document_context,sources=[{'chunk_id':s.chunk.chunk_id,'content':s.chunk.content,'metadata':s.chunk.metadata} for s in response.sources])
            if not failure:
                if not task['answerable']:row['exact_match']=int(response.abstained)
                else:
                    aliases=[task['answer']]+[a.strip() for a in task['aliases'].split('|') if a.strip()]
                    scores=[score(response.short_answer or '',a) for a in aliases];row.update(max(scores,key=lambda r:(r['exact_match'],r['token_f1'])))
        except Exception as exc:row['error']=type(exc).__name__
        calls=llm.call_records[before:]
        row.update(call_records=calls,calls=len(calls),tokens=sum(c['input_tokens']+c['output_tokens'] for c in calls) if all(c.get('input_tokens') is not None and c.get('output_tokens') is not None for c in calls) else None,elapsed_seconds=time.monotonic()-started,numerical_failure=any(c.get('numerical_failure') for c in calls),truncated=any(c.get('max_length_reached') for c in calls),raw_outputs=raw[raw_before:])
        records.append(row);write(out/'private/records.json',records);finalize(out);print(len(records),task['task_id'],row['status'],flush=True)
    write(out/'memory.json',{'peak_allocated_device0_bytes':torch.cuda.max_memory_allocated()});finalize(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--original',type=Path);p.add_argument('--confirmation',type=Path);p.add_argument('--prepare-only',action='store_true');p.add_argument('--summarize-only',action='store_true');a=p.parse_args()
    if a.prepare_only:
        if a.original is None or a.confirmation is None:p.error('Human exports required')
        print(json.dumps(prepare(a.root,a.original,a.confirmation,a.out)))
    elif a.summarize_only:finalize(a.out)
    else:
        if (a.out/'private/records.json').exists():raise FileExistsError('Never rerun existing attempts')
        run(a.root,a.out)
