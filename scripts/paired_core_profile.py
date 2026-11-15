"""Free local-reader frozen-evidence control. Private source text never becomes public output.

No hosted provider is imported. Evaluation failures are retained without retry.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding='utf-8')
    temporary.replace(path)


def excluded_ids():
    override = os.getenv('DOCMIND_PAIRED_EXCLUSIONS_FILE')
    if override:
        values=json.loads(Path(override).read_text(encoding='utf-8'))
        if not isinstance(values,list) or any(not isinstance(value,str) for value in values):raise ValueError('Invalid historical exclusions')
        return set(values)
    ids = set()
    for directory in ('reports/benchmarks', 'reports/releases'):
        for path in (ROOT/directory).rglob('*.json'):
            if path.stat().st_size > 20*1024*1024:continue
            ids.update(re.findall(r'(?<![0-9a-f])[0-9a-f]{24}(?![0-9a-f])', path.read_text(encoding='utf-8')))
    return ids


def prepare(source, out):
    import pyarrow.parquet as pq
    from calibration_closure import SOURCE_SHA
    from docmind.evaluation.paired import balanced_pool, digest
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:raise ValueError('Public source hash differs')
    excluded = excluded_ids()
    pool = balanced_pool(pq.read_table(source).to_pylist(), excluded, 140)
    if len(pool) < 30:raise ValueError('Insufficient fresh questions')
    pilot = balanced_pool(pool, set(), 20)
    evaluation = balanced_pool(pool, {r['id'] for r in pilot}, 100)
    rows = pilot + evaluation
    clean = [{'id':r['id'], 'type':r['type'], 'question':r['question'], 'reference':r['answer'],
        'titles':r['context']['title'], 'texts':[' '.join(sentences) for sentences in r['context']['sentences']]} for r in rows]
    write(out/'private/pool.json',clean)
    manifest={'source_sha256':SOURCE_SHA, 'source_attribution':'HotpotQA authors and underlying Wikipedia; CC BY-SA 4.0',
        'excluded_ids_sha256':digest(sorted(excluded)), 'excluded_count':len(excluded),
        'pilot_ids':[r['id'] for r in pilot[:10]],'reserve_pilot_ids':[r['id'] for r in pilot[10:]],
        'evaluation_pool_ids':[r['id'] for r in evaluation], 'selection_uses_outcomes':False,
        'mix':{'bridge':.6,'comparison':.4},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(out/'input_manifest.json',manifest)
    return manifest


def run(out, deadline):
    os.environ.pop('GROQ_API_KEY',None)
    os.environ.update(DOCMIND_TEST_MODE='0',DOCMIND_SECRETS_FILE=str(out/'absent.env'),
        DOCMIND_HOME=str(out/'runtime'),DOCMIND_ATTRIBUTION='0',TOKENIZERS_PARALLELISM='false')
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from docmind.agent.reader import CitedReader,CitedReActReader
    from docmind.evaluation.paired import freeze_design, score, digest
    from docmind.llm.local_reader import LocalReader
    from docmind.models import Chunk,SearchResult
    torch.set_num_threads(2)
    manifest=json.loads((out/'input_manifest.json').read_text())
    pool=json.loads((out/'private/pool.json').read_text())
    model=SentenceTransformer('intfloat/multilingual-e5-small',revision='614241f622f53c4eeff9890bdc4f31cfecc418b3',device='cuda:0')
    prepared={}
    for item in pool:
        if time.time() > deadline-120:raise TimeoutError('Setup exhausted profile reservation')
        text=[title+'\n'+content for title,content in zip(item['titles'],item['texts'])]
        # Duplicate article titles are combined before retrieval.
        groups={}
        for title,content in zip(item['titles'],text):
            groups.setdefault(title,[])
            if content not in groups[title]:groups[title].append(content)
        text=['\n'.join(values) for values in groups.values()]
        vectors=model.encode(['passage: '+t for t in text],normalize_embeddings=True)
        query=model.encode(['query: '+item['question']],normalize_embeddings=True)[0]
        order=np.argsort(-(vectors@query),kind='stable')[:5].tolist()
        prepared[item['id']]={'texts':text,'vectors':vectors,'order':order}
    del model;gc.collect();torch.cuda.empty_cache()
    reader=LocalReader(max_input_tokens=8192,max_new_tokens=512)
    write(out/'reader_identity.json',{**reader.identity,'actual_dtype':str(next(reader.model.parameters()).dtype),
        'gpu':torch.cuda.get_device_name(0),'allocated_device_count':torch.cuda.device_count()})
    if next(reader.model.parameters()).dtype != torch.float16:raise ValueError('FP16 profile required')
    class Frozen:
        def __init__(self,results):self.results=results
        def search(self,question,file_ids=None):return self.results
    def evidence(item):
        prepared_item=prepared[item['id']]
        results=[]
        for index in prepared_item['order']:
            original=prepared_item['texts'][index]
            content=reader.tokenizer.decode(reader.tokenizer.encode(original,add_special_tokens=False)[:384])
            identity=hashlib.sha256((item['id']+':'+str(index)+':'+content).encode()).hexdigest()
            chunk=Chunk(content,identity,'hotpot-public',index,chunk_id=identity,file_id=identity,
                metadata={'filename':'HotpotQA supplied article '+str(index),'language_info':{'language':'en'},
                    'document_language_info':{'language':'en'}})
            results.append(SearchResult(chunk,0,'dense'))
        return results
    def one(item,arm,pilot=False):
        start=time.monotonic();before=len(reader.call_records);results=evidence(item)
        record={'id':item['id'],'arm':arm,'type':item['type'],'status':'error','exact_match':0,
            'output_token_cap':reader.max_new_tokens,
            'evidence_sha256':digest([(r.chunk.chunk_id,r.chunk.content) for r in results])}
        try:
            remaining=deadline-time.time()-20
            if remaining <= 0:record['status']='unserved';return record
            engine=(CitedReader if arm=='single' else CitedReActReader)(Frozen(results),reader,deadline_seconds=min(180,remaining))
            engine.output_tokens=reader.max_new_tokens
            response=engine.run(item['question'])
            quote_report=response.citation_report.get('quote_validation',{})
            explicit=quote_report.get('short_answer_explicit',False)
            format_failure=response.finish_reason=='format_failure' or not explicit
            record.update(status='format_failure' if format_failure else 'completed', format_failure=format_failure,
                answer=response.answer,short_answer=response.short_answer,short_answer_words=len((response.short_answer or '').split()),
                abstained=response.abstained,release_state=response.release_state,hold_reasons=response.hold_reasons,
                evidence=response.evidence,quote_errors=quote_report.get('errors',[]),calls=response.llm_calls)
            if not pilot and not format_failure:
                record.update(score('' if response.abstained else response.short_answer or '',item['reference']))
        except Exception as exc:record['error']=type(exc).__name__
        finally:
            calls=reader.call_records[before:]
            known=all(c.get('input_tokens') is not None and c.get('output_tokens') is not None for c in calls)
            record.update(call_records=calls,calls=len(calls),tokens=sum(c['input_tokens']+c['output_tokens'] for c in calls) if known else None,
                truncated=any(c.get('max_length_reached') for c in calls),numerical_failure=any(c.get('numerical_failure') for c in calls),
                elapsed_seconds=time.monotonic()-start)
        return record
    by_id={r['id']:r for r in pool};pilot_records=[]
    def timing_pilot(ids):
        timings=[]
        for index,identifier in enumerate(ids):
            started=time.monotonic()
            order=('single','react') if index%2==0 else ('react','single')
            for arm in order:
                pilot_records.append(one(by_id[identifier],arm,True));write(out/'private/pilot_records.json',pilot_records)
            timings.append(time.monotonic()-started)
        return timings
    timings=timing_pilot(manifest['pilot_ids'])
    if any(r['truncated'] for r in pilot_records):
        reader.max_new_tokens=1024
        timings=timing_pilot(manifest['reserve_pilot_ids'])
    final_pilot=pilot_records[-20:]
    # Preserve format failures and permit a small-reader format result; numerical
    # or token-overflow failures require diagnosis instead of a misleading cohort.
    if any(r.get('numerical_failure') or r.get('status')=='error' for r in final_pilot):
        raise RuntimeError('Pilot numerical/runtime failure; final cohort remains unmeasured')
    design=freeze_design([by_id[i] for i in manifest['evaluation_pool_ids']],timings,deadline-time.time())
    design['output_token_cap']=reader.max_new_tokens
    write(out/'design.json',design)
    records=[]
    for item in design['rows']:
        for arm in item['order']:
            if time.time()>deadline-20:break
            records.append(one(by_id[item['id']],arm));write(out/'private/records.json',records)
        finalize(out)
        if time.time()>deadline-20:break
    write(out/'memory.json',{'peak_allocated_device0_bytes':torch.cuda.max_memory_allocated()})
    finalize(out)


def finalize(out):
    from docmind.evaluation.paired import summarize
    if not (out/'design.json').exists():
        write(out/'summary.json',{'status':'unmeasured; timing pilot did not freeze a final cohort'});return
    records=json.loads((out/'private/records.json').read_text()) if (out/'private/records.json').exists() else []
    write(out/'summary.json',summarize(json.loads((out/'design.json').read_text()),records))
    # Quotes, articles, questions and references remain private pending source review.
    write(out/'numeric_records.json',[{k:v for k,v in r.items() if k not in ('answer','short_answer','evidence')} for r in records])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--source',type=Path)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:finalize(out)
    else:
        if (out/'input_manifest.json').exists():raise FileExistsError('Use a fresh immutable experiment directory')
        if args.source is None:parser.error('--source is required')
        prepare(args.source,out)
        if not args.prepare_only:
            deadline=float(os.environ['DOCMIND_EVAL_DEADLINE_UNIX'])
            if not 0<deadline-time.time()<=4800:raise ValueError('Profile exceeds the locked free-compute reservation')
            run(out,deadline)
