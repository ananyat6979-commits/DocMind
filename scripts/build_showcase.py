"""Build a static replay from an explicit allowlist of already-public artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]


def build(out):
    out=out.resolve()
    if out!=ROOT/'.site-dist':raise ValueError('Use the dedicated ignored static output directory')
    out.mkdir(exist_ok=True)
    for name in ('index.html','styles.css','app.js'):shutil.copyfile(ROOT/'showcase'/name,out/name)
    target=out/'assets/demo';target.mkdir(parents=True,exist_ok=True)
    for name in ('ui-4.png','docmind-workflow.gif'):shutil.copyfile(ROOT/'assets/demo'/name,target/name)
    content={'scope':'Recorded real local-reader outputs on original public synthetic fixtures; not live inference','conditions':{}}
    for key,directory in [('original','quoted_policy_t4'),('complete','quoted_complete_schema_t4')]:
        source=ROOT/'reports/releases/r4'/directory
        planned=json.loads((source/'planned.json').read_text(encoding='utf-8'));observed=json.loads((source/'answer_records.json').read_text(encoding='utf-8'));lookup={r['id']:r for r in observed};usage=json.loads((source/'compute_usage.json').read_text(encoding='utf-8'))
        records=[]
        for case in planned:
            r=lookup.get(case['id'],{'id':case['id'],'status':'unserved'})
            texts=([case['payload']+'\n'+case['fact']] if case['placement']=='before' else [case['fact']+'\n'+case['payload']] if case['placement']=='after' else [case['fact'],case['payload']])
            records.append({**{k:case[k] for k in ('id','condition','kind','domain','family','question')},**{k:r.get(k) for k in ('status','answer','release_state','hold_reasons','finish_reason','evidence','nli','draft_target')},'sources':[{'chunk_id':hashlib.sha256((case['id']+':'+str(i)).encode()).hexdigest(),'text':text} for i,text in enumerate(texts)]})
        content['conditions'][key]={'source_commit':usage['source_commit'],'source_url':'https://github.com/ananyat6979-commits/DocMind/blob/main/reports/releases/r4/'+directory+'/answer_records.json','attempted':len(observed),'records':records}
    (out/'data.json').write_text(json.dumps(content,ensure_ascii=True,separators=(',',':')),encoding='utf-8')
    (out/'.nojekyll').write_text('',encoding='utf-8')
    return {'conditions':{k:{'requested':len(v['records']),'attempted':v['attempted']} for k,v in content['conditions'].items()},'private_inputs_read':False,'hosted_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'.site-dist');a=p.parse_args();print(json.dumps(build(a.out)))
