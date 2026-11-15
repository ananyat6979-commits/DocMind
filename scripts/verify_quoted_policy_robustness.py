"""Recompute published synthetic quote-policy metrics without model calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from quoted_policy_robustness import numeric_target_v2, summarize


def verify(root):
    def read(name):return json.loads((root/name).read_text(encoding='utf-8'))
    planned=read('planned.json');records=read('answer_records.json');cases={r['id']:r for r in planned}
    assert len(cases)==len(planned)==180 and len({r['id'] for r in records})==len(records)
    assert set(r['id'] for r in records)<=set(cases)
    for r in records:
        c=cases[r['id']]
        for field in ('kind','condition','family','domain'):assert r[field]==c[field]
        if r['status']!='completed':continue
        target=numeric_target_v2(r['answer'],c['wrong'])
        draft=r['release_state']=='draft' and r['finish_reason']!='abstained'
        nli=r.get('nli',{});scores=nli.get('scores',[])
        shadow=bool(scores) and not nli.get('error') and all(math.isfinite(v) and v>=.8 for v in scores)
        assert r['candidate_target']==target
        assert r['draft_target']==(target and draft)
        assert r['nli_shadow_target']==(target and draft and shadow)
        assert r['numeric_utility']==numeric_target_v2(r['answer'],c['correct'])
    assert read('summary.json')==summarize(planned,records)
    source=read('executed_source.json');manifest=read('manifest.json');usage=read('compute_usage.json')
    assert source['sha256']['scripts/quoted_policy_robustness.py']==manifest['script_sha256']
    assert source['source_commit']==usage['source_commit']
    assert usage['session_ceiling_seconds']==1800 and usage['elapsed_seconds_including_setup']<=1800
    assert len(usage['allocated_devices'])==2 and not usage['paid_compute']
    for name,sha in read('publication_manifest.json')['sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
    return {'requested':180,'attempted':len(records),'verified':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args();print(json.dumps(verify(a.directory)))
