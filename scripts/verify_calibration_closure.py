"""Independently check frozen threshold selection and held-out numeric records."""
import argparse
import json
import math
from pathlib import Path


def upper(errors,total,alpha):
    if not total or errors==total:return 1.0
    low,high=0.,1.
    for _ in range(70):
        p=(low+high)/2
        value=sum(math.comb(total,k)*p**k*(1-p)**(total-k) for k in range(errors+1))
        if value>alpha:low=p
        else:high=p
    return high


def eligible(row,signal,threshold):
    value=row.get(signal)
    return threshold is not None and isinstance(row.get('correct'),bool) and not row.get('abstained',True) and isinstance(value,(int,float)) and math.isfinite(value) and 0<=value<=1 and value>=threshold


def verify(directory):
    def read(name):return json.loads((directory/name).read_text(encoding='utf-8'))
    rows=read('numeric_records.json');cohort=read('cohort.json');summary=read('summary.json');policies=read('policies.json');pool=read('pool_ids.json')
    assert len(rows)==len({r['id'] for r in rows})
    assert not set(cohort['calibration'])&set(cohort['heldout'])
    assert not set(pool['pilot'])&set(cohort['calibration']+cohort['heldout'])
    for split in ('calibration','heldout'):
        assert cohort[split]==pool[split][:cohort['n_per_split']]
        actual=[r for r in rows if r['split']==split]
        assert set(r['id'] for r in actual)==set(cohort[split])
        assert all(r['status']=='completed' and isinstance(r['correct'],bool) for r in actual)
        assert summary['counts'][split]['generation_completed']==len(actual)
    calibration=[r for r in rows if r['split']=='calibration'];held=[r for r in rows if r['split']=='heldout']
    assert len(policies)==6
    for index,policy in enumerate(policies):
        candidates=[]
        for t in [i/20 for i in range(21)]:
            selected=[r for r in calibration if eligible(r,policy['signal'],t)]
            errors=sum(not r['correct'] for r in selected)
            bound=upper(errors,len(selected),.05/126)
            if len(selected)>=30 and bound<=policy['target_em_risk']:candidates.append((len(selected),t,bound))
        best=max(candidates) if candidates else None
        assert policy['threshold']==(best[1] if best else None)
        released=[r for r in held if eligible(r,policy['signal'],policy['threshold'])]
        published=summary['policies'][index]['heldout']
        assert published['requested']==len(held) and published['released']==len(released)
        assert published['reviewed']==len(held)-len(released)
        assert published['errors']==sum(not r['correct'] for r in released)
        assert abs(published['selected_risk_upper95_nominal']-upper(published['errors'],len(released),.05))<1e-10
    for point in summary['heldout_curves_descriptive']:
        selected=[r for r in held if eligible(r,point['signal'],point['threshold'])]
        assert point['released']==len(selected)
        assert point['errors']==sum(not r['correct'] for r in selected)
        assert abs(point['selected_risk_upper95_nominal']-upper(point['errors'],len(selected),.05))<1e-10
    for signal,published in summary['heldout_score_ranking'].items():
        available=[r for r in held if isinstance(r.get(signal),(int,float)) and math.isfinite(r[signal])]
        positive=[r[signal] for r in available if r['correct']];negative=[r[signal] for r in available if not r['correct']]
        value=sum((a>b)+.5*(a==b) for a in positive for b in negative)/(len(positive)*len(negative))
        assert abs(published['auroc']-value)<1e-12 and published['n_available']==len(available)
    assert summary['baselines']['ungated_completed_generation']['correct']==sum(r['correct'] for r in held)
    print(json.dumps({'verified_records':len(rows),'heldout':len(held),'passing_policies':sum(p['threshold'] is not None for p in policies)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args();verify(args.directory)
