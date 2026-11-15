"""Validate a private human timing export. Never infer labels or durations."""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import statistics


def summarize(export,key):
    if export['packet_sha256']!=key['packet_sha256']:raise ValueError('Review packet identity differs')
    planned={r['review_id']:r for r in key['records']};records=export['records']
    if len({r['review_id'] for r in records})!=len(records) or not {r['review_id'] for r in records}<=set(planned):raise ValueError('Unknown or duplicate review IDs')
    total=export['visible_ms']
    if not isinstance(total,(int,float)) or not math.isfinite(total) or not 0<=total<=1200500:raise ValueError('Review duration exceeds the bounded packet')
    if abs(sum(r['visible_ms'] for r in records)-total)>.01:raise ValueError('Item and total times differ')
    if total and not export.get('started_at'):raise ValueError('Missing review start')
    if export.get('started_at') and datetime.fromisoformat(export['exported_at'].replace('Z','+00:00'))<datetime.fromisoformat(export['started_at'].replace('Z','+00:00')):raise ValueError('Export predates start')
    cap_reached=total>=1200000
    order={r['review_id']:i for i,r in enumerate(key['records'])}
    positive=[order[r['review_id']] for r in records if r['visible_ms']>0]
    boundary=max(positive) if positive and cap_reached else None
    finished=[];timed=[]
    for row in records:
        seconds=row['visible_ms']/1000
        if not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid item duration')
        if row['question_validity'] not in ('','valid','invalid','unclear') or row['answer_correctness'] not in ('','correct','incorrect','unclear'):raise ValueError('Unknown human label')
        if not isinstance(row['finished'],bool):raise ValueError('Finished flag must be boolean')
        if row['finished']:
            if not row['question_validity'] or not row['answer_correctness'] or not row.get('finished_at'):raise ValueError('Incomplete finished record')
            if seconds==0 and not cap_reached:raise ValueError('Finished record lacks recorded time before the cap')
            if not planned[row['review_id']]['practice']:
                finished.append((row,planned[row['review_id']]))
                if seconds>0 and (boundary is None or order[row['review_id']]<boundary):timed.append((row,planned[row['review_id']]))
    stamped=sorted([r for r in records if r['finished']],key=lambda r:order[r['review_id']])
    if any(stamped[i]['finished_at']>stamped[i+1]['finished_at'] for i in range(len(stamped)-1)):timed=[]
    expected=[r for r in planned.values() if not r['practice']]
    groups={}
    for layout in ('A','B'):
        values=[r for r,k in finished if k['layout']==layout]
        times=[r['visible_ms']/1000 for r,k in timed if k['layout']==layout]
        groups[layout]={'completed_labels':len(values),'fully_timed_subset':len(times),'median_visible_seconds':statistics.median(times) if times else None,
            'mean_visible_seconds':statistics.mean(times) if times else None,'seconds_sum':sum(times),
            'question_validity':dict(Counter(r['question_validity'] for r in values)),'answer_correctness':dict(Counter(r['answer_correctness'] for r in values))}
    complete=len(finished)==len(expected)
    discordance=sum((r['answer_correctness']=='correct')!=k['benchmark_exact_match'] for r,k in finished if r['answer_correctness']!='unclear')
    wall=(datetime.fromisoformat(export['exported_at'].replace('Z','+00:00'))-datetime.fromisoformat(export['started_at'].replace('Z','+00:00'))).total_seconds() if export.get('started_at') else None
    return {'status':'validated labels with capped timing' if cap_reached else 'validated single-reviewer subset','packet_sha256':key['packet_sha256'],'selected_nonpractice':len(expected),'finished_nonpractice':len(finished),
        'missing_nonpractice':len(expected)-len(finished),'total_visible_seconds_including_practice_and_unfinished':total/1000,
        'start_to_export_wall_seconds':wall,'fully_timed_nonpractice':len(timed),'timing_censored_nonpractice':len(finished)-len(timed),
        'finished_zero_recorded_time':sum(r['finished'] and r['visible_ms']==0 for r in records),'timer_cap_reached':cap_reached,
        'between_layout_time_savings':None,'timing_limit':'The legacy timer stopped accumulating at its cap while labels could still be saved. Zero times after the cap are missing durations, not instant reviews. The final positive-duration item is conservatively treated as censored. Timing comparisons are unvalidated.',
        'layout_descriptives':groups,'question_answer_labels_are_separate':True,'human_vs_benchmark_em_disagreements':discordance,
        'sampling_complete':complete,'population_extrapolation':False,
        'scope':'Single-reviewer English public-document interface study. Layouts use different matched questions. Visible-page time includes possible idle time and excludes reading before Start; 500ms timing resolution. Missing or time-stopped items can bias completion.',
        'review_cost_claim':'Descriptive subset only. No general time-saving or Hindi attribution claim.',
        'human_claim_support_labels':0,'inter_annotator_agreement':None,'automatic_release_risk':'Insufficient evidence: no calibrated candidate releases in this study'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--export',type=Path,required=True);parser.add_argument('--private-key',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    result=summarize(json.loads(args.export.read_text(encoding='utf-8-sig')),json.loads(args.private_key.read_text(encoding='utf-8')))
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result,indent=2))
