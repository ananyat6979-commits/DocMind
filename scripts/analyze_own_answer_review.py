"""Validate private human decisions and export only bounded aggregate descriptions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(packet,linkage,labels):
    if labels.get('packet_sha256')!=packet['sha256']:raise ValueError('Review packet mismatch')
    expected={r['id']:r for r in packet['items']};route={r['id']:r for r in linkage}
    records=labels['records'];ids=[r['id'] for r in records]
    if len(ids)!=len(set(ids)) or not set(ids)<=set(expected) or set(route)!=set(expected):raise ValueError('Invalid reviewed identities')
    complete=[]
    for r in records:
        if not r.get('finished') or not r.get('human_adjudicated'):continue
        if r.get('answer_correctness') not in ('correct','incorrect','unclear'):raise ValueError('Missing correctness label')
        claims=expected[r['id']]['claims'];judgments=r.get('claims',{})
        if set(judgments)!={c['id'] for c in claims} or any(v not in ('supported','unsupported','unclear') for v in judgments.values()):raise ValueError('Missing or unexpected claim label')
        complete.append(r)
    correctness=dict(Counter(r['answer_correctness'] for r in complete));release=[r for r in complete if route[r['id']].get('release_state')=='draft']
    decisive=[r for r in release if r['answer_correctness']!='unclear'];incorrect=sum(r['answer_correctness']=='incorrect' for r in decisive)
    claims=[v for r in complete for v in r.get('claims',{}).values()]
    elapsed=float(labels.get('elapsed_ms',0))/1000
    if not 0<=elapsed<=packet['cap_seconds']:raise ValueError('Review timer exceeds the frozen allowance')
    return {'scope':'Timed single-reviewer actual-answer subset, descriptive only; no population extrapolation','packet_sha256':packet['sha256'],'requested_review_answers':len(expected),'finished_review_answers':len(complete),'unfinished_or_missing':len(expected)-len(complete),'recorded_running_seconds':elapsed,'active_work_independently_observed':False,'answer_correctness':correctness,'reviewed_drafts':len(release),'decisive_reviewed_drafts':len(decisive),'incorrect_reviewed_drafts':incorrect,'reviewed_subset_false_release_fraction':incorrect/len(decisive) if len(decisive)>=25 else None,'false_release_risk':'insufficient evidence' if len(decisive)<25 else 'reviewed-subset descriptive only; population risk remains unvalidated','correct_held_answers':sum(r['answer_correctness']=='correct' and route[r['id']].get('release_state')=='held' for r in complete),'incorrect_answerable_abstentions':sum(bool(r['answer_correctness']=='incorrect' and route[r['id']].get('abstained') and route[r['id']].get('answerable')) for r in complete),'claim_labels':dict(Counter(claims)),'attribution_recall':'unmeasured: no complete independent claim-extraction gold','human_annotation_sources':'One reviewer, also involved in source/question confirmation; no inter-annotator agreement or independent replication','missing_label_policy':'No automatic judgments or time-based completion weights substitute for missing human labels'}


def metric_sensitivity(packet, linkage, labels, records):
    """Compare automatic metrics only against identical, explicitly reviewed inputs."""
    summarize(packet, linkage, labels)
    route = {r['id']: r for r in linkage}
    items = {r['id']: r for r in packet['items']}
    rows = {r['task_id']: r for r in records}
    if len(rows) != len(records):
        raise ValueError('Repeated profile inputs')
    cross, support = Counter(), Counter()
    for label in labels['records']:
        if not label.get('finished') or not label.get('human_adjudicated'):
            continue
        identity = label['id']
        row = rows[route[identity]['task_id']]
        grouped = {}
        for e in row.get('evidence', []):
            grouped.setdefault(e.get('claim', ''), []).append(e.get('text') or e.get('quote') or '')
        claims = [{'id': 'claim-' + str(i + 1), 'claim': text, 'quote': '\n\n'.join(quotes)}
            for i, (text, quotes) in enumerate(grouped.items())]
        digest = hashlib.sha256(json.dumps({'answer': row.get('answer'), 'claims': claims}, sort_keys=True).encode()).hexdigest()
        if digest != route[identity]['input_sha256'] or claims != items[identity]['claims'] or row.get('answer') != items[identity]['answer']:
            raise ValueError('Changed answer/evidence cannot reuse human judgments')
        cross[str(row['exact_match']) + ':' + label['answer_correctness']] += 1
        for claim in claims:
            evidence = [e for e in row['evidence'] if e['claim'] == claim['claim']]
            found = bool(evidence) and all(e['chunk_id'] in {r['chunk_id'] for r in row['sources']} for e in evidence)
            located = bool(evidence) and all(e['status'] in ('literal', 'normalized') for e in evidence)
            support['citation_found=' + str(found) + ':quote_located=' + str(located) + ':' + label['claims'][claim['id']]] += 1
    return {'scope': 'Post-hoc human versus exact-match and citation-location diagnostic on the unchanged baseline; no inference or automatic human labels',
        'packet_sha256': packet['sha256'], 'answer_em_by_human_label': dict(cross),
        'claim_location_by_human_support': dict(support), 'input_identity_verified': True,
        'limitations': ['Selected English diagnostic cohort and a single reviewer involved in source confirmation',
            'Claim labels cover extracted claims only, not complete claim-extraction gold',
            'Location is not entailment; no independent NLI accuracy or attribution recall claim',
            'This sensitivity does not relabel historical calibration outcomes or estimate population false-release risk']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('labels',type=Path);p.add_argument('out',type=Path);p.add_argument('--profile',type=Path);p.add_argument('--sensitivity-out',type=Path);a=p.parse_args()
    if bool(a.profile) != bool(a.sensitivity_out):p.error('Both profile and sensitivity output are required')
    packet=json.loads((a.root/'actual-answer-packet.private.json').read_text(encoding='utf-8'));linkage=json.loads((a.root/'actual-answer-linkage.private.json').read_text(encoding='utf-8'));raw=a.labels.read_bytes();labels=json.loads(raw)
    summary=summarize(packet,linkage,labels);summary['export_sha256']=hashlib.sha256(raw).hexdigest();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_bytes((json.dumps(summary,indent=2)+'\n').encode());print(json.dumps(summary))
    if a.profile:
        sensitivity=metric_sensitivity(packet,linkage,labels,json.loads((a.profile/'private/records.json').read_text(encoding='utf-8')))
        sensitivity['export_sha256']=summary['export_sha256']
        a.sensitivity_out.write_bytes((json.dumps(sensitivity,indent=2)+'\n').encode())
