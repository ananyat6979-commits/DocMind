"""Post-hoc retrieval and paired-ranking diagnostics; never selects a release policy."""
import argparse
import json
from pathlib import Path


def analyze(directory,source,allow_download=False):
    import numpy as np
    from calibration_closure import acquire,SOURCE_SHA
    from qa_and_robustness import scores
    if not source.exists() and not allow_download:raise FileNotFoundError('Provide the pinned public source or explicitly allow download')
    pool=acquire(source);lookup={r['id']:r for values in pool.values() for r in values}
    records=json.loads((directory/'numeric_records.json').read_text(encoding='utf-8'));statistics={}
    for split in ('calibration','heldout'):
        rows=[r for r in records if r['split']==split];complete=[];incomplete=[]
        for row in rows:
            original=lookup[row['id']];em,f1=scores('' if row['abstained'] else row['answer'],original['reference'])
            if row['correct']!=bool(em) or abs(row['token_f1']-f1)>1e-12:raise ValueError('Published answer score differs from public reference')
            all_titles=set(original['supporting_titles'])<=set(row['retrieved_titles'])
            (complete if all_titles else incomplete).append(row)
        def describe(group):return {'n':len(group),'exact_match_correct':sum(r['correct'] for r in group),'exact_match':sum(r['correct'] for r in group)/len(group) if group else None}
        statistics[split]={'requested':len(rows),'all_gold_article_titles_in_top5':describe(complete),'missing_at_least_one_gold_article_title':describe(incomplete),
            'abstained':sum(r['abstained'] for r in rows),'truncated':sum(r['truncated'] for r in rows)}
    common=[r for r in records if r['split']=='heldout' and r.get('token_confidence') is not None and r.get('verification_score') is not None]
    labels=np.array([r['correct'] for r in common]);positive=np.flatnonzero(labels);negative=np.flatnonzero(~labels);rng=np.random.default_rng(7)
    values={signal:np.array([r[signal] for r in common]) for signal in ('token_confidence','verification_score')}
    def auc(a,b):return float(((a[:,None]>b)+.5*(a[:,None]==b)).mean())
    differences=[]
    for _ in range(2000):
        p=rng.choice(positive,len(positive),replace=True);n=rng.choice(negative,len(negative),replace=True)
        differences.append(auc(values['token_confidence'][p],values['token_confidence'][n])-auc(values['verification_score'][p],values['verification_score'][n]))
    paired={s:auc(v[positive],v[negative]) for s,v in values.items()}
    return {'reference_validation':'All 256 public predictions rescored against the pinned public dataset; exact-match and token-F1 agree. No human labels inferred.','source_sha256':SOURCE_SHA,
        'article_title_diagnostic':statistics,'article_title_scope':'Post-hoc source-title coverage only. Does not check which supporting sentence survived the 384-token evidence cap or whether a complete multi-hop answer is derivable.',
        'paired_ranking_common_rows':{'n':len(common),'auroc':paired,'token_minus_verifier':paired['token_confidence']-paired['verification_score'],
            'stratified_paired_bootstrap95_nominal':np.percentile(differences,[2.5,97.5]).tolist(),'seed':7,'resamples':2000,'scope':'Exploratory comparison on identical candidate answers with both scores; no held-out threshold selection'}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--allow-download',action='store_true');args=parser.parse_args()
    result=analyze(args.directory,args.source,args.allow_download);(args.directory/'failure_analysis.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result,indent=2))
