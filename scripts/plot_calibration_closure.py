"""Render descriptive held-out curves and stratified ranking uncertainty."""
import argparse
import json
from pathlib import Path


def render(directory):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    rows=[r for r in json.loads((directory/'numeric_records.json').read_text(encoding='utf-8')) if r['split']=='heldout']
    rng=np.random.default_rng(7);ranking={}
    fig,axis=plt.subplots(figsize=(8,5))
    labels={'token_confidence':'Token confidence','verification_score':'Answer verifier','combined_score':'Minimum of both'}
    for signal,label in labels.items():
        available=[r for r in rows if isinstance(r.get(signal),(int,float)) and np.isfinite(r[signal])]
        positive=np.array([r[signal] for r in available if r['correct']]);negative=np.array([r[signal] for r in available if not r['correct']])
        draws=[]
        for _ in range(2000):
            a=rng.choice(positive,len(positive),replace=True);b=rng.choice(negative,len(negative),replace=True)
            draws.append(float(((a[:,None]>b)+.5*(a[:,None]==b)).mean()))
        ranking[signal]={'auroc':summary['heldout_score_ranking'][signal]['auroc'],'stratified_bootstrap95_nominal':np.percentile(draws,[2.5,97.5]).tolist(),'n_available':len(available)}
        points=[p for p in summary['heldout_curves_descriptive'] if p['signal']==signal and p['released']]
        points.sort(key=lambda p:p['coverage'])
        line=axis.plot([p['coverage'] for p in points],[p['selected_em_risk'] for p in points],marker='.',label=label)[0]
        axis.plot([p['coverage'] for p in points],[p['selected_risk_upper95_nominal'] for p in points],linestyle=':',alpha=.65,color=line.get_color())
    for value in (.1,.2):axis.axhline(value,color='#999999',linestyle='--',linewidth=.8)
    axis.set(xlim=(0,1),ylim=(0,1),xlabel='Released / 128 requested questions',ylabel='Exact-match error among released answers',title='Held-out coverage versus risk: no calibration policy qualified')
    axis.grid(alpha=.15);axis.legend(loc='lower right');fig.text(.08,.015,'Dotted curves: nominal one-sided 95% upper bounds. Descriptive thresholds, not selected on held-out outcomes.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(directory/'coverage-risk.png',dpi=180);fig.savefig(directory/'coverage-risk.svg');plt.close(fig)
    diagnostic={'ranking':ranking,'intervals':'Stratified percentile bootstrap, 2000 draws, seed seven; nominal exploratory ranking intervals, not simultaneous policy certificates',
        'heldout_em_correct':sum(r['correct'] for r in rows),'heldout_requested':len(rows),'human_review_cost':'Awaiting actual private review labels'}
    (directory/'derived_diagnostics.json').write_bytes(json.dumps(diagnostic,indent=2).encode());print(json.dumps(diagnostic))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args();render(args.directory)
