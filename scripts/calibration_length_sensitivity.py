"""Post-hoc EM/length sensitivity, never a new calibrated policy or factual adjudication."""
import argparse
import hashlib
import json
from pathlib import Path


def analyze(path):
    all_rows=json.loads(path.read_text(encoding='utf-8'));rows=[r for r in all_rows if r['split']=='heldout']
    def stats(items):
        correct=sum(bool(r['correct']) for r in items)
        return {'n':len(items),'exact_match_correct':correct,'exact_match_accuracy':correct/len(items) if items else None,
            'em_failures_with_token_f1_at_least_half':sum(not r['correct'] and r.get('token_f1',0)>=.5 for r in items)}
    groups={'all':stats(rows),'at_most_8_words':stats([r for r in rows if len(r.get('answer','').split())<=8]),
        'more_than_8_words':stats([r for r in rows if len(r.get('answer','').split())>8])}
    for name,predicate in [('verifier_equal_one',lambda v:v==1),('verifier_at_least_0999',lambda v:v>=.999)]:
        selected=[r for r in rows if r.get('verification_score') is not None and predicate(r['verification_score'])]
        groups[name]={**stats(selected),'em_errors_over_8_words':sum(not r['correct'] and len(r.get('answer','').split())>8 for r in selected)}
    concise=[r for r in rows if len(r.get('answer','').split())<=8 and r.get('token_confidence') is not None]
    top=sorted(concise,key=lambda r:(-r['token_confidence'],r['id']))[:20]
    return {'scope':'Post-hoc descriptive sensitivity of historical English HotpotQA held-out EM to raw answer length. No factual labels inferred and no threshold selected.',
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'word_count':'Whitespace split of the recorded full answer',
        'groups':groups,'top_20_concise_token_confidence':stats(top),
        'interpretation':'Verbose answers, aliases and partial matches can fail EM. Token-F1 is also not factual correctness. The original fixed-grid policies did not qualify under their EM protocol; this is not proof of factual overconfidence or safe release risk.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Sensitivity outputs are immutable')
    result=analyze(args.source);args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['groups'],indent=2))
