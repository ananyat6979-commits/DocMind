"""Outcome-free paired-test power sensitivity under declared hypothetical discordance."""
import argparse
import json
import math
from pathlib import Path


def analyze():
    import numpy as np
    rng=np.random.default_rng(7);rows=[]
    for n in (20,50,100):
        # One-sided .025 corresponds to detecting a positive effect with a
        # two-sided .05 exact McNemar/binomial test. This is not bootstrap power.
        rejection={m:next((b for b in range(m+1) if sum(math.comb(m,k) for k in range(b,m+1))/2**m<=.025),m+1) for m in range(n+1)}
        for discordance in (.3,.5):
            for delta in (0,.05,.1,.15):
                draws=rng.multinomial(n,[(discordance+delta)/2,(discordance-delta)/2,1-discordance],size=10000)
                rejected=sum(int(b)>=rejection[int(b+c)] for b,c,_ in draws)
                rows.append({'n_pairs':n,'assumed_discordance':discordance,'assumed_react_minus_single':delta,'estimated_detection_probability':rejected/len(draws)})
    return {'scope':'Pre-outcome sensitivity under hypothetical independent paired Bernoulli outcomes; not empirical power or a promotion threshold',
        'test':'Positive direction of a two-sided .05 exact McNemar test; upper binomial tail <= .025',
        'reported_result_intervals':'Paired bootstrap intervals are separate; the simulated test is not their exact power calculation',
        'seed':7,'simulations_per_scenario':10000,'rows':rows,
        'limitations':'Shared articles can violate question independence. Fresh question IDs do not establish unseen training data, document disjointness or general loop utility.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Power sensitivity outputs are immutable')
    result=analyze();args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([r for r in result['rows'] if r['n_pairs']==100 and r['assumed_react_minus_single']==.05]))
