"""Fixed-grid selective exact-match risk control; never a factual-safety certificate."""
import math
from statistics import NormalDist

GRID=tuple(i/20 for i in range(21))
SIGNALS=("token_confidence","verification_score","combined_score")
TARGETS=(.1,.2)
FAMILY_SIZE=len(GRID)*len(SIGNALS)*len(TARGETS)


def exact_upper(errors,total,alpha=.05):
    if not isinstance(errors,int) or not isinstance(total,int) or not 0<=errors<=total or not 0<alpha<1:
        raise ValueError("Invalid binomial inputs")
    if not total or errors==total:return 1.0
    if errors==0:return -math.expm1(math.log(alpha)/total)
    def cdf(p):
        values=[math.lgamma(total+1)-math.lgamma(k+1)-math.lgamma(total-k+1)+k*math.log(p)+(total-k)*math.log1p(-p) for k in range(errors+1)]
        largest=max(values)
        return math.exp(largest)*sum(math.exp(v-largest) for v in values)
    low,high=0.0,1.0
    for _ in range(80):
        middle=(low+high)/2
        if cdf(middle)>alpha:low=middle
        else:high=middle
    return high


def selected(row,signal,threshold):
    score=row.get(signal)
    return (threshold is not None and isinstance(row.get("correct"),bool) and not row.get("abstained",True)
        and isinstance(score,(int,float)) and math.isfinite(score) and 0<=score<=1 and score>=threshold)


def fit(rows,signal,target,alpha=.05,min_selected=30):
    if signal not in SIGNALS or target not in TARGETS:raise ValueError("Use the frozen signal/target family")
    if len({r["id"] for r in rows})!=len(rows):raise ValueError("Duplicate calibration IDs")
    candidates=[]
    for threshold in GRID:
        accepted=[r for r in rows if selected(r,signal,threshold)]
        errors=sum(not r["correct"] for r in accepted)
        bound=exact_upper(errors,len(accepted),alpha/FAMILY_SIZE)
        if len(accepted)>=min_selected and bound<=target:
            candidates.append({"threshold":threshold,"n_selected":len(accepted),"errors":errors,"upper_calibration_risk":bound})
    best=max(candidates,key=lambda p:(p["n_selected"],p["threshold"])) if candidates else {
        "threshold":None,"n_selected":0,"reason":"No fixed-grid threshold meets the corrected exact bound"}
    return {**best,"signal":signal,"target_em_risk":target,"alpha":alpha,"family_size":FAMILY_SIZE,
        "scope":"Selective exact-match error under independent matched-distribution examples; distribution shift is outside the bound"}


def evaluate(rows,policy):
    accepted=[r for r in rows if selected(r,policy["signal"],policy["threshold"])]
    errors=sum(not r["correct"] for r in accepted)
    known=[r for r in rows if r.get("correct") is not None]
    return {"requested":len(rows),"prediction_labels_available":len(known),"released":len(accepted),"reviewed":len(rows)-len(accepted),
        "coverage":len(accepted)/len(rows) if rows else 0,
        "selected_em_risk":errors/len(accepted) if accepted else None,
        "selected_risk_upper95_nominal":exact_upper(errors,len(accepted)),
        "correct_predictions_withheld":sum(r["correct"] for r in known if not selected(r,policy["signal"],policy["threshold"])),
        "errors":errors,"review_cost_seconds":None,"review_cost_status":"awaiting measured human timing"}


def auc(rows,signal):
    valid=[r for r in rows if r.get("correct") is not None and isinstance(r.get(signal),(int,float)) and math.isfinite(r[signal])]
    positives=[r[signal] for r in valid if r["correct"]];negatives=[r[signal] for r in valid if not r["correct"]]
    value=sum((a>b)+.5*(a==b) for a in positives for b in negatives)/(len(positives)*len(negatives)) if positives and negatives else None
    return {"auroc":value,"n_available":len(valid),"scope":"Correctness ranking against benchmark exact match"}


def wilson_two_sided(successes,total):
    if not total:return [None,None]
    z=NormalDist().inv_cdf(.975);den=total+z*z
    center=(successes+z*z/2)/den;half=z*math.sqrt(successes*(1-successes/total)+z*z/4)/den
    return [max(0,center-half),min(1,center+half)]
