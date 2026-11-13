"""Selective prediction with disjoint calibration and explicit review cost."""
import math
import numpy as np
from docmind.evaluation.metrics import bootstrap_ci


def wilson_upper(errors, total, z=1.6448536269514722):
    if not total:
        return 1.0
    p = errors / total
    return (p + z*z/(2*total) + z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1+z*z/total)


def risk_curve(scores, correct, review_seconds=60):
    scores, correct = np.asarray(scores, float), np.asarray(correct, bool)
    if len(scores) != len(correct) or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Finite scores and correctness labels of equal nonzero length are required")
    out = []
    for threshold in [float("inf"), *sorted(set(scores), reverse=True)]:
        selected = scores >= threshold
        n = int(selected.sum())
        risk = float((~correct[selected]).mean()) if n else None
        ci = bootstrap_ci((~correct[selected]).astype(float)) if n else (None, None, None)
        out.append({"threshold": threshold if math.isfinite(threshold) else None,
                    "coverage": n / len(scores), "risk": risk, "risk_ci95": list(ci[1:]),
                    "risk_upper95": wilson_upper(int((~correct[selected]).sum()), n),
                    "n_selected": n, "n_review": len(scores)-n,
                    "review_minutes": (len(scores)-n)*review_seconds/60})
    return out


def calibrate(scores, correct, max_risk=0.1, min_selected=30):
    # Bonferroni accounts for searching several thresholds on the same calibration set.
    from statistics import NormalDist
    thresholds = sorted(set(scores), reverse=True)
    z = NormalDist().inv_cdf(1 - 0.05 / max(len(thresholds), 1))
    rows = risk_curve(scores, correct)
    valid = []
    for row in rows:
        n = row["n_selected"]
        errors = round((row["risk"] or 0) * n)
        row["calibration_upper95"] = wilson_upper(errors, n, z)
        if n >= min_selected and row["calibration_upper95"] <= max_risk:
            valid.append(row)
    return max(valid, key=lambda x: x["coverage"]) if valid else {
        "threshold": None, "coverage": 0.0, "risk": None,
        "reason": "No threshold satisfies the corrected calibration bound"}
