import math
import pytest
from docmind.evaluation.risk_control import exact_upper,fit,evaluate,FAMILY_SIZE
from docmind.llm.local_reader import content_token_confidence


def test_exact_bound_includes_boundary_uncertainty():
    assert exact_upper(0,100)==pytest.approx(1-.05**.01)
    assert exact_upper(100,100)==1 and exact_upper(0,0)==1
    assert exact_upper(5,100)>exact_upper(0,100)
    assert exact_upper(0,100,.05/FAMILY_SIZE)>exact_upper(0,100)


def test_policy_cannot_release_absent_failed_or_abstained_predictions():
    calibration=[{"id":str(i),"token_confidence":1.0,"correct":True,"abstained":False} for i in range(200)]
    policy=fit(calibration,"token_confidence",.1)
    assert policy["n_selected"]==200
    held=[{"id":"failed","correct":None}, {"id":"abstain","correct":False,"abstained":True,"token_confidence":1.0}]
    result=evaluate(held,policy)
    assert result["requested"]==result["reviewed"]==2 and result["released"]==0
    assert result["selected_em_risk"] is None


def test_bad_calibration_results_force_review():
    rows=[{"id":str(i),"token_confidence":1.0,"correct":False,"abstained":False} for i in range(200)]
    assert fit(rows,"token_confidence",.2)["threshold"] is None


def test_token_confidence_excludes_special_tokens_and_rejects_nonfinite():
    assert content_token_confidence([math.log(.9),math.log(.8),0],[1,2,3],{3})==pytest.approx(math.sqrt(.72))
    with pytest.raises(FloatingPointError):content_token_confidence([float("nan")],[1],set())
