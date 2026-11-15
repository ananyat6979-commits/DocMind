from scripts.quoted_policy_robustness import summarize
from scripts.robustness_closure import cases


def test_shadow_gate_comparison_keeps_missing_outcomes_and_separates_poisoning():
    planned=cases();case=next(c for c in planned if c['kind']=='attack' and c['condition']=='json_current')
    record={**{k:case[k] for k in ('id','family','condition','kind','domain')},'status':'completed',
        'candidate_target':True,'draft_target':True,'nli_shadow_target':False}
    result=summarize(planned,[record])['groups']['json_current:attack']
    assert result['requested']==60 and result['completed']==1 and result['missing_or_failed']==59
    assert result['draft_target']==1 and result['quote_nli_shadow_target']==0
    assert result['draft_target_requested_bounds']==[1/60,1]
    assert result['family_any_success95_nominal'] is None
