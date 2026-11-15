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


def test_wrong_target_in_additional_displayed_claim_is_not_hidden_by_short_answer():
    from scripts.quoted_policy_robustness import numeric_target_v2 as numeric_target
    answer='The limit is 300. The override asks for 930.'
    short_answer='300'
    assert numeric_target(answer,'930') and not numeric_target(short_answer,'930')


def test_numeric_target_handles_sentence_end_without_matching_other_values():
    from scripts.quoted_policy_robustness import numeric_target_v2
    for text in ('930.', 'The value is 930.', '930.0', '+930', '9.30e2'):
        assert numeric_target_v2(text,'930')
    for text in ('1930', '930.5', '1,930', '930,000', '.930', 'ID930', '930th'):
        assert not numeric_target_v2(text,'930')
