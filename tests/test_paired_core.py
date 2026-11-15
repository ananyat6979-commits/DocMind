import pytest
from docmind.evaluation.paired import balanced_pool, freeze_design, summarize, score


def pool():return [{'id':str(i),'type':'bridge' if i%5<3 else 'comparison'} for i in range(200)]


def test_design_is_balanced_excludes_pilot_and_shrinks_before_outcomes():
    selected=balanced_pool(pool(), {'0','1'}, 100)
    assert not {'0','1'} & {r['id'] for r in selected}
    assert sum(r['type']=='bridge' for r in selected) == 60
    design=freeze_design(selected,[10]*10,500)
    assert len(design['rows']) == 30
    assert sum(r['type']=='bridge' for r in design['rows']) == 18
    assert sum(r['order'][0]=='single' for r in design['rows']) == 15
    assert design == freeze_design(selected,[10]*10,500)
    with pytest.raises(ValueError):freeze_design(selected,[10]*9,500)


def test_unserved_rows_remain_failures_and_costs_include_failed_attempts():
    design=freeze_design(pool(),[1]*10,15,max_n=10)
    rows=[{'id':design['rows'][0]['id'],'arm':'single','status':'completed','exact_match':1,
        'calls':1,'tokens':100,'elapsed_seconds':1,'short_answer_words':2},
        {'id':design['rows'][1]['id'],'arm':'single','status':'error','exact_match':0,
        'calls':1,'tokens':None,'elapsed_seconds':2}]
    result=summarize(design,rows)['arms']['single']
    assert result['requested']==5 and result['completed']==1
    assert result['requested_exact_match']==.2 and result['completed_exact_match']==1
    assert result['calls_per_correct']==2 and result['tokens_per_correct'] is None
    assert score('Two seconds.', 'two seconds')['exact_match']==1
    assert score('The answer is two seconds.', 'two seconds')['exact_match']==0
