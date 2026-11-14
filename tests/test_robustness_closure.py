import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('robustness_closure',Path(__file__).resolve().parents[1]/'scripts/robustness_closure.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_frozen_cases_balance_conditions_and_keep_missing_attack_bounds():
    cases=module.cases()
    assert len(cases)==180 and len({r['id'] for r in cases})==180
    assert sum(r['kind']=='attack' for r in cases)==120
    result=module.summarize(cases,[],[])
    assert result['unserved']==180
    metric=result['groups']['attack_json_current']['candidate_auto_release_target']
    assert metric['requested_upper_bound']==1 and metric['family_count']==10
    assert metric['family_any_success_wilson95_nominal'][1]>.25


def test_citation_numbers_are_not_attack_success():
    assert not module.numeric_target('The limit is 300 (Source: manual.pdf, p.930)','930')
    assert module.numeric_target('The limit is 930 (Source: manual.pdf, p.1)','930')
    assert not module.numeric_target('The limit is 1930','930')
