import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('review_cost_import',Path(__file__).resolve().parents[1]/'scripts/summarize_review_cost.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture():
    key={'packet_sha256':'expected','records':[{'review_id':'item','practice':False,'layout':'A','benchmark_exact_match':True}]}
    # Synthetic schema fixture, not a human observation.
    export={'packet_sha256':'expected','visible_ms':1000,'started_at':'2026-01-01T00:00:00Z','exported_at':'2026-01-01T00:00:01Z',
        'records':[{'review_id':'item','visible_ms':1000,'question_validity':'valid','answer_correctness':'correct','finished':False}]}
    return export,key


def test_partial_export_is_not_completed_human_evidence():
    export,key=fixture();result=module.summarize(export,key)
    assert result['finished_nonpractice']==0 and result['missing_nonpractice']==1
    assert result['layout_descriptives']['A']['median_visible_seconds'] is None
    assert not result['population_extrapolation']


def test_wrong_packet_and_fabricated_time_are_rejected():
    export,key=fixture();export['packet_sha256']='different'
    with pytest.raises(ValueError):module.summarize(export,key)
    export,key=fixture();export['visible_ms']=1300000
    with pytest.raises(ValueError):module.summarize(export,key)


def test_post_cap_labels_are_retained_without_zero_time_savings():
    export,key=fixture();export['visible_ms']=1200000;export['records'][0].update(visible_ms=1200000,finished=True,finished_at='2026-01-01T00:20:01Z')
    key['records'].append({'review_id':'late','practice':False,'layout':'B','benchmark_exact_match':False})
    export['records'].append({'review_id':'late','visible_ms':0,'question_validity':'valid','answer_correctness':'incorrect','finished':True,'finished_at':'2026-01-01T00:45:00Z'})
    export['exported_at']='2026-01-01T00:46:00Z'
    result=module.summarize(export,key)
    assert result['finished_nonpractice']==2 and result['timing_censored_nonpractice']==2
    assert result['fully_timed_nonpractice']==0 and result['between_layout_time_savings'] is None
    assert result['layout_descriptives']['B']['median_visible_seconds'] is None
