import json
from pathlib import Path
from scripts.calibration_length_sensitivity import analyze


def test_published_length_sensitivity_recomputes_from_unchanged_records():
    result=analyze(Path('reports/releases/r3/calibration/numeric_records.json'))
    expected=json.loads(Path('reports/releases/r4/calibration_answer_length_sensitivity.json').read_text())
    assert result==expected
    assert result['groups']['more_than_8_words']=={'n':56,'exact_match_correct':0,'exact_match_accuracy':0.0,'em_failures_with_token_f1_at_least_half':5}
    assert result['groups']['at_most_8_words']['exact_match_accuracy']==.5
