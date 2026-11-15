"""Verify published own-document aggregate records without redistributing private gold."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from own_document_profile import summarize
from robustness_closure import wilson


def verify(root):
    def read(name):
        return json.loads((root / name).read_text(encoding='utf-8'))
    design, records = read('design.json'), read('numeric_records.json')
    frozen = json.loads((ROOT / 'reports/releases/r4/own_document_preflight/design.json').read_text())
    assert design == frozen
    assert len(records) == len({r['task_id'] for r in records}) == design['requested_tasks'] == 19
    assert design['pdf_questions'] == 17 and design['selection_uses_model_outcomes'] is False
    assert summarize(design, records) == read('summary.json')
    allowed = {'task_id', 'format', 'status', 'exact_match', 'answerable', 'gold_quote_location_found',
        'abstained', 'release_state', 'hold_reasons', 'token_f1', 'call_records', 'calls', 'tokens',
        'elapsed_seconds', 'numerical_failure', 'truncated'}
    call_keys = {'input_tokens', 'output_tokens', 'generation_attempted', 'max_length_reached',
        'numerical_failure', 'status', 'elapsed_seconds', 'time_to_first_token_seconds'}
    for r in records:
        assert set(r) <= allowed
        assert r['calls'] == len(r['call_records']) == 1
        assert all(set(c) <= call_keys for c in r['call_records'])
        assert r['tokens'] == sum(c['input_tokens'] + c['output_tokens'] for c in r['call_records'])
        assert r['exact_match'] in (0, 1)
        assert r['status'] == 'completed' or r['exact_match'] == 0
    costs = read('cost_and_uncertainty.json')['groups']
    for kind in ('pdf', 'docx'):
        rs = [r for r in records if r['format'] == kind]
        correct = sum(r['exact_match'] for r in rs)
        assert costs[kind]['candidate_em_nominal_wilson95'] == wilson(correct, len(rs))
        assert costs[kind]['calls'] == sum(r['calls'] for r in rs)
        assert costs[kind]['tokens_per_candidate_em_correct'] == sum(r['tokens'] for r in rs) / correct
        assert costs[kind]['seconds_per_candidate_em_correct'] == sum(r['elapsed_seconds'] for r in rs) / correct
    usage, source = read('compute_usage.json'), read('executed_source.json')
    assert usage['source_commit'] == source['source_commit'] == '500c7e0d7f372007d6ed9af49ac1cfa63428de6f'
    assert usage['experiment_exit_code'] == usage['summary_exit_code'] == 0
    assert 0 < usage['elapsed_seconds_including_setup'] <= usage['session_ceiling_seconds'] == 1800
    assert usage['reserved_gpu_seconds'] == 3600 and len(usage['allocated_devices']) == 2
    assert not usage['paid_compute'] and not read('summary.json')['hosted_calls']
    for name, expected in read('publication_manifest.json')['sha256'].items():
        assert Path(name).name == name
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    return {'requested_tasks': len(records), 'pdf_questions': 17, 'verified': True,
        'scope': 'Numerical consistency only; semantic labels and raw outputs remain private'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    print(json.dumps(verify(p.parse_args().directory)))
