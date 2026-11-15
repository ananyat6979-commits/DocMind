import importlib.util
import json
from pathlib import Path
import shutil
import pytest

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('robustness_verifier',root/'scripts/verify_robustness_closure.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_independent_verifier_rejects_false_zero_uncertainty(tmp_path):
    source=root/'reports/releases/r3/robustness'
    for name in ['manifest.json','summary.json','answer_records.json','attempted.json','executed_source.json','compute_usage.json']:shutil.copyfile(source/name,tmp_path/name)
    summary=json.loads((tmp_path/'summary.json').read_text())
    summary['groups']['attack_json_current']['candidate_auto_release_target']['family_any_success_wilson95_nominal']=[0,0]
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(AssertionError):module.verify(tmp_path)
