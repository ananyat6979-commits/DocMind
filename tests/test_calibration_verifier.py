import importlib.util
import json
from pathlib import Path
import shutil
import pytest

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('calibration_verifier',root/'scripts/verify_calibration_closure.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_independent_verifier_rejects_changed_release_count(tmp_path):
    source=root/'reports/releases/r3/calibration'
    for name in ['numeric_records.json','cohort.json','summary.json','policies.json','pool_ids.json']:shutil.copyfile(source/name,tmp_path/name)
    summary=json.loads((tmp_path/'summary.json').read_text())
    summary['policies'][0]['heldout']['released']=1
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(AssertionError):module.verify(tmp_path)
