"""A model result file never substitutes for successful process completion."""
import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("local_profile", Path(__file__).parents[1]/"scripts/profile_local_models.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def test_completed_child_is_required(tmp_path):
    result = profile.run_bounded([sys.executable, "-c", "print('finished')"], tmp_path, 20)
    assert result["process_passed"] and not result["timed_out"]
    assert result["exit_code"] == 0
    assert json.loads((tmp_path/"process.json").read_text())["process_passed"]


def test_partial_model_record_does_not_override_timeout(tmp_path):
    (tmp_path/"model_checks.json").write_text('{"checks_passed": true}')
    result = profile.run_bounded([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path, .2)
    assert result["timed_out"] and not result["process_passed"]
    assert result["exit_code"] != 0


def test_failed_child_is_not_a_pass(tmp_path):
    result = profile.run_bounded([sys.executable, "-c", "raise SystemExit(3)"], tmp_path, 20)
    assert not result["process_passed"] and not result["timed_out"]
    assert result["exit_code"] == 3
