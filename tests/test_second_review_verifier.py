"""Public aggregate integrity tests never read private reviewer exports."""

import json
import shutil
from pathlib import Path

import pytest

from scripts.verify_second_review import verify


@pytest.fixture
def aggregate(tmp_path):
    source = Path(__file__).resolve().parents[1] / "reports/releases/r7"
    for name in ("second_review_agreement.json", "review_budget_after_second.json"):
        shutil.copyfile(source / name, tmp_path / name)
    return tmp_path


def test_published_pair_counts_and_budget_recompute(aggregate):
    result = verify(aggregate)
    assert result["paired_answers"] == 16 and result["paired_claims"] == 33


def test_inflated_kappa_is_rejected(aggregate):
    path = aggregate / "second_review_agreement.json"
    report = json.loads(path.read_text())
    report["answers"]["cohen_kappa"] = 1.0
    path.write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        verify(aggregate)


def test_review_time_cannot_disappear_from_the_budget(aggregate):
    path = aggregate / "review_budget_after_second.json"
    report = json.loads(path.read_text())
    report["charged_total_seconds"] -= 600
    path.write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        verify(aggregate)
