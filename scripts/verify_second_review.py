"""Verify public paired-review aggregates without reading private labels."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.analyze_independent_review import agreement


def verify(root):
    report = json.loads(
        (root / "second_review_agreement.json").read_text(encoding="utf-8")
    )
    budget = json.loads(
        (root / "review_budget_after_second.json").read_text(encoding="utf-8")
    )
    for key in (
        "requested_second_review_answers",
        "finished_second_review_answers",
        "paired_answers",
        "unfinished_answers",
        "completed_without_matching_reference",
    ):
        assert type(report[key]) is int and 0 <= report[key] <= 17
    assert (
        report["finished_second_review_answers"] + report["unfinished_answers"]
        == report["requested_second_review_answers"]
        == 17
    )
    assert (
        report["paired_answers"] + report["completed_without_matching_reference"]
        == report["finished_second_review_answers"]
    )
    assert report["population_false_release_risk"] is None
    assert report["adjudication"].startswith("Not performed")
    for kind, allowed, maximum in (
        ("answers", {"correct", "incorrect", "unclear"}, 17),
        ("claims", {"supported", "unsupported", "unclear"}, 36),
    ):
        data = report[kind]
        pairs = []
        for key, count in data["confusion"].items():
            left, right = key.split("|")
            assert left in allowed and right in allowed
            assert type(count) is int and 0 <= count <= maximum
            pairs.extend([(left, right)] * count)
        assert len(pairs) <= maximum
        calculated = agreement(pairs)
        assert calculated["paired_labels"] == data["paired_labels"]
        for key in ("observed_agreement", "cohen_kappa"):
            expected, actual = calculated[key], data[key]
            assert (
                actual is None
                if expected is None
                else isinstance(actual, (int, float))
                and math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12)
            )
        assert data["paired_answer_clusters"] == report["paired_answers"]
        if data["resamples"]:
            assert report["paired_answers"] >= 10
            assert (
                data["resamples"]
                == data["defined_resamples"] + data["undefined_resamples"]
                == 2000
            )
            interval = data["kappa_interval95"]
            assert (
                interval is None
                or len(interval) == 2
                and -1 <= interval[0] <= interval[1] <= 1
            )
        else:
            assert data["kappa_interval95"] is None and report["paired_answers"] < 10
    assert report["answers"]["paired_labels"] == report["paired_answers"]
    for value in [
        report["packet_sha256"],
        report["input_export_sha256"]["second"],
        *report["input_export_sha256"]["primary"],
        *report["primary_packet_sha256"],
        *report["analysis_code_sha256"].values(),
    ]:
        assert len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    assert 0 <= report["recorded_second_review_seconds"] <= 600
    assert (
        budget["second_review_recorded_seconds"]
        == report["recorded_second_review_seconds"]
        <= 600
    )
    assert math.isclose(
        budget["charged_total_seconds"],
        budget["charged_before_second_review_seconds"]
        + budget["second_review_recorded_seconds"],
    )
    assert (
        budget["charged_total_seconds"]
        <= budget["total_pdf_review_allowance_seconds"]
        == 9000
    )
    assert math.isclose(
        budget["remaining_seconds"],
        9000 - budget["charged_total_seconds"],
        abs_tol=1e-9,
    )
    assert (
        budget["independent_questions_added"] == 0
        and budget["raw_labels_in_git"] is False
    )
    manifest = root / "publication_manifest.json"
    if manifest.exists():
        for name, digest in json.loads(manifest.read_text())["sha256"].items():
            assert Path(name).name == name and "/" not in name and "\\" not in name
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    return {
        "verified": True,
        "paired_answers": report["paired_answers"],
        "paired_claims": report["claims"]["paired_labels"],
        "scope": "Aggregate point metrics, denominators, budget and saved-byte integrity; private label matching and cluster intervals require private inputs",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory)))
