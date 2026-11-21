"""Consensus import must preserve original votes and incomplete cases."""

from copy import deepcopy
import hashlib
import json

import pytest

from scripts.analyze_source_adjudication import analyze


def seal(packet):
    packet.pop("sha256", None)
    packet["sha256"] = hashlib.sha256(
        json.dumps(packet, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()
    return packet


def fixture():
    item = {
        "id": "one",
        "question": "How long?",
        "answer": "Ten seconds.",
        "claims": [{"id": "c1", "claim": "Ten seconds.", "quote": "Ten seconds."}],
    }
    digest = hashlib.sha256(
        json.dumps(
            {"answer": item["answer"], "claims": item["claims"]}, sort_keys=True
        ).encode()
    ).hexdigest()
    links = [{"id": "one", "task_id": "task", "input_sha256": digest}]
    packet = seal({"items": [item], "cap_seconds": 600})
    left = {
        "packet_sha256": packet["sha256"],
        "elapsed_ms": 1000,
        "records": [
            {
                "id": "one",
                "finished": True,
                "human_adjudicated": True,
                "answer_correctness": "incorrect",
                "claims": {"c1": "unsupported"},
            }
        ],
    }
    right = deepcopy(left)
    right["records"][0].update(answer_correctness="correct", claims={"c1": "supported"})
    empty = seal({"items": [], "cap_seconds": 600})
    originals = {
        "first-original": (packet, links, left),
        "first-changed": (
            empty,
            [],
            {"packet_sha256": empty["sha256"], "elapsed_ms": 0, "records": []},
        ),
        "second": (packet, links, right),
    }
    joint_item = deepcopy(item)
    joint_item.update(
        prior_answer={"first": "incorrect", "second": "correct"},
        prior_claims={"c1": {"first": "unsupported", "second": "supported"}},
    )
    joint = seal(
        {
            "items": [joint_item],
            "cap_seconds": 600,
            "review_role": "source_adjudication",
            "original_second_packet_sha256": packet["sha256"],
        }
    )
    labels = deepcopy(right)
    labels.update(
        packet_sha256=joint["sha256"],
        schema="joint-adjudication-v1",
        participant_mode="joint",
    )
    labels["records"][0]["note"] = "Source page 1 supports the stated duration."
    return joint, links, labels, originals


def test_joint_import_does_not_rewrite_independent_votes():
    args = fixture()
    before = deepcopy(args)
    result = analyze(*args)
    assert args == before
    assert result["finished_cases"] == 1
    assert result["independent_new_review"] is False
    assert result["recomputed_independent_agreement"] is False
    assert result["population_false_release_risk"] is None


def test_single_person_is_not_reported_as_joint_consensus():
    packet, links, labels, originals = fixture()
    labels["participant_mode"] = "single"
    result = analyze(packet, links, labels, originals)
    assert result["scope"] == "Single-person source reconsideration"


@pytest.mark.parametrize(
    "fault", ["prior_vote", "input", "omitted", "rationale", "timer", "schema"]
)
def test_changed_provenance_and_invalid_exports_fail(fault):
    packet, links, labels, originals = fixture()
    if fault == "prior_vote":
        packet["items"][0]["prior_answer"]["first"] = "correct"
    elif fault == "input":
        packet["items"][0]["answer"] = "Twenty seconds."
    elif fault == "omitted":
        packet["items"] = []
        links = []
        labels["records"] = []
    elif fault == "rationale":
        labels["records"][0]["note"] = ""
    elif fault == "timer":
        labels["elapsed_ms"] = 600001
    else:
        labels["schema"] = "second-review-v1"
    seal(packet)
    labels["packet_sha256"] = packet["sha256"]
    with pytest.raises(ValueError):
        analyze(packet, links, labels, originals)


def test_unfinished_case_stays_missing():
    packet, links, labels, originals = fixture()
    labels["records"] = []
    result = analyze(packet, links, labels, originals)
    assert result["finished_cases"] == 0 and result["unfinished_cases"] == 1
    assert result["answer_labels"] == {}


def test_cli_charges_actual_time_and_refuses_overwrite(tmp_path):
    import subprocess
    import sys

    packet, links, labels, originals = fixture()
    manifest, hashes = {}, {}

    def save(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    for role, (original, original_links, votes) in originals.items():
        root = tmp_path / role
        root.mkdir()
        save(root / "actual-answer-packet.private.json", original)
        save(root / "actual-answer-linkage.private.json", original_links)
        export = save(root / "labels.json", votes)
        hashes[role] = hashlib.sha256(export.read_bytes()).hexdigest()
        manifest[role] = {"root": str(root), "labels": str(export)}
    packet["input_exports_sha256"] = hashes
    seal(packet)
    labels["packet_sha256"] = packet["sha256"]
    joint = tmp_path / "joint"
    joint.mkdir()
    save(joint / "actual-answer-packet.private.json", packet)
    save(joint / "actual-answer-linkage.private.json", links)
    exported = save(joint / "labels.json", labels)
    source = save(tmp_path / "originals.json", manifest)
    budget = save(
        tmp_path / "budget.json",
        {
            "charged_total_seconds": 8944.9269,
            "total_pdf_review_allowance_seconds": 9000,
        },
    )
    out = tmp_path / "result.json"
    command = [
        sys.executable,
        "-m",
        "scripts.analyze_source_adjudication",
        "--root",
        str(joint),
        "--labels",
        str(exported),
        "--originals",
        str(source),
        "--budget-before",
        str(budget),
        "--allowance-seconds",
        "9600",
        "--out",
        str(out),
    ]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    result = json.loads(out.read_text())
    assert result["budget"]["charged_total_seconds"] == pytest.approx(8945.9269)
    before = out.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert out.read_bytes() == before
    labels["elapsed_ms"] = 600000
    save(exported, labels)
    command[command.index("9600")] = "9000"
    command[-1] = str(tmp_path / "over-budget.json")
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert not (tmp_path / "over-budget.json").exists()
