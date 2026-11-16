"""Synthetic fixtures check data flow and bounds, never create research labels."""

from copy import deepcopy

from scripts.support_risk_component import choose_evidence, evaluate, fit, word_windows


def design():
    return {
        "models": [{}],
        "candidate_depths": [1],
        "numeric_guard_modes": [False],
        "thresholds": [0.5, 0.9],
        "alpha": 0.05,
        "policy_family_size": 2,
        "task_family_size": 6,
        "risk_target": 0.2,
        "minimum_released": 25,
    }


def rows():
    return [
        {
            "id": str(i),
            "split": "calibration",
            "model_index": 0,
            "task": "QA",
            "status": "completed",
            "eligible": True,
            "score_top1": 0.95,
            "score_top4": 0.95,
            "numeric_ok": True,
            "unsupported": False,
            "quality_problem": False,
        }
        for i in range(100)
    ]


def test_selection_cannot_read_heldout_labels():
    data = rows() + [
        {**rows()[0], "id": "test", "split": "heldout", "unsupported": True}
    ]
    original = fit(data, design())
    changed = deepcopy(data)
    changed[-1]["unsupported"] = False
    assert original["policy"] is not None and fit(changed, design()) == original
    assert evaluate(data, original["policy"], design())["unsupported_risk"] is None


def test_strict_task_bound_cannot_hide_a_missing_task_stratum():
    assert fit(rows(), design(), stratified=True)["policy"] is None


def test_word_windows_preserve_source_and_evidence_ranking_is_deterministic():
    text = (
        "The valve closes. The pressure threshold is 40 bar. Operators check the pump."
    )
    windows = word_windows(text, size=6, stride=3)
    assert all(w in text for w in windows)
    picked = choose_evidence("pressure threshold 40", windows, depth=1)
    assert len(picked) == 1 and "pressure" in picked[0] and "40" in picked[0]


def test_bounded_job_output_roundtrip_and_bomb_rejection(tmp_path, monkeypatch):
    import base64
    import json
    import zlib

    import pytest

    from scripts.support_risk_component import export_job_output, unpack_job_output

    source = tmp_path / "source"
    source.mkdir()
    for name, value in [
        (
            "numeric_records.json",
            [{"id": "a", "context_sha256": "trace", "response_sha256": "trace"}],
        ),
        ("identity.json", {}),
        ("timing.json", {}),
    ]:
        (source / name).write_text(json.dumps(value))
    output = tmp_path / "job-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    export_job_output(source)
    payload = output.read_text().strip().split("=", 1)[1]
    unpack_job_output(payload, tmp_path / "received")
    assert json.loads((tmp_path / "received/numeric_records.json").read_text()) == [
        {"id": "a"}
    ]
    bomb = base64.b64encode(zlib.compress(b"x" * 3_000_002)).decode()
    with pytest.raises(ValueError, match="Decoded result bound"):
        unpack_job_output(bomb, tmp_path / "rejected")


def test_result_channel_refuses_accidental_raw_answer_fields(tmp_path, monkeypatch):
    import json

    import pytest

    from scripts.support_risk_component import export_job_output

    source = tmp_path / "source"
    source.mkdir()
    (source / "numeric_records.json").write_text(
        json.dumps([{"id": "a", "response": "raw source must remain private"}])
    )
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    with pytest.raises(ValueError, match="Non-numeric response content"):
        export_job_output(source)


def test_numeric_output_channel_rejects_raw_response_fields(tmp_path, monkeypatch):
    import json

    import pytest

    from scripts.support_risk_component import export_job_output

    source = tmp_path / "source"
    source.mkdir()
    for name, value in [
        ("numeric_records.json", [{"id": "x", "response": "not permitted"}]),
        ("identity.json", {}),
        ("timing.json", {}),
    ]:
        (source / name).write_text(json.dumps(value))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    with pytest.raises(ValueError, match="Non-numeric response content"):
        export_job_output(source)
