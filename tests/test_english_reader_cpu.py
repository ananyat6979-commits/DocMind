"""Public-cohort separation and timing-only decisions use synthetic fixtures."""

from copy import deepcopy

import pytest

from scripts.english_reader_cpu import choose, select, sha


def source():
    return {
        "data": [
            {
                "title": "synthetic",
                "paragraphs": [
                    {
                        "context": f"Paragraph {i}.",
                        "qas": [
                            {
                                "id": f"{i}-yes",
                                "question": "How much?",
                                "is_impossible": False,
                                "answers": [{"text": str(i), "answer_start": 10}],
                            },
                            {
                                "id": f"{i}-no",
                                "question": "When?",
                                "is_impossible": True,
                                "answers": [],
                            },
                        ],
                    }
                    for i in range(240)
                ],
            }
        ]
    }


def test_contexts_are_unique_and_disjoint_from_all_prior_questions():
    data = source()
    prior = {"0-yes", "1-no"}
    pilot, final, excluded = select(data, prior)
    assert len(pilot) == 10 and len(final) == 200
    assert sum(r["answerable"] for r in pilot) == 5
    assert sum(r["answerable"] for r in final) == 100
    assert len({r["context_sha256"] for r in pilot + final}) == 210
    assert excluded == {sha("Paragraph 0."), sha("Paragraph 1.")}
    assert not {r["context_sha256"] for r in pilot + final} & excluded
    assert select(data, prior) == (pilot, final, excluded)
    # Every even prefix of the candidate pool remains balanced.
    assert all(
        sum(r["answerable"] for r in final[:n]) == n // 2 for n in range(2, 201, 2)
    )


def test_insufficient_fresh_paragraphs_fail_instead_of_reusing_contexts():
    data = source()
    data["data"][0]["paragraphs"] = data["data"][0]["paragraphs"][:100]
    with pytest.raises(ValueError):
        select(data, set())


def timing(seconds=30):
    return {
        "setup_seconds": 100,
        "charged_pilot_seconds": 400,
        "rows": [
            {"elapsed_seconds": seconds, "generation_completed": True}
            for _ in range(10)
        ],
    }


def test_cohort_decision_ignores_answer_correctness_and_reserves_budget():
    first = timing()
    second = deepcopy(first)
    second["correctness"] = "different unseen outcomes"
    assert choose(first) == choose(second)
    result = choose(first)
    assert result["cohort_size"] == 200
    assert result["remaining_cpu_seconds"] == 19800
    assert result["reserve_fraction"] == 0.2


def test_timeouts_are_included_in_conservative_cost():
    data = timing()
    data["rows"].append({"elapsed_seconds": 600, "generation_completed": False})
    result = choose(data)
    assert result["conservative_seconds_per_question"] == 600
    assert result["cohort_size"] == 0


def test_too_few_usable_timings_do_not_start_a_final_run():
    data = timing()
    data["rows"] = data["rows"][:3]
    assert choose(data)["cohort_size"] == 0


def test_frozen_decision_binds_exact_balanced_prefix():
    from scripts.english_reader_cpu import checked_decision, freeze

    _, rows, _ = select(source(), set())
    manifest = {
        "final_pool": [
            {k: r[k] for k in ("id", "context_sha256", "answerable")} for r in rows
        ]
    }
    result = freeze(timing(), manifest)
    assert checked_decision(result, manifest) == manifest["final_pool"]
    changed = deepcopy(result)
    changed["rows"][0]["id"] = "substituted"
    with pytest.raises(ValueError):
        checked_decision(changed, manifest)
    changed = deepcopy(result)
    changed["cohort_size"] = 199
    with pytest.raises(ValueError):
        checked_decision(changed, manifest)
    changed = deepcopy(manifest)
    changed["final_pool"][0]["context_sha256"] = "changed"
    with pytest.raises(ValueError):
        checked_decision(result, changed)


def test_freeze_command_records_input_hashes_and_refuses_overwrite(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys

    _, rows, _ = select(source(), set())
    manifest = {
        "final_pool": [
            {k: r[k] for k in ("id", "context_sha256", "answerable")} for r in rows
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    timing_path = tmp_path / "timing.json"
    timing_path.write_text(json.dumps(timing()))
    out = tmp_path / "decision.json"
    command = [
        sys.executable,
        "-m",
        "scripts.english_reader_cpu",
        "choose",
        "--timing",
        str(timing_path),
        "--manifest",
        str(manifest_path),
        "--out",
        str(out),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    decision = json.loads(out.read_text())
    assert decision["cohort_size"] == 200
    assert (
        decision["pilot_timing_sha256"]
        == hashlib.sha256(timing_path.read_bytes()).hexdigest()
    )
    before = out.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert out.read_bytes() == before
