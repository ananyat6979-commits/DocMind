"""Recompute bounded quality-control counts without models or private labels."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def verify(root):
    for folder, design_name in [
        ("english_policy_control", "english_policy_control_design.json"),
        ("english_policy_confirmation", "english_policy_confirmation_design.json"),
    ]:
        design = json.loads((root / design_name).read_text())
        records = json.loads((root / folder / "numeric_records.json").read_text())
        result = json.loads((root / folder / "summary.json").read_text())
        expected = {r["id"] for r in design["rows"]}
        assert len(expected) == design["n"] == 100
        assert sum(r["answerable"] for r in design["rows"]) == 75
        assert len(records) == len({(r["id"], r["condition"]) for r in records}) == 400
        assert (
            result["design_sha256"]
            == hashlib.sha256((root / design_name).read_bytes()).hexdigest()
        )
        for condition in design["conditions"]:
            rows = [r for r in records if r["condition"] == condition]
            assert {r["id"] for r in rows} == expected
            group = {
                "requested": 100,
                "completed": len(rows),
                "held_gold_inputs": sum(r["release_state"] == "held" for r in rows),
                "held_answerable_gold_inputs": sum(
                    r["release_state"] == "held" and r["answerable"] for r in rows
                ),
                "held_unanswerable_gold_inputs": sum(
                    r["release_state"] == "held" and not r["answerable"] for r in rows
                ),
                "hold_reasons": dict(
                    Counter(reason for r in rows for reason in r["hold_reasons"])
                ),
            }
            assert result["groups"][condition] == group
        assert result["new_reader_calls"] == 0
    first = json.loads((root / "english_policy_control_design.json").read_text())
    second = json.loads((root / "english_policy_confirmation_design.json").read_text())
    assert not ({r["id"] for r in first["rows"]} & {r["id"] for r in second["rows"]})
    assert not (
        {r["context_sha256"] for r in first["rows"]}
        & {r["context_sha256"] for r in second["rows"]}
    )
    rows = json.loads((root / "own_policy_replay/numeric_records.json").read_text())
    assert len(rows) == len({(r["task_id"], r["condition"]) for r in rows}) == 76
    assert all(
        set(r)
        <= {
            "task_id",
            "format",
            "condition",
            "release_state",
            "hold_reasons",
            "finish_reason",
            "human_correctness_if_identical",
        }
        for r in rows
    )
    for name, digest in json.loads((root / "publication_manifest.json").read_text())[
        "sha256"
    ].items():
        assert ".." not in Path(name).parts and not Path(name).is_absolute()
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    return {
        "verified": True,
        "control_questions": 100,
        "confirmation_questions": 100,
        "conditions": 4,
        "replay_attempts": 76,
        "scope": "Internal numerical consistency and artifact identity, not independent semantic validation",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    print(json.dumps(verify(p.parse_args().directory)))
