"""Offline batch mechanics and fixed budgets use synthetic inputs only."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.english_reader_batch_cpu import (
    choose,
    checked_decision,
    supplied_context,
    through_eos,
)


def manifest():
    return {
        "final_pool": [
            {"id": str(i), "context_sha256": str(i), "answerable": bool(i % 2)}
            for i in range(200)
        ]
    }


def timing(seconds=40):
    return {
        "setup_seconds": 100,
        "batches": [
            {
                "requested_sequences": 4,
                "elapsed_seconds": seconds,
                "generation_completed": True,
            },
            {
                "requested_sequences": 4,
                "elapsed_seconds": seconds,
                "generation_completed": True,
            },
            {
                "requested_sequences": 2,
                "elapsed_seconds": seconds / 2,
                "generation_completed": True,
            },
        ],
    }


def test_padding_after_first_eos_is_not_counted_as_generated_output():
    assert through_eos([10, 11, 99, 99, 99], {99}) == [10, 11, 99]
    assert through_eos([10, 98, 99], {98, 99}) == [10, 98]
    assert through_eos([10, 11], {99}) == [10, 11]


def test_timing_rule_binds_balanced_full_batches_and_global_cpu_ceiling():
    plan = choose(timing(), manifest())
    assert plan["cohort_size"] == 200
    assert checked_decision(plan, manifest()) == 200
    assert plan["cumulative_cpu_cap_seconds"] == 21600
    assert (
        plan["pilot_reservation_seconds"]
        + plan["prior_fp16_reservation_seconds"]
        + plan["final_wall_cap_seconds"]
        == 21600
    )
    changed = deepcopy(plan)
    changed["rows"][0]["id"] = "substituted"
    with pytest.raises(ValueError):
        checked_decision(changed, manifest())


def test_failed_and_interrupted_batches_are_not_hidden_from_cost():
    data = timing()
    data["batches"].append(
        {
            "requested_sequences": 2,
            "elapsed_seconds": 1200,
            "generation_completed": False,
        }
    )
    assert choose(data, manifest())["cohort_size"] == 0
    data["batches"] = data["batches"][:1]
    assert choose(data, manifest())["cohort_size"] == 0


def test_complete_scoped_context_retains_actual_document_language_for_abstention():
    source = SimpleNamespace(
        chunk=SimpleNamespace(
            file_id="one", metadata={"document_language_info": {"language": "fr"}}
        )
    )
    context = supplied_context(source, ["one"])
    assert context["languages"] == ["fr"]
    assert context["documents"][0]["quarantined_pages"] == []
    assert supplied_context(source, []) == {
        "partial": False,
        "languages": [],
        "documents": [],
    }


def test_new_pilot_excludes_prior_planned_pool_and_reconstructs_exactly(
    tmp_path, monkeypatch
):
    import hashlib
    import json
    import scripts.english_reader_batch_cpu as batch
    import scripts.english_reader_cpu as original

    data = {
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
    raw = json.dumps(data).encode()
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(batch, "SOURCE_SHA", digest)
    monkeypatch.setattr(original, "SOURCE_SHA", digest)
    monkeypatch.setattr(batch, "tracked_ids", lambda: set())
    pilot, final, _ = original.select(data, set())
    fields = lambda rows: [
        {k: r[k] for k in ("id", "context_sha256", "answerable")} for r in rows
    ]
    previous = {"pilot": fields(pilot), "final_pool": fields(final)}
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "input_manifest.json").write_text(json.dumps(previous))
    (prior / "cohort_decision.json").write_text(json.dumps({"cohort_size": 0}))
    source = tmp_path / "source.json"
    source.write_bytes(raw)
    out = tmp_path / "new"
    actual = batch.prepare(source, prior, out)
    assert actual["final_pool"] == previous["final_pool"]
    assert not {r["context_sha256"] for r in actual["pilot"]} & {
        r["context_sha256"] for r in previous["pilot"] + previous["final_pool"]
    }
    original.reconstruct(source, out / "input_manifest.json", tmp_path / "reproduced")
    assert (out / "private/pool.json").read_bytes() == (
        tmp_path / "reproduced/private/pool.json"
    ).read_bytes()
    (prior / "cohort_decision.json").write_text(json.dumps({"cohort_size": 40}))
    with pytest.raises(ValueError):
        batch.prepare(source, prior, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_actual_assembly_retains_selected_language_for_explicit_absence(monkeypatch):
    import json
    import time
    from docmind.config import CONFIG
    from docmind.models import Chunk, SearchResult
    from scripts.english_reader_batch_cpu import assemble_sequence

    monkeypatch.setenv("DOCMIND_ATTRIBUTION", "0")
    monkeypatch.setattr(CONFIG.review, "release_mode", "draft")
    monkeypatch.setattr(CONFIG.review, "hold_until_approved", False)
    raw = json.dumps({"abstained": True, "short_answer": "", "claims": []})

    def source(language):
        return SearchResult(
            Chunk(
                "The report describes valve operation.",
                "d",
                "paragraph.txt",
                0,
                file_id="selected",
                chunk_id="chunk",
                metadata={"document_language_info": {"language": language}},
            ),
            1,
            "supplied_paragraph",
        )

    english = assemble_sequence(
        "What is the missing launch date?", source("en"), raw, time.monotonic(), {}
    )
    assert english.document_context["languages"] == ["en"]
    assert english.document_context["documents"][0]["file_id"] == "selected"
    assert english.release_state == "draft"
    french = assemble_sequence(
        "What is the missing launch date?", source("fr"), raw, time.monotonic(), {}
    )
    assert french.document_context["languages"] == ["fr"]
    assert french.release_state == "held"
