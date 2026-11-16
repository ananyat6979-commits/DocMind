"""Same-output quality-policy replay. Raw model outputs and human labels stay private."""

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from quality_policy_control import old_module


def fingerprint(answer, evidence):
    grouped = {}
    for e in evidence:
        grouped.setdefault(e.get("claim", ""), []).append(
            e.get("text") or e.get("quote") or ""
        )
    claims = [
        {"id": "claim-" + str(i + 1), "claim": claim, "quote": "\n\n".join(quotes)}
        for i, (claim, quotes) in enumerate(grouped.items())
    ]
    return hashlib.sha256(
        json.dumps({"answer": answer, "claims": claims}, sort_keys=True).encode()
    ).hexdigest()


def run(profile, cache, review, labels, out, private_out):
    os.environ.update(
        DOCMIND_TEST_MODE="1", DOCMIND_ATTRIBUTION="0", DOCMIND_RELEASE_MODE="draft"
    )
    from docmind import language
    from docmind.agent import reader
    from docmind.review import release
    from docmind.models import Chunk, Document, SearchResult
    from docmind.evidence import validate_claims
    from scripts.analyze_own_answer_review import summarize

    records = json.loads((profile / "private/records.json").read_text())
    gold = json.loads((profile / "private/gold.json").read_text())
    task_map = {r["task_id"]: r for r in gold}
    documents = {d["filename"]: d for d in json.loads(cache.read_text())["documents"]}
    packet = json.loads((review / "actual-answer-packet.private.json").read_text())
    linkage = json.loads((review / "actual-answer-linkage.private.json").read_text())
    judgments = json.loads(labels.read_text())
    summarize(packet, linkage, judgments)
    label_by_id = {
        r["id"]: r
        for r in judgments["records"]
        if r.get("finished") and r.get("human_adjudicated")
    }
    link_by_task = {r["task_id"]: r for r in linkage}
    old_language = old_module("replay_old_language", "docmind/language.py")
    old_evidence = old_module("replay_old_evidence", "docmind/evidence.py")
    old_release = old_module("replay_old_release", "docmind/review/release.py")
    new_info, new_hint = language.language_info, language.short_language_hint
    current_gate, current_policy = release._unvalidated_language, release.apply_policy
    original_validation, original_policy = reader.validate_claims, reader.apply_policy
    revised_languages = {
        name: language.document_language([Document(**u) for u in d["units"]])
        for name, d in documents.items()
    }
    session = SimpleNamespace(calls=0, usage=lambda: {})
    numeric, revised, groups = [], [], {}
    conditions = ("frozen_v4", "language_only", "numeric_only", "combined")

    @contextmanager
    def condition(name):
        fixed = name in ("language_only", "combined")
        language.language_info = new_info if fixed else old_language.language_info
        language.short_language_hint = (
            new_hint if fixed else old_language.short_language_hint
        )
        release._unvalidated_language = (
            current_gate if fixed else old_release._unvalidated_language
        )
        reader.apply_policy = (
            current_policy if name != "frozen_v4" else old_release.apply_policy
        )
        reader.validate_claims = (
            validate_claims
            if name in ("numeric_only", "combined")
            else old_evidence.validate_claims
        )
        try:
            yield fixed
        finally:
            language.language_info, language.short_language_hint = new_info, new_hint
            release._unvalidated_language = current_gate
            reader.apply_policy, reader.validate_claims = (
                original_policy,
                original_validation,
            )

    for row in records:
        for name in conditions:
            with condition(name) as fixed:
                sources = []
                for i, s in enumerate(row["sources"]):
                    metadata = dict(s["metadata"])
                    if fixed:
                        metadata["document_language_info"] = revised_languages[
                            task_map[row["task_id"]]["source_filename"]
                        ]
                        metadata["language_info"] = new_info(s["content"])
                    sources.append(
                        SearchResult(
                            Chunk(
                                s["content"],
                                "cached",
                                metadata.get("filename", "source"),
                                i,
                                chunk_id=s["chunk_id"],
                                file_id=metadata.get("file_id", "cached"),
                                metadata=metadata,
                            ),
                            1,
                            "saved",
                        )
                    )
                if len(row["raw_outputs"]) != 1:
                    raise ValueError("Replay requires one saved output")
                result = reader.CitedReader(SimpleNamespace()).assemble(
                    row["question"],
                    sources,
                    row["raw_outputs"][0],
                    time.monotonic(),
                    session,
                )
                human = "not_reviewed_identical_input"
                link = link_by_task.get(row["task_id"])
                if (
                    link
                    and fingerprint(result.answer, result.evidence)
                    == link["input_sha256"]
                    and link["id"] in label_by_id
                ):
                    human = label_by_id[link["id"]]["answer_correctness"]
                numeric.append(
                    {
                        "task_id": row["task_id"],
                        "format": row["format"],
                        "condition": name,
                        "release_state": result.release_state,
                        "hold_reasons": result.hold_reasons,
                        "finish_reason": result.finish_reason,
                        "human_correctness_if_identical": human,
                    }
                )
                if name == "combined":
                    revised.append(
                        {
                            **row,
                            "answer": result.answer,
                            "evidence": result.evidence,
                            "short_answer": result.short_answer,
                            "release_state": result.release_state,
                            "hold_reasons": result.hold_reasons,
                            "status": "format_failure"
                            if result.finish_reason == "format_failure"
                            else "completed",
                        }
                    )
    for name in conditions:
        rows = [r for r in numeric if r["condition"] == name and r["format"] == "pdf"]
        groups[name] = {
            "requested_pdf": 17,
            "drafts": sum(r["release_state"] == "draft" for r in rows),
            "held": sum(r["release_state"] == "held" for r in rows),
            "correct_held_identical_inputs": sum(
                r["release_state"] == "held"
                and r["human_correctness_if_identical"] == "correct"
                for r in rows
            ),
            "incorrect_drafts_identical_inputs": sum(
                r["release_state"] == "draft"
                and r["human_correctness_if_identical"] == "incorrect"
                for r in rows
            ),
            "draft_inputs_needing_new_labels": sum(
                r["release_state"] == "draft"
                and r["human_correctness_if_identical"]
                == "not_reviewed_identical_input"
                for r in rows
            ),
            "hold_reasons": dict(
                Counter(reason for r in rows for reason in r["hold_reasons"])
            ),
        }
    summary = {
        "scope": "Post-hoc same-output policy replay, cached historical extraction unchanged; not fresh generation or independent validation",
        "groups": groups,
        "new_reader_calls": 0,
        "false_release_risk": "insufficient evidence",
        "human_reuse": "Only exact reviewed answer/evidence fingerprints; changed inputs are unreviewed",
        "layout_limit": "This replay does not reparse historical PDFs or apply fresh layout geometry",
        "input_sha256": {
            "records": hashlib.sha256(
                (profile / "private/records.json").read_bytes()
            ).hexdigest(),
            "cache": hashlib.sha256(cache.read_bytes()).hexdigest(),
            "human_export": hashlib.sha256(labels.read_bytes()).hexdigest(),
        },
    }
    out.mkdir(parents=True, exist_ok=False)
    private_out.mkdir(parents=True, exist_ok=False)
    (private_out / "private").mkdir()
    for path, value in [
        (out / "numeric_records.json", numeric),
        (out / "summary.json", summary),
        (private_out / "private/records.json", revised),
        (private_out / "private/gold.json", gold),
    ]:
        path.write_bytes((json.dumps(value, indent=2) + "\n").encode())
    print(json.dumps(groups))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "cache", "review", "labels", "out", "private-out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    run(a.profile, a.cache, a.review, a.labels, a.out, a.private_out)
