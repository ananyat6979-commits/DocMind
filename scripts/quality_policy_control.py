"""Frozen English gold-input routing control. No reader calls or new human labels."""

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = "8640702"


def old_module(name, file):
    source = subprocess.check_output(
        ["git", "show", BASE + ":" + file], cwd=ROOT
    ).decode()
    module = ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, "<frozen source>", "exec"), module.__dict__)
    return module


def run(design, output):
    os.environ.update(
        DOCMIND_TEST_MODE="1", DOCMIND_ATTRIBUTION="0", DOCMIND_RELEASE_MODE="draft"
    )
    from docmind import language
    from docmind.agent import reader
    from docmind.review import release
    from docmind.models import Chunk, Document, SearchResult
    from docmind.evidence import validate_claims

    frozen = json.loads(design.read_text())
    with urllib.request.urlopen(frozen["source_url"], timeout=60) as response:
        raw = response.read(5_000_001)
    if hashlib.sha256(raw).hexdigest() != frozen["source_sha256"]:
        raise ValueError("Public source changed")
    lookup = {
        q["id"]: (p["context"], q)
        for a in json.loads(raw)["data"]
        for p in a["paragraphs"]
        for q in p["qas"]
    }
    old_language = old_module("quality_old_language", "docmind/language.py")
    old_evidence = old_module("quality_old_evidence", "docmind/evidence.py")
    old_release = old_module("quality_old_release", "docmind/review/release.py")
    new_language_info, new_hint = language.language_info, language.short_language_hint
    new_policy, new_validation = release.apply_policy, validate_claims
    original_reader_policy, original_reader_validation = (
        reader.apply_policy,
        reader.validate_claims,
    )
    original_language_gate = release._unvalidated_language
    records = []
    session = SimpleNamespace(calls=0, usage=lambda: {})

    @contextmanager
    def condition(name):
        repaired_language = name in ("language_only", "combined")
        reader.validate_claims = (
            new_validation
            if name in ("numeric_only", "combined")
            else old_evidence.validate_claims
        )
        reader.apply_policy = (
            new_policy if name != "frozen_v4" else old_release.apply_policy
        )
        release._unvalidated_language = (
            original_language_gate
            if repaired_language
            else old_release._unvalidated_language
        )
        language.language_info = (
            new_language_info if repaired_language else old_language.language_info
        )
        language.short_language_hint = (
            new_hint if repaired_language else old_language.short_language_hint
        )
        try:
            yield (
                new_language_info if repaired_language else old_language.language_info,
                language.document_language
                if repaired_language
                else old_language.document_language,
            )
        finally:
            reader.validate_claims, reader.apply_policy = (
                original_reader_validation,
                original_reader_policy,
            )
            language.language_info, language.short_language_hint = (
                new_language_info,
                new_hint,
            )
            release._unvalidated_language = original_language_gate

    for row in frozen["rows"]:
        context, question = lookup[row["id"]]
        if hashlib.sha256(context.encode()).hexdigest() != row["context_sha256"]:
            raise ValueError("Control paragraph changed")
        for name in frozen["conditions"]:
            with condition(name) as (info, document_info):
                identity = document_info([Document(context, "public reference")])
                chunk = Chunk(
                    context,
                    "public reference",
                    "reference.txt",
                    0,
                    chunk_id=row["id"],
                    file_id=row["id"],
                    metadata={
                        "filename": "reference.txt",
                        "document_language_info": identity,
                        "language_info": info(context),
                    },
                )
                source = SearchResult(chunk, 1, "gold_input")

                class Store:
                    def document_context(self, file_ids):
                        return {
                            "partial": False,
                            "languages": [identity["language"]],
                            "documents": [
                                {"file_id": row["id"], "language": identity["language"]}
                            ],
                        }

                engine = reader.CitedReader(SimpleNamespace(store=Store()))
                if row["answerable"]:
                    answer = question["answers"][0]["text"]
                    start = question["answers"][0]["answer_start"]
                    assert context[start : start + len(answer)] == answer
                    payload = {
                        "short_answer": answer,
                        "claims": [
                            {
                                "text": answer,
                                "evidence": [{"chunk_id": row["id"], "quote": answer}],
                            }
                        ],
                    }
                else:
                    payload = {"abstained": True, "short_answer": "", "claims": []}
                result = engine.assemble(
                    question["question"],
                    [source],
                    json.dumps(payload),
                    time.monotonic(),
                    session,
                    [row["id"]],
                )
                records.append(
                    {
                        "id": row["id"],
                        "condition": name,
                        "answerable": row["answerable"],
                        "release_state": result.release_state,
                        "hold_reasons": result.hold_reasons,
                        "finish_reason": result.finish_reason,
                        "document_language": identity["language"],
                    }
                )
    groups = {}
    for name in frozen["conditions"]:
        rows = [r for r in records if r["condition"] == name]
        groups[name] = {
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
    result = {
        "scope": frozen["scope"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        "design_sha256": hashlib.sha256(design.read_bytes()).hexdigest(),
        "groups": groups,
        "new_reader_calls": 0,
        "limitations": [
            "Public reference inputs, not generated-answer correctness or risk calibration",
            "No new human review, paragraph-disjoint from historical artifacts but articles may overlap",
            "One fixed test, no threshold selection from its outcomes; unexpected holds retained",
            "Valid answers with repeated identical spans can still fail the unique-quote contract",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [("numeric_records.json", records), ("summary.json", result)]:
        (output / name).write_bytes((json.dumps(value, indent=2) + "\n").encode())
    print(json.dumps(groups))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--design",
        type=Path,
        default=ROOT / "reports/releases/r5/english_policy_control_design.json",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run(args.design, args.out)
