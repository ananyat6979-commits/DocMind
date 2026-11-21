"""Bounded CPU feasibility and current-reader control on fresh public paragraphs."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time

SOURCE_SHA = "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"
SEED = "english-reader-cpu-v1:"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def select(data, historical_ids):
    rows, excluded = [], set()
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            identity = sha(paragraph["context"])
            if any(q["id"] in historical_ids for q in paragraph["qas"]):
                excluded.add(identity)
            for question in paragraph["qas"]:
                rows.append(
                    {
                        "id": question["id"],
                        "title": article["title"],
                        "context": paragraph["context"],
                        "context_sha256": identity,
                        "question": question["question"],
                        "answerable": not question["is_impossible"],
                        "answers": question["answers"],
                    }
                )
    rows.sort(key=lambda row: sha(SEED + row["id"]))
    seen = set(excluded)

    def take(count):
        groups = {}
        for answerable in (False, True):
            picked = []
            for row in rows:
                if (
                    row["answerable"] == answerable
                    and row["context_sha256"] not in seen
                ):
                    seen.add(row["context_sha256"])
                    picked.append(row)
                    if len(picked) == count:
                        break
            if len(picked) != count:
                raise ValueError("Not enough fresh distinct public paragraphs")
            groups[answerable] = picked
        return [row for pair in zip(groups[False], groups[True]) for row in pair]

    pilot = take(5)
    final = take(100)
    return pilot, final, excluded


def prepare(source, out):
    if out.exists():
        raise FileExistsError("Keep previously frozen cohorts unchanged")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("Public source hash changed")
    tracked = subprocess.run(
        ["git", "ls-files", "reports"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    ids = set()
    for name in tracked:
        path = Path(name)
        if path.suffix in (".json", ".jsonl") and path.stat().st_size < 20000000:
            ids.update(
                re.findall(
                    r"(?<![0-9a-f])[0-9a-f]{24}(?![0-9a-f])",
                    path.read_text(encoding="utf-8"),
                )
            )
    pilot, final, excluded = select(json.loads(raw), ids)
    out.mkdir(parents=True)
    write(out / "private/pool.json", {"pilot": pilot, "final": final})
    manifest = {
        "source_sha256": SOURCE_SHA,
        "selection_seed": SEED,
        "historical_ids_sha256": sha(json.dumps(sorted(ids))),
        "excluded_contexts_sha256": sha(json.dumps(sorted(excluded))),
        "excluded_context_count": len(excluded),
        "pilot": [
            {
                "id": r["id"],
                "context_sha256": r["context_sha256"],
                "answerable": r["answerable"],
            }
            for r in pilot
        ],
        "final_pool": [
            {
                "id": r["id"],
                "context_sha256": r["context_sha256"],
                "answerable": r["answerable"],
            }
            for r in final
        ],
        "selection_uses_outcomes": False,
        "private_pool_sha256": hashlib.sha256(
            (out / "private/pool.json").read_bytes()
        ).hexdigest(),
    }
    write(out / "input_manifest.json", manifest)
    return manifest


def choose(timing):
    served = [
        r["elapsed_seconds"] for r in timing["rows"] if r.get("generation_completed")
    ]
    if len(served) < 4 or not all(math.isfinite(v) and v > 0 for v in served):
        return {"cohort_size": 0, "reason": "Fewer than four usable generation timings"}
    attempted = [r["elapsed_seconds"] for r in timing["rows"]]
    if not all(math.isfinite(v) and v > 0 for v in attempted):
        raise ValueError("Invalid attempted-question timing")
    per_question = max(max(attempted), 2 * statistics.median(served))
    remaining = 21600 - max(1800, timing["charged_pilot_seconds"])
    setup = 2 * timing["setup_seconds"]
    size = min(
        200, 2 * max(0, math.floor(0.8 * max(0, remaining - setup) / per_question / 2))
    )
    return {
        "cohort_size": size if size >= 40 else 0,
        "timing_supported_size": size,
        "conservative_seconds_per_question": per_question,
        "reserved_setup_seconds": setup,
        "remaining_cpu_seconds": remaining,
        "reserve_fraction": 0.2,
        "reason": "Timing-only rule; no answers, correctness or routing inspected"
        if size >= 40
        else "CPU timing cannot support forty cases within the frozen ceiling",
    }


def content_hash(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=True))


def freeze(timing, manifest):
    decision = choose(timing)
    size = decision["cohort_size"]
    selected = manifest["final_pool"][:size]
    if len(selected) != size or sum(r["answerable"] for r in selected) != size // 2:
        raise ValueError("Frozen cohort must be available and balanced")
    return {
        **decision,
        "rows": selected,
        "input_manifest_content_sha256": content_hash(manifest),
        "cohort_content_sha256": content_hash(selected),
        "selection_uses_outcomes": False,
    }


def checked_decision(decision, manifest):
    size = decision["cohort_size"]
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 40
        or size > 200
        or size % 2
    ):
        raise ValueError("A viable even cohort of forty to two hundred is required")
    if decision.get("input_manifest_content_sha256") != content_hash(manifest):
        raise ValueError("Decision refers to different frozen inputs")
    rows = manifest["final_pool"][:size]
    if len(rows) != size or sum(r["answerable"] for r in rows) != size // 2:
        raise ValueError("Selected cohort must be available and balanced")
    if decision.get("rows") != rows or decision.get(
        "cohort_content_sha256"
    ) != content_hash(rows):
        raise ValueError("Selected cohort differs from the frozen prefix")
    return rows


def run(root, phase, deadline, decision=None):
    import os

    os.environ.pop("GROQ_API_KEY", None)
    os.environ.update(
        DOCMIND_TEST_MODE="0",
        DOCMIND_ATTRIBUTION="0",
        DOCMIND_RELEASE_MODE="draft",
        DOCMIND_HOME=str(root / "runtime"),
        DOCMIND_SECRETS_FILE=str(root / "absent.env"),
        TOKENIZERS_PARALLELISM="false",
    )
    raw_pool = (root / "private/pool.json").read_bytes()
    manifest = json.loads((root / "input_manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(raw_pool).hexdigest() != manifest["private_pool_sha256"]:
        raise ValueError("Frozen question, context or reference inputs changed")
    pool = json.loads(raw_pool)
    rows = pool[phase]
    if phase == "final":
        if not decision:
            raise ValueError("Freeze a timing-only cohort before final generation")
        selected = checked_decision(decision, manifest)
        rows = rows[: len(selected)]
    out = root / phase
    out.mkdir(exist_ok=False)
    records = [
        {"id": row["id"], "context_sha256": row["context_sha256"], "status": "unserved"}
        for row in rows
    ]
    write(out / "private/records.json", records)
    timing = {"rows": [], "setup_seconds": None, "charged_pilot_seconds": None}
    started = time.time()
    import torch
    from docmind.agent.reader import CitedReader
    from docmind.language import language_info
    from docmind.llm.local_reader import LocalReader
    from docmind.models import Chunk, SearchResult

    if torch.cuda.is_available():
        raise RuntimeError("CPU-only allocation required")
    torch.set_num_threads(2)

    class CapturedReader(LocalReader):
        last_raw = None

        def complete(self, *args, **kwargs):
            self.last_raw = None
            self.last_raw = super().complete(*args, **kwargs)
            return self.last_raw

    reader = CapturedReader(device="cpu", max_input_tokens=8192, max_new_tokens=512)
    if next(reader.model.parameters()).dtype != torch.float16:
        raise ValueError("Pinned FP16 inference required for this CPU profile")
    write(
        out / "reader_identity.json",
        {
            **reader.identity,
            "device": "cpu",
            "gpu_seconds": 0,
            "actual_dtype": str(next(reader.model.parameters()).dtype),
            "torch_version": torch.__version__,
            "cpu_threads": 2,
        },
    )
    timing["setup_seconds"] = time.time() - started

    class Frozen:
        def __init__(self, result):
            self.result = result

        def search(self, question, file_ids=None):
            return [self.result]

    for index, row in enumerate(rows):
        if time.time() >= deadline - 5:
            break
        record = records[index]
        reader.last_raw = None
        record.update(status="in_progress", started_unix=time.time())
        write(out / "private/records.json", records)
        before = len(reader.call_records)
        start = time.monotonic()
        try:
            hint = language_info(row["context"])
            chunk = Chunk(
                row["context"],
                "public-paragraph",
                "paragraph.txt",
                0,
                chunk_id=row["context_sha256"],
                file_id=row["context_sha256"][:16],
                metadata={
                    "page": 1,
                    "language_info": hint,
                    "document_language_info": hint,
                },
            )
            result = CitedReader(
                Frozen(SearchResult(chunk, 1.0, "supplied_paragraph")),
                reader,
                deadline_seconds=min(600, max(1, deadline - time.time() - 5)),
            ).run(row["question"])
            record.update(
                status="completed",
                answer=result.answer,
                short_answer=result.short_answer,
                abstained=result.abstained,
                release_state=result.release_state,
                hold_reasons=result.hold_reasons,
                finish_reason=result.finish_reason,
                evidence=result.evidence,
            )
        except Exception as exc:
            record.update(status="error", error=type(exc).__name__)
        record["raw_output"] = reader.last_raw
        record["call_records"] = reader.call_records[before:]
        elapsed = time.monotonic() - start
        timing["rows"].append(
            {
                "id": row["id"],
                "elapsed_seconds": elapsed,
                "generation_completed": any(
                    r["status"] == "completed" for r in record["call_records"]
                ),
                "input_tokens": sum(r["input_tokens"] for r in record["call_records"])
                if all(
                    r.get("input_tokens") is not None for r in record["call_records"]
                )
                else None,
                "output_tokens": sum(r["output_tokens"] for r in record["call_records"])
                if all(
                    r.get("output_tokens") is not None for r in record["call_records"]
                )
                else None,
                "numerical_failure": any(
                    r.get("numerical_failure") for r in record["call_records"]
                ),
            }
        )
        timing["charged_pilot_seconds"] = time.time() - started
        write(out / "private/records.json", records)
        write(out / "timing.json", timing)
    timing["charged_pilot_seconds"] = time.time() - started
    write(out / "timing.json", timing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--phase", choices=("pilot", "final"), required=True)
    p.add_argument("--deadline-unix", type=float, required=True)
    p.add_argument("--decision", type=Path)
    p = sub.add_parser("choose")
    p.add_argument("--timing", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.out)
    elif args.command == "run":
        run(
            args.root,
            args.phase,
            args.deadline_unix,
            json.loads(args.decision.read_text()) if args.decision else None,
        )
    else:
        if args.out.exists():
            raise FileExistsError("Keep the frozen cohort decision")
        raw = args.timing.read_bytes()
        decision = freeze(
            json.loads(raw), json.loads(args.manifest.read_text(encoding="utf-8"))
        )
        decision["pilot_timing_sha256"] = hashlib.sha256(raw).hexdigest()
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(decision, handle, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")


if __name__ == "__main__":
    main()
