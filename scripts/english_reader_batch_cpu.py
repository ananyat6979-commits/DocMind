"""Separate FP32 offline batched CPU profile with fixed public-reference inputs."""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time
from types import SimpleNamespace

from scripts.english_reader_cpu import SOURCE_SHA, content_hash, sha


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_bytes(
        (
            json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        ).encode()
    )
    temp.replace(path)


def tracked_ids():
    ids = set()
    names = subprocess.run(
        ["git", "ls-files", "reports"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    for name in names:
        path = Path(name)
        if path.suffix in (".json", ".jsonl") and path.stat().st_size < 20000000:
            ids.update(
                re.findall(
                    r"(?<![0-9a-f])[0-9a-f]{24}(?![0-9a-f])",
                    path.read_text(encoding="utf-8"),
                )
            )
    return ids


def supplied_context(source, file_ids=None):
    """Use the real store's single-complete-document metadata shape."""
    selected = file_ids is None or source.chunk.file_id in file_ids
    documents = (
        [
            {
                "file_id": source.chunk.file_id,
                "filename": "paragraph.txt",
                "language": source.chunk.metadata["document_language_info"]["language"],
                "partial": False,
                "quarantined_pages": [],
            }
        ]
        if selected
        else []
    )
    return {
        "partial": False,
        "languages": [d["language"] for d in documents],
        "documents": documents,
    }


def assemble_sequence(question, source, raw_answer, batch_start, counts):
    from docmind.agent.reader import CitedReader

    store = SimpleNamespace(document_context=lambda ids: supplied_context(source, ids))
    session = SimpleNamespace(calls=1, usage=lambda: counts)
    return CitedReader(SimpleNamespace(store=store)).assemble(
        question,
        [source],
        raw_answer,
        batch_start,
        session,
        file_ids=[source.chunk.file_id],
    )


def prepare(source, prior_root, out):
    if out.exists():
        raise FileExistsError("Keep frozen inputs")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("Source changed")
    decision = json.loads((prior_root / "cohort_decision.json").read_bytes())
    if decision["cohort_size"] != 0:
        raise ValueError("The reused final pool must have remained unused")
    previous = json.loads((prior_root / "input_manifest.json").read_bytes())
    if (
        len(previous["final_pool"]) != 200
        or sum(r["answerable"] for r in previous["final_pool"]) != 100
    ):
        raise ValueError("Reuse the full balanced unused pool")
    old_ids = tracked_ids()
    old_ids.update(r["id"] for key in ("pilot", "final_pool") for r in previous[key])
    lookup, excluded = {}, set()
    for article in json.loads(raw)["data"]:
        for paragraph in article["paragraphs"]:
            identity = sha(paragraph["context"])
            if any(q["id"] in old_ids for q in paragraph["qas"]):
                excluded.add(identity)
            for q in paragraph["qas"]:
                lookup[q["id"]] = {
                    "id": q["id"],
                    "title": article["title"],
                    "context": paragraph["context"],
                    "context_sha256": identity,
                    "question": q["question"],
                    "answerable": not q["is_impossible"],
                    "answers": q["answers"],
                }
    candidates = sorted(
        lookup.values(), key=lambda r: sha("english-batch-pilot-v1:" + r["id"])
    )
    seen, groups = set(excluded), {}
    for answerable in (False, True):
        chosen = []
        for row in candidates:
            if row["answerable"] == answerable and row["context_sha256"] not in seen:
                chosen.append(row)
                seen.add(row["context_sha256"])
                if len(chosen) == 6:
                    break
        if len(chosen) != 6:
            raise ValueError("Insufficient fresh pilot paragraphs")
        groups[answerable] = chosen
    pilot = [r for pair in zip(groups[False], groups[True]) for r in pair]
    final = []
    for expected in previous["final_pool"]:
        row = lookup[expected["id"]]
        if {k: row[k] for k in ("id", "context_sha256", "answerable")} != expected:
            raise ValueError("Unused final pool differs")
        final.append(row)
    if {r["context_sha256"] for r in pilot} & {r["context_sha256"] for r in final}:
        raise ValueError("Pilot and final overlap")
    out.mkdir(parents=True)
    save(out / "private/pool.json", {"pilot": pilot, "final": final})
    manifest = {
        "source_sha256": SOURCE_SHA,
        "prior_zero_cohort_decision_sha256": hashlib.sha256(
            (prior_root / "cohort_decision.json").read_bytes()
        ).hexdigest(),
        "private_pool_sha256": hashlib.sha256(
            (out / "private/pool.json").read_bytes()
        ).hexdigest(),
        "pilot": [
            {k: r[k] for k in ("id", "context_sha256", "answerable")} for r in pilot
        ],
        "final_pool": previous["final_pool"],
        "final_pool_previously_generated": False,
        "selection_uses_outcomes": False,
        "excluded_pilot_context_count": len(excluded),
    }
    save(out / "input_manifest.json", manifest)
    return manifest


def through_eos(tokens, eos_ids):
    for index, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[: index + 1]
    return tokens


def choose(timing, manifest):
    batches = timing["batches"]
    complete = sum(
        b["requested_sequences"]
        for b in batches
        if b["generation_completed"] and b.get("batch_completed", True)
    )
    if complete < 8:
        size, reason, cost = (
            0,
            "Fewer than eight native generations in finished batches",
            None,
        )
    else:
        costs = [b["elapsed_seconds"] / b["requested_sequences"] for b in batches]
        if not costs or not all(math.isfinite(v) and v > 0 for v in costs):
            raise ValueError("Invalid attempted batch timing")
        cost = max(max(costs), statistics.median(costs)) * 1.25
        remaining = max(0, 18000 - 2 * timing["setup_seconds"])
        supported = min(200, 4 * math.floor(0.8 * remaining / cost / 4))
        size = supported if supported >= 40 else 0
        reason = "Timing-only balanced prefix" if size else "Fewer than forty cases fit"
    selected = manifest["final_pool"][:size]
    return {
        "cohort_size": size,
        "rows": selected,
        "reason": reason,
        "conservative_seconds_per_question": cost,
        "final_wall_cap_seconds": 18000,
        "pilot_reservation_seconds": 1800,
        "prior_fp16_reservation_seconds": 1800,
        "cumulative_cpu_cap_seconds": 21600,
        "reserve_fraction": 0.2,
        "input_manifest_content_sha256": content_hash(manifest),
        "cohort_content_sha256": content_hash(selected),
        "selection_uses_outcomes": False,
    }


def checked_decision(decision, manifest):
    size = decision["cohort_size"]
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or not 40 <= size <= 200
        or size % 4
    ):
        raise ValueError("Freeze a viable complete-batch cohort")
    selected = manifest["final_pool"][:size]
    if len(selected) != size or sum(r["answerable"] for r in selected) != size // 2:
        raise ValueError("Cohort must be available and balanced")
    if (
        decision["rows"] != selected
        or decision["cohort_content_sha256"] != content_hash(selected)
        or decision["input_manifest_content_sha256"] != content_hash(manifest)
    ):
        raise ValueError("Frozen cohort changed")
    return size


def run(root, phase, deadline, design_path, decision=None):
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
    manifest = json.loads((root / "input_manifest.json").read_bytes())
    raw = (root / "private/pool.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["private_pool_sha256"]:
        raise ValueError("Frozen payload changed")
    examples = json.loads(raw)[phase]
    if phase == "final":
        examples = examples[: checked_decision(decision, manifest)]
    out = root / phase
    out.mkdir(exist_ok=False)
    records = [
        {"id": r["id"], "context_sha256": r["context_sha256"], "status": "unserved"}
        for r in examples
    ]
    save(out / "private/records.json", records)
    started = time.time()
    timing = {"batches": [], "setup_seconds": None}
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LogitsProcessor,
        LogitsProcessorList,
        StoppingCriteria,
        StoppingCriteriaList,
    )
    from docmind.agent.reader import SYSTEM
    from docmind.evidence import untrusted_json
    from docmind.language import language_info
    from docmind.llm.local_reader import MODEL, REVISION
    from docmind.models import Chunk, SearchResult

    design = json.loads(design_path.read_bytes())
    if sha(SYSTEM) != design["system_prompt_sha256"]:
        raise ValueError("Reader prompt changed")
    if torch.cuda.is_available():
        raise ValueError("CPU-only allocation required")
    torch.set_num_threads(4)
    torch.manual_seed(7)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=False, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        trust_remote_code=False,
        dtype=torch.float32,
        device_map={"": "cpu"},
        attn_implementation="sdpa",
    ).eval()
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("FP32 model parameters required")
    save(
        out / "reader_identity.json",
        {
            "model": MODEL,
            "revision": REVISION,
            "dtype": "float32",
            "device": "cpu",
            "batch_size": 4,
            "cpu_threads": 4,
            "seed": 7,
            "attention": model.config._attn_implementation,
            "torch_version": torch.__version__,
            "python": platform.python_version(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("transformers", "tokenizers", "numpy")
            },
            "parameter_bytes": sum(
                p.numel() * p.element_size() for p in model.parameters()
            ),
            "system_prompt_sha256": sha(SYSTEM),
            "gpu_seconds": 0,
            "max_input_tokens": 8192,
            "max_new_tokens": 512,
            "generation_calls_unit": "native batch, up to four independent sequences",
        },
    )
    timing["setup_seconds"] = time.time() - started

    class Finite(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Nonfinite batch logits")
            return scores

    class BeforeDeadline(StoppingCriteria):
        def __init__(self, end):
            self.end = end

        def __call__(self, input_ids, scores, **kwargs):
            if time.monotonic() >= self.end:
                raise TimeoutError("Native batch deadline exceeded")
            return False

    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    for offset in range(0, len(examples), 4):
        if time.time() >= deadline - 5:
            break
        group = examples[offset : offset + 4]
        batch_records = records[offset : offset + 4]
        batch_start = time.monotonic()
        start_unix = time.time()
        generated = False
        for record in batch_records:
            record.update(
                status="in_progress", started_unix=start_unix, batch_index=offset // 4
            )
        save(out / "private/records.json", records)
        batch = {
            "batch_index": offset // 4,
            "requested_sequences": len(group),
            "generation_completed": False,
        }
        try:
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM},
                        {
                            "role": "user",
                            "content": untrusted_json(
                                {
                                    "question": r["question"],
                                    "untrusted_evidence": [
                                        {
                                            "chunk_id": r["context_sha256"],
                                            "text": r["context"],
                                        }
                                    ],
                                }
                            ),
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for r in group
            ]
            encoded = tokenizer(prompts, padding=True, return_tensors="pt")
            length = encoded.input_ids.shape[-1]
            if length > 8192:
                raise ValueError("No silent input truncation")
            end = time.monotonic() + min(600, max(1, deadline - time.time() - 5))
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=512,
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([Finite()]),
                    stopping_criteria=StoppingCriteriaList([BeforeDeadline(end)]),
                )
            generated = True
            # Persist all raw completions before any potentially slow policy work.
            decoded = [
                through_eos(output[i, length:].tolist(), eos) for i in range(len(group))
            ]
            for i, record in enumerate(batch_records):
                record.update(
                    raw_output=tokenizer.decode(decoded[i], skip_special_tokens=True),
                    call_records=[
                        {
                            "input_tokens": int(encoded.attention_mask[i].sum()),
                            "output_tokens": len(decoded[i]),
                            "status": "completed",
                            "generation_attempted": True,
                            "max_length_reached": len(decoded[i]) >= 512,
                            "numerical_failure": False,
                            "timing_unit": "shared native batch",
                        }
                    ],
                )
            save(out / "private/records.json", records)
            for example, record in zip(group, batch_records):
                raw_answer = record["raw_output"]
                counts = record["call_records"][0]
                try:
                    hint = language_info(example["context"])
                    source = SearchResult(
                        Chunk(
                            example["context"],
                            "public-paragraph",
                            "paragraph.txt",
                            0,
                            chunk_id=example["context_sha256"],
                            file_id=example["context_sha256"][:16],
                            metadata={
                                "page": 1,
                                "language_info": hint,
                                "document_language_info": hint,
                            },
                        ),
                        1.0,
                        "supplied_paragraph",
                    )
                    result = assemble_sequence(
                        example["question"], source, raw_answer, batch_start, counts
                    )
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
                    record.update(status="assembly_error", error=type(exc).__name__)
        except Exception as exc:
            for record in batch_records:
                if record["status"] == "in_progress":
                    record.update(
                        status="batch_error", error=type(exc).__name__, raw_output=None
                    )
            batch["error"] = type(exc).__name__
        batch.update(
            generation_completed=generated,
            batch_completed=True,
            elapsed_seconds=time.monotonic() - batch_start,
        )
        timing["batches"].append(batch)
        save(out / "private/records.json", records)
        save(out / "timing.json", timing)
    timing["worker_elapsed_seconds"] = time.time() - started
    save(out / "timing.json", timing)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--source", type=Path, required=True)
    q.add_argument("--prior-root", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("run")
    q.add_argument("--root", type=Path, required=True)
    q.add_argument("--phase", choices=("pilot", "final"), required=True)
    q.add_argument("--deadline-unix", type=float, required=True)
    q.add_argument("--design", type=Path, required=True)
    q.add_argument("--decision", type=Path)
    q = sub.add_parser("choose")
    q.add_argument("--timing", type=Path, required=True)
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.prior_root, args.out)
    elif args.command == "run":
        run(
            args.root,
            args.phase,
            args.deadline_unix,
            args.design,
            json.loads(args.decision.read_bytes()) if args.decision else None,
        )
    else:
        result = choose(
            json.loads(args.timing.read_bytes()), json.loads(args.manifest.read_bytes())
        )
        result["pilot_timing_sha256"] = hashlib.sha256(
            args.timing.read_bytes()
        ).hexdigest()
        with args.out.open("xb") as handle:
            handle.write((json.dumps(result, indent=2) + "\n").encode())


if __name__ == "__main__":
    main()
