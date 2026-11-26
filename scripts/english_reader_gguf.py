"""Bounded quantized CPU evaluation using the existing cited reader policy."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import tarfile
import time
import urllib.error
import urllib.request

from scripts.english_reader_batch_cpu import assemble_sequence, save
from scripts.english_reader_cpu import content_hash, sha


def download(spec, target, deadline):
    digest = hashlib.sha256()
    size = 0
    with (
        urllib.request.urlopen(spec["url"], timeout=30) as response,
        target.open("xb") as handle,
    ):
        while block := response.read(1024 * 1024):
            size += len(block)
            if size > spec["bytes"] or time.time() >= deadline:
                raise ValueError("Download exceeds frozen size or deadline")
            digest.update(block)
            handle.write(block)
    if size != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
        raise ValueError("Downloaded artifact does not match published identity")


def choose(timing, manifest, design):
    attempts = timing["attempts"]
    completed = [r["elapsed_seconds"] for r in attempts if r["generation_completed"]]
    budget = design["final_reserved_seconds"]
    cost = None
    size = 0
    reason = "Fewer than eight completed pilot sequences"
    if len(completed) >= 8:
        observed = [r["elapsed_seconds"] for r in attempts]
        if not all(math.isfinite(v) and v > 0 for v in observed):
            raise ValueError("Invalid attempt timing")
        cost = max(max(observed), 2 * statistics.median(completed))
        setup = timing["setup_seconds"]
        if not math.isfinite(setup) or setup < 0:
            raise ValueError("Invalid setup timing")
        estimate = min(200, 4 * math.floor(0.8 * max(0, budget - 2 * setup) / cost / 4))
        size = estimate if estimate >= 40 else 0
        reason = (
            "Timing-only balanced prefix" if size else "Fewer than forty questions fit"
        )
    rows = manifest["final_pool"][:size]
    return {
        "cohort_size": size,
        "rows": rows,
        "reason": reason,
        "conservative_seconds_per_question": cost,
        "final_wall_cap_seconds": budget,
        "selection_uses_outcomes": False,
        "input_manifest_content_sha256": content_hash(manifest),
        "cohort_content_sha256": content_hash(rows),
    }


def run(profile, design, phase, deadline, scratch, decision):
    os.environ.pop("GROQ_API_KEY", None)
    os.environ.update(
        DOCMIND_TEST_MODE="0",
        DOCMIND_ATTRIBUTION="0",
        DOCMIND_RELEASE_MODE="draft",
        DOCMIND_HOME=str(scratch / "runtime"),
        DOCMIND_SECRETS_FILE=str(scratch / "absent.env"),
    )
    from docmind.agent.reader import SYSTEM
    from docmind.evidence import untrusted_json
    from docmind.language import language_info
    from docmind.llm.local_reader import MODEL, REVISION
    from docmind.models import Chunk, SearchResult
    from scripts.english_reader_batch_cpu import checked_decision
    from transformers import AutoTokenizer

    started = time.time()
    manifest = json.loads((profile / "input_manifest.json").read_bytes())
    raw = (profile / "private/pool.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["private_pool_sha256"]:
        raise ValueError("Inputs changed")
    prompts = {"current": SYSTEM}
    if phase == "paired":
        prompts = {
            "previous": (profile / "previous-prompt.txt").read_text(encoding="utf-8"),
            "revised": SYSTEM,
        }
        if {arm: sha(prompt) for arm, prompt in prompts.items()} != design[
            "prompt_sha256"
        ]:
            raise ValueError("Paired prompts changed")
    elif sha(SYSTEM) != design["system_prompt_sha256"]:
        raise ValueError("Prompt changed")
    if MODEL != design["base_model"] or REVISION != design["base_revision"]:
        raise ValueError("Base model identity changed")
    examples = json.loads(raw)[phase]
    if phase == "final":
        examples = examples[: checked_decision(decision, manifest)]
    if phase == "paired":
        identities = [
            {k: row[k] for k in ("id", "context_sha256", "answerable")}
            for row in examples
        ]
        if identities != design["rows"]:
            raise ValueError("Paired cohort differs from frozen design")
    schedule = []
    for index, example in enumerate(examples):
        arms = tuple(prompts)
        if phase == "paired" and (index // 2 + index % 2) % 2:
            arms = arms[::-1]
        schedule.extend((example, arm) for arm in arms)
    out = profile / phase
    out.mkdir(exist_ok=False)
    records = [
        {
            "id": r["id"],
            "context_sha256": r["context_sha256"],
            "status": "unserved",
            **({"arm": arm} if phase == "paired" else {}),
        }
        for r, arm in schedule
    ]
    save(out / "private/records.json", records)
    timing = {"setup_seconds": None, "attempts": []}
    save(out / "timing.json", timing)
    archive, model = scratch / "engine.tar.gz", scratch / "model.gguf"
    download(design["engine"], archive, deadline)
    download(design["weights"], model, deadline)
    engine_dir = scratch / "engine"
    with tarfile.open(archive) as bundle:
        bundle.extractall(engine_dir, filter="data")
    (server_path,) = engine_dir.rglob("llama-server")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=False
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, data=None, timeout=2):
        payload = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/{path}",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=timeout) as response:
            return json.load(response)

    with (scratch / "server.private.log").open("wb") as log:
        server = subprocess.Popen(
            [
                str(server_path.resolve()),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--model",
                str(model.resolve()),
                "--threads",
                "4",
                "--threads-batch",
                "4",
                "--gpu-layers",
                "0",
                "--parallel",
                "1",
                "--ctx-size",
                "8704",
                "--no-context-shift",
            ],
            stdout=log,
            stderr=log,
        )
        try:
            while server.poll() is None and time.time() < deadline - 5:
                try:
                    if request("health")["status"] == "ok":
                        break
                except (OSError, urllib.error.URLError):
                    time.sleep(1)
            else:
                raise RuntimeError("CPU reader failed to become ready")
            timing["setup_seconds"] = time.time() - started
            save(out / "timing.json", timing)
            for index, ((example, arm), row) in enumerate(zip(schedule, records)):
                if time.time() >= deadline - 5:
                    break
                begin = time.monotonic()
                row.update(
                    status="in_progress", started_unix=time.time(), batch_index=index
                )
                save(out / "private/records.json", records)
                attempt = {"index": index, "generation_completed": False}
                try:
                    messages = [
                        {"role": "system", "content": prompts[arm]},
                        {
                            "role": "user",
                            "content": untrusted_json(
                                {
                                    "question": example["question"],
                                    "untrusted_evidence": [
                                        {
                                            "chunk_id": example["context_sha256"],
                                            "text": example["context"],
                                        }
                                    ],
                                }
                            ),
                        },
                    ]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    tokens = tokenizer(prompt).input_ids
                    if len(tokens) > 8192:
                        raise ValueError("Prompt exceeds input cap")
                    response = request(
                        "completion",
                        {
                            "prompt": tokens,
                            "temperature": 0,
                            "seed": 7,
                            "n_predict": 512,
                            "return_tokens": True,
                            "cache_prompt": False,
                        },
                        timeout=min(180, max(1, deadline - time.time() - 5)),
                    )
                    if (
                        response["truncated"]
                        or response["tokens_evaluated"] != len(tokens)
                        or response["stop_type"] not in {"eos", "limit"}
                    ):
                        raise ValueError("Backend violated frozen completion contract")
                    counts = {
                        "input_tokens": len(tokens),
                        "output_tokens": len(response["tokens"]),
                        "max_length_reached": response["stop_type"] == "limit",
                    }
                    raw_answer = response["content"]
                    row.update(raw_output=raw_answer, call_records=[counts])
                    attempt["generation_completed"] = True
                    save(out / "private/records.json", records)
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
                        example["question"], source, raw_answer, begin, counts
                    )
                    row.update(
                        status="completed",
                        answer=result.answer,
                        short_answer=result.short_answer,
                        abstained=result.abstained,
                        release_state=result.release_state,
                        hold_reasons=result.hold_reasons,
                        finish_reason=result.finish_reason,
                    )
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    row.update(
                        status="assembly_error"
                        if attempt["generation_completed"]
                        else "generation_error",
                        error=type(exc).__name__,
                    )
                attempt["elapsed_seconds"] = time.monotonic() - begin
                timing["attempts"].append(attempt)
                save(out / "private/records.json", records)
                save(out / "timing.json", timing)
                if row.get("error") == "TimeoutError":
                    break
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pilot", "final", "choose", "paired"))
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--timing", type=Path)
    args = parser.parse_args()
    design = json.loads(args.design.read_bytes())
    if args.phase == "choose":
        raw = args.timing.read_bytes()
        manifest = json.loads((args.profile / "input_manifest.json").read_bytes())
        decision = choose(json.loads(raw), manifest, design)
        decision["pilot_timing_sha256"] = hashlib.sha256(raw).hexdigest()
        with args.decision.open("xb") as handle:
            handle.write((json.dumps(decision, indent=2) + "\n").encode())
    else:
        decision = json.loads(args.decision.read_bytes()) if args.decision else None
        run(args.profile, design, args.phase, args.deadline, args.scratch, decision)


if __name__ == "__main__":
    main()
