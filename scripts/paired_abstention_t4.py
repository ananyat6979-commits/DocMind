"""Run the old and revised prompts on identical untouched supplied passages."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

from scripts.english_reader_batch_cpu import assemble_sequence, save
from scripts.english_reader_cpu import sha


def run(profile, design, deadline, out):
    os.environ.pop("GROQ_API_KEY", None)
    os.environ.update(
        DOCMIND_TEST_MODE="0",
        DOCMIND_ATTRIBUTION="0",
        DOCMIND_RELEASE_MODE="draft",
        DOCMIND_HOME=str(out / "runtime"),
        DOCMIND_SECRETS_FILE=str(out / "no.env"),
    )
    from docmind.agent.reader import SYSTEM
    from docmind.evidence import untrusted_json
    from docmind.language import language_info
    from docmind.llm.local_reader import MODEL, REVISION, LocalReader
    from docmind.models import Chunk, SearchResult
    import torch

    if torch.cuda.device_count() != 2 or any(
        "T4" not in torch.cuda.get_device_name(i) for i in range(2)
    ):
        raise RuntimeError("The frozen study requires exactly two allocated T4 devices")
    prompts = {
        "previous": (profile / "previous-prompt.txt").read_text(encoding="utf-8"),
        "revised": SYSTEM,
    }
    if {k: sha(v) for k, v in prompts.items()} != design["prompt_sha256"]:
        raise ValueError("Prompt identity changed")
    if (MODEL, REVISION) != (design["model"], design["revision"]):
        raise ValueError("Reader identity changed")
    raw = (profile / "cases.private.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != design["private_cases_sha256"]:
        raise ValueError("Frozen cases changed")
    cases = json.loads(raw)
    identities = [
        {k: row[k] for k in ("id", "context_sha256", "answerable")} for row in cases
    ]
    if identities != design["rows"]:
        raise ValueError("Cohort changed")

    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    save(
        out / "identity.json",
        {
            "allocated_devices": [torch.cuda.get_device_name(i) for i in range(2)],
            "torch": torch.__version__,
            "model": MODEL,
            "revision": REVISION,
        },
    )
    # Dispatch pairs round-robin within each stratum. Device and first-arm order are balanced.
    schedule = {
        i: [(index, case) for index, case in enumerate(cases) if (index // 2) % 2 == i]
        for i in range(2)
    }
    checkpoints = {}
    for device, group in schedule.items():
        slots = [
            {
                "id": case["id"],
                "context_sha256": case["context_sha256"],
                "arm": arm,
                "status": "unserved",
            }
            for _, case in group
            for arm in prompts
        ]
        save(out / f"private/device-{device}.json", slots)
        checkpoints[device] = {(r["id"], r["arm"]): r for r in slots}

    def worker(device):
        reader = LocalReader(
            device=f"cuda:{device}", max_input_tokens=8192, max_new_tokens=512
        )
        slots = checkpoints[device]
        for index, case in schedule[device]:
            if time.time() >= deadline - 5:
                return
            hint = language_info(case["context"])
            cid = case["context_sha256"]
            source = SearchResult(
                Chunk(
                    case["context"],
                    "public-paragraph",
                    "paragraph.txt",
                    0,
                    chunk_id=cid,
                    file_id=cid[:16],
                    metadata={
                        "page": 1,
                        "language_info": hint,
                        "document_language_info": hint,
                    },
                ),
                1.0,
                "supplied_paragraph",
            )
            # Alternate first arm within each answerability stratum and device.
            order = (
                ("previous", "revised")
                if (index // 4 + index % 2) % 2 == 0
                else ("revised", "previous")
            )
            for arm in order:
                remaining = deadline - time.time() - 5
                if remaining <= 0:
                    return
                row = slots[case["id"], arm]
                row.update(status="in_progress", started_unix=time.time())
                path = out / f"private/device-{device}.json"
                save(path, list(slots.values()))
                begin = time.monotonic()
                before = len(reader.call_records)
                try:
                    answer = reader.complete(
                        [
                            {"role": "system", "content": prompts[arm]},
                            {
                                "role": "user",
                                "content": untrusted_json(
                                    {
                                        "question": case["question"],
                                        "untrusted_evidence": [
                                            {"chunk_id": cid, "text": case["context"]}
                                        ],
                                    }
                                ),
                            },
                        ],
                        max_tokens=512,
                        deadline=time.monotonic() + min(90, remaining),
                    )
                    row["raw_output"] = answer
                    result = assemble_sequence(
                        case["question"], source, answer, begin, reader.call_records[-1]
                    )
                    row.update(
                        status="completed",
                        short_answer=result.short_answer,
                        abstained=result.abstained,
                        release_state=result.release_state,
                        hold_reasons=result.hold_reasons,
                        finish_reason=result.finish_reason,
                    )
                except Exception as exc:
                    row.update(
                        status="assembly_error"
                        if "raw_output" in row
                        else "generation_error",
                        error=type(exc).__name__,
                    )
                row.update(
                    elapsed_seconds=time.monotonic() - begin,
                    call_records=reader.call_records[before:],
                )
                save(path, list(slots.values()))

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, range(2)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    run(args.profile, json.loads(args.design.read_bytes()), args.deadline, args.out)
