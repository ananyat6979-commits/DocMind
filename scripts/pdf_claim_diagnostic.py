"""Score frozen private claim/evidence pairs without loading human judgments."""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()


def validate(payload, raw, design):
    manifest = design["input_manifest"]
    if hashlib.sha256(raw).hexdigest() != manifest["payload_sha256"]:
        raise ValueError("Frozen input hash mismatch")
    if (
        set(payload) != {"schema", "source_packet_sha256", "rows"}
        or payload["schema"] != "pdf-claim-inputs-v1"
    ):
        raise ValueError("Unexpected inference payload fields")
    if payload["source_packet_sha256"] != manifest["source_packet_sha256"]:
        raise ValueError("Source packet mismatch")
    seen = set()
    units = []
    for row in payload["rows"]:
        if set(row) != {"answer_id", "claim_id", "claim", "evidence", "input_sha256"}:
            raise ValueError("Inference inputs must exclude labels and routing")
        key = (row["answer_id"], row["claim_id"])
        if key in seen or not all(isinstance(v, str) for v in row.values()):
            raise ValueError("Invalid claim identity or text")
        seen.add(key)
        if (
            digest({k: v for k, v in row.items() if k != "input_sha256"})
            != row["input_sha256"]
        ):
            raise ValueError("Claim input changed")
        units.append({k: row[k] for k in ("answer_id", "claim_id", "input_sha256")})
    if units != manifest["units"] or len(units) != manifest["claims"]:
        raise ValueError("Frozen claim cohort changed")
    if len({r["answer_id"] for r in units}) != manifest["answers"]:
        raise ValueError("Frozen answer cohort changed")


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def score_one(judge, unit):
    start = time.monotonic()
    if not unit["evidence"].strip():
        return {"status": "missing_evidence", "score": None}
    tok = judge.tokenizer
    premise = len(tok.encode(unit["evidence"], add_special_tokens=False))
    hypothesis = len(tok.encode(unit["claim"], add_special_tokens=False))
    pair = premise + hypothesis + tok.num_special_tokens_to_add(pair=True)
    lengths = {
        "premise_tokens": premise,
        "hypothesis_tokens": hypothesis,
        "pair_tokens": pair,
        "premise_clipped": pair > 256,
    }
    if hypothesis > 128:
        return {**lengths, "status": "claim_budget", "score": None}
    scores = judge([(unit["evidence"], unit["claim"])])
    if len(scores) != 1 or not math.isfinite(scores[0]) or not 0 <= scores[0] <= 1:
        raise ValueError("Invalid entailment probability")
    return {
        **lengths,
        "status": "completed",
        "score": scores[0],
        "elapsed_seconds": time.monotonic() - start,
    }


def run(payload, raw, design, out, deadline):
    validate(payload, raw, design)
    out.mkdir(parents=True, exist_ok=False)
    records = [
        {
            **{k: unit[k] for k in ("answer_id", "claim_id", "input_sha256")},
            "model_index": index,
            "status": "unserved",
            "score": None,
        }
        for index in range(len(design["models"]))
        for unit in payload["rows"]
    ]
    checkpoint = out / "numeric_records.json"
    write(checkpoint, records)
    identity = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "design_sha256": digest(design),
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "gpu_seconds": 0,
        "hosted_calls": 0,
        "models": [],
    }
    write(out / "identity.json", identity)
    import torch
    from docmind.attribution import NLIJudge

    if torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires CPU-only allocation")
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    identity["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "tokenizers", "numpy", "sentencepiece")
    }
    count = len(payload["rows"])
    for index, model in enumerate(design["models"]):
        if time.time() >= deadline:
            break
        start = time.monotonic()
        try:
            judge = NLIJudge(
                model["model"],
                revision=model["revision"],
                local_files_only=False,
                device="cpu",
                batch_size=2,
            )
            identity["models"].append(
                {
                    **judge.identity,
                    "model_index": index,
                    "dtype": str(next(judge.model.parameters()).dtype),
                    "load_seconds": time.monotonic() - start,
                }
            )
            write(out / "identity.json", identity)
        except Exception as exc:
            for row in records[index * count : (index + 1) * count]:
                row.update(status="model_load_error", error=type(exc).__name__)
            write(checkpoint, records)
            continue
        for offset, unit in enumerate(payload["rows"]):
            if time.time() >= deadline:
                break
            row = records[index * count + offset]
            try:
                row.update(score_one(judge, unit))
            except Exception as exc:
                row.update(status="scoring_error", error=type(exc).__name__)
            write(checkpoint, records)
        del judge
        import gc

        gc.collect()
    identity["finished_unix"] = time.time()
    identity["deadline_unix"] = deadline
    write(out / "identity.json", identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args()
    raw = args.inputs.read_bytes()
    run(
        json.loads(raw),
        raw,
        json.loads(args.design.read_text()),
        args.out,
        args.deadline_unix,
    )


if __name__ == "__main__":
    main()
