"""Public-label support-risk component. No reader generation or private annotation."""

import argparse
import base64
import hashlib
import importlib.metadata
import io
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
import zipfile
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from docmind.evaluation.risk_control import exact_upper


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_bytes((json.dumps(data, indent=2) + "\n").encode())
    temp.replace(path)


def emit_log_bundle(folder):
    """Keep numeric checkpoints in free job logs, without artifact storage."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix in (".json", ".png", ".svg"):
                archive.writestr(path.name, path.read_bytes())
    raw = buffer.getvalue()
    encoded = base64.b64encode(raw).decode()
    parts = [encoded[i : i + 4096] for i in range(0, len(encoded), 4096)]
    print("SUPPORT_BUNDLE_SHA256 " + hashlib.sha256(raw).hexdigest(), flush=True)
    for i, part in enumerate(parts):
        print(f"SUPPORT_BUNDLE_PART {i + 1}/{len(parts)} {part}", flush=True)


def export_job_output(folder):
    output = os.getenv("GITHUB_OUTPUT")
    if not output:
        return
    files = {
        name: json.loads((folder / name).read_text())
        for name in ("numeric_records.json", "identity.json", "timing.json")
        if (folder / name).exists()
    }
    for row in files.get("numeric_records.json", []):
        allowed = {
            "id",
            "source_id",
            "split",
            "task",
            "context_sha256",
            "response_sha256",
            "model_index",
            "unsupported",
            "quality_problem",
            "quality",
            "span_roundtrip_valid",
            "detected_refusal",
            "numeric_baseline_ok",
            "status",
            "eligible",
            "score_top1",
            "score_top4",
            "numeric_ok",
            "pairs",
            "claims",
            "context_windows",
            "premise_clipped_pairs",
            "elapsed_seconds",
            "error",
        }
        if not set(row) <= allowed:
            raise ValueError(
                "Non-numeric response content cannot enter the result channel"
            )
        row.pop("context_sha256", None)
        row.pop("response_sha256", None)
    payload = base64.b64encode(
        zlib.compress(json.dumps(files, separators=(",", ":")).encode(), 9)
    ).decode()
    if len(payload) > 96000:
        raise ValueError("Job-output payload exceeds the frozen transfer bound")
    with open(output, "a", encoding="utf-8") as stream:
        stream.write("payload=" + payload + "\n")


def unpack_job_output(payload, folder):
    folder.mkdir(parents=True, exist_ok=False)
    if not payload:
        return
    if len(payload) > 96000:
        raise ValueError("Unexpected result payload size")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(payload, validate=True), 3_000_001)
    if len(raw) > 3_000_000 or not decoder.eof or decoder.unused_data:
        raise ValueError("Decoded result bound")
    files = json.loads(raw)
    if not set(files) <= {"numeric_records.json", "identity.json", "timing.json"}:
        raise ValueError("Unexpected result files")
    for name, value in files.items():
        write(folder / name, value)


def load_source(design, cache):
    cache.mkdir(parents=True, exist_ok=True)
    for filename, identity in design["source"]["files"].items():
        path = cache / Path(filename).name
        if not path.exists():
            with urllib.request.urlopen(identity["url"], timeout=120) as response:
                raw = response.read(35_000_001)
            if (
                len(raw) > 35_000_000
                or hashlib.sha256(raw).hexdigest() != identity["sha256"]
            ):
                raise ValueError("Source download differs from the frozen identity")
            path.write_bytes(raw)
        if hashlib.sha256(path.read_bytes()).hexdigest() != identity["sha256"]:
            raise ValueError("Cached public source changed")
    sources = {
        r["source_id"]: r
        for r in map(
            json.loads,
            (cache / "source_info.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    responses = {
        r["id"]: r
        for r in map(
            json.loads,
            (cache / "response.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    return sources, responses


def source_context(source):
    data = source["source_info"]
    if source["task_type"] == "QA":
        return data["passages"]
    return (
        data
        if isinstance(data, str)
        else json.dumps(data, sort_keys=True, ensure_ascii=False)
    )


def word_windows(text, size=64, stride=32, maximum=512):
    words = list(re.finditer(r"\S+", text))
    return [
        text[words[i].start() : words[min(i + size, len(words)) - 1].end()]
        for i in range(0, len(words), stride)
    ][:maximum]


def choose_evidence(claim, windows, depth=4):
    terms = set(re.findall(r"[^\W_]+", claim.casefold()))
    ranked = sorted(
        enumerate(windows),
        key=lambda item: (
            -len(terms & set(re.findall(r"[^\W_]+", item[1].casefold()))),
            item[0],
        ),
    )
    return [text for _, text in ranked[:depth]]


def score_case(context, response, judge, design):
    from docmind.claim_checks import numeric_conflicts
    from docmind.text import sentence_spans

    refusal = bool(
        re.match(
            r"(?is)^\s*(?:unable to answer|i (?:cannot|can't) answer|not enough information|insufficient information)",
            response,
        )
    )
    if len(context) > design["context_character_cap"] or not response.strip():
        return {
            "status": "input_bound",
            "eligible": False,
            "score_top1": None,
            "score_top4": None,
            "numeric_ok": False,
            "pairs": 0,
        }
    claims = sentence_spans(response)
    if not claims or len(claims) > design["sentence_cap"]:
        return {
            "status": "claim_cap",
            "eligible": False,
            "score_top1": None,
            "score_top4": None,
            "numeric_ok": False,
            "pairs": 0,
            "claims": len(claims),
        }
    if any(
        len(judge.tokenizer.encode(c.text, add_special_tokens=False))
        > design["hypothesis_token_cap"]
        for c in claims
    ):
        return {
            "status": "hypothesis_cap",
            "eligible": False,
            "score_top1": None,
            "score_top4": None,
            "numeric_ok": False,
            "pairs": 0,
            "claims": len(claims),
        }
    windows = word_windows(
        context,
        design["window_words"],
        design["window_stride"],
        design["max_context_windows"],
    )
    if not windows:
        return {
            "status": "empty_context",
            "eligible": False,
            "score_top1": None,
            "score_top4": None,
            "numeric_ok": False,
            "pairs": 0,
        }
    pairs, lengths, clipped = [], [], 0
    for claim in claims:
        evidence = choose_evidence(claim.text, windows)
        lengths.append(len(evidence))
        for premise in evidence:
            if (
                len(judge.tokenizer.encode(premise, add_special_tokens=False))
                + len(judge.tokenizer.encode(claim.text, add_special_tokens=False))
                + judge.tokenizer.num_special_tokens_to_add(pair=True)
                > 256
            ):
                clipped += 1
            pairs.append((premise, claim.text))
    scores = judge(pairs)
    if len(scores) != len(pairs) or not all(
        math.isfinite(v) and 0 <= v <= 1 for v in scores
    ):
        raise ValueError("Nonfinite or missing verifier outputs")
    first, best, offset = [], [], 0
    for size in lengths:
        first.append(scores[offset])
        best.append(max(scores[offset : offset + size]))
        offset += size
    return {
        "status": "completed",
        "eligible": not refusal,
        "score_top1": min(first),
        "score_top4": min(best),
        "numeric_ok": not any(numeric_conflicts(c.text, [context]) for c in claims),
        "claims": len(claims),
        "context_windows": len(windows),
        "pairs": len(pairs),
        "premise_clipped_pairs": clipped,
        "detected_refusal": refusal,
    }


def score(design_path, cache, out, model_index):
    os.environ.pop("GROQ_API_KEY", None)
    os.environ.update(
        DOCMIND_TEST_MODE="0", DOCMIND_SECRETS_FILE=str(out / "no-provider.env")
    )
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if out.exists():
        raise FileExistsError("Scored attempts are never overwritten or retried")
    out.mkdir(parents=True)
    started = time.monotonic()
    sources, responses = load_source(design, cache)
    import torch

    from docmind.attribution import NLIJudge

    torch.set_num_threads(2)
    model = design["models"][model_index]
    judge = NLIJudge(
        model["model"],
        model["revision"],
        local_files_only=False,
        device="cpu",
        batch_size=8,
    )
    identity = {
        **model,
        "device": "cpu",
        "dtype": str(next(judge.model.parameters()).dtype),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "design_sha256": hashlib.sha256(design_path.read_bytes()).hexdigest(),
        "gpu_seconds": 0,
        "hosted_calls": 0,
        "source_commit": os.getenv("GITHUB_SHA")
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        "implementation_sha256": {
            name: hashlib.sha256(
                (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for name in (
                "scripts/support_risk_component.py",
                "docmind/attribution.py",
                "docmind/claim_checks.py",
                "docmind/text.py",
                "docmind/evaluation/risk_control.py",
            )
        },
    }
    write(out / "identity.json", identity)
    records = []
    for case in design["rows"]:
        response = responses[case["id"]]
        context = source_context(sources[case["source_id"]])
        if (
            hashlib.sha256(context.encode()).hexdigest() != case["context_sha256"]
            or hashlib.sha256(response["response"].encode()).hexdigest()
            != case["response_sha256"]
        ):
            raise ValueError("Frozen case changed")
        row = {
            **case,
            "model_index": model_index,
            "unsupported": bool(response["labels"]),
            "quality_problem": response["quality"] != "good",
            "quality": response["quality"],
            "span_roundtrip_valid": all(
                response["response"][r["start"] : r["end"]] == r["text"]
                for r in response["labels"]
            ),
        }
        from docmind.claim_checks import numeric_conflicts
        from docmind.text import sentence_spans

        row["detected_refusal"] = bool(
            re.match(
                r"(?is)^\s*(?:unable to answer|i (?:cannot|can't) answer|not enough information|insufficient information)",
                response["response"],
            )
        )
        spans = sentence_spans(response["response"])
        row["numeric_baseline_ok"] = (
            bool(spans)
            and len(spans) <= design["sentence_cap"]
            and len(context) <= design["context_character_cap"]
            and not any(numeric_conflicts(span.text, [context]) for span in spans)
        )
        row.pop("context_sha256", None)
        row.pop("response_sha256", None)
        before = time.monotonic()
        if before - started >= design["model_time_cap_seconds"]:
            row.update(
                status="budget_unserved",
                eligible=False,
                score_top1=None,
                score_top4=None,
                numeric_ok=False,
                pairs=0,
            )
        else:
            try:
                row.update(score_case(context, response["response"], judge, design))
            except Exception as exc:
                row.update(
                    status="verifier_failure",
                    error=type(exc).__name__,
                    eligible=False,
                    score_top1=None,
                    score_top4=None,
                    numeric_ok=False,
                    pairs=0,
                )
        row["elapsed_seconds"] = time.monotonic() - before
        records.append(row)
        if len(records) % 10 == 0:
            write(out / "numeric_records.json", records)
            print(model_index, len(records), row["status"], flush=True)
    write(out / "numeric_records.json", records)
    write(
        out / "timing.json",
        {
            "elapsed_seconds_including_setup": time.monotonic() - started,
            "time_cap_seconds": design["model_time_cap_seconds"],
            "implementation": "NLI verification only; no new reader or production-policy change",
            "statuses": dict(Counter(r["status"] for r in records)),
        },
    )

    emit_log_bundle(out)
    export_job_output(out)


def policies(design):
    return [
        {"model_index": m, "depth": k, "numeric_guard": g, "threshold": t}
        for m, k, g, t in itertools.product(
            range(len(design["models"])),
            design["candidate_depths"],
            design["numeric_guard_modes"],
            design["thresholds"],
        )
    ]


def selected(row, policy):
    value = row.get("score_top" + str(policy["depth"]))
    return (
        row.get("eligible", False)
        and row.get("status") == "completed"
        and value is not None
        and value >= policy["threshold"]
        and (not policy["numeric_guard"] or row["numeric_ok"])
    )


def fit(rows, design, stratified=False):
    calibration = [r for r in rows if r["split"] == "calibration"]
    choices = []
    for policy in policies(design):
        group = [r for r in calibration if r["model_index"] == policy["model_index"]]
        accepted = [r for r in group if selected(r, policy)]
        errors = sum(r["unsupported"] for r in accepted)
        bound = exact_upper(
            errors, len(accepted), design["alpha"] / design["policy_family_size"]
        )
        qualifiers = (
            len(accepted) >= design["minimum_released"]
            and bound <= design["risk_target"]
            and errors / len(accepted) <= design.get("risk_point_target", 1.0)
        )
        task_bounds = {}
        if stratified:
            for task in ("QA", "Summary", "Data2txt"):
                sub = [r for r in accepted if r["task"] == task]
                task_bounds[task] = exact_upper(
                    sum(r["unsupported"] for r in sub),
                    len(sub),
                    design["alpha"] / design["task_family_size"],
                )
                qualifiers = (
                    qualifiers
                    and len(sub) >= design["minimum_released"]
                    and task_bounds[task] <= design["risk_target"]
                )
        if qualifiers:
            choices.append(
                {
                    **policy,
                    "calibration_released": len(accepted),
                    "calibration_errors": errors,
                    "calibration_upper": bound,
                    "task_upper": task_bounds,
                }
            )
    if not choices:
        return {
            "policy": None,
            "reason": "No fixed policy meets the corrected calibration bound",
            "stratified": stratified,
        }
    best = max(
        choices,
        key=lambda p: (
            p["calibration_released"],
            -p["calibration_upper"],
            p["threshold"],
            -p["model_index"],
            -p["depth"],
            p["numeric_guard"],
        ),
    )
    return {"policy": best, "stratified": stratified}


def evaluate(rows, policy, design):
    test = [
        r
        for r in rows
        if r["split"] == "heldout"
        and r["model_index"] == (policy["model_index"] if policy else 0)
    ]
    accepted = [r for r in test if policy and selected(r, policy)]
    errors = sum(r["unsupported"] for r in accepted)
    true_supported = sum(not r["unsupported"] for r in accepted)
    supported_total = sum(not r["unsupported"] for r in test)
    return {
        "requested": len(test),
        "completed": sum(r["status"] == "completed" for r in test),
        "released": len(accepted),
        "coverage": len(accepted) / len(test),
        "unsupported_released": errors,
        "support_release_precision": true_supported / len(accepted)
        if len(accepted) >= design["minimum_released"]
        else None,
        "support_release_recall": true_supported / supported_total
        if supported_total
        else None,
        "unsupported_risk": errors / len(accepted)
        if len(accepted) >= design["minimum_released"]
        else None,
        "risk_upper95_nominal": exact_upper(errors, len(accepted))
        if len(accepted) >= design["minimum_released"]
        else 1,
        "quality_problem_released": sum(r["quality_problem"] for r in accepted),
        "risk_evidence": "insufficient evidence"
        if len(accepted) < design["minimum_released"]
        else "Component-only under the stated binomial assumptions",
    }


def baseline_results(rows, design):
    test = [r for r in rows if r["split"] == "heldout" and r["model_index"] == 0]
    rules = {
        "all_responses": lambda r: True,
        "ungated_non_refusal": lambda r: not r["detected_refusal"],
        "numeric_guard_without_nli": lambda r: (
            not r["detected_refusal"] and r["numeric_baseline_ok"]
        ),
    }
    result = {}
    for name, rule in rules.items():
        accepted = [r for r in test if rule(r)]
        errors = sum(r["unsupported"] for r in accepted)
        result[name] = {
            "requested": len(test),
            "released": len(accepted),
            "coverage": len(accepted) / len(test),
            "unsupported_released": errors,
            "unsupported_risk": errors / len(accepted)
            if len(accepted) >= design["minimum_released"]
            else None,
            "risk_upper95_nominal": exact_upper(errors, len(accepted))
            if len(accepted) >= design["minimum_released"]
            else 1,
            "quality_problem_released": sum(r["quality_problem"] for r in accepted),
        }
    return result


def summarize(design_path, inputs, out):
    design = json.loads(design_path.read_text(encoding="utf-8"))
    rows = []
    sources, responses = load_source(design, out.parent / "public-source-cache")
    for index, path in enumerate(inputs):
        data = (
            json.loads((path / "numeric_records.json").read_text())
            if (path / "numeric_records.json").exists()
            else []
        )
        existing = {r["id"]: r for r in data}
        expected = {r["id"]: r for r in design["rows"]}
        if len(existing) != len(data) or not set(existing) <= set(expected):
            raise ValueError("Unexpected or repeated scored identity")
        for row in data:
            case = expected[row["id"]]
            if row["model_index"] != index or any(
                row[key] != case[key] for key in ("source_id", "split", "task")
            ):
                raise ValueError("Scored case identity differs from the frozen design")
        if (path / "identity.json").exists():
            identity = json.loads((path / "identity.json").read_text())
            if (
                identity["design_sha256"]
                != hashlib.sha256(design_path.read_bytes()).hexdigest()
                or identity["revision"] != design["models"][index]["revision"]
            ):
                raise ValueError("Model or design identity differs")

        assert len(existing) == len(data) and set(existing) <= {
            r["id"] for r in design["rows"]
        }
        assert all(r["model_index"] == index for r in data)
        for case in design["rows"]:
            if case["id"] in existing:
                rows.append(existing[case["id"]])
            else:
                response = responses[case["id"]]
                rows.append(
                    {
                        **case,
                        "model_index": index,
                        "unsupported": bool(response["labels"]),
                        "quality_problem": response["quality"] != "good",
                        "status": "unserved_after_interruption",
                        "eligible": False,
                        "score_top1": None,
                        "score_top4": None,
                        "numeric_ok": False,
                        "numeric_baseline_ok": False,
                        "detected_refusal": bool(
                            re.match(r"(?is)^\s*unable to answer", response["response"])
                        ),
                    }
                )
    expected = {r["id"]: r for r in design["rows"]}
    for row in rows:
        case = expected[row["id"]]
        assert all(row[key] == case[key] for key in ("source_id", "split", "task"))
        assert isinstance(row["unsupported"], bool)
    pooled, strict = fit(rows, design), fit(rows, design, stratified=True)
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "calibration_selection.json",
        {
            "pooled": pooled,
            "stratified": strict,
            "selection_uses_heldout_labels": False,
            "design_sha256": hashlib.sha256(design_path.read_bytes()).hexdigest(),
        },
    )
    curves = [
        {"policy": p, "heldout": evaluate(rows, p, design)} for p in policies(design)
    ]
    result = {
        "scope": design["scope"],
        "pooled_calibration_selection": pooled,
        "stratified_calibration_selection": strict,
        "pooled_heldout": evaluate(rows, pooled["policy"], design),
        "stratified_heldout": evaluate(rows, strict["policy"], design),
        "models": design["models"],
        "baselines": baseline_results(rows, design),
        "per_task": {
            task: {
                "pooled": evaluate(
                    [r for r in rows if r["task"] == task], pooled["policy"], design
                ),
                "stratified": evaluate(
                    [r for r in rows if r["task"] == task], strict["policy"], design
                ),
            }
            for task in ("QA", "Summary", "Data2txt")
        },
        "implementation": "NLI verification only; no new reader or production-policy change",
        "statuses": dict(
            Counter((str(r["model_index"]) + ":" + r["status"]) for r in rows)
        ),
        "new_reader_calls": 0,
        "gpu_seconds": 0,
        "hosted_calls": 0,
        "limitations": [
            design["assumptions"],
            "Original public human annotations, not independent labels of current PDF answers",
            "Wrong refusals and truncation are separate utility failures, not automatically unsupported spans",
            "Window selection and premise clipping can miss support; every requested response remains in denominators",
            "No current product release guarantee or domain-transfer claim",
        ],
    }
    write(out / "numeric_records.json", rows)
    write(out / "summary.json", result)
    write(out / "fixed_policy_curves.json", curves)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for index, model in enumerate(design["models"]):
        points = [
            r
            for r in curves
            if r["policy"]["model_index"] == index
            and r["policy"]["depth"] == 4
            and not r["policy"]["numeric_guard"]
            and r["heldout"]["unsupported_risk"] is not None
        ]
        points.sort(key=lambda r: r["heldout"]["coverage"])
        ax.plot(
            [r["heldout"]["coverage"] for r in points],
            [r["heldout"]["unsupported_risk"] for r in points],
            marker="o",
            label=model["id"],
        )
    for label, value in result["baselines"].items():
        if value["unsupported_risk"] is not None:
            ax.scatter(
                value["coverage"],
                value["unsupported_risk"],
                marker="x",
                s=65,
                label=label,
            )
    ax.axhline(
        design["risk_target"],
        color="black",
        linestyle="--",
        label="8% component risk target",
    )
    ax.set(
        xlabel="Held-out response coverage",
        ylabel="Unsupported response fraction",
        title="Public support component, fixed policies",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "support-risk-coverage.png", dpi=180)
    fig.savefig(out / "support-risk-coverage.svg")
    plt.close(fig)
    print(json.dumps(result))
    emit_log_bundle(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["score", "summarize", "unpack"])
    p.add_argument(
        "--design",
        type=Path,
        default=ROOT / "reports/releases/r6/support_risk_design.json",
    )
    p.add_argument("--cache", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-index", type=int, choices=[0, 1])
    p.add_argument("--inputs", type=Path, nargs=2)
    a = p.parse_args()
    if a.action == "unpack":
        unpack_job_output(os.getenv("SUPPORT_PAYLOAD", ""), a.out)
    elif a.action == "score":
        if a.cache is None or a.model_index is None:
            p.error("Scoring requires cache and model index")
        score(a.design, a.cache, a.out, a.model_index)
    else:
        if a.inputs is None:
            p.error("Summary requires both model score directories")
        summarize(a.design, a.inputs, a.out)
