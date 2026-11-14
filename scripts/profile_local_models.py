"""Bounded offline CPU model diagnostic with stages and timeout stack traces.

Outputs are private diagnostics, not generation quality or service latency results.
"""
import argparse
import atexit
import faulthandler
import hashlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_bytes(json.dumps(value, indent=2, ensure_ascii=True).encode("utf-8"))


def child(out):
    sys.path.insert(0, str(ROOT))
    for key in list(os.environ):
        if key.startswith("DOCMIND_") or key == "GROQ_API_KEY":
            os.environ.pop(key, None)
    os.environ.update(DOCMIND_TEST_MODE="0", DOCMIND_HOME=str(out / "runtime"),
        DOCMIND_SECRETS_FILE=str(out / "no-provider.env"), DOCMIND_DEVICE="cpu",
        DOCMIND_TOOL_PROTOCOL="react", HF_HUB_OFFLINE="1", DOCMIND_ATTRIBUTION="1",
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false")
    started = time.monotonic()
    def stage(name, **fields):
        row = {"stage": name, "elapsed_seconds": time.monotonic()-started, **fields}
        with (out / "stages.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    faulthandler.enable()
    faulthandler.dump_traceback_later(30, repeat=True)
    atexit.register(lambda: stage("atexit", threads=[t.name for t in threading.enumerate()]))
    stage("child_started")
    import torch
    torch.set_num_threads(2)
    stage("torch_imported")
    from docmind.config import CONFIG
    from docmind.ingestion.pipeline import ingest
    from docmind.agent.react import ReActAgent
    from docmind.llm.demo import DemoLLM
    CONFIG.review.audit_rate = 0.0
    stage("application_imported")
    fixture = "Coolant must stay below 300 degrees during startup. Operators verify the pump every shift. "*4
    source = out / "coolant.txt"
    source.write_text(fixture, encoding="utf-8")
    ingest(source)
    stage("ingested")
    runner = ReActAgent(llm=DemoLLM())
    stage("query_object_created")
    response = runner.run("What is the coolant limit during startup?")
    stage("query_completed")
    attribution = response.citation_report.get("attribution") or {}
    assert "300" in response.answer and response.citation_report.get("validity") == 1.0
    assert response.sources and all(s.retriever == "dense" for s in response.sources)
    assert attribution.get("n_claims", 0) > 0 and not attribution.get("error")
    assert math.isfinite(attribution["supported_fraction"])
    write(out / "model_checks.json", {
        "scope": "Synthetic offline CPU compatibility observation; no human or provider validation",
        "versions": {key: md.version(key) for key in ("torch", "transformers", "sentence-transformers")},
        "embedding": [CONFIG.embedding.model_name, CONFIG.embedding.model_revision],
        "retrieval_mode": CONFIG.retriever.mode,
        "fixture_sha256": hashlib.sha256(fixture.encode()).hexdigest(),
        "citation_validity": response.citation_report["validity"],
        "n_claims": attribution["n_claims"], "nli_finite": True,
        "provider_calls": 0, "model_downloads": False})
    stage("top_level_completed")


def run_bounded(command, out, timeout):
    """Record success only when the child actually exits with status zero."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be positive and finite")
    started = time.monotonic()
    with (out / "stdout.log").open("wb") as stdout, (out / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        launch_seconds = time.monotonic()-started
        timed_out = False
        try:
            code = process.wait(timeout=max(.001, timeout-launch_seconds))
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            code = process.wait(timeout=15)
    result = {"scope": "Bounded local diagnostic, not an API latency benchmark",
        "timeout_seconds": timeout, "launch_seconds": launch_seconds,
        "wall_seconds": time.monotonic()-started, "exit_code": code,
        "timed_out": timed_out, "process_passed": not timed_out and code == 0}
    write(out / "process.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.child:
        child(out)
        return 0
    out.mkdir(parents=True, exist_ok=False)
    result = run_bounded([sys.executable, "-u", str(Path(__file__).resolve()),
        "--child", "--out", str(out)], out, args.timeout)
    print(json.dumps(result), flush=True)
    return 0 if result["process_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
