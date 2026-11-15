"""Loopback HTTP service measurement with isolated state and an explicit demo reader.

Optional real E5/NLI models exercise CPU retrieval and attribution. Neither mode
measures a hosted/local generative reader, multi-tenant scale or production SLA.
"""
import argparse
from collections import Counter
import concurrent.futures
import hashlib
import json
import os
import math
import sqlite3
from pathlib import Path
import platform
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def summarize(rows, requested, elapsed):
    import numpy as np
    completed = [row for row in rows if row.get("status") == 200]
    durations = [row["latency_ms"] for row in completed]
    return {"requested": requested, "recorded": len(rows), "completed": len(completed),
        "failed": sum(row.get("status") not in (None, 200) or bool(row.get("error")) for row in rows),
        "unserved": requested-len(rows)+sum(row.get("unserved", False) for row in rows),
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "elapsed_seconds": elapsed, "completed_per_second": len(completed)/elapsed if elapsed else None,
        "completed_only_p50_ms": float(np.percentile(durations, 50)) if durations else None,
        "completed_only_p95_ms": float(np.percentile(durations, 95)) if durations else None,
        "all_completed_have_request_id": bool(completed) and all(row["request_id_present"] for row in completed)}


def stored_attribution_summary(path):
    """Verify completed NLI from stored response metadata, without publishing text."""
    if not path.exists():
        return {"responses": 0, "completed_nli": 0, "judges": []}
    try:
        with sqlite3.connect("file:"+path.as_posix()+"?mode=ro", uri=True) as connection:
            rows = connection.execute("SELECT response_info FROM reviews").fetchall()
    except sqlite3.OperationalError:
        return {"responses": 0, "completed_nli": 0, "judges": []}
    completed = []
    for row in rows:
        attribution = json.loads(row[0]).get("citation_report", {}).get("attribution") or {}
        fraction = attribution.get("supported_fraction")
        if (not attribution.get("error") and attribution.get("n_claims", 0) > 0
                and isinstance(fraction, (int, float)) and math.isfinite(fraction)
                and isinstance(attribution.get("judge"), dict)):
            completed.append(attribution)
    identities = {json.dumps(a["judge"], sort_keys=True) for a in completed}
    return {"responses": len(rows), "completed_nli": len(completed),
        "judges": [json.loads(identity) for identity in sorted(identities)],
        "scope": "Stored NLI execution metadata only, not human support or correctness"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--real-models", action="store_true")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--deadline", type=float, default=420)
    args = parser.parse_args()
    if not 1 <= args.requests <= 200 or not 1 <= args.concurrency <= 8 or not 10 <= args.deadline <= 600:
        parser.error("Use 1-200 requests, 1-8 workers and a 10-600 second total deadline")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started+args.deadline
    rows = []
    report = {"scope": "Single-process loopback HTTP, synthetic document and demo generation; not a production or generative-reader SLA",
        "real_embedding_and_nli": args.real_models, "provider_calls": 0, "model_downloads": False,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "concurrency": args.concurrency, "requested": args.requests, "deadline_seconds": args.deadline,
        "quota_override_per_minute": 1000,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    server = None
    with tempfile.TemporaryDirectory(prefix="docmind_http_") as tmp:
        env = {k: v for k, v in os.environ.items() if not k.startswith("DOCMIND_") and k != "GROQ_API_KEY"}
        key = secrets.token_hex(24)
        env.update(DOCMIND_HOME=tmp, DOCMIND_TEST_MODE="0" if args.real_models else "1",
            DOCMIND_API_KEY=key, DOCMIND_SECRETS_FILE=str(Path(tmp)/"missing.env"),
            DOCMIND_REQUESTS_PER_MINUTE="1000", DOCMIND_REVIEW_AUDIT_RATE="0",
            DOCMIND_ATTRIBUTION="1" if args.real_models else "0",
            DOCMIND_DEVICE="cpu", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
        data = Path(tmp)/"data"
        data.mkdir()
        fixture = "Coolant must stay below 300 degrees during startup. Operators verify the pump every shift. "*4
        (data/"coolant.txt").write_text(fixture, encoding="utf-8")
        report["fixture_sha256"] = hashlib.sha256(fixture.encode()).hexdigest()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = "http://127.0.0.1:"+str(port)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def request(path, body=None):
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Total measurement deadline reached")
            req = urllib.request.Request(base+path, data=json.dumps(body).encode() if body is not None else None,
                headers={"X-API-Key": key, "Content-Type": "application/json"})
            with opener.open(req, timeout=min(120, remaining)) as response:
                return response.status, json.load(response), bool(response.headers.get("X-Request-ID"))
        command = ("from docmind.config import CONFIG; CONFIG.llm.provider='demo'; "
            "import sys,json,importlib.metadata as md; from pathlib import Path; "
            "packages={d.metadata.get('Name','').lower():d.version for d in md.distributions()}; "
            "identity={'embedding_backend':CONFIG.embedding.backend,'embedding_model':CONFIG.embedding.model_name,"
            "'embedding_revision':CONFIG.embedding.model_revision,'retrieval_mode':CONFIG.retriever.mode,"
            "'versions':{k:packages.get(k) for k in ['torch','transformers','sentence-transformers','fastapi','uvicorn']}}; "
            "Path(sys.argv[1]).write_text(json.dumps(identity),encoding='utf-8'); "
            "import uvicorn; uvicorn.run('docmind.api.main:app',host='127.0.0.1',port="+str(port)+",log_level='warning',proxy_headers=False)")
        with (out/"server.log").open("wb") as log:
            try:
                server = subprocess.Popen([sys.executable, "-c", command, str(out/"runtime.json")], cwd=ROOT, env=env, stdout=log, stderr=log)
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("Local API exited before readiness")
                    try:
                        request("/ready")
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(.1)
                report["server_ready_seconds"] = time.monotonic()-started
                ingest_start = time.monotonic()
                _, job, _ = request("/ingest/jobs", {"path": "coolant.txt", "expected_language": "en"})
                while True:
                    _, status, _ = request("/ingest/jobs/"+job["job_id"])
                    if status["status"] == "completed":
                        break
                    if status["status"] == "failed":
                        raise RuntimeError("Queued ingestion failed")
                    time.sleep(.1)
                report["queued_ingestion_seconds"] = time.monotonic()-ingest_start
                question = {"question": "What is the coolant limit during startup?"}
                cold = time.monotonic()
                request("/query", question)
                report["cold_query_seconds"] = time.monotonic()-cold
                def ask(index):
                    begin = time.monotonic()
                    if begin >= deadline:
                        return {"index": index, "unserved": True}
                    try:
                        code, payload, request_id = request("/query", question)
                        return {"index": index, "status": code, "request_id_present": request_id,
                            "latency_ms": (time.monotonic()-begin)*1000,
                            "review_status": payload["review_status"], "application_latency_ms": payload["latency_ms"]}
                    except Exception as exc:
                        return {"index": index, "status": getattr(exc, "code", None),
                            "latency_ms": (time.monotonic()-begin)*1000, "error": type(exc).__name__}
                load_start = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    rows = list(pool.map(ask, range(args.requests)))
                report["warm_load"] = summarize(rows, args.requests, time.monotonic()-load_start)
            except Exception as exc:
                report["setup_error"] = type(exc).__name__+": "+str(exc)
                report["warm_load"] = summarize(rows, args.requests, time.monotonic()-started)
            finally:
                if server is not None and server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                        report["forced_kill"] = False
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=10)
                        report["forced_kill"] = True
                report["server_exit_code"] = server.returncode if server is not None else None
        if args.real_models:
            report["stored_attribution"] = stored_attribution_summary(Path(tmp)/"indexes"/"docmind.db")
            report["neural_path_verified"] = (
                report["stored_attribution"]["responses"] == args.requests+1
                and report["stored_attribution"]["completed_nli"] == args.requests+1)
    if (out/"runtime.json").exists():
        report["runtime"] = json.loads((out/"runtime.json").read_text(encoding="utf-8"))
    report["total_wall_seconds"] = time.monotonic()-started
    report["shutdown_scope"] = "Harness termination, not a graceful-shutdown benchmark"
    (out/"rows.jsonl").write_bytes("".join(json.dumps(row)+"\n" for row in rows).encode())
    (out/"summary.json").write_bytes(json.dumps(report, indent=2).encode())
    print(json.dumps(report, indent=2), flush=True)
    return 0 if (not report.get("setup_error") and report["warm_load"]["completed"] == args.requests
        and (not args.real_models or report.get("neural_path_verified"))) else 1


if __name__ == "__main__":
    raise SystemExit(main())
