"""Isolated ASGI service load test using offline stand-ins, not a model SLA."""
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    with tempfile.TemporaryDirectory(prefix="docmind_load_") as tmp:
        os.environ["DOCMIND_HOME"]=tmp
        os.environ["DOCMIND_TEST_MODE"]="1"
        os.environ["DOCMIND_API_KEY"]="isolated-load-test-key"
        os.environ["DOCMIND_REQUESTS_PER_MINUTE"]="500"
        os.environ.pop("GROQ_API_KEY",None)
        import numpy as np
        from fastapi.testclient import TestClient
        from docmind.config import CONFIG
        from docmind.api.main import app
        CONFIG.review.audit_rate=0
        with TestClient(app,headers={"X-API-Key":"isolated-load-test-key"}) as client:
            started=time.perf_counter()
            response=client.post("/uploads/jobs",files={"file":("load.txt",b"The coolant temperature limit during startup is 300 degrees. Operators inspect the coolant pump every shift.","text/plain")})
            response.raise_for_status();job=response.json()["job_id"]
            while time.perf_counter()-started<30:
                status=client.get("/ingest/jobs/"+job).json()
                if status["status"] in {"completed","failed"}:break
                time.sleep(.1)
            if status["status"]!="completed":raise RuntimeError("Queued ingestion did not complete")
            ingestion=time.perf_counter()-started
            def ask(index):
                begin=time.perf_counter()
                result=client.post("/query",json={"question":"What is the coolant temperature limit?"})
                return {"latency_ms":(time.perf_counter()-begin)*1000,"status":result.status_code,
                        "request_id_present":bool(result.headers.get("X-Request-ID"))}
            started=time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(ask,range(200)))
            elapsed=time.perf_counter()-started
        report={"scope":"In-process ASGI, hash embeddings and extractive stand-in; no hosted model calls",
            "python":sys.version.split()[0],"queries":200,"concurrency":4,"quota_override_per_minute":500,
            "queued_upload_seconds":ingestion,"wall_seconds":elapsed,"throughput_per_second":200/elapsed,
            "p50_ms":float(np.percentile([r["latency_ms"] for r in rows],50)),
            "p95_ms":float(np.percentile([r["latency_ms"] for r in rows],95)),
            "errors":sum(r["status"]!=200 for r in rows),"all_request_ids_present":all(r["request_id_present"] for r in rows),
            "status_counts":{str(status):sum(r["status"]==status for r in rows) for status in sorted({r["status"] for r in rows})}}
        (ROOT/"reports"/"service_load.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))

if __name__=="__main__":main()
