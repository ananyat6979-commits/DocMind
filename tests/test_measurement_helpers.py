import importlib.util
import json
from pathlib import Path


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/"scripts"/(name+".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_review_document_content_cannot_end_script():
    helper = module("prepare_pdf_quality_review")
    value = {"filename": "</script><script>alert(1)</script>", "text": "<img onerror=alert(1)>"}
    encoded = helper.safe_json(value)
    assert "<" not in encoded and ">" not in encoded
    assert json.loads(encoded) == value


def test_http_load_keeps_failed_and_unserved_requests():
    helper = module("load_http_service")
    rows = [{"status": 200, "latency_ms": 20, "request_id_present": True},
            {"status": 503, "latency_ms": 30, "error": "HTTPError"},
            {"unserved": True}]
    summary = helper.summarize(rows, requested=4, elapsed=2)
    assert summary["completed"] == 1 and summary["failed"] == 1
    assert summary["unserved"] == 2
    assert summary["completed_only_p50_ms"] == 20
    assert summary["completed_per_second"] == .5
