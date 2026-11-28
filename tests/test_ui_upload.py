"""Exercise the upload language and extraction feedback through the UI."""

import io
import json
from urllib.parse import urlsplit

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize(
    ("selected_language", "query"),
    [("Auto-detect", ""), ("Hindi", "expected_language=hi")],
)
def test_ui_upload_passes_language_and_shows_extraction_failure(
    monkeypatch, selected_language, query
):
    uploaded_queries = []
    message = "The PDF text failed extraction quality checks."
    action = "Upload a Unicode export from the source document."

    def respond(request, timeout):
        path = urlsplit(request.full_url)
        if path.path == "/whoami":
            result = {"actor_id": "reader", "roles": ["ingestor"]}
        elif path.path == "/documents":
            result = []
        elif path.path == "/uploads/jobs":
            assert request.get_method() == "POST"
            assert b"%PDF-1.4 fixture" in request.data
            uploaded_queries.append(path.query)
            result = {"job_id": "job-1"}
        elif path.path == "/ingest/jobs/job-1":
            result = {
                "status": "failed",
                "error_detail": {"message": message, "action": action},
            }
        else:
            pytest.fail(f"Unexpected route: {path.path}")
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr("urllib.request.urlopen", respond)
    app = AppTest.from_string(
        "from docmind.ui_remote import render\nrender()", default_timeout=15
    )
    app.session_state["personal_access_key"] = "test-key"
    app.run()
    app.sidebar.file_uploader[0].set_value(
        ("sample.pdf", b"%PDF-1.4 fixture", "application/pdf")
    ).run()
    app.sidebar.selectbox[0].set_value(selected_language)
    next(
        button for button in app.sidebar.button if button.label == "Upload document"
    ).click().run()
    next(
        button for button in app.sidebar.button if button.label == "Check upload status"
    ).click().run()

    assert not app.exception
    assert uploaded_queries == [query]
    assert message in [widget.value for widget in app.sidebar.warning]
    assert action in [widget.value for widget in app.sidebar.caption]
