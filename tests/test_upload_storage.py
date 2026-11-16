"""Portable uploads and failure cleanup, with no parser or model execution."""

from io import BytesIO
from types import SimpleNamespace
import pytest
from fastapi import HTTPException
from docmind.api.main import _persist_upload, _upload_name
from docmind.config import CONFIG


@pytest.mark.parametrize(
    "name",
    [
        "x" * 256 + ".txt",
        "क" * 100 + ".txt",
        "con.txt",
        "NUL.pdf",
        "COM1.txt",
        "bad:name.txt",
        "bad.txt.",
    ],
)
def test_nonportable_filenames_fail_before_writing(name):
    with pytest.raises(HTTPException) as error:
        _persist_upload(SimpleNamespace(filename=name, file=BytesIO(b"hello")))
    assert error.value.status_code == 422
    assert not list(CONFIG.data_dir.rglob("*.upload"))


def test_windows_and_posix_paths_keep_only_the_leaf():
    assert _upload_name("C:\\fakepath\\paper.txt") == "paper.txt"
    assert _upload_name("../../paper.txt") == "paper.txt"


def test_failed_promotion_removes_temp_and_hides_os_details(monkeypatch):
    from pathlib import Path

    def fail(*args):
        raise OSError("private server path and storage details")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(HTTPException) as error:
        _persist_upload(SimpleNamespace(filename="paper.txt", file=BytesIO(b"hello")))
    assert error.value.status_code == 503
    assert "private" not in error.value.detail
    assert not list(CONFIG.data_dir.rglob("*.upload"))


def test_read_failure_removes_partial_upload():
    class Broken:
        def read(self, size):
            raise OSError("reader failed")

    with pytest.raises(HTTPException):
        _persist_upload(SimpleNamespace(filename="paper.txt", file=Broken()))
    assert not list(CONFIG.data_dir.rglob("*.upload"))
