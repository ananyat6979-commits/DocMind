"""Resolve real Compose configuration without launching a daemon or provider."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_compose_forwards_explicit_retrieval_profile(tmp_path):
    docker = shutil.which("docker")
    required = os.getenv("DOCMIND_REQUIRE_COMPOSE") == "1"

    def unavailable(reason):
        if required:
            pytest.fail(reason)
        pytest.skip(reason)

    if not docker:
        unavailable("Docker CLI unavailable")
    try:
        available = subprocess.run(
            [docker, "compose", "version"], capture_output=True, timeout=15
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        unavailable("Compose availability probe failed: " + type(exc).__name__)
    if available.returncode:
        unavailable("Compose plugin unavailable")
    root = Path(__file__).resolve().parents[1]
    credential = tmp_path / "credentials.json"
    credential.write_text("{}", encoding="utf-8")
    environment = {
        **os.environ,
        "GROQ_API_KEY": "",
        "DOCMIND_LLM_MODEL": "",
        "DOCMIND_CREDENTIALS_PATH": str(credential),
        "DOCMIND_RETRIEVAL_MODE": "hybrid_rerank",
        "DOCMIND_EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "DOCMIND_EMBEDDING_REVISION": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "DOCMIND_USE_RERANKER": "1",
        "DOCMIND_MIN_RERANK_SCORE": "0.7",
        "DOCMIND_RELEASE_MODE": "strict",
        "DOCMIND_ATTRIBUTION": "1",
        "DOCMIND_ANSWER_STRATEGY": "react",
        "DOCMIND_QUERY_TIMEOUT_SECONDS": "19.5",
    }
    result = subprocess.run(
        [
            docker,
            "compose",
            "--env-file",
            str(root / ".env.example"),
            "config",
            "--format",
            "json",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    resolved = json.loads(result.stdout)["services"]["api"]["environment"]
    for key in (
        "DOCMIND_RETRIEVAL_MODE",
        "DOCMIND_EMBEDDING_MODEL",
        "DOCMIND_EMBEDDING_REVISION",
        "DOCMIND_USE_RERANKER",
        "DOCMIND_MIN_RERANK_SCORE",
        "DOCMIND_RELEASE_MODE",
        "DOCMIND_ATTRIBUTION",
        "DOCMIND_ANSWER_STRATEGY",
        "DOCMIND_QUERY_TIMEOUT_SECONDS",
    ):
        assert resolved[key] == environment[key]
    assert resolved["GROQ_API_KEY"] == ""

    environment.update(
        DOCMIND_DATABASE_URL="postgresql://runtime:not-used@postgres/docmind",
        DOCMIND_DB_ADMIN_PASSWORD="not-used",
        DOCMIND_MIGRATION_DSN="postgresql://admin:not-used@postgres/docmind",
        DOCMIND_DB_PASSWORD="not-used",
        DOCMIND_DB_TENANTS="test",
    )
    persistent = subprocess.run(
        [
            docker,
            "compose",
            "--env-file",
            str(root / ".env.example"),
            "-f",
            "compose.postgres.yml",
            "config",
            "--format",
            "json",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    inherited = json.loads(persistent.stdout)["services"]["api"]["environment"]
    for key in (
        "DOCMIND_RELEASE_MODE",
        "DOCMIND_ATTRIBUTION",
        "DOCMIND_ANSWER_STRATEGY",
        "DOCMIND_QUERY_TIMEOUT_SECONDS",
    ):
        assert inherited[key] == environment[key]


@pytest.mark.parametrize("required", [False, True])
def test_compose_probe_timeout_is_explicit_and_never_a_pass(
    tmp_path, monkeypatch, required
):
    monkeypatch.setattr(shutil, "which", lambda name: "synthetic-docker")
    monkeypatch.setenv("DOCMIND_REQUIRE_COMPOSE", "1" if required else "0")
    calls = []

    def timeout(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    outcome = pytest.fail.Exception if required else pytest.skip.Exception
    with pytest.raises(outcome, match="Compose availability probe failed"):
        test_compose_forwards_explicit_retrieval_profile(tmp_path)
    assert calls == [["synthetic-docker", "compose", "version"]]


@pytest.mark.parametrize("failure", ["configuration", "timeout"])
def test_working_compose_configuration_failures_are_not_skipped(
    tmp_path, monkeypatch, failure
):
    from types import SimpleNamespace

    monkeypatch.setattr(shutil, "which", lambda name: "synthetic-docker")
    calls = []

    def broken(command, **kwargs):
        calls.append(command)
        if command[-1] == "version":
            return SimpleNamespace(returncode=0)
        if failure == "configuration":
            raise subprocess.CalledProcessError(
                1, command, stderr="Invalid configuration fixture"
            )
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", broken)
    error = (
        subprocess.CalledProcessError
        if failure == "configuration"
        else subprocess.TimeoutExpired
    )
    with pytest.raises(error):
        test_compose_forwards_explicit_retrieval_profile(tmp_path)
    assert len(calls) == 2
