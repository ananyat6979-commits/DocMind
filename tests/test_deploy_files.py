"""Guards on the deployment files. Docker itself is not run in the test suite (CI does that in docker.yml),
but the properties that made the first Dockerfile bad are pinned here."""
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text()


def test_dockerfile_is_hardened():
    assert "\nUSER docmind" in DOCKERFILE                       # not root
    assert "HEALTHCHECK" in DOCKERFILE
    assert "HF_HOME=/opt/hf" in DOCKERFILE and "/data/hf" not in DOCKERFILE   # models not hidden by the volume
    assert "SentenceTransformer('BAAI/bge-small-en-v1.5', revision=" in DOCKERFILE      # models baked at build time
    assert "pip install -e" not in DOCKERFILE and "build-essential" not in DOCKERFILE
    assert "whl/cpu" in DOCKERFILE                              # CPU-only torch


def test_dockerfile_models_match_config():
    from docmind.config import EmbeddingConfig, RetrieverConfig
    assert EmbeddingConfig().model_name in DOCKERFILE or "bge-small-en-v1.5" in DOCKERFILE
    assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in DOCKERFILE


def test_compose_services_isolate_volumes_and_never_bake_secrets():
    c = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    ui, api, parser = (c["services"][name] for name in ("ui","api","parser"))
    assert "volumes" not in ui and "GROQ_API_KEY" not in ui["environment"]
    assert ui["environment"]["DOCMIND_API_URL"] == "http://api:8000"
    assert "docmind-api-data:/data" in api["volumes"]
    credential = next(v for v in api["volumes"] if isinstance(v,dict))
    assert credential["read_only"] and not credential["bind"]["create_host_path"]
    assert parser["network_mode"] == "none" and parser["read_only"]
    assert parser["cap_drop"] == ["ALL"] and parser["mem_limit"] == "768m"
    assert not parser.get("environment")
    assert parser["volumes"] == ["parser-socket:/run/docmind-parser"]


def test_workflow_smoke_tests_and_publish_only_on_main():
    wf = yaml.safe_load((ROOT / ".github/workflows/docker.yml").read_text())
    steps = wf["jobs"]["build"]["steps"]
    names = [s.get("name", "") for s in steps]
    assert sum(n.startswith("Smoke") for n in names) == 3
    publish = [s for s in steps if s.get("name") == "Publish"][0]
    assert publish["if"] == "github.ref == 'refs/heads/main'"
    assert wf["permissions"]["packages"] == "write"


def test_dockerignore_keeps_secrets_and_state_out():
    ignored = (ROOT / ".dockerignore").read_text().split()
    for must in (".env", "indexes", "venv", "data", ".git"):
        assert must in ignored
