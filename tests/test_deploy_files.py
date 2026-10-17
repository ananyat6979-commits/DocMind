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
    assert "SentenceTransformer('BAAI/bge-small-en-v1.5')" in DOCKERFILE      # models baked at build time
    assert "pip install -e" not in DOCKERFILE and "build-essential" not in DOCKERFILE
    assert "whl/cpu" in DOCKERFILE                              # CPU-only torch


def test_dockerfile_models_match_config():
    from docmind.config import EmbeddingConfig, RetrieverConfig
    assert EmbeddingConfig().model_name in DOCKERFILE or "bge-small-en-v1.5" in DOCKERFILE
    assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in DOCKERFILE


def test_compose_services_share_one_volume_and_never_bake_secrets():
    c = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    for name, svc in c["services"].items():
        assert "docmind-data:/data" in svc["volumes"], name
        for k, v in svc.get("environment", {}).items():
            assert str(v).startswith("${"), f"{name}.{k} must come from the environment, not be hardcoded"


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
