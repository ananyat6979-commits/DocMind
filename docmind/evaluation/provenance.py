"""Content-addressed evaluation manifests. Credentials are never serialized."""
import hashlib
import json
import platform
import os
import importlib.metadata
import subprocess
from dataclasses import asdict
from pathlib import Path
from docmind.config import CONFIG


def fingerprint(testset, verified_only=True, file_ids=None, runtime=None):
    config = asdict(CONFIG)
    config["llm"].pop("groq_api_key", None)
    config["review"].pop("reviewer_pin", None)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD"], stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        revision, diff = "unknown", b""
    code = hashlib.sha256(diff)
    root = Path(__file__).resolve().parents[2]
    for path in sorted((root / "docmind").rglob("*.py")):
        code.update(str(path.relative_to(root)).encode())
        code.update(path.read_bytes())
    settings = {name: os.getenv(name, default) for name, default in {
        "DOCMIND_TOOL_PROTOCOL":"react", "DOCMIND_ATTRIBUTION":"0" if os.getenv("DOCMIND_TEST_MODE")=="1" else "1",
        "DOCMIND_NLI_MODEL":"typeform/distilbert-base-uncased-mnli", "DOCMIND_NLI_REVISION":"",
        "DOCMIND_PARSER_MODE":"service", "DOCMIND_DENSE_INDEX":"flat"}.items()}
    versions: dict[str, str | None] = {}
    for name in ("transformers","torch","sentence-transformers","tokenizers","regex","pypdfium2","python-docx"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    manifest = {"settings":settings,"versions":versions,"testset_sha256": hashlib.sha256(Path(testset).read_bytes()).hexdigest(),
                "config": config, "verified_only": verified_only, "file_ids": file_ids,
                "git_commit": revision, "code_sha256": code.hexdigest(),
                "python": platform.python_version(), "runtime": runtime}
    encoded = json.dumps(manifest, sort_keys=True, default=str).encode()
    manifest["run_id"] = hashlib.sha256(encoded).hexdigest()
    return manifest
