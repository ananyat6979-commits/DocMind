"""Content-addressed evaluation manifests. Credentials are never serialized."""
import hashlib
import json
import platform
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
    manifest = {"testset_sha256": hashlib.sha256(Path(testset).read_bytes()).hexdigest(),
                "config": config, "verified_only": verified_only, "file_ids": file_ids,
                "git_commit": revision, "code_sha256": code.hexdigest(),
                "python": platform.python_version(), "runtime": runtime}
    encoded = json.dumps(manifest, sort_keys=True, default=str).encode()
    manifest["run_id"] = hashlib.sha256(encoded).hexdigest()
    return manifest
