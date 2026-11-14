"""Resolve real Compose configuration without launching a daemon or provider."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import pytest


def test_compose_forwards_explicit_retrieval_profile(tmp_path):
    docker=shutil.which("docker")
    if not docker:pytest.skip("Docker CLI unavailable")
    available=subprocess.run([docker,"compose","version"],capture_output=True,timeout=15)
    if available.returncode:pytest.skip("Compose plugin unavailable")
    root=Path(__file__).resolve().parents[1]
    credential=tmp_path/"credentials.json";credential.write_text("{}",encoding="utf-8")
    environment={**os.environ,"GROQ_API_KEY":"","DOCMIND_LLM_MODEL":"","DOCMIND_CREDENTIALS_PATH":str(credential),
                 "DOCMIND_RETRIEVAL_MODE":"hybrid_rerank","DOCMIND_EMBEDDING_MODEL":"BAAI/bge-small-en-v1.5",
                 "DOCMIND_EMBEDDING_REVISION":"5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                 "DOCMIND_USE_RERANKER":"1","DOCMIND_MIN_RERANK_SCORE":"0.7"}
    result=subprocess.run([docker,"compose","--env-file",str(root/".env.example"),"config","--format","json"],cwd=root,env=environment,capture_output=True,text=True,check=True,timeout=30)
    resolved=json.loads(result.stdout)["services"]["api"]["environment"]
    for key in ("DOCMIND_RETRIEVAL_MODE","DOCMIND_EMBEDDING_MODEL","DOCMIND_EMBEDDING_REVISION","DOCMIND_USE_RERANKER","DOCMIND_MIN_RERANK_SCORE"):
        assert resolved[key]==environment[key]
    assert resolved["GROQ_API_KEY"]==""
