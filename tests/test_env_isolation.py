"""Offline test mode must be hermetic: no .env loading, no live LLM, even if a real key exists."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = """
import dotenv
calls = []
dotenv.load_dotenv = lambda *a, **k: calls.append(1)
from docmind.config import CONFIG
print(len(calls), CONFIG.llm.provider)
"""


def _run(env_extra, remove=()):
    env = dict(os.environ, PYTHONPATH=str(ROOT), DOCMIND_HOME=str(ROOT / ".tmp_env_iso"))
    for k in remove:
        env.pop(k, None)
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True).stdout.split()


def test_test_mode_does_not_read_dotenv_and_ignores_a_real_key():
    assert _run({"DOCMIND_TEST_MODE": "1", "GROQ_API_KEY": "gsk_realistic_key"}) == ["0", "demo"]


def test_normal_mode_still_loads_dotenv():
    assert _run({"GROQ_API_KEY": "x"}, remove=("DOCMIND_TEST_MODE",)) == ["1", "groq"]
