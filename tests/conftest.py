"""Offline test setup: no model downloads, no API keys, isolated database per test."""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="docmind_test_")
os.environ["DOCMIND_TEST_MODE"] = "1"
os.environ["DOCMIND_HOME"] = _TMP
os.environ.pop("GROQ_API_KEY", None)

import pytest  # noqa: E402

from docmind.config import CONFIG  # noqa: E402
from docmind import embeddings  # noqa: E402


class ScriptedLLM:
    """Returns pre-scripted replies in order; records every call."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, stop=None, **kwargs):
        self.calls.append({"messages": [dict(m) for m in messages], "stop": stop})
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        return self.replies.pop(0)


@pytest.fixture(autouse=True)
def isolated(tmp_path):
    CONFIG.db_path = tmp_path / "test.db"
    CONFIG.data_dir = tmp_path / "data"
    CONFIG.reports_dir = tmp_path / "reports"
    CONFIG.ensure_dirs()
    CONFIG.retriever.min_rerank_score = None
    CONFIG.review.min_top_score = None
    CONFIG.review.audit_rate = 0.0
    CONFIG.review.enabled = True
    CONFIG.max_agent_iterations = 5
    embeddings.reset_embedding_model()
    yield


@pytest.fixture
def make_file(tmp_path):
    def _make(name: str, text: str):
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        return p
    return _make


@pytest.fixture
def scripted():
    return ScriptedLLM


ALPHA = (
    "The Alpha Protocol governs thermal regulation in the reactor. "
    "Coolant temperature must stay below 300 degrees during startup. "
    "Operators verify the coolant pump every shift. "
    "Emergency shutdown is triggered when the pressure exceeds 40 bar. "
    "The shutdown valve closes within two seconds of the trigger."
)
BETA = (
    "The Beta Handbook describes the cafeteria menu rotation. "
    "Soup is served on Mondays and Thursdays in the main hall. "
    "Vegetarian options include lentil curry and mushroom risotto. "
    "Allergen labels are printed on every tray card. "
    "Dessert is limited to one portion per employee."
)
