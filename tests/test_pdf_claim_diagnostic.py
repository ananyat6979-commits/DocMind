"""Synthetic scoring contracts do not supply research labels."""

import hashlib
import json

import pytest

from scripts.pdf_claim_diagnostic import digest, score_one, validate


def fixture():
    row = {
        "answer_id": "a",
        "claim_id": "c",
        "claim": "Ten seconds.",
        "evidence": "The limit is ten seconds.",
    }
    row["input_sha256"] = digest(row)
    payload = {
        "schema": "pdf-claim-inputs-v1",
        "source_packet_sha256": "source",
        "rows": [row],
    }
    raw = json.dumps(payload).encode()
    design = {
        "input_manifest": {
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
            "source_packet_sha256": "source",
            "claims": 1,
            "answers": 1,
            "units": [{k: row[k] for k in ("answer_id", "claim_id", "input_sha256")}],
        }
    }
    return payload, raw, design


def test_exact_private_input_contract():
    validate(*fixture())


@pytest.mark.parametrize("change", ["label", "text", "duplicate", "omitted"])
def test_changed_inputs_or_label_leakage_rejected(change):
    payload, _, design = fixture()
    if change == "label":
        payload["rows"][0]["human_support"] = "supported"
    elif change == "text":
        payload["rows"][0]["claim"] = "Twenty seconds."
    elif change == "duplicate":
        payload["rows"].append(payload["rows"][0])
    else:
        payload["rows"] = []
    raw = json.dumps(payload).encode()
    design["input_manifest"]["payload_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        validate(payload, raw, design)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def num_special_tokens_to_add(self, pair):
        return 3


class Judge:
    tokenizer = Tokenizer()

    def __init__(self, score):
        self.score = score
        self.called = []

    def __call__(self, pairs):
        self.called.extend(pairs)
        return [self.score]


def test_score_uses_exact_displayed_strings_and_reports_clipping():
    judge = Judge(0.1)
    unit = {"claim": "Ten seconds.", "evidence": "word " * 260}
    result = score_one(judge, unit)
    assert judge.called == [(unit["evidence"], unit["claim"])]
    assert result["status"] == "completed" and result["score"] == 0.1
    assert result["premise_clipped"] is True
    assert "evidence" not in result and "claim" not in result


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_nonfinite_and_invalid_probabilities_fail(score):
    with pytest.raises(ValueError):
        score_one(Judge(score), {"claim": "Ten.", "evidence": "Ten."})


def test_missing_evidence_and_overlong_claim_remain_unscored():
    judge = Judge(0.9)
    missing = score_one(judge, {"claim": "Ten.", "evidence": " "})
    overlong = score_one(judge, {"claim": "word " * 129, "evidence": "Ten."})
    assert missing["status"] == "missing_evidence" and missing["score"] is None
    assert overlong["status"] == "claim_budget" and overlong["score"] is None
    assert not judge.called
