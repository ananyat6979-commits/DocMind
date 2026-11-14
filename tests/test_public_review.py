import pytest
from scripts.summarize_public_review import summarize
from docmind.evaluation.release import digest


def fixture():
    source = [{"premise": "P", "hypothesis": "H", "label": 0}]
    key = {"tasks": [{"review_id": "a", "source_index": 0, "original_label": 0}]}
    export = {"schema": "public-label-review-v1", "dataset": "facebook/xnli", "language": "hi", "independent_review": True, "active_seconds": 10,
              "records": [{"review_id": "a", "input_sha256": digest(["P", "H"]), "label": 1, "finished": True}]}
    return export, key, source


def test_review_disagreement_is_not_population_error():
    result = summarize(*fixture())
    assert result["disagreements"] == 1
    assert result["population_label_error_rate"] is None


def test_review_rejects_tampered_inputs():
    export, key, source = fixture()
    export["records"][0]["input_sha256"] = "changed"
    with pytest.raises(ValueError, match="input mismatch"):
        summarize(export, key, source)


def test_review_rejects_unattested_or_over_budget():
    export, key, source = fixture()
    export["independent_review"] = False
    with pytest.raises(ValueError, match="attestation"):
        summarize(export, key, source)
    export["independent_review"] = True
    export["active_seconds"] = 1501
    with pytest.raises(ValueError, match="budget"):
        summarize(export, key, source)
