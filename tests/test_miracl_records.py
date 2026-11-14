import pytest
from scripts.verify_miracl_records import verify_row


def test_ndcg_uses_all_gold_and_rejects_forged_score():
    row = {"predicted_ids": ["a"], "recall5": 0.5, "ndcg5": 1 / (1 + 1 / __import__("math").log2(3)), "mrr5": 1.0}
    verify_row(row, {"a": 1, "b": 1})
    row["ndcg5"] = 1.0
    with pytest.raises(ValueError, match="metric mismatch"):
        verify_row(row, {"a": 1, "b": 1})


def test_duplicate_ranking_cannot_inflate_recall():
    with pytest.raises(ValueError, match="duplicate ranking"):
        verify_row({"predicted_ids": ["a", "a"]}, {"a": 1})
