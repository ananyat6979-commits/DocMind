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


def test_lexical_runner_refuses_to_overwrite_published_measurements(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    output=tmp_path/"published";output.mkdir()
    marker=output/"manifest.json";marker.write_text("original",encoding="utf-8")
    runner=Path(__file__).resolve().parents[1]/"scripts/miracl_lexical.py"
    result=subprocess.run([sys.executable,str(runner),"--inputs",str(tmp_path/"missing"),"--out",str(output)],capture_output=True,text=True)
    assert result.returncode!=0 and "Output already exists" in result.stderr
    assert marker.read_text(encoding="utf-8")=="original" and list(output.iterdir())==[marker]


def test_default_selection_rejects_an_inflated_paired_gain():
    from scripts.verify_miracl_records import verify_selection
    rows=[{"language":"hi","mode":"base","id":"q","recall5":0.2},{"language":"hi","mode":"candidate","id":"q","recall5":0.7}]
    entry={"baseline":"base","treatment":"candidate","n":1,"metrics":{"recall5":{"difference":0.9,"ci95":[0.9,0.9]}}}
    with pytest.raises(ValueError,match="selection interval mismatch"):
        verify_selection(rows,{"contrasts":{"hi_candidate_minus_base":entry}},{"n_per_language":{"hi":1}})
