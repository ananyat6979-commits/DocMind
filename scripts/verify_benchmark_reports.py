"""Verify published measurements offline. No model, network or service key required."""
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports" / "benchmarks"


def read(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def rows(path):
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")) as stream:
        return [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))) for line in stream if line.strip()]


def close(actual, expected):
    assert math.isclose(actual, expected, abs_tol=1e-10), (actual, expected)


def split_check(folder):
    split = read(folder / "split_ids.json")
    assert len(set(split["calibration"])) == len(split["calibration"]) == 200
    assert len(set(split["heldout"])) == len(split["heldout"]) == 200
    assert not set(split["calibration"]) & set(split["heldout"])
    return split


def verify():
    manifest = read(BASE / "manifest.json")
    for name, digest in manifest["files"].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Artifact changed: {name}"
        if path.suffix == ".json":
            read(path)
    split = split_check(BASE / "hotpot")
    grouped = defaultdict(list)
    for row in rows(BASE / "hotpot" / "records.jsonl.gz"):
        assert row["id"] in split[row["split"]]
        gold = set(row["gold_title_sha256"])
        rel = [title in gold for title in row["ranked_title_sha256"]]
        assert gold and len(set(row["ranked_title_sha256"])) == len(rel)
        computed = {
            "recall2": sum(rel[:2]) / len(gold),
            "recall5": sum(rel[:5]) / len(gold),
            "hit5": float(any(rel[:5])),
            "mrr": next((1 / (i + 1) for i, hit in enumerate(rel[:5]) if hit), 0),
            "ndcg5": sum(hit / math.log2(i + 2) for i, hit in enumerate(rel[:5])) / sum(1 / math.log2(i + 2) for i in range(min(5, len(gold)))),
            "all_support2": float(sum(rel[:2]) == len(gold)),
        }
        for key, value in computed.items():
            close(row[key], value)
        group = f"{row['embedder']}|headers={row['headers']}|{row['mode']}"
        grouped[(row["split"], group)].append(row)
    summary = read(BASE / "hotpot" / "summary.json")["results"]
    for (kind, name), values in grouped.items():
        assert len(values) == len({r["id"] for r in values}) == 200
        if kind == "heldout":
            for metric in ("recall2", "recall5", "hit5", "mrr", "ndcg5", "all_support2"):
                close(sum(r[metric] for r in values) / 200, summary[name][metric]["mean"])
    # Human evidence labels are retained in every attribution denominator.
    judges = rows(BASE / "judges" / "records.jsonl")
    judge_summary = read(BASE / "judges" / "summary.json")["results"]
    for model, result in judge_summary.items():
        claim_sets = {}
        for kind in ("calibration", "heldout"):
            values = [r for r in judges if r["model"] == model and r["split"] == kind]
            assert len(values) == (500 if kind == "calibration" else 1000)
            claim_sets[kind] = {r["claim_id"] for r in values}
            tp = sum(r["supported"] and r["label"] for r in values)
            fp = sum(r["supported"] and not r["label"] for r in values)
            fn = sum(not r["supported"] and r["label"] for r in values)
            expected = result[kind]
            assert (tp, fp, fn) == (expected["tp"], expected["fp"], expected["fn"])
            close(tp / (tp + fp) if tp + fp else 0, expected["precision"])
            close(tp / (tp + fn) if tp + fn else 0, expected["recall"])
            assert all(r["quote_roundtrip"] and r["end"] > r["start"] >= 0 for r in values)
        assert not claim_sets["calibration"] & claim_sets["heldout"]
    for name in ("selective", "critic"):
        folder = BASE / name
        split = split_check(folder)
        summary = read(folder / "summary.json")
        titles = {}
        for kind in ("calibration", "heldout"):
            values = rows(folder / (kind + ".jsonl"))
            assert len(values) == len({r["id"] for r in values}) == 200
            assert {r["id"] for r in values} == set(split[kind])
            assert sum(r["answerable"] for r in values) == 100
            titles[kind] = {r["title"] for r in values}
            for row in values:
                close(float(row["correct"]), row["exact_match"])
                assert 0 <= row["confidence"] <= 1
            if kind == "heldout":
                for metric in ("exact_match", "token_f1"):
                    close(sum(r[metric] for r in values) / 200, summary["baseline"][metric]["mean"])
                policies = summary["policies"] if name == "selective" else summary["critic_policies"]
                for policy in policies.values():
                    threshold = policy["calibration"]["threshold"]
                    selected = [r for r in values if threshold is not None and r["confidence"] >= threshold]
                    report = policy["heldout"]
                    assert len(selected) == report["n_auto"]
                    assert report["n_review"] + report["n_auto"] == 200
                    close(len(selected) / 200, report["decision_coverage"])
                    if selected:
                        close(sum(not r["correct"] for r in selected) / len(selected), report["risk"])
                    else:
                        assert report["risk"] is None
        assert not titles["calibration"] & titles["heldout"]
    assert not set(read(BASE / "selective/split_ids.json")["heldout"]) & set(read(BASE / "critic/split_ids.json")["heldout"])
    testset = read(ROOT / "data/testset.json")
    corpus = read(ROOT / "data/public/corpus.json")
    documents = {r["id"]: r for r in corpus["documents"]}
    assert len(testset) == len(documents) == 200
    assert sum(r["answerable"] for r in testset) == 100
    for row in testset:
        assert row["verified"] and row["context_sha256"] == documents[row["id"]]["context_sha256"]
        assert row["evidence_file_id"] == row["context_sha256"][:16]
        assert bool(row["gold_spans"]) == row["answerable"]
        assert [r["text"] for r in row["gold_spans"]] == row["references"]
        for span in row["gold_spans"]:
            assert span["start"] >= 0 and span["end"] - span["start"] == len(span["text"])
    answers = rows(BASE / "reader_small/answers.jsonl")
    for mode, result in read(BASE / "reader_small/summary.json")["qa"].items():
        values = [r for r in answers if r["mode"] == mode]
        assert len(values) == len({r["id"] for r in values}) == 200
        for metric in ("exact_match", "token_f1"):
            close(sum(r[metric] for r in values) / 200, result[metric]["mean"])
    print(f"Verified {len(manifest['files'])} artifact hashes, public splits, rank metrics, attribution counts and answer denominators.")


if __name__ == "__main__":
    verify()
