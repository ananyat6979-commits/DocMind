"""Fetch original public paragraphs and verify every retained human annotation."""
import hashlib
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "data/public/corpus.json").read_text(encoding="utf-8"))
    labels = {row["id"]: row for row in json.loads((ROOT / "data/testset.json").read_text(encoding="utf-8"))}
    with urllib.request.urlopen(manifest["source"]["url"], timeout=120) as response:
        dataset = json.load(response)
    canonical = json.dumps(dataset, sort_keys=True, ensure_ascii=False).encode()
    if hashlib.sha256(canonical).hexdigest() != manifest["source"]["canonical_json_sha256"]:
        raise ValueError("Public source differs from the measured version")
    lookup = {qa["id"]: (paragraph["context"], qa) for article in dataset["data"]
              for paragraph in article["paragraphs"] for qa in paragraph["qas"]}
    out = ROOT / "data/public/documents"
    out.mkdir(parents=True, exist_ok=True)
    for item in manifest["documents"]:
        if not re.fullmatch(r"[a-f0-9]{24}", item["id"]):
            raise ValueError("Invalid public document id")
        content, original = lookup[item["id"]]
        label = labels[item["id"]]
        if hashlib.sha256(content.encode()).hexdigest() != item["context_sha256"]:
            raise ValueError("Paragraph hash mismatch")
        spans = [{"start": a["answer_start"], "end": a["answer_start"] + len(a["text"]), "text": a["text"]}
                 for a in original["answers"]]
        if label["question"] != original["question"] or label["answerable"] == original["is_impossible"] or spans != label["gold_spans"]:
            raise ValueError("Human question or annotation changed")
        if any(content[s["start"]:s["end"]] != s["text"] for s in spans):
            raise ValueError("Answer span does not round-trip to the original paragraph")
        (out / (item["id"] + ".txt")).write_text(content, encoding="utf-8", newline="")
    print(f"Verified and materialized {len(manifest['documents'])} public paragraphs and their original human labels.")


if __name__ == "__main__":
    main()
