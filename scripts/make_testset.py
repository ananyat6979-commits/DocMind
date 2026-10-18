"""
Build a HUMAN-VERIFIED testset from your own documents.

The LLM drafts question/answer pairs from random chunks (using the larger judge
model), then YOU accept, edit or reject each one in the terminal. Only verified
items are written. Unverified synthetic questions are not a benchmark.

  python scripts/make_testset.py --docs data/mydoc.pdf --n 40 --out data/testset.json
  python scripts/make_testset.py --docs data/mydoc.pdf --unanswerable 10 --out data/testset.json

Cloud workflow (no typing prompts in a notebook): draft there, verify locally.
  python scripts/make_testset.py --docs data/mydoc.pdf --n 40 --unanswerable 10 --draft-only --out data/testset.draft.json
  python scripts/verify_testset.py --in data/testset.draft.json --out data/testset.json

Each saved item has gold_substrings (the exact span that answers the question),
which powers the free retrieval ablation.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docmind.config import CONFIG  # noqa: E402
from docmind.ingestion.pipeline import ingest  # noqa: E402
from docmind.llm.groq_client import GroqLLM  # noqa: E402
from docmind.store import CorpusStore  # noqa: E402

PROMPT = """From the passage below, write ONE question a reader of the whole document might ask, answerable
ONLY from this passage, plus the short answer and the exact verbatim span (max 25 words) that contains the answer.
Return strict JSON: {{"question": "...", "ground_truth": "...", "gold_span": "..."}}

Passage:
{passage}"""

UNANSWERABLE_PROMPT = """Here is a sample of passages from a document:
{passages}

Write ONE plausible question that a reader might ask about this document's topic but that the passages do
NOT answer. Return strict JSON: {{"question": "..."}}"""


def _json(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--unanswerable", type=int, default=8)
    ap.add_argument("--out", default="data/testset.json")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--draft-only", action="store_true", help="no prompts: write unverified drafts for scripts/verify_testset.py")
    args = ap.parse_args()

    store, file_ids = CorpusStore(), []
    for d in args.docs:
        file_ids += ingest(d, store=store).file_ids
    chunks, _ = store.load_corpus(file_ids)
    chunks = [c for c in chunks if len(c.content) > 300]
    random.Random(args.seed).shuffle(chunks)
    llm = GroqLLM(model=CONFIG.eval.judge_model)

    out_path = Path(args.out)
    items = json.loads(out_path.read_text()) if out_path.exists() else []
    next_id = len(items) + 1

    def save():
        out_path.write_text(json.dumps(items, indent=2, ensure_ascii=False))

    if not args.draft_only:
        print("a = accept, e = edit answer, s = skip, q = quit\n")
    for c in chunks[: args.n]:
        d = _json(llm.complete([{"role": "user", "content": PROMPT.format(passage=c.content)}], temperature=0.2))
        if not d.get("question") or d.get("gold_span", "") not in c.content:
            continue
        if args.draft_only:
            items.append({"id": f"q{next_id:03d}", "question": d["question"], "ground_truth": d["ground_truth"],
                          "answerable": True, "gold_substrings": [d["gold_span"]], "file": c.filename(),
                          "context_preview": c.content[:700], "citation": c.citation(), "verified": False})
            next_id += 1
            save()
            continue
        print("-" * 70, f"\n{c.citation()}\nQ: {d['question']}\nA: {d['ground_truth']}\nSPAN: {d['gold_span']}")
        k = input("[a/e/s/q] > ").strip().lower()
        if k == "q":
            break
        if k == "e":
            d["ground_truth"] = input("answer > ").strip() or d["ground_truth"]
            k = "a"
        if k == "a":
            items.append({"id": f"q{next_id:03d}", "question": d["question"], "ground_truth": d["ground_truth"],
                          "answerable": True, "gold_substrings": [d["gold_span"]],
                          "file": c.filename(), "verified": True})
            next_id += 1
            save()

    sample = "\n---\n".join(c.content[:300] for c in chunks[:8])
    for _ in range(args.unanswerable):
        d = _json(llm.complete([{"role": "user", "content": UNANSWERABLE_PROMPT.format(passages=sample)}], temperature=0.8))
        if not d.get("question"):
            continue
        if args.draft_only:
            items.append({"id": f"q{next_id:03d}", "question": d["question"], "ground_truth": "", "answerable": False,
                          "gold_substrings": [], "context_preview": sample[:700], "verified": False})
            next_id += 1
            save()
            continue
        print("-" * 70, f"\nUNANSWERABLE? {d['question']}")
        if input("truly NOT answered by the document? [y/n] > ").strip().lower() == "y":
            items.append({"id": f"q{next_id:03d}", "question": d["question"], "ground_truth": "",
                          "answerable": False, "gold_substrings": [], "verified": True})
            next_id += 1
            save()
    save()
    print(f"\nSaved {len(items)} verified items to {out_path}")


if __name__ == "__main__":
    main()
