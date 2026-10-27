"""
Build a HUMAN-VERIFIED testset from your own documents.

The LLM drafts question/answer pairs from random chunks (using the larger judge
model), then YOU accept, edit or reject each one in the terminal. Only verified
items are written. Unverified synthetic questions are not a benchmark.

  python scripts/make_testset.py --docs data/mydoc.pdf --n 40 --out data/testset.json
  python scripts/make_testset.py --docs data/mydoc.pdf --unanswerable 10 --out data/testset.json

Keys at every prompt: a accept, e edit the answer, s skip, q save and quit. Anything else re-asks.
Draft only the unanswerable part later with --n 0. Use --max-page N to exclude appendices and references.

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

Prefer questions about the document's main ideas, methods, findings or procedures. Do NOT ask about
table cells, example prompts, file names, figure labels or formatting details.

Passage:
{passage}"""

UNANSWERABLE_PROMPT = """Here is a sample of passages from a document:
{passages}

Write ONE plausible question that a reader might ask about this document's topic but that the passages do
NOT answer. Return strict JSON: {{"question": "..."}}"""


def is_prose(text: str) -> bool:
    """True for running text, False for table dumps, reference lists and code-like chunks. Questions drafted
    from those are trivia ('what is the value in the third column'), not a useful benchmark."""
    n = len(text)
    if n < 400:
        return False
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    sentences = len(re.findall(r"[.!?](?:\s|$)", text))
    return letters / n >= 0.78 and digits / n <= 0.05 and sentences >= 3


def _json(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except ValueError:
        return {}


def select_chunks(chunks, include_tables=False, max_page=None):
    """Chunks worth drafting from: running prose only, and (for paged documents) not past max_page."""
    out = []
    for c in chunks:
        page = c.metadata.get("page")
        if max_page and page and page > max_page:
            continue
        if include_tables:
            if len(c.content) > 300:
                out.append(c)
        elif is_prose(c.content):
            out.append(c)
    return out


_FILLER = frozenset("a an and are as at be by do does did for from has have how in is it its of on or that the this to was were "
                    "what which who whom when where why with used use using specific".split())


def _tokens(text):
    """Content words only, so filler such as 'the', 'are' and 'of' cannot hide a rephrased duplicate."""
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _FILLER}


def too_similar(question, seen, threshold=0.6):
    """Near-duplicate check (Jaccard on word sets) so you are never asked the same question three times."""
    a = _tokens(question)
    for other in seen:
        b = _tokens(other)
        if a and b and len(a & b) / len(a | b) >= threshold:
            return True
    return False


class Quit(Exception):
    pass


def ask(prompt, allowed):
    """Ask until the answer is one of `allowed`. Stray text re-asks instead of silently skipping."""
    while True:
        k = input(prompt).strip().lower()
        if k in allowed:
            if k == "q":
                raise Quit()
            return k
        print(f"   Type one of: {' '.join(sorted(allowed))}  (then Enter)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=30, help="how many answerable questions to draft (0 = skip that part)")
    ap.add_argument("--unanswerable", type=int, default=8)
    ap.add_argument("--out", default="data/testset.json")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-page", type=int, help="ignore pages after this one (use it to exclude appendices and references)")
    ap.add_argument("--include-tables", action="store_true", help="also draft from table-like and numeric chunks")
    ap.add_argument("--draft-only", action="store_true", help="no prompts: write unverified drafts for scripts/verify_testset.py")
    args = ap.parse_args()

    store, file_ids = CorpusStore(), []
    for d in args.docs:
        file_ids += ingest(d, store=store).file_ids
    all_chunks, _ = store.load_corpus(file_ids)
    chunks = select_chunks(all_chunks, args.include_tables, args.max_page)
    print(f"{len(chunks)} of {len(all_chunks)} chunks are eligible for drafting"
          + ("" if args.include_tables else " (running prose only; --include-tables widens)")
          + (f", pages up to {args.max_page}" if args.max_page else ""))
    if not chunks:
        sys.exit("No eligible chunks. Try --include-tables or a larger --max-page.")
    if args.n and len(chunks) < args.n:
        print(f"Only {len(chunks)} eligible chunks, so at most that many questions will be drafted. Add more documents for a bigger testset.")
    random.Random(args.seed).shuffle(chunks)
    llm = GroqLLM(model=CONFIG.eval.judge_model, reasoning_effort=CONFIG.eval.judge_reasoning_effort)

    out_path = Path(args.out)
    items = json.loads(out_path.read_text()) if out_path.exists() else []
    next_id = len(items) + 1
    seen_questions = [it["question"] for it in items]

    def save():
        out_path.write_text(json.dumps(items, indent=2, ensure_ascii=False))

    def add(item):
        nonlocal next_id
        items.append({"id": f"q{next_id:03d}", **item})
        seen_questions.append(item["question"])
        next_id += 1
        save()

    try:
        if not args.draft_only:
            print("Keys: a accept | e edit the answer | s skip | q save and quit\n")
        for c in chunks[: args.n]:
            d = _json(llm.complete([{"role": "user", "content": PROMPT.format(passage=c.content)}], temperature=0.2))
            if not d.get("question") or d.get("gold_span", "") not in c.content or too_similar(d["question"], seen_questions):
                continue
            if args.draft_only:
                add({"question": d["question"], "ground_truth": d["ground_truth"], "answerable": True,
                     "gold_substrings": [d["gold_span"]], "file": c.filename(),
                     "context_preview": c.content[:700], "citation": c.citation(), "verified": False})
                continue
            print("-" * 70, f"\n{c.citation()}\nQ: {d['question']}\nA: {d['ground_truth']}\nSPAN: {d['gold_span']}")
            k = ask("[a/e/s/q] > ", {"a", "e", "s", "q"})
            if k == "s":
                continue
            if k == "e":
                d["ground_truth"] = input("answer > ").strip() or d["ground_truth"]
            add({"question": d["question"], "ground_truth": d["ground_truth"], "answerable": True,
                 "gold_substrings": [d["gold_span"]], "file": c.filename(), "verified": True})

        retriever = None
        for i in range(args.unanswerable):
            # a fresh random sample per question, so the drafts differ from each other
            pool = random.Random(args.seed * 1000 + i).sample(chunks, min(6, len(chunks)))
            sample = "\n---\n".join(c.content[:300] for c in pool)
            d = _json(llm.complete([{"role": "user", "content": UNANSWERABLE_PROMPT.format(passages=sample)}], temperature=0.8))
            if not d.get("question") or too_similar(d["question"], seen_questions):
                continue
            if args.draft_only:
                add({"question": d["question"], "ground_truth": "", "answerable": False, "gold_substrings": [],
                     "context_preview": sample[:700], "verified": False})
                continue
            print("-" * 70, f"\nUNANSWERABLE? {d['question']}")
            if retriever is None:
                from docmind.retrieval.hybrid import HybridRetriever
                retriever = HybridRetriever(store)
            print("Closest passages in the whole corpus (read them; the question must NOT be answered here or elsewhere):")
            for r in retriever.search(d["question"], top_k=3, file_ids=file_ids):
                print(f"   [{r.score:6.2f}] {r.chunk.citation()}\n      {r.chunk.content[:280]!r}")
            k = ask("answered nowhere in the document? y = truly unanswerable, n = reject, q = save and quit > ", {"y", "n", "q"})
            if k == "y":
                add({"question": d["question"], "ground_truth": "", "answerable": False, "gold_substrings": [], "verified": True})
    except (Quit, KeyboardInterrupt, EOFError):
        print("\nStopped. Progress so far is saved.")
    finally:
        save()
    n_ans = sum(1 for it in items if it["answerable"] and it.get("verified"))
    n_un = sum(1 for it in items if not it["answerable"] and it.get("verified"))
    print(f"\nSaved {len(items)} items to {out_path} ({n_ans} verified answerable, {n_un} verified unanswerable).")
    print("Resume later with --seed 2 (different chunks); use --n 0 to add only unanswerable questions.")


if __name__ == "__main__":
    main()
