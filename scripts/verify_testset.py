"""
Human verification of a drafted testset. Pure Python, no heavy dependencies:
runs fine on a laptop with the plain standard library.

  python scripts/verify_testset.py --in data/testset.draft.json --out data/testset.json

For each unverified draft you see the question, the proposed answer, the exact
span, and the passage it came from. Keys:
  a accept   e edit the answer then accept   s skip   d delete   q save and quit
Unanswerable drafts ask: is this truly NOT answered by the document?  y/n
Only items you accept are marked verified. Re-running resumes where you stopped.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    drafts = json.loads(Path(args.src).read_text(encoding="utf-8"))
    out_path = Path(args.out)
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    rej_path = out_path.with_suffix(".rejected.json")       # remembered so deleted drafts do not come back
    rejected = set(json.loads(rej_path.read_text(encoding="utf-8"))) if rej_path.exists() else set()
    seen_q = {d["question"] for d in done} | rejected

    def save():
        out_path.write_text(json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8")
        rej_path.write_text(json.dumps(sorted(rejected), ensure_ascii=False), encoding="utf-8")

    def ask(prompt):
        try:
            return input(prompt).strip().lower()
        except EOFError:                                      # stdin closed: save progress and stop
            return "q"

    todo = [d for d in drafts if not d.get("verified") and d["question"] not in seen_q]
    print(f"{len(todo)} drafts to review, {len(done)} already verified.\n")
    for i, d in enumerate(todo, 1):
        print("=" * 72)
        print(f"[{i}/{len(todo)}] {d.get('citation', '')}")
        print(f"PASSAGE: {d.get('context_preview', '(none)')[:700]}\n")
        print(f"Q: {d['question']}")
        if d["answerable"]:
            print(f"A: {d['ground_truth']}\nSPAN (must appear verbatim in the doc): {d['gold_substrings'][0]}")
            k = ask("[a]ccept / [e]dit / [s]kip / [d]elete / [q]uit > ")
            if k == "e":
                d["ground_truth"] = input("corrected answer > ").strip() or d["ground_truth"]
                k = "a"
        else:
            ans = ask("Is this truly NOT answered anywhere in the document? [y/n] > ")
            k = "q" if ans == "q" else ("a" if ans == "y" else "d")
        if k == "q":
            break
        if k == "d":
            rejected.add(d["question"])
            save()
        if k == "a":
            d["verified"] = True
            d.pop("context_preview", None)
            done.append(d)
            save()
    save()
    n_u = sum(1 for d in done if not d["answerable"])
    print(f"\nSaved {len(done)} verified items ({len(done) - n_u} answerable, {n_u} unanswerable) to {out_path}")
    if len(done) < 50 or n_u < 10:
        print("Reminder: aim for at least 50 verified items and at least 10 unanswerable for usable confidence intervals.")


if __name__ == "__main__":
    main()
