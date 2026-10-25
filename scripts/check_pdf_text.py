"""
Check how well a PDF's text extracts, and pick the word-gap tolerance that fixes glued words.

  python scripts/check_pdf_text.py data/corpus/paper.pdf

Some PDFs are tightly kerned, so the default extraction produces 'WeevaluateAgentS4onOSWorldV2'. That
breaks keyword search and makes the answer spans in a testset unreadable. This script extracts a sample of
pages at several tolerances, scores each (very long tokens suggest glued words, lone letters suggest
over-splitting), and tells you which DOCMIND_PDF_X_TOLERANCE to use.

Decide this BEFORE building a testset and BEFORE ingesting. If you change it later, delete
indexes/docmind.db and ingest again, otherwise stored text and testset spans will not match.
"""
import argparse
import re
import sys
from pathlib import Path
from statistics import mean

TOLERANCES = [3.0, 2.0, 1.5, 1.0, 0.7]


def text_metrics(text: str) -> dict:
    toks = text.split()
    if not toks:
        return {"tokens": 0, "long_ratio": 0.0, "single_ratio": 0.0, "avg_len": 0.0, "score": 1.0}
    long_tokens = [t for t in toks if len(re.sub(r"[^A-Za-z]", "", t)) >= 20 and "/" not in t and "@" not in t]
    singles = [t for t in toks if len(t) == 1 and t.isalpha() and t not in ("a", "A", "I")]
    lr, sr = len(long_tokens) / len(toks), len(singles) / len(toks)
    return {"tokens": len(toks), "long_ratio": lr, "single_ratio": sr,
            "avg_len": mean(len(t) for t in toks), "score": lr + sr}


def sample_pages(n_pages: int, k: int = 6) -> list:
    if n_pages <= k:
        return list(range(n_pages))
    step = n_pages / (k + 1)
    return sorted({int(step * (i + 1)) for i in range(k)})


def evaluate(pdf_path: str, pages=None, tolerances=TOLERANCES) -> dict:
    import pdfplumber
    rows, previews = [], {}
    with pdfplumber.open(pdf_path) as pdf:
        idx = pages if pages is not None else sample_pages(len(pdf.pages))
        for tol in tolerances:
            text = "\n".join((pdf.pages[i].extract_text(x_tolerance=tol) or "") for i in idx)
            m = text_metrics(text)
            m["tolerance"] = tol
            rows.append(m)
            previews[tol] = text[:400].replace("\n", " ")
        n_pages = len(pdf.pages)
    usable = [r for r in rows if r["tokens"] > 0]
    best = min(usable, key=lambda r: (round(r["score"], 4), -r["tolerance"])) if usable else None
    return {"pages_total": n_pages, "pages_sampled": [i + 1 for i in idx], "rows": rows,
            "best": best["tolerance"] if best else None, "previews": previews}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--pages", help="comma-separated 1-based page numbers (default: an even sample)")
    args = ap.parse_args()
    if not Path(args.pdf).exists():
        print(f"File not found: {args.pdf}")
        return 1
    pages = [int(p) - 1 for p in args.pages.split(",")] if args.pages else None
    res = evaluate(args.pdf, pages)
    print(f"{args.pdf}: {res['pages_total']} pages, sampled pages {res['pages_sampled']}\n")
    print("tolerance | tokens | long-token share | lone-letter share | score (lower is better)")
    for r in res["rows"]:
        mark = "  <-- best" if r["tolerance"] == res["best"] else ""
        print(f"{r['tolerance']:>9} | {r['tokens']:>6} | {r['long_ratio']:>16.4f} | {r['single_ratio']:>17.4f} | {r['score']:.4f}{mark}")
    if res["best"] is None:
        print("\nNo extractable text on the sampled pages (scanned PDF?). OCR is required.")
        return 1
    print(f"\nSample at the best tolerance ({res['best']}):\n  {res['previews'][res['best']]}\n")
    if res["best"] == 3.0:
        print("The default (3.0) is fine. Nothing to change.")
    else:
        print(f"Use it:   export DOCMIND_PDF_X_TOLERANCE={res['best']}")
        print("Then delete indexes/docmind.db* and ingest again. Read the sample above to confirm the words are separated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
