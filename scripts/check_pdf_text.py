"""Inspect the production parser's text and extraction-quality flags."""
import argparse
import json
from statistics import mean
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

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
    return list(range(n_pages)) if n_pages <= k else [round((i+1)*(n_pages-1)/(k+1)) for i in range(k)]


def evaluate(path):
    from docmind.ingestion.loader import load_document
    units = load_document(path)
    return {"parser":"pypdfium2", "pages":len(units), "samples":[{
        "page":u.metadata.get("page"),"preview":u.content[:500],"quality":u.metadata.get("extraction_quality"),
        "text_metrics":text_metrics(u.content)} for i,u in enumerate(units) if i in sample_pages(len(units))]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("pdf");args=parser.parse_args()
    if not Path(args.pdf).is_file():
        print("File not found");return 1
    print(json.dumps(evaluate(args.pdf),ensure_ascii=True,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
