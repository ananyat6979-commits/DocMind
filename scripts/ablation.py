"""
Retrieval ablation (free, no LLM): dense vs sparse vs hybrid vs hybrid+rerank,
optionally fixed vs semantic chunking. Produces the numbers that confirm or
refute the README's reranker claim, with confidence intervals.

  python scripts/ablation.py --testset data/testset.json --docs data/mydoc.pdf
  python scripts/ablation.py --testset data/testset.json --docs data/mydoc.pdf --chunkers
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docmind.config import CONFIG  # noqa: E402
from docmind.evaluation.retrieval_eval import compare_chunkers, evaluate_modes, load_testset, to_markdown  # noqa: E402
from docmind.ingestion.pipeline import ingest  # noqa: E402
from docmind.retrieval.hybrid import HybridRetriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True)
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--chunkers", action="store_true", help="also compare fixed vs semantic chunking")
    ap.add_argument("--include-unverified", action="store_true", help="also use items no human has verified (trial runs only)")
    args = ap.parse_args()

    testset = load_testset(args.testset, verified_only=not args.include_unverified)
    file_ids = []
    for d in args.docs:
        file_ids += ingest(d).file_ids
    result = evaluate_modes(testset, HybridRetriever(), k=args.k, file_ids=file_ids)
    md = to_markdown(result)
    if args.chunkers:
        ch = compare_chunkers(testset, args.docs, k=args.k)
        result["chunkers"] = ch
        md += "\n\nChunking (hybrid+rerank):\n\n| Strategy | chunks | Precision@k | Hit@k | MRR |\n|---|---|---|---|---|\n"
        for name, r in ch.items():
            m = r["modes"]["hybrid_rerank"]
            md += f"| {name} | {r['n_chunks']} | {m['precision']['mean']:.3f} | {m['hit']['mean']:.3f} | {m['mrr']['mean']:.3f} |\n"
    out = CONFIG.reports_dir
    (out / "retrieval_latest.json").write_text(json.dumps(result, indent=2))
    (out / "retrieval_latest.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
