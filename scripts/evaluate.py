"""
RAGAS evaluation with Groq as judge. Resumable.

  python scripts/evaluate.py --testset data/testset.json --docs data/mydoc.pdf
  python scripts/evaluate.py --testset data/testset.json --skip-generate      # re-score only
  python scripts/evaluate.py --testset data/testset.json --limit 10           # cheap trial

Outputs: reports/generations.jsonl, reports/ragas_scores.jsonl, reports/ragas_latest.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")

from docmind.evaluation.harness import run_evaluation  # noqa: E402
from docmind.ingestion.pipeline import ingest  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="DocMind RAGAS evaluation (Groq judge)")
    ap.add_argument("--testset", required=True)
    ap.add_argument("--docs", nargs="*", default=[], help="files/dirs to ingest first (idempotent)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--metrics", nargs="*", choices=["faithfulness", "answer_relevancy", "context_precision", "context_recall"])
    ap.add_argument("--include-unverified", action="store_true", help="also use items no human has verified (trial runs only)")
    args = ap.parse_args()

    if not Path(args.testset).exists():
        sys.exit(f"Testset not found: {args.testset}")
    file_ids = None
    for d in args.docs:
        rep = ingest(d)
        file_ids = (file_ids or []) + rep.file_ids
    summary = run_evaluation(args.testset, skip_generate=args.skip_generate, skip_score=args.skip_score,
                             limit=args.limit, file_ids=file_ids, metrics=args.metrics,
                             verified_only=not args.include_unverified)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
