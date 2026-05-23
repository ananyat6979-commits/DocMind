"""
Evaluation script. Run on Kaggle for large testsets.
Usage: python scripts/evaluate.py --testset data/testset.json
"""
import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")

from docmind.evaluation.harness import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="DocMind RAGAS evaluation")
    parser.add_argument("--testset", required=True, help="Path to testset JSON")
    parser.add_argument("--output", default="evaluation_results.json")
    args = parser.parse_args()

    if not Path(args.testset).exists():
        print(f"Testset not found: {args.testset}")
        print("Format: [{\"question\": \"...\", \"ground_truth\": \"...\"}, ...]")
        return

    print(f"\nRunning RAGAS evaluation on: {args.testset}\n")
    metrics = run_evaluation(args.testset, output_path=args.output)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"  {metric:<25} {value:.4f}")
    print("="*50)
    print(f"\nFull results saved to: {args.output}")
    print("Log these numbers in your README and resume.\n")


if __name__ == "__main__":
    main()
