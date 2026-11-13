"""Plot descriptive held-out curves from frozen records; never select thresholds here."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/benchmarks"


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name, label, color in (("selective", "3B token confidence", "#245a81"), ("critic", "7B same-model critique", "#b04b28")):
        report = json.loads((BASE / name / "summary.json").read_text())
        points = [p for p in report["risk_curve"] if p["risk"] is not None]
        axes[0].plot([p["coverage"] for p in points], [p["risk"] for p in points], label=label, color=color)
        axes[1].plot([p["review_minutes"] for p in points], [p["risk"] for p in points], label=label, color=color)
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.set_ylabel("Exact-match error among accepted decisions")
        ax.grid(alpha=.2)
        for target in (.1, .2, .3):
            ax.axhline(target, linestyle=":", color="grey", alpha=.5)
    axes[0].set_xlabel("Automatic decision coverage")
    axes[0].set_xlim(0, 1)
    axes[1].set_xlabel("Review minutes, assuming 60 seconds per case")
    axes[1].set_xlim(0, 200)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Descriptive threshold sweeps on separate 200-question held-out sets\nAll calibrated policies selected zero automatic coverage", fontsize=11)
    fig.savefig(BASE / "coverage-risk.png", dpi=180)
    plt.close(fig)
    result = json.loads((BASE / "hotpot/summary.json").read_text())["results"]
    variants = ["dense", "sparse", "hybrid", "hybrid_rerank_depth5", "hybrid_rerank_depth10"]
    values = [result["BAAI/bge-small-en-v1.5|headers=True|" + mode]["recall5"] for mode in variants]
    means = np.array([v["mean"] for v in values])
    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    ax.errorbar(means, range(len(means)), xerr=np.array([[v["mean"] - v["lo"] for v in values], [v["hi"] - v["mean"] for v in values]]), fmt="o", capsize=4, color="#245a81")
    ax.set_yticks(range(5), ["Dense", "Sparse", "Hybrid", "Rerank depth 5", "Rerank depth 10"])
    ax.invert_yaxis()
    ax.set_xlim(.78, 1)
    ax.set_xlabel("Recall@5 with 95% question-bootstrap interval")
    ax.set_title("HotpotQA: BGE with headers, 200 distractor questions")
    ax.grid(axis="x", alpha=.2)
    fig.savefig(BASE / "retrieval.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
