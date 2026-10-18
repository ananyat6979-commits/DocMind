"""
One-command evidence run for Kaggle / Colab (also works locally).

  python scripts/cloud_run.py --docs data/mydoc.pdf --testset data/testset.json

Does, in order: ingest -> retrieval ablation (free) -> RAGAS with Groq judge (resumable)
-> writes reports/ -> bundles everything into docmind_results.zip.

Exit codes: 0 done | 2 Groq daily cap hit (progress saved, just run the same command again later)
            1 error

The bundle contains reports/ and indexes/docmind.db. Unzip it into the project
folder on a local machine and the UI there reuses the embeddings computed in the cloud (no re-embedding).
"""
import argparse
import json
import logging
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("cloud_run")

from docmind.config import CONFIG  # noqa: E402
from docmind.evaluation.retrieval_eval import evaluate_modes, load_testset, to_markdown, compare_chunkers  # noqa: E402
from docmind.ingestion.pipeline import ingest  # noqa: E402
from docmind.llm.groq_client import RateLimitExhausted  # noqa: E402
from docmind.retrieval.hybrid import HybridRetriever  # noqa: E402


def bundle(out_zip: Path) -> Path:
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(CONFIG.reports_dir.glob("*")):
            if f.is_file():
                z.write(f, f"reports/{f.name}")
        if Path(CONFIG.db_path).exists():
            z.write(CONFIG.db_path, "indexes/docmind.db")
    return out_zip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--limit", type=int, help="score only the first N questions (cheap trial)")
    ap.add_argument("--chunkers", action="store_true", help="also compare fixed vs semantic chunking")
    ap.add_argument("--skip-ragas", action="store_true", help="ablation only (no Groq key needed)")
    ap.add_argument("--include-unverified", action="store_true")
    ap.add_argument("--out", default="docmind_results.zip")
    args = ap.parse_args()

    t0 = time.time()
    verified_only = not args.include_unverified
    testset = load_testset(args.testset, verified_only=verified_only)
    log.info(f"testset: {len(testset)} {'verified ' if verified_only else ''}items "
             f"({sum(1 for t in testset if not t['answerable'])} unanswerable)")
    if not args.skip_ragas and not CONFIG.llm.groq_api_key:
        log.error("GROQ_API_KEY is not set (Kaggle: Add-ons > Secrets; Colab: the key icon). Or use --skip-ragas.")
        return 1

    file_ids = []
    for d in args.docs:
        rep = ingest(d)
        file_ids += rep.file_ids
        log.info(f"ingest {d}: {[(f.filename, f.status, f.n_chunks) for f in rep.files]}")

    CONFIG.reports_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_modes(testset, HybridRetriever(), k=5, file_ids=file_ids)
    md = to_markdown(result)
    if args.chunkers:
        ch = compare_chunkers(testset, args.docs, k=5)
        result["chunkers"] = ch
        md += "\n\nChunking (hybrid+rerank):\n\n| Strategy | chunks | Precision@5 | Hit@5 | MRR |\n|---|---|---|---|---|\n"
        for name, r in ch.items():
            m = r["modes"]["hybrid_rerank"]
            md += f"| {name} | {r['n_chunks']} | {m['precision']['mean']:.3f} | {m['hit']['mean']:.3f} | {m['mrr']['mean']:.3f} |\n"
    (CONFIG.reports_dir / "retrieval_latest.json").write_text(json.dumps(result, indent=2))
    (CONFIG.reports_dir / "retrieval_latest.md").write_text(md)
    print("\n" + md + "\n")

    code = 0
    if not args.skip_ragas:
        from docmind.evaluation.harness import run_evaluation
        try:
            summary = run_evaluation(args.testset, limit=args.limit, file_ids=file_ids, verified_only=verified_only)
            print(json.dumps(summary, indent=2))
        except RateLimitExhausted as exc:
            log.warning(f"{exc}\nProgress is saved in reports/. Run the SAME command again later; it resumes.")
            code = 2

    z = bundle(Path(args.out))
    log.info(f"bundle: {z} ({z.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as exc:       # user-fixable problems: message, not a traceback
        log.error(str(exc))
        sys.exit(1)
