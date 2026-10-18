# Metrics: what is recorded, where, and how

Run `python scripts/record_metrics.py` to consolidate everything into `reports/METRICS.md` and `reports/metrics_latest.json`. A section without evidence says NOT RUN and names the command that produces it. No value is ever estimated or filled in.

| Area | Metrics | Produced by | Evidence file |
|---|---|---|---|
| Stack and environment | real-stack verdict, package versions, CUDA, commit, embedder throughput and dimension, reranker ordering, Groq reachability and latency, end-to-end verified answer | `scripts/stack_check.py` | `reports/stack_check.json` |
| Corpus | documents, chunks, chunk length distribution, chunks under the minimum size | `record_metrics.py` (reads the database) | `reports/metrics_latest.json` |
| Retrieval | precision, hit rate, MRR and nDCG at k for dense, sparse, hybrid and hybrid plus rerank, bootstrap 95% intervals, paired relative improvement, suggested abstention threshold, fixed vs semantic chunking | `scripts/ablation.py` | `reports/retrieval_latest.json` |
| Answer quality | RAGAS faithfulness, answer relevancy, context precision, context recall with 95% intervals | `scripts/evaluate.py` | `reports/ragas_latest.json` |
| Answer behaviour | citation validity, answers with invalid or no citation, correct and false abstention, unsupported answers on unanswerable questions, finish reasons, review rate | `scripts/evaluate.py` | `reports/generations.jsonl` |
| Cost and speed | latency p50 and p95, LLM calls per answer, prompt and completion tokens per answer | `scripts/evaluate.py` | `reports/generations.jsonl` |
| Pilot | adoption, repeat use, helpfulness, abstention and review rates, latency, review queue; hours saved from a stopwatch baseline | the app, `scripts/pilot_report.py` | events table, `pilot/baseline_timing.csv` |
| Offline tests | pass and fail counts (stand-ins only) | `record_metrics.py --run-tests` | `reports/metrics_latest.json` |

## Order of a complete measurement
1. `python scripts/stack_check.py` on a machine that can run the real models (Codespaces or a Kaggle notebook). Requires `GROQ_API_KEY`. If it does not report `REAL STACK VERIFIED`, none of the later numbers describe the real system.
2. `python scripts/make_testset.py` (or the draft and verify workflow), producing at least 50 verified questions, at least 10 of them unanswerable.
3. `python scripts/ablation.py --testset data/testset.json --docs <files> --chunkers`
4. `python scripts/evaluate.py --testset data/testset.json --docs <files>`
5. `python scripts/record_metrics.py --run-tests`, then `python scripts/update_readme.py`, then commit `reports/`.

## Reading the numbers honestly
- With fewer than about 50 questions the confidence intervals are wide. Report the interval, not only the mean.
- A metric computed with the reranker disabled or with `DOCMIND_TEST_MODE=1` is not a result for the real system. `stack_check.py` records which mode was active.
- The offline test suite proves the code paths work. It does not measure model quality.
