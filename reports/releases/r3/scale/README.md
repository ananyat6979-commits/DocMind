# Persistent ANN on real embeddings

The completed profile persisted 50,000 unique public HotpotQA article paragraphs with pinned multilingual E5-small embeddings. This replaces neither the historical Gaussian test nor the smaller MIRACL corpus test; those inputs and conclusions remain distinct. PostgreSQL 17.12 and pgvector 0.8.7 ran on a free Linux runner with four reported CPUs and two Torch threads.

For 100 public questions, unscoped application-path Recall@10 against exact search was **95.2% [92.8, 97.3]**, using a seed-seven 2000-query bootstrap. The partial tenant HNSW index appears in the recorded query plan. The index uses normalized 384-dimensional real embeddings and search depth 128. Scoped application queries use exact filtered search. Index fidelity is not semantic answer quality, and missing exact neighbors remain a real approximation cost.

| Concurrency | Completed / requested | Requests/s | p50 ms | p95 ms |
|---:|---:|---:|---:|---:|
| 1 | 200/200 | 44.27 | 22.48 | 23.95 |
| 4 | 200/200 | 89.71 | 44.39 | 52.29 |
| 8 | 200/200 | 96.83 | 81.53 | 98.81 |
| 16 | 200/200 | 100.26 | 155.03 | 193.78 |

These are database/application vector-search calls with query vectors precomputed. They include fresh store objects and database connections in the concurrency sweep. They exclude query embedding, NLI, generation and HTTP, so 100 requests/s must not be advertised as an end-to-end service rate. Higher concurrency gave diminishing throughput and materially longer latency in this fixed-order profile. The separate real-model HTTP test is much slower and uses a small synthetic corpus.

Embedding took 1339.10 seconds and document/vector ingestion 84.18 seconds; the profile itself took 1465.84 seconds. The chunks relation including its indexes occupied 328785920 bytes, about 313.6 MiB. A real database restart was followed by a fresh process that verified all 50,000 rows and searched a retained vector. Incremental insert/delete behavior and worker recovery have separate live tests. This is not a multi-host, churn-duration or failover benchmark.

The first free CPU attempt timed out during embedding and wrote no corpus rows. Its [failure record](initial_cpu_timeout.json) remains unchanged. The completed retry used the same profile with a 70-minute step cap and an 85-minute job ceiling. Free runner timing varied substantially; the retry is not evidence of an optimization. No additional GPU or paid compute was used.

Source selection is hash-deterministic over the pinned public parquet, with SHA-256 `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`. The selected-paragraph identity is recorded in `summary.json`. [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) declares CC BY-SA 4.0; credit its authors and underlying Wikipedia contributors. Paragraph text and full embedding matrices stay outside Git. Source commit and successful CI artifact provenance are in [provenance](provenance.json).

```sh
python scripts/verify_scale_profile.py reports/releases/r3/scale
```
