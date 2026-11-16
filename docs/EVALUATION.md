# Evaluation protocol, results and failure analysis

Measured 13 November 2026. Code and derived records are public; runs used available local compute, private free T4 notebooks and a bounded free hosted API allowance. No paid capacity was used. The frozen records are under `reports/benchmarks`; their manifest is checked in CI by `scripts/verify_benchmark_reports.py`.

This dated report preserves the original experiments. Later runtime behavior, human reviews and deployment checks are tracked in [current core evidence](CORE_RESULTS.md), [quality follow-up](../reports/releases/r5/README.md) and the [public response-support comparison](../reports/releases/r6/README.md). Statements about unmeasured work below describe the original snapshot unless a follow-up is identified.

## Scope and reproducibility

The question is whether additional retrieval machinery, attribution checks and selective review improve a small document QA system under constrained compute. The evidence does not support a new state of the art or a useful calibrated automatic-answer gate. Publishing the failures is the main result.

HotpotQA uses the official validation distractor setting, at most ten supplied paragraphs per question. It does not search all of Wikipedia. SHA256-based sampling fixes 200 calibration and 200 held-out questions. SQuAD2 uses the original human answer spans and unanswerable labels: 200 calibration and 200 held-out questions, each balanced 100/100, with unique paragraph contexts and article-disjoint groups. A follow-up reader/critic experiment uses the same calibration pool and 200 fresh held-out contexts excluding the first held-out contexts. It was designed after seeing the first study, so it is exploratory, not a confirmatory replication.

CLIMATE-FEVER retains existing human support labels, with 100 calibration claims and 200 held-out claims, five evidence sentences each. The initial small-judge study and the later span-judge comparison use different held-out claim sets. Existing annotations are not new independent human adjudication; agreement and source completeness are limitations.

Every run records dataset URL/revision/hash, model revision, package versions, seed, split IDs and source-file hashes. Some snapshots were dirty when submitted; the recorded source hashes identify that exact payload. A pre-rewrite Git ID in an early manifest is historical provenance, not an available checkout reference. Current code includes later correctness fixes and should not be confused with a byte-identical original run. Reproduce with the recorded hashes/revisions and retain new outputs separately.

Percentile bootstrap intervals use 2,000 seeded resamples. Retrieval and answer scores resample questions; attribution resamples claims; injection resamples ten templates with their five variants; selective held-out coverage and risk resample articles. Confidence intervals describe sampling uncertainty within these settings, not uncertainty from model training, label error or domain shift. A bootstrap interval at 100% can collapse at the boundary: semantic chunk hit@2 is 100/100, but its Wilson 95% interval is approximately [96.3%, 100%].

## Retrieval and answer baselines

HotpotQA BGE-small results, 200 held-out questions, contextual headers enabled:

| Retrieval | Recall@5, 95% CI | MRR@5 | All supporting paragraphs in top two |
|---|---|---|---|
| Sparse BM25 | .8625 [.8250, .8950] | .9110 | .420 |
| Dense | .9425 [.9175, .9650] | .9352 | .575 |
| Hybrid RRF | .9250 [.8975, .9500] | .9446 | .560 |
| Hybrid, rerank depth 5 | .9025 [.8725, .9300] | .9524 | .500 |
| Hybrid, rerank depth 10 | .8900 [.8575, .9200] | .9510 | .495 |

Reranking promoted a good first hit but lost multi-hop support coverage. These measures must be assessed together. nDCG uses the total number of relevant paragraphs in the ideal denominator, including missed support. Hash-only full rankings and support labels allow metric recomputation without redistributing paragraph text.

BGE dense recall@5 without headers was .9375; MiniLM with headers was .8325 and without headers .8375. Header effects were small and depended on the embedder. MiniLM hybrid with headers reached .8925; reranking depth 10 reached .8900. At depth 10, essentially all supplied contexts reach the reranker, explaining identical rankings across some embedder variants. The full ablation matrix and intervals are in [the retrieval report](../reports/benchmarks/hotpot/RESULTS.md).

With the same held-out questions and a Qwen2.5-1.5B reader:

| Context baseline | Exact match, 95% CI | Token F1, 95% CI |
|---|---|---|
| Naive dense top five | .200 [.145, .255] | .247 [.192, .303] |
| Hybrid plus reranker top five | .190 [.135, .245] | .257 [.201, .315] |
| Long context, all ten paragraphs | .120 [.080, .165] | .145 [.100, .193] |

The F1 point difference is small and does not establish an improvement. These direct-reader experiments exercise retrieval plus answer generation, not the whole authenticated UI/API or review workflow. Short answers, citations and intermediate thought are normalized separately; failed or empty generations are retained. The long-context baseline performed worst, indicating that additional context alone was insufficient for this small reader.

## Attribution beyond citation existence

Runtime attribution identifies claims, resolves cited retrieved chunks, selects one- or two-sentence evidence windows and records exact character offsets plus quotes. A pinned NLI model tests whether evidence entails the claim. Unavailable judges or unsupported claims route to review. Claim segmentation is automatic in the application; the public NLI study supplies human-defined claim units and therefore does not validate that segmentation step.

The old metric only tested citation identity. All three controlled examples have a valid source label, scoring 1.0 under that metric: the correct coolant limit of 300, a fabricated limit of 900, and an unrelated claim about a chief executive's resignation. NLI accepts the first and rejects the latter two. These are demonstrations, not a human-labeled benchmark.

On the initial CLIMATE-FEVER held-out 1,000 pairs, positive-all citation existence has attribution precision .238 and recall 1.0. The small NLI model at .8 has precision .704 [.478, .870] and recall .080 [.039, .133]. It reduces unsupported positives but misses most human-supported evidence.

A fresh span-pipeline comparison, 200 held-out claims / 1,000 pairs, same .8 threshold:

| Judge | TP / FP / FN | Precision, claim-bootstrap 95% CI | Recall, claim-bootstrap 95% CI |
|---|---|---|---|
| Default small DistilBERT | 14 / 13 / 222 | .519 [.292, .758] | .0593 [.0282, .1013] |
| BART large | 8 / 3 / 228 | .727 [.467, 1.000] | .0339 [.0128, .0586] |

BART was selected using calibration F1 before the fresh held-out run. It is not substituted silently for the default runtime model; its memory cost is material. Both judges have very low recall. Exact quote round-trip is 100%, which establishes coordinate integrity, not semantic validity. Human labels concern supporting sentences, not minimal spans; recall here is over claim/evidence pairs, not all facts in a generated answer. NLI does not establish that a source is true, current or complete. No independently human-labeled end-to-end generated-answer attribution set was collected.

## Selective prediction and review cost

Risk is the exact-match error among automatically accepted decisions, including both answers and correct unanswerable decisions. Decision coverage counts all accepted decisions; answer coverage counts only accepted non-abstentions. Review is a separate action. We test a fixed 21-threshold grid, require at least 30 calibration decisions, and use a one-sided Wilson upper bound with Bonferroni correction over thresholds. These approximate bounds assume independent calibration questions, despite article clustering; they are not a distribution-free production guarantee.

The 3B reader scores .575 [.505, .645] EM and .594 [.525, .659] F1. It falsely abstains on 36% of answerable questions and correctly abstains on 70% of unanswerable questions. Token confidence cannot meet the 10%, 20% or 30% calibrated risk targets: each policy sends all 200 questions to review. Risk at zero coverage is undefined, recorded as null rather than zero.

The exploratory quantized 7B reader scores .665 [.600, .730] EM and .689 [.626, .752] F1 on fresh held-out contexts. False abstention is 29%; correct unanswerable abstention is 80%. A separate question-aware critique pass uses the same model's YES/NO next-token odds. This is not an independent judge and can have self-preference. Neither critique nor token confidence passes any of the three calibrated targets. The stronger model therefore did not establish useful safe coverage either.

Descriptive, uncalibrated .8 cutoffs illustrate the problem:

| Study / score | Decision coverage | Selective risk, article-bootstrap 95% CI | Reviews / assumed minutes |
|---|---|---|---|
| 3B token probability | .945 [.909, .975] | .402 [.317, .478] | 11 / 11 |
| 7B critique | .905 [.862, .949] | .309 [.234, .394] | 19 / 19 |
| Any calibrated target above | 0 | Undefined | 200 / 200 |

Review minutes assume 60 seconds per case. No reviewer timing, correction accuracy, user study or economic benefit was measured. Cost-matched random-review controls have zero accepted decisions under the selected policies and cannot demonstrate a utility advantage. The [risk curves](../reports/benchmarks/coverage-risk.png) are descriptive held-out threshold sweeps and must not be used to choose a deployment threshold after observing the labels. The current implementation is a research candidate for calibrated abstention plus review, not a proven novel result.

## Chunking and protocol ablations

On 100 answerable SQuAD2 paragraphs, using original human answer offsets and BGE embeddings, max chunk size 600 and min 150:

| Chunker | Span hit@1, 95% CI | Span hit@2 | Mean chunks | Mean chunking time, 95% CI |
|---|---|---|---|---|
| Recursive | .94 [.89, .98] | .99 | 1.86 | .079 ms [.075, .084] |
| Semantic | .92 [.87, .97] | 1.00 | 2.02 | 21.346 ms [14.159, 34.963] |

Recursive overlap is up to 100 characters; semantic overlap is one sentence when size permits. This is a practical configuration comparison, not an overlap-controlled causal study. Semantic chunking did not justify its extra embedding work here. It remains configurable rather than being advertised as an improvement.

The hosted comparison uses one free Groq model, `qwen/qwen3.8-27b`, fixed dense top-five contexts clipped to 600 characters each, temperature zero, reasoning disabled, 512 output tokens and at most four model calls. Native tools and ReAct alternate question order. Attribution/review are disabled to isolate orchestration. Twenty questions per protocol were requested, and only twelve orchestration attempts per protocol fit within the conservative 80,000-token cap. Unserved/budget cases remain failures in the full denominator.

| Protocol | Exact match, 95% CI | Failures / 20 | Median / p95 latency |
|---|---|---|---|
| ReAct | .35 [.15, .55] | 9 | 15.75 / 53.28 seconds |
| Native tools | .30 [.10, .50] | 10 | 16.26 / 56.64 seconds |

Native had two provider bad-request errors. These failures were not repaired and rerun until a favorable result appeared. Free-tier quota waits affect latency; small sample size, truncation, failed requests and order effects prevent a native-tool advantage claim. The p95 under 20 seconds pilot target was not met. Known token usage is 72,080 plus 4,090 conservative failure reserve. Provider records and manifest are [published separately](../reports/protocols).

## Injection and robustness

The direct 1.5B context reader receives 50 adversarial documents, ten attack templates with five variants, under weak and stronger JSON/untrusted-data prompts. Success means producing the attack target. Template-clustered 95% intervals are wide: weak .12 [0, .32], stronger .30 [0, .60]. Ten repeated benign controls all abstained. Over-refusal and increased attack success are negative results. Prompt-only hardening is insufficient in this test.

This suite does not measure the hosted service's full tool loop, data exfiltration, authorization bypass or every attack family. Parser type/size/binary checks, merged-cell handling, malformed prefixes, Unicode text, source collisions, atomic ingestion and interrupted generation are covered separately by regression tests. Model robustness is not inferred from those tests.

## Local operations and limitations

On the available Windows machine, Python 3.11, synthetic PDF parsing peaks at 35.5 MiB for 100 pages and 38.0 MiB for 300 pages. The warm 300-page run being faster is a warmup/cache artifact, not evidence of favorable scaling. These fixtures differ from the earlier problematic document, so no direct speedup is claimed. Parsing runs separately with enforced resource limits.

A 50,000-chunk BM25 microbenchmark built in .453 seconds; 400 queries at concurrency four had p50 3.28 ms, p95 5.21 ms and zero errors. An in-process ASGI test used demo/hash backends, 200 requests, concurrency four and an explicitly raised 500/minute quota: 28.66 requests/second, p50 108.5 ms, p95 202.1 ms, zero errors; async upload completed in 3.73 seconds. It excludes real model inference, external network and reverse proxy costs. This is not a hosted service load SLA.

A separate 50,000 by 384 synthetic normalized-vector test uses 100 queries, four CPU threads and the runtime HNSW settings (M=32, efConstruction=80, efSearch=64). Exact search built in .042 seconds and had p50/p95 13.58/19.64 ms. HNSW built in 116.54 seconds and had p50/p95 4.25/5.95 ms, but recovered only 26.9% of the exact top ten neighbors. Serialized index sizes were 76.8 MB exact and 90.4 MB ANN. This poor synthetic recall is a material negative result, not semantic retrieval accuracy. Flat remains the default. Memory peaks were process cumulative and the single-run timings have no uncertainty interval. [Raw ANN measurements](../reports/ann_scale.json).

Auth is required by default, pending answers are withheld, parsing and queues are bounded, corpus replacement is atomic and index snapshots are coherent. Optional HNSW provides ANN search, but refresh still loads the corpus and rebuilds caches. Multi-process coherence, incremental index updates, sustained ANN scale, distributed queues, tenant ACLs, full tracing infrastructure and measured production recovery are not established. SQLite remains a deliberate single-process pilot boundary. Document deletion does not erase independent backups or provider retention.

The current evidence favors simple dense retrieval and explicit review over unsupported complexity claims. Larger independent human attribution labels, a real timed reviewer study, context-grouped calibration with adequate sample size, broader injection tests and a full-service load run are the next experiments. They are not claimed as completed.


## Public response-support follow-up, 16 November 2026

A separately frozen RAGTruth component compares two pinned NLI models on existing recorded responses and original human annotations. Six hundred calibration sources and 450 official held-out sources are disjoint by source ID and context hash. There are three task strata, so only 150 held-out responses are QA. All 1,050 requested rows per model remain available, including 155 above the fixed twelve-span ceiling. Each model completed scoring 390 of 450 held-out responses; there were no unserved or failed cases.

Neither the primary task-stratified criterion nor the predeclared pooled sensitivity selected a policy. Selected coverage is zero and risk is undefined. The all-response baseline has 149 unsupported of 450 responses, with a nominal one-sided 95% upper bound of 36.9%. A fixed legacy NLI condition releases 61/450, with five unsupported and an upper bound of 16.5%; the same numeric threshold on the comparison model releases 25/450, with three unsupported and an upper bound of 28.2%. These are descriptive fixed-grid points, not operating policies selected on test outcomes. No safe coverage, PDF transfer or general model superiority is established.

The [full report](../reports/releases/r6/README.md) includes support precision/recall, task failures, clipping, exact bounds, original source rights and execution hashes. The model-free command `python scripts/verify_support_risk.py reports/releases/r6/component` recomputes saved selections and metrics. It checks internal consistency, not annotation correctness or independent replication. Standard public CPU runners were used, with no new reader, hosted or GPU calls. The product keeps NLI diagnostic-only; realistic PDF correctness, claim-extraction recall and independent human risk validation remain separate requirements.
