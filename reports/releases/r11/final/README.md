# English reader scores below always abstaining on 92 supplied passages

All 92 frozen SQuAD2 development questions completed. The quantized reader scored 23.9% candidate short-answer exact match (22/92) and 32.7% token F1. The balanced always-abstain control scored 50.0% on both metrics. The paired exact-match difference was -26.1 percentage points with a nominal question-bootstrap 95% interval of [-42.4, -8.7]. Resampling the 28 source articles gave [-45.0, -4.0] as a sensitivity check.

| Cohort | Exact match | Token F1 |
|---|---:|---:|
| All 92 | 23.9% [15.2, 32.6] | 32.7% [23.9, 41.6] |
| 46 answerable | 47.8% [34.8, 60.9] | 65.4% [54.1, 76.2] |
| 46 unanswerable | 0/46 | 0/46 |

The table uses nominal question-bootstrap 95% intervals. With zero credited unanswerable cases, the resampling distribution is degenerate. That does not establish a zero population success rate. Whole-article intervals for the full cohort were [14.3%, 34.9%] for exact match and [22.4%, 44.7%] for token F1.

The scorer flagged 30 valid short answers on unanswerable questions, 19 invalid auxiliary answer fields and one format failure across the cohort. Completion measures service. Answer scores reflect the auxiliary field or a valid explicit abstention. The policy marked 74 outputs as drafts and held 18. Candidate scores include the held outputs. Release state alone does not establish factual correctness or entailment.

The timing-only decision selected 92 of 200 candidate questions before pilot answer inspection. The final notebook used the pinned quantized weights, tokenizer, runtime and reader policy specified in the [protocol](../PROTOCOL.md). It took 7510.78 wall seconds, including 182.63 seconds of setup, with a median question time of 73.24 seconds. No completed sequence reached the output-token cap. No GPU or hosted calls were used. The final reservation was 16200 free CPU seconds and the total allocation including the three pilots stayed within 21600 seconds.

These supplied-passage public short-answer results apply to the selected cohort and this execution profile. The public labels were not independently audited here and pretraining contamination is unknown. This study leaves PDF answer correctness, retrieval performance, and Hindi attribution unmeasured. Earlier SQuAD2 results use different readers and cohorts. The [numeric record](diagnostics.json), [timing](timing.json), [compute identity](compute.json) and [manifest](publication_manifest.json) support reproduction of the aggregates.

Run `python -m scripts.verify_quantized_english reports/releases/r11` to check cohort identities, score aggregates and run provenance. Raw sources, references and outputs are retained privately.
