# Public response-support component

Frozen before inference on 2026-11-16. This is a separate verification component using existing human annotations, not a new reader experiment, a realistic-PDF accuracy study or an automatic production release policy.

## Inputs and rights

RAGTruth revision c103204b9ce28d6bbad859304bf30de72b8ed8fe provides original manual case and word-span annotations. One response is selected deterministically per source, without filtering on support, quality or model scores. Use 600 calibration sources from the official training split, 200 each for QA, Summary and Data2txt. Use all 450 source-distinct official test cases, 150 per task. Context hashes are disjoint. This is 450 held-out responses, not 500, and the mixed-task result is not 450 QA questions.

Pin and verify source bytes, response hashes and context hashes. Raw inputs and word annotations remain outside Git. The repository has an MIT notice, but its QA sources derive from MS MARCO's noncommercial research data and other original source rights are not cleared by a repository license. No commercial clearance, dataset re-licensing or source-text redistribution is asserted.

References: [RAGTruth paper](https://aclanthology.org/2024.acl-long.585/), [dataset documentation](https://github.com/ParticleMedia/RAGTruth/blob/main/README.md), [repository notice](https://github.com/ParticleMedia/RAGTruth/blob/main/LICENSE), [MS MARCO terms](https://microsoft.github.io/msmarco/).

## Fixed methods

Compare the existing pinned NLI model with the pinned small DeBERTa NLI comparison. The comparison model reports SNLI and MultiNLI training, not RAGTruth training; unknown pretraining overlap is not independently audited. Its model card declares Apache-2.0. This does not clear source-data rights.

[Comparison model card](https://huggingface.co/cross-encoder/nli-deberta-v3-small).

Split responses with the project's offset-preserving sentence service. More than twelve spans, hypotheses above 128 tokenizer tokens, missing evidence or oversized inputs are withheld and retained in denominators. Evidence consists of overlapping 64-word windows with stride 32, ranked by lexical overlap. Score at depths one and four. Response support is the minimum across spans of the maximum selected evidence entailment score. Report premise clipping; it can lose support. This is not complete evidence retrieval, authority verification or claim-extraction recall.

The unsupported-response target means at least one original annotated span, including factual-but-unsupported implicit-true spans. Incorrect refusals and truncated responses are additional reported utility failures; their human quality flags never choose routing. Text-detected refusals are excluded from gated answer coverage. Supported-response recall alone does not establish answer usefulness.

Baselines are all responses, ungated non-refusals and the numeric guard without NLI. They are gating baselines on identical recorded answers, not a framework-based generation comparison. Do not call source availability a real citation-existence measurement: these outputs do not follow the product's citation contract.

## Selection and uncertainty

The fixed family has two models, depths one/four, numeric guard off/on and thresholds 0.5/0.8/0.9/0.95/0.99: forty policies. Choose using calibration only. The pooled selection requires at least 25 released examples, observed unsupported fraction at most 5% and a one-sided exact upper bound at most 8%, using alpha 0.05/40. The stricter selection additionally requires the bound in every task stratum, using alpha 0.05/120 and at least 25 releases in each. Break ties by coverage, smaller bound, higher threshold, lower model index, smaller depth, then the numeric guard on.

The stratified selection is primary; pooled selection is a predeclared sensitivity, not a replacement chosen after seeing test outcomes. Apply the chosen policies once to held-out cases. Display the full fixed-policy curves for descriptive comparison; never select from their held-out outcomes. Binomial bounds are conditional on independent, matched-distribution examples. Source disjointness does not establish exchangeability or transfer to PDF answers. Report per-task outcomes and both supported-response precision/recall and quality failures. Below 25 released examples, report insufficient evidence. A zero-release result is retained, not tuned away.

## Resources and publication

Use standard public Linux CPU runners, two CPU threads per process and at most 1800 seconds per model including setup. Use separate hash-locked evaluation additions. No GPU time, hosted reader calls, paid annotation, larger runners or additional accounts are used. Failed and interrupted cases remain requested rows; missing predictions are explicitly unserved. Outputs are never overwritten and failed model evaluations are not retried for better scores.

To avoid artifact-storage charges, transfer bounded numeric job outputs and retain compressed numeric checkpoints and plots in job logs. Do not upload dataset text, model weights, artifact archives or caches. Standard public-repository compute and job-log storage are covered by the documented free paths: [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions).

Publish numeric predictions, immutable identities, code hashes, calibration choices, held-out results, plots and negative outcomes. Do not change the live product's diagnostic-only NLI contract from this component result. A separate sufficiently large, human-reviewed PDF study is required for that decision.
