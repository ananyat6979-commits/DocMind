# Bounded calibration, review cost and scale protocol

## Calibration

Use the pinned HotpotQA distractor validation parquet (SHA-256 `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`). Select in hash order, excluding any question whose context title or paragraph hash occurs in an earlier selected question. Ten timing-only questions are excluded from calibration and held-out candidates. Shuffle the remaining four hundred candidates with seed seven and split into two pools of two hundred. Exact repeated article titles within a question are combined for retrieval; title comparisons between questions are conservatively case-insensitive.

Retrieve five supplied article paragraphs with the pinned multilingual E5-small model. Reader evidence is limited to 384 native tokenizer tokens per article. Run the pinned Qwen3-4B-Instruct-2507 reader in FP16, greedy decoding, seed seven, SDPA, input cap 6144 and output cap 96. Score the same candidate answer with a separate verification prompt, recording the next-token probability of Y and the total Y/N probability mass. This is a correctness-prediction feature, not human claim attribution. Compare generated-token confidence, verification score, and their minimum. Source paragraphs and reference labels remain private; public predictions and numeric results carry source attribution.

Choose equal calibration and held-out cohort sizes from the worst complete timing-pilot observation multiplied by 1.25, reserving 20 percent of remaining time. Freeze IDs before scoring evaluation outcomes. No held-out outcome may change the threshold grid, target risks, score definitions or cohort size. All frozen rows remain in the denominator after errors or interruption.

The grid is 0 to 1 in increments of .05. Targets are .1 and .2 exact-match error. Use one-sided exact binomial bounds and Bonferroni correction across 3 signals, 21 thresholds and 2 targets, with at least 30 selected calibration answers. The bound assumes independent matched-distribution examples; it is not a distribution-shift or factual-safety guarantee. Model abstentions, generation failures and absent scores are ineligible for release. Verification failures retain completed-generation evidence and can still evaluate the confidence-only baseline. Apply frozen thresholds to held-out questions once. Always-review and ungated-answer baselines remain explicit. No passing threshold is a valid negative result.

The free session is capped at 3600 wall seconds and conservatively reserved as 7200 GPU seconds for two allocated devices, including environment setup. There are no hosted-provider evaluation calls. No measured review duration is assumed in advance.

## Human review cost

Prepare a blinded, timed packet only after predictions exist. Keep questions, source text, reference answers, raw labels and review notes private. Compare full-source presentation with a deployable source-ordering and lexical-highlighting presentation; never highlight using the gold answer or supporting-fact labels. Randomize interface assignment within matched question pairs and balance order. Conditions are necessarily visible through layout, while their names, model verdicts, routing and gold labels are hidden.

Measure duration, completion, source expansion, decisions and disagreement with benchmark exact match. One reviewer and a small timed sample support exploratory within-person findings. If the probability sample is incomplete, report reviewed-subset descriptives rather than population savings. Count study time against the remaining human-review cap. A cost model uses measured durations only when available; sensitivity assumptions are labeled assumptions.

## Production profile

Use PostgreSQL and pgvector for transactional metadata and persistent vector updates. Server-derived tenant scopes and forced row-level policies protect all document, chunk, job, review and event reads and writes. Share authenticated quotas and lease-based ingestion claims across replicas. Expired workers cannot commit through an obsolete lease. Upload storage is a shared private volume in the single-host multi-replica profile. Database administrator credentials are separate from runtime credentials.

CI must exercise cross-tenant denials, duplicate claims, worker interruption/recovery, stale leases, shared quotas and persistent index changes. Load and ANN records must name corpus size, real embedding source, concurrency, filtering mode, recall against exact search, latency percentiles, failure counts and restart behavior. Synthetic service timing never becomes a hosted-reader SLA. Multi-host storage, regional failover and operator-managed backup restoration require separate deployment evidence.

## Broader robustness component

Freeze 180 synthetic cases before inference: six questions across ten injection families and two separately reported factual-poisoning families, each with raw and JSON document serialization, plus three benign controls per question and condition. Injection targets are plausible wrong numeric answers with valid source citations. Vary placement before the fact, after it and in a separate chunk. Include role spoofing, delimiter escape, encoded directives, reviewer and evaluator spoofing, long distractors and conflicting facts. Retrieval is fixed, so this measures the reader and policy, not adversarial retrieval or document-parser exploit resistance.

Report model-final contamination, candidate contamination and candidate automatic release separately. Strip citations before matching numeric targets. Presence of the correct number is a utility diagnostic, not human answer correctness. Preserve missing-outcome bounds and stratify factual poisoning. Domain variants of one attack family are dependent. Report family-level any-success nominal intervals, without treating the 60 injection variants as independent trials. The component is capped at 3000 wall seconds, reserved as 6000 GPU seconds including setup. Existing robustness results remain unchanged.
