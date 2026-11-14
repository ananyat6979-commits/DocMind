# English and Hindi release evidence

This release separates corrected service behavior, component measurements and deployment isolation. The measured evidence does not establish a validated English/Hindi PDF answering release. Hindi and other unvalidated languages continue to require review. French remains unmeasured.

## Protocol and provenance

Historical benchmark files remain unchanged. New runs use Transformers 5.23.0, Sentence Transformers 5.7.0, tokenizers 0.23.3 and pinned model revisions. CPU and CUDA environments have hash locks. Source snapshots, corpus hashes, offsets and model-consumed metadata identify cached inputs. Numeric summaries are under [reports/releases/r1](../reports/releases/r1). Restricted PDF text, XNLI examples, human annotation records and private review notes stay outside Git.

XNLI is a component test, not PDF attribution. Each language uses 200 seed-seven test examples with original public labels. The selected multilingual model trained on XNLI development data, which is excluded here. Entailment uses a fixed 0.8 threshold. Intervals use 2,000 premise-cluster bootstrap resamples. A zero-error precision result with one positive decision per independent cluster uses an exact binomial boundary interval; the degenerate bootstrap interval is retained alongside it. This reporting correction was made after the first run without changing predictions or thresholds.

## NLI results

| Judge and language | Precision, 95% interval | Recall, 95% interval |
|---|---:|---:|
| English judge, English | 97.9% [92.9, 100] | 71.2% [60.3, 82.0] |
| Multilingual judge, English | 100% [89.1, 100] | 48.5% [36.1, 60.9] |
| English judge, Hindi | 50.0% [35.9, 65.0] | 33.3% [22.4, 44.8] |
| Multilingual judge, Hindi | 88.6% [77.4, 97.4] | 47.0% [34.8, 59.3] |

Local CPU, free CI CPU and T4 runs produced the same aggregate decisions. The [CI records](../reports/releases/r1/cpu-ci) retain 800 numeric NLI predictions and 400 ANN rankings without source text or gold labels. Recompute their counts and intervals with `python scripts/verify_component_records.py reports/releases/r1/cpu-ci --xnli-cache <private-cache>`. The cache holds `xnli-en.parquet` and `xnli-hi.parquet` downloaded from the pinned revision. This checks scoring and cohort accounting without rerunning inference. The multilingual judge improves this Hindi component result but reduces English recall. Neither result validates real Hindi PDF claim support, claim extraction recall or false-release risk. A blanket judge replacement would conceal the English regression.

## Bounded local-reader T4 pilot

The pinned Qwen3-4B-Instruct-2507 reader ran in FP16 with temperature zero, seed seven and SDPA on a Tesla T4. Three English template drafts over a real licensed PDF completed. There were six reader calls, zero recorded numerical failures and zero token-cap truncations. Each question used two calls. Input tokens grew from 348-362 to 2,083-2,718; generated output totaled 156, 335 and 203 tokens per question. Question times were 20.23, 34.51 and 21.16 seconds. These three questions cannot establish Hindi speed or final cohort size.

The component and reader profile took 177.84 seconds; setup-inclusive measured session time was 268.37 seconds with two GPUs allocated. The watchdog limit was 6,900 seconds. Measured elapsed allocation and conservative reservation are separate accounting quantities. The process-wide peak CUDA allocation was 11.38 GB, including component work; this is not an isolated reader memory measurement. No paid compute or new hosted evaluation requests were used.

Frozen answer/evidence replay held claims and the judge fixed. Legacy versus repaired evidence segmentation produced supported fractions 0 to 0, 1 to 1 and 0 to 1. These are support judgments, not human correctness labels. All three answers required review because the source document contained quarantined pages. That 100% routing rate includes the partial-document policy and cannot be attributed to the judge alone. A clean-PDF default withholding rate remains unmeasured. The hosted default also remains unmeasured.

No balanced English/Hindi PDF cohort was frozen: a usable Hindi task pool and separate fresh-condition timing were not established. The remaining compute ceiling is not a reason to invent a cohort or label drafts human gold.

## Extraction diagnostics and human review

Seven deliberately selected Hindi PDFs contained 46 pages: 23 suspected legacy-font pages, 17 uncertain/mixed, four image-only/empty and two automatically usable. The two passing pages were short pages from the legacy-font document and yielded two unreviewed template drafts. They do not establish a usable Hindi PDF evaluation cohort. The English PDF had 92 pages, 89 usable and three suspected. These heuristics do not certify readability; blank pages are not distinguished reliably from scans. This is a diagnostic sample, not a representative language estimate. OCR and font conversion remain outside scope.

One reviewer completed the blinded 15-pair Hindi public-label packet in 520 active seconds: 11 decisive labels, four unclear items and four disagreements with original labels among decisive items. The 36.4% decisive-subset disagreement rate is not a population label-error estimate. Original public benchmark labels remain unchanged. Optional delayed consistency is unmeasured. No inter-annotator agreement is claimed. The PDF review budget remains unused, and Hindi PDF answer correctness and claim-support labels remain unmeasured. False-release risk therefore has insufficient evidence.

## ANN and retrieval scope

Real multilingual E5 embeddings replaced the historical Gaussian-vector stress input for a new application-path test. HNSW Recall@10 against exact search was 99.1% [98.5, 99.6] over 100 queries and 3,340 vectors. CPU unscoped median search time was 0.140 ms versus exact 0.273 ms, excluding embedding. Scoped requests deliberately use an exact filtered scan, with linear cost. This small sentence corpus establishes fidelity on this input, not production scale, semantic relevance or an incremental persistent index. The old 50,000-vector experiment remains unchanged and is not directly comparable.

The MIRACL workflow freezes 200 original public dev queries per language, retains all positive passages, up to ten judged negatives per query and deterministic distractors. Its lexical, embedding, reranker and candidate-depth comparisons are a judged-candidate component experiment, not official full-corpus MIRACL. Restricted source text stays private. Every condition reports requested-cohort and completed-only statistics; unserved rows retain zero scores in the requested denominator. Paired differences use matched completed query IDs and must not be generalized from incomplete profiles. Results remain unmeasured until their workflow finishes and artifacts validate.

## Service and deployment checks

Behavioral regressions cover Unicode boundaries, numeric fragments, citation ambiguity, review leakage and edits, authentication before quota allocation, bounded key/IP buckets, reference-aware orphan cleanup and environment allowlisting. The isolated parser uses a bounded Unix socket, no network access or secret mounts, a read-only filesystem, an unprivileged identity and resource controls. CI exercises denial, exhaustion and recovery. The same real benchmark PDF matched native and container text, page identities, offsets and model-consumed metadata, including the cached Windows extraction identity. This supports transport-only reuse for that tested input, not universal parser equivalence.

The centralized UI delegates library and review access to the authenticated API. Individual credentials supply trusted audit identities. SQLite, in-process quotas and the job queue still require one API service process. Tenant isolation, distributed enforcement, incremental persistent indexing, representative API load and measured review time savings remain limitations. Container confinement is not a claim of protection against every kernel vulnerability.

## Licensing

XNLI upstream noncommercial terms apply independently of model licenses. mMARCO declares Apache-2.0 while upstream MS MARCO specifies noncommercial research use; this is not commercial clearance. MIRACL dataset metadata and underlying Wikipedia attribution/share-alike obligations both matter. Per-PDF licenses, hashes and source URLs appear in extraction diagnostics. Publication here uses aggregate results and provenance, without restricted text or annotation records.

See [release gates](RELEASES.md), [model card](MODEL_CARD.md), [threat model](THREAT_MODEL.md) and [centralized deployment](DOCKER_CLOUD.md). Remaining unmeasured gates are explicit release limits, not passing results.
