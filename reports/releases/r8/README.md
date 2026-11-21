# Both PDF judges endorse every scored claim; negative-label evidence is insufficient

This exploratory diagnostic scores the exact displayed claim/evidence inputs from the same 17 selected English PDF answers used in R5 and R7. It adds no reader generation, independent questions or Hindi evidence. The [cohort, thresholds and compute design](pdf_attribution_design.json) were committed in `7abb6ff` before inference; the separate-reference analysis was committed in `45eca09` before inspecting predictions. An always-support control was added after prediction inspection and is explicitly descriptive. Historical results and original human votes remain unchanged.

## Fixed inputs and completion

Two pinned NLI judges ran in FP32 on a private free CPU notebook, with the existing 256-token pair and 128-token hypothesis limits. Each scored **35 of 36 requested claims**. One claim had no evidence and remains `missing_evidence`, not an inferred negative. No completed premise was clipped, no claim exceeded its token limit and no scoring or numerical failures occurred. The same 17 answer clusters remain the question denominator.

The legacy judge's score range was 0.97497 to 0.99842; the comparison judge's was 0.84941 to 0.99463. Both therefore predict support for every scored claim at the primary 0.8 threshold and descriptive 0.5 sensitivity. Both equal an always-support baseline on these inputs. There is no demonstrated classification advantage between the judges here.

## Original references remain separate

The following counts apply identically to both judges at both thresholds. These are support judgments about displayed claims, not answer correctness or claim-extraction recall.

| Reference | Decisive labels requested | Completed predictions with decisive labels | TP / FP / FN / TN | Precision | Recall |
|---|---:|---:|---|---:|---:|
| First reviewer | 36 | 35 | 32 / 3 / 0 / 0 | 91.43% | 100% |
| Second reviewer | 32 | 31 | 31 / 0 / 0 / 0 | 100% | 100% |
| Concordant decisive subset | 30 | 29 | 29 / 0 / 0 / 0 | 100% | 100% |

The first review has 32 supported and four unsupported labels. Three of its unsupported judgments concern scored claims; all three are disputed by the second review, which called two supported and one unclear. The only unsupported claim agreed by both reviewers is the claim with no evidence, which neither model scored. The second review also has one unclear claim and three missing claim judgments from an unfinished answer. The concordant subset is selected on agreement and cannot be treated as independent gold.

Consequently, the second-review scored subset contains **zero decisive negative labels**. Its apparent perfect precision and recall do not validate false-positive detection. Joint adjudication, if returned, will be an additive sensitivity rather than a replacement for either original reference.

The predeclared 2000 seed-19 answer-cluster bootstrap gives first-reference precision [82.86%, 100%]. Recall is a degenerate [100%, 100%] in this selected sample. Second-reference and concordant intervals are also degenerate [100%, 100%]. All 2000 ratios were defined. These mechanical conditional resampling intervals cannot represent unseen failure modes or certify safety. Paired common-input differences are zero with [0, 0] intervals because the two judges make identical binary predictions, not because population equivalence has been established.

## Location, overlap and support

The original frozen response records contain 37 evidence entries: 34 normalized locations, two literal locations and one missing entry. Entries and grouped claims have different denominators. The old citation-existence score was not retained in these quote-first records and is **not reconstructed as 1.0**. A location match is not a support label.

A separately marked post-hoc text-overlap check found four of the 35 nonempty claims as literal substrings of their evidence and 30 after NFC plus whitespace collapse. This describes a quote-heavy task; it did not alter model inputs, thresholds or cohort selection. High conditional recall here does not overturn the earlier low-recall span-selection and public mixed-task benchmarks, which use different inputs and tasks.

## Compute and reproduction

The first CPU attempt failed in environment setup before producing predictions. The repaired second attempt used the identical frozen source/input bundle and hash-locked packages. It completed in 242.86 seconds including 211.11 seconds of setup. Reserving 60 seconds for the failed attempt charges 302.86 seconds against the original 1800-second ceiling. No GPU time, hosted calls, paid services or human-review time were used for this scoring study. See [compute provenance](compute.json) and [exact runtime/model identities](scoring_identity.json).

[Numeric records](numeric_records.json) retain every requested claim/model slot, including missing evidence. [Aggregate analysis](pdf_attribution_summary.json) contains separate original-reference metrics, missing counts and paired comparisons. Raw source text, human labels, rationales and the private input payload remain outside Git. The public numeric records can be checked against the frozen input manifest; reproducing human-reference metrics requires the private source exports. `python -m scripts.analyze_pdf_claim_diagnostic --help` documents the validated private analysis inputs. Output paths must be new.

This study supplies a real-PDF diagnostic, not independent population validation, calibrated coverage, safe automatic release, general attribution-extraction recall or Hindi support. It does not justify enabling NLI as a release gate. A larger independently reviewed sample with genuine supported and unsupported evidence is still needed.
