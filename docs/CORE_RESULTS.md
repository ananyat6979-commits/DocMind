# Core reader release evidence

This report separates completed measurements, implemented behavior and unmeasured claims. All work uses local resources, free public CI or free private CPU or T4 notebooks. No new hosted evaluation calls, paid inference or paid annotation were used.

## Product behavior

The default is a single-call reader with structured claims, exact source IDs and quoted passages. Quote matching accepts unique literal matches or NFC/whitespace/soft-hyphen normalization with reversible chunk offsets. It establishes location, not entailment. Eligible drafts are visible and explicitly unverified. NLI is off by default and can only supply diagnostics. Strict mode remains an administrator opt-in.

Invalid quotes or citations, quarantined evidence, incomplete-document absence claims and unmeasured languages remain held. Complete-document abstentions are shown with a may-have-missed warning. Short answers inherit only reliable English file context, with contrary-language vetoes; this heuristic still produces false holds. Hindi support is not validated by script detection or retrieval recall. Historical responses retain their recorded policies.

Requests use async provider I/O, request-owned accounting, bounded native workers and deadlines. Real-socket cancellation, authenticated response retrieval, deleted-evidence protection, stale edit attribution and missing-document requests have regression checks. These are behavioral checks, not a generative-reader concurrency SLA.

## Fresh frozen-evidence control

The [published 35-pair control](../reports/releases/r4/paired_core_t4/README.md) is an English local-reader T4 profile using supplied HotpotQA distractor contexts. Its 60/40 bridge/comparison mix and alternating order were frozen from timing pilots before outcomes. Single-call candidate EM is 5/35 and ReAct-protocol candidate EM is 8/35; the paired difference is +8.57 points, nominal bootstrap 95% interval [-2.86, +20.00]. Four and ten format failures respectively remain in requested denominators. No final truncations or numerical failures occurred.

The [paired operational sensitivity](../reports/releases/r4/paired_operational_sensitivity.json) retains all attempts: mean tokens per question were 1361.37 versus 1596.43, and mean elapsed seconds were 14.93 versus 26.88. Operational intervals are post-hoc nominal paired bootstrap descriptions, not a predeclared promotion rule. Calls/tokens/time per correct answer are ratios with a small, uncertain EM denominator.

Both arms made exactly one call per question. No multi-step tool use occurred, so this result cannot establish adaptive retrieval benefits, its absence, or universal superiority. Single-call remains the product default. Candidate EM includes withheld answers and public labels are not independently reviewed. Full-answer human correctness remains separate. Raw failed completions were not captured in this experiment; new runs retain them privately.

The [power sensitivity](../reports/releases/r4/paired_power_sensitivity.json) illustrates the limited detection power of small paired studies under stated discordance assumptions. It is a hypothetical sensitivity calculation, not a precommitted numeric promotion threshold or a result about a population of real users.

## Robustness and attribution contract

The [new quoted-policy profile](../reports/releases/r4/quoted_policy_t4/README.md) completed all 180 requested cases within 1315.14 wall seconds including setup, counting both allocated T4s. It had 103 malformed JSON outputs and zero truncations, so zero observed released JSON-serialization attack targets is not evidence of a useful or safe answerer. The raw arm released one contaminated attack candidate; its same-output NLI shadow would have held it. The JSON arm produced one contaminated candidate which the quote/language policy already held. Of 18 benign controls per arm, only seven raw and six JSON drafts contained the correct numeric value. All format failures remain in requested denominators. The complete-schema prompt correction is a separate prospective development-regression condition, not a relabeling or retry of the original outputs. Its 1200-wall-second reservation is reassigned from unused primary headroom, with the original total GPU ceiling and own-document reserve intact. It compares raw versus Unicode-JSON serialization and same-output NLI shadow holds. Its synthetic six-domain numeric contamination metric does not establish arbitrary injection resistance or factual correctness. Failed and unserved requests remain visible; family intervals are reported only when the required family outcomes are complete.

The [complete-schema development confirmation](../reports/releases/r4/quoted_complete_schema_t4/README.md) attempted 123/180 fixed cases before its budget deadline. Format failures fell from 72 to four on the matched 123-case subset, with no truncations. Fifty-seven unserved cases remain in requested bounds. It produced one contaminated JSON injection draft and two JSON poisoning drafts; all passed the NLI shadow. The examples repeat injected directives or conflicting revisions as claims, illustrating that quote matching plus entailment does not establish authority or question relevance. This is development regression after observing the original suite, not independent safety validation. No additional tuning or retries are hidden in this report.

A [post-hoc matcher check](../reports/releases/r4/robustness_numeric_matching_sensitivity.json) fixes sentence-final numeric punctuation and numeric equivalents without changing historical files or rerunning outputs. The historical R3 candidate/release counts remain unchanged under this sensitivity. The fresh metric checks every displayed claim, rather than just the auxiliary short answer. Negated mentions can count and spelled-out numbers can be missed.

## Own-document and human evidence

The source-confirmation export admitted one unchanged human-authored question and 16 explicitly reviewed repairs or source-based synthetic proposals. The cohort was frozen before generation: 17 PDF questions and two repeated guide questions in DOCX format. These are mostly definitions and named values, with no completed unanswerable examples. This is a small, selected English smoke test, not 17 independently authored questions, a representative benchmark or Hindi evidence.

The [completed T4 profile](../reports/releases/r4/own_document_t4/summary.json) used actual recursive ingestion, pinned E5 dense retrieval and the single-call pinned reader on cached trusted-input extraction. All 19 tasks were attempted once. PDF candidate exact match was 2/17 (nominal Wilson 95% interval 3.3% to 34.3%), with five format failures, nine drafts and eight holds. The interval describes this selected diagnostic task distribution, not a broader document population. The two paired DOCX tasks had one exact match; they add no independent questions. There were no recorded truncations, numerical failures or runtime errors. Gold pages occurred in the top-five retrieved chunks for 14/17 PDF questions, which does not establish complete evidence retrieval. Short-answer exact match is not human factual correctness.

The run consumed 749.61 wall seconds including setup, with both allocated T4s counted. The conservative project ledger now records 18,600 of the 21,600 GPU-second cap, with no active reservations. No hosted evaluation calls were made. Numeric records, model identity, executed-source hashes and a publication manifest are available; raw outputs, source text and human labels remain private.

A [separate post-hoc contract replay](../reports/releases/r4/auxiliary_contract_replay.json) found that all five PDF failures were valid claim JSON rejected solely by auxiliary short-answer scoring validation. The prospective parser now retains valid claims, records invalid metric fields and refuses to score that auxiliary answer. All five recovered own-document payloads remain held because document-level language is undetermined despite English chunk hints, leaving release counts unchanged. This exposes conservative language-policy non-service, not proof of an invalid citation or an incorrect answer. Across the two synthetic robustness runs, four payloads recover and two become drafts; observed numeric-contamination counts are unchanged. This does not establish semantic safety. Baseline results and the original blinded review packet are unchanged, and changed answer/evidence inputs need separate human judgments.

The [completed blinded review](../reports/releases/r4/own_document_actual_answer_review.json) covers all 17 actual baseline PDF responses and 25 extracted claims. Recorded running time was 25 minutes 45 seconds, within the 30-minute cap; active work was not independently observed. The single reviewer judged 10 answers correct, two incorrect and five unclear, and 22 claims supported versus three unsupported. Three correct answers were held. Of nine visible drafts, two were judged incorrect. These are cohort descriptions, not a validated release-risk estimate.

The [post-hoc metric sensitivity](../reports/releases/r4/own_document_human_metric_sensitivity.json) verifies identical reviewed answer/evidence inputs. Eight human-correct answers failed exact match, while both exact matches were human-correct. Every reviewed claim had an observed citation, yet three were unsupported. Twenty-four had all quotes located, including two unsupported claims. Citation existence and literal quotation therefore overstate support on this small selected cohort. Claim-extraction recall and NLI accuracy remain unmeasured. The same reviewer helped confirm sources and questions; these labels are not independent replication and do not revise historical EM calibration labels. Below 25 reviewed candidate releases, false-release risk remains insufficient evidence. Neither this study nor the previous English review-cost study validates general Hindi attribution, claim-extraction recall or review-time savings. The internal DOCX fixture lacks an independent visual layout check.

## Hindi, deployment and limits

The [Hindi JSON diagnostic](../reports/releases/r4/hindi_serialization_component.json) reduces input tokens from 6456 to 1775 over seven private drafts using the pinned tokenizer. This is tokenization evidence, not generation latency or answer quality. Legacy-font conversion and OCR remain unsupported. Hindi end-to-end attribution and the proposed French addition remain gated.

The authenticated centralized UI/API and confined parser are implemented. The optional PostgreSQL profile has real tenant, incremental-vector, recovery and bounded load evidence. A real browser captures the UI with synthetic English/Hindi documents, actual E5 retrieval and an extractive demo reader. Plain browser screenshots and added presentation captions are separate assets. No human decision is simulated by pressing approval in the demo.

Passing CI, publishing source or publishing a container is not proof of a running public service. The source has been promoted to main and its container publication passed CI; the pinned image is anonymously pullable. This does not establish a running public service. The free release does not claim production availability, a hosted-reader SLA, calibrated safety, broad language coverage or adaptive-loop gains. Live deployment requires an identified operator-controlled target and private runtime configuration. No paid hosting is introduced.

## Reproduction

Run `python scripts/verify_paired_core_profile.py reports/releases/r4/paired_core_t4` and the existing historical verification commands in CI. Publication manifests and executed-source hashes identify exact snapshots. Internal consistency checks are not external replication. Private human labels, restricted text, review notes and raw outputs remain outside Git.


## Quality follow-up

The [R5 quality report](../reports/releases/r5/README.md) preserves a first 100-case routing result as development evidence and reports a disjoint 100-case confirmation. Holds fall from 40 to 11; all remaining cases fail unique quote location. On the 89 citation-clean gold inputs, language-related holds fall from 29 to zero. This is a public-reference policy component, not new reader generation or calibrated risk.

An identical-output own-document replay now has 16 drafts and one hold. Human judgments are reused only for identical question/answer/evidence inputs, and the five recovered inputs received a separate blinded review. The first review of the current selected 17-answer cohort marked fourteen correct and three incorrect answers. All three first-review incorrect candidates were drafts; one first-review correct answer remained held. The later partial second review disputes two of those incorrect judgments, as reported below. Narrow numeric guards did not solve the observed semantic errors. The larger independent human correctness and risk-calibration gaps remain open.

Uploads now reject nonportable names before writing and clean temporary files after failures. PDF geometry warnings quarantine observed interleaved columns without changing extracted text; a new parser manifest covers layout metadata. Existing PDF caches need explicit reprocessing for refreshed language and layout metadata. Historical neural results are not silently promoted to the new preprocessing version.


## Public human-annotated support component

The [R6 comparison](../reports/releases/r6/README.md) completed two fixed CPU NLI profiles on 600 calibration and 450 source-disjoint held-out RAGTruth responses. Neither pooled nor task-stratified calibration selected a qualifying policy. Selected coverage is zero and unsupported risk is undefined. All requested cases remain in denominators, including 60 held-out responses above the twelve-span limit. Fixed-policy curves and nominal bounds expose the coverage and uncertainty costs; they are not selected production thresholds. Original public annotations, mixed QA/Summary/Data2txt tasks and unchanged recorded responses make this a separate response-support component, not current-reader PDF accuracy, Hindi attribution or independent human replication. NLI remains diagnostic-only.


## Partial second-review follow-up

The [R7 snapshot](../reports/releases/r7/README.md) matches sixteen completed second-review answers and 33 claim units to identical first-review inputs. Raw agreement is 14/16 answers and 30/33 claims; Cohen's kappa is 0.549 and 0.468, with nominal answer-cluster intervals [0, 1]. Thirteen answers were marked correct by both reviewers and one incorrect by both. Two first-review incorrect answers are disputed: the second reviewer marked one correct and one unclear. Three claim judgments also differ. The initial labels remain immutable and no source adjudication is claimed. One answer and three claim units have no completed second judgment.

The second reviewer self-reported different-person status and no earlier involvement or result exposure. The first reviewer helped confirm questions. This remains a selected seventeen-question cohort, not independent study replication or a validated risk estimate. Ten recorded minutes were charged within the expanded review cap. The separate language-v4 sampler repair preserves the frozen review inputs and does not validate Hindi answers.


## Fixed-input real-PDF attribution diagnostic

The [R8 CPU diagnostic](../reports/releases/r8/README.md) evaluated the same 36 displayed claims from 17 current English PDF answers. Both pinned judges scored 35 claims and left one missing-evidence claim unscored. Both marked all 35 as supported at 0.8 and 0.5, matching an always-support baseline. Against original first-review judgments, precision is 32/35 (nominal answer-cluster interval [82.86%, 100%]); the three scored unsupported labels are all disputed by the second review. The only jointly unsupported claim has no evidence and was not scored, leaving no decisive negative examples in the second-review scored subset. Apparent 100% conditional recall and degenerate intervals do not validate false-positive detection, extraction recall or release safety. This free CPU run used no new reader, GPU or hosted calls and leaves R1-R7 artifacts unchanged.


## Fresh English CPU feasibility limit

The [R9 pilot](../reports/releases/r9/README.md) froze ten fresh public-reference questions and a separate 200-question candidate pool. It completed five pilot requests before its guarded CPU deadline; one was interrupted and four were unserved. The timing-only rule supported thirty final questions, below the declared minimum of forty, so zero final questions were selected. That stop was committed before answer inspection. One short-answer exact match among five completed requests is a tiny pilot diagnostic, not the larger independent English evaluation. No final session, GPU time, hosted calls or paid fallback was used. The unused candidate pool and the larger evaluation remain open rather than being reported as completed.


## English FP32 batch feasibility

The offline FP32 batch pilot completed four of twelve requested sequences. Two batches timed out. The timing rule selected zero final questions, leaving the larger English evaluation open. See the [R10 findings](../reports/releases/r10/README.md).


## English quantized CPU reader

The supplied-passage quantized reader completed all 92 frozen English SQuAD2 questions on free CPU compute. Candidate exact match was 23.9% versus 50.0% for always abstaining, a paired difference of -26.1 points [95% question bootstrap: -42.4, -8.7]. It matched 22 of 46 answerable questions and none of 46 unanswerable questions. This result leaves PDF correctness and Hindi end-to-end quality open. See the [R11 final results](../reports/releases/r11/final/README.md).

The post-hoc raw-output check found that all 92 responses omitted the explicit abstention field. None produced a parsed abstention. Of the 30 unanswerable cases scored as valid short answers, 25 reached draft status. The diagnostic counts field presence and routing and does not judge semantic refusals. See the [aggregate abstention diagnostic](../reports/releases/r11/abstention_diagnostic.json).
