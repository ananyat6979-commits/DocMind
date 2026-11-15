# Core reader release evidence

This report separates completed measurements, implemented behavior and unmeasured claims. All work uses local resources, free public CI or free private T4 notebooks. No new hosted evaluation calls, paid inference or paid annotation were used.

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

The first private authoring export has five finished questions. Four need explicit answer/scope/quote confirmation; they are not silently repaired into gold. The user authorized 60 additional minutes of human review, taking the aggregate PDF/review allowance to 150 minutes. A 20-minute prefilled packet contains four proposed repairs and 30 source-based synthetic drafts. Only explicit complete human decisions can enter the frozen cohort. Proposed synthetic questions are never described as human authored.

The own-document runner uses the actual recursive ingestion, pinned E5 dense retrieval and single-call local reader, with trusted PDFs/DOCX parsed once and cached with text, page identities, quality metadata and hashes. It requires the frozen human export before generation, never retries evaluation failures and reports requested-cohort metrics. Repeated PDF/DOCX questions are a paired format diagnostic, not extra independent questions. The internal DOCX fixture has not had an independent visual layout check. Actual generated-answer correctness and claim support require a separate blinded timed packet; a source-confirmation label does not validate a model answer.

Below 25 reviewed candidate releases, false-release risk remains insufficient evidence. Even larger reviewed subsets need uncertainty and selection limitations. Neither the previous English review-cost study nor the planned small own-document smoke test establishes general Hindi attribution or review-time savings.

## Hindi, deployment and limits

The [Hindi JSON diagnostic](../reports/releases/r4/hindi_serialization_component.json) reduces input tokens from 6456 to 1775 over seven private drafts using the pinned tokenizer. This is tokenization evidence, not generation latency or answer quality. Legacy-font conversion and OCR remain unsupported. Hindi end-to-end attribution and the proposed French addition remain gated.

The authenticated centralized UI/API and confined parser are implemented. The optional PostgreSQL profile has real tenant, incremental-vector, recovery and bounded load evidence. A real browser captures the UI with synthetic English/Hindi documents, actual E5 retrieval and an extractive demo reader. Plain browser screenshots and added presentation captions are separate assets. No human decision is simulated by pressing approval in the demo.

Passing CI, publishing source or publishing a container is not proof of a running public service. The source has been promoted to main and its container publication passed CI; the pinned image is anonymously pullable. This does not establish a running public service. The free release does not claim production availability, a hosted-reader SLA, calibrated safety, broad language coverage or adaptive-loop gains. Live deployment requires an identified operator-controlled target and private runtime configuration. No paid hosting is introduced.

## Reproduction

Run `python scripts/verify_paired_core_profile.py reports/releases/r4/paired_core_t4` and the existing historical verification commands in CI. Publication manifests and executed-source hashes identify exact snapshots. Internal consistency checks are not external replication. Private human labels, restricted text, review notes and raw outputs remain outside Git.
