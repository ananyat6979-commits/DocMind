# Independent release gates

R0 fixes defects and establishes shared Unicode boundaries. R1 measures English and Hindi without waiting for R2 deployment isolation. French is the next candidate and remains gated until measured.

## R0 behavior

Chunking, claims and evidence share original-text boundaries. Soft wraps and numeric fragments survive. Search representations normalize Unicode, Latin accents, join controls and decimal digits without rewriting source evidence. Hindi character trigrams are an optional experimental representation, not a default quality claim.

Page extraction records distinguish usable Unicode, suspected legacy-font corruption, image-only/empty and uncertain/mixed content. Known corruption is quarantined. Remaining quality flags and partial documents require review. No heuristic certifies Hindi readability. OCR and legacy-font conversion are outside this release.

Authentication precedes authenticated quota allocation. Per-key and actual-peer limits are bounded. Unreferenced failed uploads are removed while document and queued/running-job references are protected. Parser subprocess environments start from an allowlist. Pending and rejected answers remain hidden, including claim/evidence payloads. Edited answers carry no stale model attribution. Final responses require authenticated retrieval. Library deletion requires the independent administrator credential.

## R1 evidence contract

The diagnostic question pool is synthetic and template-based, with at most 200 records per language. Proposed answers and exact evidence locations are drafts until a human adjudicates them. Public component benchmarks and PDF tasks are separate. English automatic checks do not become human verification. Historical measurements remain immutable; new versions receive new baselines and cache directories.

The local-reader T4 profile has a six-hour total ceiling, two-hour checkpointed sessions and a 20 percent reserve. Pilot timings determine a balanced cohort before its outcomes are inspected. Cost includes baseline capture, fresh answers, loading, retrieval, NLI and replay. Interrupted cohort rows stay in the requested denominator. The hosted default remains unmeasured. No additional hosted evaluation requests or paid fallback are authorized.

Human PDF work is capped at 90 minutes: 10 minutes for a five-item rubric pilot, 50 for Hindi answer correctness, 25 for claim support and 5 buffer. Public-label review has a separate 30-minute cap. Blinded, randomized routing strata must include candidate releases and withholds. Shadow decisions are distinct from enforced withholding. Missing labels and duration-based stopping are reported. Population weights require a completed preselected probability sample. Fewer than 25 reviewed candidate releases yields insufficient evidence about false-release risk. Small claim samples remain exploratory and cannot establish claim-extraction recall. Single-reviewer consistency is not inter-annotator agreement.

## R2 isolation gate

The default PDF/DOCX parser mode is a separate service. Missing service access fails closed. Native parsing is trusted-input development mode only and requires both `DOCMIND_PARSER_MODE=local` and `DOCMIND_ALLOW_UNSANDBOXED_PARSING=1`. Resource limits alone are not a sandbox.

The service release requires filesystem and network confinement, bounded socket transport and output, resource recovery tests, individual credentials and centralized enforcement. Neural results may be reused only when text, page identities, offsets and model-consumed metadata are equivalent. Changed extraction invalidates affected downstream artifacts.

## Publication

Keep restricted text, translations, annotation files and private review notes outside Git. Establish redistribution terms per document and dataset. Record XNLI noncommercial terms and the upstream MS MARCO research restriction alongside mMARCO's declared Apache license. Model licenses do not override upstream dataset restrictions. Automatic annotation never substitutes for the human review budget.

## Core-reader follow-up

The [locked follow-up](CORE_RELEASE_PLAN.md) and [current evidence](CORE_RESULTS.md) document the later unverified quoted-draft default, bounded concurrency, fresh local-reader control and authorized 150-minute aggregate human-review allowance. These changes supersede the initial 90-minute planning cap without changing the six-hour GPU ceiling or historical results. Hindi and French remain gated until their relevant measurements pass.
