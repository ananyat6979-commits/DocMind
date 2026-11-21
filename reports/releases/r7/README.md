# Partial second review: high raw agreement, weak reliability evidence

This snapshot records a blinded second review of the same selected English PDF answers used in the quality follow-up. It does not add independently authored questions, new reader generation or new Hindi evidence. The original votes remain unchanged, and no adjudication had been performed at this snapshot.

## Fixed inputs and completed work

The [protocol](../../../docs/SECOND_REVIEW_PROTOCOL.md) was committed before second-review labels. It fixed all 17 current answers and 36 extracted claim/evidence units, a new seeded order and a ten-minute recorded timer. The builder did not load prior rating exports. Source PDFs and page images were bundled privately; prior votes, routing, NLI scores and result summaries were hidden.

The second reviewer completed **16 of 17 answers and 33 of 36 claim units** at the 600-second cap. All sixteen completed answers matched an earlier judgment by exact task, question and displayed answer/evidence fingerprint. One unfinished answer and its three claim units remain missing for this comparison. No automatic labels or completion weights fill that gap.

The reviewer declared a different person, no involvement in earlier question/label preparation, and no exposure to earlier results. This is self-reported independence, not externally verified identity or independent study replication. The first reviewer also helped confirm the source questions. The selected question pool remains seventeen unique PDF questions, not thirty-three independent questions.

## Paired judgments

| Dimension | Paired units | Raw agreement | Cohen's kappa | Nominal clustered 95% interval |
|---|---:|---:|---:|---:|
| Answer correctness | 16 answers | 14/16, 87.5% | 0.549 | [0.000, 1.000] |
| Claim support | 33 claims in 16 answers | 30/33, 90.9% | 0.468 | [0.000, 1.000] |

Both intervals resample the sixteen answer clusters, retaining their claim units together, using 2000 seed-19 resamples. Kappa was undefined in 80 answer resamples and 14 claim resamples. The reported percentile intervals condition on the remaining defined resamples and are descriptive. These broad intervals do not establish reliable agreement for a broader task population.

The labels are imbalanced. On the paired answers, the first reviewer marked thirteen correct and three incorrect; the second marked fourteen correct, one incorrect and one unclear. Thirteen answers were marked correct by both, and one incorrect by both. One first-review incorrect answer was marked correct by the second reviewer, and another was marked unclear. Those two disputes are unresolved.

On the paired claims, both reviewers marked 29 supported and one unsupported. Two first-review unsupported claims were marked supported by the second, and another was marked unclear. Those three claim disputes are unresolved. Unclear is retained as a separate category, not dropped or merged into a favorable label. High raw agreement with these marginals is insufficient evidence of strong label reliability.

The [aggregate record](second_review_agreement.json) retains confusion matrices, missing-item counts, bootstrap degeneracy, self-reported context, export hashes and analysis-code hashes. Raw labels, notes, task linkage and source text remain private. The initial agreement numbers are not improved by replacing either review with the other or by any future consensus exercise.

## What this changes and what it does not

The earlier single-reviewer result was fourteen correct and three incorrect among seventeen current displayed answers. The second review supports thirteen correct and one incorrect on identical paired inputs while disputing two of the earlier incorrect judgments. The remaining answer has no completed second judgment. This exposes reference-label uncertainty; it is not a new reader-accuracy improvement. No adjudicated gold set or population false-release estimate is created.

The frozen cohort has fewer than 25 reviewed candidate releases. A second rating of the same answer does not increase the number of independent cases. General release safety, claim-extraction recall, review-time savings, question-author independence and Hindi PDF attribution remain unvalidated. The written rubric and short review allowance do not establish domain expertise, active work time or a causal explanation for disagreements.

The [time ledger](review_budget_after_second.json) charges 600 recorded seconds, bringing the PDF-review total to 8944.9269 of the expanded 9000-second allowance, with 55.0731 seconds remaining. Recorded running time is not independently observed active work. This second-review cap was not extended after outcomes. Any later source adjudication needs its own prospective allowance and remains separate from these initial judgments.

## Runtime repair and migration

A separate code inspection found corrupted Hindi sentence delimiters in the language sampler. `language-v4` uses ASCII Unicode escapes for danda and double danda, and does not accidentally split on possessive apostrophes. Behavioral regressions cover Hindi deduplication, English possessives and cached-v3 ingestion.

Cached v3 documents retain their metadata and chunk IDs on ordinary reads. Explicit reprocessing refreshes language metadata and creates new chunk IDs, with the existing stale/deleted-evidence protections. Source text is preserved. [Migration instructions](../../../docs/API_MIGRATION.md#language-sampler-v4) describe that distinction. This repair does not remove Hindi review holds, regenerate the frozen answers or relabel historical neural measurements.

CI now checks Linux on Python 3.11, 3.12 and 3.13, plus Windows on Python 3.11. Linux requires a functioning Compose plugin and real configuration validation. Windows can report an unavailable optional plugin after a bounded probe. Once available, configuration errors and timeouts still fail the check. Parser confinement remains a separate Linux-container gate.

## Verification and access limits

Run `python -m scripts.verify_second_review reports/releases/r7` to recompute aggregate point agreement and kappa, denominators, budget arithmetic and saved-byte hashes. It does not independently recover private label matching or reproduce answer-cluster intervals from marginal confusion matrices. Exact paired import and clustered resampling require the preserved private packet, linkage and exports, using `python -m scripts.analyze_independent_review`.

The review analysis and sampler checks use existing local resources and standard public CI. They make no new generative-reader evaluation or hosted-provider calls, use no GPU evaluation time, and use no paid annotation. See [image publication](image_publication.json) for the runtime image's exact source, immutable digest and build inventory/provenance. Build provenance is not evidence of factual-answer safety.
