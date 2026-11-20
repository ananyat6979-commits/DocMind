# Second-reviewer protocol

Frozen before second-reviewer labels. This follows up the selected English PDF diagnostic; it does not expand its question population or create a new reader experiment.

## Inputs and blinding

Use the exact 17 current displayed answers and 36 extracted claim/evidence units from the published quality replay. The source profile SHA-256 is `111232d9b35c1cb7a595bc9f421442be92ad6e5db0940d8750c68802a1d57f36`. The private second-review packet SHA-256 is `49688553dc9c5c58949ead87a65f6f28749e34fe4a09cfa351418ac655d2aae5`.

The builder reads the frozen profile and source assets, not earlier rating exports. Include every primary PDF answer in seeded random order, seed 19. The reviewer sees the question, actual current answer, complete source PDF/page image and actual claim/evidence units. Proposed reference answers, prior judgments, routing states, NLI scores and result tables are excluded. The portable handoff contains only HTML, instructions, the two English source PDFs and required page images. Private linkage, reference labels, analysis and packet metadata files are excluded from the archive.

A self-reported anonymous code, R2, distinguishes the export. Before starting, the reviewer declares whether they are a different person, helped create questions or earlier labels, and have seen prior results. No choice is preselected and every truthful response is accepted. Independence is not assumed: only different-person=yes, prior-involvement=no and prior-results=no supports a self-reported independent second judgment. This is not external identity verification or independent study replication.

## Work and time

Reserve 600 recorded running seconds from the remaining 655.0731-second PDF-review allowance, leaving 55.0731 seconds unreserved. Previously charged time is 8344.9269 seconds against the expanded 9000-second ceiling. Creating or testing a packet creates no human labels and consumes no human-review time. Actual imported timer data, rather than the reservation, will be charged.

The reviewer works alone, without discussing ratings or consulting prior results. Correctness is judged against the full source. Claim support is judged against the evidence actually displayed. Missing evidence stays missing. Pause for genuine breaks. At the timer cap, export completed and unfinished records; completing fewer answers is expected. Unfinished items remain missing, without automatic labels, completion weights or population extrapolation.

## Analysis

Validate packet hashes, answer/evidence fingerprints, task IDs, exact questions, label domains, timer bounds and reviewer context. Aggregate outputs retain hashes of every input label export and the executed analysis sources, plus the Python version; private absolute paths are not published. Match primary judgments from their original and changed-answer packets only for identical inputs. A changed answer, question or evidence set cannot inherit a reference judgment. Repeated matching references require adjudication.

Report paired answer correctness using correct/incorrect/unclear, and claim support using supported/unsupported/unclear. Retain unclear as its own category. Publish counts, confusion matrices, observed agreement and multiclass Cohen's kappa. Kappa is undefined for zero pairs or degenerate marginals; perfect constant-label agreement is not silently reported as kappa=1.

If at least ten paired answer clusters finish, use 2000 seed-19 bootstrap resamples of answers, retaining all their claim pairs together. Report nominal percentile 95% kappa intervals, defined and undefined resample counts, and the fact that intervals condition on defined kappa. Below ten clusters, leave the interval unestimated. Small selected samples and timed partial completion limit all agreement results.

Neither reviewer overwrites the other. Disagreements remain unresolved until a separate explicit adjudication. Agreement does not establish correctness, attribution recall, calibrated safety, Hindi support or population false-release risk. The selected cohort and fewer than 25 reviewed candidate releases retain their existing limitations.

## Reproduction and privacy

`python -m scripts.prepare_independent_review <new-private-root> <frozen-profile> <source-asset-root>` builds an immutable portable packet. `python -m scripts.analyze_independent_review --help` documents the paired import arguments. These tools make no hosted calls and use no GPU. Public synthetic regression tests exercise blinding, path confinement, changed inputs, repeated references, degenerate kappa and clustered intervals. Synthetic test ratings are not research labels.

Keep the handoff and raw exports private and outside Git. Only bounded aggregate results may be published after actual labels are returned and verified. No second-review outcome is claimed by this protocol.
