# Quality follow-up: fixed inputs, explicit limits

This release repairs language routing, portable upload handling and observed PDF layout ambiguity. It adds narrow numeric-conflict holds. All measurements below use local CPU work; no new reader calls, hosted evaluation calls or GPU time were used. Historical artifacts remain unchanged.

## English routing controls

The initial fixed 100-question SQuAD 2.0 control improved holds from 35 to 28, which remained unsatisfactory. It is retained as development evidence. A separate 100-question confirmation was frozen before the revised fragment handling was tested. Its question IDs and paragraph hashes exclude historical records and the inspected control; articles may overlap. Each cohort contains 75 answerable and 25 unanswerable questions.

| Confirmation condition | Held reference inputs | Held answerable | Held unanswerable |
|---|---:|---:|---:|
| Frozen prior policy | 40/100 | 35/75 | 5/25 |
| Language repair only | 11/100 | 11/75 | 0/25 |
| Numeric checks only | 40/100 | 35/75 | 5/25 |
| Combined | 11/100 | 11/75 | 0/25 |

The remaining eleven cases all have repeated minimal quotes that fail the unchanged unique-location rule. Among the 89 citation-clean reference inputs, language-related holds fell from 29 to zero. This is routing with public reference answers, not generated-reader accuracy, a full-corpus retrieval benchmark or calibrated risk. The overall 5% hold target is not met. No threshold or cohort was changed after inspecting confirmation outcomes.

Public reference labels are used as published, without new independent human review. Source: [Stanford SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/), distributed under CC BY-SA 4.0. Raw source text is not redistributed here. The pinned source hash, fixed selection and numeric outcomes are published. Download the source through the runner to reproduce; use a checkout containing the frozen baseline revision.

## Same-output own-document replay

On the 17 primary PDF responses, the prior policy has nine drafts and eight holds. The combined repair has sixteen drafts and one hold. Human-correct held answers decrease from three to one on identical reviewed answer/evidence inputs. The remaining correct hold has an ambiguous quote. The two known incorrect drafts remain drafts: numeric checks do not solve semantic support or question relevance.

Five recovered answer/evidence inputs differ from the originally reviewed placeholders. Their old labels were not reused. A separate blinded review completed all five in 522.757 recorded running seconds: four answers were judged correct and one incorrect; ten claims were supported and one unsupported.

The [combined identical-input review](current_answer_human_review.json) now covers all 17 current PDF answers: 14 correct and three incorrect. Sixteen are visible drafts, including all three incorrect answers; one correct answer remains held for an ambiguous quote. Across 36 extracted claims, 32 were supported and four unsupported. These are unchanged reader completions with repaired parsing and routing, not a new generation experiment or proof of improved intrinsic reader accuracy. The two original incorrect drafts and one newly recovered incorrect draft show that the semantic correctness gap remains open. Fewer than 25 reviewed candidate releases still gives insufficient evidence about population false-release risk. This replay uses cached historical extraction and does not apply new layout geometry or regenerate answers.

## Runtime defects and limits

- Filenames are checked before writes for bounded UTF-8/UTF-16 length, forbidden characters and reserved device names. Temporary uploads are removed after read or promotion failures. Storage failures return a generic 503; ingestion responses do not expose raw parser exceptions.
- Numeric checks hold literal numbers absent from quoted evidence and explicit opposing bounds at the same value. Common exact time/length conversions are handled. These checks do not verify entities, arithmetic inference, authority, general negation or semantic entailment. Passing is not verification.
- A bounded PDF geometry check flags repeated within-line column jumps. Flagged pages are quarantined with original text preserved; unavailable geometry is explicit. Tests include an actual interleaved two-column PDF and native-worker postprocessing. This does not implement general layout reconstruction, table semantics, OCR or header removal.
- A fresh 92-page parser equivalence input has no observed layout warning. This is not proof that every page has correct reading order. Changed layout metadata receives a new transport manifest; prior neural artifacts are not declared extraction-equivalent.
- Existing chunk metadata stays readable. Previously cached PDFs need an explicit reparse to acquire layout assessment and refreshed document-language metadata. Cached text and DOCX recipes remain readable. No old stored answer is silently relabeled.
- Changed runtime modules are formatted and have additional F, B, I, E701 and E702 checks in CI. This is scoped cleanup, not a claim that the whole repository has adopted every lint rule.

## Reproduction

`python scripts/verify_quality_control.py reports/releases/r5` checks numeric consistency without models or private labels. To reproduce the frozen gold-input control, run `python scripts/quality_policy_control.py --design reports/releases/r5/english_policy_confirmation_design.json --out <new-directory>` after installing the light test dependencies. This downloads the pinned public source and performs language inference, but no reader generation. Output directories are never overwritten.

Human unsupported-claim calibration, a second reviewer, larger realistic PDF evaluation and the long-term coverage targets remain open. No benchmark here validates a nonzero safe automatic-release policy.
