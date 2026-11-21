# Fresh English supplied-paragraph reader control

This protocol tests the pinned reader on fresh SQuAD2 public-reference questions. It is a supplied-paragraph component, separate from PDF parsing, retrieval quality and human full-answer correctness. Original public labels are not independently reviewed here. Prior training contamination is unknown. The selected paragraphs are disjoint from 800 paragraphs used in tracked earlier studies; different paragraphs may share an article.

The frozen inputs contain ten pilot questions and a separate balanced 200-question candidate pool. The pilot uses only private free CPU resources, the existing pinned FP16 reader and actual quote validation and draft policy. No GPU, hosted calls or paid fallback is permitted. The six-hour CPU ceiling reserves 1800 seconds for the pilot and at most 19800 for a final run, including setup. This does not spend or expand the separate GPU ledger. The 600-second research request deadline is not the product default or a production latency claim.

Before inspecting model answers, correctness or routing, freeze the final size using the timing rule in [the design](english_reader_design.json). The decision includes the timing-file hash, canonical input-manifest hash and exact selected question IDs. An unavailable, odd, substituted or changed cohort fails validation. A timing-supported cohort below forty stops the final experiment. Pilot cases never enter final metrics.

## Reconstruct the committed inputs

Download the exact public source linked in the design. Its SHA-256 is checked before processing. Use `python -m scripts.english_reader_cpu prepare --source SOURCE_JSON --frozen-manifest reports/releases/r9/input_manifest.json --out NEW_PRIVATE_DIRECTORY`, replacing the two uppercase path arguments. The output directory must be new and outside tracked publication paths.

The frozen-manifest option reconstructs the recorded questions, paragraphs and public reference spans in their original order. It verifies the complete private pool byte hash. Recognized LF and CRLF JSON layout is supported to reproduce the initial Windows snapshot without changing escaped source strings or offsets. The actual study pool has been reconstructed byte for byte, not merely matched by question count. A changed source, paragraph, answerability flag or payload hash fails closed.

Calling `prepare` without a frozen manifest selects a new study while excluding currently tracked question IDs and their paragraphs. It does not reproduce an old study: once that study is committed, its own IDs are historical. Reproduction therefore always uses the frozen manifest. Raw source, reference spans and model responses stay in private ignored directories.

## Freeze and score

After the pilot is terminal, `python -m scripts.english_reader_cpu choose --timing TIMING_JSON --manifest reports/releases/r9/input_manifest.json --out NEW_DECISION_JSON` records the timing-only decision. It refuses to overwrite an existing decision. Commit that artifact before inspecting pilot quality or starting final generation. No final result is claimed by this protocol.

Failures, invalid auxiliary fields, explicit abstentions and unserved cases remain distinct. Blank or malformed responses cannot earn credit as correct abstentions. Candidate short-answer EM/F1 and completed-only metrics have separate denominators; a correct candidate may still be held. Always-abstain is the same-cohort control. [The pre-final supplement](uncertainty_supplement.json) adds article-cluster uncertainty and paired baseline differences. Neither nominal interval family establishes population release safety. Historical R1-R8 artifacts remain unchanged.

Source and license: [Stanford SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), CC BY-SA 4.0, with attribution to the dataset authors and underlying Wikipedia.
