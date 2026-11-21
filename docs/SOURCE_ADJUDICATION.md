# Source adjudication after paired review

This follow-up is separate from the frozen second-review protocol and R7 agreement snapshot. The first and second reviewers discuss every completed paired case with an answer or claim disagreement against the source. A third reviewer is not required. Three selected cases contain two answer disagreements and three claim disagreements; all nine claim units remain visible for context.

The private packet allows at most 600 additional recorded seconds. The prospective total allowance is 9600 seconds, with 8944.9269 seconds already charged. Only actual exported time is charged. Packet creation, synthetic browser checks and automatic validation create no human labels.

Participants declare joint discussion or single-person reconsideration before starting. Single-person work cannot be described as joint consensus or an independent third judgment. Prior votes are visible, so neither mode is independent review. Answer correctness uses the full source; claim support uses the exact displayed evidence. Unclear is a valid outcome. Each completed case requires a short source-based rationale. Unfinished cases remain missing.

`python -m scripts.analyze_source_adjudication --help` describes the additive importer. Its private originals manifest maps first-original, first-changed and second to their packet root and raw labels path. The importer checks original export hashes, packet hashes, exact question/answer/evidence identity, the original votes, complete disagreement selection, label domains, participant declaration, rationale and time limits. Supply the immutable prior budget with `--budget-before` and the separately authorized total with `--allowance-seconds 9600`.

The output path must be new. Original exports and R7 agreement results are never rewritten. Any later public adjudication summary is additive and reports unfinished or unresolved cases. Do not recompute independent agreement after consensus, count these cases as new questions, or infer population release safety from this disagreement-selected subset. Keep the packet, source files, rationales and raw labels outside Git.
