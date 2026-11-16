# Locked quality follow-up

Locked before fresh control outcomes on 2026-11-16. Scope: answer correctness, false language holds, stronger English controls, and reproduced defects that compromise those paths. No new infrastructure, paid services, hosted evaluation calls or expansion of the 21600 GPU-second ceiling. Historical results and private audit material remain unchanged.

1. Reproduce and fix upload filename failures and temporary-file leaks. Sanitize ingestion error responses. Add behavioral regressions.
2. Repair language routing using document context and explicit short-text handling. Never infer support from caller-supplied expected language. Keep Hindi, Hinglish and unmeasured languages gated. Evaluate foreign-language negative controls, including French, German and Finnish. Do not choose confidence thresholds from four external examples.
3. Add bounded deterministic checks for unsupported numeric values and explicit opposing numeric bounds. These detect narrow contradictions and are not entailment, authority verification or general entity verification. Measure both added holds and protection on identical saved outputs. Keep semantic support explicitly unverified.
4. Freeze a fresh English gold-input policy diagnostic, excluding historical question IDs, before inspecting routing outcomes. Public reference answers are controls, not newly human-verified labels or generated-reader accuracy. Keep this separate from saved-output replay and actual end-to-end results. No new generation is required to measure policy correctness.
5. Reproduce multi-column extraction failure and quarantine confirmed ambiguous reading order. No automatic OCR, header removal or text reordering without separate validation. Preserve offsets and invalidate changed preprocessing explicitly.
6. Refactor and format touched modules, expand useful lint gates incrementally, and retain runtime behavior tests. Do not rewrite historical sources or reformat the whole repository as a substitute for product work.
7. Publish paired counts, failures, negative controls and limitations. Run typing, regression, real-container and database checks as appropriate; commit and push focused verified changes.

Stronger reader, larger independently annotated samples, second-reviewer agreement and broad factual-risk calibration remain separate evidence requirements. Existing caps are not evidence that those requirements have been satisfied.
