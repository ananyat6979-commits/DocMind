# Quantized CPU pilot supports a 92-question English evaluation

The pilot completed all twelve generations in 993.14 seconds on private free CPU compute, including 164.34 seconds of setup. The frozen rule used a conservative cost of 136.89 seconds per question and selected 92 questions within the remaining 16200-second reservation. This balanced cohort contains 46 answerable and 46 unanswerable questions from the unused SQuAD2 pool. The decision was committed in `833b959` before pilot answers were retrieved. See the [timing record](pilot_timing.json) and [cohort decision](cohort_decision.json).

The pilot achieved 25.0% exact match and 37.3% token F1. The nominal question-level exact-match interval was [0%, 50%]. Ten responses were released as drafts and two were held. Three auxiliary answer fields were invalid, and three answers scored as responses to unanswerable questions. The twelve cases are feasibility diagnostics. See the [numeric results](pilot_diagnostics.json).

The final run uses the same pinned quantized weights, CPU runtime, tokenizer, reader prompt and policy. It has been submitted with the frozen cohort. Final evaluation results are pending. See the [protocol](PROTOCOL.md) and [submission identity](final_submission.json).
