# Offline FP32 batch profile on free CPUs

This is a separate execution profile of the pinned Qwen3-4B-Instruct-2507 reader. It changes model parameter precision to FP32, offline batch size to four and thread count to four. It does not replace the stopped R9 FP16 single-sequence study, provide a paired precision-quality comparison or validate the hosted default.

The original system prompt, source serialization, quote checks and draft policy are retained. Selected complete-document metadata follows the real store schema, including the detected source language when the reader abstains. A supplied paragraph is the complete query scope. PDF ingestion, retrieval quality and real API latency are outside this component.

## Inputs and labels

The final pool is the same balanced 200-question pool that R9 never generated. The new twelve-question pilot (three complete batches) uses different source paragraphs, excluding all prior planned and attempted contexts. Source bytes, input identities and the unused-pool condition are checked. Source/reference strings stay private. Public SQuAD2 labels have not been independently audited here and possible pretraining contamination is unknown.

The frozen-manifest reconstruction command in `scripts.english_reader_cpu` also reconstructs these inputs byte for byte from the pinned public source. The private input-pool hash must match before inference. The [design](english_batch_design.json) and [input manifest](input_manifest.json) are frozen before model execution.

## Compute and stopping

The original cumulative CPU-study ceiling remains 21600 seconds: R9 reserved 1800, this new pilot reserves 1800 and a possible final session receives at most 18000. All limits include environment setup and model loading. No GPU, hosted-model calls, paid fallback or new human annotation is used.

After the pilot, inspect timing only. Require at least eight completed native generations in fully finished batches. Use 1.25 times the worst attempted batch time per requested sequence, including failed or interrupted batches. Reserve twice setup plus 20 percent of the remaining final-session allowance. Select a balanced prefix in complete batches, at most 200 questions. A supported size below forty stops final evaluation. Commit that decision before inspecting pilot answers, correctness or routing. No failed generation is rerun.

Batch calls and logical sequence completions have different units. EOS padding is excluded from generated-token counts. All raw completions are checkpointed before assembly. A native batch failure affects the batch; it does not establish that every sequence had nonfinite logits. Interrupted counts and unknown outcomes stay unknown. These are offline throughput measurements, not a production concurrency or latency claim.

## Scoring and uncertainty

Candidate short-answer EM/F1, answerability strata, failed/invalid fields, cap reaches and missing requests remain separate. Blank outputs and errors cannot become correct abstentions. Requested-cohort and completed-only metrics retain their denominators. Always-abstain is evaluated on the same requested questions, with a paired difference. Use the predeclared nominal question bootstrap and whole-article sensitivity, including the ten-article minimum. Human full-answer correctness, claim-extraction recall and population release risk are not inferred from these automatic metrics.

No default model, gate or language-support change follows automatically from this study. R1-R9 artifacts remain unchanged. See the design for exact model revision, prompt identity, source license and environment-lock hashes.
