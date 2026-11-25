# CPU pilot stopped before the final English evaluation

The fresh English study did not advance to its final cohort. The ten-question CPU pilot hit its time limit after five completed generations, one interrupted case and four unserved cases. The precommitted timing rule supported only thirty final questions, below its minimum of forty, so the [frozen decision](cohort_decision.json) selects **zero final questions**. The separate 200-question candidate pool remains unused.

The design and original inputs were committed in `5b98590` before inference. The timing-only stop decision was committed in `a374702` before downloading or inspecting answer files. No failed generation was rerun and no reader, precision or budget was silently substituted. The original pilot used the unchanged pinned LocalReader and CitedReader methods, supplied with one complete public paragraph per question. This was not PDF ingestion, retrieval evaluation, hosted-default quality or a CPU service SLA.

## Timing and compute

The private free notebook provided four CPU cores and 33.66 GB RAM; inference used two threads and the pinned Qwen3-4B-Instruct-2507 model with FP16 parameters and SDPA. Five completed question times were 229.21 to 355.81 seconds. The interrupted sixth case had run for 107.15 seconds when the outer deadline stopped it. Its token counts and numerical outcome remain unknown.

The model and environment setup took 250.59 seconds. The worker's guarded run ended after 1740.80 seconds, reserving the remaining portion of the 1800-second pilot ceiling for shutdown. The study charges that full 1800-second reservation. No final CPU session, GPU allocation, hosted inference or paid fallback was started. The separate project GPU ledger remains unchanged at 18,600 of 21,600 GPU-seconds used.

The frozen sizing rule uses the larger of the longest attempted question and twice the median completed-generation question time: 502.26 seconds per question. After reserving twice setup time and 20 percent of the remaining 19,800-second CPU allowance, only thirty balanced questions fit. The declared forty-question minimum therefore stopped the experiment. This is a stop under this study's compute rule, not a claim that the platform forbids larger runs.

## Post-decision pilot diagnostics

Only after the stop decision was committed were model outputs retrieved. The pilot has one exact short-answer match among ten requested questions, or one among five completed requests. Nominal question-bootstrap intervals are [0%, 30%] and [0%, 60%], respectively. These tiny, timed, partially served samples cannot establish reader quality or a population failure rate. The requested metric mixes model behavior with compute non-service; the completed-only subset has a different answerability mix from the requested cohort. Do not read either as a model comparison.

Among completed requests, two valid short answers were scored, one answer was returned on a publicly labeled impossible question, and two auxiliary metric fields were invalid. No completed response had a JSON format failure, recorded output-cap truncation or observed nonfinite logits. The interrupted call's outcome is unknown. Four completed candidates were drafts and one was held; short-answer metrics do not label their full prose as human-correct or human-incorrect. No human answer or claim labels were created here.

The requested-cohort always-abstain control scores 50% on the balanced ten questions, while the five completed requests contain three unanswerable questions. Comparisons involving the five non-served requests mainly reflect service availability. Article-cluster intervals are not estimated: the pilot has only eight source articles, below the predeclared minimum of ten. The [numeric diagnostics](pilot_diagnostics.json) retain all requested cases and explicit missing states, without source text or raw completions.

## Reproduction and limits

See the [protocol](PROTOCOL.md), [fixed design](english_reader_design.json), [input identities](input_manifest.json), [timings](pilot_timing.json), [compute checkpoint](pilot_compute.json) and [reader identity](reader_identity.json). The public source reconstructs the original private input pool byte for byte using the frozen-manifest command. Raw paragraphs, reference spans and generated responses stay outside Git. Existing R1-R8 results are unchanged.

The larger independent English current-reader evaluation remains open. This pilot does not validate calibrated coverage, full-answer correctness, Hindi support or automatic release safety. It supplies feasibility and failure evidence; it does not close those research gaps.
