# Paired English prompt comparison on the quantized CPU reader

This study uses the exact 24 previously frozen R12 pairs and both prompt hashes. The T4 attempts produced no answers. The CPU engine, weights, tokenizer and generation parameters match the successful R11 run. Both arms see the same supplied passage and question. First-arm order alternates within the answerable and unanswerable strata. No reference text is sent to the generator.

The primary endpoint is valid explicit abstention among 12 unanswerable questions. Compare paired counts and report a nominal paired question-bootstrap interval. Report false abstention and short-answer exact match among 12 answerable questions, overall candidate exact match and token F1, completion, format outcomes and release routing. Include missing requests as errors in the requested denominators, and keep timed-out token counts unknown. The sample supports a small supplied-passage comparison with broad uncertainty.

The free CPU reservation is 7200 wall seconds including setup. The worker deadline is 7140 seconds. Earlier CPU pilots are charged their full 5400 seconds of reservations. The closed R11 final job is charged 7560 seconds after rounding observed runtime upward to the minute. The prior debit plus this reservation is 20160 seconds against the six-hour CPU ceiling. GPU and hosted usage are both zero. Prior study records retain their original reservation accounting.

The paired comparison estimates prompt effects for this quantized runtime and selected public-label sample. It leaves hosted quality, claim support, production release risk and Hindi evaluation separate. Publish only aggregate metrics, numeric records and private artifact hashes.
