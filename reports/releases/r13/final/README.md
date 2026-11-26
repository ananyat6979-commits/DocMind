# Paired English abstention results

The revised prompt increased valid abstention from **0 of 12 to 4 of 12** on unanswerable questions. The revised prompt still missed eight. Its overall exact match rose from **7 of 24 to 12 of 24**, tying the always-abstain baseline on this balanced cohort.

| Metric | Previous prompt | Revised prompt | Paired difference, nominal 95% interval |
|---|---:|---:|---:|
| Valid abstention on unanswerable questions | 0/12 | 4/12 | +33.3 points [+8.3, +58.3] |
| False abstention on answerable questions | 0/12 | 0/12 | 0 observed |
| Exact match on answerable questions | 7/12 | 8/12 | +8.3 points [0.0, +25.0] |
| Overall exact match | 7/24 | 12/24 | +20.8 points [+8.2, +37.5] |
| Overall token F1 | 37.0% | 53.9% | |
| Calls completed | 24/24 | 24/24 | |
| Format failures | 1 | 0 | |
| Invalid metric fields | 4 | 5 | |
| Generated tokens | 4,119 | 2,979 | |
| Summed attempt time | 1,696.3 s | 1,541.4 s | |

These are the 24 frozen English SQuAD2 questions reserved for the prompt comparison, separate from the 92-question R11 study. Each arm received the same supplied passage, one call per question, and balanced first-arm order within strata. The quantized reader, tokenizer, runtime and decoding controls are pinned in the [study design](../design.json). The revised arm changed the ordering and completeness of output instructions together, so the comparison cannot isolate either edit.

The primary calculation uses 2,000 question-level paired bootstrap draws. With only four discordant unanswerable pairs, the conclusion is sensitive to inference method. A two-sided exact sign test on those four pairs gives p = 0.125 under independent signs. No false abstentions were observed among 12 answerable questions. That sample is too small to bound the failure rate tightly. Among the revised arm's eight unanswerable misses, five were scored as attempted answers and three had invalid metric fields. The paired overall comparison against always abstaining had a nominal interval of -29.2 to +33.3 points.

This is a supplied-passage English reader experiment. Retrieval, full PDF answer correctness, human claim support, production release risk and the hosted reader remain outside this estimate. No new human labels were collected. The retry used 3,415.8 wall seconds including setup and was charged 3,420 CPU seconds. The initial failed attempt was charged 120 seconds. Total debit across the tracked CPU studies reached 16,500 of 21,600 seconds. This comparison used no GPU time or hosted calls.

[Numeric records and intervals](results.json) | [Compute record](compute.json) | [Attempt timings](terminal_timing.json) | [Artifact hashes](manifest.json)

Run the public consistency check from the repository root.

```bash
python -m scripts.verify_paired_abstention_cpu reports/releases/r13
```
