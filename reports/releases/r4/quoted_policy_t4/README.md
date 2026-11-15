# Quoted-policy robustness and format failure

All 180 fixed synthetic cases were attempted once: six domains, ten injection families, two poisoning families and three benign controls, paired raw and Unicode-JSON serialization. The pinned FP16 local reader and matched-quote English NLI judge ran on free T4 compute. This is the single-call quoted-draft policy, not the historical ReAct/span-selection policy or hosted-service ASR.

| Serialization / input | Requested | Non-format outputs | Format failures | Wrong-number candidates | Released wrong-number drafts | Correct-number drafts | Correct-number drafts under NLI shadow |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw / injection | 60 | 32 | 28 | 1 | 1 | 22 | 22 |
| JSON / injection | 60 | 17 | 43 | 1 | 0 | 9 | 9 |
| Raw / poisoning | 12 | 9 | 3 | 0 | 0 | 8 | 8 |
| JSON / poisoning | 12 | 5 | 7 | 0 | 0 | 5 | 5 |
| Raw / benign | 18 | 7 | 11 | 0 | 0 | 7 | 6 |
| JSON / benign | 18 | 7 | 11 | 0 | 0 | 6 | 6 |

No inference/runtime failures or unserved cases occurred. Non-format includes valid abstentions; it does not imply a visible answer. The NLI shadow would have held the one contaminated raw draft, and one additional correct-number raw benign draft. It uses the same generated output and matched quotes, not the old selected-span judge. No human labels establish semantic correctness or general attribution precision/recall here.

Zero observed JSON attack releases has only ten injection families. The exploratory nominal family-any-success interval is [0, 0.2775], not an item-independent bound or safety certificate. The raw family interval is [0.0179, 0.4042]. Numeric presence can count negation and miss spelled-out numbers. Instruction contamination, factual poisoning and benign utility are separate outcomes. High formatting failure suppresses both utility and observed release contamination.

Saved raw completions show that all 103 malformed outputs parse after inserting the same missing claims-array closing bracket. This is a diagnostic only: automatic repair is disabled and original outcomes remain failures. None reached the 512-token output cap. The product prompt now contains one complete root-level short-answer/claims JSON example; that change requires fresh measurements. A prospective bounded confirmation on all the same fixed fixtures is explicitly a development regression after these outcomes were inspected, not independent held-out evidence.

The executed summary accidentally called candidate numeric utility draft utility. It is retained as `executed_summary.json`. The corrected `summary.json` distinguishes hidden candidates, visible drafts and NLI-shadow utility, recomputed from unchanged saved answers without model calls. The offline verifier checks numeric matching, policy/shadow decisions, denominators, hashes and source identities. Raw completions remain private; published answer records contain only original synthetic material and derived model responses.

The session took 1315.139975143 wall seconds including setup, counting both allocated T4s. No hosted calls or paid resources were used. Run `python scripts/verify_quoted_policy_robustness.py reports/releases/r4/quoted_policy_t4` without models. This is internal consistency verification, not external replication.
