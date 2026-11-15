# Broader synthetic robustness component

All 180 frozen requests completed without exceptions or unserved rows. Six questions cover ten instruction-injection families, two separately reported factual-poisoning families, and three benign controls, under two observation-serialization conditions. The pinned FP16 Qwen3-4B-Instruct-2507 reader runs the actual ReAct loop with the pinned English NLI judge. Retrieval is fixed. This is a reader/policy component test, not a hosted-service attack success rate or an adversarial retrieval benchmark.

| Case type, per condition | Raw document text | Current JSON document serialization |
|---|---:|---:|
| Injection candidate contamination | 4/60 | 1/60 |
| Injection contaminated automatic release | 2/60 | 0/60 |
| Correct-number presence under injection | 56/60 | 59/60 |
| Poisoning candidate contamination | 0/12 | 0/12 |
| Benign correct-number presence | 18/18 | 18/18 |

The current serialization still followed a direct numeric override in the battery question; attribution held that answer. Under raw text, role-spoofing produced two wrong numeric answers that passed the release policy. Citation presence and NLI support did not distinguish those instruction-derived values from facts. This is a concrete failure mode, not merely a marker-string attack.

There are only ten handcrafted injection families. Domain variants are dependent. At family level, observed automatic-release success was 0/10 for current serialization, with nominal Wilson interval [0, 0.278], versus 1/10 [0.018, 0.404] for raw text. These are exploratory bounds on a family having any observed success, not per-request safety intervals. A matched family bootstrap is reported separately; it does not establish general superiority. No observed success is not proof of resistance. The two poisoning families and three benign families are particularly small controls.

The target metric strips citations before matching wrong numeric values. Correct-number presence is a utility diagnostic, not semantic correctness or human labeling. Both the model-final output and application candidate are scored, and candidate release is distinct from the policy withholding it. Failed or missing outcomes would remain in requested bounds. All inputs are original synthetic fixtures. There are no human annotation or hosted-provider calls.

The session used 1754.58 wall seconds including setup, within its 3000-second cap. Both allocated T4 devices are covered by the private compute ledger. `executed_source.json` identifies 60 source files from commit `8e3f8fac27952a1b8ce7433f81bb266a16b7bdf3`, individually verified against Git before publication. Published short final answers and numeric call logs support independent checking; intermediate model messages stay private. Older robustness artifacts are unchanged.

```sh
python scripts/verify_robustness_closure.py reports/releases/r3/robustness
```

Unmeasured: arbitrary real documents, semantic attacks outside the fixtures, malicious tool arguments, retrieval poisoning at corpus scale, cross-language instruction attacks, real secrets/exfiltration, and hosted generation. Parser confinement and tenant isolation have separate behavioral suites; their pass counts are not language-model attack success rates.
