# Frozen-evidence local-reader control

Thirty-five fresh HotpotQA distractor questions used a 60/40 bridge/comparison mix, alternating arm order, identical initial evidence and no evaluation retries. Pilot timings fixed N before final outcomes. The pinned Qwen3-4B-Instruct-2507 reader ran FP16 with SDPA on a T4. Both allocated T4s count against the project ceiling.

| Arm | Candidate EM / requested | Completed-format EM | Format failures | Calls / correct | Tokens / correct | Seconds / correct |
|---|---|---|---|---|---|---|
| Single call | 5/35 (14.3%) | 5/31 (16.1%) | 4/35 | 7.0 | 9529.6 | 104.5 |
| ReAct protocol | 8/35 (22.9%) | 8/25 (32.0%) | 10/35 | 4.375 | 6984.4 | 117.6 |

The paired requested-EM difference is +8.57 percentage points, nominal paired bootstrap 95% interval [-2.86, +20.00] with 2000 resamples and seed seven. This does not establish a quality improvement or equivalence. Most importantly, every final question used one call in both arms: the run did not exercise multi-step retrieval or tool use. It is a frozen-evidence protocol/cost control, not evidence that adaptive retrieval helps or fails. Single-call remains the product default.

The initial 512-token pilot had four truncations in 20 attempts. The predeclared fresh second pilot used 1024 tokens, with zero truncations and three format failures in 20 attempts. The frozen final design retained 1024 tokens, with zero final truncations or numerical failures. The initial reader-identity snapshot still says 512; it is preserved alongside the final design rather than silently rewritten. Peak device-zero allocated memory was 9405160960 bytes. Setup, pilots and evaluation took 2587.998 seconds wall time, charged conservatively for both allocated devices.

Candidate EM includes held answers and measures agreement with public short-answer labels, not human correctness. Only 20 single-call and 17 ReAct candidates were visible drafts; respectively four and five of those matched EM. Those ratios are not false-release risk. English language holds and invalid quotes remain visible in the diagnostic counts. Format-failure raw completions were not saved, so detailed causes cannot be reconstructed. Later code fixes are not retroactively measured by this snapshot. The hosted default, Hindi generation and adaptive retrieval remain unmeasured here.

The source snapshot, numeric attempts, reader identity, pilot diagnostics, requested design and archive hashes accompany this report. Restricted article text and raw answers remain private. Public source: HotpotQA authors and underlying Wikipedia, CC BY-SA 4.0, pinned dataset revision in the input manifest; this is retrieval within supplied distractor contexts, not full Wikipedia.

Run `python scripts/verify_paired_core_profile.py reports/releases/r4/paired_core_t4` without model downloads. This is internal consistency verification, not external replication.
