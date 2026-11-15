# Locked core-release plan

Locked 2026-11-15 before fresh neural outcomes. Scope is the document-answer product, fair controls and a visible demonstration. Only existing local compute, free CI and free Kaggle resources are permitted. No paid fallback, new framework dependency or unrelated infrastructure work is authorized. Audit notes, private human labels, secrets and restricted source text remain outside Git. Historical artifacts remain unchanged.

## Engineering first

Implement deterministic structured quotes and unverified cited drafts, then authentication-preserving async request execution with a 30-second deadline, bounded CPU workers and four hosted calls per process. Use request-owned state and accounting, disabled SDK retries and at most one interactive transient retry; quota errors fail immediately. NLI is off by default and diagnostic only. Optional strict mode requires human approval. Missing/invalid quotes, quarantined cited pages and unmeasured languages remain held. A clean-page answer may be shown with explicit missing-page context; a partial-document abstention remains held. A complete-document abstention is visible with a may-have-missed warning.

Short answers inherit only reliably English cited-file context, with contrary language signals vetoing release. Unknown short text additionally needs a source-identical fragment; numbers/English numeric units have explicit handling. Hindi and Hinglish remain held. This heuristic is not measured language accuracy. New empty corpora default to recursive chunking; existing recipes and old metadata stay readable. No automatic re-chunking or atomic re-embedding command is included. Representation changes require a separate corpus; library-version-only changes warn without rewriting vectors or historical provenance.

Build and test quote normalization before inference: unique case-sensitive literal match first, otherwise NFC plus Unicode-whitespace collapse and explicit soft-hyphen removal. Preserve literal hyphens, punctuation, escaped quotes and source offsets. Repeated occurrences are ambiguous and held. All offsets name their scope as retrieved chunk content. JSON preserves Hindi but explicitly escapes U+0085/U+2028/U+2029 and directional controls. Location is not claim entailment.

## Frozen local-reader experiment

Primary control: at most 100 fresh public HotpotQA distractor questions, 60 percent bridge and 40 percent comparison, excluding historical question IDs. The single-call reader and bounded ReAct loop receive the same initial five retrieved passages, with identical chunk content and deterministic quote checks. Alternate arm order, seed seven. This tests cost-benefit under frozen evidence, not the full value of adaptive retrieval. Run a separate diagnostic of up to 40 balanced questions with adaptive retrieval, reusing an identical captured baseline only. Report it separately.

Use Qwen3-4B-Instruct-2507 revision cdbee75f17c01a7cc42f958dc650907174af0554, FP16, greedy decoding and seed seven. This checkpoint is non-thinking-only; do not introduce an unsupported thinking flag. Ten timing-only pilot questions precede the frozen cohort. Start at 512 output tokens. If truncation occurs, permit one 1024-token pilot on fresh IDs, preserving the first pilot as failed/limited. Confirm finite logits, actual dtype, attention path and memory. Format failures, truncation and wrong answers are separate outcomes; no retries of evaluation failures.

Freeze cohort size using conservative pilot timings before inspecting evaluation outcomes, including setup, baseline capture, loop calls and fresh generation, with a 20 percent reserve. Main experiments reserve at most 4800 wall seconds over two allocated T4s, or 9600 GPU-seconds; Hindi diagnostics reserve at most 1800 wall seconds, or 3600 GPU-seconds. The closed-use upper bound is 7080 GPU-seconds. All use remains within 21600 GPU-seconds, and each checkpointed session is at most two hours. Verify free quota before submission. Account for both allocated devices even when one is used.

Report requested-cohort and completed-only EM, format failures, truncation, abstentions, calls, tokens and latency per correct answer, paired intervals and uncertainty/power sensitivity. Keep unserved rows as failures in requested denominators. No preordained two-times-token promotion rule. Single-call stays the default for this release; no universal loop conclusion follows from small samples or format failure. The hosted default remains unmeasured.

## Own-document and Hindi evidence

Prepare 30 private draft opportunities, half project guides rendered as PDF/DOCX and half NIST SP 800-63B-4, covering values, tables, facts, multiple sentences and unanswerable cases. Preorder authoring before model outputs. Remaining human budget is 1462.8131 seconds: allocate 19 minutes to question/evidence authoring and adjudication, three to disagreements and two as buffer. Expect 15 to 25 completed gold records, or fewer. Only actual user-reviewed questions, aliases and evidence receive human status. Freeze this gold before generation. English automatic checks and synthetic drafts are never described as human labels. Later disagreements remain separate diagnostics.

Hindi NCERT extraction/routing demonstrations and ASCII-versus-Unicode prompt token/runtime diagnostics stay separate from accuracy studies. Keep restricted pages/quotes private. Do not claim validated Hindi PDF attribution or pool Hindi with English accuracy. Unfinished human evidence remains unmeasured.

## Optional hosted confirmation and presentation

No new Groq evaluation calls are currently enabled. A future exploratory confirmation requires both an explicit CLI opt-in and DOCMIND_ALLOW_HOSTED_EVALUATION=1, only after committing local results. Freeze 100 paired questions, 500 attempted calls, one million conservatively charged tokens and two wall hours, with no retries or paid fallback. Dry-run order/hash and worst-case budget are prerequisites. Preserve unknown usage and unserved rows.

Create a truthful 75-second local demo, with no copyrighted NCERT extracts, credentials or fabricated neural results. README opening states what the product does, its unverified draft contract and what the evidence establishes. Ship focused commits after behavior, typing and provenance checks; publish only measurements actually completed within the caps.


## Pre-inference feedback incorporated

The structured output now includes a bounded short-answer field linked to displayed claim text. Score EM/F1 on that field and stratify by answer length; full-answer human correctness remains separate. Existing EM-risk findings are not factual-risk certification. New draft visibility is a product tradeoff, never a claim of greater safety. Compare NLI on/off as shadow decisions on identical new robustness outputs, and retain any lost protection. A new-policy 180-case run and labeled false-release checks remain required evidence; do not substitute old-policy numbers. Fewer than 25 reviewed releases warrant insufficient-evidence reporting, not a safety conclusion.

Shrink adaptive comparison B first, then the primary cohort, using pilot timings rather than outcome quality. The own-document preparation target is reduced to 30 pre-located draft opportunities, evenly divided across project guides and NIST; the expected completed human subset remains 15 to 25 or fewer. This is an end-to-end smoke test with wide intervals, not comparative validation. Changes add invisible Unicode-tag, zero-width-space and word-joiner escaping; seeded combining-text property tests; and a total parser receive deadline. Extraction and language checks remain fallible heuristics, including on legacy-font pages without hints.

Scoped vector search deliberately uses exact linear work in selected documents. Report selected chunk count, latency and timeout failures; do not present unscoped ANN throughput as scoped or end-to-end latency. Extra infrastructure beyond these correctness changes is deferred.

The bounded secondary reservation may prioritize the new quoted-policy 180-case robustness comparison before Hindi diagnostics. Both together remain within 1800 wall seconds and 3600 conservatively charged GPU-seconds; missing work remains unmeasured. This does not add compute to the project ceiling.
