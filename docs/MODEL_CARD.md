# Model and measurement card

DocMind retrieves document passages and applies citation and entailment checks before release or human review. It is an experimental document QA service, with English controls and a measured-language gate. Hindi support is a measurement objective, not a validated general capability. French remains gated.

The historical English stack uses pinned BGE-small, MS MARCO MiniLM reranking and an English MNLI entailment classifier. The English/Hindi candidate uses multilingual E5-small with required query/passage prefixes, mMARCO MiniLM reranking and multilingual MiniLM NLI. The local-reader candidate is pinned Qwen3-4B-Instruct-2507 in FP16 with deterministic decoding. Exact revisions are recorded in the runner and per-run manifests. These candidates do not replace the hosted default by assertion.

The multilingual NLI model was trained on XNLI development data. Independent component measurements exclude that split. XNLI sentence-pair performance does not establish claim-versus-PDF attribution, especially in Hindi. Public labels are not independently reviewed until the bounded user review is completed.

The diagnostic PDF pool uses fixed templates over explicit source spans. Proposed answers and exact location checks remain synthetic drafts. Human question validity, answer correctness and claim support are separate. The cohort is selected from conservative pilot time measurements before its outcomes are examined. Failures and unserved rows remain in the requested denominator.

Historical reports describe their own dependencies, inputs and task restrictions. They are immutable and do not validate changed tokenization, Transformers versions or inference settings. New CPU/CUDA locks and representation identities require fresh baselines. ANN overlap against exact neighbors measures index fidelity, not semantic QA quality.

The project code license does not grant commercial clearance for all model training data. XNLI has noncommercial terms. mMARCO declares Apache-2.0 while upstream MS MARCO specifies noncommercial research use. The model cards, upstream datasets and individual PDFs must each be checked. Restricted raw text, translations and human annotation files stay outside Git until redistribution rights are established.

Human review is bounded to 90 minutes for PDF work and a separate 30 minutes for public labels. English controls receive automatic checks only. Fewer than 25 reviewed candidate releases yields insufficient evidence about false-release risk. Exploratory claim labels cannot validate general Hindi attribution or claim-extraction recall. Single-reviewer consistency is not inter-annotator agreement.

See [release gates](RELEASES.md), [evaluation protocol](EVALUATION.md) and [threat model](THREAT_MODEL.md). Missing measurements stay explicitly unmeasured.
