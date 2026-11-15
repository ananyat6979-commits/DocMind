# Threat model

The service handles untrusted uploads, retrieved instructions, model output and unauthenticated network traffic. Authorized readers share a library within the local profile or their authenticated tenant. Personal accounts establish audit identity and operation permissions. The opt-in PostgreSQL profile derives tenant scopes from the server credential registry and enforces forced row policies across documents, chunks, reviews, jobs and events.

## Boundaries and controls

The API authenticates before allocating key quota state. Personal credential records contain token hashes, actor IDs and roles. Request bodies cannot choose trusted identities. Review and administrator permissions are independent. Response retrieval checks ownership and current review state. Pending/rejected answers and stale attribution from human edits are not released.

The UI delegates authority to the API and has no library volume or generation key. The API uses bounded uploads, reference-aware cleanup, a durable bounded queue and a single SQLite writer model in the default profile. The PostgreSQL option uses transactional persistent vectors, a shared quota function and expiring ingestion leases with commit-time fencing. Completed upload copies remain while referenced; interrupted unreferenced copies are eligible for a startup sweep after the grace period.

The separate parser receives bytes over a bounded Unix socket. Its container has no application volume or credential mount. It runs without network-family sockets, Linux capabilities or writable application files, and has bounded memory, CPU, process count, wall time and output. Environment construction uses an allowlist. The container workflows exercise isolation, recovery and transport equivalence. A local native parser is resource-limited trusted-input development mode and has no filesystem/network sandbox.

Document text is serialized as untrusted JSON. This serialization is not a prompt-injection proof. Unique quote location, citation disambiguation and human review add checks but cannot establish safety against arbitrary documents. The default shows eligible unverified drafts; quoted text can still contradict or fail to support a claim. NLI diagnostics cannot release or withhold a draft. Strict human approval is an administrator setting. Real evidence failures and unvalidated languages remain held; complete-document abstentions are visible as potentially missed retrieval. Model artifacts are pinned and remote model code is disabled in the local reader. Dependency and model revisions are part of fresh measurement identities.

## Residual risks

Container boundaries depend on the host kernel and runtime. The parser serializes work, so a malicious authorized client can occupy it until timeout. The default in-memory rate limiter and SQLite queue assume one API process; the PostgreSQL profile shares quotas and worker leases between processes. Invalid-key requests do not consume legitimate quota but still consume authentication CPU. There is no distributed abuse perimeter, external secret manager or general malware scanner. PostgreSQL tenant policies protect service operations, not a fully compromised runtime process that can issue arbitrary SQL and change its tenant setting. Administrator database credentials are separate from runtime credentials.

Extraction quality heuristics can quarantine readable content or miss corruption. Unicode ratios alone do not validate Hindi. OCR and legacy-font conversion are unsupported. Unmeasured languages require review. Short and Romanized Hindi text can remain undetermined.

NLI confidence is not a calibrated truth probability. Existing selective-prediction experiments did not establish useful safe automatic coverage. Small human subsets do not validate general multilingual attribution or claim-extraction recall. Hosted generation remains outside the local-reader profile's evidence.

An adversarial document can assert a false fact directly. Quote existence and high NLI entailment cannot authenticate that fact, and the historical suite includes successful raw role-spoofing cases. The original quoted-draft robustness profile completed 180 cases with 103 malformed JSON outputs; its low observed contamination therefore cannot establish a useful safe answerer. A complete-schema prompt correction has its own bounded prospective confirmation. Older release-gate results do not validate either new prompt. See the core evidence for source snapshots, utility and NLI-shadow tradeoffs. Human review is not a demonstrated safety guarantee.
