# Bounded calibration, robustness and infrastructure evidence

This follow-up closes implementation and measurement gaps within named profiles. It does not relabel historical results, validate the hosted provider or finish the separate Hindi PDF claim-label work.

## Calibration

The [document-disjoint HotpotQA component](../reports/releases/r3/calibration/README.md) completed 128 calibration and 128 held-out questions after a ten-question timing pilot. Provided article titles and paragraph hashes are disjoint across all selected questions. The grid, signals, error targets and family correction were frozen before held-out outcomes. Every frozen request completed. None of the six fixed-grid policies qualified at 10 or 20 percent exact-match error.

Token-confidence AUROC was 0.757 [0.655, 0.851]; the separate verification prompt was 0.562 [0.495, 0.623]. Those are nominal bootstrap ranking intervals with different feature-availability counts, not a paired improvement or calibrated release guarantee. The verifier assigned probability 1.0 to 96 held-out candidates, of which 64 failed exact match. The result exposes a concrete overconfidence failure. Public benchmark contamination in model training is not assessed, exact match can miss aliases, and a coarse frozen grid does not exclude every possible alternative thresholding method.

The zero-release policy entails 128 review decisions for 128 held-out requests. This is workload, not completed human annotation. Actual review time is [pending a validated export](../reports/releases/r3/review_cost/status.json). The private 20-minute English timing packet compares full-source display against source ordering and lexical highlighting without gold-answer hints. It has two practice and 27 sampled items, with missing labels and time-based stopping explicitly retained. This cannot validate Hindi attribution or general staffing cost.

## Robustness

The [broader reader/policy component](../reports/releases/r3/robustness/README.md) completed 180 synthetic cases across six questions, ten instruction families, two poisoning families and three benign controls under paired observation serializations. Current JSON serialization produced 1/60 contaminated injection candidates and 0/60 contaminated automatic releases; raw text produced 4/60 and 2/60. Only ten injection families were tested, so family uncertainty is wide. The observed improvement is narrow and exploratory. One direct numeric override still reached the current candidate, and raw role-spoofing passed NLI and citations twice.

This suite uses the actual pinned FP16 local reader and ReAct loop, fixed retrieval and the English NLI judge. It does not test hostile retrieval at corpus scale, cross-language attacks, arbitrary tool misuse or real-secret exfiltration. Parser isolation and API authorization have separate behavioral checks. They are not language-model attack success rates.

## Persistent service profile

The [opt-in PostgreSQL profile](POSTGRES_DEPLOYMENT.md) adds persistent transactional pgvector indexing, explicit tenant scopes, forced row policies, shared bounded quotas, per-tenant upload roots and renewable ingestion leases. Runtime credentials are separate from migration credentials. Corpus transactions reject stale leases; expired claims can be recovered by another worker. Synchronous upload and deletion use the authenticated tenant store. Newly generated responses cannot record already deleted sources. Operational counters are scoped to the current tenant and worker.

Live CI covers isolation for documents, chunk prefixes, neighbors, vectors, jobs, reviews and events; tenant-header spoofing; concurrent version updates; shared quotas across two API processes; terminated claimant recovery; and expiry during a transaction. A quota backend outage fails closed. A separate real-model HTTP profile verifies retrieval and stored NLI execution rather than inferring neural work from HTTP status. Load timing remains distinct from hosted generation.

A 50,000-paragraph CPU scale attempt timed out during embedding before any document writes. Its [failure record](../reports/releases/r3/scale/initial_cpu_timeout.json) is retained. An unchanged profile is rerunning with a 70-minute embedding/search step and 85-minute job cap on free CPU CI. No ANN quality or throughput number is claimed until that output is verified. Its real E5 vectors, exact-search comparison, scoped path, concurrency sweep and database restart are separate from historical Gaussian-vector and smaller-corpus experiments.

The checked-in Compose target is one host with two workers and a shared private upload volume. Multi-host object storage, regional failover, operator backup schedules and external availability remain outside its evidence. A restore procedure is checked separately with synthetic state. Passing CI is not proof of an operating production deployment.

## Reproducibility and cost

Source revisions, model revisions, dependency identities, requested denominators, failure records and numeric predictions are published for the two completed GPU components. Independent model-free verifiers run in CI. Human packets, raw labels, restricted source text and private review notes remain outside Git. No new hosted evaluation calls, paid annotation or paid compute were used. The exact free-account snapshot after both sessions completed was 3504.539 used seconds with zero active reservations; rounding upward and counting both allocated devices bounds cumulative use at 7080 GPU-seconds against the 21600-second ceiling. The repeated scale run uses free CPU CI.
