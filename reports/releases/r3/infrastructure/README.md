# Persistent-backend infrastructure checks

The first two-process HTTP profile completed 24/24 warm requests at concurrency four. Both independent API processes used the same PostgreSQL corpus, credential scope, upload directory and job queue. A job enqueued through one process was observed through the other. The query URLs alternate explicitly, so each process handled twelve warm requests. Actual stored metadata verifies 26 completed NLI executions, including one cold request per process. E5 retrieval and the pinned English attribution model were real; generation used the deterministic demo reader. There were no provider calls or model downloads.

Warm p50/p95 latency was 1022/1498 ms and completion throughput was 3.82 requests/second. Readiness took 1.07 seconds, cross-process queued ingestion 8.90 seconds, and cold query times were 0.48 and 7.84 seconds. These use one small synthetic document on a free CI runner. They do not establish a 50,000-document HTTP SLA or generative-reader latency. The quota was explicitly raised to 1000 requests/minute for the measurement. Do not compare these figures directly to the earlier single-process, lower-concurrency SQLite fixture test.

The frozen source was `9a47aa5d7cfc170f2b8883909191cd36a0c48f40`; its script hash was checked against Git before publication. The later socket-peer entry-point hardening is separately checked and does not change these original measurements. Further runs, if published, retain their own source identity.

The synthetic same-cluster restore procedure passed at source `6c7d85ca9225f2ac1d7841463e4d52126fcba889`. A consistent `pg_dump` restored into a fresh database preserved vector bytes, pending review state, a queued job, events and tenant denials through the runtime role. The dump and synthetic source text stay private. This is a tested procedure, not an operator backup schedule or cross-region recovery claim.

Live backend tests also cover forced row policies, runtime DDL denial, chunk prefix/neighbor isolation, concurrent corpus versions, two-process shared key/IP quotas, expired-worker recovery, commit-time lease fencing, source deletion races and tenant-bound checkpoint identities. Those are behavioral contracts, not language-model quality metrics. Source and workflow identities are in [provenance](provenance.json).

```sh
python scripts/verify_infrastructure_profiles.py reports/releases/r3/infrastructure
```
