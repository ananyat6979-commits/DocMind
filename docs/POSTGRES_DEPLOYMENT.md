# Persistent single-host deployment

This opt-in profile uses PostgreSQL 17 and pgvector 0.8.7, two API worker processes, the isolated parser service, the centralized UI, and a shared private upload volume. Dense search reads the persistent HNSW index directly; ingestion and deletion update metadata, vector bytes and vector entries in the same transaction. It does not load the full corpus into each API worker. Scoped queries use exact filtered search. Each provisioned tenant gets a partial HNSW index. Only normalized 384-dimensional embeddings are admitted. Other dimensions require a deliberate schema migration.

On a Linux Docker host, create private credentials once:

```sh
python scripts/init_production.py --tenant example
chmod 700 .secrets/production
# Build, start the database, then migrate using the separate administrator identity.
docker compose --env-file .secrets/production/compose.env -f compose.postgres.yml build
docker compose --env-file .secrets/production/compose.env -f compose.postgres.yml up -d postgres
docker compose --env-file .secrets/production/compose.env -f compose.postgres.yml --profile tools run --rm migrate
docker compose --env-file .secrets/production/compose.env -f compose.postgres.yml up -d api parser ui
```

Read your initial key privately from `.secrets/production/api-keys.txt`. Replace the initial administrator with individual credentials in the private registry. Rotate the key after setup. The registry is read on authentication; adding a new tenant also requires migration provisioning and an API restart so its ingestion worker starts. Keep the environment file and key file outside Git and outside build contexts. They are ignored by both defaults. On Windows, set equivalent private ACLs before copying credentials to a deployment host.

The runtime database role cannot migrate, bypass row policies, or inspect the quota table. Every corpus, review, event and job operation runs with a server-derived tenant setting and forced row policies. Clients cannot choose a tenant using a request field or header. A compromised runtime process or arbitrary runtime SQL can choose that setting, so row policies do not confine a fully compromised service. Administrator database credentials are supplied only to the database and one-off migration service.

Authenticated quotas are shared in the database, with independent per-key and peer-IP limits. Authentication failures never allocate authenticated quota rows. A quota outage rejects requests rather than falling back to per-process counters. The quota function serializes short updates to keep its state bounded; load measurements must include that bottleneck. Forwarded IP headers are not trusted, so a reverse proxy aggregates clients unless a separate trusted-proxy design is implemented.

Ingestion claims have renewable 60-second leases and at most three attempts. Another process can reclaim an expired lease. Store transactions check the lease before work and before commit; an old claimant cannot commit after expiry. Execution can repeat, while deterministic file identities make corpus effects replaceable. Uploads must remain on the shared private volume. A live job is never failed merely because another API worker starts.

The checked-in Compose profile binds UI and API only to loopback. An operator supplies TLS ingress for external access, monitoring, database backups, backup restore drills and storage capacity planning. Two workers on one host are not regional failover. The profile does not migrate existing SQLite data automatically: re-ingest documents and transfer reviewed labels using reviewed export procedures. Do not copy source databases across tenants.

Transport and indexing changes do not validate answer accuracy. Reader profiles and historical evaluation artifacts remain separate. Do not enable automatic Hindi or other unmeasured-language answers because a deployment check passed.


## Backup procedure and bounded restore check

The `postgres` CI suite uses synthetic committed corpus vectors, pending reviews, events and a queued job. It runs `pg_dump -Fc`, restores into a fresh database in the same cluster, then verifies byte-identical vectors, review/job state and tenant denials through runtime credentials. The dump remains private; only a boolean summary is retained as an artifact. This tests the procedure, not a configured operator backup schedule or cross-cluster disaster recovery.

An operator can create a local private dump with:

```sh
mkdir -p .secrets/backups
chmod 700 .secrets/backups
docker compose --env-file .secrets/production/compose.env -f compose.postgres.yml exec -T postgres pg_dump -U postgres -Fc docmind > .secrets/backups/docmind.dump
chmod 600 .secrets/backups/docmind.dump
```

Protect and copy backups according to the deployment's retention and encryption policy. Restoring to another cluster also requires role provisioning and extension availability. Do not restore over a running production database without an operator-controlled recovery plan. Private credential registries and operator keys require separate protected backup and rotation procedures.
