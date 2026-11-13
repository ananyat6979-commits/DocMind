# Security and deployment boundary

DocMind is a single-user or shared pilot, with one process per SQLite corpus. It has shared API, reviewer and administrator credentials, not tenant isolation or per-document ACLs. Use TLS at a reverse proxy before remote access. Bind the native API to loopback until that boundary is configured. Do not use confidential documents with the hosted reader: retrieved passages leave the machine.

## Access and secrets

The API fails closed without `DOCMIND_API_KEY`. Send it in `X-API-Key`. The UI requires `DOCMIND_APP_PASSWORD`; library mutations additionally require `DOCMIND_ADMIN_PIN`. Review operations require `DOCMIND_REVIEWER_PIN`. Set `DOCMIND_TEST_MODE=0` in deployments. Test mode bypasses the UI password and live models; the API still requires its key.

Keep credentials in an ignored `.env`, a private `DOCMIND_SECRETS_FILE`, or host secrets. Never include keys in Git, notebooks, source archives or logs. Revoke exposed keys. The deployment image excludes environment files and local state. Rotate shared credentials by updating the secret and restarting the service. Client-supplied user names are not trusted reviewer identities.

## Untrusted input

Uploads are limited to 50 MiB and validated by type, signature and extracted content. PDF parsing is limited to 500 pages, 200,000 characters per page and 8 million total characters. DOCX decompression is bounded by entry count, total expanded bytes, per-entry bytes and compression ratio. A separate parsing process has a 512 MiB memory limit, 45 seconds CPU and 60 seconds parent wall timeout. These controls contain common resource exhaustion cases; they do not establish that every malicious parser input is safe.

Document text is untrusted. Citation identity and NLI checks cannot make prompt injection impossible. The measured injection suite failed, with wide confidence intervals; see [the experiment](docs/EVALUATION.md). Unsupported attribution or judge failure requires review, with pending answers withheld by default. Human approval can still be wrong.

## Operations and retention

The async ingestion queue is durable and bounded to 20 waiting jobs. Requests have bounded bodies and text fields, request IDs, structured timing logs and per-key quotas. Quotas and metrics are local to one process and reset on restart. They are not a distributed denial-of-service defense. Health is a public liveness endpoint; readiness, metrics and library access require credentials.

SQLite contains document text, embeddings, review content and telemetry. Protect the data directory and backups with OS permissions and encryption where appropriate. Deletion removes the document rows, associated review content and owned upload copies. Secure deletion is enabled for SQLite rows, but external backups, filesystem snapshots, transaction journals, logs, memory and the hosted provider may retain copies. There is no automatic retention scheduler or certified erasure. Use an explicit retention policy for any pilot.

## Vulnerability reports

Use the repository's private vulnerability reporting option when available, or contact the owner through their GitHub profile to arrange a private channel. Do not place credentials, private documents or exploitation details in a public issue. Include the affected revision, a minimal harmless reproduction and expected versus observed behavior.
