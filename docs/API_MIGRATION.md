# API migration from v1 to v2

## Ingestion

`POST /ingest` now accepts only paths below `DOCMIND_HOME/data`. Use
`POST /upload` for multipart document uploads. Ingestion returns `file_ids`,
`chunks_added`, and `cached` instead of only a chunk count.

## Querying

`POST /query` accepts an optional `file_ids` list to scope retrieval. Its
response now includes a response ID, abstention status, review status, review
reasons, citation validity, cited-source flags, and LLM call count.

## Review and feedback

`POST /feedback` records useful or not useful feedback. `GET /review/pending`
and `POST /review/{response_id}` expose the review queue. Configure
`DOCMIND_API_KEY` and `DOCMIND_REVIEWER_PIN` before exposing these endpoints.

## Compatibility

Clients written for v1 must update their response parsing and use `/upload`
when they previously sent arbitrary server paths to `/ingest`.

## Authentication and pending answers

Missing API configuration now returns 503, and an invalid `X-API-Key` returns 401. Review access additionally requires `X-Reviewer-Pin`. Pending answers are withheld by default; clients must inspect review status rather than treating an empty pending response as an approved answer. Source entries include `file_id` and disambiguated labels when filenames collide.

## Background ingestion and operations

Use `POST /uploads/jobs` for a multipart upload or `POST /ingest/jobs` for an allowed server path. Both return 202 and a job ID. Poll `GET /ingest/jobs/{job_id}`; queued jobs can be cancelled with DELETE. Existing synchronous routes remain available. `/health` reports liveness only. `/ready` and `/metrics` require the API key. Request IDs are returned and correlated with structured logs.

## Corpus migration

Embedding signatures now include model revision. Existing corpora without compatible lineage must be rebuilt into an isolated directory, then switched over deliberately. Changing chunker configuration does not silently reuse cached chunks. Back up authorized data before migration.
