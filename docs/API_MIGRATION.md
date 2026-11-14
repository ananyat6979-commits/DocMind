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


## E5 dense default

The measured English/Hindi component release defaults to pinned multilingual E5-small with dense retrieval. Existing BGE vectors, even with 384 dimensions, are a different representation. The identity also includes inference-library versions. `/ready` and `/query` return 503 when a nonempty corpus does not match the active identity; the original documents remain listed.

Keep the old store and authorized uploads for rollback. Create an isolated `DOCMIND_HOME` for the new corpus and re-ingest retained source files through the authenticated ingestion path. For Compose, stop the old stack while retaining its volumes, then start a separately named project such as `docker compose -p docmind-e5 up --build`. Re-upload authorized source files to its new API library. Historical responses and review labels remain associated with their original inputs.

`DOCMIND_RETRIEVAL_MODE=hybrid_rerank` explicitly enables the legacy-style search path. `DOCMIND_USE_RERANKER` controls whether that optional scorer is available; dense requests do not invoke it. Clear old `DOCMIND_MIN_RERANK_SCORE` values when switching modes. They cannot be applied to cosine scores. A configured reranker review threshold without reranker scores routes to review. Any new threshold requires a fresh calibration study; these component results do not validate a risk bound.


## Extraction quality policy v2

The policy now flags repeated long digit artifacts, while retaining ordinary numeric fragments and distinct identifier tables. Re-ingest retained uploads into a fresh isolated store to apply the new quarantine decisions. Existing production responses retrieved from stale or missing extraction-policy lineage require review. Re-uploading the same bytes does not silently reuse an old preprocessing cache; an explicit rebuild or isolated ingestion is required.

The new parser equivalence manifest is `reports/releases/r2/parser_equivalence_quality_v2.json`. Historical manifests and neural results remain unchanged. Quality metadata changed, so historical downstream artifacts are not automatically evidence for this policy.
