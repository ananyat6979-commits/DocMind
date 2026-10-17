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
