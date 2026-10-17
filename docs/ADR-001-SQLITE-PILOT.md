# ADR 001: SQLite for the pilot corpus

## Decision

Use one SQLite database as the source of truth for documents, chunks,
embeddings, review decisions, and pilot events. Build FAISS and BM25 indexes
in memory from the database when the corpus version changes.

## Rationale

This keeps the pilot free, local, portable, and usable without Docker. It also
prevents stale pickled index files from diverging from stored chunk metadata.

## Consequences

The application is single-node and should run one process for a shared pilot.
It does not provide tenant isolation, distributed writes, or durable free-tier
hosting. A later multi-user deployment will move binary documents to object
storage and metadata plus vector search to a managed database.
