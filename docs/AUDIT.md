# DocMind audit: bottlenecks and gaps

Reviewed against commit `abd4c51` of the original repository (about 2,000 lines of Python). "Fixed" means the change is in this repo with a test. "Measure" means it needs your data or your Groq key.

## A. Correctness bugs

| # | Where | Problem | Status |
|---|---|---|---|
| 1 | `agent/react.py` L133 | Dangling `else` pairs with `if final_answer is None`, so every successful answer was overwritten with "I was unable to find...". Reproduced against the original code. Also made RAGAS score failure strings. | Fixed (`test_successful_answer_is_not_overwritten`) |
| 2 | `agent/react.py` L121 | `elif all_sources` attached to `if reasoning_trace`, unreachable; `last` undefined path. | Fixed (single `_decide_answer`) |
| 3 | `agent/react.py` L78 | Accepted a Final Answer with zero retrieved sources on the last iteration, i.e. answering from memory. | Fixed (ungrounded answers become abstentions) |
| 4 | `ingestion/pipeline.py` L93 | `ingest()` returned early if `faiss.index` existed, so a second upload reused the first file's index. | Fixed (hash-keyed multi-document store) |
| 5 | `pipeline.py` / `dense.py` | SQLite (all rows, random UUIDs, duplicates on re-ingest) and the pickled index could disagree. | Fixed (SQLite is the single source; indexes derived) |
| 6 | `app.py` L84-88 | "Rebuild index" checkbox stayed checked, so every rerun re-ingested and wiped history. | Fixed (button) |
| 7 | `agent/tools.py` | Empty, yet `get_chunk_context` was advertised in the prompt. | Fixed (implemented, neighbour lookup in document order) |
| 8 | `retrieval/sparse.py` | `scores > 0` with `BM25Okapi`: negative IDF on small corpora returned nothing. Tokenizer left punctuation attached ("data," != "data"). | Fixed (own BM25, Lucene IDF, regex tokenizer) |
| 9 | `ingestion/chunker.py` | A tiny leading fragment was emitted alone below `min_chunk_size`; one oversized sentence was silently truncated by the 512-token embedder. | Fixed |
| 10 | `ingestion/loader.py` | `.doc` mapped to a loader that cannot read it; DOCX tables ignored; scanned PDFs vanished silently. | Fixed (clear errors, tables flattened) |
| 11 | `README.md` | `pip install -e .` with no `pyproject.toml`; unpinned `ragas>=0.1.14`. RAGAS 0.4.3 fails to import against current LangChain. | Fixed (pyproject, pins) |
| 12 | `evaluation/harness.py` | No judge configured (RAGAS falls back to its default provider and fails without that provider's key); `result["metric"]` changed meaning across RAGAS versions; Groq rejects `n>1` used by answer relevancy. | Fixed (Groq judge, `to_pandas`, `bypass_n`) |

## B. Security

| Problem | Status |
|---|---|
| `/ingest` accepted any server path; ingest then query read arbitrary files | Fixed: restricted to `data/`, plus `/upload` with size and type limits |
| Pickle files loaded from disk (code execution if the folder is tampered with) | Fixed: no pickles |
| Model output rendered with `unsafe_allow_html=True` (document-borne HTML injection) | Fixed |
| Prompt injection via document text | Mitigated: observations declared untrusted in the prompt, citations verified. Not eliminated. |
| No auth on the API | Optional `DOCMIND_API_KEY`, `DOCMIND_REVIEWER_PIN` |

## C. Performance bottlenecks

| Bottleneck | Evidence | Status |
|---|---|---|
| Groq free-tier token caps are small and model-specific; the original README claimed 100K tokens/min, and the models it named were shut down on 2026-08-16. | console.groq.com/docs/rate-limits, /docs/deprecations | Retries honour `Retry-After`; fail fast on daily cap; model ids are configuration with a clear `model not found` error listing available models |
| Prompt grows each ReAct step; same chunks re-sent on repeated searches | code | Fixed: seen chunks listed by id only |
| Reranker model reloaded per `HybridRetriever()` (every upload) | code | Fixed: process singleton |
| Every sentence embedded for chunking, then every chunk embedded again; rebuild re-embedded | code | Rebuild no longer re-embeds (stored vectors). Sentence-then-chunk double pass remains: Measure |
| BM25 scoring looped in Python over all docs | code | Fixed: inverted index |
| One MLflow run per query, importing mlflow at startup | code | Moved to eval runs; per-query data in SQLite events |
| Per-stage latency unknown | none recorded | `AgentResponse.timings` now splits `llm_ms` and `retrieval_ms`. Measure p50/p95 via pilot report |

## D. Measurement gaps (claims without evidence)

| Claim | Why it is unsupported | How it is now measured |
|---|---|---|
| "40-60% precision improvement" from the reranker | No experiment existed | `scripts/ablation.py`: precision@5 dense vs hybrid vs hybrid+rerank, paired bootstrap CI |
| "no hallucinated sources" | Sources shown were retrieved, but citations in the answer text were never checked | `agent/citations.py` verifies every citation; report `citation_validity` |
| Semantic chunker "better" | Never compared to fixed | `scripts/ablation.py --chunkers` |
| RAGAS scores | Never run; testset is 5 questions about a document not in the repo | `scripts/make_testset.py` (human-verified), `scripts/evaluate.py` |
| bge-small "outperforms MiniLM" | MTEB is not your corpus | Optional: add `--embedder` ablation (see ROADMAP) |

## E. Remaining gaps

- Scanned PDFs need OCR (`ocrmypdf`). Tables inside PDFs are plain text only.
- Small-model compliance with the ReAct format is imperfect; the review queue exists because of that.
- Abstention threshold `DOCMIND_MIN_RERANK_SCORE` needs calibration on your labels (the ablation prints a suggestion).
- Single-process SQLite is fine for a pilot; not for dozens of concurrent writers.
- Free-tier Groq sends passages off-machine.
