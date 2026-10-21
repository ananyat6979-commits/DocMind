# DocMind

Question answering over your PDF, DOCX and text files. Hybrid retrieval, cross-encoder reranking, a from-scratch ReAct agent, **verified citations**, and a **human review queue**.

> Status: the measured results below are filled in by scripts from real runs. A section that says "Not yet run" means exactly that.

## What is different from a generic RAG demo
- Every `(Source: file, p.N)` in an answer is checked against the passages actually retrieved. Invalid citations are removed and counted.
- The agent abstains (`NOT_FOUND`) instead of guessing, and abstentions go to a human.
- Answers are routed to a review queue by policy (uncited, invalid citation, low confidence, abstained, plus a random audit sample). Reviewer corrections export back into the evaluation set.
- Multi-document library with per-query scoping. Ingestion is hash-addressed and idempotent.
- Evaluation is built in: RAGAS with a Groq judge, a free retrieval ablation, bootstrap confidence intervals.

## Architecture
```
file -> loader (PDF pages / DOCX sections+tables / TXT)
     -> semantic chunker (NLTK + bge-small, percentile breakpoints)
     -> embeddings + chunks stored in SQLite (single source of truth)
                  |
                  +-- derived in-memory caches, rebuilt when corpus_version changes:
                      FAISS IndexFlatIP (dense)    BM25 inverted index (sparse)
                              \                      /
                               RRF fusion (k=60) -> cross-encoder rerank -> abstention gate
                                                |
                    ReAct agent (search_documents, get_chunk_context), Groq-hosted LLM
                                                |
                    citation verifier -> abstention decision -> review policy
                                                |
              Streamlit UI / FastAPI  <->  review queue + usage events (SQLite)
```

## Setup (Windows PowerShell)
```powershell
cd D:\Code\Projects\MachineLearning\DocMind
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt      # runtime + evaluation + tests
pip install -e .
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
copy .env.example .env      # then put GROQ_API_KEY=gsk_... in .env
```
Mac/Linux: `source venv/bin/activate` and `cp .env.example .env`.

## Run
```powershell
streamlit run app.py                                   # UI
uvicorn docmind.api.main:app --reload                  # API, docs at /docs
python scripts/demo.py --doc data\your.pdf             # CLI
python -m pytest -q --ignore=tests/test_ragas_plumbing.py   # 70+ offline tests, no downloads, no API key (use python -m, not bare pytest)
```
Offline demo without a key: `$env:DOCMIND_TEST_MODE="1"` (hash embedder, extractive demo LLM, never for evaluation).

## No Docker, or a weak laptop?
- Container without local Docker: GitHub Actions builds and tests the image, Codespaces runs it: [docs/DOCKER_CLOUD.md](docs/DOCKER_CLOUD.md).
- Heavy steps (ingest, ablation, RAGAS) free on Kaggle or Colab, results copied back: [docs/CLOUD.md](docs/CLOUD.md) and `notebooks/docmind_cloud.ipynb`.

## Evaluate (run in this order)
```powershell
python scripts/make_testset.py --docs data\your.pdf --n 40 --unanswerable 10 --out data\testset.json   # you verify each item
python scripts/ablation.py --testset data\testset.json --docs data\your.pdf --chunkers                  # free, no LLM
python scripts/evaluate.py --testset data\testset.json --docs data\your.pdf                             # RAGAS, Groq judge, resumable
python scripts/stack_check.py                                                                           # proves the real models run
python scripts/record_metrics.py                                                                        # writes reports/METRICS.md
python scripts/update_readme.py                                                                         # writes the blocks below
```
Only human-verified questions are used by default (`--include-unverified` for trial runs). Groq's free plan has small per-minute and per-day token caps that depend on the model (see console.groq.com/docs/rate-limits), and hosted models are retired from time to time, so model ids are configuration: run `python scripts/list_models.py`, then set `DOCMIND_LLM_MODEL` and `DOCMIND_JUDGE_MODEL`. `evaluate.py` checkpoints and resumes, so a daily cap only delays you.

### Stack verification
<!-- STACK:START -->
Not yet run. Execute `python scripts/stack_check.py`.
<!-- STACK:END -->

### Results: RAGAS and answer-level
<!-- RAGAS:START -->
Not yet run. Execute `python scripts/evaluate.py --testset data/testset.json`.
<!-- RAGAS:END -->

### Results: retrieval ablation
<!-- RETRIEVAL:START -->
Not yet run. Execute `python scripts/ablation.py --testset data/testset.json --docs <your docs>`.
<!-- RETRIEVAL:END -->

### Pilot
Not yet run. Protocol in [docs/PILOT.md](docs/PILOT.md); requirements and success metrics in [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).

## Human review
Flags: `abstained`, `no_citations`, `invalid_citation`, `low_retrieval_confidence`, `no_clean_final_answer`, `random_audit` (rate `DOCMIND_REVIEW_AUDIT_RATE`, default 0.10). Reviewers approve, correct or reject in the Review tab or via `GET /review/pending` and `POST /review/{id}`. Set `DOCMIND_REVIEWER_PIN` to protect it. Approved and corrected items export to `data/reviewed_testset.json`.

## Configuration (environment)
`GROQ_API_KEY`, `DOCMIND_LLM_MODEL`, `DOCMIND_REASONING_EFFORT`, `DOCMIND_MAX_TOKENS`, `DOCMIND_CHUNKER`, `DOCMIND_HOME`, `DOCMIND_MIN_RERANK_SCORE`, `DOCMIND_REVIEW_MIN_SCORE`, `DOCMIND_REVIEW_AUDIT_RATE`, `DOCMIND_REVIEWER_PIN`, `DOCMIND_API_KEY`, `DOCMIND_JUDGE_MODEL`. All other knobs are in `docmind/config.py`.

## Limitations
Scanned PDFs need OCR first. English text only. Free-tier Groq sends retrieved text off-machine, so do not use confidential documents. SQLite is single-node. Full list: [docs/AUDIT.md](docs/AUDIT.md). Next steps: [docs/ROADMAP.md](docs/ROADMAP.md).

## License
MIT
