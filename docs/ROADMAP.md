# Roadmap: from solid project to defensible engineering artifact

Honest framing: no one can certify "top 0.0000001%". What separates a memorable project is that every claim has a number, every number has a method, and anyone can reproduce it. Do these in order; each step produces evidence.

## 1. Reproducible benchmark (1 day)
Add one public document so strangers can rerun your numbers (data/ is gitignored, so today nobody can reproduce the eval).
```powershell
mkdir data\public
# download a public PDF you are allowed to redistribute into data\public\, then:
python scripts/make_testset.py --docs data/public/paper.pdf --n 40 --unanswerable 10 --out data/testset.json
git add -f data/testset.json data/public/*.pdf
```
Target: at least 50 verified questions, at least 10 unanswerable. Fewer than this gives confidence intervals too wide to mean anything.

## 2. Evidence run (half a day, mostly waiting on rate limits)
```powershell
python scripts/ablation.py --testset data/testset.json --docs data/public/paper.pdf --chunkers
python scripts/evaluate.py --testset data/testset.json --docs data/public/paper.pdf
python scripts/update_readme.py
```
If the 70B judge hits the 100K tokens/day cap, rerun the same command tomorrow: it resumes. Or add a payment method on Groq for one cheap run.

## 3. Regression gate in CI
Commit `reports/ragas_latest.json`. Add a workflow step that runs `python scripts/evaluate.py --limit 15` on a schedule (needs `GROQ_API_KEY` as a repo secret) and fails if faithfulness or citation validity drops more than 0.05 from the committed value. This is how eval-driven development is shown, not described.

## 4. Deploy with real numbers
```powershell
docker build -t docmind .
docker run -p 8501:8501 -p 8000:8000 -e GROQ_API_KEY=gsk_xxx -v docmind-data:/data docmind
```
Docker is optional (see docs/CLOUD.md for the no-Docker path). Correction to an earlier version of this file: Hugging Face Spaces is no longer a free container host, since creating Gradio or Docker Spaces now needs a paid plan. For hosting, see docs/CLOUD.md. Then load test the API and record p50/p95:
```powershell
pip install locust
# locustfile: POST /query with 20 questions from the testset, 3 users
locust -f loadtest.py --headless -u 3 -r 1 -t 5m --host http://localhost:8000
```
Report p50/p95 and error rate. The expected bottleneck is Groq tokens per minute, not your code; the per-stage `timings` field will show it.

## 5. Experiments worth running (each is a table in the README)
- Query rewriting / HyDE vs plain query (retrieval ablation, free).
- `bge-small` vs `bge-base` vs `all-MiniLM-L6-v2` embedder (add `--embedder`; one corpus rebuild each).
- Reranker top-k (3, 5, 8) vs faithfulness and tokens per answer (cost curve).
- Calibrated abstention threshold: sweep `DOCMIND_MIN_RERANK_SCORE`, plot hallucination rate on unanswerable vs false abstention rate on answerable.
- Sentence-level groundedness check (NLI model) as a second guard behind the citation check.
- Contextual chunk headers (prepend document title and section to each chunk before embedding).

## 6. Data-engineering angle
Ingestion is already idempotent and hash-addressed. Make it a pipeline: watch a folder, ingest incrementally, record lineage (file hash, chunker version, embedding model) per chunk, and add a `doc ingest --since` backfill. Store chunk and event tables as Parquet and query with DuckDB for the pilot dashboard.

## 7. Operability
Structured JSON logs with `response_id` on every line; export `events` to Prometheus or Langfuse; add `/metrics`. Alert on abstention rate and review-queue age.
