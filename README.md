# DocMind

Question answering over PDF, DOCX and text files, with hybrid retrieval, quoted evidence, claim-level attribution checks and a human review queue. This is a single-corpus pilot with published measurements and known failures.

## Measured results

Runs used available local compute and private free-tier T4 notebooks. No paid inference or compute was purchased. Intervals below are 95%; see the [technical write-up](docs/EVALUATION.md) for protocols, baselines, splits and limitations.

| Experiment | Held-out result | What it establishes |
|---|---|---|
| HotpotQA, dense BGE with headers, 200 questions | Recall@5 **94.25% [91.75, 96.50]** | Retrieval within the ten supplied distractor contexts |
| HotpotQA, hybrid plus reranker, same questions | Recall@5 **89.00% [85.75, 92.00]** | Reranking hurt support coverage in this setting |
| HotpotQA, 1.5B reader, dense top five | Exact match **20.0% [14.5, 25.5]** | High retrieval recall did not imply good answers |
| SQuAD2, 3B reader, 200 human-labeled questions | Exact match **57.5% [50.5, 64.5]** | Includes 100 answerable and 100 unanswerable items |
| SQuAD2, exploratory 7B reader, 200 fresh contexts | Exact match **66.5% [60.0, 73.0]** | Different held-out set; not a paired model comparison |
| CLIMATE-FEVER, selected span judge, 200 claims / 1,000 pairs | Attribution precision **72.7% [46.7, 100]**, recall **3.39% [1.28, 5.86]** | Human evidence labels, very low support recall |
| Selective prediction, both SQuAD2 studies | **0% automatic coverage** at calibrated 10%, 20% and 30% risk targets | No useful calibrated gate was demonstrated |
| Document injection, 50 cases per prompt | Attack success **12% [0, 32]** versus **30% [0, 60]** | Stronger prompt failed to improve this direct-reader test |

These are bounded experiments, not a full-wiki leaderboard, a multi-tenant deployment, or a safety certification. The review loop is implemented, but its human time savings have not been measured. [Records and provenance](reports/benchmarks), [provider protocol comparison](reports/protocols/summary.json), [local service load](reports/service_load.json).

## Run locally

Python 3.11 or newer is required. From this checkout:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pip install -e .
Copy-Item .env.example .env
```

Set the keys and passwords in `.env` privately. The UI requires `DOCMIND_APP_PASSWORD`; the API requires `DOCMIND_API_KEY`; review operations require `DOCMIND_REVIEWER_PIN`; document deletion and rebuild in the UI require `DOCMIND_ADMIN_PIN`. Missing credentials disable the corresponding access. Select a currently available hosted model with `python scripts/list_models.py`, then set `DOCMIND_LLM_MODEL`. `DOCMIND_SECRETS_FILE` can point to a private dotenv file outside the checkout. Never put secrets in a notebook or archive.

```powershell
streamlit run app.py
uvicorn docmind.api.main:app --host 127.0.0.1 --port 8000
python -m pytest -q
python scripts/verify_benchmark_reports.py
```

The test suite uses hash embeddings and a demo reader without downloading models. `DOCMIND_TEST_MODE=1` uses explicit development backends and must stay off in a deployed service. Real startup downloads the pinned retrieval and attribution models unless they are cached. Docker bakes those weights into the image; see [deployment](docs/DOCKER_CLOUD.md).

## How it works

Uploads are validated and parsed in a resource-limited subprocess. PDF pages, DOCX tables and text become chunks with stable document identity, source metadata and ingestion lineage. SQLite stores the corpus and embeddings atomically. Dense FAISS and sparse BM25 snapshots are built together; optional HNSW reduces dense search cost. RRF fusion and an optional cross-encoder rank candidates.

The reader can use ReAct or native tools. Citation labels must resolve to retrieved sources; ambiguous chunk prefixes and same-name document labels are rejected. Claim attribution selects one- or two-sentence evidence with exact offsets and applies NLI. A valid citation label alone is never presented as entailment. Failed or unsupported attribution routes to review, and pending answers are withheld by default.

The UI and API support approve, correct and reject decisions. Review exports remain separate from the frozen public test set. Structured request logs carry request IDs and timings; `/health` is public liveness, while `/ready`, `/metrics`, queries and the library require authentication. This is a shared authorization boundary, not per-user access control.

## Reproduce the evidence

```powershell
python scripts/materialize_public_corpus.py
python scripts/verify_benchmark_reports.py
python scripts/public_benchmark.py --device cuda --batch-size 32 --allow-download
python scripts/attribution_benchmark.py --device cuda
python scripts/qa_and_robustness.py --device cuda
python scripts/selective_benchmark.py --device cuda
python scripts/judge_comparison.py --device cuda
python scripts/critic_benchmark.py
```

The last three commands require GPU libraries beyond the lightweight test dependencies. See [free-compute instructions](docs/CLOUD.md) before running them. The public corpus downloader verifies source hashes and all 200 original human annotations. Experiments publish explicit source snapshots because some runs preceded later correctness fixes. Rerunning writes new outputs; keep the published artifacts unchanged or regenerate their manifest deliberately.

For your own documents, draft labels with `scripts/make_testset.py`, verify them with `scripts/verify_testset.py`, then use `scripts/evaluate.py`. Reference scoring is local by default. Failed questions remain in the denominator. Optional legacy judge integration is outside the supported dependency set.

## Limits and security

English-oriented retrieval and NLI; scanned PDFs need OCR. Hosted inference sends retrieved text to Groq. The default small attribution model has particularly poor measured recall. Prompt injection remains possible. ANN is optional, but corpus refresh still rebuilds in-memory caches, so storage and ingestion are not proven at service scale. Run one application process per corpus; the two Compose services use separate corpora. Synthetic local throughput does not predict hosted inference latency.

Read [security and retention](SECURITY.md), [evaluation and failure analysis](docs/EVALUATION.md), and [public data attribution](data/public/README.md).

## License

Application code: MIT. Public benchmark data has separate upstream terms, described in the data card.
