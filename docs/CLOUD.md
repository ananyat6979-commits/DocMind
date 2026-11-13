# Reproduce on free compute

The published runs used private Kaggle T4 notebooks with internet enabled and no billing upgrade. They uploaded an allowlist of source files and public benchmark scripts, not private documents, environment files or credentials. The experiment provenance is in `reports/benchmarks`; availability and quotas are account-specific.

## GPU experiments

Use a fresh private notebook and a clean checkout. Install the runtime requirements plus `pandas`, `pyarrow`, `accelerate` and `matplotlib`. The quantized 7B follow-up additionally requires `bitsandbytes>=0.50.2`. Use a compatible CUDA PyTorch wheel supplied by the notebook runtime. The exact recorded versions for the first run are in `reports/benchmarks/runtime-packages.txt`; the model and dataset revisions are in each summary. Stop if CUDA is unavailable rather than silently running a different experiment.

Run the commands in README in dependency order: public retrieval first, initial attribution and direct reader next, selective study, span judge comparison, then the exploratory critic follow-up. Scripts write under `reports/hotpot`, `reports/attribution`, `reports/qa_robustness`, `reports/selective`, `reports/judges` and `reports/critic`. Inspect their output paths and source hashes. Published records are copied under `reports/benchmarks` with corpus text omitted and source identities hashed. That publication transformation does not alter labels or measured outputs. Preserve the original private output archive separately.

Use the fixed 200 calibration / 200 held-out splits. Do not choose thresholds on held-out labels or silently discard failed questions. The 7B study excludes earlier held-out contexts and is explicitly exploratory. Public source downloads and GPU weights require network access. The local offline report checker does not.

## Hosted protocol comparison

Only `scripts/protocol_benchmark.py` needs a hosted key. Store it privately outside Git and set `DOCMIND_SECRETS_FILE` to that path, or use the notebook's secret facility. Never write it in code. The script enforces a conservative 80,000-token limit, records failures and leaves budget-unserved rows in the denominator. The published run exhausted its allowance; rerunning is a new experiment and needs a fresh allowed quota. Do not enable billing to reproduce these results.

## Small local checks

`python -m pytest -q` and `python scripts/verify_benchmark_reports.py` require no model downloads or live inference. `scripts/measure_local.py` and `scripts/load_service.py` record synthetic parser, BM25 and demo-service measurements. Their results are not estimates of real hosted inference capacity. A four-gigabyte laptop is unsuitable for the full neural ablation matrix.

## State and secrets

Run a single process per corpus. Keep the database, review content and upload directory private; SQLite includes source text. Transfer only data you are authorized to process. Do not upload a whole working directory: ignored files can still contain secrets. Use a source allowlist and inspect the archive before upload. No particular free hosting service or indefinite GPU availability is guaranteed by this project.
