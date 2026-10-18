# Running DocMind without Docker, and on free cloud compute

## Overview
- **Docker is optional.** The app runs natively in a Python virtual environment. For a container, build and run it in the cloud: see [DOCKER_CLOUD.md](DOCKER_CLOUD.md).
- **Heavy work on free notebooks.** Kaggle or Colab can run ingestion, the retrieval ablation and the RAGAS evaluation. Embeddings are stored in SQLite, so ingest in the cloud, copy one file (`docmind.db`) to a local machine, and local runs only embed each short query.
- **Hosting.** As of October 2026, Hugging Face Spaces no longer offers free compute for Gradio or Docker Spaces (see the Spaces documentation), so it is not listed below.

## Choosing where to run what

| Goal | Free option | Caveats |
|---|---|---|
| Evidence run (ablation and RAGAS) | Kaggle notebook (CPU is enough, GPU optional) or Colab | Kaggle needs a phone-verified account for Internet access. Quotas change, see kaggle.com/docs. Groq's own limits still apply. |
| Pilot with 2 to 3 users | A local machine running natively, plus a free Cloudflare quick tunnel during agreed sessions | The machine must stay on. Set `DOCMIND_APP_PASSWORD` first. Not 24/7. |
| 24/7 pilot | Oracle Cloud Always Free VM, or Streamlit Community Cloud | Not verified here. Oracle needs a card at sign-up and capacity varies. Streamlit Community Cloud has tight memory, so torch plus two models may not fit. |
| Short demo from a notebook | A cloudflared tunnel from Kaggle or Colab | Read the platform's terms first. Sessions end without warning. Not suitable for a pilot. |

## 0. Protect the API key
- Keep `GROQ_API_KEY` only in `.env` (gitignored) or the host's secret store. Never put it in an archive, a notebook cell, or a commit.
- If a key was ever shared or committed, revoke it at console.groq.com (API Keys) and create a new one.
- To share a project without secrets or a virtual environment (PowerShell, from the project folder):
```powershell
Compress-Archive -Path (Get-ChildItem -Force | Where-Object { $_.Name -notin 'venv','.venv','.env','.git','mlruns','indexes','__pycache__' }) -DestinationPath ..\docmind_clean.zip
```

## 1. First local run
Run one command per line.
```powershell
cd <project folder>
venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q --ignore=tests/test_ragas_plumbing.py
python scripts\demo.py --doc data\<your file>.pdf
```
Use `python -m pytest` rather than bare `pytest`, so the virtual environment's interpreter is used.

## 2. Evidence run on Kaggle
1. Create a clean zip (command above). On kaggle.com choose Datasets, New Dataset, upload `docmind_clean.zip` and your PDFs, and set visibility to **Private**.
2. New Notebook, File, Import Notebook, upload `notebooks/docmind_cloud.ipynb`.
3. Add the private dataset as data. Settings: Internet **On**. Add-ons, Secrets: `GROQ_API_KEY`. Accelerator: None is fine.
4. Run the cells in order. Cell 5a drafts questions (unverified). Download `data/testset.draft.json`.
5. Verify the drafts locally. This human step is what makes the benchmark honest:
```powershell
python scripts\verify_testset.py --in data\testset.draft.json --out data\testset.json
```
6. Upload `data/testset.json` into the notebook's `data/` folder, then run cell 5b. If Groq's daily cap stops it, the run exits with code 2; run the same cell later and it resumes.
7. Save Version, Save and Run All. Download `docmind_results.zip` from Output and unzip it into the project folder:
```powershell
Expand-Archive -Force $HOME\Downloads\docmind_results.zip -DestinationPath .
python scripts\record_metrics.py
python scripts\update_readme.py
```
Colab works the same way: upload the zip and PDFs in the Files panel and store `GROQ_API_KEY` with the key icon.

Status: the scripts the notebook calls are covered by automated tests (`tests/test_cloud_workflow.py`). The notebook itself has not been verified on Kaggle or Colab; if a cell fails, open an issue with the error output.

## 3. Pilot on a local machine through a free tunnel
```powershell
winget install --id Cloudflare.cloudflared
$env:DOCMIND_APP_PASSWORD = "<long random password>"
$env:DOCMIND_REVIEWER_PIN = "<pin>"
streamlit run app.py
```
In a second PowerShell window:
```powershell
cloudflared tunnel --url http://localhost:8501
```
It prints an `https://....trycloudflare.com` URL. Share it with pilot users together with the password, only for the agreed time. Anyone with the URL and password can use the Groq quota and sees the shared library, so use non-confidential documents. Stop the tunnel with Ctrl+C when finished.

## 4. Docker without local Docker
See [DOCKER_CLOUD.md](DOCKER_CLOUD.md): GitHub Actions builds and tests the image in the cloud, and GitHub Codespaces runs it and provides a URL.

## 5. Low-RAM machine mode (about 4 GB)
Close the browser and other heavy applications first, then:
```powershell
$env:DOCMIND_USE_RERANKER = "0"     # skips the cross-encoder, saves roughly half a GB
$env:DOCMIND_MAX_ITERATIONS = "3"   # fewer LLM calls, fewer Groq tokens
streamlit run app.py
```
Retrieval is then hybrid-only and the abstention gate is inactive. `scripts/ablation.py` refuses to run in this mode, so reranker numbers cannot be produced by accident. Run evaluations in the cloud.

## 6. Smaller installs
- `requirements.txt` holds only runtime dependencies. RAGAS, LangChain and MLflow live in `requirements-eval.txt` and are needed only for evaluation.
- CPU-only torch is smaller: `pip install torch --index-url https://download.pytorch.org/whl/cpu` before the other requirements.
- Ingest once (in the cloud or locally); later runs reuse the stored embeddings.
