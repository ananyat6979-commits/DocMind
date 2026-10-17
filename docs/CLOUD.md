# Running DocMind without Docker, and using free cloud compute

## Short answers
- **Docker is optional, but you can still have it.** The app runs natively in your venv. If you want the container anyway, build and run it in the cloud: [DOCKER_CLOUD.md](DOCKER_CLOUD.md).
- **Free cloud for the heavy work:** Kaggle or Colab notebooks. DocMind stores embeddings in SQLite, so you can ingest in the cloud and copy one file (`docmind.db`) back to your laptop; the laptop then only embeds each short query.
- **Free hosting for the pilot:** the honest options are below. Hugging Face Spaces is **no longer free** for this kind of app (checked on the HF docs page today: creating Gradio or Docker Spaces needs a paid plan; only Static Spaces are free). My earlier roadmap suggested it; that was wrong.

## What to use for what

| Goal | Best free option | Caveats |
|---|---|---|
| Evidence run (ablation + RAGAS) | Kaggle notebook (CPU is enough; GPU optional) or Colab | Kaggle needs a phone-verified account for Internet. Quotas and session lengths change, check kaggle.com/docs. Groq limits still apply, not Kaggle's. |
| Pilot with 2 to 3 users | Your laptop, run natively, plus a free Cloudflare quick tunnel during agreed sessions | Laptop must be on. Set `DOCMIND_APP_PASSWORD` first. Not 24/7. |
| 24/7 pilot | Oracle Cloud "Always Free" VM, or Streamlit Community Cloud | Not verified by me. Oracle needs a card for sign-up and capacity varies. Streamlit Community Cloud has tight memory, torch plus two models may not fit. |
| Short demo from a notebook | cloudflared tunnel from Kaggle/Colab | Read the platform's terms first; sessions end without warning; do not use for the pilot. |

## 0. Do this first: rotate your Groq key
The archive you shared contained your `.env`, so that key has left your machine. Treat it as exposed.
1. console.groq.com > API Keys > delete the old key > create a new one.
2. Put the new key only in `.env` (`GROQ_API_KEY=gsk_...`). It is gitignored; keep it out of zips.
3. To share your project without secrets or the 1.7 GB venv (PowerShell, from the project folder):
```powershell
Compress-Archive -Path (Get-ChildItem -Force | Where-Object { $_.Name -notin 'venv','.venv','.env','.git','mlruns','indexes','__pycache__' }) -DestinationPath ..\docmind_clean.zip
```

## 1. Your first real run on the laptop (your own PDF, no placeholders)
Type one command per line. (In your last output, several commands were pasted together and their output interleaved; and `data\your.pdf` was my placeholder, not a file.)
```powershell
cd D:\Code\Projects\MachineLearning\DocMind
venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q --ignore=tests/test_ragas_plumbing.py
python scripts\demo.py --doc data\The_Refrigeration_Problem_v5_Final.pdf
```
Use `python -m pytest`, not bare `pytest`. Bare `pytest` ran a global Python's pytest that had no `dotenv`, which was your ImportError.

## 2. Evidence run on Kaggle (free)
1. Make a clean zip (command above), then kaggle.com > Datasets > New Dataset > upload `docmind_clean.zip` and your PDFs. Set visibility **Private**.
2. New Notebook > File > Import Notebook > upload `notebooks/docmind_cloud.ipynb`.
3. Add data: your private dataset. Settings: Internet **On**. Add-ons > Secrets > `GROQ_API_KEY` (the new key). Accelerator: None is fine.
4. Run the cells in order. Cell 5a drafts questions (unverified). Download `data/testset.draft.json`.
5. On your laptop, verify the drafts yourself (this is the human step that makes the benchmark honest):
```powershell
python scripts\verify_testset.py --in data\testset.draft.json --out data\testset.json
```
6. Upload `data/testset.json` into the notebook's `data/` folder (or the dataset), then run cell 5b. If Groq's daily cap stops it you get exit code 2; run the same cell tomorrow and it resumes.
7. Notebook > Save Version > Save & Run All. Download `docmind_results.zip` from Output and unzip into your project folder:
```powershell
Expand-Archive -Force $HOME\Downloads\docmind_results.zip -DestinationPath .
python scripts\update_readme.py
```
Colab works the same way: upload the zip and PDFs in the Files panel and use the key icon for `GROQ_API_KEY`.

I tested the scripts the notebook calls (including a dry run of `cloud_run.py`) locally. I could not run the notebook on Kaggle or Colab from here, so if a cell fails, send me the error.

## 3. Pilot on your laptop through a free tunnel
```powershell
winget install --id Cloudflare.cloudflared
$env:DOCMIND_APP_PASSWORD = "choose-something-long"
$env:DOCMIND_REVIEWER_PIN = "1234"
streamlit run app.py
```
In a second PowerShell window:
```powershell
cloudflared tunnel --url http://localhost:8501
```
It prints an `https://....trycloudflare.com` URL. Share it with the pilot users plus the password, only for the agreed time. Anyone with URL and password can use your Groq quota and sees the shared library, so use non-confidential documents. Close the tunnel when done (Ctrl+C).

## 4. Docker without Docker on your laptop
See [DOCKER_CLOUD.md](DOCKER_CLOUD.md): GitHub Actions builds and tests the image in the cloud, and GitHub Codespaces (free monthly allowance) runs it and gives you a URL.

## Low-RAM laptop mode (4 GB machines)
If you must run locally on a small machine, close the browser and other heavy apps first, then:
```powershell
$env:DOCMIND_USE_RERANKER = "0"     # skips the cross-encoder, saves roughly half a GB
$env:DOCMIND_MAX_ITERATIONS = "3"   # fewer LLM calls, fewer Groq tokens
streamlit run app.py
```
The app will work, but retrieval is hybrid-only and the abstention gate is inactive. `scripts/ablation.py` refuses to run in this mode, so reranker numbers can never be produced by accident. Do evaluations in the cloud.

## Lighter laptop footprint
- `requirements.txt` no longer includes RAGAS, LangChain or MLflow (they are in `requirements-eval.txt`, needed only for evaluation), so a runtime install is much smaller.
- CPU-only torch is smaller still: `pip install torch --index-url https://download.pytorch.org/whl/cpu` before the other requirements.
- Ingest once (in the cloud or locally); later runs reuse the stored embeddings.
