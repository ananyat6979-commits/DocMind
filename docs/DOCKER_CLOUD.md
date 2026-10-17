# Docker without Docker on your laptop

Two free GitHub services do the work: **Actions** builds and tests the image on GitHub's machines, and **Codespaces** gives you a cloud machine that has Docker so you can run it and get a URL.

## What is verified and what is not
- Verified: the Dockerfile passes `hadolint`, the workflow passes `actionlint`, the compose file parses, and 5 regression tests pin the Dockerfile's properties (non-root, health check, models baked in, CPU torch).
- **Not verified: I could not build or run the image myself** (no container runtime where I work). Treat the first Actions run as the real test. It has three smoke tests that fail loudly; if it fails, paste me the log of the failing step.
- Free limits (GitHub's billing docs): personal free accounts get 120 core-hours and 15 GB storage per month for Codespaces, so a 2-core codespace gives about 60 hours. With no payment method on file, usage is blocked, not billed. Actions minutes for private repos are limited on the free plan; public repos are free. Check github.com/settings/billing.

## Step 1: push your fixed code to your repo (PowerShell)
Your repo is `ananyat6979-commits/DocMind`. Go to the folder that contains `.git`, then:
```powershell
git status                                   # note what is already modified
Expand-Archive -Force $HOME\Downloads\DocMind-fixed-v3.zip -DestinationPath $env:TEMP\dm
Copy-Item -Recurse -Force $env:TEMP\dm\DocMind\* .
git add -A
git status --short                           # CHECK: no .env, no venv, no indexes, no PDFs. Stop here if you see any.
git commit -m "Fix agent bugs, multi-document ingestion, evaluation, review queue, Docker cloud build"
git push origin main
```
`.env`, `venv/`, `indexes/` and `data/*` (except `testset.json`) are gitignored, so your key and PDFs stay local.

## Step 2: watch the cloud build (no Docker needed)
GitHub > your repo > **Actions** tab > workflow **docker**. A push that touches the Dockerfile or code triggers it, or press **Run workflow**. Green means: image built, both models load with the network disabled, the full ingest and agent pipeline ran inside the image, and the UI reported healthy. On `main` it then publishes `ghcr.io/ananyat6979-commits/docmind:latest`.

## Step 3: run it in Codespaces
1. First add secrets: GitHub > Settings > Codespaces > **New secret**, repository access = DocMind. Create `GROQ_API_KEY` (your **new** key), `DOCMIND_APP_PASSWORD` (long, random) and `DOCMIND_REVIEWER_PIN`. They must exist before you create the codespace (if you add them later, rebuild it).
2. Repo page > green **Code** button > **Codespaces** tab > **Create codespace on main** (2-core is enough).
3. In the codespace terminal:
```bash
docker version                      # should print a Client and a Server section
docker compose up --build ui
```
The first build takes several minutes (it downloads CPU torch and bakes the models). Later starts are quick.
4. VS Code shows a pop-up for port 8501, or open the **Ports** tab and click the globe. The URL looks like `https://<name>-8501.app.github.dev`. Ports are private (only you) by default. To let pilot users in, right-click the port > Port Visibility > Public, and keep the password on: anyone with the URL can reach the login page.
5. API as well: `docker compose --profile api up --build api` and forward port 8000.

### Running costs your hours
The codespace stops after an idle period (30 minutes by default; adjustable in Settings > Codespaces). **Stop it yourself when you are done**: Codespaces page > the codespace menu > Stop. Data survives stop and start, and is deleted when the codespace is deleted.

### Back up and restore the database
```bash
docker compose stop ui
docker compose cp ui:/data/indexes/docmind.db ./docmind_backup.db     # then right-click it in Explorer > Download
```
Load a database produced by the Kaggle notebook (its embeddings are reused, nothing is recomputed):
```bash
docker compose create ui              # harmless if it already exists; cp needs the container to exist
docker compose stop ui
docker compose run --rm --no-deps --entrypoint "" ui mkdir -p /data/indexes
docker compose cp ./docmind.db ui:/data/indexes/docmind.db
docker compose run --rm --no-deps -u root --entrypoint "" ui chown docmind:docmind /data/indexes/docmind.db
docker compose up ui
```

## Alternative: make Docker work on the laptop, lightly
Docker Desktop is the heavy part. Docker Engine inside WSL2 is much lighter, but it still needs CPU virtualization enabled in BIOS (Task Manager > Performance > CPU shows "Virtualization: Enabled"). If that says Disabled, no local Docker will work and the cloud route above is the answer. Otherwise:
```powershell
wsl --install -d Ubuntu
```
Inside Ubuntu: `curl -fsSL https://get.docker.com | sh` then `sudo usermod -aG docker $USER`. Cap WSL's memory in `%UserProfile%\.wslconfig`:
```
[wsl2]
memory=4GB
processors=2
```
Then `wsl --shutdown` and reopen. If you tell me the actual error Docker Desktop gives, I can be more specific.

## What does not work
Kaggle and Colab notebooks do not provide a Docker daemon, as far as I know, so use them for the evidence run (docs/CLOUD.md), not for containers.

## If something fails
| Symptom | Likely cause |
|---|---|
| Actions fails at "Build" with a download error | A transient network or PyPI/PyTorch index error. Press Re-run jobs. |
| Smoke 1 fails | A model is not baked in: HF download failed during build. Re-run, and send me the log if it repeats. |
| Smoke 3 fails | Look at the `docker logs dm` output printed in that step. |
| Codespace UI says GROQ_API_KEY missing | The secret was added after the codespace was created. Rebuild the codespace. |
| File upload shows a 403 | Streamlit's XSRF check behind the proxy. The image already disables it; confirm you run the compose file, not a bare `streamlit run`. |
| Login page missing | `DOCMIND_APP_PASSWORD` is empty. Set the secret and restart. |
