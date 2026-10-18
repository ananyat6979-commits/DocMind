# Docker without local Docker

Two free GitHub services do the work. **Actions** builds and tests the image on GitHub's machines. **Codespaces** provides a cloud machine with Docker so the image can be run and reached by URL.

## Status
- The Dockerfile passes `hadolint`, the workflows pass `actionlint`, the compose file parses, and `tests/test_deploy_files.py` pins the Dockerfile's key properties (non-root user, health check, models baked in, CPU-only torch).
- The image build itself is verified by the `docker` workflow, which runs on relevant pushes to `main` and can be started manually. It has three smoke tests that fail loudly.
- Free limits (GitHub's billing documentation): personal free accounts get 120 core-hours and 15 GB of storage per month for Codespaces, so a 2-core codespace gives about 60 hours. Without a payment method on file, usage is blocked rather than billed. Actions minutes for private repositories are limited on the free plan; public repositories are free. See github.com/settings/billing.

## Step 1: push the code
From the folder that contains `.git`:
```powershell
git status
git add -A
git status --short
git commit -m "<message>"
git push origin main
```
Check the `git status --short` output before committing: there must be no `.env`, `venv`, `indexes`, database files or PDFs. The repository's `.gitignore` excludes them.

## Step 2: watch the cloud build
GitHub, Actions tab, workflow **docker**. A push that touches the Dockerfile or the code triggers it, or use **Run workflow**. A green run means the image built, both models load with the network disabled, the full ingest and agent pipeline ran inside the image, and the UI reported healthy. On `main` it then publishes `ghcr.io/<owner>/docmind:latest`.

## Step 3: run it in Codespaces
1. Add secrets first: GitHub, Settings, Codespaces, **New secret**, with repository access to this repository. Create `GROQ_API_KEY`, `DOCMIND_APP_PASSWORD` and `DOCMIND_REVIEWER_PIN`. They must exist before the codespace is created (if added later, rebuild it).
2. On the repository page choose **Code**, **Codespaces**, **Create codespace on main** (2-core is enough).
3. In the codespace terminal:
```bash
docker version                      # should print a Client and a Server section
docker compose up --build ui
```
The first build takes several minutes (CPU torch download and model baking). Later starts are quick.
4. Open port 8501 from the Ports tab. The URL looks like `https://<name>-8501.app.github.dev`. Ports are private by default. To admit pilot users, set Port Visibility to Public and keep the password enabled.
5. API as well: `docker compose --profile api up --build api` and forward port 8000.

### Running costs hours
A codespace stops after an idle period (30 minutes by default, adjustable in Settings, Codespaces). Stop it manually when finished. Data survives stop and start, and is deleted when the codespace is deleted.

### Back up and restore the database
```bash
docker compose stop ui
docker compose cp ui:/data/indexes/docmind.db ./docmind_backup.db     # then download it from the Explorer
```
Load a database produced by the cloud notebook (its embeddings are reused, nothing is recomputed):
```bash
docker compose create ui              # harmless if it exists; cp needs the container to exist
docker compose stop ui
docker compose run --rm --no-deps --entrypoint "" ui mkdir -p /data/indexes
docker compose cp ./docmind.db ui:/data/indexes/docmind.db
docker compose run --rm --no-deps -u root --entrypoint "" ui chown docmind:docmind /data/indexes/docmind.db
docker compose up ui
```

## Alternative: lightweight Docker on a local Windows machine
Docker Desktop is the heavy part. Docker Engine inside WSL2 is much lighter, but it needs CPU virtualization enabled in BIOS (Task Manager, Performance, CPU shows "Virtualization: Enabled"). If that reads Disabled, local Docker cannot work and the Codespaces route above applies. Otherwise:
```powershell
wsl --install -d Ubuntu
```
Inside Ubuntu: `curl -fsSL https://get.docker.com | sh`, then `sudo usermod -aG docker $USER`. Cap WSL's memory in `%UserProfile%\.wslconfig`:
```
[wsl2]
memory=4GB
processors=2
```
Then run `wsl --shutdown` and reopen.

## Limitations
Kaggle and Colab notebooks do not provide a Docker daemon, so use them for the evidence run ([CLOUD.md](CLOUD.md)), not for containers.

## Troubleshooting
| Symptom | Likely cause |
|---|---|
| Actions fails at "Build" with a download error | A transient network or package-index error. Re-run the failed jobs. |
| Smoke 1 fails | A model was not baked in because the download failed during build. Re-run; if it repeats, open an issue with the log. |
| Smoke 3 fails | See the `docker logs dm` output printed in that step. |
| Codespace UI reports a missing GROQ_API_KEY | The secret was added after the codespace was created. Rebuild the codespace. |
| File upload returns 403 | Streamlit's XSRF check behind the proxy. The image disables it; confirm the compose file is used rather than a bare `streamlit run`. |
| No login page | `DOCMIND_APP_PASSWORD` is empty. Set the secret and restart. |
