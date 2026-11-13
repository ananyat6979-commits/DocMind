# Container deployment

The Docker image uses Python 3.11, CPU PyTorch, pinned retrieval and attribution weights under `/opt/hf`, and a non-root user. Runtime state belongs under `/data`. Network access is disabled in the model-loading smoke check, proving that the configured default weights were included during the build.

## Local run

Create `.env` from `.env.example` and privately fill the UI password, administrator PIN, reviewer PIN, provider key and a currently available model ID. Then run:

```powershell
docker compose up --build ui
```

For the API, set `DOCMIND_API_KEY` and run:

```powershell
docker compose --profile api up --build api
```

The UI uses port 8501 and the API port 8000. Missing credentials disable access. Compose intentionally gives the services separate data volumes; do not point concurrent processes at the same corpus. Review state and uploaded documents therefore do not transfer between the two services automatically. Put remote access behind TLS and a suitable network boundary. No public hosting provider is required for a local run.

## Build verification

The `docker` GitHub Actions workflow builds the image, loads all three neural models offline, performs demo ingestion/query and checks UI liveness. A manual run on a working branch builds and smoke-tests without publishing. Main-branch runs additionally publish the existing container package. The `ci` workflow separately tests Python 3.11, 3.12 and 3.13, critical lint checks and frozen report integrity without live inference credentials.

Model downloads make the image substantially larger than a basic web service. This build is not proof that all models fit in four gigabytes alongside other applications. HNSW is optional and full index refresh remains memory-bound. Liveness means the process is running; authenticated `/ready` performs basic configuration/database checks and is not a full neural inference probe.

## Troubleshooting

Inspect build logs for dependency/download failures. Inspect smoke logs for model-cache or permission failures. If uploads return 403 behind a proxy, configure trusted proxy origins and forwarding rather than disabling CORS/XSRF. If access is denied, verify the corresponding secret and restart. Persistent volumes may retain data after containers are removed; follow your retention policy before deleting them.
