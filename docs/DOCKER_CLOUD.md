# Centralized deployment

The UI calls one authenticated API. It has no generation key or library volume. The API owns the corpus, review state, quota and trusted identities. Readers share one library; tenant and per-document authorization are outside this release.

Create private personal credentials before starting containers:

```powershell
python scripts/create_credential.py --actor owner --roles reader ingestor reviewer admin
$env:DOCMIND_CREDENTIALS_PATH = "$env:USERPROFILE/.config/docmind/credentials.json"
docker compose up --build
```

Read the generated personal `.key` file privately and enter it in the UI. The service mounts only the hashed credential registry, read-only. Set the provider key and model in the shell or private environment configuration. Never put keys in Git. Remove a credential record to revoke it; changes apply on the next request. Protect private files with host access permissions and use TLS for remote access.

The API and UI bind localhost ports 8000 and 8501. The API's data volume is the only library. A separate parser receives bytes over a Unix socket. Its container has no network namespace access, accepts only Unix-family sockets, drops capabilities, runs as UID 10001, uses a read-only filesystem and a bounded temporary filesystem, and has CPU, memory and process limits. It receives no application credential file or database mount. A malformed or exhausted worker returns a bounded rejection and the service handles the next request.

`deployment/compose.parser.yml` starts the isolated parser independently. The `parser isolation` workflow exercises denied network access, absent secrets, environment scrubbing, memory exhaustion and recovery. Passing these checks is a release gate. A working-branch push does not publish an image or establish that the container checks have passed.

Native parsing requires both `DOCMIND_PARSER_MODE=local` and `DOCMIND_ALLOW_UNSANDBOXED_PARSING=1` and is restricted to trusted development/benchmark inputs. The local UI additionally requires `DOCMIND_UI_MODE=local` and `DOCMIND_ALLOW_LOCAL_UI=1`. Resource-limited native parsing has no filesystem or network confinement.

CPU and CUDA environments are recorded under `locks/`, including hashes. The CPU image installs the Linux Python 3.11 lock and pinned model revisions. The full image's offline neural smoke checks remain separate from the small parser-isolation checks. The local Docker daemon must be running to execute either suite locally.

Liveness only proves the HTTP process runs. Authenticated readiness checks generation configuration and database access, not all neural inference paths. SQLite state, in-process quotas and the ingestion queue support one service process. Distributed operation, tenant authorization, incremental persistent indexes and kernel isolation beyond the configured container boundary remain limitations.
