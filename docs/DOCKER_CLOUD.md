# Centralized deployment

The UI calls one authenticated API. It has no generation key or library volume. The API owns the corpus, review state, quota and trusted identities. Readers share one library; tenant and per-document authorization are outside this release.

Create private personal credentials before starting containers:

```powershell
python scripts/create_credential.py --actor owner --roles reader ingestor reviewer admin
$env:DOCMIND_CREDENTIALS_PATH = "$env:USERPROFILE/.config/docmind/credentials.json"
docker compose up --build
```

On Linux, set `DOCMIND_CREDENTIALS_GID` to the registry group ID printed by the helper. The registry is group-readable for the API; personal key files remain owner-only. Windows Docker file-sharing uses the host access controls.

Read the generated personal `.key` file privately and enter it in the UI. The service mounts only the hashed credential registry, read-only. Set the provider key and model in the shell or private environment configuration. Never put keys in Git. Remove a credential record to revoke it; changes apply on the next request. Protect private files with host access permissions and use TLS for remote access.

The API and UI bind localhost ports 8000 and 8501. The API's data volume is the only library. A separate parser receives bytes over a Unix socket. Its container has no network namespace access, accepts only Unix-family sockets, drops capabilities, runs as UID 10001, uses a read-only filesystem and a bounded temporary filesystem, and has CPU, memory and process limits. It receives no application credential file or database mount. A malformed or exhausted worker returns a bounded rejection and the service handles the next request.

`deployment/compose.parser.yml` starts the isolated parser independently. The `parser isolation` workflow exercises denied network access, absent secrets, environment scrubbing, memory exhaustion and recovery. Passing these checks is a release gate. A working-branch push does not publish an image or establish that the container checks have passed.

Native parsing requires both `DOCMIND_PARSER_MODE=local` and `DOCMIND_ALLOW_UNSANDBOXED_PARSING=1` and is restricted to trusted development/benchmark inputs. The local UI additionally requires `DOCMIND_UI_MODE=local` and `DOCMIND_ALLOW_LOCAL_UI=1`. Resource-limited native parsing has no filesystem or network confinement.

CPU and CUDA environments are recorded under `locks/`, including hashes. The CPU image installs the Linux Python 3.11 lock and pinned model revisions. The full image's offline neural smoke checks remain separate from the small parser-isolation checks. The local Docker daemon must be running to execute either suite locally.

Liveness only proves the HTTP process runs. Authenticated readiness checks generation configuration, database access and corpus embedding identity, not all neural inference paths. SQLite state, in-process quotas and the ingestion queue support one service process. The default profile has no distributed operation or tenant authorization. The opt-in PostgreSQL profile below addresses shared state and tenant libraries; multi-host storage, regional failover and kernel isolation beyond the configured container boundary remain limitations.

Compose forwards the explicit retrieval mode, embedding identity, optional reranker identity and score-policy settings. Known default model revisions are pinned in configuration. Alternate models require their exact revision and a custom image with those weights cached before offline startup. The standard image caches E5-small, legacy BGE, the English reranker and the English NLI judge.


## Bounded HTTP diagnostics

`python scripts/load_http_service.py --out reports/release_workspace/http-check` starts a temporary loopback API, uses synthetic data and offline stand-ins, and records real socket HTTP behavior. Add `--real-models --requests 12 --concurrency 2` only in a CPU environment containing the pinned, cached E5 and NLI models. Model downloads and provider generation are disabled in both modes. The Docker workflow exercises the real-model variant with external networking disabled.

Startup, queued ingestion, one cold query, and warm HTTP load are separate measurements. Requested, failed and unserved counts accompany completed-only percentiles. This is one process and a repeated tiny fixture with demo generation, not a hosted-reader SLA, representative corpus-scale load or multi-tenant validation. Server termination by the harness does not validate graceful shutdown.

For native model startup investigation, `python scripts/profile_local_models.py --out reports/release_workspace/model-check` records stages and repeated thread stacks under a finite process deadline. A model-output file alone never makes a timed-out process pass. Diagnostic directories may contain local paths and remain private.


The runtime UID does not own application source or the baked model cache. Persistent application data is assigned to that UID; temporary runtime files belong in `/tmp`. The offline model-loading check runs with a read-only root filesystem and separate temporary `/data` and `/tmp` mounts. Ownership hardening is validated separately from parser confinement and does not provide tenant isolation or protection against every kernel exploit.


Dense mode builds only the dense index view; sparse mode builds only the lexical view. Explicit mode changes rebuild the required coherent snapshot, and concurrent searches retain the snapshot they selected. Hybrid modes build both views. Corpus changes still require full snapshot refreshes, and corpus loading still reads stored vectors. This removes unnecessary view construction without providing incremental persistent indexing or a corpus-scale memory bound.


## Persistent multi-process option

The default SQLite Compose deployment remains a one-process profile. [PostgreSQL deployment](POSTGRES_DEPLOYMENT.md) provides a separate two-worker profile with forced tenant row policies, persistent pgvector writes, shared quotas and renewable ingestion leases. It uses the same confined parser and centralized UI. Live CI covers tenant denials, synchronized quotas, killed-worker recovery and stale writes. Database search and neural HTTP load have separate bounded measurement records under `reports/releases/r3`; neither is a hosted-reader SLA. Follow the profile's migration and private-credential procedure rather than scaling the SQLite service.

Compose forwards `DOCMIND_RELEASE_MODE` (default `draft`), `DOCMIND_ATTRIBUTION` (default `0`), `DOCMIND_ANSWER_STRATEGY` (default `single`) and `DOCMIND_QUERY_TIMEOUT_SECONDS` (default `30`) to the API in both profiles. Set `DOCMIND_RELEASE_MODE=strict` in the private environment configuration to require human approval. These are operator settings, not query-body permissions.

## Published image provenance

The application base image is pinned by digest. Main publication requests a BuildKit SBOM and full build provenance. A separate main-only job signs a build-provenance attestation for the exact pushed image digest; its OIDC and attestation permissions are confined to that job. Working-branch builds cannot use that signing job. These controls identify build origin and inventory, not absence of vulnerabilities or commercial clearance for model training data.

After publication, verify the immutable digest with `gh attestation verify oci://ghcr.io/ananyat6979-commits/docmind@sha256:DIGEST --repo ananyat6979-commits/DocMind`. Replace DIGEST with the published value. The image bakes in pinned model weights and is substantially larger than the static showcase. The public model card records dataset/model licensing limitations; the code license does not override them.

## One-command offline inspection

After pulling the image, this runs the actual default retrieval/quote pipeline on a tiny original synthetic corpus, using cached E5 and an extractive demonstration reader:

```sh
docker run --rm --network none --read-only --tmpfs /tmp:rw,nosuid,size=512m ghcr.io/ananyat6979-commits/docmind@sha256:f0771fd5914dd781b338e6d8baeaf1aba9a7db19845253ab9de75c50b620ce9f python scripts/offline_demo.py
```

It requires no API key, opens no listening port and makes no hosted call. It prints an explicitly unverified draft and exact quoted evidence. It is a component demonstration, not a generative accuracy baseline or a substitute for the authenticated UI/API/parser deployment. Use a published immutable digest in place of latest for a fixed source version. A fresh image pull downloads the baked model weights; they are not fetched at runtime.
