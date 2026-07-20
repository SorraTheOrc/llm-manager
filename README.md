LLM Manager
===========

This repository contains a local copy of the llm proxy with an added model statistics panel and SSE status broadcasts. It was pushed to `SorraTheOrc/llm-manager` for development and review.

Key files
- `proxy/server.py` — server, SSE, and the model stats query helper
- `proxy/tests/` — unit tests and Playwright tests (e.g. `test_query_llama_status.py`, `status-stats.spec.js`)

Quick start (development)
1. Create and activate a virtualenv:
   ```
   python3 -m venv .venv
   . .venv/bin/activate
   ```
2. Install Python test/runtime deps:
   ```
   pip install -r requirements.txt || pip install pytest pytest-asyncio httpx fastapi requests pyyaml
   ```
3. Run the proxy server (from repo root):
   ```
   cd proxy
   . .venv/bin/activate && python -m uvicorn server:app --host 127.0.0.1 --port 3000
   ```

Testing
- Unit tests (example):
  ```
  . .venv/bin/activate && python -m pytest proxy/tests/test_query_llama_status.py -q
  ```
- Full test suite (proxy):
  ```
  . .venv/bin/activate && python -m pytest -q
  ```

Linting
-------

This project uses [ruff](https://docs.astral.sh/ruff/) for Python linting
and formatting. Configuration is in `.ruff.toml` at the repo root.

### Quick start

.ruff.toml is auto-discovered from the repo root, so commands can be run from any location:

```bash
# Run all lint checks
.venv/bin/ruff check .

# Auto-fix fixable issues
.venv/bin/ruff check --fix .
```

### Rule set

| Category | Rules | Description |
|----------|-------|-------------|
| F (Pyflakes) | `F` | Logic errors, undefined names, unused imports |
| E (Pycodestyle errors) | `E` | Formatting errors (except line length) |
| W (Pycodestyle warnings) | `W` | Formatting warnings (blank line whitespace) |
| I (isort) | `I` | Import ordering |
| N (Naming) | `N` | PEP 8 naming conventions |
| UP (pyupgrade) | `UP` | Modern Python idioms |
| RUF100 | `RUF100` | Unused `# noqa` directives |

Line length is set to **120** characters (project convention). The E501
rule is disabled since line-length is advisory.

### Test directory exceptions

Test files in `tests/` and `proxy/tests/` allow `assert` statements,
`print()` calls, and unused imports (common in test helpers). These
exceptions are configured via `per-file-ignores` in `.ruff.toml`.

### Remaining issues

After auto-fix, the remaining warnings are non-auto-fixable issues that
require manual attention. Run `ruff check .` to see the current count.

- Playwright UI tests (requires Node/npm and browser install):
  ```
  npm i -D playwright && npx playwright install
  npx playwright test proxy/tests/status-stats.spec.js
  ```

Worklog / tracking
- Associated worklog item: `LP-0MN5AW8DE1KJVAKX` (Add stats to the main page for the currently loaded model)

Notes
- The repository was pushed to `git@github.com:SorraTheOrc/llm-manager.git` and `origin` is configured to point to it.
- Before running integration or Playwright tests ensure the local llama-server (or a compatible mock) is available and any long model loading has completed — requests may return 503 while the model loads.

Configuration
-------------

Key server configuration in `proxy/config.yaml` under `server:`:

| Key | Default | Description |
|-----|---------|-------------|
| `slot_management.slot_pool_size` | `4` | Number of slots (GPU contexts) for job-level slot ownership. |
| `slot_management.slot_queue_max_depth` | `16` | Maximum jobs waiting in queue when all slots busy. |
| `slot_management.slot_job_timeout_seconds` | `300.0` | Seconds of inactivity before releasing a job's slot. The timeout check skips slots with an active request in flight to prevent premature slot release during long streaming responses. |
| `slot_management.slot_queue_overflow_retry_after` | `900` | Seconds in Retry-After header on queue overflow. |

When `slot_management` is present in config, the JobScheduler assigns each
multi-turn conversation (session) to a slot for its entire lifetime,
eliminating save/restore overhead between turns. When absent, the previous
hash-based slot assignment with save/restore is used.

Existing session slot settings (used when `slot_management` is absent):

| Key | Default | Description |
|-----|---------|-------------|
| `session_slot_pool_size` | `1` | Number of parallel dispatch sessions. Controls how many concurrent Pi agent sessions can hold dispatch leases simultaneously. Also sets llama-server's `--parallel` flag. |
| `session_slot_save_path` | `./slot-cache` | Directory for KV cache snapshots. |
| `session_slot_timeout_seconds` | `3.0` | Slot save/restore timeout in seconds. |

Contributing
- Open issues and PRs in the `SorraTheOrc/llm-manager` repo. If you want changes merged upstream to `rgardler/llm`, open a PR from this repo to the upstream repository.

License
- See upstream project for license information.

---

## Session Recording

The proxy automatically records all session message payloads (prompts,
completions, embeddings) to disk for debugging, auditing, and analysis. 
Recording is **always on** and uses non-blocking I/O (``asyncio.create_task``)
to avoid impacting request latency.

### Configuration

Session recording is configured under the ``session_recording`` key in
``proxy/config.yaml``:

```yaml
session_recording:
  path: proxy/session-recordings/   # Directory for recording files
```

The ``path`` defaults to ``proxy/session-recordings/`` if not specified.

### Directory Structure

Recordings are organized on disk by session ID with timestamps:

```
<recording-path>/
  <session-id>/
    <timestamp>-request.json            # client → proxy (original request)
    <timestamp>-proxy_to_provider-request.json  # proxy → provider (after processing)
    <timestamp>-response.json          # provider → client (assembled response)
```

Each JSON file contains:
- ``session_id`` — The session identifier
- ``direction`` — One of ``client_to_proxy``, ``proxy_to_provider``, or ``provider_to_client``
- ``timestamp`` — ISO8601 timestamp of when the recording was written
- ``payload`` — The actual request/response message payload

### What gets recorded

- **Client → Proxy**: The original request payload as received from the client
  (messages, model, stream flag, etc.)
- **Proxy → Provider**: The request payload after proxy processing
  (session handling, model overrides, system prompt injection)
- **Provider → Client**: The final assembled response from the provider.
  For streaming responses chunks are fully assembled into a single JSON
  document before writing.

### What is NOT recorded

- HTTP headers (authorization, cookies, internal routing metadata)
- Proxy-internal secrets or API keys
- Individual SSE chunks (only the assembled response is recorded)

### Admin Endpoint

Recordings can be listed and retrieved via the admin HTTP API (requires
access to the proxy server):

```
# List all sessions with recordings
GET /admin/sessions

# List recordings for a specific session
GET /admin/sessions/<session-id>/recordings

# Retrieve a specific recording file
GET /admin/sessions/<session-id>/recordings/<filename>
```

Example response for listing recordings:

```json
{
  "session_id": "abc-123",
  "recordings": [
    {
      "filename": "2026-07-06T10:00:00.000000-request.json",
      "timestamp": "2026-07-06T10:00:00.000000",
      "direction": "client_to_proxy",
      "file_size": 1234
    }
  ]
}
```

When a session has no recordings, the endpoint returns HTTP 404 with a
descriptive message.

### Security Considerations

- Recordings contain user prompts and model responses, which may include
  PII or sensitive business data.
- Recordings are stored on the local filesystem only. There is no
  automatic sharing, network transmission, or cloud upload.
- The recording directory should be restricted to the user running the
  proxy (e.g., ``chmod 700``).
- Cleanup/retention policies should be configured separately to prevent
  disk space exhaustion. See the [Disk Space Management](#disk-space-management) section for
  cleanup scripts and retention policies.

---

## Disk Space Management

This project includes automated cleanup scripts to manage disk usage from
stale model caches, unused container images, stopped containers, and pi agent
session logs.

### Cleanup Scripts

All scripts are in `scripts/` and support `--dry-run` (preview without
deletion) and `--json` (machine-readable output).

| Script | Purpose |
|--------|---------|
| `scripts/cleanup-model-cache.sh` | Remove HuggingFace models not in `models.ini` and duplicate llama.cpp GGUF cache |
| `scripts/cleanup-container-images.sh` | Remove container images not referenced by project scripts or Containerfile |
| `scripts/cleanup-stopped-containers.sh` | Remove exited containers stopped for more than 7 days |
| `scripts/cleanup-pi-sessions.sh` | Remove old pi agent session logs (keep last 90 days / 50 sessions per workspace) |
| `scripts/cleanup-all.sh` | Orchestrator that runs all cleanup scripts |
| `scripts/install-cron.sh` | Install daily crontab entries for automated cleanup |

### Quick Start

```bash
# Preview what would be deleted
./scripts/cleanup-all.sh --dry-run

# Run actual cleanup
./scripts/cleanup-all.sh

# Install daily cron job (runs at 3 AM)
./scripts/install-cron.sh
```

### Retention Policies

| Category | Default Retention | Configuration |
|----------|------------------|--------------|
| Model cache | Keep only models listed in `models.ini` | Edit `models.ini` to add/remove models |
| Container images | Keep only images referenced by `scripts/` and `Containerfile` | `LLM_CLEANUP_ALL=1` to remove cross-project images |
| Stopped containers | Remove containers exited for >7 days | `--max-age DAYS` flag |
| Pi session logs | Keep last 90 days AND last 50 sessions per workspace | `--retention-days`, `--keep-sessions` flags |

### Cron Automation

By default, cleanup runs daily at **3:00 AM**. Slot cache cleanup
(if installed) runs at **4:00 AM** to avoid resource contention.

#### Individual Toggles

Each cleanup category can be independently disabled via environment
variables:

```bash
# Disable specific cleanup categories
CLEANUP_ENABLE_MODELS=0     # Skip model cache cleanup
CLEANUP_ENABLE_CONTAINERS=0 # Skip container and image cleanup
CLEANUP_ENABLE_PI=0         # Skip pi session log cleanup

# Combine with the orchestrator
CLEANUP_ENABLE_PI=0 ./scripts/cleanup-all.sh
```

> **Note:** Stopped container cleanup and container image cleanup are both
> controlled by `CLEANUP_ENABLE_CONTAINERS`. Stopped containers are removed
> before images to release image references held by stopped containers.
> Use the `--yes` flag with `cleanup-stopped-containers.sh` for non-interactive
> / cron use. Without `--yes` in a non-TTY environment, the script will prompt
> with a clear message instructing use of `--yes`.

#### Manual Cron Installation

```bash
# Install cron entries (daily at 3 AM)
./scripts/install-cron.sh

# Preview cron entries without installing
./scripts/install-cron.sh --dry-run

# Uninstall cron entries
./scripts/install-cron.sh --uninstall

# Use a custom hour (e.g., 2 AM)
CLEANUP_HOUR=2 ./scripts/install-cron.sh
```

Logs are written to `/var/log/pi-cleanup.log` by default. Log rotation
should be configured via the system's logrotate.

### Related Work

- Slot cache cleanup (LP-0MQMC4MNU002QJK4) — separate script
  `scripts/cleanup-slot-cache.sh` for KV cache snapshot retention.
  Runs at 4:00 AM to avoid conflict with the main cleanup at 3:00 AM.
  See the slot cache cleanup epic for details.

## GPU Offload (ROCm)

The proxy supports AMD ROCm GPU acceleration for LLM inference via the
router-mode llama-server. GPU offload is configured in `models.ini`:

```ini
[global]
ngl = 99          # GPU layers (99 = offload all layers to GPU)
```

### How it works

- Router-mode startup in `start-llama.sh` reads `[global] ngl` from `models.ini`
  and passes it as `-ngl <value>` to the llama-server command.
- This applies to ALL model workers spawned by the router (both Qwen3 and the
  mxbai-embed embeddings preset).
- Single-model mode also uses `-ngl 99` for GPU offload.

### Rollback to CPU-only

If ROCm instability occurs, operators can roll back to CPU-only mode:

1. **Quick (env var):** `LLAMA_NGL=0 ./start-llama.sh router`
2. **Permanent:** Set `ngl = 0` in `[global]` in `models.ini` and restart.

### Verification

- **Unit tests:** `proxy/tests/test_gpu_offload_verification.py` validates
  models.ini parsing, env var propagation, and command construction.
- **Live verification:** See `docs/gpu-offload-verification.md` for health
  checks, ROCm confirmation steps, and baseline measurement.
- **Guard tests:** All guard tests pass. A regression in GPU offload
  settings will cause test failures.

### Pre-requisites

- AMD GPU with ROCm drivers
- llama-server compiled with `-DGGML_HIP=ON`
- ROCm environment variables in `start-llama.sh`:
  - `HSA_OVERRIDE_GFX_VERSION=11.5.1`
  - `ROCM_LLVM_PRE_VEGA=1`
