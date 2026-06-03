# LLama Proxy Server

A proxy server that routes OpenAI-compatible API requests to either a local llama-server or remote API services based on configuration.

## Features

- **Unified API Endpoint**: Single endpoint for all LLM requests, regardless of backend
- **Web UI**: Built-in dashboard for monitoring, model switching, and API testing
- **Local Model Management**: Automatically starts/stops llama-server with the correct model
- **Remote API Routing**: Forward requests to OpenAI, Anthropic, or other OpenAI-compatible APIs
- **Hot Model Switching**: Automatically switches local models when a different model is requested
- **Streaming Support**: Full support for streaming responses (SSE)
- **Request/Response Logging**: Comprehensive logging with time-based rotation. Console output for STREAM CHUNK messages now prints only the streamed text content (delta.content) to reduce noisy JSON envelopes in the terminal; rotating file logs continue to record the full JSON chunk records unchanged.
- **Request + Token Counters**: In-memory counters with periodic JSON persistence
- **Session-Based Incremental Ingestion**: Reduce CPU and latency with per-session KV cache reuse
- **Live Log Tail + Stats**: `/logs` UI and `/logs/tail` SSE stream for logs/counts/tokens
-- Systemd integration details removed: the repository no longer distributes systemd unit files. Run the proxy manually or manage service units outside this repo.

## Requirements

- Python 3.10+
- Distrobox with a container named `llama` containing llama-server (llama.cpp)
-- `/home/rgardler/projects/llm/start-llama.sh` script for starting llama-server

See `../LLAMA_README.md` for distrobox setup instructions.

## Installation

### Quick Install

```bash
cd ~/projects/llm/proxy
sudo ./install.sh
```

This will:
1. Create a Python virtual environment
2. Install dependencies
3. Create the log directory

### Manual Install

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create log directory (optional - server will use ./logs/ as fallback)
sudo mkdir -p /var/log/llama-proxy
sudo chown $USER:$USER /var/log/llama-proxy

<!-- systemd unit installation removed from repo distribution -->
<!-- If you wish to run the proxy as a systemd unit, create and manage service files outside of this repository. -->
```

## Testing

To run the test suite, activate the virtual environment contained in the `proxy` folder. The repository's install script creates a `.venv` under `proxy/` — activate it and run the tests as follows:

```bash
cd proxy
source .venv/bin/activate
# (optional) install/update dependencies
pip install -r requirements.txt
# run tests
pytest -q

# refactor parity baseline (used during proxy/server.py extraction)
pytest -m refactor_parity -q
```

The `refactor_parity` selection covers the contract paths that must remain stable during refactor slices:
- routing/model lifecycle (`test_model_lifecycle_router_unit.py`, `test_model_lifecycle_router_integration.py`, `test_backend_resilience.py`)
- session/delta behavior (`test_incremental_ingestion.py`, `test_session_manager.py`)
- observability/status/log endpoints (`test_llama_local_status.py`, `test_logs.py`)

If you prefer not to activate the virtualenv, you can run pytest directly from the venv binary:

```bash
proxy/.venv/bin/pytest -q
```

## proxyctl (CLI)

A small bash CLI `proxyctl` is included to manage a user-local proxy process. It supports: `start`, `stop`, `restart`, `status`, and `logs`.

Installation example:

```sh
sudo install -m 0755 proxy/proxyctl /usr/local/bin/proxyctl
```

Usage examples:

```sh
proxyctl start    # start using proxy/config.yaml llama_start_script or start-llama.sh
proxyctl status   # show running status and PID
proxyctl logs     # tail the proxy logs
proxyctl stop     # stop the running proxy
```

The script respects `LLAMA_START_SCRIPT` environment variable and `proxy/config.yaml` `server.llama_start_script` entry when determining what to run.

## Configuration

Edit `config.yaml` to configure the server:

```yaml
# Server settings
server:
  host: "0.0.0.0"
  port: 8000
  llama_start_script: "/home/rgardler/projects/llm/start-llama.sh"
  llama_router_mode: true
  llama_router_preload:
    - "embeddings"
    - "gemma4"    # preload the configured default model (gemma4)
  llama_models_max: 2
  distrobox_name: "llama"  # Distrobox container where llama-server runs
  llama_server_port: 8080
  llama_startup_timeout: 300
  session_single_flight_mode: "queue"
  session_single_flight_max_queue_depth: 1
  session_slot_save_path: "/home/rgardler/projects/llm/slot-cache"
  session_slot_pool_size: 2
  session_slot_timeout_seconds: 3.0
  session_guardrail_max_runtime_seconds: 120
  session_guardrail_max_completion_tokens: 2048
  session_guardrail_repetition_min_pattern_chars: 64
  session_guardrail_repetition_min_repeats: 10
  session_guardrail_invalidate_on_cutoff: true
  session_guardrail_invalidate_on_repetition: false
  session_require_restore_signal: false

# Default model to load on startup
# Set the default to `gemma4` in examples and docs; other models (e.g., `gpt120`) remain available.
default_model: "gemma4"

# Logging configuration
logging:
  directory: "/var/log/llama-proxy"
  rotation_hours: 6
  retention_days: 90
  level: "INFO"

# Model routing
models:
  qwen3:
    type: "local"
    llama_model: "qwen3"
    force_full_prompt: true  # Disable delta routing for this model
    aliases:
      - "qwen3"
      - "qwen3-coder"
      - "qwen3*"          # Wildcard: matches qwen3-32b, qwen3-coder-instruct, etc.

  openai:
    type: "remote"
    endpoint: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"
    aliases:
      - "gpt-4"
      - "gpt-4-turbo"
      - "gpt-*"           # Wildcard: matches any model starting with gpt-
      - "o1-*"            # Wildcard: matches o1-preview, o1-mini, etc.
```

### Wildcard Patterns in Aliases

Aliases support wildcard patterns using fnmatch syntax:

| Pattern | Description | Example Matches |
|---------|-------------|-----------------|
| `*` | Matches any sequence of characters | `gpt-*` matches `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo` |
| `?` | Matches any single character | `gpt-?` matches `gpt-4`, `gpt-5` but not `gpt-4o` |
| `[seq]` | Matches any character in seq | `gpt-[34]` matches `gpt-3`, `gpt-4` |
| `[!seq]` | Matches any character not in seq | `gpt-[!3]*` matches `gpt-4` but not `gpt-3.5` |

**Priority:** Exact matches always take precedence over wildcard patterns. If a model name matches both an exact alias and a wildcard pattern, the exact match wins.

**Example:** With these aliases:
```yaml
qwen3:
  aliases: ["qwen3", "qwen3-coder"]   # Exact matches

openai:
  aliases: ["gpt-*", "o1-*"]          # Wildcards route ALL gpt- and o1- models to OpenAI
```

- Request for `gpt-4` → Routes to OpenAI
- Request for `gpt-4o-mini` → Routes to OpenAI
- Request for `o1-preview` → Routes to OpenAI
- Request for `qwen3-coder` → Routes to local qwen3

### Model Types

**Local Models** (`type: local`)
- Served by the local llama-server
- `llama_model`: Name passed to `start-llama.sh`
- Server automatically switches models when needed

**Remote Models** (`type: remote`)
- Forwarded to external API endpoints
- `endpoint`: Base URL of the API
- `api_key_env`: Environment variable containing the API key
- `headers`: Additional headers to include (optional)

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LLAMA_PROXY_CONFIG` | Path to config file (default: `./config.yaml`) |
| `OPENAI_API_KEY` | API key for OpenAI |
| `ANTHROPIC_API_KEY` | API key for Anthropic |

Systemd-specific instructions removed. If you run systemd units outside this repository, add environment variables via `systemctl edit <unit>` or your system's preferred method.

## Usage

### Starting the Server

**Development:**
```bash
source .venv/bin/activate
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

**Production:** Run the proxy under your platform's service manager or keep it as a manually started process:

```bash
# Manual start (recommended for repo-managed installs)
sudo systemctl start llama-proxy  # only if you created and installed a unit externally
```

### Web UI

The proxy includes a built-in web interface for monitoring and testing. Access it at:

```
http://localhost:8000/
```

#### Features

**Status Bar**
- Real-time display of current model, server health, and llama-server status
- Automatic refresh on page load

**Quick Links**
- Direct links to API documentation, health endpoint, and model list
- "Load Model" buttons to quickly switch between local models

**Configured Models Table**
- Lists all models from config.yaml with their type (local/remote) and aliases
- "Load Model" buttons for local models to trigger model switching

**Quick Test Section**
- Interactive chat interface for testing the currently loaded model
- Supports streaming responses (Server-Sent Events)
- Enter a prompt and see real-time model output

**API Passthrough Endpoints**
- Test buttons for each OpenAI-compatible endpoint:
  - Chat Completions (`POST /v1/chat/completions`)
  - Text Completions (`POST /v1/completions`)
  - List Models (`GET /v1/models`)
  - Embeddings (`POST /v1/embeddings`)
- Model dropdown selector to choose which model to test
- Shows example request JSON and actual response
- Useful for verifying API compatibility and debugging

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "ready": true,
  "current_model": "qwen3",
  "llama_server_running": true,
  "backend_signals": {
    "connect_failures": 0,
    "read_failures": 0,
    "timeout_failures": 0,
    "other_failures": 0,
    "concurrency_rejects": 0
  }
}
```

#### List Models
```bash
curl http://localhost:8000/v1/models
```

#### Chat Completions
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

#### Streaming Chat Completions
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Admin Endpoints

#### Reload Configuration
```bash
curl -X POST http://localhost:8000/admin/reload-config
```

#### Switch Model
```bash
curl -X POST http://localhost:8000/admin/switch-model/qwen2.5
```

#### Stop LLama Server
```bash
curl -X POST http://localhost:8000/admin/stop-server
```

#### Dump Current Counters (debug)
```bash
curl http://localhost:8000/admin/dump-counts
```

Returns current in-memory request and token counters:

```json
{
  "counts": {
    "POST /v1/chat/completions -> local": 1
  },
  "tokens": {
    "sent:POST /v1/chat/completions -> local": 42,
    "recv:POST /v1/chat/completions -> local": 128,
    "total_sent": 42,
    "total_recv": 128
  }
}
```

#### Reset Counters
```bash
curl -X POST http://localhost:8000/admin/reset-counts
```

Resets in-memory request/token counters and triggers immediate persistence.

## Session-Based Incremental Ingestion

The proxy supports session-based incremental prompt ingestion to reduce CPU usage and latency for multi-turn conversations.

### Strict restore policy (important)

Delta routing is only gated on explicit backend restore evidence when `server.session_require_restore_signal` is enabled. The default configuration (`false`) favors optimistic delta forwarding while still invalidating on history mismatch.

- If restore evidence is missing in API headers/body, the proxy performs compatibility checks against llama-server logs:
  - session-specific restore phrases when available, and
  - restore markers (`restored context checkpoint`, `load_session`, etc.) that appear in log lines appended during the active request window.
- If neither API nor log evidence is found and strict restore is enabled, the proxy sends the full prompt and sets `X-Session-Fallback-Reason: missing_restore_signal`.
- If message history was edited, the proxy invalidates the session and falls back with `X-Session-Fallback-Reason: history_mismatch`.
- When no previous history exists, requests are full-ingestion by design.

### How It Works

1. Client sends a chat completion request with an `X-Session-Id` header (preferred), or one of the compatible headers `session_id`, `X-Client-Request-Id`, or `X-Session-Affinity` (the proxy generates a UUID v4 if none are present).
2. The proxy tracks full message history for each session.
3. For subsequent requests, the proxy computes a delta against prior history.
4. The proxy forwards delta messages **only** when strict restore confirmation has been observed for that session; otherwise it forwards the full prompt.
5. The proxy returns `X-Session-Id`, `X-Session-Created`, `X-Session-Delta`, and (when applicable) `X-Session-Fallback-Reason`.
6. Sessions expire after 3 hours of inactivity and are automatically cleaned up.

### Limitations

- **KV cache ownership**: The proxy never stores or mutates KV tensors; llama-server owns the cache. The proxy only passes session metadata and deltas so llama-server can restore/cache internally.
- **Editing earlier messages invalidates the KV cache**: If a client modifies any earlier message in the conversation, the proxy detects the mismatch and falls back to sending the full history, invalidating the previous session and creating a new one.
- **Context window limits**: llama-server's KV cache has finite capacity. Very long conversations may exceed the context window.
- **Ephemeral sessions**: Sessions are held in memory and are lost when the proxy restarts. Cross-restart persistence is not supported in this version.
- **Per-model delta disable**: Set `force_full_prompt: true` (or `disable_delta: true`) in a model config to always send full history. Use this for models that force full prompt reprocessing (SWA/hybrid/recurrent cache behavior).

### Using Sessions

#### Python Example

```python
import httpx

# Start a session - no X-Session-Id header
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gemma4",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)

# The proxy returns X-Session-Id in the response headers
session_id = response.headers.get("x-session-id")
print(f"Session ID: {session_id}")

# Next turn - include X-Session-Id header (or session_id/X-Client-Request-Id)
response2 = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gemma4",
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "Tell me about sessions."}
        ]
    },
    headers={"X-Session-Id": session_id}
)

# X-Session-Delta header shows incremental ingestion was used
delta = response2.headers.get("x-session-delta")
print(f"Delta request: {delta}")  # "true" when only new messages were sent
```

#### Curl Example

```bash
# First request - proxy generates a session ID
SESSION_ID=$(curl -s -D - http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma4", "messages": [{"role": "user", "content": "Hello"}]}' \
  | grep -i "^x-session-id:" | tr -d '\r' | cut -d' ' -f2)

echo "Session ID: $SESSION_ID"

# Second request - use the session ID
# Only new messages need to be sent (the proxy computes the delta)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: $SESSION_ID" \
  -d '{
    "model": "gemma4",
    "messages": [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi there!"},
      {"role": "user", "content": "What are sessions?"}
    ]
  }'
```

### Session Client Example

A complete Python client example is provided in `examples/session_client.py`:

```bash
# Run 3-turn conversation with session reuse
python examples/session_client.py --model gemma4 --turns 3

# Run comparison benchmark (with vs without session)
python examples/session_client.py --compare
```

### Session Admin Endpoints

```bash
# List all active sessions
GET /admin/sessions

curl http://localhost:8000/admin/sessions

# Delete a specific session
curl -X DELETE http://localhost:8000/admin/sessions/<session-id>
```

### Session Metrics

Session and restore metrics are available on `/admin/metrics`:

```bash
curl http://localhost:8000/admin/metrics
```

Important fields:
- `session_metrics.sessions_active`
- `session_metrics.sessions_created_total`
- `session_metrics.sessions_expired_total`
- `restore_success_total`
- `restore_fallback_total` (per-reason map, e.g. `missing_restore_signal`, `history_mismatch`)
- `delta_payload_bytes_total`
- `single_flight_metrics` (queue/reject/active session counters)
- `guardrail_metrics` (cutoff + invalidation counters)
- `backend_ready`
- `backend_signals`

### Single-flight + guardrails

The proxy enforces **per-session single-flight** by default. Only one in-flight
request per session is allowed; additional requests are queued or rejected based
on config.

Config keys:
- `server.session_single_flight_mode` — `queue` (default) or `reject`
- `server.session_single_flight_max_queue_depth` — max waiting requests per session

Guardrails stop runaway responses and invalidate sessions when configured:
- `server.session_guardrail_max_runtime_seconds` — cutoff streaming after N seconds
- `server.session_guardrail_max_completion_tokens` — cutoff on excessive output
- `server.session_guardrail_repetition_min_pattern_chars` — repetition pattern length
- `server.session_guardrail_repetition_min_repeats` — repetition count to trigger cutoff
- `server.session_guardrail_invalidate_on_cutoff` — invalidate session after runtime/token cutoff
- `server.session_guardrail_invalidate_on_repetition` — invalidate session after repetition cutoff

When a guardrail triggers, `/admin/metrics` exposes `guardrail_metrics` with the
cutoff reason and invalidation counters for observability.

### Slot persistence (KV save/restore)

To avoid llama-server invalidating KV checkpoints between turns, the proxy can
restore and save slot snapshots on each request. This requires llama-server to
run with `--slot-save-path` pointing at the same path.

Config keys:
- `server.session_slot_save_path` — directory for slot snapshot files
- `server.session_slot_pool_size` — number of slots; should match llama-server `--parallel`
- `server.session_slot_timeout_seconds` — timeout for save/restore calls

The proxy restores the slot before each request and saves it after the response.
To avoid slot mismatches, keep `session_slot_pool_size` aligned with
llama-server's `--parallel` setting; for single-slot debugging set both to 1.
If a session is invalidated (history mismatch or guardrail), the slot file is
removed to avoid stale restores. Ensure llama-server is launched with
`--slot-save-path` (see `start-llama.sh` / `models.ini`) and, for SWA/hybrid
models, `--swa-full` to prevent checkpoint invalidation.

### Operator verification checklist

Use these steps to validate strict restore behavior in your environment:

1. Start a chat request without `X-Session-Id`; capture returned `X-Session-Id`.
2. Repeat with the same session id and appended messages.
3. Check headers:
   - `X-Session-Delta: true` indicates delta forwarding.
   - `X-Session-Delta: false` plus `X-Session-Fallback-Reason` explains fallback.
4. If fallback is `missing_restore_signal`, inspect llama-server logs for session-specific restore lines (for example phrases containing `load_session` and the same session id).
5. Check `/admin/metrics`:
   - `restore_success_total` increases when backend restore evidence is observed.
   - `restore_fallback_total` increments by reason when strict policy blocks delta.
   - `delta_payload_bytes_total` grows only when delta forwarding is used.
6. Confirm payload reduction over repeated turns (target baseline >=30% reduction for representative conversations).

## Prometheus metrics

The proxy exposes Prometheus exposition-format metrics at `/metrics` (text/plain).
Available metrics (best-effort):

- `llama_process_rss_bytes` (gauge) — process RSS in bytes.
- `llama_model_rss_bytes{model="..."}` (gauge) — estimated per-model RSS (when multiple models are loaded the proxy divides process RSS evenly across models as an approximation).
- `llama_model_load_events_total{model="...",event="load|unload"}` (counter) — model lifecycle events.
- `llama_models_loaded` (gauge) — number of loaded models reported by router-mode.

Example Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: 'llama-proxy'
    static_configs:
      - targets: ['localhost:8000']
```

Alerting rules (warning at 75% of 90GB; critical at 90GB) are provided in `monitoring/llama_memory_alerts.yaml`.
A minimal Grafana dashboard JSON is included at `monitoring/grafana_llama_memory_dashboard.json`.

## Model Switching Behavior

When a request specifies a model that differs from the currently loaded model:

1. The proxy stops the current llama-server
2. Starts llama-server with the new model
3. Waits for the server to be ready (up to `llama_startup_timeout` seconds)
4. Forwards the request

If loading fails, the proxy returns HTTP 503 with a `Retry-After` header:

```json
{
  "error": "model_loading",
  "message": "Failed to load model qwen2.5",
  "retry_after": 30
}
```

Clients should handle this by retrying after the specified delay.

### Crash-path safeguards

The proxy now adds bounded backend retries for transient local transport failures
(connect/read/timeout). Retry behavior is controlled by:

- `server.backend_retry_attempts`
- `server.backend_retry_base_delay_seconds`
- `server.backend_retry_max_delay_seconds`
- `server.backend_retry_jitter_ratio`

Concurrency pressure is controlled by `server.max_concurrent_queries` (default 4).
When the guard rejects a request, the proxy returns a 503 and increments
`backend_signals.concurrency_rejects`.

A watchdog loop monitors the child llama-server process. If the process exits,
health switches to `degraded`, backend readiness is gated to `ready: false`, and
router mode attempts a best-effort restart.

#### Fault-injection validation (reproducible)

Run the crash-path repro script:

```bash
cd proxy
./scripts/fault-injection-backend-crash.sh
```

Artifacts are written under `proxy/logs/fault-injection/run-<timestamp>/`.
Expected triage signatures include:

- `backend_retry path=... signal=connect_failures|read_failures|timeout_failures`
- `concurrency_reject active=... max=...`
- `watchdog detected llama-server exit code=...`
- `watchdog router restart backend_ready=...`

Use `/health` and `/admin/metrics` snapshots from the run directory to verify
readiness transitions and backend signal counters.

### Router Mode (Multi-Model)

When `llama_router_mode` is enabled, the proxy launches llama-server in router mode and
preloads the embeddings model plus the configured primary model. Requests are routed
to the appropriate model without stopping the server. The router exposes management
endpoints like `GET /models` and `POST /models/load`.

## Logging

Logs are written with time-based rotation:

**Production:** `/var/log/llama-proxy/` (if you configure logging to this directory when running under a service manager)
**Development (no root):** `./logs/` (fallback when `/var/log/llama-proxy` is not writable)

### Log Files

| File | Description | Rotation |
|------|-------------|----------|
| `proxy.log` | Proxy server logs (requests, responses, lifecycle) | Every 6 hours, 90 days retention |
| `llama-server.log` | llama-server stdout/stderr | On each restart, last 15 kept |
| `request_counts.json` | Persisted request counters (endpoint keys) | Updated periodically and on reset |
| `token_counts.json` | Persisted token counters (endpoint keys + totals) | Updated periodically and on reset |

### Proxy Log Settings
- **Rotation**: Every 6 hours
- **Retention**: 90 days
- **Format**: `TIMESTAMP - LEVEL - MESSAGE`

Log entries include:
- Incoming requests (method, path, target, body preview)
- Responses (status code, body preview)
- Streaming chunks (debug level)
- Server lifecycle events

### llama-server Logs

The llama-server output is captured to `llama-server.log`. Each time the server restarts (e.g., model switch), the current log is rotated:
- `llama-server.log` → `llama-server.1.log` → `llama-server.2.log` → ...
- The last 15 log files are retained

View logs:
```bash
# Development logs
tail -f ./logs/proxy.log
tail -f ./logs/llama-server.log

# Production logs
tail -f /var/log/llama-proxy/proxy.log
tail -f /var/log/llama-proxy/llama-server.log

# Systemd journal
journalctl -u llama-proxy -f
```

## Troubleshooting

### Server won't start
1. Check if port 8000 is in use: `lsof -i :8000`
2. Check logs: `journalctl -u llama-proxy -n 50`
3. Verify config syntax: `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`

### Model fails to load
1. Test start script manually: `/home/rgardler/projects/llm/start-llama.sh qwen3`
2. Check llama-server port: `lsof -i :8080`
3. Increase `llama_startup_timeout` in config

### Remote API errors
1. Verify API key is set: `echo $OPENAI_API_KEY`
2. Check endpoint URL in config
3. Review proxy logs for error details

## Testing

The project includes end-to-end tests using Playwright to verify the Web UI and SSE functionality.

### Python tests (pytest)

The proxy backend includes Python tests. To run them from the `proxy` folder:

```bash
# from repository root
cd proxy
source .venv/bin/activate   # if you have a virtualenv
# or create one: python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Some integration tests expect a local proxy/llama-server to be running (see test files for details).

### Setup

```bash
# Install test dependencies
npm install

# Install Playwright browsers
npx playwright install chromium
```

### Running Tests

```bash
# Run all tests
npx playwright test

# Run specific test file
npx playwright test status-bar.spec.js

# Run tests matching a pattern
npx playwright test --grep "SSE"

# Run with headed browser (visible)
npx playwright test --headed

# Run diagnostic tests
npx playwright test diagnostic.spec.js
```

### Test Coverage

The test suite covers:

- **Status bar elements** - Verifies IDs and initial state
- **SSE connection** - Confirms EventSource connects to `/events`
- **SSE initial status** - Validates initial status message is received
- **Model switch via API** - Tests that external API calls trigger UI updates via SSE
- **Load Model button** - Tests UI button triggers status updates
- **API passthrough tests** - Tests model switching when using test buttons

## Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Client    │────▶│  LLama Proxy    │────▶│  llama-server   │
│             │     │  (port 8000)    │     │  (port 8080)    │
└─────────────┘     └────────┬────────┘     └─────────────────┘
                             │
                             │              ┌─────────────────┐
                             └─────────────▶│  OpenAI API     │
                                            └─────────────────┘
                                            ┌─────────────────┐
                                            │  Anthropic API  │
                                            └─────────────────┘
```

## License

See repository root for license information.
