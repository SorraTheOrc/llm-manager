# LLama Proxy Server

A proxy server that routes OpenAI-compatible API requests to either a local llama-server or remote API services based on configuration.

## Features

- **Unified API Endpoint**: Single endpoint for all LLM requests, regardless of backend
- **Web UI**: Built-in dashboard for monitoring, model switching, and API testing
- **Local Model Management**: Automatically starts/stops llama-server with the correct model
- **Remote API Routing**: Forward requests to OpenAI, Anthropic, or other OpenAI-compatible APIs
- **Hot Model Switching**: Automatically switches local models when a different model is requested
- **Provider Fallback**: Automatic failover between providers per model with configurable cooldown and circuit breaker
- **Streaming Support**: Full support for streaming responses (SSE)
- **Client Disconnect Detection**: Automatically detects client disconnections during streaming, cancels in-flight backend processing, releases scheduler slots, removes queued jobs, and maintains accurate `active_queries` counters
- **Request/Response Logging**: Comprehensive logging with time-based rotation. INFO-level request log lines now include the resolved session ID (`session_id=<value>`), assigned slot ID (`slot=<value>` or `slot=none`), and a body preview that excludes system-prompt content to prevent sensitive system-prompt data from leaking into logs. Console output for STREAM CHUNK messages now prints only the streamed text content (delta.content) to reduce noisy JSON envelopes in the terminal; rotating file logs continue to record the full JSON chunk records unchanged.
- **Request + Token Counters**: In-memory counters with periodic JSON persistence
- **Session-Based Incremental Ingestion**: Reduce CPU and latency with per-session KV cache reuse
- **Live Log Tail + Stats**: `/logs` UI and `/logs/tail` SSE stream for logs/counts/tokens
- **Host-first Deployment**: systemd service units for llama-server and proxy with host-based startup model
-- Systemd integration details removed: the repository no longer distributes systemd unit files. Run the proxy manually or manage service units outside this repo. See [Host-first deployment](#host-first-deployment) for example systemd units.

## Request Logging

The proxy logs every incoming request at **INFO** level via the `llama-proxy` logger. This is visible in normal proxy logs (no need for debug mode).

### Log Format

#### Local proxy path (llama-server)

```
[local] POST http://localhost:8080/v1/chat/completions body={...} session_id=sess-abc123 slot=7 session={"x-session-id": "sess-abc123"}
```

#### Remote proxy path

```
[remote] POST http://localhost:8080/v1/chat/completions -> https://api.openai.com/v1/chat/completions body={...} session_id=sess-abc123 slot=none session={"x-session-id": "sess-abc123"}
```

### Fields

| Field | Description |
|-------|-------------|
| `session_id=<value>` | The resolved session ID (internal identifier used by the session manager). Omitted when no session is resolved. |
| `slot=<value>` | The assigned slot identifier. Uses `"none"` when no slot is assigned or for the remote proxy path. Uses `"queued"` when the request is waiting in the scheduler queue. |
| `body={...}` | A truncated preview of the request body (first 500 chars). **System message content is excluded** from the preview to prevent sensitive system-prompt data from appearing in logs. Only user-facing message content (role="user" and role="assistant") is included. |
| `session={...}` | The session-related headers extracted from the request (X-Session-Id, X-Client-Request-Id, X-Session-Affinity). |

### System Prompt Redaction

The body preview automatically filters out messages with `role: "system"` to prevent sensitive system-prompt content from leaking into proxy logs. Only `role: "user"` and `role: "assistant"` messages are included in the preview.

## Host-first deployment

The repository supports two deployment models for running llama-server:

- **Host-first (systemd)** — llama-server runs directly on the host, managed by systemd. The example unit files in `docs/systemd/` call `start-llama.sh` directly for direct host execution.
- **Proxy-managed (container)** — When the proxy manages llama-server startup via `start_llama_server()` (in `proxy/proxy/lifecycle.py`), it uses the configured `llama_start_script` (default: `scripts/podman_start_llama.sh`) which runs llama-server inside a podman container.

### Host vs. Container startup

| Mode | How it starts | Managed by | Use case |
|------|--------------|------------|----------|
| **Host-first (host-direct)** | Proxy's `start_llama_server()` calls `start-llama.sh` directly on the host (single attempt) before falling back to the configured container script | Proxy lifecycle | When `llama_allow_host_fallback: true` (default), preferred path for proxy-managed startups |
| **Proxy-managed (container)** | Proxy's `start_llama_server()` runs the configured `llama_start_script` (e.g., `scripts/podman_start_llama.sh`) | Proxy lifecycle | Development, ad-hoc proxy startups, or when host-direct fails |
| **Host-first (systemd)** | Systemd unit calls `start-llama.sh` directly on the host | `systemctl --user` or `sudo systemctl` | Production deployments, reboot-safe supervision |

The container path is **not deprecated** and remains fully supported. It is the default configured `llama_start_script` in `config.yaml`. The host-first fallback (`llama_allow_host_fallback: true`) attempts host-direct startup first and falls back to the container script if the host-direct attempt fails. This provides the best of both worlds: fast host-direct startup when available, with automatic fallback to container-based operation.

To disable the host-first fallback and use the container script exclusively (matching the original pre-host-first behavior), set:

```yaml
server:
  llama_allow_host_fallback: false
```

### Startup behavior (proxy-managed mode)

The proxy starts llama-server via the configured `llama_start_script` (from `proxy/proxy/lifecycle.py`):
1. The proxy invokes the configured `llama_start_script` with the target model
2. If the start script fails, the proxy retries up to 4 times with exponential backoff
3. If all attempts fail, the proxy reports a clear error message

### Configuration

```yaml
server:
  llama_start_script: /path/to/start-llama.sh  # Script to start llama-server
  llama_allow_host_fallback: true               # Allow host-direct start as fallback
```

**`llama_allow_host_fallback`** (boolean, default: `true` in the shipped config):
- When `true`: the proxy may attempt starting llama-server directly on the host (via the configured script) if the primary container-based startup fails.
- When `false`: the proxy uses only the configured `llama_start_script` (container-based) and does NOT attempt host-direct fallback.
- To disable host-fallback, set `llama_allow_host_fallback: false` in the `server:` section of `config.yaml`.

> **Note:** When running under systemd (host-first mode), the systemd unit calls `start-llama.sh` directly and bypasses the proxy\'s startup logic entirely. The `llama_allow_host_fallback` config option only affects the proxy-managed startup path.

### Systemd service units

Example systemd unit files are provided in `docs/systemd/`:
- `docs/systemd/llama-server.service` — llama-server service unit
- `docs/systemd/llama-proxy.service` — proxy server service unit

Both units document two deployment approaches:
- **User service** (recommended) — runs under the operator's login session, uses `~/.config/systemd/user/`
- **System service** — runs independently of user sessions, uses `/etc/systemd/system/`

See the individual unit files for setup commands, configuration, and verification steps.

### Log paths

The proxy distinguishes between development and production log paths:

| Mode | Proxy logs | llama-server logs |
|------|------------|-------------------|
| Development | `./logs/proxy.log` | `./logs/llama-server.log` |
| Production (user service) | `$XDG_STATE_HOME/llama-proxy/logs/proxy.log` | `$XDG_STATE_HOME/llama-server/logs/llama-server.log` |
| Production (system service) | `/var/log/llama-proxy/proxy.log` | `/var/log/llama-server/llama-server.log` |

### Verification

Verify services are running after deployment:

```bash
# Check llama-server health
curl http://localhost:8080/health

# Check proxy health
curl http://localhost:8000/health

# View systemd logs (user service)
journalctl --user -u llama-server.service -f
journalctl --user -u llama-proxy.service -f

# View systemd logs (system service)
sudo journalctl -u llama-server.service -f
sudo journalctl -u llama-proxy.service -f
```

## Requirements

## Requirements

- Python 3.10+
- Podman with a container image containing llama-server (llama.cpp)
- `/home/rgardler/projects/llm/start-llama.sh` script for starting llama-server

See `../LLAMA_README.md` for build and setup instructions.

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
proxyctl start              # start using proxy/config.yaml llama_start_script or start-llama.sh
proxyctl start --dev        # start dev instance (port 8001, DEBUG logging, auto-reload)
proxyctl status             # show running status and PID
proxyctl status --dev       # show dev instance status
proxyctl logs               # tail the proxy logs
proxyctl logs --dev         # tail dev instance logs
proxyctl stop               # stop the running proxy
proxyctl stop --dev         # stop the dev instance
```

The script respects `LLAMA_START_SCRIPT` environment variable and `proxy/config.yaml` `server.llama_start_script` entry when determining what to run. Use the `--dev` flag to run in development mode, or set `LLAMA_PROXY_DEV=1` to force dev mode via environment variable.

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
  llama_models_max: 1
  llama_server_port: 8080
  llama_startup_timeout: 300
  session_single_flight_mode: "queue"
  session_single_flight_max_queue_depth: 1
  session_slot_save_path: "/home/rgardler/projects/llm/slot-cache"
  session_slot_pool_size: 1
  session_slot_timeout_seconds: 3.0
  session_guardrail_max_runtime_seconds: 1800
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

# Audit model configuration
# The audit skill uses these settings to determine which model to call
# for audit operations. See proxy/provider_resolver.py for details.
audit_model: "deepseek-v4-flash-free"
audit_model_fallbacks:
  - "openrouter/free"
  - "deepseek-v4-flash"

# Model routing
models:
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

  provider-a:
    type: "remote"
    endpoint: "https://api.provider-a.com/v1"
    api_key_env: "PROVIDER_A_KEY"
    aliases:
      - "my-model-*"      # Wildcard: matches my-model-anything
      - "my-model"
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

provider-a:
  aliases: ["my-model-*"]            # Wildcards route ALL my-model-* to provider-a
```

- Request for `my-model` → Routes to provider-a
- Request for `my-model-extra` → Routes to provider-a
- Request for `qwen3-coder` → Routes to local qwen3

### Friendly Model Aliases

In addition to model IDs and wildcard patterns, the proxy supports short,
human-friendly aliases that map directly to configured model presets.
These are useful when callers should not need to know internal preset IDs.

**Example configuration:**
```yaml
qwen3-next:
  providers:
    - name: local-qwen3-next
      type: local
      llama_model: Qwen3-Next
  aliases:
    - qwen3-next
    - qwen3-coder-next
    - plan       # Route requests for model="plan" to Qwen3-Next
    - code       # Route requests for model="code" to Qwen3-Next
```

Requests with `model: "plan"` or `model: "code"` are routed to the `qwen3-next`
preset. The same case-insensitive and wildcard precedence rules apply:

- `"plan"`, `"Plan"`, `"PLAN"` → all resolve to `qwen3-next`
- `"code"`, `"Code"`, `"CODE"` → all resolve to `qwen3-next`
- Exact aliases (`plan`, `code`) take precedence over wildcard patterns

Friendly aliases are listed in the `/v1/models` response and can be discovered
programmatically by inspecting the `aliases` field of each model entry.

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

### Audit Model Configuration

The audit skill (`skill/audit/`) uses the `audit_model` and `audit_model_fallbacks`
settings to determine which model to call for audit operations. The resolver
maps short model names to provider-prefixed model identifiers.

#### Configuration

```yaml
audit_model: "deepseek-v4-flash-free"
audit_model_fallbacks:
  - "openrouter/free"
  - "deepseek-v4-flash"
```

#### Resolution

The resolver maintains a static mapping table (in `proxy/provider_resolver.py`) that
maps well-known short names to provider-prefixed model IDs:

| Short name | Resolved IDs |
|------------|-------------|
| `deepseek-v4-flash-free` | `opencode/deepseek-v4-flash-free`, `openrouter/openrouter/free`, `opencode-go/deepseek-v4-flash` |
| `deepseek-v4-flash` | `opencode/deepseek-v4-flash`, `opencode-go/deepseek-v4-flash`, `openrouter/deepseek/deepseek-v4-flash` |
| `openrouter/free` | `openrouter/openrouter/free`, `opencode/deepseek-v4-flash-free`, `opencode-go/deepseek-v4-flash` |
| `free-model` | `openrouter/openrouter/free`, `opencode/deepseek-v4-flash-free`, `opencode-go/deepseek-v4-flash` |

Canonical model IDs were discovered via `pi --list-models`:
- `opencode/deepseek-v4-flash-free`
- `opencode-go/deepseek-v4-flash`
- `openrouter/openrouter/free`
- `openrouter/deepseek/deepseek-v4-flash`
- `opencode/deepseek-v4-flash`

#### Fallback Behaviour

When the primary `audit_model` fails to resolve, the resolver attempts each
fallback in order. If all names fail, startup validation logs a warning (lenient
mode, default) or errors (strict mode). The default strictness is lenient to
avoid blocking startup when models are temporarily unavailable.

#### Startup Validation

At server startup the resolver validates that at least one valid model can be
resolved. Configuration:

```yaml
# Not yet exposed in config.yaml — resolved programmatically from the resolver
# Default: strict=false (lenient — logs warnings but does not fail startup)
```

#### Observability

The resolver produces structured logs and a lightweight metric:

- **Log**: `INFO` on successful resolution, `WARNING` on unresolvable names
- **Metric**: `provider_resolver_unresolved_total` — count of unresolvable
  lookups keyed by short name

#### Migration Note

The SorrasAgents integration will consume this resolver in a separate work item
(LP-0MQGO0PWJ001R5QI covers the resolver implementation; SorrasAgents update
follows). Until then, the resolver is available for standalone testing via the
CLI:

```bash
cd proxy && source .venv/bin/activate
python -m proxy.provider_resolver deepseek-v4-flash-free openrouter/free
# Output: Resolved to: opencode/deepseek-v4-flash-free, openrouter/openrouter/free, opencode-go/deepseek-v4-flash
```

### Provider Fallback

Each model can define an ordered list of `providers` for automatic failover. The `providers` list replaces the old top-level `endpoint`, `api_key_env`, and `headers` fields on each model entry. This is a **breaking change** — existing flat-format model entries must be migrated to the provider list format.

When a request arrives for a model with a `providers` list, the proxy tries each provider in order. If a provider fails (connection error, timeout, HTTP 4xx/5xx, or slot exhaustion for local models), the proxy immediately tries the next provider in the list. This continues until a provider succeeds or all providers are exhausted.

#### Provider Entry Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Unique identifier for this provider entry |
| `type` | string | yes | `"local"` or `"remote"` |
| `endpoint` | string | remote | Base URL of the remote API |
| `api_key_env` | string | remote | Environment variable containing the API key |
| `headers` | dict | remote (optional) | Additional headers to include |
| `llama_model` | string | local | Name of the local model |

#### Example: Remote Fallback

```yaml
models:
  mimo-v2.5:
    providers:
      - name: remote-primary
        type: remote
        endpoint: https://api.provider-a.com/v1
        api_key_env: PROVIDER_A_KEY
      - name: remote-fallback
        type: remote
        endpoint: https://api.provider-b.com/v1
        api_key_env: PROVIDER_B_KEY
    aliases:
      - mimo*
```

In this example, requests for `mimo-v2.5` first try the primary remote provider. If that fails, they fall back to the secondary provider.

#### Example: Local-to-Remote Fallback

```yaml
models:
  hybrid-model:
    providers:
      - name: local-llama
        type: local
        llama_model: Qwen3
      - name: remote-fallback
        type: remote
        endpoint: https://api.provider-a.com/v1
        api_key_env: PROVIDER_A_KEY
    aliases:
      - hybrid*
```

In this example, requests first try the local `Qwen3` model. If the local server is unavailable, has no available slots, or returns errors, the request falls back to the remote provider.

#### Unavailability Detection

A provider is considered unavailable when:
- **Connection error**: DNS failure, connection refused, timeout
- **HTTP error**: Response with status >= 400 (4xx or 5xx)
- **Slot exhaustion** (local models): `available_slots == 0` and `total_slots > 0`

#### Cooldown / Circuit Breaker

After a provider fails, it is marked as unavailable for a cooldown period. During cooldown, subsequent requests skip that provider and try the next available one. This prevents wasting time on known-bad endpoints.

- **Default cooldown**: 60 seconds
- **Configuration**: Set `server.provider_cooldown_seconds` in `config.yaml`
- **Retry-After**: If the upstream response includes a `Retry-After` header, the larger of the configured cooldown and the header value is used
- **State**: Cooldown state is in-memory only and resets when the proxy restarts
- **Scope**: Cooldown state is global across all sessions within a single proxy process.
  When a provider fails in one session, all other sessions immediately see it as
  unavailable until the cooldown expires. Multi-worker deployments (multiple
  proxy processes) have independent cooldown state per worker.

##### Cross-Request Stall Circuit Breaker (Tier 3)

In addition to the per-failure cooldown above (Tier 2), the proxy includes a
cross-request stall circuit breaker (Tier 3) that tracks stall frequency per
provider within a sliding time window. When a provider exceeds the stall
threshold within the window, it is marked unavailable via the same cooldown
mechanism, affecting subsequent requests immediately.

This prevents unreliable providers from stalling on every request, since the
per-stream retry count resets between requests. A provider that stalls
repeatedly across requests is quarantined after the threshold is exceeded.

- **Sliding window**: 300 seconds (default)
- **Stall threshold**: 3 stalls (default) within window triggers cooldown
- **Cooldown duration**: 180 seconds (default) — separate from provider_cooldown_seconds
- **Configuration keys** (optional, defaults apply when absent):
  - `server.upstream_stall_window_seconds` (default: 300)
  - `server.upstream_stall_threshold` (default: 3)
  - `server.upstream_stall_cooldown_seconds` (default: 180)
- **State**: In-memory only, resets on proxy restart. Shared across all sessions.
- **Integration**: Uses the same `mark_provider_unavailable()` mechanism as Tier 2.
  Stalls during cooldown are recorded but do not extend the cooldown.

#### All Providers Exhausted

When all providers are exhausted:
- **Slot exhaustion** (all providers were local and had no slots): Returns HTTP 429 (Too Many Requests) with `Content-Type: text/plain` and body `"Model server busy: 0/<total_slots> slots available. Retry later."` (no `Retry-After` header).
- **Other errors**: Returns HTTP 503 with JSON body containing `retry_after` field.

#### Observability

When a fallback occurs:
- **Response header**: `X-Provider: <provider-name>` is added to the response, indicating which provider handled the request.
- **Logging**: An INFO-level log is emitted:
  ```
  Fallback triggered for model=v1/chat/completions, from=remote-primary, to=remote-fallback, reason=HTTP 502
  ```

#### Migration Guide

To migrate from the old flat format to the new `providers` list format:

1. Replace the top-level `endpoint`, `api_key_env`, and `headers` fields with a `providers` list.
2. Each entry in the `providers` list must have a `name`, `type`, and type-specific fields.
3. For single-provider models, simply wrap the existing config in a `providers` list.

**Before (old format):**
```yaml
models:
  my-model:
    type: remote
    endpoint: https://api.provider-a.com/v1
    api_key_env: PROVIDER_A_KEY
    aliases:
      - my-model
      - my-model-*
```

**After (new format):**
```yaml
models:
  my-model:
    providers:
      - name: my-provider
        type: remote
        endpoint: https://api.provider-a.com/v1
        api_key_env: PROVIDER_A_KEY
    aliases:
      - my-model
      - my-model-*
```

**Important:** The old top-level `endpoint`/`api_key_env` format is **deprecated** and will be removed in a future release. All models must use the `providers` list format.

### Custom System Prompts

Each model entry may include a `system_prompt` configuration to inject a default
system prompt into all requests targeting that model or any of its aliases.

#### Configuration

```yaml
models:
  assistant-model:
    type: local
    llama_model: gemma4
    aliases:
      - "assistant"
      - "asst"
    system_prompt:
      mode: "prepend"       # "override" or "prepend" (required)
      file: "proxy/prompts/assistant.txt"
```

#### Modes

| Mode | Behaviour |
|------|-----------|
| `override` | Replaces **all** client-supplied system messages with the prompt file content. |
| `prepend`  | Inserts the prompt file content as the first system message **before** any client-supplied system messages. |

If `system_prompt` is present but `mode` is missing or invalid, config validation
fails at startup with a clear error message.

#### Prompt file resolution precedence

When a request arrives, the proxy looks for the prompt file in this order:

1. **Local override**: `.sorraAgents/prompts/<alias>.txt` (project-local, not committed)
2. **Repo default**: The path specified in `system_prompt.file`, resolved against the repo root
3. **No prompt** if neither exists

This allows operators to deploy per-instance prompt overrides without modifying
the repository. Override files are looked up by alias name (e.g.,
`.sorraAgents/prompts/assistant.txt` for the `assistant` alias).

#### File format and limits

- Files must be **plain text UTF-8**.
- Maximum file size: **64 KB**. Files larger than this are ignored and logged.
- Non-UTF-8 files are ignored and logged.
- **No caching**: prompt files are read on every request.

#### Security considerations

- **Do not store secrets or credentials in prompt files.** Prompts are plaintext
  and may appear in logs, upstream request payloads, or the debug endpoint.
- Local override files (`.sorraAgents/prompts/`) are outside the repository
  and should be managed via deployment tooling, not committed.
- Enable the debug endpoint (see below) only for development or QA.

#### Example prompt file

Create `proxy/prompts/assistant.txt`:

```
You are a helpful and knowledgeable assistant. Your responses should be
concise, accurate, and well-structured. When asked to write code, include
comments and follow best practices.
```

#### Debug endpoint

When enabled, the debug endpoint at `GET /debug/prompt?alias=<alias>` returns
sanitized information about the resolved prompt:

```json
{
  "alias": "assistant",
  "resolved": true,
  "mode": "prepend",
  "source_path": "/path/to/prompt.txt",
  "content_preview": "You are a helpful...",
  "size_bytes": 182
}
```

To enable, set `server.debug: true` in `config.yaml`:

```yaml
server:
  debug: true
```

By default, the endpoint returns only the first 200 characters of content
(`content_preview`). Pass `&full=true` to get the full content when debug
mode is enabled. Without debug mode, the endpoint is accessible only from
localhost (127.0.0.1).

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LLAMA_PROXY_CONFIG` | Path to config file (default: `./config.yaml`) |
| `LLAMA_PROXY_DEV` | Set to `1` to enable dev mode (alternative to `--dev` flag) |
| `LLAMA_START_SCRIPT` | Override the start script path (used by proxyctl) |
| `OPENAI_API_KEY` | API key for OpenAI |
| `ANTHROPIC_API_KEY` | API key for Anthropic |
| `PROXY_PORT` | Override proxy web server port (default: 8000 prod, 8001 dev) |
| `LLAMA_SERVER_PORT` | Override llama-server backend port (default: 8080 prod, 8081 dev) |
| `PORT` | Override backend port (alias for LLAMA_SERVER_PORT) |
| `XDG_STATE_HOME` | Base dir for state (defaults to `~/.local/state`) |

### Upstream Timeout Configuration

The proxy uses two separate timeout values for upstream remote connections:

| Config Key | Default | Description |
|-----------|---------|-------------|
| `server.upstream_idle_timeout_seconds` | `30` | Per-chunk idle timeout for SSE streaming. When the upstream stops sending data mid-stream without closing the connection, the proxy waits this long for the next chunk before detecting a stall. Reduced from 60s to 30s for faster stall detection (LP-0MRFEXXVC001RYKB). Operators with long-thinking models may increase this value. |
| `server.upstream_retry_connect_timeout_seconds` | `30` | Timeout for establishing a retry connection after a stall. Decoupled from the idle timeout so operators can tune retry connection timeouts independently (typically shorter). |
| `server.upstream_retry_max_attempts` | `3` | Maximum number of retry attempts (initial attempt + retries) for a stalled upstream stream. Aligned with Pi's default maxRetries=3. |
| `server.upstream_retry_base_delay_seconds` | `2.0` | Base delay for exponential backoff between retries. The actual delay is `min(base_delay * 2^attempt, max_delay)`. Aligned with Pi's default maxRetryDelayMs=60000. |
| `server.upstream_retry_max_delay_seconds` | `60.0` | Maximum delay between retries (cap on exponential backoff). |
| `server.upstream_request_timeout_seconds` | `120` | Upstream request-level timeout (LP-0MRF77A0E0026B9T). Caps the read timeout for the initial HTTP response from the upstream provider. Prevents 15+ minute silent hangs when the upstream is slow to respond or silently returns empty content. Different from the per-chunk idle timeout (`upstream_idle_timeout_seconds`) which detects mid-stream stalls. |
| `server.upstream_empty_retry_max_attempts` | `1` | Maximum number of additional attempts when an upstream returns a semantically empty response (no content, stopReason: stop, total_tokens: 0). Default: 1 retry. |
| `server.upstream_empty_retry_base_delay_seconds` | `2.0` | Base delay (in seconds) before retrying on empty response. |
| `server.upstream_stall_window_seconds` | `300` | Sliding window duration (seconds) for the cross-request stall circuit breaker (Tier 3). Stalls older than this are ignored when counting toward the threshold. |
| `server.upstream_stall_threshold` | `3` | Number of stalls within the sliding window that triggers the circuit breaker to mark the provider unavailable for the cooldown duration. |
| `server.upstream_stall_cooldown_seconds` | `180` | Cooldown duration (seconds) applied when the stall circuit breaker threshold is exceeded. Separate from `provider_cooldown_seconds` (Tier 2 cooldown). |

The retry connection timeout (`upstream_retry_connect_timeout_seconds`) controls how long the proxy waits for a retry connection to be established before counting the retry as failed and either retrying again (with exponential backoff) or exhausting retries. The per-chunk idle timeout (`upstream_idle_timeout_seconds`) controls how long the proxy waits between SSE chunks before detecting a stall.

### Remote HTTP Client Configuration

The proxy uses a shared, pooled `httpx.AsyncClient` for all remote upstream requests, replacing per-request client creation. This enables connection reuse (TCP/TLS keepalive) and configurable connection-level timeouts.

Configuration is under `server.remote_http_client`:

| Config Key | Default | Description |
|-----------|---------|-------------|
| `server.remote_http_client.connect_timeout_seconds` | `30` | Timeout for establishing new connections to upstream providers. |
| `server.remote_http_client.read_timeout_seconds` | `300` | Read timeout for the entire response. Set generously to avoid interfering with per-request adaptive timeouts. |
| `server.remote_http_client.pool_connections` | `50` | Maximum number of connections in the pool. |
| `server.remote_http_client.pool_keepalive_connections` | `10` | Maximum number of idle keepalive connections to maintain. |
| `server.remote_http_client.keepalive_seconds` | `60` | Time in seconds before an idle keepalive connection is closed. |

The pool is initialized at server startup and closed on shutdown. Per-request adaptive timeouts (see Adaptive Timeouts) still apply via the `timeout` parameter passed to each request, independent of the pool's connection-level timeouts.

Retry behavior uses bounded exponential backoff: `delay = min(base_delay * 2^attempt, max_delay)`. After all retry attempts are exhausted, the provider-level cooldown (see Provider Fallback) applies separately, preventing immediate retry by subsequent requests.

### Development Mode

The proxy supports a development mode that allows running a dev instance side-by-side with the production proxy. Dev mode uses alternate ports, DEBUG logging, and auto-reload for rapid iteration.

#### Dev Mode Ports

| Component | Default (prod) | Default (dev) | Overridable via |
|-----------|---------------|---------------|----------------|
| Proxy web server | 8000 | 8001 | `PROXY_PORT` env var |
| llama-server backend | 8080 | 8081 | `LLAMA_SERVER_PORT` or `PORT` env var |

#### Using Dev Mode with proxyctl

```bash
# Start dev instance (auto-reload + DEBUG logging)
proxyctl start --dev

# Check dev instance status
proxyctl status --dev

# View dev logs
proxyctl logs --dev

# Restart dev instance
proxyctl restart --dev

# Stop dev instance
proxyctl stop --dev
```

#### Direct uvicorn invocation

You can also run the proxy directly with uvicorn for development:

```bash
source .venv/bin/activate
export LLAMA_PROXY_DEV=1
python -m uvicorn proxy.server:app --host 0.0.0.0 --port 8001 --reload --log-level debug
```

#### Dev Mode Details

- **Opt-in only**: Dev mode must be explicitly enabled via `--dev` flag or `LLAMA_PROXY_DEV=1` environment variable
- **Separate PID file**: Dev instances use `proxy.dev.pid` to avoid conflicts with production
- **Separate log directory**: Dev logs go to `$XDG_STATE_HOME/llama-proxy-dev/logs/` (or `~/.local/state/llama-proxy-dev/logs/`)
- **Auto-reload**: Code changes trigger automatic server restarts during development
- **DEBUG logging**: All log levels are emitted for maximum visibility during development
- **No production impact**: Dev mode does not modify production config or defaults

> **Warning**: Ensure no other service is listening on port 8001 or 8081 before starting the dev instance.

Systemd-specific instructions removed. If you run systemd units outside this repository, add environment variables via `systemctl edit <unit>` or your system's preferred method.

## Usage

### Starting the Server

**With proxyctl (recommended):**
```bash
# Production (default port 8000)
proxyctl start

# Development (port 8001, DEBUG logging, auto-reload)
proxyctl start --dev
```

**Direct uvicorn:**
```bash
source .venv/bin/activate
# Production
python -m uvicorn proxy.server:app --host 0.0.0.0 --port 8000

# Development (with auto-reload and DEBUG logging)
LLAMA_PROXY_DEV=1 python -m uvicorn proxy.server:app --host 0.0.0.0 --port 8001 --reload --log-level debug
```

Note on start-proxy.sh hardening

The bundled `proxy/scripts/start-proxy.sh` script now prefers the virtualenv Python interpreter (`.venv/bin/python3`) when available, falls back to the system `python3`, and will set `PYTHONPATH` to the repository root if it is not already set to avoid import errors when running from the repository checkout.

Before launching the server the script also checks whether the selected port (default `8000`, or overridden with `--port`) is already in use on the local host. If the port is occupied the script exits with a helpful message indicating the port and suggesting `proxyctl start --dev` or running with a different `--port` value.

This makes manual invocations of the proxy more robust across developer environments and reduces confusing import or bind errors when starting the server directly.
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

#### LLama Local Status

Internal endpoint that returns a small JSON object describing the current status
of the local llama.cpp (llama-server) process.

```bash
curl http://localhost:8000/llama/local/status
```

Response:
```json
{
  "active_query": false,
  "model_switch_in_progress": false,
  "current_model": "qwen3",
  "llama_server_running": true,
  "available_slots": 2,
  "total_slots": 4
}
```

**Fields:**

| Field                     | Type          | Description                                                                 |
|---------------------------|---------------|-----------------------------------------------------------------------------|
| `active_query`            | `bool`        | `true` while a request is being processed (at least one in-flight request). |
| `model_switch_in_progress`| `bool`        | `true` during a background model load or model switch.                     |
| `current_model`           | `string|null` | Name of the currently loaded model, or `null` when no model is loaded.     |
| `llama_server_running`    | `bool`        | `true` when the llama-server process is running and responsive.            |
| `available_slots`         | `int`         | Number of model-serving slots that are currently idle (not processing).     |
| `total_slots`             | `int`         | Total number of model-serving slots configured on the llama-server.         |

**Performance:**

The endpoint is designed to remain responsive even under load. The underlying
`query_llama_status()` call is wrapped with a configurable timeout
(default: 1 second, overridable via the `STATUS_QUERY_TIMEOUT` environment
variable). If the query times out, safe defaults (mostly `false`/`null`) are
returned and `llama_server_running` is set to `false`.

This endpoint requires no authentication (internal) and is rate-unlimited.

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

#### Lease Release

Proactively release the dispatch lease for the caller's session, allowing other
sessions to acquire the local backend without waiting for the idle timeout
(default 180s) or disconnect detection.

```bash
curl -X POST http://localhost:8000/v1/leases/release \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess-abc123"}'
```

Response (200 — lease released or no matching lease):
```json
{"status": "ok"}
```

Response (400 — missing/empty `session_id`):
```json
{"detail": "session_id is required"}
```

The endpoint is **idempotent**: calling it with a `session_id` that has no
matching lease returns `200 OK` (no-op). This is useful for automated
workflows that want to clean up after completing a task without knowing
whether the lease is still active.

**Important:** This endpoint only releases the dispatch lease record from
`local_dispatch_records`. It does **not** release the scheduler slot (job
ownership), which is managed separately via disconnect detection or timeout.

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

## Client Disconnect Detection

The proxy automatically detects when a client disconnects during a streaming response and performs cleanup to prevent resource leaks and false overload signals.

### How It Works

1. The proxy periodically calls `request.is_disconnected()` (every 10 SSE chunks) during streaming in both local and remote proxy handlers.
2. When a disconnect is detected:
   - The streaming loop is broken immediately
   - The httpx connection to the backend is closed
   - The JobScheduler slot is released (if one was allocated) via `scheduler.remove_job()`
   - The `active_queries` counter is correctly decremented
   - Any queued jobs for the disconnected session are removed from the queue
3. Cleanup code runs in a `finally` block, ensuring resources are released even on generator closure.

### Configuration

Client disconnect cleanup timeout can be configured in `config.yaml`:

```yaml
server:
  disconnect_cleanup_timeout: 5.0  # seconds, default: 5.0
```

This timeout controls how long the proxy waits for cleanup operations (e.g., closing the httpx connection) to complete. If a cleanup operation exceeds this timeout, it is cancelled and the proxy continues with remaining cleanup steps.

### Behavior on Non-Streaming Requests

For non-streaming requests, the proxy reads the full backend response before returning. Client disconnect during non-streaming processing is detected by Starlette's built-in mechanisms, and resources are cleaned up in the request handler's error paths.

### Queue and Slot Cleanup

When a client disconnects:
- **While queued**: The job is removed from the JobScheduler queue and marked as cancelled. When a slot becomes available, cancelled jobs are skipped.
- **While a slot is allocated**: The slot is released immediately for use by another session. The `active_queries` counter is decremented.

### Observability

The proxy logs client disconnect events at INFO level with the session ID and slot ID:
```
client_disconnect session=<session_id> slot=<slot_id>
```

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
- **Context window limits**: llama-server's KV cache has finite capacity. The Qwen3 model is configured with a **128k (131,072) token context window**. Very long conversations may exceed this limit. The context window size is set in [`models.ini`](../models.ini) (router mode) and [`start-llama.sh`](../start-llama.sh) (single-model mode).

  > **Resource note**: A 128k context window increases RAM usage for llama-server. On GPU with 64 GB+ VRAM, running Qwen3.6-35B-A3B at 128k context is feasible but may require disabling mmap (`--no-mmap`) and using an appropriate quantization (Q5_K_M or lower). For hosts with less memory, consider reducing the context size or switching to a smaller model. The canonical size can be adjusted by changing `ctx-size` in `models.ini` and `CONTEXT` in `start-llama.sh`.
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
- `server.session_guardrail_max_token_rate` — token-rate guardrail threshold (tokens/second). Default `0` = disabled.
- `server.session_guardrail_token_rate_window_seconds` — rolling window (seconds) used to compute sustained tokens/sec.

When a guardrail triggers, `/admin/metrics` exposes `guardrail_metrics` with the
cutoff reason and invalidation counters for observability.

Token-rate guardrail calibration and notes:
- Disabled by default. Operators must explicitly enable it by setting `server.session_guardrail_max_token_rate`.
- Measurement is best-effort: token counting uses the existing `count_text_tokens` utility and SSE chunk boundaries, so rates are approximate.
- The guardrail requires a sustained violation over the full rolling window (`session_guardrail_token_rate_window_seconds`) to avoid cutting short legitimate short bursts.
- To calibrate a threshold:
  1. Enable the Prometheus token-rate metrics (see below) in a non-enforcing, observability-only run — start with a very high threshold or deploy in a canary environment.
  2. Observe `llama_token_rate_gauge{session_id="..."}` (instantaneous gauge) or compute a percentile from `llama_token_rate_histogram` over representative traffic, for example:
     ```promql
     histogram_quantile(0.95, sum(rate(llama_token_rate_histogram_bucket[5m])) by (le))
     ```
  3. Choose a threshold comfortably above your normal operating percentile (e.g., above 99th percentile of normal traffic), and test in canary before broad rollout.

Note: token-rate measurement is approximate and intended as a pragmatic guardrail; prefer conservative thresholds and monitor metrics closely after enabling.

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
- `proxy_http_errors_total{endpoint="...",status="...",reason="..."}` (counter) — HTTP errors by endpoint, status class, and reason. Incremented via `record_http_error(endpoint, status, reason)` at each 5xx response. Reasons include: `backend_error`, `backend_unavailable`, `self_healing`, and `slot_exhaustion`. See [5xx Error Runbook](docs/runbook-5xx.md) for investigation guidance.

Token-rate observability metrics (exposed on `/metrics`):

- `llama_token_rate_gauge{session_id="..."}` (gauge) — Best-effort instantaneous tokens/second observed for an active session. Useful for short-term spikes and live dashboards.
- `llama_token_rate_histogram{session_id="..."}` (histogram) — Bucketed distribution of observed token rates per session. Use Prometheus `histogram_quantile()` over a reasonable window (e.g., 5m) to compute operational percentiles for calibration.

Example queries:
- 95th percentile token-rate across all sessions:
```promql
histogram_quantile(0.95, sum(rate(llama_token_rate_histogram_bucket[5m])) by (le))
```
- Current token-rate gauge for a specific session:
```promql
llama_token_rate_gauge{session_id="<session-id>"}
```

These metrics are best-effort and will no-op when the `prometheus_client` library is not available. They are scoped per-session to help identify high-rate sessions that might indicate a runaway generator.

Example Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: 'llama-proxy'
    static_configs:
      - targets: ['localhost:8000']
```

A fully configured Prometheus instance with alert rules, data retention,
and systemd service setup is documented in [monitoring/README.md](../monitoring/README.md).

### Alerting Rules

- **Llama memory**: Warning at 75% of 90GB; critical at 90GB — `monitoring/llama_memory_alerts.yaml`.
- **Proxy 5xx errors**: Critical alert when `rate(proxy_http_errors_total{endpoint="/v1/chat/completions",status="5xx"}[5m]) > 5` for 5 minutes — `monitoring/proxy_5xx_alerts.yaml`.

A minimal Grafana dashboard JSON is included at `monitoring/grafana_llama_memory_dashboard.json` with panels for llama-server RSS, models loaded, and proxy 5xx error rate.

### Deploying Prometheus and Grafana

See [monitoring/README.md](../monitoring/README.md) for step-by-step deployment
instructions covering Prometheus and Grafana binary installation, systemd user
service setup, Grafana datasource and dashboard provisioning, and
verification steps.

### Runbook

See [5xx Error Runbook](docs/runbook-5xx.md) for on-call investigation, remediation, and escalation steps when the `ProxyHttpErrorsHigh` alert fires.

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

#### Adaptive Timeouts

The proxy supports adaptive timeouts that scale based on prompt size. This
prevents unnecessary timeouts for large prompts while keeping reasonable
limits for small requests.

Enable with:

- `server.llama_adaptive_timeout_enabled` (default `false`)
- `server.llama_adaptive_timeout_base_seconds` (default `60`)
- `server.llama_adaptive_timeout_per_token_seconds` (default `0.01`)
- `server.llama_request_timeout` (default `300`) — used as max timeout cap

The adaptive timeout formula is:
```
timeout = min(base + per_token * estimated_tokens, max_timeout)
```

For example, with default settings:
- Small prompt (100 tokens): `60 + 0.01 * 100 = 61s`
- Medium prompt (1000 tokens): `60 + 0.01 * 1000 = 70s`
- Large prompt (10000 tokens): `60 + 0.01 * 10000 = 160s`
- Very large prompt (25000 tokens): capped at `300s`

Token estimation uses a heuristic of ~4 bytes per token for UTF-8 text.

Concurrency pressure is controlled by `server.max_concurrent_queries` (default 4).
When the guard rejects a request, the proxy returns a 503 and increments
`backend_signals.concurrency_rejects`.

A watchdog loop monitors the backend process and, in router mode, also checks
for unhealthy worker children (for example zombie/defunct states). If a crash or
worker failure is detected, health switches to `degraded`, `backend_reachable`
becomes `false`, and router mode starts automatic self-healing.

Self-healing uses exponential backoff and is capped by:

- `server.llama_self_heal_max_attempts` (default `3`)
- `server.llama_self_heal_window_seconds` (default `300`)
- `server.llama_self_heal_backoff_base_seconds` (default `1`)
- `server.llama_self_heal_retry_after_seconds` (default `30`)

While self-healing is active, proxy requests return HTTP `503` with
`Retry-After: 30` and a machine-readable `backend_recovery_in_progress` payload.

If the backend is unavailable (either `backend_ready` is `false` or the backend
process has not been started), requests to `/v1/chat/completions` and other
completions endpoints immediately return HTTP `503` with a `backend_unavailable`
payload and `Retry-After` header, **without** attempting to connect to the
backend. This eliminates the window between a backend crash and watchdog
detection where clients would previously receive raw 500 errors.

When self-healing is not active, existing backend failure behavior is unchanged.

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
readiness transitions and backend signal counters. `/health` now includes
`backend_reachable`, `self_healing_in_progress`, and `backend_recovery`.

### Router Mode (Multi-Model)

When `llama_router_mode` is enabled, the proxy launches llama-server in router mode and
preloads the embeddings model plus the configured primary model. Requests are routed
to the appropriate model without stopping the server. The router exposes management
endpoints like `GET /models` and `POST /models/load`.

#### `llama_models_max` — Choosing a value

The `llama_models_max` config option (passed as `--models-max` to llama-server)
limits how many models can be loaded concurrently in router mode. When a new model
is requested beyond this limit, llama-server evicts the least-recently-used model
(triggering an `unload_lru` event). This is normal but excessive eviction/reload
churn degrades latency.

Recommended values for common setups:

| Setup | `llama_models_max` | Rationale |
|-------|-------------------|-----------|
| **Single model** (e.g., one chat model only) | `1` | Only one model needed; no eviction risk. |
| **Dual model** (embed + one chat model, preloaded both) | `2` | Embeddings + chat model both fit without eviction. |
| **Multi-model** (3+ models, e.g., embed + several chat models) | `3`–`4` | Allows embed + 2–3 chat models. Monitor eviction rates to tune. |
| **Heavy multi-model** (5+ models with varied sizes) | `5`–`8` | Requires sufficient GPU memory. Watch for OOM or eviction churn. |

The proxy monitors llama-server logs for `unload_lru` events and emits a warning
when 3 or more evictions occur within a 5-minute rolling window. The threshold
is configurable via environment variables:

- `LLAMA_UNLOAD_LRU_WINDOW_MINUTES` — rolling window duration (default: `5`)
- `LLAMA_UNLOAD_LRU_THRESHOLD` — events that trigger a warning (default: `3`)

If you see eviction warnings, increase `llama_models_max` or reduce the number
of concurrently used models.

### Slot Cache Retention

Slot snapshot files (`slot_*.bin`) accumulate in the directory configured by
`session_slot_save_path` (default: `slot-cache/`). A cleanup script is provided
to prevent disk-space exhaustion:

**Script:** `scripts/cleanup-slot-cache.sh`

**Default retention policy:**
- Remove slot files older than **7 days** (`--max-age-days`, default: `7`)
- Retain the **3 most recently modified** files per unique slot prefix
  (`--keep-recent`, default: `3`)

**Usage:**

```bash
# Preview deletions without removing anything
./scripts/cleanup-slot-cache.sh --dry-run

# Run cleanup with defaults (7 days, keep 3 per prefix)
./scripts/cleanup-slot-cache.sh

# Custom retention
./scripts/cleanup-slot-cache.sh --max-age-days 14 --keep-recent 5

# Custom slot-cache directory
./scripts/cleanup-slot-cache.sh --path /custom/slot-cache-path
```

The script is idempotent and exits 0 on success. Errors (e.g., un-deletable files)
are logged as warnings without causing a non-zero exit.

**Automated cleanup (cron):**

To run the cleanup automatically every evening at 10 PM, add the following
crontab entry (run `crontab -e`):

```cron
0 22 * * * /home/rgardler/projects/llm/scripts/cleanup-slot-cache.sh >> /home/rgardler/projects/llm/logs/cleanup-slot-cache.log 2>&1
```

Adjust the path to match your repository location.

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

## TTS (Text-to-Speech) /v1/audio/speech

The proxy supports text-to-speech synthesis via the OpenAI-compatible
`/v1/audio/speech` endpoint.  Requests are forwarded to a local
[qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) TTS server that
uses a quantised Qwen3-TTS model for inference on AMD ROCm / HIP hardware.

### Endpoint

```
POST /v1/audio/speech
Content-Type: application/json

{
  "model": "qwen3-tts",
  "input": "Text to convert to speech.",
  "voice": "default",
  "response_format": "wav"
}
```

**Request fields**

| Field | Required | Description |
|-------|----------|-------------|
| `model` | Yes | Model identifier (passed through to tts-server) |
| `input` | Yes | Text to synthesise (must be non-empty) |
| `voice` | No | Speaker voice (base model has no pre-registered speakers; omit or leave empty) |
| `response_format` | No | Output format (currently only `wav` is supported) |

**Response**  
- **Success (200):** Audio file with `Content-Type: audio/wav` (24 kHz, mono, 16-bit PCM WAV).
- **Error (400):** Missing or empty `input`, missing `model`, or empty body.
- **Error (502):** TTS server is unreachable or returned an error.

### Curl Examples

```bash
# Basic usage
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts",
    "input": "Hello, this is a test of the text-to-speech system."
  }' \
  --output speech.wav

# With explicit voice parameter (may fail on base model)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts",
    "input": "The quick brown fox jumps over the lazy dog.",
    "voice": "default",
    "response_format": "wav"
  }' \
  --output fox.wav
```

### Configuration

The TTS server is configured under the `server:` section of `config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `tts_server_host` | `localhost` | Host the tts-server listens on |
| `tts_server_port` | `8081` | Port the tts-server listens on |
| `tts_start_script` | `proxy/scripts/start-qwentts.sh` | Path to the startup script |
| `tts_model_path` | `""` (auto-detect) | Talker LM GGUF path |
| `tts_codec_path` | `""` (auto-detect) | Codec / tokenizer GGUF path |
| `tts_enabled` | `true` | Set to `false` to skip TTS server startup |

### Lifecycle

The TTS server lifecycle is managed by the proxy:

- **Startup:** Started asynchronously alongside llama-server during proxy
  `lifespan()` startup.  The proxy waits up to 30 seconds for the TTS server
  to become reachable on its configured port.
- **Shutdown:** Stopped gracefully (SIGTERM, 30 s timeout, then SIGKILL) during
  proxy shutdown, before llama-server is stopped.
- **Health checks:** Probed automatically on the configured port before
  marking the server as ready.

### Building the TTS Server

The TTS server is built from the `qwentts.cpp` submodule:

```bash
# Prerequisites — ROCm / HIP SDK installed, gfx1151-capable GPU
cd qwentts.cpp
cmake -B build \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151
cmake --build build -j$(nproc)

# Binaries are placed in build/bin/
ls build/bin/tts-server  # the server binary
```

*See `qwentts.cpp/BUILD_NOTES.md` for the exact build flags used on this machine.*

### Models

GGUF-quantised models are downloaded from
[Serveurperso/Qwen3-TTS-GGUF](https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF)
and placed in `qwentts.cpp/models/`:

- **Talker LM:** `qwen-talker-1.7b-base-Q8_0.gguf` (~2.0 GB, 1.7B params, Q8_0 quant)
- **Codec / Tokenizer:** `qwen-tokenizer-12hz-Q8_0.gguf` (~278 MB)

The startup script auto-detects model files in `qwentts.cpp/models/` unless
`tts_model_path` / `tts_codec_path` are explicitly set in `config.yaml`.

### Integration Tests

End-to-end integration tests that exercise a live tts-server are located in:

```
pytest proxy/tests/test_tts_integration.py -v
```

These tests are **automatically skipped** when the tts-server is not running
(default port `localhost:8081`).  To run them:

1. Start the proxy (which starts the tts-server): `proxyctl start`
2. In another terminal: `pytest proxy/tests/test_tts_integration.py -v`

Alternatively, start the tts-server directly:

```bash
bash proxy/scripts/start-qwentts.sh --port 8081
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

### TTS server won't start
1. Check the TTS server port: `lsof -i :8081`
2. Verify the tts-server binary exists: `ls -la qwentts.cpp/tts-server`
3. Start the TTS server manually and check output:
   ```bash
   bash proxy/scripts/start-qwentts.sh --port 8081
   ```
4. Verify GPU is accessible: `rocm-smi` (should list gfx1151)
5. Review proxy logs for TTS-specific errors: `tail -f ./logs/proxy.log | grep -i tts`

### TTS server starts but health check fails
1. Verify correct port: `curl -X POST http://localhost:8081/v1/audio/speech -H "Content-Type: application/json" -d '{"model":"test","input":"ping"}'`
2. Check for port conflicts with other services (llama-server uses :8080)
3. Increase `startup_timeout` if the model takes longer than 30 s to load on GPU

### Audio response is garbled or silent
1. Verify WAV header: `head -c 44 speech.wav | xxd | head -5` (should start with `RIFF`)
2. Ensure the output file is opened in binary mode: `--output speech.wav`
3. Base model has no pre-registered speakers — omit `voice` parameter (use empty string if needed)
4. Check sample rate: `ffprobe speech.wav` (expected: 24000 Hz, mono, 16-bit PCM)

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

### Internal Module Structure

The proxy server (`proxy/proxy/`) has been refactored from a monolithic `server.py`
into a modular composition:

| Module | Responsibility |
|--------|---------------|
| `server.py` | Bootstrap, composition wiring, module-level globals, lifespan, `main()` |
| `router.py` | Core proxy routing (`proxy_to_local`, `proxy_to_remote`) and request/response logging (`log_request`, `log_response`, `log_response_chunk`) |
| `handlers.py` | HTTP route handlers (FastAPI APIRouter) — `/health`, `/v1/models`, `/metrics`, `/admin/*` |
| `lifecycle.py` | Model lifecycle orchestration, model loading, refcounting, background loads, router-model loading |
| `backend_health.py` | Backend recovery, self-healing, watchdog monitoring, worker-health checks, router model health monitoring |
| `session.py` | Session coordination, delta/fallback/single-flight, restore signal detection, `ContentOnlyConsoleHandler` |
| `observability.py` | Backend signal counters, SSE client sets (`sse_clients`, `log_tail_clients`), persistence loops |
| `metrics.py` | Prometheus metrics helpers (gauges, counters, exposition format) |
| `ui.py` | Web UI endpoints (dashboard at `/`, log viewer at `/logs`, SSE streaming) |
| `templates/` | HTML template files for Web UI (`index.html`, `view_logs.html`) |
| `session_manager.py` | Core session manager class (session create/lookup/expiry) |

Each module uses a lazy `_srv()` import pattern to access `server.py` module-level
state without circular imports. Backward-compatibility re-exports in `server.py`
preserve existing test imports.

## License

See repository root for license information.
