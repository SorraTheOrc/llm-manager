# LLama Proxy Server

A proxy server that routes OpenAI-compatible API requests to either a local llama-server or remote API services based on configuration.

## Features

- **Unified API Endpoint**: Single endpoint for all LLM requests, regardless of backend
- **Web UI**: Built-in dashboard for monitoring, model switching, and API testing
- **Local Model Management**: Automatically starts/stops llama-server with the correct model
- **Remote API Routing**: Forward requests to OpenAI, Anthropic, or other OpenAI-compatible APIs
- **Hot Model Switching**: Automatically switches local models when a different model is requested
- **Streaming Support**: Full support for streaming responses (SSE)
- **Request/Response Logging**: Comprehensive logging with time-based rotation
- **Request + Token Counters**: In-memory counters with periodic JSON persistence
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

## Configuration

Edit `config.yaml` to configure the server:

```yaml
# Server settings
server:
  host: "0.0.0.0"
  port: 8000
  llama_start_script: "/home/rgardler/projects/llm/start-llama.sh"
  distrobox_name: "llama"  # Distrobox container where llama-server runs
  llama_server_port: 8080
  llama_startup_timeout: 300

# Default model to load on startup
default_model: "qwen3"

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
  "current_model": "qwen3",
  "llama_server_running": true
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
