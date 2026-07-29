#!/usr/bin/env bash
set -euo pipefail

# Start the proxy application
# Usage: ./scripts/start-proxy.sh [--restart] [uvicorn-args...]
#
# Flags:
#   --restart   Kill all running proxy/llama-server/TTS processes before starting
#
# Automatically resolves required API keys from:
#   1. Environment variables (already set)
#   2. ~/.pi/agent/auth.json as fallback

VENV_DIR=".venv"
VENV_PY="$VENV_DIR/bin/python3"
VENV_ACTIVATE="$VENV_DIR/bin/activate"
PY_BIN=""

# Prefer venv python if present, fall back to system python3, then python
if [ -x "$VENV_PY" ]; then
  PY_BIN="$VENV_PY"
elif [ -x "$VENV_DIR/bin/python" ]; then
  PY_BIN="$VENV_DIR/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="$(command -v python)"
else
  echo "Error: no Python interpreter found. Please install Python 3 or create a .venv." >&2
  exit 1
fi

# Source venv activate if present (this keeps behavior consistent for users)
if [ -f "$VENV_ACTIVATE" ]; then
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
fi

# Determine repo root and set PYTHONPATH if not set
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -z "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$REPO_ROOT"
  echo "Notice: PYTHONPATH not set, defaulting to repo root: $REPO_ROOT" >&2
fi

# Determine port: default 8000 unless overridden by --port or PROXY_PORT/PORT env var
PORT="${PROXY_PORT:-${PORT:-8000}}"
RESTART=0
UVICORN_ARGS=()
prev=""
for arg in "$@"; do
  if [ "$prev" = "--port" ] || [ "$prev" = "-p" ]; then
    PORT="$arg"
    UVICORN_ARGS+=("--port" "$arg")
    prev=""
  else
    case "$arg" in
      --port=*)
        PORT="${arg#*=}"
        UVICORN_ARGS+=("$arg")
        ;;
      --port)
        prev="--port"
        ;;
      -p)
        prev="-p"
        ;;
      --restart)
        RESTART=1
        ;;
      *)
        UVICORN_ARGS+=("$arg")
        ;;
    esac
  fi
done

# Ports used by backend services (llama-server on 8080, TTS on 8081)
LLAMA_PORT=8080
TTS_PORT=8081

# ---------------------------------------------------------------------------
# Port helpers
# ---------------------------------------------------------------------------

# _port_in_use <port>  ->  0 if in use, 1 if free
_port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn | awk '{print $4}' | grep -Eq ":$port$|\.$port$"
  elif command -v netstat >/dev/null 2>&1; then
    netstat -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":$port$|\.$port$"
  else
    "$PY_BIN" -c "
import socket, sys
s = socket.socket()
s.settimeout(0.5)
try:
    s.connect(('127.0.0.1', $port))
except Exception:
    sys.exit(1)
else:
    sys.exit(0)
" 2>/dev/null
  fi
}

# _wait_for_port_release <port> [timeout]  ->  0 on success, 1 on timeout
# Polls until the port is free (ECONNREFUSED), with a default 10s timeout.
_wait_for_port_release() {
  local port="$1"
  local timeout="${2:-10}"
  local deadline
  deadline="$(python3 -c "import time; print(time.monotonic() + $timeout)")"
  while true; do
    local now
    now="$(python3 -c "import time; print(time.monotonic())")"
    if python3 -c "import sys; sys.exit(0 if $now > $deadline else 1)" 2>/dev/null; then
      return 1  # timeout
    fi
    if ! _port_in_use "$port"; then
      return 0  # port is free
    fi
    sleep 0.5
  done
}

# --restart: kill all running proxy-related processes before starting
if [ "$RESTART" -eq 1 ]; then
  echo "Restart requested: stopping running proxy services..."

  # Phase 1: graceful SIGTERM
  pkill -f 'uvicorn proxy\.server' 2>/dev/null || true
  pkill -f 'llama-server' 2>/dev/null || true
  pkill -f 'qwentts' 2>/dev/null || true
  pkill -f 'tts-server' 2>/dev/null || true
  sleep 3

  # Phase 2: force-kill any survivors (graceful shutdown may hang — e.g.,
  # asyncio tasks that don't cancel cleanly leaving a zombie process).
  pkill -9 -f 'uvicorn proxy\.server' 2>/dev/null || true
  pkill -9 -f 'llama-server' 2>/dev/null || true
  pkill -9 -f 'qwentts' 2>/dev/null || true
  pkill -9 -f 'tts-server' 2>/dev/null || true
  sleep 2

  # Phase 3: fuser fallback — kill any leftover processes holding our ports
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "$LLAMA_PORT/tcp" 2>/dev/null || true
    fuser -k "$TTS_PORT/tcp" 2>/dev/null || true
  fi

  # Phase 4: wait until all ports are confirmed free (blocking, up to 10s each)
  local failed=0
  if ! _wait_for_port_release "$LLAMA_PORT"; then
    echo "Warning: llama-server port $LLAMA_PORT did NOT become free within 10s after kill" >&2
    failed=1
  fi
  if ! _wait_for_port_release "$TTS_PORT"; then
    echo "Warning: TTS port $TTS_PORT did NOT become free within 10s after kill" >&2
    failed=1
  fi
  if ! _wait_for_port_release "${PORT}"; then
    echo "Warning: proxy port $PORT did NOT become free within 10s after kill" >&2
    failed=1
  fi

  if [ "$failed" -eq 0 ]; then
    echo "All ports freed successfully."
  fi
  echo "Done. Starting fresh..."
fi

# Check if the proxy port is already in use
PORT_IN_USE=0
if _port_in_use "${PORT}"; then
  PORT_IN_USE=1
fi

if [ "$PORT_IN_USE" -eq 1 ]; then
  echo "Error: port $PORT is already in use. Is another proxy or service running?" >&2
  echo "If you intended to run in development mode, use --port <port> to specify a different port." >&2
  exit 1
fi

# ---- Resolve API keys from config.yaml ---------------------------------

CONFIG_FILE="$REPO_ROOT/config.yaml"
AUTH_FILE="$HOME/.pi/agent/auth.json"

resolve_api_keys() {
  local missing=()

  # Extract all unique api_key_env values from config.yaml
  while IFS='' read -r env_var; do
    [[ -z "$env_var" ]] && continue

    # Already set in environment — nothing to do
    if [[ -n "${!env_var:-}" ]]; then
      echo "[env] $env_var already set from environment"
      continue
    fi

    # Try to resolve from pi's auth.json
    if [[ -f "$AUTH_FILE" ]]; then
      resolved="$(resolve_from_auth_json "$env_var")"
      if [[ -n "$resolved" ]]; then
        export "$env_var=$resolved"
        echo "[env] $env_var resolved from ~/.pi/agent/auth.json"
        continue
      fi
    fi

    # Not found anywhere — annotate with which model(s) need it
    local models_using
    models_using="$($PY_BIN -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
models = []
for name, model in cfg.get('models', {}).items():
    for p in model.get('providers', []):
        if p.get('api_key_env') == '$env_var':
            models.append(name)
print(', '.join(models))
" 2>/dev/null || echo 'unknown')"
    missing+=("$env_var  (required by: $models_using)")
  done < <($PY_BIN -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
keys = set()
for name, model in cfg.get('models', {}).items():
    for p in model.get('providers', []):
        env_key = p.get('api_key_env')
        if env_key:
            keys.add(env_key)
for k in sorted(keys):
    print(k)
")

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo ""
    echo "ERROR: The following API key environment variables are not set"
    echo "       and could not be resolved from ~/.pi/agent/auth.json:"
    for key in "${missing[@]}"; do
      echo "  - $key"
    done
    echo ""
    echo "Set each as an environment variable before starting the proxy, for example:"
    echo "  export GITHUB_TOKEN=ghp_..."
    echo "  export OPENCODE_API_KEY=sk-..."
    echo ""
    echo "Or add the key to \$AUTH_FILE under the matching provider name"
    return 1
  fi
}

# Map api_key_env name to auth.json key.
# Prefers opencode-go over opencode when resolving OPENCODE_API_KEY.
resolve_from_auth_json() {
  local env_var="$1"

  $PY_BIN -c "
import json, sys

key_name = '$env_var'

try:
    with open('$AUTH_FILE') as f:
        auth = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    sys.exit(1)

# Lowercase key for lookup
key = key_name.lower()

# Prefer opencode-go over opencode for OPENCODE_API_KEY
if key == 'opencode_api_key':
    for preferred in ('opencode-go', 'opencode'):
        if preferred in auth and auth[preferred].get('type') == 'api_key':
            print(auth[preferred]['key'])
            sys.exit(0)

# Exact lowercase match
if key in auth and auth[key].get('type') == 'api_key':
    print(auth[key]['key'])
    sys.exit(0)

# Strip _API_KEY suffix
if key.endswith('_api_key'):
    stem = key[:-8]
    if stem in auth and auth[stem].get('type') == 'api_key':
        print(auth[stem]['key'])
        sys.exit(0)

sys.exit(1)
"
}

echo "=== LLM Proxy API Key Check ==="
resolve_api_keys

echo ""
echo "=== Starting proxy server ==="

# Exec uvicorn using chosen python binary
exec "$PY_BIN" -m uvicorn proxy.server:app --host 0.0.0.0 --port "$PORT" "${UVICORN_ARGS[@]+${UVICORN_ARGS[@]}}"
