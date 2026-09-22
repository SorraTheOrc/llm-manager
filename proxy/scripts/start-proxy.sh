#!/usr/bin/env bash
set -euo pipefail

# Start the proxy application
# Usage: ./scripts/start-proxy.sh [--restart] [--verbose] [uvicorn-args...]
#
# Flags:
#   --restart   Kill all running proxy/llama-server/TTS processes before starting
#   --keep-llama-server
#               Proxy-only restart: preserve the co-located llama-server so its
#               warm KV / prompt cache survives (LP-0MUCEFCL6001ZXYN). With
#               --restart and no explicit flag, llama-server is preserved
#               automatically when its recorded signature matches the selected
#               config (same mode + config hash); it is restarted when the
#               model or parallel config actually changes.
#   --verbose   Enable verbose per-chunk SSE logging (STREAM CHUNK lines at INFO level)
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
VERBOSE=0
KEEP_LLAMA_SERVER=0
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
      --keep-llama-server)
        # LP-0MUCEFCL6001ZXYN: proxy-only restart — preserve the co-located
        # llama-server so its warm KV / prompt cache survives.
        KEEP_LLAMA_SERVER=1
        ;;
      --verbose)
        # Enable verbose per-chunk SSE logging (STREAM CHUNK lines at INFO
        # level). Consumed here rather than passed to uvicorn, which rejects
        # unknown CLI flags (LP-0MS9GAN2P002NR4M).
        VERBOSE=1
        ;;
      *)
        UVICORN_ARGS+=("$arg")
        ;;
    esac
  fi
done

# Translate --verbose into the env var read by proxy/utils.setup_logging
if [ "$VERBOSE" -eq 1 ]; then
  export LLAMA_PROXY_VERBOSE=1
  echo "Verbose mode enabled: per-chunk SSE logging at INFO level" >&2
fi

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

  # LP-0MUCEFCL6001ZXYN: decide whether to preserve the co-located
  # llama-server. A proxy-only restart (unchanged local backend config) must
  # keep it running so the warm KV / prompt cache survives; a genuine
  # model/parallel change (e.g. a mode switch) must restart it.
  KEEP_LLAMA=0
  if [ "$KEEP_LLAMA_SERVER" -eq 1 ]; then
    KEEP_LLAMA=1
  else
    _sig_mode=""
    _sig_sha=""
    _restart_mode="fast"
    if [ -f "$REPO_ROOT/.mode" ]; then
      _restart_mode="$(tr -d '[:space:]' < "$REPO_ROOT/.mode")"
    fi
    _restart_cfg="$REPO_ROOT/config-$_restart_mode.yaml"
    [ -f "$_restart_cfg" ] || _restart_cfg="$REPO_ROOT/config.yaml"
    _restart_sha=""
    if command -v sha256sum >/dev/null 2>&1 && [ -f "$_restart_cfg" ]; then
      _restart_sha="$(sha256sum "$_restart_cfg" | cut -d' ' -f1)"
    fi
    _sig_file="$REPO_ROOT/.llama_server_signature.json"
    if [ -f "$_sig_file" ] && [ -n "$_restart_sha" ]; then
      _sig_mode="$("$PY_BIN" -c "import json,sys; print(json.load(open(sys.argv[1])).get('mode',''))" "$_sig_file" 2>/dev/null || true)"
      _sig_sha="$("$PY_BIN" -c "import json,sys; print(json.load(open(sys.argv[1])).get('config_sha256',''))" "$_sig_file" 2>/dev/null || true)"
    fi
    if [ -n "$_sig_mode" ] && [ "$_sig_mode" = "$_restart_mode" ] && [ "$_sig_sha" = "$_restart_sha" ]; then
      KEEP_LLAMA=1
    fi
  fi

  if [ "$KEEP_LLAMA" -eq 1 ]; then
    echo "Preserving running llama-server (unchanged local backend config)."
  fi

  # Phase 1: graceful SIGTERM
  pkill -f 'uvicorn proxy\.server' 2>/dev/null || true
  if [ "$KEEP_LLAMA" -eq 0 ]; then
    pkill -f 'llama-server' 2>/dev/null || true
  fi
  pkill -f 'qwentts' 2>/dev/null || true
  pkill -f 'tts-server' 2>/dev/null || true
  sleep 3

  # Phase 2: force-kill any survivors (graceful shutdown may hang — e.g.,
  # asyncio tasks that don't cancel cleanly leaving a zombie process).
  pkill -9 -f 'uvicorn proxy\.server' 2>/dev/null || true
  if [ "$KEEP_LLAMA" -eq 0 ]; then
    pkill -9 -f 'llama-server' 2>/dev/null || true
  fi
  pkill -9 -f 'qwentts' 2>/dev/null || true
  pkill -9 -f 'tts-server' 2>/dev/null || true
  sleep 2

  # Phase 3: fuser fallback — kill any leftover processes holding our ports
  if command -v fuser >/dev/null 2>&1; then
    if [ "$KEEP_LLAMA" -eq 0 ]; then
      fuser -k "$LLAMA_PORT/tcp" 2>/dev/null || true
    fi
    fuser -k "$TTS_PORT/tcp" 2>/dev/null || true
  fi

  # Phase 4: wait until all ports are confirmed free (blocking, up to 10s each)
  failed=0
  if [ "$KEEP_LLAMA" -eq 0 ] && ! _wait_for_port_release "$LLAMA_PORT"; then
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

# ---- Resolve operating mode and config file -----------------------------
# The persisted mode (proxy/.mode, written by POST /admin/set-mode) selects
# the config profile: fast -> config-fast.yaml, cheap -> config-cheap.yaml.
# Absent/invalid state defaults to fast (current behavior). LLAMA_PROXY_CONFIG
# is exported so the server's load_config() uses the same profile, and API
# keys are resolved from the SELECTED config (cheap mode has no remote
# providers, so missing cloud keys must not fail startup).

MODE_FILE="$REPO_ROOT/.mode"
MODE="fast"
if [ -f "$MODE_FILE" ]; then
  MODE_CONTENT="$(tr -d '[:space:]' < "$MODE_FILE")"
  case "$MODE_CONTENT" in
    fast|cheap) MODE="$MODE_CONTENT" ;;
    *) echo "Warning: unknown mode '$MODE_CONTENT' in $MODE_FILE, defaulting to fast" >&2 ;;
  esac
fi

CONFIG_FILE="$REPO_ROOT/config-$MODE.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Warning: $CONFIG_FILE not found, falling back to config.yaml" >&2
  CONFIG_FILE="$REPO_ROOT/config.yaml"
fi
export LLAMA_PROXY_CONFIG="$CONFIG_FILE"
echo "Operating mode: $MODE (config: $CONFIG_FILE)" >&2

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
