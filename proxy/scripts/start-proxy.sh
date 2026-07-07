#!/usr/bin/env bash
set -euo pipefail

# Start the proxy application
# Usage: ./scripts/start-proxy.sh [uvicorn-args...]
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
prev=""
for arg in "$@"; do
  if [ "$prev" = "--port" ] || [ "$prev" = "-p" ]; then
    PORT="$arg"
    prev=""
  else
    case "$arg" in
      --port=*)
        PORT="${arg#*=}"
        ;;
      --port)
        prev="--port"
        ;;
      -p)
        prev="-p"
        ;;
      *)
        ;;
    esac
  fi
done

# Check if the port is already in use. Prefer ss/netstat for a fast local check,
# fall back to a Python connect test when those tools are unavailable.
PORT_IN_USE=0
if command -v ss >/dev/null 2>&1; then
  if ss -ltn | awk '{print $4}' | grep -Eq ":${PORT}$|\.${PORT}$"; then
    PORT_IN_USE=1
  fi
elif command -v netstat >/dev/null 2>&1; then
  if netstat -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${PORT}$|\.${PORT}$"; then
    PORT_IN_USE=1
  fi
else
  # Fallback: try to connect using Python
  if "$PY_BIN" - <<PYTEST 2>/dev/null
import socket,sys
port = int(${PORT})
s=socket.socket()
s.settimeout(0.5)
try:
    s.connect(('127.0.0.1', port))
except Exception:
    sys.exit(0)
else:
    sys.exit(1)
PYTEST
  then
    PORT_IN_USE=1
  fi
fi

if [ "$PORT_IN_USE" -eq 1 ]; then
  echo "Error: port $PORT is already in use. Is another proxy or service running?" >&2
  echo "If you intended to run in development mode use: proxyctl start --dev (uses port 8001), or run this script with --port <port>." >&2
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
exec "$PY_BIN" -m uvicorn proxy.server:app --host 0.0.0.0 --port "$PORT" "$@"
