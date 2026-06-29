#!/usr/bin/env bash
set -euo pipefail

# Start the proxy application
# Usage: ./scripts/start-proxy.sh [uvicorn-args...]

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

# Exec uvicorn using chosen python binary
exec "$PY_BIN" -m uvicorn proxy.server:app --host 0.0.0.0 --port "$PORT" "$@"
