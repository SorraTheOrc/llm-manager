#!/usr/bin/env bash
set -euo pipefail

# Start the proxy application
# Usage: ./scripts/start-proxy.sh [uvicorn-args...]

VENV_ACTIVATE=".venv/bin/activate"

if [ -f "$VENV_ACTIVATE" ]; then
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
else
  echo "Warning: virtualenv activate script not found at '$VENV_ACTIVATE'."
  echo "Continuing without activating virtualenv." >&2
fi

# Default host and port can be overridden by passing extra uvicorn args
exec python -m uvicorn proxy.server:app --host 0.0.0.0 --port 8000 "$@"
