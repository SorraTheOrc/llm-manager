#!/usr/bin/env bash
set -euo pipefail

# Start the proxy application
# Usage: ./scripts/start-proxy.sh [--restart] [--reason "<text>"] [--verbose] [--use-env-keys] [uvicorn-args...]
#
# Flags:
#   --restart   Kill all running proxy/llama-server/TTS processes before starting
#   --reason    Restart reason echoed into the new process's startup log
#               (LP-0MVAPNC2900145RH; flag > LLAMA_PROXY_RESTART_REASON >
#               `manual (no reason provided)`).
#   --keep-llama-server
#               Proxy-only restart: preserve the co-located llama-server so its
#               warm KV / prompt cache survives (LP-0MUCEFCL6001ZXYN). With
#               --restart and no explicit flag, llama-server is preserved
#               automatically when its recorded signature matches the effective
#               config; it is restarted when the local-backend config actually
#               changes. The signature hashes the merged config (the base alone
#               since the mode overlays were removed, LP-0MVDT94F0003VHUA AC4)
#               via the shared merged_config_sha256 helper
#               (LP-0MUJRVYQH002IEHW); pre-upgrade raw-file-hash signatures
#               mismatch once and conservatively restart llama-server.
#   --verbose   Enable verbose per-chunk SSE logging (STREAM CHUNK lines at INFO level)
#   --use-env-keys
#               Keep the environment's provider API keys even when they
#               DISAGREE with ~/.pi/agent/auth.json. Stale-key detection is
#               fail-closed by default: a mismatch refuses to start, so a
#               rotated auth.json can never be silently ignored. Unattended
#               restart paths (herdr/agent-spawned) cannot pass flags, so they
#               may instead set LLAMA_PROXY_USE_ENV_KEYS=1. This flag is
#               consumed here and never forwarded to uvicorn
#               (LP-0MVF7DRMK001NHT8).
#
# Automatically resolves required API keys from:
#   1. Environment variables (already set) — verified against auth.json; a
#      stale value refuses startup unless --use-env-keys /
#      LLAMA_PROXY_USE_ENV_KEYS=1 is given
#   2. ~/.pi/agent/auth.json as fallback
#
# Every resolved provider key is logged as an 8-character fingerprint only;
# a full key is never printed (LP-0MVF7DRMK001NHT8).

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
RESTART_REASON=""
# LP-0MVF7DRMK001NHT8: stale-key detection is fail-closed by default. The
# opt-out is either the --use-env-keys flag or LLAMA_PROXY_USE_ENV_KEYS=1
# (unattended restarts cannot pass flags).
USE_ENV_KEYS=0
if [ "${LLAMA_PROXY_USE_ENV_KEYS:-}" = "1" ]; then
  USE_ENV_KEYS=1
fi
UVICORN_ARGS=()
prev=""
for arg in "$@"; do
  if [ "$prev" = "--reason" ]; then
    # --reason <text>: consumed here, never forwarded to uvicorn
    RESTART_REASON="$arg"
    prev=""
  elif [ "$prev" = "--port" ] || [ "$prev" = "-p" ]; then
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
      --reason)
        prev="--reason"
        ;;
      --reason=*)
        RESTART_REASON="${arg#*=}"
        ;;
      --keep-llama-server)
        # LP-0MUCEFCL6001ZXYN: proxy-only restart — preserve the co-located
        # llama-server so its warm KV / prompt cache survives.
        KEEP_LLAMA_SERVER=1
        ;;
      --use-env-keys)
        # LP-0MVF7DRMK001NHT8 AC3: explicit opt-out of fail-closed stale-key
        # detection. Consumed here — uvicorn rejects unknown CLI flags.
        USE_ENV_KEYS=1
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
  # LP-0MVAPNC2900145RH: resolve the restart reason (--reason flag beats
  # LLAMA_PROXY_RESTART_REASON env, which beats the default) and export it so
  # the new server process can echo it into its startup log right before the
  # `Starting LLama Proxy Server` marker.
  RESTART_REASON="${RESTART_REASON:-${LLAMA_PROXY_RESTART_REASON:-manual (no reason provided)}}"
  export LLAMA_PROXY_RESTART_REASON="$RESTART_REASON"
  echo "Restart requested: stopping running proxy services..."
  echo "Restart reason: $RESTART_REASON"

  # LP-0MUCEFCL6001ZXYN: decide whether to preserve the co-located
  # llama-server. A proxy-only restart (unchanged local backend config) must
  # keep it running so the warm KV / prompt cache survives; a genuine
  # model/parallel change must restart it.
  KEEP_LLAMA=0
  if [ "$KEEP_LLAMA_SERVER" -eq 1 ]; then
    KEEP_LLAMA=1
  else
    _sig_sha=""
    _restart_cfg="$REPO_ROOT/config.yaml"
    # LP-0MUJRVYQH002IEHW: hash the MERGED config (base + overlay) via the
    # shared canonical helper — NOT the raw file — so a base-only
    # local-backend change (e.g. local_model_ctx_size, llama_server_port)
    # also changes the signature and forces a llama-server restart. With the
    # mode axis removed (LP-0MVDT94F0003VHUA AC4) the effective config is
    # the base alone. An empty hash (Python unavailable / helper failure)
    # means "do not preserve".
    _restart_sha="$("$PY_BIN" -c "
import sys
sys.path.insert(0, '$REPO_ROOT/proxy')
from config_merge import merged_config_sha256
print(merged_config_sha256('$REPO_ROOT/config.yaml', '$_restart_cfg'))
" 2>/dev/null || true)"
    _sig_file="$REPO_ROOT/.llama_server_signature.json"
    if [ -f "$_sig_file" ] && [ -n "$_restart_sha" ]; then
      _sig_sha="$("$PY_BIN" -c "import json,sys; print(json.load(open(sys.argv[1])).get('config_sha256',''))" "$_sig_file" 2>/dev/null || true)"
    fi
    if [ -n "$_sig_sha" ] && [ "$_sig_sha" = "$_restart_sha" ]; then
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

CONFIG_FILE="$REPO_ROOT/config.yaml"
echo "Using config: $CONFIG_FILE" >&2

AUTH_FILE="$HOME/.pi/agent/auth.json"

# ---------------------------------------------------------------------------
# API-key resolution + stale-key detection (LP-0MVF7DRMK001NHT8)
# ---------------------------------------------------------------------------

# enumerate_api_key_envs
#
# Print every unique api_key_env declared by the effective config, sorted.
enumerate_api_key_envs() {
  "$PY_BIN" -c "
import sys
sys.path.insert(0, '$REPO_ROOT/proxy')
from config_merge import load_merged
cfg = load_merged('$REPO_ROOT/config.yaml', '$CONFIG_FILE')
keys = set()
for name, model in cfg.get('models', {}).items():
    for p in model.get('providers', []):
        env_key = p.get('api_key_env')
        if env_key:
            keys.add(env_key)
for k in sorted(keys):
    print(k)
"
}

# models_requiring <env_var>
#
# Print the comma-separated model names that declare <env_var>.
models_requiring() {
  "$PY_BIN" -c "
import sys
sys.path.insert(0, '$REPO_ROOT/proxy')
from config_merge import load_merged
cfg = load_merged('$REPO_ROOT/config.yaml', '$CONFIG_FILE')
models = []
for name, model in cfg.get('models', {}).items():
    for p in model.get('providers', []):
        if p.get('api_key_env') == '$1':
            models.append(name)
print(', '.join(models))
" 2>/dev/null || echo 'unknown'
}

# check_api_key <env_var>
#
# Run the resolver's ``--check`` mode and print ONE report line whose five
# fields are separated by the ASCII unit separator (0x1f). Tab is deliberately
# avoided: it is IFS whitespace, so ``read`` would collapse the empty fields an
# env-unset (``resolved``) or auth.json-less (``auth_missing``) check produces.
#
#   status <US> env fingerprint <US> auth.json fingerprint <US>
#   resolved key <US> auth.json key name(s) tried
#
# The resolver exit code is passed through (0 match/resolved/env_only,
# 3 stale mismatch, 4 unresolved/auth_missing). Callers MUST invoke this
# inside an ``if`` guard so ``set -e`` never aborts before the report is
# printed, and must never echo ``resolved_key``: only the 8-char fingerprints
# are ever logged (AC2).
check_api_key() {
  local env_var="$1"
  "$PY_BIN" -c "
import contextlib
import io
import json
import sys

sys.path.insert(0, '$REPO_ROOT/proxy')
from api_key_resolver import main as resolver_main

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = resolver_main(['--check', '$AUTH_FILE', '$env_var'])
payload = json.loads(buf.getvalue().strip().splitlines()[-1])
print('\x1f'.join([
    payload.get('status', ''),
    payload.get('env_fingerprint', ''),
    payload.get('auth_fingerprint', ''),
    payload.get('resolved_key', ''),
    ', '.join(payload.get('candidate_names') or []),
]))
sys.exit(rc)
"
}

resolve_api_keys() {
  local missing=()
  local mismatched_vars=()
  local mismatch_lines=()
  local env_var="" status="" env_fp="" auth_fp="" resolved_key="" candidates=""
  local report="" check_rc=0 models_using=""

  while IFS='' read -r env_var; do
    [[ -z "$env_var" ]] && continue

    # Keep the resolver call inside the ``if`` condition: ``--check`` exits 3
    # (stale mismatch) and 4 (nothing usable), and a bare assignment would trip
    # ``set -e`` before the failure report below is printed.
    report=""
    check_rc=0
    if report="$(check_api_key "$env_var")"; then
      check_rc=0
    else
      check_rc=$?
    fi

    status="" env_fp="" auth_fp="" resolved_key="" candidates=""
    IFS=$'\x1f' read -r status env_fp auth_fp resolved_key candidates <<< "$report"

    case "$status" in
      match)
        # Environment key agrees with auth.json: keep it (no export needed) and
        # log the fingerprint so a stale key is diagnosable from the log alone.
        echo "[env] $env_var from environment (matches auth.json, fp=$env_fp)"
        ;;
      resolved)
        # Environment unset: export the auth.json key, log its fingerprint.
        export "$env_var=$resolved_key"
        echo "[env] $env_var resolved from ~/.pi/agent/auth.json (fp=$auth_fp)"
        ;;
      mismatch)
        if [ "$USE_ENV_KEYS" -eq 1 ]; then
          echo "WARNING: $env_var in the environment differs from ~/.pi/agent/auth.json (env fp=$env_fp, auth.json fp=$auth_fp) — keeping the environment key because the stale-key opt-out is set; a rotated auth.json is IGNORED for this restart." >&2
        else
          mismatched_vars+=("$env_var")
          mismatch_lines+=("$env_var  (environment fp=${env_fp:-<empty>}, auth.json fp=$auth_fp; auth.json key(s) tried: ${candidates:-unknown}) — stale environment key, MISMATCH")
        fi
        ;;
      env_only|unresolved|auth_missing)
        if [[ -n "$env_fp" ]]; then
          # auth.json unavailable / no matching entry: keep the env key but say
          # that staleness could not be verified.
          echo "WARNING: $env_var from environment only (fp=$env_fp) — ~/.pi/agent/auth.json has no matching entry, so key staleness cannot be verified." >&2
        else
          # Nothing usable anywhere: fall through to the aggregated error.
          models_using="$(models_requiring "$env_var")"
          missing+=("$env_var  (required by: $models_using)")
        fi
        ;;
      *)
        # Fail closed on an unparseable report rather than starting blind.
        echo "ERROR: key check failed for $env_var (resolver exit $check_rc, no report); cannot verify the API key source." >&2
        models_using="$(models_requiring "$env_var")"
        missing+=("$env_var  (key check failed; required by: $models_using)")
        ;;
    esac
  done < <(enumerate_api_key_envs)

  if [[ ${#mismatched_vars[@]} -gt 0 ]]; then
    echo "" >&2
    echo "ERROR: stale API key environment variable(s) detected — refusing to start." >&2
    echo "       The environment disagrees with $AUTH_FILE, so a rotated" >&2
    echo "       auth.json would be silently ignored:" >&2
    for line in "${mismatch_lines[@]}"; do
      echo "  - $line" >&2
    done
    echo "" >&2
    echo "Fix one of these, then restart:" >&2
    echo "  1. Unset the stale variable(s) so the auth.json key is used:" >&2
    for var in "${mismatched_vars[@]}"; do
      echo "       unset $var" >&2
    done
    echo "  2. Re-export the current key from $AUTH_FILE." >&2
    echo "  3. Explicitly keep the environment keys (stale-key detection is" >&2
    echo "     then disabled for this restart):" >&2
    echo "       --use-env-keys   or   LLAMA_PROXY_USE_ENV_KEYS=1" >&2
    echo "" >&2
    return 1
  fi

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

echo "=== LLM Proxy API Key Check ==="
resolve_api_keys

echo ""
echo "=== Starting proxy server ==="

# Exec uvicorn using chosen python binary
exec "$PY_BIN" -m uvicorn proxy.server:app --host 0.0.0.0 --port "$PORT" "${UVICORN_ARGS[@]+${UVICORN_ARGS[@]}}"
