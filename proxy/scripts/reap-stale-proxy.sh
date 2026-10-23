#!/usr/bin/env bash
# Reap wedged LLM-proxy twin processes (LP-0MVLHMAWV008EKTI AC2).
#
# On 2026-10-23 two `python3 -m uvicorn proxy.server:app` processes were alive
# at once: the newest owned the :8000 listener while its predecessor lingered
# in ep_poll with no listener but with proxy.log open for write — so it could
# still rotate/truncate the log another process was writing, which is how a
# 3 h 35 m verification window vanished. start-proxy.sh's launch-time check
# only refused on a *listening* port, so a bound-less twin was never caught.
#
# A *listening* proxy can never be reaped by this script: start-proxy.sh's
# port check runs first and refuses. Any process matching the server command
# line for a given port is therefore either a legitimate server that lost its
# listener or a stale duplicate — in both cases it must not keep a second
# writer attached to proxy.log, so it is terminated (SIGTERM, grace, SIGKILL).
#
# Usage:
#   reap-stale-proxy.sh [--port N] [--grace S] [--dry-run]
#   --port N     only consider processes started with --port N (default 8000)
#   --grace S    seconds between SIGTERM and SIGKILL (default 5)
#   --dry-run    report what would be reaped without killing anything
#
# Exit status: 0 on success (including "nothing to reap"); 2 on usage errors.
set -uo pipefail

PORT=8000
GRACE=5
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --port=*)
      PORT="${1#*=}"
      ;;
    --port)
      shift
      PORT="${1:-}"
      ;;
    --grace=*)
      GRACE="${1#*=}"
      ;;
    --grace)
      shift
      GRACE="${1:-}"
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      cat <<'USAGE'
Reap wedged LLM-proxy twin processes (LP-0MVLHMAWV008EKTI AC2).

Usage:
  reap-stale-proxy.sh [--port N] [--grace S] [--dry-run]

  --port N     only consider processes started with --port N (default 8000)
  --grace S    seconds between SIGTERM and SIGKILL (default 5)
  --dry-run    report what would be reaped without killing anything

A listening proxy is never reaped: start-proxy.sh's port check refuses first.
USAGE
      exit 0
      ;;
    *)
      echo "reap-stale-proxy: unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
  echo "reap-stale-proxy: invalid port: $PORT" >&2
  exit 2
fi

command -v pgrep >/dev/null 2>&1 || {
  echo "reap-stale-proxy: pgrep not available; cannot scan for stale proxies" >&2
  exit 0
}

# The full command line of a launched server is
#   python3 -m uvicorn proxy.server:app --host 0.0.0.0 --port N [args...]
# Match on the port so a legitimate dev instance on another port is never
# touched, and never match this script's own process (its argv has no
# "uvicorn proxy.server" text).
PATTERN="uvicorn proxy\.server.*--port[ =]${PORT}([[:space:]]|\$)"

_cmdline() {
  # Join /proc/<pid>/cmdline NULs with spaces (matches what pgrep matches on).
  tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null
}

# Re-verify against the live /proc cmdline so a recycled pid is never killed.
_matches() {
  local pid="$1"
  [ -r "/proc/$pid/cmdline" ] || return 1
  _cmdline "$pid" | grep -Eq "$PATTERN"
}

# shellcheck disable=SC2009
PIDS="$(pgrep -f "$PATTERN" 2>/dev/null || true)"

# Exclude our own pid and parent (this script / start-proxy.sh) if they ever
# matched the pattern (defensive; they should not).
SELF=$$
PPID_SELF=$PPID

REAPED=0
FOUND=0
for pid in $PIDS; do
  [ "$pid" = "$SELF" ] && continue
  [ "$pid" = "$PPID_SELF" ] && continue
  if _matches "$pid"; then
    FOUND=$((FOUND + 1))
    cmdline="$(_cmdline "$pid")"
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "reap-stale-proxy: would reap $pid: $cmdline"
      continue
    fi
    echo "reap-stale-proxy: stale proxy twin $pid has no listener on port $PORT; terminating (grace ${GRACE}s): $cmdline"
    kill -TERM "$pid" 2>/dev/null || true
    REAPED=$((REAPED + 1))
  fi
done

if [ "$DRY_RUN" -eq 1 ]; then
  if [ "$FOUND" -eq 0 ]; then
    echo "reap-stale-proxy: no stale proxy process would be reaped for port $PORT"
  fi
  exit 0
fi

# Grace period, then the kill -9 backstop for survivors (mirrors
# start-proxy.sh's two-phase teardown).
if [ "$REAPED" -gt 0 ]; then
  sleep "$GRACE"
  for pid in $PIDS; do
    [ "$pid" = "$SELF" ] && continue
    [ "$pid" = "$PPID_SELF" ] && continue
    if _matches "$pid" && kill -0 "$pid" 2>/dev/null; then
      echo "reap-stale-proxy: forcing kill of surviving twin $pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
fi

if [ "$REAPED" -eq 0 ]; then
  echo "reap-stale-proxy: no stale proxy process found for port $PORT"
else
  echo "reap-stale-proxy: reaped $REAPED stale proxy process(es) for port $PORT"
fi
exit 0
