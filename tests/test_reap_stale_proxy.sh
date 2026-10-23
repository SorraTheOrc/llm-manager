#!/usr/bin/env bash
# Harness for proxy/scripts/reap-stale-proxy.sh
# (LP-0MVLHMAWV008EKTI AC2).
#
# The 2026-10-23 incident left a second `uvicorn proxy.server:app` process
# alive with no listener but with proxy.log open for write. start-proxy.sh only
# refused when the *port* was taken, so the wedged twin survived every start
# and could rotate/truncate the shared log.
#
# This harness spawns stand-in processes whose command line matches the real
# server shape (a python process carrying `uvicorn proxy.server:app ... --port
# N` in its argv, holding a log file open) and asserts the reaper:
#   1. terminates the wedged twin for the requested port;
#   2. leaves a proxy-shaped process on a *different* port alone;
#   3. leaves unrelated processes alone;
#   4. reports without killing under --dry-run, and is a no-op when clean;
#   5. NEVER reaps while something is listening on the port — a serving proxy
#      must never be killed by the safety-net script (AGENTS.md: never kill the
#      proxy).
#
# No production proxy, llama-server, TTS, port 8000 or live log is touched.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REAPER="$REPO_ROOT/proxy/scripts/reap-stale-proxy.sh"

PASS=0
FAIL=0
pass() { PASS=$((PASS + 1)); echo "PASS: $*"; }
fail() { FAIL=$((FAIL + 1)); echo "FAIL: $*"; }

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

# _spawn_proxy_shaped <port> <logfile>
# A stand-in for a wedged proxy twin: argv carries the uvicorn proxy.server
# command line and the process holds a log file open for write.
_spawn_proxy_shaped() {
  local port="$1" logfile="$2"
  TWIN_LOG="$logfile" python3 -c '
import os, time
fh = open(os.environ["TWIN_LOG"], "a")
fh.write("twin holding the log\n")
fh.flush()
time.sleep(120)
' "uvicorn proxy.server:app --host 0.0.0.0 --port ${port}" >/dev/null 2>&1 &
  PIDS+=("$!")
  echo "$!"
}

_wait_gone() {
  local pid="$1" deadline=$((SECONDS + 15))
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      return 1
    fi
    sleep 0.2
  done
  return 0
}

TMP_DIR="$(mktemp -d)"
trap 'cleanup; rm -rf "$TMP_DIR"' EXIT

PORT_TWIN=$((20000 + RANDOM % 20000))
PORT_OTHER=$((PORT_TWIN + 1))
PORT_CLEAN=$((PORT_OTHER + 1))

# --- 0. sanity: the reaper exists ------------------------------------------
if [ ! -x "$REAPER" ]; then
  fail "reaper script missing or not executable: $REAPER"
  echo "=== Results: $PASS passed, $FAIL failed ==="
  exit 1
fi

# --- 1. reaps the wedged twin, leaves the other port alone ------------------
TWIN_PID="$(_spawn_proxy_shaped "$PORT_TWIN" "$TMP_DIR/twin.log")"
OTHER_PID="$(_spawn_proxy_shaped "$PORT_OTHER" "$TMP_DIR/other.log")"
sleep 1

if kill -0 "$TWIN_PID" 2>/dev/null; then
  pass "stand-in twin is running before reaping (pid $TWIN_PID)"
else
  fail "stand-in twin failed to start"
fi

OUT="$(bash "$REAPER" --port "$PORT_TWIN" 2>&1)"
RC=$?
if [ "$RC" -eq 0 ]; then
  pass "reaper exits 0"
else
  fail "reaper exited $RC: $OUT"
fi

if _wait_gone "$TWIN_PID"; then
  pass "wedged twin on port $PORT_TWIN was reaped"
else
  fail "wedged twin on port $PORT_TWIN survived"
fi

if kill -0 "$OTHER_PID" 2>/dev/null; then
  pass "proxy-shaped process on the other port was left alone"
else
  fail "reaper killed a process on a different port (out of scope)"
fi

if echo "$OUT" | grep -q "$TWIN_PID"; then
  pass "reaper reports the reaped pid"
else
  fail "reaper output does not name the reaped pid: $OUT"
fi

# --- 2. unrelated processes are untouched ----------------------------------
sleep 120 &
UNRELATED_PID=$!
PIDS+=("$UNRELATED_PID")
bash "$REAPER" --port "$PORT_TWIN" >/dev/null 2>&1
if kill -0 "$UNRELATED_PID" 2>/dev/null; then
  pass "unrelated process untouched"
else
  fail "reaper killed an unrelated process"
fi
kill -9 "$UNRELATED_PID" 2>/dev/null || true

# --- 3. --dry-run reports without killing ---------------------------------
PORT_DRY=$((PORT_CLEAN + 1))
DRY_PID="$(_spawn_proxy_shaped "$PORT_DRY" "$TMP_DIR/dry.log")"
sleep 1
DRY_OUT="$(bash "$REAPER" --port "$PORT_DRY" --dry-run 2>&1)"
if kill -0 "$DRY_PID" 2>/dev/null; then
  pass "--dry-run does not kill"
else
  fail "--dry-run killed the stand-in twin"
fi
if echo "$DRY_OUT" | grep -qi "would reap"; then
  pass "--dry-run reports what it would reap"
else
  fail "--dry-run output missing the would-reap report: $DRY_OUT"
fi
bash "$REAPER" --port "$PORT_DRY" >/dev/null 2>&1
_wait_gone "$DRY_PID" && pass "cleanup reap after --dry-run" \
  || fail "cleanup reap after --dry-run failed"

# --- 5. a live listener on the port is never reaped ------------------------
# A healthy proxy holds the listener: the reaper must refuse, not kill it.
PORT_LISTENING=$((PORT_CLEAN + 2))
LISTENER_PID="$(_spawn_proxy_shaped "$PORT_LISTENING" "$TMP_DIR/listening.log")"
# Give the stand-in a real listener on the requested port.
python3 - "$PORT_LISTENING" <<'PY' >/dev/null 2>&1 &
import socket, sys, time
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", port))
s.listen(1)
time.sleep(120)
PY
SOCKET_PID=$!
PIDS+=("$SOCKET_PID")
sleep 1

LISTEN_OUT="$(bash "$REAPER" --port "$PORT_LISTENING" 2>&1)"
if kill -0 "$LISTENER_PID" 2>/dev/null; then
  pass "a serving proxy on the port is never reaped"
else
  fail "reaper killed a proxy-shaped process whose port has a live listener"
fi
if echo "$LISTEN_OUT" | grep -qi "listener"; then
  pass "reaper explains the listener interlock"
else
  fail "reaper output missing the listener interlock message: $LISTEN_OUT"
fi
kill -9 "$LISTENER_PID" "$SOCKET_PID" 2>/dev/null || true

# --- 6. a clean port is a no-op -------------------------------------------
CLEAN_OUT="$(bash "$REAPER" --port "$PORT_CLEAN" 2>&1)"
RC=$?
if [ "$RC" -eq 0 ] && echo "$CLEAN_OUT" | grep -qi "no stale"; then
  pass "clean port reports no stale proxy process and exits 0"
else
  fail "clean port run rc=$RC output=$CLEAN_OUT"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] || exit 1
