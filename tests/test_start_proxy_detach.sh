#!/usr/bin/env bash
# Regression test for start-proxy.sh launching the proxy detached by default.
#
# Pins AC3 of LP-0MVGT0E6W000Q8CY (hardening proxy start against a foreground
# invocation being killed by tool-call timeouts). The failure mode: the script
# ended with `exec uvicorn ...`, so uvicorn ran inside whatever process group
# launched the script. An agent tool timeout SIGKILLs that whole group, which
# takes the proxy down mid-startup with no `Shutting down` line — and because
# the same invocation had already run `--restart`, the proxy is left offline.
#
# The test proves the launched server survives `SIGKILL` of the launcher's
# entire process group:
#   1. the launcher runs in its own process group (session leader) and the
#      sandboxed fake uvicorn binds an ephemeral port;
#   2. the fake server is launched exactly once (no detach re-entry) and its
#      pgid differs from the launcher's (it lives in its own session);
#   3. the launcher prints a single handoff line (child PID, log path,
#      health-check command) and exits 0;
#   4. the child's stdout/stderr go to the log file, so the caller's stdout
#      reaches EOF promptly (the caller's pipe is not held open — otherwise
#      `... | tail -20` still blocks until the tool timeout);
#   5. `kill -9 -<launcher-pgid>` leaves the listener answering.
#
# Sandbox (mirrors tests/test_start_proxy_keys.sh and test_start_proxy_mode.sh):
# a copy of start-proxy.sh, a fake `python3` that stands in for uvicorn and
# records its pid/pgid/sid, an isolated $HOME, stubbed ss/netstat port probes
# and an ephemeral port. The live proxy, port 8000, llama-server and the TTS
# server are never touched, and only PIDs this harness recorded are killed
# (never a blanket pkill — the proxy shares the host with every consumer).
#
# The launcher is started from the sandbox (not the repo root) so the script
# resolves the fake `python3` from PATH rather than a repository `.venv` — see
# the launcher wrapper below.
#
# Usage:
#   bash tests/test_start_proxy_detach.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_SCRIPT="$SCRIPT_DIR/../proxy/scripts/start-proxy.sh"
REAL_PROXY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/proxy"
PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); echo "PASS: $*"; }
fail() { FAIL=$((FAIL + 1)); echo "FAIL: $*"; }

SANDBOX=""
LAUNCHER_PID=""
LAUNCHER_PGID=""
SERVER_PID=""
READER_PID=""

cleanup() {
    local rc=$?
    # Kill ONLY processes this harness started, addressed by recorded PID/PGID.
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -s TERM "$SERVER_PID" 2>/dev/null || true
        local i
        for i in $(seq 1 40); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 0.05
        done
        kill -s KILL "$SERVER_PID" 2>/dev/null || true
    fi
    if [ -n "$LAUNCHER_PID" ] && kill -0 "$LAUNCHER_PID" 2>/dev/null; then
        local own_pgid
        own_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
        if [ -n "$LAUNCHER_PGID" ] && [ "$LAUNCHER_PGID" != "$own_pgid" ]; then
            kill -s KILL -- "-$LAUNCHER_PGID" 2>/dev/null || true
        else
            kill -s KILL "$LAUNCHER_PID" 2>/dev/null || true
        fi
    fi
    if [ -n "$READER_PID" ] && kill -0 "$READER_PID" 2>/dev/null; then
        kill -s KILL "$READER_PID" 2>/dev/null || true
    fi
    rm -rf "${SANDBOX:-}"
    if [ "$rc" -ne 0 ] && [ "$FAIL" -eq 0 ]; then
        FAIL=$((FAIL + 1))
        echo "FAIL: unexpected error (exit $rc)"
    fi
    echo ""
    echo "=== Results: $PASS passed, $FAIL failed ==="
    [ "$FAIL" -eq 0 ]
}
trap cleanup EXIT

if [ ! -f "$SOURCE_SCRIPT" ]; then
    fail "start-proxy.sh not found at $SOURCE_SCRIPT"
    exit 1
fi

# ---- Sandbox ---------------------------------------------------------------

SANDBOX="$(mktemp -d)"
PROXY_DIR="$SANDBOX/proxy"
mkdir -p "$PROXY_DIR/scripts"
cp "$SOURCE_SCRIPT" "$PROXY_DIR/scripts/start-proxy.sh"
chmod +x "$PROXY_DIR/scripts/start-proxy.sh"

# Mirror the dependency-light proxy package (config_merge for config/key
# enumeration, api_key_resolver for the key check). Copying the whole top-level
# *.py set keeps the harness working if the detach change adds a sibling helper.
mkdir -p "$PROXY_DIR/proxy"
cp "$REAL_PROXY_DIR/proxy/"*.py "$PROXY_DIR/proxy/"
for _required in config_merge.py api_key_resolver.py; do
    if [ ! -f "$PROXY_DIR/proxy/$_required" ]; then
        fail "could not mirror proxy/proxy/$_required into the sandbox"
        exit 1
    fi
done

# Isolated HOME: the script reads $HOME/.pi/agent/auth.json. The fixture config
# declares no api_key_env, so no provider key is required.
mkdir -p "$SANDBOX/home/.pi/agent"
printf '{}\n' > "$SANDBOX/home/.pi/agent/auth.json"

# Minimal config: a single local provider, no api_key_env (so the run never
# needs a real provider key and never reaches the network).
cat > "$PROXY_DIR/config.yaml" <<'YAML'
models:
  test-model:
    providers:
      - name: test-local
        type: local
        llama_model: Test
server:
  session_slot_pool_size: 1
YAML

# Ephemeral port for the fake server; never 8000/8080/8081.
pick_free_port() {
    /usr/bin/python3 - <<'PY'
import socket

sock = socket.socket()
sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1])
sock.close()
PY
}

TEST_PORT="$(pick_free_port)"
if [ -z "$TEST_PORT" ]; then
    fail "could not pick an ephemeral port"
    exit 1
fi

# Fake uvicorn: a minimal HTTP listener that stands in for the proxy and
# records the PID/PGID/SID of the process that ends up serving. Each launch
# appends one state line, so the test can prove it was launched exactly once.
SERVER_PY="$SANDBOX/fake_server.py"
cat > "$SERVER_PY" <<'PY'
import os
import socket
import sys

port = int(sys.argv[1])
srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", port))
srv.listen(16)
print("FAKE_SERVER listening port=%d pid=%d" % (port, os.getpid()), flush=True)
while True:
    try:
        conn, _ = srv.accept()
    except OSError:
        break
    try:
        conn.recv(1024)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
    except OSError:
        pass
    finally:
        conn.close()
PY

STATE_FILE="$SANDBOX/fake-uvicorn.state"
FAKE_PY="$SANDBOX/python3"
cat > "$FAKE_PY" <<'PY'
#!/usr/bin/env bash
# Intercept the final `exec <python> -m uvicorn ...`; delegate everything else
# to the real interpreter (the script uses python for its config/key helpers).
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "uvicorn" ]; then
    port=""
    prev=""
    for a in "$@"; do
        if [ "$prev" = "--port" ]; then port="$a"; fi
        prev="$a"
    done
    pid="$$"
    pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
    sid="$(ps -o sid= -p "$pid" | tr -d ' ')"
    printf 'pid=%s pgid=%s sid=%s port=%s\n' "$pid" "$pgid" "$sid" "$port" >> "$FAKE_UVICORN_STATE"
    echo "FAKE_UVICORN_STDOUT pid=$pid pgid=$pgid sid=$sid port=$port args=$*"
    echo "FAKE_UVICORN_STDERR pid=$pid pgid=$pgid sid=$sid port=$port" >&2
    exec /usr/bin/python3 "$FAKE_SERVER_PY" "$port"
fi
exec /usr/bin/python3 "$@"
PY
chmod +x "$FAKE_PY"

# Stub port probes: report no listeners, so the sandboxed script never probes
# the live proxy ports (and never sees a real service on our ephemeral port).
for _fake in ss netstat; do
    printf '#!/usr/bin/env bash\nexit 0\n' > "$SANDBOX/$_fake"
    chmod +x "$SANDBOX/$_fake"
done

# ---- Launcher --------------------------------------------------------------
# The wrapper records its own PID and PGID (it is the session leader started
# with `setsid`) before handing over to start-proxy.sh, and records the
# script's exit code. It runs the script exactly as an agent/caller would, but
# with a sandboxed log file instead of the default /tmp/proxy-startup.log.

LAUNCHER_PID_FILE="$SANDBOX/launcher.pid"
LAUNCHER_RC_FILE="$SANDBOX/launcher.rc"
LOG_FILE="$SANDBOX/detached.log"
CALLER_OUT="$SANDBOX/caller.out"
CALLER_ERR="$SANDBOX/caller.err"
PIPE="$SANDBOX/caller.pipe"
EOF_RC_FILE="$SANDBOX/reader.rc"

cat > "$SANDBOX/launcher.sh" <<'SH'
#!/usr/bin/env bash
echo "$$ $(ps -o pgid= -p $$ | tr -d ' ')" > "$LAUNCHER_PID_FILE"
# Run from the sandbox, never from the repository root: start-proxy.sh prefers
# <cwd>/.venv/bin/python3, so a launcher started in the repo root would pick the
# real interpreter and try to import the real app instead of the fake uvicorn on
# PATH (the sandbox mirrors only the top-level proxy package).
cd "$SANDBOX" || exit 1
bash "$SANDBOX/proxy/scripts/start-proxy.sh" --port "$TEST_PORT" --log-file "$LOG_FILE" "$@"
rc=$?
echo "$rc" > "$LAUNCHER_RC_FILE"
exit "$rc"
SH
chmod +x "$SANDBOX/launcher.sh"

export SANDBOX TEST_PORT LOG_FILE LAUNCHER_PID_FILE LAUNCHER_RC_FILE
export FAKE_UVICORN_STATE="$STATE_FILE" FAKE_SERVER_PY="$SERVER_PY"
export HOME="$SANDBOX/home"
export PATH="$SANDBOX:$PATH"

# ---- Helpers ---------------------------------------------------------------

port_answers() {
    /usr/bin/python3 - "$TEST_PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", port))
    sock.sendall(b"GET /health HTTP/1.0\r\n\r\n")
    data = sock.recv(64)
except OSError:
    sys.exit(1)
finally:
    sock.close()
sys.exit(0 if data else 1)
PY
}

wait_for_listener() {
    local i
    for i in $(seq 1 150); do
        if port_answers; then return 0; fi
        sleep 0.1
    done
    return 1
}

wait_for_file() {
    # Usage: wait_for_file <path> <tenths-of-a-second-budget>
    local path="$1" budget="$2" i
    for i in $(seq 1 "$budget"); do
        if [ -s "$path" ]; then return 0; fi
        sleep 0.1
    done
    return 1
}

state_field() {
    awk -v key="$1" '{
        for (i = 1; i <= NF; i++) {
            split($i, kv, "=")
            if (kv[1] == key) { print kv[2]; exit }
        }
    }' "$STATE_FILE" 2>/dev/null
}

caller_output() {
    cat "$CALLER_OUT" "$CALLER_ERR" 2>/dev/null
}

show_diagnostics() {
    echo "--- diagnostics ---"
    echo "launcher pid/pgid: ${LAUNCHER_PID:-<none>}/${LAUNCHER_PGID:-<none>}"
    echo "launcher exit code: $(cat "$LAUNCHER_RC_FILE" 2>/dev/null || echo '<did not exit>')"
    echo "fake-uvicorn state: $(cat "$STATE_FILE" 2>/dev/null || echo '<none>')"
    echo "caller stdout: $(cat "$CALLER_OUT" 2>/dev/null || echo '<empty>')"
    echo "caller stderr: $(cat "$CALLER_ERR" 2>/dev/null || echo '<empty>')"
    echo "detached log ($LOG_FILE): $(cat "$LOG_FILE" 2>/dev/null || echo '<missing>')"
    echo "--- end diagnostics ---"
}

# ---- Run the launcher ------------------------------------------------------

echo "=== start-proxy.sh Detached-Launch Survival Tests ==="
echo ""

mkfifo "$PIPE"
# The reader only sees EOF once every writer closed the pipe: if the launched
# server inherits the caller's stdout, this blocks until the timeout expires.
( timeout 20 cat "$PIPE" > "$CALLER_OUT"; echo $? > "$EOF_RC_FILE" ) &
READER_PID=$!

setsid bash "$SANDBOX/launcher.sh" > "$PIPE" 2> "$CALLER_ERR" &

if wait_for_file "$LAUNCHER_PID_FILE" 100; then
    read -r LAUNCHER_PID LAUNCHER_PGID < "$LAUNCHER_PID_FILE"
    pass "launcher started (pid $LAUNCHER_PID, pgid $LAUNCHER_PGID)"
else
    fail "launcher never recorded its PID"
    show_diagnostics
    exit 1
fi

if [ "$LAUNCHER_PGID" = "$LAUNCHER_PID" ]; then
    pass "launcher runs in its own process group (pgid == pid)"
else
    fail "launcher is not a process-group leader (pid $LAUNCHER_PID, pgid $LAUNCHER_PGID)"
fi

if wait_for_listener; then
    pass "the launched server answers on the ephemeral port $TEST_PORT"
else
    fail "nothing is listening on the ephemeral port $TEST_PORT"
    show_diagnostics
    exit 1
fi

SERVER_PID="$(state_field pid)"
SERVER_PGID="$(state_field pgid)"
SERVER_SID="$(state_field sid)"

# ---- Assertion 1: the launcher exits (it hands off, it does not serve) -----

if wait_for_file "$LAUNCHER_RC_FILE" 100; then
    LAUNCHER_RC="$(cat "$LAUNCHER_RC_FILE")"
    if [ "$LAUNCHER_RC" -eq 0 ]; then
        pass "launcher exited 0 after the handoff"
    else
        fail "launcher exited $LAUNCHER_RC (expected 0)"
    fi
else
    fail "launcher never exited — it is still running the server in the foreground"
fi

# ---- Assertion 2: the caller's stdout reaches EOF promptly ------------------

if wait_for_file "$EOF_RC_FILE" 250; then
    READER_RC="$(cat "$EOF_RC_FILE")"
    if [ "$READER_RC" -eq 0 ]; then
        pass "caller's stdout reached EOF promptly (child does not hold the pipe)"
    else
        fail "caller's stdout stayed open (reader exit $READER_RC): the launched server inherited the caller's pipe"
    fi
else
    fail "caller's stdout never reached EOF within 25s"
fi

# ---- Assertion 3: launched exactly once, in its own session ----------------

if [ -f "$STATE_FILE" ]; then
    LAUNCH_COUNT="$(grep -c . "$STATE_FILE")"
else
    LAUNCH_COUNT=0
fi
if [ "$LAUNCH_COUNT" -eq 1 ]; then
    pass "the server was launched exactly once (no detach re-entry)"
else
    fail "the server was launched $LAUNCH_COUNT time(s) (expected exactly 1)"
fi

if [ -n "$SERVER_PGID" ] && [ "$SERVER_PGID" != "$LAUNCHER_PGID" ]; then
    pass "the server's pgid ($SERVER_PGID) differs from the launcher's ($LAUNCHER_PGID)"
else
    fail "the server shares the launcher's process group (server pgid '${SERVER_PGID:-<unknown>}', launcher pgid '$LAUNCHER_PGID')"
fi

if [ -n "$SERVER_PGID" ] && [ "$SERVER_PGID" != "$LAUNCHER_PGID" ] && [ "$SERVER_SID" = "$SERVER_PGID" ]; then
    pass "the server is detached into its own session (sid == pgid == $SERVER_SID)"
else
    fail "the server is not in its own session (sid '${SERVER_SID:-<unknown>}', pgid '${SERVER_PGID:-<unknown>}', launcher pgid '$LAUNCHER_PGID')"
fi

# ---- Assertion 4: a single handoff line (PID, log path, health check) ------

# The child's own output (when it wrongly inherits the caller's pipe) also
# names the log file, via the forwarded --log-file argument; exclude it so the
# handoff assertion cannot pass on the harness's/fake server's own output.
HANDOFF_LINES="$(caller_output | grep -F -- "$LOG_FILE" | grep -viE '^FAKE_UVICORN' || true)"
HANDOFF_COUNT=0
if [ -n "$HANDOFF_LINES" ]; then
    HANDOFF_COUNT="$(printf '%s\n' "$HANDOFF_LINES" | grep -c .)"
fi

if [ "$HANDOFF_COUNT" -eq 1 ]; then
    pass "the launcher printed exactly one handoff line"
else
    fail "expected exactly one handoff line naming $LOG_FILE, got $HANDOFF_COUNT"
fi

if [ "$HANDOFF_COUNT" -eq 1 ]; then
    HANDOFF="$HANDOFF_LINES"
    HANDOFF_PID_OK=0
    for token in $(printf '%s\n' "$HANDOFF" | grep -oE '[0-9]+'); do
        if [ "$token" = "$SERVER_PID" ] || [ "$token" = "$SERVER_PGID" ] || [ "$token" = "$SERVER_SID" ]; then
            HANDOFF_PID_OK=1
        fi
    done
    if [ "$HANDOFF_PID_OK" -eq 1 ]; then
        pass "the handoff line names the detached child PID (server pid $SERVER_PID)"
    else
        fail "the handoff line names no PID from the detached session (server pid $SERVER_PID, pgid $SERVER_PGID): $HANDOFF"
    fi

    if printf '%s\n' "$HANDOFF" | grep -qiE 'curl.*health|health.*curl'; then
        pass "the handoff line gives the health-check command"
    else
        fail "the handoff line gives no curl health-check command: $HANDOFF"
    fi
fi

# ---- Assertion 5: the child's stdout/stderr go to the log file -------------

if [ -f "$LOG_FILE" ] && grep -q '^FAKE_UVICORN_STDOUT' "$LOG_FILE"; then
    pass "the child's stdout is redirected to the log file"
else
    fail "the child's stdout is not in $LOG_FILE"
fi

if [ -f "$LOG_FILE" ] && grep -q '^FAKE_UVICORN_STDERR' "$LOG_FILE"; then
    pass "the child's stderr is redirected to the log file"
else
    fail "the child's stderr is not in $LOG_FILE"
fi

if caller_output | grep -q 'FAKE_UVICORN'; then
    fail "the child's output leaked into the caller's stdout/stderr"
else
    pass "the child's output never reaches the caller's stdout/stderr"
fi

# ---- Assertion 6: the server survives SIGKILL of the launcher's group ------

OWN_PGID="$(ps -o pgid= -p $$ | tr -d ' ')"
if [ -z "$LAUNCHER_PGID" ] || [ "$LAUNCHER_PGID" = "$OWN_PGID" ]; then
    fail "refusing to SIGKILL: launcher pgid '${LAUNCHER_PGID:-<unknown>}' is unset or the harness's own"
else
    kill -s KILL -- "-$LAUNCHER_PGID" 2>/dev/null || true
    sleep 1
    if port_answers; then
        pass "the server still answers after SIGKILL of the launcher's process group (-$LAUNCHER_PGID)"
    else
        fail "the server died with the launcher's process group (-$LAUNCHER_PGID)"
    fi
fi

if [ "$FAIL" -gt 0 ]; then
    show_diagnostics
fi

# The cleanup trap prints the results footer; force the documented non-zero
# exit on failure (a trap's last command does not change the script status).
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
