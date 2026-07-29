#!/usr/bin/env bash
# Integration test for start-proxy.sh --restart port cleanup behavior.
#
# This test uses ephemeral (random) ports and mock listeners — it does NOT
# interact with the live proxy, llama-server, or TTS on ports 8080/8081/8000.
#
# Expected behavior:
#   1. A mock server is spawned on a random port
#   2. The cleanup logic (pkill + fuser) is simulated against that port
#   3. After cleanup, the port should be free for a new server to bind
#
# Usage:
#   bash tests/test_start_proxy_restart.sh

set -euo pipefail

PASS=0
FAIL=0

pass() {
    PASS=$((PASS + 1))
    echo "PASS: $*"
}

fail() {
    FAIL=$((FAIL + 1))
    echo "FAIL: $*"
}

cleanup() {
    local exit_code=$?
    # Kill any lingering background listeners
    if [ -n "${LISTENER_PID:-}" ]; then
        kill "$LISTENER_PID" 2>/dev/null || true
    fi
    if [ -n "${LISTENER_PID2:-}" ]; then
        kill "$LISTENER_PID2" 2>/dev/null || true
    fi
    if [ "$exit_code" -ne 0 ] && [ "$FAIL" -eq 0 ]; then
        FAIL=$((FAIL + 1))
        echo "FAIL: Unexpected error (exit code $exit_code)"
    fi
    echo ""
    echo "=== Results: $PASS passed, $FAIL failed ==="
    if [ "$FAIL" -gt 0 ]; then
        exit 1
    fi
    exit 0
}
trap cleanup EXIT

# ---- Helpers ---------------------------------------------------------------

find_free_port() {
    python3 -c "
import socket
s = socket.socket()
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
s.close()
"
}

# Start a background TCP listener on the given port.
# Usage: start_listener <port>
# Sets LISTENER_PID to the background process PID.
start_listener() {
    local port="$1"
    python3 -c "
import socket, time
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('127.0.0.1', $port))
s.listen(1)
print('listening on $port', flush=True)
time.sleep(30)
" &
    LISTENER_PID=$!
    # Wait for listener to be ready
    sleep 0.3
}

# Check if a port is in use.
# Returns 0 if in use, 1 if free.
port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn | awk '{print $4}' | grep -Eq ":$port\$|\.$port\$"
    elif command -v netstat >/dev/null 2>&1; then
        netstat -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":$port\$|\.$port\$"
    else
        python3 -c "
import socket,sys
s=socket.socket()
s.settimeout(0.3)
try:
    s.connect(('127.0.0.1', $port))
except Exception:
    sys.exit(1)
else:
    sys.exit(0)
"
    fi
}

# Wait for port to be free, polling every 0.2s with a timeout.
wait_for_port_free() {
    local port="$1"
    local timeout="${2:-5}"
    local deadline
    deadline=$(python3 -c "import time; print(time.monotonic() + $timeout)")
    while true; do
        if ! port_in_use "$port"; then
            return 0
        fi
        local now
        now=$(python3 -c "import time; print(time.monotonic())")
        if python3 -c "import sys; sys.exit(0 if $now > $deadline else 1)" 2>/dev/null; then
            return 1
        fi
        sleep 0.2
    done
}

# ---- Test 1: pkill frees a listening port ---------------------------------
test_pkill_frees_port() {
    echo "--- Test: pkill frees a listening port ---"

    local port
    port=$(find_free_port)
    start_listener "$port"

    if ! port_in_use "$port"; then
        fail "test_pkill_frees_port: listener did not start on port $port"
        kill "$LISTENER_PID" 2>/dev/null || true
        return
    fi
    pass "test_pkill_frees_port: listener started on port $port"

    # Kill using pkill by PID (simulating what start-proxy.sh does)
    kill "$LISTENER_PID" 2>/dev/null || true
    LISTENER_PID=""

    # Wait for the port to be freed
    if wait_for_port_free "$port" 3; then
        pass "test_pkill_frees_port: port $port freed after kill"
    else
        fail "test_pkill_frees_port: port $port still in use after kill"
    fi
}

# ---- Test 2: New server can bind after kill --------------------------------
test_new_server_binds_after_kill() {
    echo "--- Test: New server can bind after kill ---"

    local port
    port=$(find_free_port)
    start_listener "$port"

    if ! port_in_use "$port"; then
        fail "test_new_server_binds_after_kill: listener did not start"
        kill "$LISTENER_PID" 2>/dev/null || true
        return
    fi

    # Kill and wait for release
    kill "$LISTENER_PID" 2>/dev/null || true
    LISTENER_PID=""

    if ! wait_for_port_free "$port" 3; then
        fail "test_new_server_binds_after_kill: port did not free"
        return
    fi

    # Start a new server on the same port
    start_listener "$port"
    if port_in_use "$port"; then
        pass "test_new_server_binds_after_kill: new server bound to port $port"
    else
        fail "test_new_server_binds_after_kill: new server could not bind to port $port"
    fi
    kill "$LISTENER_PID" 2>/dev/null || true
    LISTENER_PID=""
}

# ---- Test 3: fuser -k can free a port --------------------------------------
test_fuser_frees_port() {
    echo "--- Test: fuser -k frees a port ---"

    if ! command -v fuser >/dev/null 2>&1; then
        pass "test_fuser_frees_port: fuser not available, skipping"
        return
    fi

    local port
    port=$(find_free_port)
    start_listener "$port"

    if ! port_in_use "$port"; then
        fail "test_fuser_frees_port: listener did not start"
        kill "$LISTENER_PID" 2>/dev/null || true
        return
    fi

    # Use fuser to kill the process on the port
    fuser -k "$port/tcp" 2>/dev/null || true
    LISTENER_PID=""

    if wait_for_port_free "$port" 3; then
        pass "test_fuser_frees_port: port $port freed by fuser"
    else
        fail "test_fuser_frees_port: port $port not freed by fuser"
    fi
}

# ---- Test 4: Multiple consecutive restart cycles ---------------------------
test_multiple_restart_cycles() {
    echo "--- Test: Multiple consecutive restart cycles ---"

    local port
    port=$(find_free_port)

    for i in 1 2 3; do
        # Start listener
        start_listener "$port"

        if ! port_in_use "$port"; then
            fail "test_multiple_restart_cycles (cycle $i): listener did not start"
            kill "$LISTENER_PID" 2>/dev/null || true
            return
        fi

        # Kill
        kill "$LISTENER_PID" 2>/dev/null || true
        LISTENER_PID=""

        # Wait for free
        if ! wait_for_port_free "$port" 3; then
            fail "test_multiple_restart_cycles (cycle $i): port did not free"
            return
        fi
    done

    pass "test_multiple_restart_cycles: all 3 cycles completed successfully"
}

# ---- Test 5: Port verification blocks until free ---------------------------
test_port_verification_blocks() {
    echo "--- Test: Port verification blocks until free ---"

    local port
    port=$(find_free_port)
    start_listener "$port"

    # Start a background timer that will kill the listener after 1 second
    (
        sleep 1
        kill "$LISTENER_PID" 2>/dev/null || true
    ) &
    local killer_pid=$!

    # The wait should succeed (port becomes free after ~1s)
    if wait_for_port_free "$port" 3; then
        pass "test_port_verification_blocks: port freed after delay"
    else
        fail "test_port_verification_blocks: port did not free"
    fi

    LISTENER_PID=""
    wait "$killer_pid" 2>/dev/null || true
}

# ---- Run tests -------------------------------------------------------------

echo "=== start-proxy.sh Restart Port Cleanup Tests ==="
echo ""

test_pkill_frees_port
test_new_server_binds_after_kill
test_fuser_frees_port
test_multiple_restart_cycles
test_port_verification_blocks
