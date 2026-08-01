#!/usr/bin/env bash
# Tests for the LIVE_PORT_KILL_TESTS opt-in gate on test_start_proxy_restart.sh.
#
# test_start_proxy_restart.sh spawns real OS processes (python3 TCP listeners)
# and kills them using port-based mechanisms. It MUST NOT run by default — only
# when LIVE_PORT_KILL_TESTS=1 is set explicitly. This test verifies the gate
# (the skip path). The on-demand path is exercised manually by operators.
#
# Usage:
#   bash tests/test_start_proxy_restart_gate.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="$SCRIPT_DIR/test_start_proxy_restart.sh"
PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); echo "PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }

if [ ! -f "$TARGET" ]; then
    fail "test_start_proxy_restart.sh not found at $TARGET"
    echo "=== Results: $PASS passed, $FAIL failed ==="
    exit 1
fi

# --- Test 1: default run (no env var) must be skipped -----------------------
output=$(bash "$TARGET" 2>&1)
rc=$?

if [ "$rc" -ne 0 ]; then
    fail "default run should exit 0 (skip), got exit $rc"
elif echo "$output" | grep -q "SKIP"; then
    pass "default run prints SKIP notice"
else
    fail "default run should print a SKIP notice, got: $output"
fi

if echo "$output" | grep -qE "^PASS:|--- Test:"; then
    fail "default run must NOT execute the port-kill tests"
else
    pass "default run does not execute port-kill tests"
fi

# --- Test 2: empty value also skips ------------------------------------------
output=$(LIVE_PORT_KILL_TESTS=0 bash "$TARGET" 2>&1)
rc=$?

if [ "$rc" -ne 0 ]; then
    fail "LIVE_PORT_KILL_TESTS=0 run should exit 0 (skip), got exit $rc"
elif echo "$output" | grep -q "SKIP"; then
    pass "LIVE_PORT_KILL_TESTS=0 run prints SKIP notice"
else
    fail "LIVE_PORT_KILL_TESTS=0 run should print a SKIP notice, got: $output"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
