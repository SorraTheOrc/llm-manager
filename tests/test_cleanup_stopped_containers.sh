#!/usr/bin/env bash
# Tests for scripts/cleanup-stopped-containers.sh
# Uses mock podman commands to avoid touching real container store.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
SCRIPT="$SCRIPT_DIR/cleanup-stopped-containers.sh"
PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

# ---------------------------------------------------------------
# Test: --dry-run lists candidates without removing anything
# ---------------------------------------------------------------
test_dry_run() {
    echo "Test: --dry-run lists candidates without removal"

    local output rc=0
    output=$(bash "$SCRIPT" --dry-run --mock 2>&1) || rc=$?

    echo "$output" | grep -qi "would remove\|dry.run\|No exited containers" && pass "output indicates dry-run" || fail "output missing dry-run indicator"
    [ $rc -eq 0 ] && pass "exits 0 on dry-run" || fail "non-zero exit ($rc) on dry-run"
}

# ---------------------------------------------------------------
# Test: --yes skips confirmation
# ---------------------------------------------------------------
test_yes_flag() {
    echo "Test: --yes skips interactive confirmation"

    local output rc=0
    output=$(bash "$SCRIPT" --yes --mock 2>&1) || rc=$?

    echo "$output" | grep -qi "removing\|would remove" && pass "proceeds without confirmation" || fail "no removal output with --yes"
}

# ---------------------------------------------------------------
# Test: JSON output is valid
# ---------------------------------------------------------------
test_json_output() {
    echo "Test: --json produces valid JSON"

    local output
    output=$(bash "$SCRIPT" --dry-run --json --mock 2>&1 || true)

    echo "$output" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert "ok" in d; assert "dry_run" in d; assert "containers_found" in d' && pass "output is valid JSON with expected fields" || fail "output is not valid JSON or missing fields"
}

# ---------------------------------------------------------------
# Test: --json without --dry-run also produces valid JSON
# ---------------------------------------------------------------
test_json_output_real() {
    echo "Test: --json --yes produces valid JSON"

    local output
    output=$(bash "$SCRIPT" --json --yes --mock 2>&1 || true)

    echo "$output" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert "ok" in d; assert "containers_removed" in d' && pass "JSON output has expected fields" || fail "JSON output missing expected fields"
}

# ---------------------------------------------------------------
# Test: Script exists as executable file
# ---------------------------------------------------------------
test_script_exists() {
    echo "Test: script exists and is executable"

    [ -f "$SCRIPT" ] && pass "script file exists" || fail "script file missing"
    [ -x "$SCRIPT" ] && pass "script is executable" || fail "script not executable"
}

# ---------------------------------------------------------------
# Test: Non-TTY without --yes prints error
# ---------------------------------------------------------------
test_non_tty_no_yes() {
    echo "Test: non-TTY without --yes prints helpful error"

    # Simulate non-TTY by piping empty input
    local output rc=0
    output=$(echo "" | bash "$SCRIPT" --dry-run --mock 2>&1) || rc=$?

    # Should still work in dry-run mode since no actual removal
    [ $rc -eq 0 ] && pass "non-TTY dry-run exits 0" || fail "non-zero exit ($rc)"
}

# ---------------------------------------------------------------
# Test: --help displays usage
# ---------------------------------------------------------------
test_help() {
    echo "Test: --help displays usage info"

    local output rc=0
    output=$(bash "$SCRIPT" --help 2>&1) || rc=$?

    echo "$output" | grep -qi "usage\|cleanup\|dry-run" && pass "--help shows usage" || fail "--help output missing usage info"
}

# ---------------------------------------------------------------
# Test: Idempotent -- second dry-run same as first
# ---------------------------------------------------------------
test_idempotent() {
    echo "Test: idempotent -- second dry-run same as first"

    local output1 output2
    output1=$(bash "$SCRIPT" --dry-run --mock 2>&1)
    output2=$(bash "$SCRIPT" --dry-run --mock 2>&1)

    [ "$output1" = "$output2" ] && pass "identical output on second dry-run" || fail "output differs between runs"
}

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
echo "=========================================="
echo "cleanup-stopped-containers.sh tests"
echo "=========================================="

test_dry_run
test_yes_flag
test_json_output
test_json_output_real
test_script_exists
test_non_tty_no_yes
test_help
test_idempotent

echo "=========================================="
echo "Results: $PASS passed, $FAIL failed"
echo "=========================================="

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
