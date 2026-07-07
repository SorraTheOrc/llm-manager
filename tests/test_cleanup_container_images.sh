#!/usr/bin/env bash
# Tests for scripts/cleanup-container-images.sh
# Uses mock podman commands to avoid touching real image store.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
SCRIPT="$SCRIPT_DIR/cleanup-container-images.sh"
TESTS_TMPDIR=""
PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

cleanup_tmp() {
    if [ -n "$TESTS_TMPDIR" ] && [ -d "$TESTS_TMPDIR" ]; then
        rm -rf "$TESTS_TMPDIR"
    fi
    TESTS_TMPDIR=""
}

# ---------------------------------------------------------------
# Test: --dry-run lists candidates without removing anything
# ---------------------------------------------------------------
test_dry_run() {
    echo "Test: --dry-run lists candidates without removal"
    TESTS_TMPDIR="$(mktemp -d)"

    # Run with --dry-run using mock podman (no-op)
    local output rc=0
    output=$(bash "$SCRIPT" --dry-run --mock 2>&1) || rc=$?

    echo "$output" | grep -qi "dry.run\|would remove\|would delete" && pass "output indicates dry-run" || fail "output missing dry-run indicator"
    [ $rc -eq 0 ] && pass "exits 0 on dry-run" || fail "non-zero exit ($rc) on dry-run"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Idempotent -- running twice produces same result
# ---------------------------------------------------------------
test_idempotent() {
    echo "Test: idempotent -- second run same as first"
    TESTS_TMPDIR="$(mktemp -d)"

    local output1 output2
    output1=$(bash "$SCRIPT" --dry-run --mock 2>&1)
    output2=$(bash "$SCRIPT" --dry-run --mock 2>&1)

    [ "$output1" = "$output2" ] && pass "identical output on second dry-run" || fail "output differs between runs"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: JSON output is valid
# ---------------------------------------------------------------
test_json_output() {
    echo "Test: --json produces valid JSON"
    TESTS_TMPDIR="$(mktemp -d)"

    local output
    output=$(bash "$SCRIPT" --dry-run --json --mock 2>&1 || true)

    echo "$output" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert "ok" in d or "dry_run" in d' && pass "output is valid JSON" || fail "output is not valid JSON"

    cleanup_tmp
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
# Main
# ---------------------------------------------------------------
echo "=========================================="
echo "cleanup-container-images.sh tests"
echo "=========================================="

test_dry_run
test_idempotent
test_json_output
test_script_exists

echo "=========================================="
echo "Results: $PASS passed, $FAIL failed"
echo "=========================================="

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
