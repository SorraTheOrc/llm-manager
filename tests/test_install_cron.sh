#!/usr/bin/env bash
# Tests for scripts/install-cron.sh
# Verifies the default log path is user-writable and cron entries
# reference it (LP-0MSK99IO2004EE9T: /var/log/pi-cleanup.log was not
# writable, so cleanup cron jobs silently never ran).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
SCRIPT="$SCRIPT_DIR/install-cron.sh"
PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

# ---------------------------------------------------------------
# Test: default CLEANUP_LOG resolves to a user-writable path
# ---------------------------------------------------------------
test_default_log_path_is_writable() {
    echo "Test: default CLEANUP_LOG resolves to user-writable path"

    local fake_home
    fake_home="$(mktemp -d)"
    local output
    output=$(HOME="$fake_home" bash "$SCRIPT" --dry-run 2>&1)

    if echo "$output" | grep -q "Log file: ${fake_home}/logs/pi-cleanup.log"; then
        pass "dry-run reports log file under \$HOME/logs"
    else
        fail "dry-run log file is not under \$HOME/logs (got: $(echo "$output" | grep 'Log file:'))"
    fi

    if echo "$output" | grep -q ">> ${fake_home}/logs/pi-cleanup.log 2>&1"; then
        pass "cron entries redirect to the writable path"
    else
        fail "cron entries do not redirect to the writable path"
    fi

    if echo "$output" | grep -q "/var/log/pi-cleanup.log"; then
        fail "cron entries still reference /var/log/pi-cleanup.log"
    else
        pass "no hardcoded /var/log/pi-cleanup.log remains"
    fi

    if [ -d "$fake_home/logs" ]; then
        pass "log directory was created under \$HOME"
    else
        fail "log directory was not created under \$HOME"
    fi

    rm -rf "$fake_home"
}

# ---------------------------------------------------------------
# Test: explicit CLEANUP_LOG override is respected
# ---------------------------------------------------------------
test_explicit_log_override() {
    echo "Test: explicit CLEANUP_LOG override is respected"

    local tmp
    tmp="$(mktemp -d)"
    local log_path="$tmp/custom/cleanup.log"
    local output
    output=$(CLEANUP_LOG="$log_path" bash "$SCRIPT" --dry-run 2>&1)

    if echo "$output" | grep -q "Log file: $log_path"; then
        pass "dry-run reports the overridden log path"
    else
        fail "dry-run did not report the overridden log path"
    fi

    if echo "$output" | grep -q ">> $log_path 2>&1"; then
        pass "cron entries redirect to the overridden path"
    else
        fail "cron entries do not redirect to the overridden path"
    fi

    if [ -d "$tmp/custom" ]; then
        pass "override log directory was created"
    else
        fail "override log directory was not created"
    fi

    rm -rf "$tmp"
}

# ---------------------------------------------------------------
test_default_log_path_is_writable
test_explicit_log_override

echo ""
echo "Results: $PASS passed, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
