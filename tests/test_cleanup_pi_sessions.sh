#!/usr/bin/env bash
# Tests for scripts/cleanup-pi-sessions.sh
# Uses temporary directories to avoid touching real pi session logs.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
SCRIPT="$SCRIPT_DIR/cleanup-pi-sessions.sh"
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

# Create a session file with a given timestamp
make_session() {
    local dir="$1"
    local timestamp="$2"
    local uuid="${3:-00000000-0000-0000-0000-000000000000}"
    local content="${4:-}"
    mkdir -p "$dir"
    local filename="${timestamp}Z_${uuid}.jsonl"
    echo "$content" > "$dir/$filename"
    # Touch the file with the exact timestamp (fallback if mtime differs)
    touch -t "$(echo "$timestamp" | sed 's/[^0-9]//g' | cut -c1-12)" "$dir/$filename" 2>/dev/null || true
}

# ---------------------------------------------------------------
# Test: --dry-run lists candidates without deleting anything
# ---------------------------------------------------------------
test_dry_run() {
    echo "Test: --dry-run lists candidates without deletion"
    TESTS_TMPDIR="$(mktemp -d)"
    local sessions="$TESTS_TMPDIR/sessions"
    local ws="$sessions/--test-project--"
    mkdir -p "$ws"

    # Create 3 old sessions (over 90 days) and 1 recent session
    make_session "$ws" "2026-01-15T10-00-00" "1111" "old session 1"
    make_session "$ws" "2026-02-20T10-00-00" "2222" "old session 2"
    make_session "$ws" "2026-03-10T10-00-00" "3333" "old session 3"
    # Recent session - within 90 days of "now"
    make_session "$ws" "2026-07-01T10-00-00" "4444" "recent session"

    local output
    output=$(PI_SESSIONS_DIR="$sessions" bash "$SCRIPT" --dry-run --mock-now "2026-07-07" 2>&1 || true)

    # Check dry-run doesn't delete anything
    [ -f "$ws/2026-07-01T10-00-00Z_4444.jsonl" ] && pass "recent session not deleted (dry-run)" || fail "recent session deleted"
    [ -f "$ws/2026-01-15T10-00-00Z_1111.jsonl" ] && pass "old session not deleted (dry-run)" || fail "old session deleted"

    # Check output lists candidates
    echo "$output" | grep -q "dry.run\|would remove\|would delete\|Candidates for removal" && pass "output indicates dry-run" || fail "output missing dry-run indicator"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Retains logs from the last 90 days
# ---------------------------------------------------------------
test_retention_days() {
    echo "Test: retains logs from the last 90 days"
    TESTS_TMPDIR="$(mktemp -d)"
    local sessions="$TESTS_TMPDIR/sessions"
    local ws="$sessions/--test-project--"
    mkdir -p "$ws"

    # Create sessions at various ages
    make_session "$ws" "2026-04-15T10-00-00" "1111" "just inside 90 days"
    make_session "$ws" "2026-06-15T10-00-00" "2222" "recent"
    make_session "$ws" "2026-01-01T10-00-00" "3333" "old (outside 90 days)"
    make_session "$ws" "2025-12-01T10-00-00" "4444" "very old"

    local output
    output=$(PI_SESSIONS_DIR="$sessions" bash "$SCRIPT" --mock-now "2026-07-07" 2>&1 || true)
    local rc=$?

    # After cleanup, old sessions should be gone
    [ ! -f "$ws/2026-01-01T10-00-00Z_3333.jsonl" ] && pass "removed old session (outside 90 days)" || fail "old session not removed"
    [ ! -f "$ws/2025-12-01T10-00-00Z_4444.jsonl" ] && pass "removed very old session" || fail "very old session not removed"
    [ -f "$ws/2026-04-15T10-00-00Z_1111.jsonl" ] && pass "kept session at 83 days old (within 90 day window)" || fail "borderline session removed"
    [ -f "$ws/2026-06-15T10-00-00Z_2222.jsonl" ] && pass "kept recent session" || fail "recent session removed"
    [ $rc -eq 0 ] && pass "exits 0 on success" || fail "non-zero exit ($rc)"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Retains the last 50 sessions per project workspace
# ---------------------------------------------------------------
test_keep_last_50() {
    echo "Test: retains last 50 sessions per workspace"
    TESTS_TMPDIR="$(mktemp -d)"
    local sessions="$TESTS_TMPDIR/sessions"
    local ws="$sessions/--test-project--"
    mkdir -p "$ws"

    # Create 60 sessions (all within 90 days to bypass age filter)
    for i in $(seq 1 60); do
        local day=$(printf "%02d" $(( (i % 28) + 1 )))
        local hour=$(printf "%02d" $(( (i % 24) )))
        make_session "$ws" "2026-06-${day}T${hour}-00-00" "uuid-$(printf '%04d' $i)" "session $i"
    done

    local output
    output=$(PI_SESSIONS_DIR="$sessions" bash "$SCRIPT" --keep-sessions 50 --mock-now "2026-07-07" 2>&1 || true)

    # Count remaining sessions
    local remaining
    remaining=$(ls "$ws"/*.jsonl 2>/dev/null | wc -l)
    [ "$remaining" -le 50 ] && pass "at most 50 sessions remain (got $remaining)" || fail "too many sessions remain ($remaining)"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Handles workspaces with fewer than 50 sessions
# ---------------------------------------------------------------
test_fewer_than_50() {
    echo "Test: handles workspaces with fewer than 50 sessions"
    TESTS_TMPDIR="$(mktemp -d)"
    local sessions="$TESTS_TMPDIR/sessions"
    local ws="$sessions/--test-project--"
    mkdir -p "$ws"

    # Create only 10 sessions
    for i in $(seq 1 10); do
        make_session "$ws" "2026-06-${i}T10-00-00" "uuid-$(printf '%04d' $i)" "session $i"
    done

    local output
    output=$(PI_SESSIONS_DIR="$sessions" bash "$SCRIPT" --keep-sessions 50 --mock-now "2026-07-07" 2>&1 || true)

    local remaining
    remaining=$(ls "$ws"/*.jsonl 2>/dev/null | wc -l)
    [ "$remaining" -eq 10 ] && pass "all 10 sessions retained (fewer than cap)" || fail "expected 10, got $remaining"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Idempotent -- running twice produces identical result
# ---------------------------------------------------------------
test_idempotent() {
    echo "Test: idempotent -- second run same as first"
    TESTS_TMPDIR="$(mktemp -d)"
    local sessions="$TESTS_TMPDIR/sessions"
    local ws="$sessions/--test-project--"
    mkdir -p "$ws"

    # Create mixed sessions
    make_session "$ws" "2026-01-01T10-00-00" "1111" "old"
    make_session "$ws" "2026-06-15T10-00-00" "2222" "recent"
    make_session "$ws" "2026-02-01T10-00-00" "3333" "old"
    make_session "$ws" "2026-07-01T10-00-00" "4444" "very recent"

    # First run
    PI_SESSIONS_DIR="$sessions" bash "$SCRIPT" --mock-now "2026-07-07" 2>/dev/null || true
    local after_first
    after_first=$(ls "$ws"/*.jsonl 2>/dev/null | wc -l)

    # Second run
    PI_SESSIONS_DIR="$sessions" bash "$SCRIPT" --mock-now "2026-07-07" 2>/dev/null || true
    local after_second
    after_second=$(ls "$ws"/*.jsonl 2>/dev/null | wc -l)

    [ "$after_first" = "$after_second" ] && pass "identical session count after both runs ($after_first)" || fail "count differs: $after_first vs $after_second"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Handles missing session log directories gracefully
# ---------------------------------------------------------------
test_missing_directory() {
    echo "Test: handles missing session log directories gracefully"
    TESTS_TMPDIR="$(mktemp -d)"
    local sessions="$TESTS_TMPDIR/nonexistent"

    local output rc=0
    output=$(PI_SESSIONS_DIR="$sessions" bash "$SCRIPT" --mock-now "2026-07-07" 2>&1) || rc=$?

    [ $rc -eq 0 ] && pass "exits 0 with missing directory" || fail "non-zero exit ($rc) with missing directory"
    echo "$output" | grep -qi "does not exist\|not found\|no such\|nothing to clean" && pass "logs warning about missing directory" || fail "no warning for missing directory"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Handles empty session log directories gracefully
# ---------------------------------------------------------------
test_empty_directory() {
    echo "Test: handles empty session log directories gracefully"
    TESTS_TMPDIR="$(mktemp -d)"
    local sessions="$TESTS_TMPDIR/sessions"
    mkdir -p "$sessions/--empty-project--"

    local output rc=0
    output=$(PI_SESSIONS_DIR="$sessions" bash "$SCRIPT" --mock-now "2026-07-07" 2>&1) || rc=$?

    [ $rc -eq 0 ] && pass "exits 0 with empty directory" || fail "non-zero exit ($rc) with empty directory"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: --dry-run --json produces valid JSON
# ---------------------------------------------------------------
test_dry_run_json() {
    echo "Test: --dry-run --json produces valid JSON"
    TESTS_TMPDIR="$(mktemp -d)"
    local sessions="$TESTS_TMPDIR/sessions"
    local ws="$sessions/--test-project--"
    mkdir -p "$ws"

    make_session "$ws" "2026-01-01T10-00-00" "1111" "old"
    make_session "$ws" "2026-07-01T10-00-00" "2222" "recent"

    local output
    output=$(PI_SESSIONS_DIR="$sessions" bash "$SCRIPT" --dry-run --json --mock-now "2026-07-07" 2>&1 || true)

    echo "$output" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert "ok" in d or "dry_run" in d' && pass "output is valid JSON" || fail "output is not valid JSON"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Custom retention-days flag works
# ---------------------------------------------------------------
test_custom_retention_days() {
    echo "Test: custom --retention-days flag works"
    TESTS_TMPDIR="$(mktemp -d)"
    local sessions="$TESTS_TMPDIR/sessions"
    local ws="$sessions/--test-project--"
    mkdir -p "$ws"

    # Create sessions at various ages
    make_session "$ws" "2026-06-01T10-00-00" "1111" "36 days ago"
    make_session "$ws" "2026-05-01T10-00-00" "2222" "67 days ago"
    make_session "$ws" "2026-03-01T10-00-00" "3333" "128 days ago"

    # With retention-days=30, only sessions in last 30 days should be kept
    # 2026-06-01 is 36 days from 2026-07-07 => outside 30 day window
    local output
    output=$(PI_SESSIONS_DIR="$sessions" bash "$SCRIPT" --retention-days 30 --mock-now "2026-07-07" 2>&1 || true)

    [ ! -f "$ws/2026-06-01T10-00-00Z_1111.jsonl" ] && pass "session at 36 days removed (30 day window)" || fail "session at 36 days not removed"
    [ ! -f "$ws/2026-05-01T10-00-00Z_2222.jsonl" ] && pass "session at 67 days removed" || fail "session at 67 days not removed"
    [ ! -f "$ws/2026-03-01T10-00-00Z_3333.jsonl" ] && pass "session at 128 days removed" || fail "session at 128 days not removed"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Custom PI_SESSIONS_DIR env var works
# ---------------------------------------------------------------
test_custom_sessions_dir() {
    echo "Test: custom PI_SESSIONS_DIR env var works"
    TESTS_TMPDIR="$(mktemp -d)"
    local sessions="$TESTS_TMPDIR/custom-sessions"
    local ws="$sessions/--custom-project--"
    mkdir -p "$ws"

    make_session "$ws" "2026-01-01T10-00-00" "1111" "old"
    make_session "$ws" "2026-07-01T10-00-00" "2222" "recent"

    local output
    output=$(PI_SESSIONS_DIR="$sessions" bash "$SCRIPT" --mock-now "2026-07-07" 2>&1 || true)

    [ ! -f "$ws/2026-01-01T10-00-00Z_1111.jsonl" ] && pass "old session removed from custom dir" || fail "old session not removed from custom dir"
    [ -f "$ws/2026-07-01T10-00-00Z_2222.jsonl" ] && pass "recent session kept in custom dir" || fail "recent session not kept in custom dir"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
echo "========================================="
echo "cleanup-pi-sessions.sh tests"
echo "========================================="

test_dry_run
test_retention_days
test_keep_last_50
test_fewer_than_50
test_idempotent
test_missing_directory
test_empty_directory
test_dry_run_json
test_custom_retention_days
test_custom_sessions_dir

echo "========================================="
echo "Results: $PASS passed, $FAIL failed"
echo "========================================="

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
