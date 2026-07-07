#!/usr/bin/env bash
# Tests for scripts/cleanup-all.sh
# Uses mock/stub child scripts to verify invocation and flag propagation.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
SCRIPT="$SCRIPT_DIR/cleanup-all.sh"
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
# Helper: create mock cleanup scripts that record their invocation
# ---------------------------------------------------------------
make_mock_script() {
    local path="$1"
    local name="$2"
    mkdir -p "$(dirname "$path")"
    cat > "$path" <<MOCK
#!/usr/bin/env bash
# Mock $name - records invocation arguments
# Produces JSON output when --json is passed
for arg in "\$@"; do
    if [ "\$arg" = "--json" ]; then
        echo '{"ok":true,"dry_run":false,"items_removed":0,"freed_bytes":0,"errors":[]}'
        break
    fi
done
echo "\$(basename \$0) \$@" >> "${TESTS_TMPDIR}/invocations.txt"
exit 0
MOCK
    chmod +x "$path"
}

make_failing_mock() {
    local path="$1"
    local name="$2"
    mkdir -p "$(dirname "$path")"
    cat > "$path" <<MOCK
#!/usr/bin/env bash
# Mock $name - always fails
# Produces JSON output when --json is passed
for arg in "\$@"; do
    if [ "\$arg" = "--json" ]; then
        echo '{"ok":false,"dry_run":false,"items_removed":0,"freed_bytes":0,"errors":["mock failure"]}'
        break
    fi
done
echo "\$(basename \$0) \$@" >> "${TESTS_TMPDIR}/invocations.txt"
exit 1
MOCK
    chmod +x "$path"
}

# ---------------------------------------------------------------
# Test: Orchestrator calls all cleanup scripts
# ---------------------------------------------------------------
test_calls_all_scripts() {
    echo "Test: orchestrator calls all cleanup scripts"
    TESTS_TMPDIR="$(mktemp -d)"
    local mock_dir="$TESTS_TMPDIR/scripts"
    mkdir -p "$mock_dir"

    make_mock_script "$mock_dir/cleanup-model-cache.sh" "model-cache"
    make_mock_script "$mock_dir/cleanup-container-images.sh" "container-images"
    make_mock_script "$mock_dir/cleanup-pi-sessions.sh" "pi-sessions"

    # Set PATH so the orchestrator finds our mocks
    local output
    output=$(PATH="$mock_dir:$PATH" CLEANUP_MODEL_CACHE="$mock_dir/cleanup-model-cache.sh" \
             CLEANUP_CONTAINER_IMAGES="$mock_dir/cleanup-container-images.sh" \
             CLEANUP_PI_SESSIONS="$mock_dir/cleanup-pi-sessions.sh" \
             bash "$SCRIPT" 2>&1 || true)

    [ -f "$TESTS_TMPDIR/invocations.txt" ] && pass "invocations recorded" || fail "no invocations recorded"

    local model_count container_count pi_count
    model_count=$(grep -c "cleanup-model-cache.sh" "$TESTS_TMPDIR/invocations.txt" 2>/dev/null || echo 0)
    container_count=$(grep -c "cleanup-container-images.sh" "$TESTS_TMPDIR/invocations.txt" 2>/dev/null || echo 0)
    pi_count=$(grep -c "cleanup-pi-sessions.sh" "$TESTS_TMPDIR/invocations.txt" 2>/dev/null || echo 0)

    [ "$model_count" -ge 1 ] && pass "model-cache script called" || fail "model-cache not called"
    [ "$container_count" -ge 1 ] && pass "container-images script called" || fail "container-images not called"
    [ "$pi_count" -ge 1 ] && pass "pi-sessions script called" || fail "pi-sessions not called"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: --dry-run flag propagates to all child scripts
# ---------------------------------------------------------------
test_dry_run_propagation() {
    echo "Test: --dry-run flag propagates to child scripts"
    TESTS_TMPDIR="$(mktemp -d)"
    local mock_dir="$TESTS_TMPDIR/scripts"
    mkdir -p "$mock_dir"

    make_mock_script "$mock_dir/cleanup-model-cache.sh" "model-cache"
    make_mock_script "$mock_dir/cleanup-container-images.sh" "container-images"
    make_mock_script "$mock_dir/cleanup-pi-sessions.sh" "pi-sessions"

    local output
    output=$(PATH="$mock_dir:$PATH" \
             CLEANUP_MODEL_CACHE="$mock_dir/cleanup-model-cache.sh" \
             CLEANUP_CONTAINER_IMAGES="$mock_dir/cleanup-container-images.sh" \
             CLEANUP_PI_SESSIONS="$mock_dir/cleanup-pi-sessions.sh" \
             bash "$SCRIPT" --dry-run 2>&1 || true)

    local dry_run_count
    dry_run_count=$(grep -c -- "--dry-run" "$TESTS_TMPDIR/invocations.txt" 2>/dev/null || echo 0)
    [ "$dry_run_count" -ge 3 ] && pass "--dry-run propagated to all 3 scripts (found $dry_run_count)" || fail "--dry-run not propagated to all scripts (found $dry_run_count)"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Orchestrator exits non-zero if a child script fails
# ---------------------------------------------------------------
test_child_failure() {
    echo "Test: exits non-zero if child script fails"
    TESTS_TMPDIR="$(mktemp -d)"
    local mock_dir="$TESTS_TMPDIR/scripts"
    mkdir -p "$mock_dir"

    make_mock_script "$mock_dir/cleanup-model-cache.sh" "model-cache"
    make_failing_mock "$mock_dir/cleanup-container-images.sh" "container-images"
    make_mock_script "$mock_dir/cleanup-pi-sessions.sh" "pi-sessions"

    local rc=0
    PATH="$mock_dir:$PATH" \
    CLEANUP_MODEL_CACHE="$mock_dir/cleanup-model-cache.sh" \
    CLEANUP_CONTAINER_IMAGES="$mock_dir/cleanup-container-images.sh" \
    CLEANUP_PI_SESSIONS="$mock_dir/cleanup-pi-sessions.sh" \
    bash "$SCRIPT" 2>&1 || rc=$?

    [ $rc -ne 0 ] && pass "exits non-zero ($rc) on child failure" || fail "should exit non-zero on child failure"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Orchestrator is idempotent
# ---------------------------------------------------------------
test_idempotent() {
    echo "Test: idempotent -- second run produces same result"
    TESTS_TMPDIR="$(mktemp -d)"
    local mock_dir="$TESTS_TMPDIR/scripts"
    mkdir -p "$mock_dir"

    make_mock_script "$mock_dir/cleanup-model-cache.sh" "model-cache"
    make_mock_script "$mock_dir/cleanup-container-images.sh" "container-images"
    make_mock_script "$mock_dir/cleanup-pi-sessions.sh" "pi-sessions"

    local output1 output2
    output1=$(PATH="$mock_dir:$PATH" \
              CLEANUP_MODEL_CACHE="$mock_dir/cleanup-model-cache.sh" \
              CLEANUP_CONTAINER_IMAGES="$mock_dir/cleanup-container-images.sh" \
              CLEANUP_PI_SESSIONS="$mock_dir/cleanup-pi-sessions.sh" \
              bash "$SCRIPT" --dry-run 2>&1 || true)
    local rc1=$?

    output2=$(PATH="$mock_dir:$PATH" \
              CLEANUP_MODEL_CACHE="$mock_dir/cleanup-model-cache.sh" \
              CLEANUP_CONTAINER_IMAGES="$mock_dir/cleanup-container-images.sh" \
              CLEANUP_PI_SESSIONS="$mock_dir/cleanup-pi-sessions.sh" \
              bash "$SCRIPT" --dry-run 2>&1 || true)
    local rc2=$?

    [ "$rc1" = "$rc2" ] && pass "same exit code on both runs ($rc1)" || fail "exit code differs: $rc1 vs $rc2"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Orchestrator --dry-run --json produces valid JSON
# ---------------------------------------------------------------
test_dry_run_json() {
    echo "Test: --dry-run --json produces valid JSON"
    TESTS_TMPDIR="$(mktemp -d)"
    local mock_dir="$TESTS_TMPDIR/scripts"
    mkdir -p "$mock_dir"

    make_mock_script "$mock_dir/cleanup-model-cache.sh" "model-cache"
    make_mock_script "$mock_dir/cleanup-container-images.sh" "container-images"
    make_mock_script "$mock_dir/cleanup-pi-sessions.sh" "pi-sessions"

    local output
    output=$(PATH="$mock_dir:$PATH" \
             CLEANUP_MODEL_CACHE="$mock_dir/cleanup-model-cache.sh" \
             CLEANUP_CONTAINER_IMAGES="$mock_dir/cleanup-container-images.sh" \
             CLEANUP_PI_SESSIONS="$mock_dir/cleanup-pi-sessions.sh" \
             bash "$SCRIPT" --dry-run --json 2>&1 || true)

    echo "$output" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert "ok" in d or "dry_run" in d' && pass "output is valid JSON" || fail "output is not valid JSON"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
echo "========================================="
echo "cleanup-all.sh tests"
echo "========================================="

test_calls_all_scripts
test_dry_run_propagation
test_child_failure
test_idempotent
test_dry_run_json

echo "========================================="
echo "Results: $PASS passed, $FAIL failed"
echo "========================================="

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
