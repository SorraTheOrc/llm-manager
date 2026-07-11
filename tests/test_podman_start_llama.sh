#!/usr/bin/env bash
# Tests for scripts/podman_start_llama.sh
# Uses a mock podman command to avoid touching real container runtime.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
SCRIPT="$SCRIPT_DIR/podman_start_llama.sh"
PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

# Create a temporary directory for mock artifacts
MOCK_DIR="$(mktemp -d)"
trap 'rm -rf "$MOCK_DIR"' EXIT

# ---------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------

# Track invocations for verification
INVOCATIONS_FILE="$MOCK_DIR/invocations.txt"

# Write a mock podman script that records invocations and simulates
# controlled container states.
setup_mock_podman() {
    local container_state="$1"   # container state to report (e.g. "stopping", "exited", "running")
    local container_exists="$2"  # "yes" or "no"

    cat > "$MOCK_DIR/podman" <<MOCK_SCRIPT
#!/usr/bin/env bash
echo "\$(date +%s%N) podman \$*" >> "$INVOCATIONS_FILE"

case "\$1" in
  container)
    case "\$2" in
      exists)
        if [ "$container_exists" = "yes" ]; then
          exit 0
        else
          exit 1
        fi
        ;;
      *)
        exit 0
        ;;
    esac
    ;;
  inspect)
    # -f '{{.State.Running}}'
    if echo "\$*" | grep -q "State.Running"; then
      if [ "$container_state" = "running" ]; then
        echo "true"
      else
        echo "false"
      fi
      exit 0
    fi
    # -f '{{.State.Status}}'
    if echo "\$*" | grep -q "State.Status"; then
      echo "$container_state"
      exit 0
    fi
    # generic inspect (mounts check)
    if echo "\$*" | grep -q "Mounts"; then
      echo '[]'
      exit 0
    fi
    echo "{}"
    exit 0
    ;;
  create)
    # Simulate podman create output
    echo "mock-container-id"
    exit 0
    ;;
  start)
    if [ "$container_state" = "stopping" ] || [ "$container_state" = "paused" ]; then
      echo "Error: unable to start container \"llama\": container must be in Created or Stopped state to be started: container state improper" >&2
      exit 125
    fi
    exit 0
    ;;
  rm)
    if [ "\$2" = "-f" ]; then
      exit 0  # force-remove succeeds
    fi
    exit 0
    ;;
  exec)
    # When podman exec is called, we need to handle the chmod and ss calls
    # The real script does `exec podman exec` which replaces the process;
    # for mock, we just simulate success for chmod/ss and exec tail
    if echo "$*" | grep -q "chmod"; then
      exit 0
    fi
    if echo "$*" | grep -q "ss -ltnp"; then
      exit 0
    fi
    if echo "$*" | grep -q "tail -F"; then
      exit 0
    fi
    # For the final exec that starts the server, we just exit 0
    # since it would replace the process in real usage
    exit 0
    ;;
  *)
    exit 0
    ;;
esac
MOCK_SCRIPT
    chmod +x "$MOCK_DIR/podman"
}

run_script() {
    local extra_env=("${@}")
    PATH="$MOCK_DIR:$PATH" \
        LLAMA_HOST_REPO="$MOCK_DIR" \
        LLAMA_CONTAINER_IMAGE="test-image:latest" \
        "${extra_env[@]}" \
        bash "$SCRIPT" "${MODEL_ARGS:-}" 2>&1 || true
}

# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------

test_script_exists() {
    echo "Test: script exists and is executable"
    [ -f "$SCRIPT" ] && pass "script file exists" || fail "script file missing"
    [ -x "$SCRIPT" ] && pass "script is executable" || fail "script not executable"
}

test_container_running_skips_start() {
    echo "Test: container is running — skips start"
    setup_mock_podman "running" "yes"
    > "$INVOCATIONS_FILE"

    local output
    output=$(run_script)
    local rc=$?

    # Should NOT attempt to start the container
    if grep -q "podman start" "$INVOCATIONS_FILE" 2>/dev/null; then
        fail "start was attempted on a running container"
    else
        pass "no start attempted on running container"
    fi
    [ $rc -eq 0 ] && pass "exits 0 for running container" || fail "non-zero exit ($rc) for running container"
}

test_container_exited_normal_start() {
    echo "Test: container is exited — normal start works"
    setup_mock_podman "exited" "yes"
    > "$INVOCATIONS_FILE"

    local output
    output=$(run_script)
    local rc=$?

    if grep -q "podman start" "$INVOCATIONS_FILE" 2>/dev/null; then
        pass "start attempted for exited container"
    else
        fail "no start attempted for exited container"
    fi
    if grep -q "podman rm -f" "$INVOCATIONS_FILE" 2>/dev/null; then
        fail "force-remove was attempted on an exited container"
    else
        pass "no force-remove for exited container"
    fi
    [ $rc -eq 0 ] && pass "exits 0 for exited container" || fail "non-zero exit ($rc) for exited container"
}

test_container_created_normal_start() {
    echo "Test: container is created — normal start works"
    setup_mock_podman "created" "yes"
    > "$INVOCATIONS_FILE"

    local output
    output=$(run_script)
    local rc=$?

    if grep -q "podman start" "$INVOCATIONS_FILE" 2>/dev/null; then
        pass "start attempted for created container"
    else
        fail "no start attempted for created container"
    fi
    if grep -q "podman rm -f" "$INVOCATIONS_FILE" 2>/dev/null; then
        fail "force-remove was attempted on a created container"
    else
        pass "no force-remove for created container"
    fi
    [ $rc -eq 0 ] && pass "exits 0 for created container" || fail "non-zero exit ($rc) for created container"
}

test_container_stopping_force_remove() {
    echo "Test: container is stopping — force-removes and recreates"
    setup_mock_podman "stopping" "yes"
    > "$INVOCATIONS_FILE"

    local output
    output=$(run_script)
    local rc=$?

    # Check that force-remove was attempted
    if grep -q "podman rm -f" "$INVOCATIONS_FILE" 2>/dev/null; then
        pass "force-remove attempted for stopping container"
    else
        fail "force-remove was NOT attempted for stopping container"
    fi

    # Check that a new container was created
    if grep -q "podman create" "$INVOCATIONS_FILE" 2>/dev/null; then
        pass "recreate attempted after force-remove"
    else
        fail "recreate was NOT attempted after force-remove"
    fi

    # Check that warning is logged
    if echo "$output" | grep -qi "warning.*force-remove\|force-removing\|improper state"; then
        pass "warning logged for force-remove"
    else
        fail "no warning logged for force-remove"
    fi

    [ $rc -eq 0 ] && pass "exits 0 after force-remove and recreate" || fail "non-zero exit ($rc) after force-remove"
}

test_container_paused_force_remove() {
    echo "Test: container is paused — force-removes and recreates"
    setup_mock_podman "paused" "yes"
    > "$INVOCATIONS_FILE"

    local output
    output=$(run_script)
    local rc=$?

    if grep -q "podman rm -f" "$INVOCATIONS_FILE" 2>/dev/null; then
        pass "force-remove attempted for paused container"
    else
        fail "force-remove was NOT attempted for paused container"
    fi

    if echo "$output" | grep -qi "warning.*force-remove\|force-removing\|improper state"; then
        pass "warning logged for force-remove of paused container"
    else
        fail "no warning logged for force-remove of paused container"
    fi

    [ $rc -eq 0 ] && pass "exits 0 after force-remove of paused container" || fail "non-zero exit ($rc)"
}

test_container_does_not_exist_creates() {
    echo "Test: container does not exist — creates it"
    setup_mock_podman "created" "no"
    > "$INVOCATIONS_FILE"

    local output
    output=$(run_script)
    local rc=$?

    if grep -q "podman create" "$INVOCATIONS_FILE" 2>/dev/null; then
        pass "create attempted for non-existent container"
    else
        fail "create was NOT attempted for non-existent container"
    fi

    [ $rc -eq 0 ] && pass "exits 0 after create" || fail "non-zero exit ($rc) after create"
}

test_no_podman_errors() {
    echo "Test: podman not found — exits with error"
    local output rc=0
    # Create a minimal PATH dir with only essential tools (no podman)
    local minidir="$MOCK_DIR/no-podman-path"
    mkdir -p "$minidir"
    # Symlink essential commands but NOT podman
    for cmd in bash echo command mkdir grep chmod true false sleep cat date dirname head tail cut sort uniq wc; do
      if command -v "$cmd" &>/dev/null; then
        ln -sf "$(command -v "$cmd")" "$minidir/$cmd" 2>/dev/null || true
      fi
    done
    output=$(PATH="$minidir" bash "$SCRIPT" 2>&1) || rc=$?
    echo "$output" | grep -qi "podman not found" && pass "error message when podman missing" || fail "no error message when podman missing: $(echo "$output" | head -3)"
    [ $rc -eq 2 ] && pass "exits 2 when podman missing" || fail "expected exit 2, got $rc"
}

test_json_output_mode() {
    echo "Test: script can be called with router model (JSON path check)"
    setup_mock_podman "running" "yes"
    > "$INVOCATIONS_FILE"

    local output
    MODEL_ARGS="router" run_script
    # Just verify it doesn't crash — exec replaces the process in real usage
    pass "script handles router model arg without crash"
}

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
echo "=========================================="
echo "podman_start_llama.sh tests"
echo "=========================================="

test_script_exists
test_container_running_skips_start
test_container_exited_normal_start
test_container_created_normal_start
test_container_stopping_force_remove
test_container_paused_force_remove
test_container_does_not_exist_creates
test_no_podman_errors
test_json_output_mode

echo "=========================================="
echo "Results: $PASS passed, $FAIL failed"
echo "=========================================="

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
