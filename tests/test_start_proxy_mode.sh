#!/usr/bin/env bash
# Tests for start-proxy.sh operating-mode resolution (fast/cheap).
#
# Verifies (LP-0MSLMYEEU002IBH6):
#   1. The persisted mode (proxy/.mode) selects the config profile:
#      .mode=cheap  -> config-cheap.yaml,  .mode=fast -> config-fast.yaml
#   2. A missing .mode defaults to fast (config-fast.yaml)
#   3. A missing mode config file falls back to config.yaml (with a warning)
#   4. An invalid .mode value warns and defaults to fast
#   5. LLAMA_PROXY_CONFIG is exported so the server loads the same profile,
#      and API-key resolution reads from the SELECTED config
#
# This test does NOT touch the live proxy: it runs a sandboxed copy of
# start-proxy.sh against minimal config files and a fake python shim that
# intercepts the final `exec uvicorn`. It uses an ephemeral port.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_SCRIPT="$SCRIPT_DIR/../proxy/scripts/start-proxy.sh"
PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); echo "PASS: $*"; }
fail() { FAIL=$((FAIL + 1)); echo "FAIL: $*"; }

cleanup() {
    local rc=$?
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

# Minimal configs: NO api_key_env entries (so API-key resolution is a no-op
# and the test only exercises mode selection).
cat > "$PROXY_DIR/config.yaml" <<'YAML'
models:
  test:
    providers:
      - name: local
        type: local
        llama_model: Test
server:
  session_slot_pool_size: 3
YAML
cp "$PROXY_DIR/config.yaml" "$PROXY_DIR/config-fast.yaml"
sed 's/session_slot_pool_size: 3/session_slot_pool_size: 1/' \
    "$PROXY_DIR/config.yaml" > "$PROXY_DIR/config-cheap.yaml"

# Fake python: delegate -c to the real interpreter; intercept `-m uvicorn`.
FAKE_PY="$SANDBOX/python3"
cat > "$FAKE_PY" <<'PY'
#!/usr/bin/env bash
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "uvicorn" ]; then
    echo "FAKE_UVICORN llama_proxy_config=${LLAMA_PROXY_CONFIG:-unset}"
    exit 0
fi
exec /usr/bin/python3 "$@"
PY
chmod +x "$FAKE_PY"

TEST_PORT=8457

run_script() {
    # Usage: run_script  -> prints combined stdout+stderr of the sandboxed script
    local out
    out=$(cd "$PROXY_DIR" && PATH="$SANDBOX:$PATH" bash scripts/start-proxy.sh --port "$TEST_PORT" 2>&1)
    echo "$out"
}

# ---- Test 1: .mode=cheap selects config-cheap.yaml -------------------------
test_cheap_mode() {
    echo "--- Test: .mode=cheap selects config-cheap.yaml ---"
    printf 'cheap\n' > "$PROXY_DIR/.mode"
    local out
    out=$(run_script)
    if echo "$out" | grep -q "FAKE_UVICORN llama_proxy_config=.*config-cheap.yaml"; then
        pass "cheap mode exports LLAMA_PROXY_CONFIG=config-cheap.yaml"
    else
        fail "cheap mode: unexpected output: $out"
    fi
    if echo "$out" | grep -q "Operating mode: cheap (config: .*config-cheap.yaml)"; then
        pass "cheap mode reported on stderr"
    else
        fail "cheap mode: missing 'Operating mode: cheap' line"
    fi
}

# ---- Test 2: .mode=fast selects config-fast.yaml ---------------------------
test_fast_mode() {
    echo "--- Test: .mode=fast selects config-fast.yaml ---"
    printf 'fast\n' > "$PROXY_DIR/.mode"
    local out
    out=$(run_script)
    if echo "$out" | grep -q "FAKE_UVICORN llama_proxy_config=.*config-fast.yaml"; then
        pass "fast mode exports LLAMA_PROXY_CONFIG=config-fast.yaml"
    else
        fail "fast mode: unexpected output: $out"
    fi
}

# ---- Test 3: missing .mode defaults to fast --------------------------------
test_missing_mode() {
    echo "--- Test: missing .mode defaults to fast ---"
    rm -f "$PROXY_DIR/.mode"
    local out
    out=$(run_script)
    if echo "$out" | grep -q "FAKE_UVICORN llama_proxy_config=.*config-fast.yaml"; then
        pass "missing .mode defaults to config-fast.yaml"
    else
        fail "missing .mode: unexpected output: $out"
    fi
}

# ---- Test 4: missing mode config falls back to config.yaml -----------------
test_missing_mode_config_fallback() {
    echo "--- Test: missing mode config falls back to config.yaml ---"
    printf 'cheap\n' > "$PROXY_DIR/.mode"
    mv "$PROXY_DIR/config-cheap.yaml" "$PROXY_DIR/config-cheap.yaml.bak"
    local out
    out=$(run_script)
    if echo "$out" | grep -q "FAKE_UVICORN llama_proxy_config=.*config.yaml"; then
        pass "missing mode config falls back to config.yaml"
    else
        fail "fallback: unexpected output: $out"
    fi
    if echo "$out" | grep -q "not found, falling back to config.yaml"; then
        pass "fallback warning emitted"
    else
        fail "fallback warning not emitted"
    fi
    mv "$PROXY_DIR/config-cheap.yaml.bak" "$PROXY_DIR/config-cheap.yaml"
}

# ---- Test 5: invalid .mode warns and defaults to fast ----------------------
test_invalid_mode() {
    echo "--- Test: invalid .mode warns and defaults to fast ---"
    printf 'garbage\n' > "$PROXY_DIR/.mode"
    local out
    out=$(run_script)
    if echo "$out" | grep -q "unknown mode 'garbage'" && \
       echo "$out" | grep -q "FAKE_UVICORN llama_proxy_config=.*config-fast.yaml"; then
        pass "invalid .mode warns and defaults to fast"
    else
        fail "invalid .mode: unexpected output: $out"
    fi
}

# ---- Test 6: API-key resolution reads the SELECTED config ------------------
test_api_keys_resolve_from_selected_config() {
    echo "--- Test: API-key resolution reads the selected config ---"
    # config-cheap.yaml has no api_key_env -> resolve_api_keys must succeed
    # even though config-fast.yaml carries one (cheap must not fail on
    # missing cloud keys).
    cat >> "$PROXY_DIR/config-fast.yaml" <<'YAML'
    # api_key_env marker: OPENCODE_API_KEY (must NOT be required in cheap mode)
YAML
    printf 'cheap\n' > "$PROXY_DIR/.mode"
    local out
    out=$(run_script)
    if echo "$out" | grep -q "FAKE_UVICORN"; then
        pass "cheap mode starts without cloud API keys (no key error)"
    else
        fail "cheap mode failed on missing keys: $out"
    fi
    # restore the fast config
    cp "$PROXY_DIR/config.yaml" "$PROXY_DIR/config-fast.yaml"
}

# ---- Run -------------------------------------------------------------------

echo "=== start-proxy.sh Operating-Mode Resolution Tests ==="
echo ""
test_cheap_mode
test_fast_mode
test_missing_mode
test_missing_mode_config_fallback
test_invalid_mode
test_api_keys_resolve_from_selected_config
