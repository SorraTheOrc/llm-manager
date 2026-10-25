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
#      and API-key resolution reads from the MERGED config (base + overlay)
#
# Fixture fidelity (LP-0MSX2FMN5006HYN5): the sandbox cheap/fast configs are
# built from the REAL config-cheap.yaml / config-fast.yaml at test time (slot
# count, per-period ctx_size, contention policy, cold-cache threshold) — no
# sed approximations — and the fake python shim reports the effective per-slot
# context using the same formula as provider.py
# (effective_per_slot_threshold: ctx_size // slots - 4096 headroom). The Web
# UI slots section text is also verified against the real cheap slot count.
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

# ---- Fixtures drawn from the real configs ---------------------------------
# Extract the mode-relevant values from the actual config-cheap.yaml /
# config-fast.yaml / config.yaml at test time so drift in the deployed
# profiles fails the test instead of passing with fabricated numbers.
REAL_CONFIG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/proxy"

# The sandboxed start-proxy.sh imports the dependency-light merge helper from
# its repo root (proxy/proxy/config_merge.py); mirror it so the merged-config
# API-key path is actually exercised rather than silently skipped.
mkdir -p "$PROXY_DIR/proxy"
cp "$REAL_CONFIG_DIR/proxy/config_merge.py" "$PROXY_DIR/proxy/config_merge.py"

# Effective server-level value from the MERGED profile (base config.yaml +
# the mode overlay). Mode files are overlays that no longer repeat inherited
# values, so a raw grep would miss base-declared keys such as
# local_model_ctx_size / session_slot_pool_size.
server_val() {
    /usr/bin/python3 -c "
import sys
sys.path.insert(0, '$REAL_CONFIG_DIR/proxy')
from config_merge import load_merged
cfg = load_merged('$REAL_CONFIG_DIR/config.yaml', '$1')
print(cfg.get('server', {}).get('$2', ''))
"
}

CHEAP_SLOTS="$(server_val "$REAL_CONFIG_DIR/config-cheap.yaml" session_slot_pool_size)"
CHEAP_POLICY="$(server_val "$REAL_CONFIG_DIR/config-cheap.yaml" contention_queue_policy)"
CHEAP_COLD="$(server_val "$REAL_CONFIG_DIR/config-cheap.yaml" local_large_context_cold_cache_threshold)"
CHEAP_LOCAL_CTX="$(server_val "$REAL_CONFIG_DIR/config-cheap.yaml" local_model_ctx_size)"

FAST_SLOTS="$(server_val "$REAL_CONFIG_DIR/config-fast.yaml" session_slot_pool_size)"
FAST_POLICY="$(server_val "$REAL_CONFIG_DIR/config-fast.yaml" contention_queue_policy)"
FAST_COLD="$(server_val "$REAL_CONFIG_DIR/config-fast.yaml" local_large_context_cold_cache_threshold)"
FAST_LOCAL_CTX="$(server_val "$REAL_CONFIG_DIR/config-fast.yaml" local_model_ctx_size)"

BASE_SLOTS="$(server_val "$REAL_CONFIG_DIR/config.yaml" session_slot_pool_size)"

# Sanity: the extraction must have found real values (guards against a
# refactored YAML layout silently dropping the fixture fidelity).
if [ -z "$CHEAP_SLOTS" ] || [ -z "$CHEAP_LOCAL_CTX" ] || [ -z "$FAST_SLOTS" ] || [ -z "$FAST_LOCAL_CTX" ]; then
    fail "could not extract mode-relevant values from real configs (cheap slots=$CHEAP_SLOTS ctx=$CHEAP_LOCAL_CTX; fast slots=$FAST_SLOTS ctx=$FAST_LOCAL_CTX)"
    exit 1
fi

# Minimal configs: NO api_key_env entries (so API-key resolution is a no-op
# and the test only exercises mode selection), but they carry the FULL
# mode-relevant delta of the real profiles.
cat > "$PROXY_DIR/config.yaml" <<YAML
models:
  test:
    providers:
      - name: local
        type: local
        llama_model: Test
server:
  session_slot_pool_size: $BASE_SLOTS
YAML

cat > "$PROXY_DIR/config-fast.yaml" <<YAML
models:
  test:
    providers:
      - name: local
        type: local
        llama_model: Test
server:
  session_slot_pool_size: $FAST_SLOTS
  contention_queue_policy: $FAST_POLICY
  local_large_context_cold_cache_threshold: $FAST_COLD
  local_model_ctx_size: $FAST_LOCAL_CTX
YAML

cat > "$PROXY_DIR/config-cheap.yaml" <<YAML
models:
  test:
    providers:
      - name: local
        type: local
        llama_model: Test
server:
  session_slot_pool_size: $CHEAP_SLOTS
  contention_queue_policy: $CHEAP_POLICY
  local_large_context_cold_cache_threshold: $CHEAP_COLD
  local_model_ctx_size: $CHEAP_LOCAL_CTX
YAML

# Pristine copy of the fast fixture, restored after Test 6 mutates it.
cp "$PROXY_DIR/config-fast.yaml" "$PROXY_DIR/config-fast.yaml.pristine"

# Fake python: delegate other calls to the real interpreter; intercept
# `-m uvicorn` and report the selected config plus the effective per-slot
# context computed exactly like provider.py::effective_per_slot_threshold
# (ctx_size // slots - 4096 headroom) so the test can assert cheap/fast
# capacity matches the real mode profiles (LP-0MSX2FMN5006HYN5).
FAKE_PY="$SANDBOX/python3"
cat > "$FAKE_PY" <<'PY'
#!/usr/bin/env bash
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "uvicorn" ]; then
    echo "FAKE_UVICORN llama_proxy_config=${LLAMA_PROXY_CONFIG:-unset}"
    if [ -n "${LLAMA_PROXY_CONFIG:-}" ] && [ -f "$LLAMA_PROXY_CONFIG" ]; then
        /usr/bin/python3 -c '
import re
import sys

src = open(sys.argv[1]).read()


def server_val(key):
    m = re.search(r"^  %s:\s*(\S+)" % key, src, re.M)
    return m.group(1) if m else None


slots = server_val("session_slot_pool_size")
policym = re.search(r"^  contention_queue_policy:\s*(\S+)", src, re.M)
cold = server_val("local_large_context_cold_cache_threshold")
local_ctx = server_val("local_model_ctx_size")
# No per-period slot schedule (LP-0MTZRM5HV0007S0V): the static
# local_model_ctx_size is the total context across the pool slots.
ctx = local_ctx
effective = 0
if ctx is not None and slots is not None:
    try:
        per_slot = int(ctx) // int(slots)
        if per_slot > 4096:
            effective = per_slot - 4096
    except (ValueError, ZeroDivisionError):
        effective = 0
print("FAKE_MODE_FACTS slots=%s ctx_size=%s policy=%s cold_cache_threshold=%s local_model_ctx_size=%s effective_per_slot_threshold=%s" % (
    slots, ctx, policym.group(1) if policym else None, cold, local_ctx, effective))
' "$LLAMA_PROXY_CONFIG" 2>/dev/null || true
    fi
    exit 0
fi
exec /usr/bin/python3 "$@"
PY
chmod +x "$FAKE_PY"

# ---- Sandboxed restart-signature fakes (LP-0MUJRVYQH002IEHW) --------------
# The --restart tests must never touch live processes or ports. These fakes
# are found first on PATH (the sandbox dir also holds the fake python shim)
# and merely log their invocation; no live pkill/fuser ever runs.
KILL_LOG="$SANDBOX/kill.log"

for _fake in pkill fuser; do
    cat > "$SANDBOX/$_fake" <<SH
#!/usr/bin/env bash
echo "$_fake \$*" >> "$KILL_LOG"
exit 0
SH
    chmod +x "$SANDBOX/$_fake"
done

# Fake ss/netstat report no listeners so _wait_for_port_release() returns
# immediately and never probes the live proxy ports (8080/8081).
for _fake in ss netstat; do
    cat > "$SANDBOX/$_fake" <<'SH'
#!/usr/bin/env bash
exit 0
SH
    chmod +x "$SANDBOX/$_fake"
done

# Compute the merged-config sha256 exactly as start-proxy.sh does (via the
# shared config_merge helper mirrored into the sandbox).
merged_hash() {
    # $1 = base path, $2 = overlay path
    /usr/bin/python3 -c "
import sys
sys.path.insert(0, '$PROXY_DIR/proxy')
from config_merge import merged_config_sha256
print(merged_config_sha256('$1', '$2'))
"
}

_write_signature() {
    # $1 = mode label, $2 = config_sha256
    cat > "$PROXY_DIR/.llama_server_signature.json" <<JSON
{"mode": "$1", "model": "router", "parallel": 1, "router_mode": true, "config_sha256": "$2"}
JSON
}

run_restart() {
    # Sandboxed --restart run; resets the fake-kill log first.
    # --foreground keeps the intercepted final `exec uvicorn` in this captured
    # stream; start-proxy.sh detaches by default (LP-0MVGT0E6W000Q8CY) and the
    # restart assertions below read the script's own output.
    : > "$KILL_LOG"
    local out
    out=$(cd "$PROXY_DIR" && PATH="$SANDBOX:$PATH" bash scripts/start-proxy.sh --foreground --restart --port "$TEST_PORT" 2>&1)
    echo "$out"
}

TEST_PORT=8457

run_script() {
    # Usage: run_script  -> prints combined stdout+stderr of the sandboxed script
    # --foreground is REQUIRED: the fake python shim intercepts the final
    # `exec uvicorn` and reports the effective mode; detaching by default
    # (LP-0MVGT0E6W000Q8CY) would route that output to the log file instead.
    local out
    out=$(cd "$PROXY_DIR" && PATH="$SANDBOX:$PATH" bash scripts/start-proxy.sh --foreground --port "$TEST_PORT" 2>&1)
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
    # The sandbox profiles declare no api_key_env -> resolve_api_keys must
    # succeed in cheap mode even though the fast fixture exists.
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
    # restore the pristine fast fixture (NOT a copy of config.yaml — the
    # rich fast fixture carries the real ctx/policy/threshold values that
    # later tests (per-slot context) assert on)
    cp "$PROXY_DIR/config-fast.yaml.pristine" "$PROXY_DIR/config-fast.yaml"
}

# ---- Test 9: base-declared api_key_env is read from the merged config ------
test_api_keys_from_merged_base() {
    echo "--- Test: api_key_env declared only in the base config is detected ---"
    # The mode overlays no longer repeat api_key_env entries; a regression to
    # raw-mode-file reads would silently miss this base-only key.
    cp "$PROXY_DIR/config.yaml" "$PROXY_DIR/config.yaml.bak"
    cat > "$PROXY_DIR/config.yaml" <<YAML
models:
  test:
    providers:
      - name: local
        type: local
        llama_model: Test
  only-base:
    providers:
      - name: missing-remote
        type: remote
        endpoint: https://example.invalid
        api_key_env: TEST_MERGE_MISSING_KEY_XYZ
server:
  session_slot_pool_size: $BASE_SLOTS
YAML
    printf 'fast\n' > "$PROXY_DIR/.mode"
    local out
    out=$(run_script)
    if echo "$out" | grep -q "TEST_MERGE_MISSING_KEY_XYZ"; then
        pass "base-only api_key_env detected via the merged config"
    else
        fail "base-only api_key_env not detected: $out"
    fi
    if echo "$out" | grep -q "FAKE_UVICORN"; then
        fail "proxy started despite a genuinely missing API key"
    else
        pass "startup aborted on the genuinely missing API key"
    fi
    cp "$PROXY_DIR/config.yaml.bak" "$PROXY_DIR/config.yaml"
    rm -f "$PROXY_DIR/config.yaml.bak"
}

# ---- Test 7: per-slot context matches provider.py formula -----------------
test_per_slot_context() {
    echo "--- Test: per-slot context matches provider.py effective_per_slot_threshold ---"

    printf 'cheap\n' > "$PROXY_DIR/.mode"
    local out
    out=$(run_script)
    if echo "$out" | grep -q "FAKE_MODE_FACTS slots=$CHEAP_SLOTS ctx_size=$CHEAP_LOCAL_CTX policy=$CHEAP_POLICY cold_cache_threshold=$CHEAP_COLD local_model_ctx_size=$CHEAP_LOCAL_CTX effective_per_slot_threshold=83285"; then
        pass "cheap per-slot context = 83285 ($CHEAP_LOCAL_CTX//$CHEAP_SLOTS - 4096)"
    else
        fail "cheap per-slot context: unexpected output: $out"
    fi

    printf 'fast\n' > "$PROXY_DIR/.mode"
    out=$(run_script)
    if echo "$out" | grep -q "FAKE_MODE_FACTS slots=$FAST_SLOTS ctx_size=$FAST_LOCAL_CTX policy=$FAST_POLICY cold_cache_threshold=$FAST_COLD local_model_ctx_size=$FAST_LOCAL_CTX effective_per_slot_threshold=258048"; then
        pass "fast per-slot context = 258048 ($FAST_LOCAL_CTX//$FAST_SLOTS - 4096)"
    else
        fail "fast per-slot context: unexpected output: $out"
    fi
}

# ---- Test 8: Web UI slots section reflects real cheap capacity -------------
test_web_ui_slots_text() {
    echo "--- Test: Web UI slots section reflects real cheap/fast capacity ---"
    local index_html="$SCRIPT_DIR/../proxy/templates/index.html"
    if [ ! -f "$index_html" ]; then
        fail "index.html not found at $index_html"
        return
    fi
    # grep the FILE directly (not a pipe): under `set -o pipefail` a large
    # `echo "$content" | grep -q <match>` SIGPIPEs when grep exits early,
    # making the check flaky (LP-0MTZRM5HV0007S0V touched this test).
    if grep -q "${CHEAP_SLOTS}-slot local pool" "$index_html"; then
        pass "Web UI slots section says '${CHEAP_SLOTS}-slot local pool' (cheap = $CHEAP_SLOTS slots)"
    else
        fail "Web UI slots section missing '${CHEAP_SLOTS}-slot local pool' text"
    fi
    if [ "$CHEAP_SLOTS" -ne 1 ] && grep -q "1-slot local pool" "$index_html"; then
        fail "Web UI slots section still says stale '1-slot local pool' (cheap is $CHEAP_SLOTS slots)"
    else
        pass "Web UI slots section has no stale '1-slot local pool' text"
    fi
}

# ---- Restart-signature tests (LP-0MUJRVYQH002IEHW) -------------------------
# The sandbox base carries a base-only local-backend key the overlays omit
# (llama_server_port), so a base-only change is invisible in the raw overlay
# bytes but must still change the merged-config signature.

_ensure_base_base_only_key() {
    # Give the sandbox base a local-backend key absent from both overlays.
    if ! grep -q "^  llama_server_port:" "$PROXY_DIR/config.yaml"; then
        printf '  llama_server_port: 8080\n' >> "$PROXY_DIR/config.yaml"
    fi
}

# AC3: unchanged merged config preserves the warm llama-server.
test_restart_preserves_unchanged_merged_config() {
    echo "--- Test: --restart preserves llama-server when merged config unchanged ---"
    _ensure_base_base_only_key
    printf 'fast\n' > "$PROXY_DIR/.mode"
    local sha
    sha="$(merged_hash "$PROXY_DIR/config.yaml" "$PROXY_DIR/config-fast.yaml")"
    _write_signature fast "$sha"

    local out
    out="$(run_restart)"
    if echo "$out" | grep -q "Preserving running llama-server"; then
        pass "unchanged merged config: llama-server preserved"
    else
        fail "unchanged merged config: expected preserve, got: $out"
    fi
    if grep -q "llama-server" "$KILL_LOG"; then
        fail "unchanged merged config: fake pkill saw a llama-server kill ($(cat "$KILL_LOG"))"
    else
        pass "unchanged merged config: fake pkill never saw a llama-server kill"
    fi
}

# AC2: a base-only local-backend change forces a restart (overlay bytes
# unchanged — the raw-file hash would have matched).
test_restart_on_base_only_change() {
    echo "--- Test: --restart restarts llama-server on a base-only change ---"
    _ensure_base_base_only_key
    printf 'fast\n' > "$PROXY_DIR/.mode"
    local sha_before
    sha_before="$(merged_hash "$PROXY_DIR/config.yaml" "$PROXY_DIR/config-fast.yaml")"
    _write_signature fast "$sha_before"

    # Base-only edit; the fast overlay (and its raw bytes) are untouched.
    sed -i 's/^  llama_server_port: .*/  llama_server_port: 9090/' "$PROXY_DIR/config.yaml"
    local sha_after
    sha_after="$(merged_hash "$PROXY_DIR/config.yaml" "$PROXY_DIR/config-fast.yaml")"
    if [ "$sha_before" = "$sha_after" ]; then
        fail "base-only change did not change the merged hash"
        return
    fi

    local out
    out="$(run_restart)"
    if echo "$out" | grep -q "Preserving running llama-server"; then
        fail "base-only change: llama-server wrongly preserved: $out"
    else
        pass "base-only change: llama-server not preserved (KEEP_LLAMA=0)"
    fi
    if grep -q "llama-server" "$KILL_LOG"; then
        pass "base-only change: fake pkill killed llama-server"
    else
        fail "base-only change: llama-server kill not observed"
    fi

    # Restore the pre-change base value for later tests.
    sed -i 's/^  llama_server_port: .*/  llama_server_port: 8080/' "$PROXY_DIR/config.yaml"
}

# AC4: an overlay-only change still forces a restart (regression guard).
test_restart_on_overlay_only_change() {
    echo "--- Test: --restart restarts llama-server on an overlay-only change ---"
    printf 'cheap\n' > "$PROXY_DIR/.mode"
    local sha_before
    sha_before="$(merged_hash "$PROXY_DIR/config.yaml" "$PROXY_DIR/config-cheap.yaml")"
    _write_signature cheap "$sha_before"

    # Overlay-only edit: session_slot_pool_size lives in config-cheap.yaml.
    sed -i 's/^  session_slot_pool_size: .*/  session_slot_pool_size: 7/' "$PROXY_DIR/config-cheap.yaml"
    local sha_after
    sha_after="$(merged_hash "$PROXY_DIR/config.yaml" "$PROXY_DIR/config-cheap.yaml")"
    if [ "$sha_before" = "$sha_after" ]; then
        fail "overlay-only change did not change the merged hash"
        return
    fi

    local out
    out="$(run_restart)"
    if echo "$out" | grep -q "Preserving running llama-server"; then
        fail "overlay-only change: llama-server wrongly preserved: $out"
    else
        pass "overlay-only change: llama-server not preserved (KEEP_LLAMA=0)"
    fi
    if grep -q "llama-server" "$KILL_LOG"; then
        pass "overlay-only change: fake pkill killed llama-server"
    else
        fail "overlay-only change: llama-server kill not observed"
    fi
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
test_api_keys_from_merged_base
test_per_slot_context
test_web_ui_slots_text
test_restart_preserves_unchanged_merged_config
test_restart_on_base_only_change
test_restart_on_overlay_only_change
