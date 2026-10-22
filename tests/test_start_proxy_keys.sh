#!/usr/bin/env bash
# Tests for start-proxy.sh stale provider API-key detection.
#
# Pins the AC1–AC4 shell contract of the parent work item
# (LP-0MVD2RF7H008B7OT) before the fail-closed implementation lands
# (implementation child LP-0MVF7DRMK001NHT8):
#
#   1. env var set AND matching auth.json  -> proxy starts; a short fingerprint
#      of the key is logged without the plaintext key
#   2. env var set AND differing           -> startup refused (non-zero exit,
#      uvicorn never exec'd); the error names the env var and BOTH fingerprints,
#      and the plaintext env key is never printed
#   3. differing env var + explicit opt-out (--use-env-keys, and separately
#      LLAMA_PROXY_USE_ENV_KEYS=1)         -> proxy starts, a warning is logged,
#      and the environment key wins
#   4. env var unset + resolvable auth.json-> the auth.json key is exported and
#      its fingerprint is logged (regression guard for the existing path)
#   5. auth.json missing + env var unset   -> the aggregated missing-key error
#      path is unchanged
#   6. --use-env-keys is consumed by start-proxy.sh (never forwarded to
#      uvicorn) and documented in the script's usage header
#   7. a mixed setup (one env-matched key + one key resolved from auth.json)
#      starts normally — only a genuine mismatch blocks
#   8. the mismatch report names the auth.json key name(s) that were tried, so
#      the failure is self-diagnosing
#
# Harness pattern (mirrors tests/test_start_proxy_mode.sh): a sandboxed copy of
# start-proxy.sh, a fake `python3` that intercepts the final `exec uvicorn`, an
# isolated $HOME holding a fixture auth.json, and an ephemeral port. The live
# proxy, port 8000, llama-server and the TTS server are never touched: the
# port probes (ss/netstat) are stubbed to report no listeners and uvicorn is
# never executed.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_SCRIPT="$SCRIPT_DIR/../proxy/scripts/start-proxy.sh"
REAL_PROXY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/proxy"
PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); echo "PASS: $*"; }
fail() { FAIL=$((FAIL + 1)); echo "FAIL: $*"; }

SANDBOX=""

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

# ---- Fixtures --------------------------------------------------------------
# Fixture keys are >= 12 chars and their first 8 characters are distinct, so a
# fingerprint assertion can tell the two keys apart without ever printing the
# plaintext key.
AUTH_KEY="MATCHKEY_aaaaaaaaaaaa"
STALE_KEY="STALEKEY_bbbbbbbbbbbb"
AUTH_FP="${AUTH_KEY:0:8}"
STALE_FP="${STALE_KEY:0:8}"
ENV_VAR="OPENCODE_API_KEY"
TEST_PORT=8461

# ---- Sandbox ---------------------------------------------------------------

SANDBOX="$(mktemp -d)"
PROXY_DIR="$SANDBOX/proxy"
mkdir -p "$PROXY_DIR/scripts"
cp "$SOURCE_SCRIPT" "$PROXY_DIR/scripts/start-proxy.sh"
chmod +x "$PROXY_DIR/scripts/start-proxy.sh"

# Mirror the dependency-light proxy package (config_merge for the merged-config
# key enumeration, api_key_resolver for the comparison helper). Copying the
# whole top-level *.py set keeps the harness working when the implementation
# child adds a new sibling helper module.
mkdir -p "$PROXY_DIR/proxy"
cp "$REAL_PROXY_DIR/proxy/"*.py "$PROXY_DIR/proxy/"
for _required in config_merge.py api_key_resolver.py; do
    if [ ! -f "$PROXY_DIR/proxy/$_required" ]; then
        fail "could not mirror proxy/proxy/$_required into the sandbox"
        exit 1
    fi
done

# Isolated HOME with the fixture auth.json (the script reads
# $HOME/.pi/agent/auth.json).
mkdir -p "$SANDBOX/home/.pi/agent"
AUTH_FILE="$SANDBOX/home/.pi/agent/auth.json"
cat > "$AUTH_FILE" <<JSON
{
  "opencode-go": {"type": "api_key", "key": "$AUTH_KEY"}
}
JSON

# Minimal config declaring exactly one covered api_key_env.
cat > "$PROXY_DIR/config.yaml" <<YAML
models:
  test-model:
    providers:
      - name: test-remote
        type: remote
        endpoint: https://example.invalid
        api_key_env: $ENV_VAR
server:
  session_slot_pool_size: 1
YAML

# Fake python: delegate every call to the real interpreter except the final
# `-m uvicorn`, which is intercepted. It reports the uvicorn argv (so the test
# can prove --use-env-keys is not forwarded) and only ever echoes a key
# FINGERPRINT, never the plaintext key.
FAKE_PY="$SANDBOX/python3"
cat > "$FAKE_PY" <<'PY'
#!/usr/bin/env bash
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "uvicorn" ]; then
    key="${OPENCODE_API_KEY:-}"
    echo "FAKE_UVICORN args=$* key_fp=${key:0:8} gh_fp=${GITHUB_TOKEN:0:8}"
    exit 0
fi
exec /usr/bin/python3 "$@"
PY
chmod +x "$FAKE_PY"

# Stub port probes: report no listeners so the sandboxed script never touches
# the live proxy ports.
for _fake in ss netstat; do
    cat > "$SANDBOX/$_fake" <<'SH'
#!/usr/bin/env bash
exit 0
SH
    chmod +x "$SANDBOX/$_fake"
done

# ---- Run helpers -----------------------------------------------------------

RUN_LOG="$SANDBOX/last-run.log"
LAST_RC=0
# Per-run environment: empty means "unset for this run".
OPENCODE_VALUE=""
USE_ENV_KEYS=""

run_proxy() {
    # Usage: run_proxy [start-proxy.sh args...]
    #
    # --foreground is REQUIRED here: this harness intercepts the final
    # `exec uvicorn` via the fake python3 shim and asserts on the shim's output.
    # start-proxy.sh now detaches by default (LP-0MVGT0E6W000Q8CY), which would
    # send the shim's output to the log file instead of this captured stream and
    # make every key assertion vacuous.
    local -a envp=(
        "HOME=$SANDBOX/home"
        "PATH=$SANDBOX:$PATH"
        "PYTHONPATH=$PROXY_DIR:$PROXY_DIR/proxy"
    )
    [ -n "$OPENCODE_VALUE" ] && envp+=("$ENV_VAR=$OPENCODE_VALUE")
    [ -n "$USE_ENV_KEYS" ] && envp+=("LLAMA_PROXY_USE_ENV_KEYS=$USE_ENV_KEYS")

    local out
    out=$(cd "$PROXY_DIR" && env -u "$ENV_VAR" -u LLAMA_PROXY_USE_ENV_KEYS \
        "${envp[@]}" bash scripts/start-proxy.sh --foreground --port "$TEST_PORT" "$@" 2>&1)
    LAST_RC=$?
    printf '%s\n' "$out" > "$RUN_LOG"
}

out_has() {
    grep -qF -- "$1" "$RUN_LOG"
}

out_matches() {
    grep -qE -- "$1" "$RUN_LOG"
}

# Script-emitted log lines only (our fake uvicorn echo is excluded so the
# assertions cannot pass on the harness's own output).
NON_UVICORN_LOG="$SANDBOX/non-uvicorn.log"
strip_uvicorn() {
    grep -v '^FAKE_UVICORN' "$RUN_LOG" > "$NON_UVICORN_LOG"
}

script_log_has() {
    strip_uvicorn
    grep -qF -- "$1" "$NON_UVICORN_LOG"
}

script_log_matches_i() {
    strip_uvicorn
    grep -qiE -- "$1" "$NON_UVICORN_LOG"
}

# True when a line emitted by the SCRIPT (i.e. not our fake uvicorn line)
# contains both patterns. Used to prove the startup log itself carries the
# fingerprint, independently of the fake uvicorn echo.
script_log_has_pair() {
    awk -v a="$1" -v b="$2" \
        '!/^FAKE_UVICORN/ && index($0, a) && index($0, b) {found=1} END {exit !found}' \
        "$RUN_LOG"
}

started() { out_has "FAKE_UVICORN"; }
refused() { ! started; }

# ---- Test 1: env var matches auth.json -> start + fingerprint logged -------

test_env_matches_auth_json() {
    echo "--- Test: env var matching auth.json starts the proxy and logs the fingerprint ---"
    OPENCODE_VALUE="$AUTH_KEY"
    USE_ENV_KEYS=""
    run_proxy

    if [ "$LAST_RC" -eq 0 ] && started; then
        pass "matching env key: proxy starts (uvicorn exec reached)"
    else
        fail "matching env key: startup failed (exit $LAST_RC): $(cat "$RUN_LOG")"
    fi

    if script_log_has_pair "$ENV_VAR" "$AUTH_FP"; then
        pass "matching env key: $ENV_VAR logged with 8-char fingerprint $AUTH_FP"
    else
        fail "matching env key: no startup log line carries $ENV_VAR + fingerprint $AUTH_FP"
    fi

    if out_matches "FAKE_UVICORN .*key_fp=$AUTH_FP"; then
        pass "matching env key: the matching env key is the one in effect"
    else
        fail "matching env key: uvicorn did not receive the env key"
    fi

    if out_has "$AUTH_KEY"; then
        fail "matching env key: plaintext key leaked into stdout/stderr"
    else
        pass "matching env key: plaintext key never printed"
    fi

    if out_has "ERROR"; then
        fail "matching env key: unexpected error output: $(cat "$RUN_LOG")"
    else
        pass "matching env key: no error output"
    fi
}

# ---- Test 2: env var differs -> fail closed, fingerprints, no plaintext ----

test_env_differs_refuses_startup() {
    echo "--- Test: env var differing from auth.json refuses startup ---"
    OPENCODE_VALUE="$STALE_KEY"
    USE_ENV_KEYS=""
    run_proxy

    if [ "$LAST_RC" -ne 0 ]; then
        pass "differing env key: exit code non-zero ($LAST_RC)"
    else
        fail "differing env key: expected non-zero exit, got 0"
    fi

    if refused; then
        pass "differing env key: uvicorn never exec'd"
    else
        fail "differing env key: proxy started despite the stale env key"
    fi

    if script_log_matches_i "refus|error|mismatch|differ"; then
        pass "differing env key: refusal is reported loudly"
    else
        fail "differing env key: no refusal/error message: $(cat "$RUN_LOG")"
    fi

    if script_log_matches_i "${ENV_VAR}.*(refus|error|mismatch|differ|stale)|(refus|error|mismatch|differ|stale).*${ENV_VAR}"; then
        pass "differing env key: the error names the offending env var"
    else
        fail "differing env key: error does not name $ENV_VAR alongside the mismatch"
    fi

    if script_log_has_pair "$ENV_VAR" "$STALE_FP"; then
        pass "differing env key: environment fingerprint $STALE_FP logged"
    else
        fail "differing env key: environment fingerprint $STALE_FP not logged"
    fi

    if script_log_has "$AUTH_FP"; then
        pass "differing env key: auth.json fingerprint $AUTH_FP logged"
    else
        fail "differing env key: auth.json fingerprint $AUTH_FP not logged"
    fi

    # Self-diagnosing message: name the auth.json key(s) the resolver tried so
    # the operator can fix auth.json without reading the resolver source.
    # GITHUB-free assertion: "opencode-go" only appears in the candidate list.
    if script_log_has "opencode-go"; then
        pass "differing env key: the report names the auth.json key(s) tried"
    else
        fail "differing env key: the report does not name the auth.json key(s) tried"
    fi

    if out_has "$STALE_KEY"; then
        fail "differing env key: plaintext env key leaked into stdout/stderr"
    else
        pass "differing env key: plaintext env key never printed"
    fi

    if out_has "$AUTH_KEY"; then
        fail "differing env key: plaintext auth.json key leaked into stdout/stderr"
    else
        pass "differing env key: plaintext auth.json key never printed"
    fi
}

# ---- Test 3: explicit opt-out -> warn and let the env key win --------------

test_opt_out_uses_env_key() {
    echo "--- Test: --use-env-keys opt-out warns and keeps the env key ---"
    OPENCODE_VALUE="$STALE_KEY"
    USE_ENV_KEYS=""
    run_proxy --use-env-keys

    if [ "$LAST_RC" -eq 0 ] && started; then
        pass "--use-env-keys: proxy starts despite the differing env key"
    else
        fail "--use-env-keys: startup failed (exit $LAST_RC): $(cat "$RUN_LOG")"
    fi

    if script_log_matches_i "warn.*${ENV_VAR}|${ENV_VAR}.*warn"; then
        pass "--use-env-keys: the warning names $ENV_VAR"
    else
        fail "--use-env-keys: no warning naming $ENV_VAR: $(cat "$RUN_LOG")"
    fi

    if out_matches "FAKE_UVICORN .*key_fp=$STALE_FP"; then
        pass "--use-env-keys: the environment key wins"
    else
        fail "--use-env-keys: the environment key did not win"
    fi

    if out_has "$STALE_KEY"; then
        fail "--use-env-keys: plaintext env key leaked into stdout/stderr"
    else
        pass "--use-env-keys: plaintext env key never printed"
    fi

    echo "--- Test: LLAMA_PROXY_USE_ENV_KEYS=1 opt-out warns and keeps the env key ---"
    OPENCODE_VALUE="$STALE_KEY"
    USE_ENV_KEYS="1"
    run_proxy

    if [ "$LAST_RC" -eq 0 ] && started; then
        pass "LLAMA_PROXY_USE_ENV_KEYS=1: proxy starts despite the differing env key"
    else
        fail "LLAMA_PROXY_USE_ENV_KEYS=1: startup failed (exit $LAST_RC): $(cat "$RUN_LOG")"
    fi

    if script_log_matches_i "warn.*${ENV_VAR}|${ENV_VAR}.*warn"; then
        pass "LLAMA_PROXY_USE_ENV_KEYS=1: the warning names $ENV_VAR"
    else
        fail "LLAMA_PROXY_USE_ENV_KEYS=1: no warning naming $ENV_VAR: $(cat "$RUN_LOG")"
    fi

    if out_matches "FAKE_UVICORN .*key_fp=$STALE_FP"; then
        pass "LLAMA_PROXY_USE_ENV_KEYS=1: the environment key wins"
    else
        fail "LLAMA_PROXY_USE_ENV_KEYS=1: the environment key did not win"
    fi
}

# ---- Test 4: env var unset -> resolve from auth.json (regression guard) ----

test_env_unset_resolves_from_auth_json() {
    echo "--- Test: env var unset resolves from auth.json and logs the fingerprint ---"
    OPENCODE_VALUE=""
    USE_ENV_KEYS=""
    run_proxy

    if [ "$LAST_RC" -eq 0 ] && started; then
        pass "env unset: proxy starts with the auth.json key"
    else
        fail "env unset: startup failed (exit $LAST_RC): $(cat "$RUN_LOG")"
    fi

    if out_matches "FAKE_UVICORN .*key_fp=$AUTH_FP"; then
        pass "env unset: the auth.json key is exported to the proxy"
    else
        fail "env unset: the auth.json key was not exported"
    fi

    if script_log_has_pair "$ENV_VAR" "$AUTH_FP"; then
        pass "env unset: resolved key logged with fingerprint $AUTH_FP"
    else
        fail "env unset: no startup log line carries $ENV_VAR + fingerprint $AUTH_FP"
    fi

    if out_has "$AUTH_KEY"; then
        fail "env unset: plaintext key leaked into stdout/stderr"
    else
        pass "env unset: plaintext key never printed"
    fi
}

# ---- Test 5: auth.json missing + env var unset -> existing error path ------

test_auth_json_missing_keeps_missing_key_error() {
    echo "--- Test: auth.json missing with no env key keeps the aggregated error ---"
    mv "$AUTH_FILE" "$AUTH_FILE.bak"
    OPENCODE_VALUE=""
    USE_ENV_KEYS=""
    run_proxy
    mv "$AUTH_FILE.bak" "$AUTH_FILE"

    if [ "$LAST_RC" -ne 0 ]; then
        pass "auth.json missing: exit code non-zero ($LAST_RC)"
    else
        fail "auth.json missing: expected non-zero exit, got 0"
    fi

    if refused; then
        pass "auth.json missing: uvicorn never exec'd"
    else
        fail "auth.json missing: proxy started without any resolvable key"
    fi

    if out_has "ERROR" && out_has "$ENV_VAR"; then
        pass "auth.json missing: aggregated error names $ENV_VAR"
    else
        fail "auth.json missing: aggregated error missing: $(cat "$RUN_LOG")"
    fi

    if out_has "could not be resolved"; then
        pass "auth.json missing: existing 'could not be resolved' wording preserved"
    else
        fail "auth.json missing: existing error wording changed or lost: $(cat "$RUN_LOG")"
    fi
}

# ---- Test 7: mixed env/auth.json setup starts normally ---------------------

test_mixed_env_and_auth_json_setup_starts() {
    echo "--- Test: mixed env + auth.json keys start normally (only mismatches block) ---"
    local second_var="GITHUB_TOKEN"
    local second_key="MIXEDKEY_cccccccccc"
    local second_fp="${second_key:0:8}"

    cp "$AUTH_FILE" "$AUTH_FILE.single"
    cat > "$AUTH_FILE" <<JSON
{
  "opencode-go": {"type": "api_key", "key": "$AUTH_KEY"},
  "github_token": {"type": "api_key", "key": "$second_key"}
}
JSON
    cp "$PROXY_DIR/config.yaml" "$PROXY_DIR/config.yaml.single"
    cat > "$PROXY_DIR/config.yaml" <<YAML
models:
  test-model:
    providers:
      - name: test-remote
        type: remote
        endpoint: https://example.invalid
        api_key_env: $ENV_VAR
      - name: test-github
        type: remote
        endpoint: https://example.invalid
        api_key_env: $second_var
server:
  session_slot_pool_size: 1
YAML

    OPENCODE_VALUE="$AUTH_KEY"
    USE_ENV_KEYS=""
    run_proxy

    # Restore the single-key fixtures first so a failure here cannot poison the
    # remaining runs.
    mv "$PROXY_DIR/config.yaml.single" "$PROXY_DIR/config.yaml"
    mv "$AUTH_FILE.single" "$AUTH_FILE"

    if [ "$LAST_RC" -eq 0 ] && started; then
        pass "mixed setup: proxy starts (env-matched + auth.json-resolved keys)"
    else
        fail "mixed setup: startup failed (exit $LAST_RC): $(cat "$RUN_LOG")"
    fi

    if script_log_has_pair "$ENV_VAR" "$AUTH_FP"; then
        pass "mixed setup: env-matched $ENV_VAR logged with fingerprint"
    else
        fail "mixed setup: env-matched $ENV_VAR not logged with fingerprint"
    fi

    if script_log_has_pair "$second_var" "$second_fp"; then
        pass "mixed setup: $second_var resolved from auth.json with fingerprint"
    else
        fail "mixed setup: $second_var not resolved/logged from auth.json"
    fi

    if out_matches "FAKE_UVICORN .*gh_fp=$second_fp"; then
        pass "mixed setup: the auth.json-resolved key reached the proxy"
    else
        fail "mixed setup: the auth.json-resolved key was not exported"
    fi

    if out_has "$second_key"; then
        fail "mixed setup: plaintext auth.json key leaked into stdout/stderr"
    else
        pass "mixed setup: plaintext auth.json key never printed"
    fi
}

# ---- Test 6: --use-env-keys consumed (not forwarded) and documented --------

test_use_env_keys_flag_consumed_and_documented() {
    echo "--- Test: --use-env-keys is consumed and documented ---"
    OPENCODE_VALUE="$AUTH_KEY"
    USE_ENV_KEYS=""
    run_proxy --use-env-keys

    if [ "$LAST_RC" -eq 0 ] && started; then
        pass "--use-env-keys: recognised by start-proxy.sh (startup succeeded)"
    else
        fail "--use-env-keys: startup failed (exit $LAST_RC): $(cat "$RUN_LOG")"
    fi

    if out_matches "FAKE_UVICORN args=.*--use-env-keys"; then
        fail "--use-env-keys: forwarded to uvicorn"
    else
        pass "--use-env-keys: never forwarded to uvicorn"
    fi

    # The script documents its flags in the leading usage header (same
    # convention as the pytest check for --keep-llama-server,
    # proxy/tests/test_restart_lifecycle.py).
    local header
    header="$(awk '/^VENV_DIR=/{exit} {print}' "$SOURCE_SCRIPT" | grep '^#')"
    if printf '%s\n' "$header" | grep -qF -- "--use-env-keys"; then
        pass "--use-env-keys documented in the usage header"
    else
        fail "--use-env-keys not documented in the usage header"
    fi
}

# ---- Run -------------------------------------------------------------------

echo "=== start-proxy.sh Stale API-Key Detection Tests ==="
echo ""
test_env_matches_auth_json
test_env_differs_refuses_startup
test_opt_out_uses_env_key
test_env_unset_resolves_from_auth_json
test_auth_json_missing_keeps_missing_key_error
test_use_env_keys_flag_consumed_and_documented
test_mixed_env_and_auth_json_setup_starts

# The cleanup trap prints the results footer; force the documented non-zero
# exit on failure (a trap's last command does not change the script status).
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
