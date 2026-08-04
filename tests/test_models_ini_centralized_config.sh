#!/usr/bin/env bash
# Tests for start-llama.sh reading quantization and ctx-size from models.ini
# Verifies that models.ini is used as the single source of truth.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$SCRIPT_DIR/start-llama.sh"
MODELS_INI="$SCRIPT_DIR/models.ini"
PASS=0
FAIL=0
TESTS_TMPDIR=""

pass() { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

cleanup_tmp() {
    if [ -n "$TESTS_TMPDIR" ] && [ -d "$TESTS_TMPDIR" ]; then
        rm -rf "$TESTS_TMPDIR"
    fi
    TESTS_TMPDIR=""
}

# ---------------------------------------------------------------
# Test: get_quantization helper extracts quant suffix from hf-repo
# ---------------------------------------------------------------
test_get_quantization() {
    echo "Test: get_quantization extracts quant suffix from hf-repo"

    local result
    # Test with a known hf-repo format
    result=$(bash -c '
        source_config() {
            local target_model="$1"
            local ini_file="$2"

            awk -v target="$target_model" '\''BEGIN { found=0; repo="" }
            /^\[/ {
                gsub(/\[|\]/, "")
                if (tolower($0) == tolower(target)) {
                    found=1
                } else {
                    found=0
                }
            }
            found && /^hf-repo/ {
                gsub(/.*=/, "")
                gsub(/^[ \t]+|[ \t]+$/, "")
                repo=$0
                exit
            }
            END { if (repo != "") print repo }'\'' "$ini_file"
        }
        quant_from_hf_repo() {
            local hf_repo="$1"
            echo "$hf_repo" | awk -F: '\''{if (NF>1) print $NF}'\''
        }

        tmp=$(mktemp)
        cat > "$tmp" <<"INIE"
[qwen3]
hf-repo = unsloth/Qwen3.6-35B-A3B-GGUF:Q5_K_M
ctx-size = 131072
INIE
        repo=$(source_config "qwen3" "$tmp")
        quant_from_hf_repo "$repo"
        rm -f "$tmp"
    ') || true

    [ "$result" = "Q5_K_M" ] && pass "quantization extracted as Q5_K_M (got: $result)" || fail "expected Q5_K_M, got: $result"
}

# ---------------------------------------------------------------
# Test: get_quantization returns empty when hf-repo has no quant suffix
# ---------------------------------------------------------------
test_get_quantization_no_suffix() {
    echo "Test: get_quantization returns empty when hf-repo has no quant suffix"

    local result
    result=$(bash -c '
        tmp=$(mktemp)
        cat > "$tmp" <<"INIE"
[test-model]
hf-repo = org/model-name-gguf
ctx-size = 4096
INIE
        repo=$(awk -v target="test-model" '\''BEGIN { found=0; repo="" }
            /^\[/ { gsub(/\[|\]/, ""); if (tolower($0) == tolower(target)) found=1; else found=0 }
            found && /^hf-repo/ { gsub(/.*=/, ""); gsub(/^[ \t]+|[ \t]+$/, ""); repo=$0; exit }
            END { if (repo != "") print repo }'\'' "$tmp")
        echo "$repo" | awk -F: '\''{if (NF>1) print $NF}'\''
        rm -f "$tmp"
    ') || true

    [ -z "$result" ] && pass "empty when no quant suffix (got: '$result')" || fail "expected empty, got: $result"
}

# ---------------------------------------------------------------
# Test: Script reads CONTEXT from models.ini
# ---------------------------------------------------------------
test_ctx_override_from_models_ini() {
    echo "Test: CONTEXT is overridden from models.ini"

    TESTS_TMPDIR="$(mktemp -d)"

    # Create test models.ini with known values
    cat > "$TESTS_TMPDIR/models.ini" << 'INI'
[qwen3]
hf-repo = unsloth/Qwen3.6-35B-A3B-GGUF:Q5_K_M
ctx-size = 99999
INI

    # Run script - it will fail at llama-server but config is printed first
    local output rc=0
    output=$(LLAMA_MODELS_PRESET="$TESTS_TMPDIR/models.ini" bash "$SCRIPT" qwen3 2>&1 || rc=$?) || true

    echo "$output" | grep -q "CONTEXT=99999" && pass "CONTEXT=99999 from models.ini overrides hardcoded 131072" || fail "CONTEXT not overridden (output: $(echo "$output" | grep 'CONTEXT='))"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Script reads QUANTIZATION from models.ini hf-repo suffix
# ---------------------------------------------------------------
test_quant_override_from_models_ini() {
    echo "Test: QUANTIZATION is read from models.ini hf-repo suffix"

    TESTS_TMPDIR="$(mktemp -d)"

    cat > "$TESTS_TMPDIR/models.ini" << 'INI'
[qwen3]
hf-repo = unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M
ctx-size = 131072
INI

    local output rc=0
    output=$(LLAMA_MODELS_PRESET="$TESTS_TMPDIR/models.ini" bash "$SCRIPT" qwen3 2>&1 || rc=$?) || true

    echo "$output" | grep -q "QUANTIZATION=Q4_K_M" && pass "QUANTIZATION=Q4_K_M from models.ini (override hardcoded Q8_0)" || fail "QUANTIZATION not overridden from models.ini (output: $(echo "$output" | grep 'QUANTIZATION='))"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Logging shows source (models.ini vs fallback)
# ---------------------------------------------------------------
test_logging_shows_source() {
    echo "Test: Startup logs show source of values"

    TESTS_TMPDIR="$(mktemp -d)"

    cat > "$TESTS_TMPDIR/models.ini" << 'INI'
[qwen3]
hf-repo = unsloth/Qwen3.6-35B-A3B-GGUF:Q5_K_M
ctx-size = 88888
INI

    local output rc=0
    output=$(LLAMA_MODELS_PRESET="$TESTS_TMPDIR/models.ini" bash "$SCRIPT" qwen3 2>&1 || rc=$?) || true

    # Check for source indicators for context
    echo "$output" | grep -qi "models.ini" && pass "log mentions models.ini as source" || fail "log does not mention models.ini"

    # Check for specific source indicator (either from models.ini or fallback)
    grep -q "Read ctx-size=88888 from" <<< "$output" && pass "log shows ctx-size read from models.ini" || fail "log does not show ctx-size source"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Logging shows quantization source from models.ini
# ---------------------------------------------------------------
test_logging_shows_quant_source() {
    echo "Test: Startup log shows quantization source from models.ini"

    TESTS_TMPDIR="$(mktemp -d)"

    cat > "$TESTS_TMPDIR/models.ini" << 'INI'
[qwen3]
hf-repo = unsloth/Qwen3.6-35B-A3B-GGUF:Q5_K_M
ctx-size = 131072
INI

    local output rc=0
    output=$(LLAMA_MODELS_PRESET="$TESTS_TMPDIR/models.ini" bash "$SCRIPT" qwen3 2>&1 || rc=$?) || true

    echo "$output" | grep -q "Read quantization=Q5_K_M from" && pass "log shows quantization read from models.ini" || fail "log does not show quantization source (output: $(echo "$output" | grep -i 'quantization'))"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Logging shows fallback when no models.ini entry
# ---------------------------------------------------------------
test_logging_shows_quant_fallback() {
    echo "Test: Startup log shows quantization fallback"

    TESTS_TMPDIR="$(mktemp -d)"

    # Create models.ini WITHOUT qwen3
    cat > "$TESTS_TMPDIR/models.ini" << 'INI'
[other-model]
hf-repo = org/other:Q2_K
ctx-size = 4096
INI

    local output rc=0
    output=$(LLAMA_MODELS_PRESET="$TESTS_TMPDIR/models.ini" bash "$SCRIPT" qwen3 2>&1 || rc=$?) || true

    echo "$output" | grep -q "No hf-repo found in" && pass "log shows hf-repo fallback message" || fail "log missing hf-repo fallback (output: $(echo "$output" | grep -i 'hf-repo\\|quantization'))"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Hardcoded fallback when model not in models.ini
# ---------------------------------------------------------------

test_hardcoded_fallback() {
    echo "Test: Hardcoded defaults used when model not in models.ini"

    TESTS_TMPDIR="$(mktemp -d)"

    # Create models.ini WITHOUT the test model
    cat > "$TESTS_TMPDIR/models.ini" << 'INI'
[other-model]
hf-repo = org/other:Q2_K
ctx-size = 4096
INI

    local output rc=0
    output=$(LLAMA_MODELS_PRESET="$TESTS_TMPDIR/models.ini" bash "$SCRIPT" qwen3 2>&1 || rc=$?) || true

    # qwen3 hardcodes QUANTIZATION=Q8_0 and CONTEXT=131072
    echo "$output" | grep -q "QUANTIZATION=Q8_0" && pass "hardcoded QUANTIZATION=Q8_0 used as fallback" || fail "hardcoded quantization not used (output: $(echo "$output" | grep 'QUANTIZATION='))"
    echo "$output" | grep -q "CONTEXT=131072" && pass "hardcoded CONTEXT=131072 used as fallback" || fail "hardcoded context not used (output: $(echo "$output" | grep 'CONTEXT='))"
    # Should mention fallback
    echo "$output" | grep -qi "fallback\|No ctx-size found\|using CONTEXT" && pass "fallback message present" || fail "no fallback message found"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: models.ini has [global] section with ngl
# ---------------------------------------------------------------
test_global_ngl_from_models_ini() {
    echo "Test: [global] ngl is parsed from models.ini"

    TESTS_TMPDIR="$(mktemp -d)"

    # Create a test models.ini with [global] section
    cat > "$TESTS_TMPDIR/models.ini" << 'INI'
[global]
ngl = 99
ctx-size = 8192
slot-save-path = /tmp/slot-cache

[mxbai-embed]
hf-repo = magicunicorn/mxbai-embed-large-v1-Q8_0-GGUF:Q8_0
ctx-size = 2048
INI

    # Use awk to parse [global] ngl
    local result
    result=$(awk 'BEGIN { found=0; val="" }
    /^\[/ { gsub(/\[|\]/, ""); found=0; if (tolower($0) == "global") found=1 }
    found && /^ngl/ { gsub(/.*=/, ""); gsub(/^[ \t]+|[ \t]+$/, ""); val=$0; exit }
    END { if (val != "") print val }' "$TESTS_TMPDIR/models.ini")

    [ "$result" = "99" ] && pass "[global] ngl parsed as 99 (got: $result)" || fail "expected [global] ngl=99, got: $result"

    cleanup_tmp
}

test_global_ngl_defaults_to_80() {
    echo "Test: Production models.ini [global] ngl matches running server"

    local result
    result=$(awk 'BEGIN { found=0; val="" }
    /^\[/ { gsub(/\[|\]/, ""); found=0; if (tolower($0) == "global") found=1 }
    found && /^ngl/ { gsub(/.*=/, ""); gsub(/^[ \t]+|[ \t]+$/, ""); val=$0; exit }
    END { if (val != "") print val }' "$MODELS_INI")

    # ngl=80 was set in commit 64d8aeb (benchmarking and fine tuning) and
    # matches the running llama-server (-ngl 80). The test previously
    # expected 99 which was the pre-benchmark value (LP-0MSB1ILZI003FFAU).
    [ "$result" = "80" ] && pass "Production [global] ngl is 80 (got: $result)" || fail "expected [global] ngl=80, got: $result"

    cleanup_tmp
}

test_global_ngl_zero_disables_gpu() {
    echo "Test: [global] ngl=0 disables GPU offload"

    TESTS_TMPDIR="$(mktemp -d)"

    cat > "$TESTS_TMPDIR/models.ini" << 'INI'
[global]
ngl = 0
ctx-size = 8192
INI

    local result
    result=$(awk 'BEGIN { found=0; val="" }
    /^\[/ { gsub(/\[|\]/, ""); found=0; if (tolower($0) == "global") found=1 }
    found && /^ngl/ { gsub(/.*=/, ""); gsub(/^[ \t]+|[ \t]+$/, ""); val=$0; exit }
    END { if (val != "") print val }' "$TESTS_TMPDIR/models.ini")

    [ "$result" = "0" ] && pass "[global] ngl=0 parsed correctly (got: $result)" || fail "expected [global] ngl=0, got: $result"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Production models.ini has per-model ctx-size (LP-0MSAZXXDY005AWA1)
# ---------------------------------------------------------------
# llama.cpp router only cascades `[*]` sections as the global preset;
# a `[global]` section is parsed as a model named "global" and its keys
# are never applied to other models. Per-model ctx-size must therefore
# live in each model's own section.
# ---------------------------------------------------------------
test_per_model_ctx_size_present() {
    echo "Test: Production models.ini [Qwen3] has per-model ctx-size"

    local ctx
    ctx=$(awk 'BEGIN { found=0; ctx="" }
    /^\[/ { gsub(/\[|\]/, ""); found=0; if (tolower($0) == "qwen3") found=1 }
    found && /^ctx-size/ { gsub(/.*=/, ""); gsub(/^[ \t]+|[ \t]+$/, ""); ctx=$0; exit }
    END { if (ctx != "") print ctx }' "$MODELS_INI")

    [ -n "$ctx" ] && pass "[Qwen3] ctx-size=$ctx present in models.ini" || fail "[Qwen3] ctx-size missing from models.ini"

    # ctx-size must not live only in [global] (which llama.cpp ignores)
    local global_ctx
    global_ctx=$(awk 'BEGIN { found=0; ctx="" }
    /^\[/ { gsub(/\[|\]/, ""); found=0; if (tolower($0) == "global") found=1 }
    found && /^ctx-size/ { gsub(/.*=/, ""); gsub(/^[ \t]+|[ \t]+$/, ""); ctx=$0; exit }
    END { if (ctx != "") print ctx }' "$MODELS_INI")

    [ -z "$global_ctx" ] && pass "[global] carries no ctx-size (only per-model sections)" || fail "[global] still has ctx-size=$global_ctx (ignored by llama.cpp router)"
}

# ---------------------------------------------------------------
# Test: Production models.ini [Qwen3] has per-model cache-type-k/v
# (LP-0MSDCLQ2W001LGWC: KV-cache quantization for decode speed)
# ---------------------------------------------------------------
# KV cache type is a first-class llama-server preset option
# (LLAMA_ARG_CACHE_TYPE_K / _V); router mode reads it directly from
# models.ini, so it must live in the model's own section like ctx-size.
# ---------------------------------------------------------------
test_per_model_cache_type_present() {
    echo "Test: Production models.ini [Qwen3] has per-model cache-type-k/v"

    local ctk ctv
    ctk=$(awk 'BEGIN { found=0; val="" }
    /^\[/ { gsub(/\[|\]/, ""); found=0; if (tolower($0) == "qwen3") found=1 }
    found && /^cache-type-k/ { gsub(/.*=/, ""); gsub(/^[ \t]+|[ \t]+$/, ""); val=$0; exit }
    END { if (val != "") print val }' "$MODELS_INI")

    ctv=$(awk 'BEGIN { found=0; val="" }
    /^\[/ { gsub(/\[|\]/, ""); found=0; if (tolower($0) == "qwen3") found=1 }
    found && /^cache-type-v/ { gsub(/.*=/, ""); gsub(/^[ \t]+|[ \t]+$/, ""); val=$0; exit }
    END { if (val != "") print val }' "$MODELS_INI")

    [ -n "$ctk" ] && pass "[Qwen3] cache-type-k=$ctk present in models.ini" || fail "[Qwen3] cache-type-k missing from models.ini"
    [ -n "$ctv" ] && pass "[Qwen3] cache-type-v=$ctv present in models.ini" || fail "[Qwen3] cache-type-v missing from models.ini"
    [ "$ctk" = "q8_0" ] && pass "[Qwen3] cache-type-k is q8_0 (got: $ctk)" || fail "[Qwen3] cache-type-k expected q8_0, got: $ctk"
    [ "$ctv" = "q8_0" ] && pass "[Qwen3] cache-type-v is q8_0 (got: $ctv)" || fail "[Qwen3] cache-type-v expected q8_0, got: $ctv"
}

# ---------------------------------------------------------------
# Test: start-llama.sh reads cache-type from models.ini (single-model)
# ---------------------------------------------------------------
test_cache_type_override_from_models_ini() {
    echo "Test: CACHE_TYPE_K/V are read from models.ini"

    TESTS_TMPDIR="$(mktemp -d)"

    cat > "$TESTS_TMPDIR/models.ini" << 'INI'
[qwen3]
hf-repo = unsloth/Qwen3.6-35B-A3B-GGUF:Q5_K_M
ctx-size = 131072
cache-type-k = q4_0
cache-type-v = q8_0
INI

    local output rc=0
    output=$(LLAMA_MODELS_PRESET="$TESTS_TMPDIR/models.ini" bash "$SCRIPT" qwen3 2>&1 || rc=$?) || true

    echo "$output" | grep -q "CACHE_TYPE_K=q4_0" && pass "CACHE_TYPE_K=q4_0 from models.ini" || fail "CACHE_TYPE_K not read from models.ini (output: $(echo "$output" | grep 'CACHE_TYPE_K='))"
    echo "$output" | grep -q "CACHE_TYPE_V=q8_0" && pass "CACHE_TYPE_V=q8_0 from models.ini" || fail "CACHE_TYPE_V not read from models.ini (output: $(echo "$output" | grep 'CACHE_TYPE_V='))"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: cache-type flags are passed to llama-server (single-model)
# ---------------------------------------------------------------
test_cache_type_flags_passed() {
    echo "Test: --cache-type-k/--cache-type-v are passed to llama-server"

    TESTS_TMPDIR="$(mktemp -d)"

    cat > "$TESTS_TMPDIR/models.ini" << 'INI'
[qwen3]
hf-repo = unsloth/Qwen3.6-35B-A3B-GGUF:Q5_K_M
ctx-size = 131072
cache-type-k = q4_0
cache-type-v = q8_0
INI

    # Intercept the llama-server invocation: use a fake binary that records args
    cat > "$TESTS_TMPDIR/llama-server" << 'FAKE'
#!/usr/bin/env bash
echo "ARGS: $*"
exit 0
FAKE
    chmod +x "$TESTS_TMPDIR/llama-server"

    local output rc=0
    output=$(LLAMA_SERVER_BIN="$TESTS_TMPDIR/llama-server" LLAMA_MODELS_PRESET="$TESTS_TMPDIR/models.ini" bash "$SCRIPT" qwen3 2>&1 || rc=$?) || true

    echo "$output" | grep -q "ARGS:.*--cache-type-k q4_0" && pass "--cache-type-k q4_0 passed to llama-server" || fail "--cache-type-k missing from llama-server args (output: $(echo "$output" | grep 'ARGS:') )"
    echo "$output" | grep -q "ARGS:.*--cache-type-v q8_0" && pass "--cache-type-v q8_0 passed to llama-server" || fail "--cache-type-v missing from llama-server args (output: $(echo "$output" | grep 'ARGS:') )"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: cache-type defaults to f16 (llama-server default) when unset
# ---------------------------------------------------------------
test_cache_type_default_when_unset() {
    echo "Test: cache-type defaults to f16 when not in models.ini"

    TESTS_TMPDIR="$(mktemp -d)"

    cat > "$TESTS_TMPDIR/models.ini" << 'INI'
[qwen3]
hf-repo = unsloth/Qwen3.6-35B-A3B-GGUF:Q5_K_M
ctx-size = 131072
INI

    local output rc=0
    output=$(LLAMA_MODELS_PRESET="$TESTS_TMPDIR/models.ini" bash "$SCRIPT" qwen3 2>&1 || rc=$?) || true

    echo "$output" | grep -q "CACHE_TYPE_K=f16 (default)" && pass "CACHE_TYPE_K=f16 (default) when unset" || fail "CACHE_TYPE_K default missing (output: $(echo "$output" | grep 'CACHE_TYPE_K='))"
    echo "$output" | grep -q "CACHE_TYPE_V=f16 (default)" && pass "CACHE_TYPE_V=f16 (default) when unset" || fail "CACHE_TYPE_V default missing (output: $(echo "$output" | grep 'CACHE_TYPE_V='))"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Script exists and is executable
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
echo "start-llama.sh models.ini configuration tests"
echo "=========================================="

test_script_exists
test_per_model_ctx_size_present
test_per_model_cache_type_present
test_global_ngl_from_models_ini
test_global_ngl_defaults_to_80
test_global_ngl_zero_disables_gpu
test_get_quantization
test_get_quantization_no_suffix
test_ctx_override_from_models_ini
test_quant_override_from_models_ini
test_cache_type_override_from_models_ini
test_cache_type_flags_passed
test_cache_type_default_when_unset
test_logging_shows_source
test_logging_shows_quant_source
test_logging_shows_quant_fallback
test_hardcoded_fallback

echo "=========================================="
echo "Results: $PASS passed, $FAIL failed"
echo "=========================================="

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
