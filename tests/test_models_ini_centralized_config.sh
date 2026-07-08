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
test_get_quantization
test_get_quantization_no_suffix
test_ctx_override_from_models_ini
test_quant_override_from_models_ini
test_logging_shows_source
test_logging_shows_quant_source
test_logging_shows_quant_fallback
test_hardcoded_fallback

echo "=========================================="
echo "Results: $PASS passed, $FAIL failed"
echo "=========================================="

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
