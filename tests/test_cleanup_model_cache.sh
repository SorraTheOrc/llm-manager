#!/usr/bin/env bash
# Tests for scripts/cleanup-model-cache.sh
# Uses temporary directories to avoid touching real caches.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
SCRIPT="$SCRIPT_DIR/cleanup-model-cache.sh"
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

# Create a 101 MB dummy GGUF file (above the 100 MB threshold)
make_gguf() {
    local path="$1"
    dd if=/dev/zero of="$path" bs=1M count=101 2>/dev/null
}

# ---------------------------------------------------------------
# Test: --dry-run outputs candidates without deleting anything
# ---------------------------------------------------------------
test_dry_run() {
    echo "Test: --dry-run outputs candidates without deletion"
    TESTS_TMPDIR="$(mktemp -d)"
    local ini="$TESTS_TMPDIR/models.ini"
    local hub="$TESTS_TMPDIR/hub"
    local llama="$TESTS_TMPDIR/llama.cpp"
    mkdir -p "$hub/models--kept--model" "$hub/models--stale--model" "$llama"
    printf '[kept]\nhf-repo = kept/model:Q5_K_M\n' > "$ini"
    make_gguf "$llama/ggml-org_gemma-4-31B-it-GGUF_gemma-4-31B-it-Q8_0.gguf"

    local output
    output=$(HF_HUB_CACHE="$hub" LLAMA_CPP_CACHE="$llama" MODELS_INI="$ini" \
             bash "$SCRIPT" --dry-run 2>&1 || true)

    [ -d "$hub/models--kept--model" ] && pass "kept model still exists" || fail "kept model was deleted"
    [ -d "$hub/models--stale--model" ] && pass "stale model still exists" || fail "stale model was deleted by dry-run"
    [ -f "$llama/ggml-org_gemma-4-31B-it-GGUF_gemma-4-31B-it-Q8_0.gguf" ] && pass "GGUF file still exists" || fail "GGUF file was deleted by dry-run"
    echo "$output" | grep -q "stale--model" && pass "output mentions stale model" || fail "output missing stale model reference"
    echo "$output" | grep -q "gemma" && pass "output mentions GGUF removal" || fail "output missing GGUF reference"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: --dry-run --json produces valid JSON
# ---------------------------------------------------------------
test_dry_run_json() {
    echo "Test: --dry-run --json produces valid JSON"
    TESTS_TMPDIR="$(mktemp -d)"
    local ini="$TESTS_TMPDIR/models.ini"
    local hub="$TESTS_TMPDIR/hub"
    local llama="$TESTS_TMPDIR/llama.cpp"
    mkdir -p "$hub/models--kept--model" "$hub/models--stale--model" "$llama"
    printf '[kept]\nhf-repo = kept/model:Q5_K_M\n' > "$ini"
    make_gguf "$llama/ggml-org_gemma-4-31B-it-GGUF_gemma-4-31B-it-Q8_0.gguf"

    local output
    output=$(HF_HUB_CACHE="$hub" LLAMA_CPP_CACHE="$llama" MODELS_INI="$ini" \
             bash "$SCRIPT" --dry-run --json 2>&1 || true)

    echo "$output" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert "ok" in d or "dry_run" in d' && pass "output is valid JSON" || fail "output is not valid JSON"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Idempotent -- running twice produces same result
# ---------------------------------------------------------------
test_idempotent() {
    echo "Test: idempotent -- second run produces identical state"
    TESTS_TMPDIR="$(mktemp -d)"
    local ini="$TESTS_TMPDIR/models.ini"
    local hub="$TESTS_TMPDIR/hub"
    local llama="$TESTS_TMPDIR/llama.cpp"
    mkdir -p "$hub/models--kept--model" "$hub/models--stale--model" "$llama"
    printf '[kept]\nhf-repo = kept/model:Q5_K_M\n' > "$ini"
    make_gguf "$llama/ggml-org_gemma-4-31B-it-GGUF_gemma-4-31B-it-Q8_0.gguf"

    # First run (actual deletion)
    HF_HUB_CACHE="$hub" LLAMA_CPP_CACHE="$llama" MODELS_INI="$ini" bash "$SCRIPT" 2>/dev/null || true

    local kept_after1 stale_after1 gguf_after1
    kept_after1=$([ -d "$hub/models--kept--model" ] && echo "yes" || echo "no")
    stale_after1=$([ -d "$hub/models--stale--model" ] && echo "yes" || echo "no")
    gguf_after1=$([ -f "$llama/ggml-org_gemma-4-31B-it-GGUF_gemma-4-31B-it-Q8_0.gguf" ] && echo "yes" || echo "no")

    # Second run
    HF_HUB_CACHE="$hub" LLAMA_CPP_CACHE="$llama" MODELS_INI="$ini" bash "$SCRIPT" 2>/dev/null || true

    local kept_after2 stale_after2 gguf_after2
    kept_after2=$([ -d "$hub/models--kept--model" ] && echo "yes" || echo "no")
    stale_after2=$([ -d "$hub/models--stale--model" ] && echo "yes" || echo "no")
    gguf_after2=$([ -f "$llama/ggml-org_gemma-4-31B-it-GGUF_gemma-4-31B-it-Q8_0.gguf" ] && echo "yes" || echo "no")

    [ "$kept_after1" = "$kept_after2" ] && pass "kept model preserved on second run" || fail "kept model state changed"
    [ "$stale_after1" = "$stale_after2" ] && pass "stale model state unchanged" || fail "stale model state changed"
    [ "$gguf_after1" = "$gguf_after2" ] && pass "GGUF file state unchanged" || fail "GGUF file state changed"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Correctly parses models.ini and identifies configured models
# ---------------------------------------------------------------
test_parses_models_ini() {
    echo "Test: correctly parses models.ini and identifies configured models"
    TESTS_TMPDIR="$(mktemp -d)"
    local ini="$TESTS_TMPDIR/models.ini"
    local hub="$TESTS_TMPDIR/hub"
    mkdir -p "$hub/models--model1--model" "$hub/models--model2--model" "$hub/models--model3--model"

    printf '[model1]\nhf-repo = owner/model1:Q5_K_M\n\n[model2]\nhf-repo = owner/model2:Q8_0\n\n[model3]\nhf-repo = owner/model3:Q4_K_M\n' > "$ini"

    local output
    output=$(HF_HUB_CACHE="$hub" MODELS_INI="$ini" bash "$SCRIPT" --dry-run 2>&1 || true)

    echo "$output" | grep -q "model1" && pass "identified model1" || fail "missed model1"
    echo "$output" | grep -q "model2" && pass "identified model2" || fail "missed model2"
    echo "$output" | grep -q "model3" && pass "identified model3" || fail "missed model3"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Correctly identifies stale models not in models.ini
# ---------------------------------------------------------------
test_identifies_stale_models() {
    echo "Test: identifies stale models not in models.ini"
    TESTS_TMPDIR="$(mktemp -d)"
    local ini="$TESTS_TMPDIR/models.ini"
    local hub="$TESTS_TMPDIR/hub"
    mkdir -p "$hub/models--owner--configured" "$hub/models--stale--one" "$hub/models--stale--two"

    printf '[configured]\nhf-repo = owner/configured:Q5_K_M\n' > "$ini"

    local output
    output=$(HF_HUB_CACHE="$hub" MODELS_INI="$ini" bash "$SCRIPT" --dry-run 2>&1 || true)

    echo "$output" | grep -q "stale--one" && pass "identified stale-one for removal" || fail "missed stale-one"
    echo "$output" | grep -q "stale--two" && pass "identified stale-two for removal" || fail "missed stale-two"
    ! echo "$output" | grep -q "Would remove.*owner--configured" && pass "configured model not listed for removal" || fail "configured model listed for removal"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Identifies duplicate llama.cpp GGUF cache file
# ---------------------------------------------------------------
test_identifies_dup_gguf() {
    echo "Test: identifies duplicate llama.cpp GGUF cache"
    TESTS_TMPDIR="$(mktemp -d)"
    local ini="$TESTS_TMPDIR/models.ini"
    local hub="$TESTS_TMPDIR/hub"
    local llama="$TESTS_TMPDIR/llama.cpp"
    mkdir -p "$hub/models--kept--model" "$llama"
    printf '[kept]\nhf-repo = kept/model:Q5_K_M\n' > "$ini"
    make_gguf "$llama/ggml-org_gemma-4-31B-it-GGUF_gemma-4-31B-it-Q8_0.gguf"

    local output
    output=$(HF_HUB_CACHE="$hub" LLAMA_CPP_CACHE="$llama" MODELS_INI="$ini" bash "$SCRIPT" --dry-run 2>&1 || true)

    echo "$output" | grep -q "gemma.*gguf\|GGUF\|llama.cpp" && pass "GGUF file identified for removal" || fail "GGUF file not identified"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Handles missing models.ini gracefully
# ---------------------------------------------------------------
test_missing_models_ini() {
    echo "Test: handles missing models.ini gracefully"
    TESTS_TMPDIR="$(mktemp -d)"
    local ini="$TESTS_TMPDIR/nonexistent.ini"
    local hub="$TESTS_TMPDIR/hub"
    mkdir -p "$hub/models--test--model"

    local output rc=0
    output=$(HF_HUB_CACHE="$hub" MODELS_INI="$ini" bash "$SCRIPT" --dry-run 2>&1) || rc=$?

    [ $rc -ne 0 ] && pass "exits non-zero with missing ini" || fail "should exit non-zero"
    [ -d "$hub/models--test--model" ] && pass "no models deleted" || fail "models were deleted despite missing ini"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Handles models.ini with no hf-repo entries
# ---------------------------------------------------------------
test_no_hf_repo_entries() {
    echo "Test: handles models.ini with no hf-repo entries"
    TESTS_TMPDIR="$(mktemp -d)"
    local ini="$TESTS_TMPDIR/models.ini"
    local hub="$TESTS_TMPDIR/hub"
    mkdir -p "$hub/models--test--model"

    printf '[section]\nctx-size = 8192\n' > "$ini"

    local output rc=0
    output=$(HF_HUB_CACHE="$hub" MODELS_INI="$ini" bash "$SCRIPT" --dry-run 2>&1) || rc=$?

    [ -d "$hub/models--test--model" ] && pass "no models deleted with no hf-repo" || fail "models were deleted with no hf-repo"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Test: Exit code is 0 on success
# ---------------------------------------------------------------
test_exit_code_success() {
    echo "Test: exits 0 on success"
    TESTS_TMPDIR="$(mktemp -d)"
    local ini="$TESTS_TMPDIR/models.ini"
    local hub="$TESTS_TMPDIR/hub"
    mkdir -p "$hub/models--kept--model"
    printf '[kept]\nhf-repo = kept/model:Q5_K_M\n' > "$ini"

    local output rc=0
    output=$(HF_HUB_CACHE="$hub" MODELS_INI="$ini" bash "$SCRIPT" --dry-run 2>&1) || rc=$?

    [ $rc -eq 0 ] && pass "exits 0 on dry-run success" || fail "non-zero exit ($rc) on dry-run"

    cleanup_tmp
}

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
echo "========================================="
echo "cleanup-model-cache.sh tests"
echo "========================================="

test_dry_run
test_dry_run_json
test_idempotent
test_parses_models_ini
test_identifies_stale_models
test_identifies_dup_gguf
test_missing_models_ini
test_no_hf_repo_entries
test_exit_code_success

echo "========================================="
echo "Results: $PASS passed, $FAIL failed"
echo "========================================="

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
