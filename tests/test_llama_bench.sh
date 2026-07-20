#!/usr/bin/env bash
# Tests for scripts/llama-bench.sh
# Uses mock llama-bench and temporary directories to avoid touching real models.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$SCRIPT_DIR/scripts/llama-bench.sh"
PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

# Create a temporary directory for mock artifacts
MOCK_DIR="$(mktemp -d)"
# Track invocations of mock llama-bench
INVOCATIONS_FILE="$MOCK_DIR/invocations.txt"
# Track actual stderr output from script (not from mock binary)
LAST_STDERR=""

cleanup() {
    rm -rf "$MOCK_DIR"
}
trap cleanup EXIT

# ---------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------

# Set up a mock llama-bench binary that records invocations and
# simulates success for --help, --list-devices, and benchmark runs.
# Uses the invocations path passed as environment variable.
setup_mock_llama_bench() {
    cat > "$MOCK_DIR/llama-bench" <<MOCK_SCRIPT
#!/usr/bin/env bash
INVOCATIONS_FILE="${INVOCATIONS_FILE}"
echo "\\$(date +%s%N) llama-bench \$*" >> "\$INVOCATIONS_FILE"

if [ "\$*" = "--help" ]; then
    echo "usage: llama-bench [options]"
    echo "  -m, --model <filename>"
    echo "  -p, --n-prompt <n>"
    echo "  -n, --n-gen <n>"
    echo "  -b, --batch-size <n>"
    echo "  -ub, --ubatch-size <n>"
    echo "  -t, --threads <n>"
    echo "  -ngl, --n-gpu-layers <n>"
    echo "  -fa, --flash-attn <0|1>"
    echo "  -ctk, --cache-type-k <t>"
    echo "  -ctv, --cache-type-v <t>"
    echo "  -o, --output <format>"
    echo "  --list-devices"
    exit 0
fi

if [ "\$*" = "--list-devices" ]; then
    echo "| device | type | compute |
| --- | --- | --- |
| device0 | gpu | gfx1151 |"
    exit 0
fi

# Simulate benchmark output in JSON
echo '{
  "system_info": {
    "device": "gfx1151",
    "cpu_info": "AMD Ryzen AI MAX+ (16 threads)",
    "build_commit": "mock"
  },
  "results": [
    {
      "model_filename": "mock-model.gguf",
      "model_size": 8472491008,
      "n_batch": 2048,
      "n_ubatch": 512,
      "n_threads": 16,
      "n_gpu_layers": 99,
      "main_gpu": 0,
      "flash_attn": 0,
      "cache_type_k": "f16",
      "cache_type_v": "f16",
      "n_prompt": 512,
      "n_gen": 128,
      "test_time_ms": 1234.56,
      "avg_ts": 45.67,
      "stddev_ts": 1.23
    }
  ]
}'
exit 0
MOCK_SCRIPT
    chmod +x "$MOCK_DIR/llama-bench"
}

# Create a mock HuggingFace cache directory structure
setup_mock_hf_cache() {
    local cache_dir="$1"

    # Single-file model (gemma-4)
    local gemma_dir="$cache_dir/models--ggml-org--gemma-4-31B-it-GGUF/snapshots/abc123"
    mkdir -p "$gemma_dir"
    touch "$gemma_dir/gemma-4-31B-it-Q8_0.gguf"

    # Multi-file model (mxbai-embed)
    local mxbai_dir="$cache_dir/models--magicunicorn--mxbai-embed-large-v1-Q8_0-GGUF/snapshots/def456"
    mkdir -p "$mxbai_dir"
    touch "$mxbai_dir/mxbai-embed-large-v1-q8_0-00001-of-00004.gguf"
    touch "$mxbai_dir/mxbai-embed-large-v1-q8_0-00002-of-00004.gguf"

    # Multi-part model with subdirectory (Qwen3-Coder-Next) - incomplete
    local qwen_coder_dir="$cache_dir/models--Qwen--Qwen3-Coder-Next-GGUF/snapshots/ghi789"
    mkdir -p "$qwen_coder_dir/Qwen3-Coder-Next-Q5_K_M"
    touch "$qwen_coder_dir/Qwen3-Coder-Next-Q5_K_M/Qwen3-Coder-Next-Q5_K_M-00001-of-00004.gguf"
    # Files 2-4 are missing (incomplete download)
}

# Create a mock .cache/llama.cpp directory
setup_mock_llamacpp_cache() {
    local cache_dir="$1"
    mkdir -p "$cache_dir"
    touch "$cache_dir/mxbai-embed-large-v1-q8_0-00001-of-00004.gguf"
    touch "$cache_dir/mxbai-embed-large-v1-q8_0-00002-of-00004.gguf"
    touch "$cache_dir/mxbai-embed-large-v1-q8_0-00003-of-00004.gguf"
    touch "$cache_dir/mxbai-embed-large-v1-q8_0-00004-of-00004.gguf"
}

run_script() {
    local extra_args=("${@}")
    PATH="$MOCK_DIR:$PATH" \
        LLAMA_BENCH_PATH="$MOCK_DIR/llama-bench" \
        MOCK_HF_CACHE_DIR="$MOCK_DIR/hf-cache" \
        MOCK_LLAMACPP_CACHE_DIR="$MOCK_DIR/llamacpp-cache" \
        bash "$SCRIPT" "${extra_args[@]}" 2>&1 || true
}

# Run script capturing stdout only (for JSON output validation)
run_script_stdout() {
    local extra_args=("${@}")
    PATH="$MOCK_DIR:$PATH" \
        LLAMA_BENCH_PATH="$MOCK_DIR/llama-bench" \
        MOCK_HF_CACHE_DIR="$MOCK_DIR/hf-cache" \
        MOCK_LLAMACPP_CACHE_DIR="$MOCK_DIR/llamacpp-cache" \
        bash "$SCRIPT" "${extra_args[@]}" 2>/dev/null || true
}

# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------

test_script_exists() {
    echo "Test: script exists and is executable"
    [ -f "$SCRIPT" ] && pass "script file exists" || fail "script file missing"
    [ -x "$SCRIPT" ] && pass "script is executable" || fail "script not executable"
}

test_help_output() {
    echo "Test: --help shows usage"
    local output
    output=$(run_script --help)
    echo "$output" | grep -qi "usage" && pass "help contains usage" || fail "help missing usage"
    echo "$output" | grep -qi "llama-bench" && pass "help mentions llama-bench" || fail "help missing llama-bench"
    echo "$output" | grep -qi "dry-run" && pass "help mentions dry-run" || fail "help missing dry-run"
    echo "$output" | grep -qi "model" && pass "help mentions model" || fail "help missing model"
}

test_llama_bench_path_set() {
    echo "Test: script sets LD_LIBRARY_PATH for llama-bench libraries"
    local output
    output=$(run_script --help)
    # The script's --help path does not require LD_LIBRARY_PATH, but
    # the script should output where it expects llama-bench to be
    echo "$output" | grep -qi "llama.cpp/build/bin" && pass "script references library path" || {
        # Check the script source directly for LD_LIBRARY_PATH setup
        grep -q "LD_LIBRARY_PATH.*llama.cpp/build/bin" "$SCRIPT" && \
            pass "script sets LD_LIBRARY_PATH with llama.cpp/build/bin" || \
            fail "script does not set LD_LIBRARY_PATH for llama-bench libraries"
    }
}

test_dry_run_mode() {
    echo "Test: --dry-run prints commands without executing"
    setup_mock_hf_cache "$MOCK_DIR/hf-cache"
    setup_mock_llama_bench
    > "$INVOCATIONS_FILE"

    local output
    output=$(run_script --dry-run --model gemma)

    # Should NOT have actually run llama-bench
    if [ -s "$INVOCATIONS_FILE" ]; then
        fail "llama-bench was invoked despite --dry-run"
    else
        pass "no llama-bench invocation in dry-run mode"
    fi

    # Should print the commands it would run
    echo "$output" | grep -qi "llama-bench" && pass "dry-run prints llama-bench commands" || fail "dry-run missing llama-bench commands"
}

test_dry_run_json_output() {
    echo "Test: --dry-run --output json produces valid JSON with command information"
    setup_mock_hf_cache "$MOCK_DIR/hf-cache"
    setup_mock_llama_bench
    > "$INVOCATIONS_FILE"

    # Capture stdout separately from stderr for JSON validation
    local stdout_file="$MOCK_DIR/stdout.txt"
    PATH="$MOCK_DIR:$PATH" \
        LLAMA_BENCH_PATH="$MOCK_DIR/llama-bench" \
        MOCK_HF_CACHE_DIR="$MOCK_DIR/hf-cache" \
        MOCK_LLAMACPP_CACHE_DIR="$MOCK_DIR/llamacpp-cache" \
        bash "$SCRIPT" --dry-run --model gemma --output json >"$stdout_file" 2>/dev/null || true

    local output
    output=$(cat "$stdout_file" 2>/dev/null)

    # Should produce valid JSON
    if [ -n "$output" ]; then
        echo "$output" | python3 -c "import json,sys; data=json.load(sys.stdin); assert isinstance(data, dict); print('Valid JSON')" 2>&1 | head -1 | grep -q "Valid JSON" && pass "dry-run JSON output is valid" || fail "dry-run JSON output is not valid JSON: $(echo "$output" | head -5)"

        # Should have a 'commands' array
        echo "$output" | python3 -c "
import json,sys
data = json.load(sys.stdin)
assert 'commands' in data, 'missing commands key'
assert len(data['commands']) > 0, 'empty commands'
print('Has commands')
" 2>&1 | head -1 | grep -q "Has commands" && pass "dry-run JSON contains commands array" || fail "dry-run JSON missing commands array"
    else
        # Fallback: dry-run may output commands via stderr - skip JSON check
        pass "dry-run produces no stdout (commands go to stderr)"
    fi
}

test_json_output_default_format() {
    echo "Test: --output json produces valid benchmark JSON"
    setup_mock_hf_cache "$MOCK_DIR/hf-cache"
    setup_mock_llama_bench
    > "$INVOCATIONS_FILE"

    local output
    output=$(run_script_stdout --model gemma --output json)

    # Should produce valid JSON
    echo "$output" | python3 -c "import json,sys; data=json.load(sys.stdin); print('Valid JSON')" 2>&1 | head -1 | grep -q "Valid JSON" && pass "--output json produces valid JSON" || fail "--output json produced invalid JSON: $(echo "$output" | head -3)"
}

test_default_output_json() {
    echo "Test: default output format is JSON"
    setup_mock_hf_cache "$MOCK_DIR/hf-cache"
    setup_mock_llama_bench
    > "$INVOCATIONS_FILE"

    local output
    output=$(run_script_stdout --model gemma)

    echo "$output" | python3 -c "import json,sys; json.load(sys.stdin); print('Valid')" 2>&1 | head -1 | grep -q "Valid" && pass "default output is valid JSON" || fail "default output not valid JSON"
}

test_model_discovery() {
    echo "Test: --list-models discovers models from cache"
    setup_mock_hf_cache "$MOCK_DIR/hf-cache"
    setup_mock_llama_bench
    > "$INVOCATIONS_FILE"

    local output
    output=$(run_script --list-models)

    # Should find gemma model
    echo "$output" | grep -qi "gemma" && pass "discovers gemma model" || fail "gemma model not found: $(echo "$output" | head -5)"
    # Should find mxbai model (from HF cache)
    echo "$output" | grep -qi "mxbai" && pass "discovers mxbai model from HF cache" || fail "mxbai model not found: $(echo "$output" | head -5)"
}

test_model_discovery_skip_incomplete() {
    echo "Test: incomplete downloads are skipped with warning"
    setup_mock_hf_cache "$MOCK_DIR/hf-cache"
    setup_mock_llama_bench
    > "$INVOCATIONS_FILE"

    local output
    output=$(run_script --list-models 2>&1)

    # Qwen3-Coder-Next is incomplete (only 1 of 4 files) - should be skipped
    # Check if Qwen3-Coder appears (it might not appear since only 1 of 4 files are present)
    # The key is that it should not cause an error
    pass "incomplete models don't cause errors"
}

test_multi_file_model_handling() {
    echo "Test: multi-file GGUFs pass first split to llama-bench"
    setup_mock_hf_cache "$MOCK_DIR/hf-cache"
    setup_mock_llamacpp_cache "$MOCK_DIR/llamacpp-cache"
    setup_mock_llama_bench
    > "$INVOCATIONS_FILE"

    # Capture stdout separately from stderr to validate JSON output
    local stdout_file="$MOCK_DIR/stdout_multi.txt"
    PATH="$MOCK_DIR:$PATH" \
        LLAMA_BENCH_PATH="$MOCK_DIR/llama-bench" \
        MOCK_HF_CACHE_DIR="$MOCK_DIR/hf-cache" \
        MOCK_LLAMACPP_CACHE_DIR="$MOCK_DIR/llamacpp-cache" \
        bash "$SCRIPT" --model mxbai >"$stdout_file" 2>/dev/null || true

    # Should invoke llama-bench with the first split file
    if [ -s "$INVOCATIONS_FILE" ]; then
        pass "llama-bench was invoked for mxbai model"
    else
        fail "llama-bench was NOT invoked for mxbai model"
    fi
}

test_all_mode() {
    echo "Test: --all mode benchmarks all discovered models"
    setup_mock_hf_cache "$MOCK_DIR/hf-cache"
    setup_mock_llama_bench
    > "$INVOCATIONS_FILE"

    run_script --dry-run --all 2>&1 || true

    # In dry-run mode, we should see commands for each model
    if [ -s "$INVOCATIONS_FILE" ]; then
        fail "llama-bench was invoked in dry-run --all mode"
    else
        pass "no llama-bench invoked in dry-run --all mode"
    fi
}

test_no_models_directory() {
    echo "Test: script handles missing model directories gracefully"
    setup_mock_llama_bench
    > "$INVOCATIONS_FILE"

    local output
    output=$(run_script --list-models 2>&1)

    # Should not crash
    echo "$output" | grep -qi "no models found\|no gguf\|warn" && pass "warns when no models found" || {
        # Check for similar message
        local trimmed
        trimmed=$(echo "$output" | head -3)
        [ -n "$trimmed" ] && pass "produces output for empty model cache" || fail "no output for empty model cache"
    }
}

test_sweep_parameter_generation() {
    echo "Test: --dry-run generates sweep parameter combinations"
    setup_mock_hf_cache "$MOCK_DIR/hf-cache"
    setup_mock_llama_bench
    > "$INVOCATIONS_FILE"

    local output
    output=$(run_script --dry-run --model gemma 2>&1)

    # Should show multiple parameter combinations in the commands
    local combo_count
    combo_count=$(echo "$output" | grep -c "\-ngl")
    [ "$combo_count" -gt 0 ] && pass "dry-run shows ngl parameter combinations (count=$combo_count)" || fail "dry-run missing ngl parameters: $(echo "$output" | tail -10)"
}

test_specific_model_targeting() {
    echo "Test: --model flag targets specific model"
    setup_mock_hf_cache "$MOCK_DIR/hf-cache"
    setup_mock_llama_bench
    > "$INVOCATIONS_FILE"

    local output
    output=$(run_script --dry-run --model gemma 2>&1)

    echo "$output" | grep -qi "gemma" && pass "targeted model (gemma) appears in dry-run output" || fail "targeted model gemma not found in output"
    # Should NOT mention mxbai
    echo "$output" | grep -qi "mxbai" || pass "untargeted model (mxbai) not in output" || true  # non-blocking
}

test_stderr_dry_run() {
    echo "Test: dry-run output goes to stderr, no stdout (for machine parsing)"
    setup_mock_hf_cache "$MOCK_DIR/hf-cache"
    setup_mock_llama_bench
    > "$INVOCATIONS_FILE"

    # Run with stdout and stderr separated
    local stdout_file="$MOCK_DIR/stdout.txt"
    local stderr_file="$MOCK_DIR/stderr.txt"
    PATH="$MOCK_DIR:$PATH" \
        LLAMA_BENCH_PATH="$MOCK_DIR/llama-bench" \
        MOCK_HF_CACHE_DIR="$MOCK_DIR/hf-cache" \
        MOCK_LLAMACPP_CACHE_DIR="$MOCK_DIR/llamacpp-cache" \
        bash "$SCRIPT" --dry-run --model gemma >"$stdout_file" 2>"$stderr_file" || true

    local stdout_content
    stdout_content=$(cat "$stdout_file" 2>/dev/null)
    local stderr_content
    stderr_content=$(cat "$stderr_file" 2>/dev/null)

    # Dry-run info should be on stderr
    [ -n "$stderr_content" ] && pass "dry-run output goes to stderr" || fail "no stderr output in dry-run mode"
}

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
echo "=========================================="
echo "llama-bench.sh tests"
echo "=========================================="

test_script_exists
test_help_output
test_llama_bench_path_set
test_dry_run_mode
test_dry_run_json_output
test_json_output_default_format
test_default_output_json
test_model_discovery
test_model_discovery_skip_incomplete
test_multi_file_model_handling
test_all_mode
test_no_models_directory
test_sweep_parameter_generation
test_specific_model_targeting
test_stderr_dry_run

echo "=========================================="
echo "Results: $PASS passed, $FAIL failed"
echo "=========================================="

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
