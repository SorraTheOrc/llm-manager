#!/usr/bin/env bash
# llama-bench.sh — Wrapper around llama-bench for model discovery,
# parameter sweeps, and result collection on Strix Halo (gfx1151) hardware.
#
# Usage:
#   ./scripts/llama-bench.sh --help
#   ./scripts/llama-bench.sh --all                      # benchmark every discovered model
#   ./scripts/llama-bench.sh --model gemma              # benchmark a specific model
#   ./scripts/llama-bench.sh --dry-run --model gemma    # preview commands without running
#   ./scripts/llama-bench.sh --list-models              # list discovered models
#   ./scripts/llama-bench.sh --output json --model gemma # explicit JSON output
#
# Output: By default, results are printed as JSON to stdout.
# In --dry-run mode, planned commands are printed to stderr.
#
set -euo pipefail

# ──────────────────────────────────────────────
# Defaults (tuned for Strix Halo: 16-core CPU, AMD GPU gfx1151)
# ──────────────────────────────────────────────
DEFAULT_NGPU=99
DEFAULT_THREADS=16
DEFAULT_BATCH=2048
DEFAULT_UBATCH=512
DEFAULT_N_PROMPT=512
DEFAULT_N_GEN=128
DEFAULT_CTK="f16"
DEFAULT_CTV="f16"
DEFAULT_FA=1
DEFAULT_REPETITIONS=3

# ──────────────────────────────────────────────
# Sweep ranges (used when --all is specified)
# ──────────────────────────────────────────────
SWEEP_NGPU=(99 80 60)
SWEEP_THREADS=(16 12 8)
SWEEP_BATCH=(4096 2048 1024)
SWEEP_UBATCH=(512 256)
SWEEP_FA=(1)
SWEEP_CTK=("f16" "q8_0")
SWEEP_CTV=("f16" "q8_0")
SWEEP_N_PROMPT=(512 1024)
SWEEP_N_GEN=(128 256)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Allow override for testing
LLAMA_BENCH="${LLAMA_BENCH_PATH:-/usr/local/bin/llama-bench}"
LLAMA_CPP_BUILD="${LLAMA_CPP_BUILD_DIR:-$HOME/llama.cpp/build/bin}"

# HuggingFace cache (allow override for testing)
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HF_HUB_CACHE="${HF_HOME}/hub"

# llama.cpp local cache
LLAMACPP_CACHE="${LLAMACPP_CACHE_DIR:-$HOME/.cache/llama.cpp}"

# ──────────────────────────────────────────────
# Helper: fix library path so llama-bench can find its shared libraries
# ──────────────────────────────────────────────
fix_library_path() {
    if [ -d "$LLAMA_CPP_BUILD" ]; then
        # Add to LD_LIBRARY_PATH if not already present
        case ":${LD_LIBRARY_PATH:-}:" in
            *":${LLAMA_CPP_BUILD}:"*) ;;
            *) export LD_LIBRARY_PATH="${LLAMA_CPP_BUILD}:${LD_LIBRARY_PATH:-}" ;;
        esac
    fi
}

# ──────────────────────────────────────────────
# Helper: find llama-bench binary
# ──────────────────────────────────────────────
find_llama_bench() {
    if [ -x "$LLAMA_BENCH" ]; then
        echo "$LLAMA_BENCH"
        return 0
    fi
    # Try the build directory
    local build_bench="${LLAMA_CPP_BUILD}/llama-bench"
    if [ -x "$build_bench" ]; then
        echo "$build_bench"
        return 0
    fi
    # Try PATH
    local path_bench
    path_bench=$(command -v llama-bench 2>/dev/null || true)
    if [ -n "$path_bench" ]; then
        echo "$path_bench"
        return 0
    fi
    return 1
}

# ──────────────────────────────────────────────
# Helper: discover GGUF model files from cache dirs
# ──────────────────────────────────────────────
discover_models() {
    local -a models=()
    local -a seen_names=()

    # Helper to check if a model name has already been seen
    _is_seen() {
        local name="$1"
        local seen
        for seen in "${seen_names[@]:-}"; do
            [ "$seen" = "$name" ] && return 0
        done
        return 1
    }

    # Discover from HF cache (snapshots directory structure)
    if [ -d "$HF_HUB_CACHE" ]; then
        # Look for GGUF files in models--*/snapshots/*/ directories (standard HF cache)
        # Use -o -type l to handle symlinks (HF cache snapshots are symlinks to blobs)
        while IFS= read -r -d '' gguf; do
            local name
            name=$(basename "$gguf" .gguf)
            # Skip mmproj files (vision encoders, not standalone LLMs)
            echo "$name" | grep -qi "mmproj" && continue
            # For multi-file GGUFs, only register the first split
            if echo "$name" | grep -qP '00001-of-\d+$'; then
                local base_name="${name%-00001-of-*}"
                _is_seen "$base_name" && continue
                seen_names+=("$base_name")
                models+=("$gguf")
            elif echo "$name" | grep -qP '\d{5}-of-\d{5}$'; then
                # Skip non-first splits
                continue
            else
                # Single-file model
                _is_seen "$name" && continue
                seen_names+=("$name")
                models+=("$gguf")
            fi
        done < <(find "$HF_HUB_CACHE" \( -type f -o -type l \) -name "*.gguf" -not -name "*.downloadInProgress" -print0 2>/dev/null)
    fi

    # Discover from llama.cpp local cache
    if [ -d "$LLAMACPP_CACHE" ]; then
        while IFS= read -r -d '' gguf; do
            local name
            name=$(basename "$gguf" .gguf)
            if echo "$name" | grep -qP '00001-of-\d+$'; then
                local base_name="${name%-00001-of-*}"
                _is_seen "$base_name" && continue
                seen_names+=("$base_name")
                models+=("$gguf")
            elif echo "$name" | grep -qP '\d{5}-of-\d{5}$'; then
                continue
            else
                _is_seen "$name" && continue
                seen_names+=("$name")
                models+=("$gguf")
            fi
        done < <(find "$LLAMACPP_CACHE" \( -type f -o -type l \) -name "*.gguf" -print0 2>/dev/null)
    fi

    # Check for incomplete downloads (multi-file models with missing parts)
    local -a valid_models=()
    local model
    for model in "${models[@]:-}"; do
        local model_name
        model_name=$(basename "$model" .gguf)
        # Check if this is a multi-file model
        if echo "$model_name" | grep -qP '00001-of-\d+$'; then
            local total_parts
            total_parts=$(echo "$model_name" | sed -n 's/.*-00001-of-0*\([0-9]*\)\.gguf$/\1/p')
            local base_path
            base_path="${model%-00001-of-*}"
            local all_present=true
            local i
            for ((i=2; i<=total_parts; i++)); do
                local part_file
                part_file="$(dirname "$model")/${base_path}-$(printf "%05d-of-%05d" "$i" "$total_parts").gguf"
                if [ ! -f "$part_file" ]; then
                    all_present=false
                    break
                fi
            done
            if [ "$all_present" = true ]; then
                valid_models+=("$model")
            else
                echo "[WARN] Skipping incomplete model: $(basename "$model" | sed 's/-00001-of-[0-9]*\.gguf$//') (missing part $i of $total_parts)" >&2
            fi
        else
            valid_models+=("$model")
        fi
    done

    # If no models found, output empty
    if [ ${#valid_models[@]} -eq 0 ]; then
        return 0
    fi

    # Return models
    local m
    for m in "${valid_models[@]:-}"; do
        echo "$m"
    done
}

# ──────────────────────────────────────────────
# Helper: extract a human-readable model name from a GGUF path
# ──────────────────────────────────────────────
model_display_name() {
    local path="$1"
    local filename
    filename=$(basename "$path" .gguf)
    # Remove multi-file suffix for display
    filename=$(echo "$filename" | sed 's/-00001-of-[0-9]*$//')
    echo "$filename"
}

# ──────────────────────────────────────────────
# Helper: check if a GGUF file is valid (exists and has size > 0)
# ──────────────────────────────────────────────
is_valid_gguf() {
    [ -f "$1" ] && [ -s "$1" ]
}

# ──────────────────────────────────────────────
# Build a single llama-bench command line
# ──────────────────────────────────────────────
build_cmd() {
    local model_path="$1"
    shift
    local -a extra_args=("$@")

    fix_library_path
    local bench
    bench=$(find_llama_bench) || {
        echo "ERROR: llama-bench not found" >&2
        return 1
    }

    local -a cmd=("$bench" -m "$model_path" -o json)

    # Add any extra arguments passed via sweep or CLI
    if [ ${#extra_args[@]} -gt 0 ]; then
        cmd+=("${extra_args[@]}")
    fi

    echo "${cmd[@]}"
}

# ──────────────────────────────────────────────
# Run a single benchmark and output JSON result
# ──────────────────────────────────────────────
run_benchmark() {
    local cmd_line
    cmd_line=$(build_cmd "$@") || return 1

    if [ "$DRY_RUN" = "true" ]; then
        echo "[DRY-RUN] $cmd_line" >&2
    else
        # shellcheck disable=SC2086
        eval "$cmd_line"
    fi
}

# ──────────────────────────────────────────────
# Generate sweep parameter combinations
# ──────────────────────────────────────────────
sweep_combinations() {
    local ngpu threads batch ubatch fa ctk ctv nprompt ngen

    for ngpu in "${SWEEP_NGPU[@]}"; do
        for threads in "${SWEEP_THREADS[@]}"; do
            for batch in "${SWEEP_BATCH[@]}"; do
                for ubatch in "${SWEEP_UBATCH[@]}"; do
                    for fa_v in "${SWEEP_FA[@]}"; do
                        for ctk_v in "${SWEEP_CTK[@]}"; do
                            for ctv_v in "${SWEEP_CTV[@]}"; do
                                for nprompt_v in "${SWEEP_N_PROMPT[@]}"; do
                                    for ngen_v in "${SWEEP_N_GEN[@]}"; do
                                        echo "-ngl $ngpu -t $threads -b $batch -ub $ubatch -fa $fa_v -ctk $ctk_v -ctv $ctv_v -p $nprompt_v -n $ngen_v"
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
}

# ──────────────────────────────────────────────
# Single-model benchmark
# ──────────────────────────────────────────────
benchmark_single() {
    local model_path="$1"
    local display_name
    display_name=$(model_display_name "$model_path")

    echo "[INFO] Benchmarking: $display_name" >&2

    # Use default parameters if not overriding
    local -a params=(
        -ngl "$DEFAULT_NGPU"
        -t "$DEFAULT_THREADS"
        -b "$DEFAULT_BATCH"
        -ub "$DEFAULT_UBATCH"
        -fa "$DEFAULT_FA"
        -ctk "$DEFAULT_CTK"
        -ctv "$DEFAULT_CTV"
        -p "$DEFAULT_N_PROMPT"
        -n "$DEFAULT_N_GEN"
        -r "$DEFAULT_REPETITIONS"
    )

    run_benchmark "$model_path" "${params[@]}"
}

# ──────────────────────────────────────────────
# Full parameter sweep for a model
# ──────────────────────────────────────────────
benchmark_sweep() {
    local model_path="$1"
    local display_name
    display_name=$(model_display_name "$model_path")

    echo "[INFO] Sweep: $display_name" >&2

    local combo
    while IFS= read -r combo; do
        [ -z "$combo" ] && continue
        if [ "$DRY_RUN" = "true" ]; then
            echo "[DRY-RUN] llama-bench -m $model_path -o json $combo -r $DEFAULT_REPETITIONS" >&2
        else
            # shellcheck disable=SC2086
            eval "llama-bench -m \"$model_path\" -o json $combo -r $DEFAULT_REPETITIONS"
        fi
    done < <(sweep_combinations)
}

# ──────────────────────────────────────────────
# List discovered models with metadata
# ──────────────────────────────────────────────
list_models() {
    local models
    mapfile -t models < <(discover_models)

    if [ ${#models[@]} -eq 0 ]; then
        echo "No GGUF models found in HF cache or llama.cpp cache." >&2
        return 0
    fi

    local model
    for model in "${models[@]}"; do
        local name
        name=$(model_display_name "$model")
        local size
        size=$(stat -c%s "$model" 2>/dev/null || echo 0)
        local size_hr
        size_hr=$(numfmt --to=iec-i "$size" 2>/dev/null || echo "${size}B")
        echo "$name  ($size_hr)  $model"
    done
}

# ──────────────────────────────────────────────
# Output models as JSON
# ──────────────────────────────────────────────
list_models_json() {
    local models
    mapfile -t models < <(discover_models)

    echo "["
    local first=true
    local model
    for model in "${models[@]}"; do
        $first || echo ","
        first=false
        local name
        name=$(model_display_name "$model")
        local size
        size=$(stat -c%s "$model" 2>/dev/null || echo 0)
        echo "  {"
        echo "    \"name\": \"$name\","
        echo "    \"path\": \"$model\","
        echo "    \"size_bytes\": $size"
        echo -n "  }"
    done
    echo ""
    echo "]"
}

# ──────────────────────────────────────────────
# Print JSON results from a directory of benchmark outputs
# ──────────────────────────────────────────────
aggregate_results() {
    local results_dir="$1"

    echo "{"
    echo "  \"system\": \"Strix Halo (AMD Ryzen AI MAX+ 395, gfx1151)\","
    echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"results\": ["

    local first=true
    local result_file
    for result_file in "$results_dir"/*.json; do
        [ -f "$result_file" ] || continue
        $first || echo ","
        first=false
        cat "$result_file"
    done

    echo ""
    echo "  ]"
    echo "}"
}

# ──────────────────────────────────────────────
# Usage / help
# ──────────────────────────────────────────────
usage() {
    cat >&2 <<'EOF'
llama-bench.sh — Benchmark wrapper for llama.cpp's llama-bench

USAGE:
  llama-bench.sh [OPTIONS]

OPTIONS:
  -h, --help              Show this help message

  --all                   Run full parameter sweep across all discovered models
  --model <name>          Benchmark a specific model (matches by name substring)
  --sweep                 Run full sweep on the specified model (default:
                          single-run with optimal defaults)

  --dry-run               Print commands without executing them (stderr)
  --output <format>       Output format (default: json)
  --list-models           List discovered models and exit
  --list-models-json      List discovered models as JSON

  --ngl <n>               Number of GPU layers (default: 99)
  --threads <n>           Number of CPU threads (default: 16)
  --batch <n>             Batch size (default: 2048)
  --ubatch <n>            Micro batch size (default: 512)
  --fa <0|1>              Flash attention (default: 1)
  --ctk <type>            KV cache type K (default: f16)
  --ctv <type>            KV cache type V (default: f16)
  --prompt <n>            Prompt length (default: 512)
  --gen <n>               Generation length (default: 128)
  --repetitions <n>       Number of repetitions per test (default: 3)

EXAMPLES:
  # Quick benchmark with defaults
  llama-bench.sh --model gemma

  # Full parameter sweep
  llama-bench.sh --sweep --model gemma

  # Benchmark all discovered models (single run each)
  llama-bench.sh --all

  # Full sweep on all models
  llama-bench.sh --all --sweep

  # Preview commands without running
  llama-bench.sh --dry-run --all

  # List discovered models
  llama-bench.sh --list-models

  # List models as JSON
  llama-bench.sh --list-models-json

NOTES:
  - Automatically sets LD_LIBRARY_PATH to include ~/llama.cpp/build/bin/
  - Multi-file GGUF models use the first split file (llama-bench handles them)
  - Incomplete downloads are skipped with a warning
  - Default parameters are tuned for Strix Halo (16 cores, gfx1151 GPU)
EOF
}

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
main() {
    # Parse options
    local mode="single"            # single, sweep, list, list-json
    local target_model=""
    local output_format="json"
    DRY_RUN="${DRY_RUN:-false}"

    # Override detection defaults
    HF_HUB_CACHE="${MOCK_HF_CACHE_DIR:-$HF_HUB_CACHE}"
    LLAMACPP_CACHE="${MOCK_LLAMACPP_CACHE_DIR:-$LLAMACPP_CACHE}"

    # Allow DRY_RUN to be set via environment
    [ "${DRY_RUN:-false}" = "true" ] && DRY_RUN=true

    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            --all)
                mode="sweep"
                shift
                ;;
            --sweep)
                mode="sweep"
                shift
                ;;
            --model)
                if [ $# -lt 2 ]; then
                    echo "ERROR: --model requires a model name" >&2
                    exit 1
                fi
                target_model="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --output)
                if [ $# -lt 2 ]; then
                    echo "ERROR: --output requires a format" >&2
                    exit 1
                fi
                output_format="$2"
                shift 2
                ;;
            --list-models)
                mode="list"
                shift
                ;;
            --list-models-json)
                mode="list-json"
                shift
                ;;
            --ngl)
                shift 2
                ;;
            --threads)
                shift 2
                ;;
            --batch)
                shift 2
                ;;
            --ubatch)
                shift 2
                ;;
            --fa)
                shift 2
                ;;
            --ctk)
                shift 2
                ;;
            --ctv)
                shift 2
                ;;
            --prompt)
                shift 2
                ;;
            --gen)
                shift 2
                ;;
            --repetitions)
                shift 2
                ;;
            *)
                echo "ERROR: Unknown option: $1" >&2
                usage
                exit 1
                ;;
        esac
    done

    # Fix library paths so llama-bench can run
    fix_library_path

    # Verify llama-bench is available
    if ! find_llama_bench >/dev/null 2>&1; then
        echo "ERROR: llama-bench not found at $LLAMA_BENCH or in PATH" >&2
        echo "  Install: cd ~/llama.cpp && cmake --build build --config Release -j\$(nproc)" >&2
        echo "  Or set LLAMA_BENCH_PATH to the llama-bench binary location" >&2
        exit 1
    fi

    case "$mode" in
        list)
            list_models
            exit 0
            ;;
        list-json)
            list_models_json
            exit 0
            ;;
        single|sweep)
            # Discover models
            local models
            mapfile -t models < <(discover_models)

            if [ ${#models[@]} -eq 0 ]; then
                echo "No GGUF models found." >&2
                exit 1
            fi

            # Filter by target model if specified
            local -a target_models=()
            if [ -n "$target_model" ]; then
                local model
                for model in "${models[@]}"; do
                    local name
                    name=$(model_display_name "$model")
                    if echo "$name" | grep -qi "$target_model"; then
                        target_models+=("$model")
                    fi
                done
                if [ ${#target_models[@]} -eq 0 ]; then
                    echo "ERROR: No models matching '$target_model' found." >&2
                    echo "Available models:" >&2
                    local m
                    for m in "${models[@]}"; do
                        echo "  - $(model_display_name "$m")" >&2
                    done
                    exit 1
                fi
                models=("${target_models[@]}")
            fi

            # Run benchmarks
            if [ "$mode" = "sweep" ]; then
                local md
                for md in "${models[@]}"; do
                    benchmark_sweep "$md"
                done
            else
                local md
                for md in "${models[@]}"; do
                    benchmark_single "$md"
                done
            fi
            ;;
        *)
            echo "ERROR: Unknown mode: $mode" >&2
            exit 1
            ;;
    esac
}

main "$@"
