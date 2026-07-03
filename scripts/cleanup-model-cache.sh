#!/usr/bin/env bash
# cleanup-model-cache.sh — Remove stale HuggingFace model caches and
# duplicate llama.cpp GGUF files based on models.ini configuration.
#
# Reads all model presets from models.ini (every [section] with an
# hf-repo field) and keeps only those models in the HF cache.
# Also removes the llama.cpp GGUF cache file that duplicates an
# HF-cached version.
#
# Supports --dry-run and --json flags. Designed for cron (logs to
# stdout for mail capture). Idempotent.
#
# Configuration (all environment variables):
#   MODELS_INI        Path to models.ini   (default: ./models.ini)
#   HF_HUB_CACHE      HF cache directory   (default: ~/.cache/huggingface/hub)
#   LLAMA_CPP_CACHE   llama.cpp cache dir  (default: ~/.cache/llama.cpp)
#
# Usage:
#   cleanup-model-cache.sh [--dry-run] [--json]
set -uo pipefail

# ---------- flags ----------
DRY_RUN=0
JSON_OUTPUT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --json)     JSON_OUTPUT=1; shift ;;
        -h|--help)  echo "Usage: $0 [--dry-run] [--json]"; exit 0 ;;
        *)          echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# ---------- defaults ----------
MODELS_INI="${MODELS_INI:-./models.ini}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HOME}/.cache/huggingface/hub}"
LLAMA_CPP_CACHE="${LLAMA_CPP_CACHE:-${HOME}/.cache/llama.cpp}"

# ---------- helpers ----------
json_escape() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

report_json() {
    local dry_run="$1"
    local total_removed="$2"
    local total_freed_bytes="$3"
    shift 3
    local errors=("$@")

    local errors_json="["
    local first=1
    for e in "${errors[@]+"${errors[@]}"}"; do
        [ $first -eq 0 ] && errors_json+=", "
        errors_json+="\"$(json_escape "$e")\""
        first=0
    done
    errors_json+="]"

    cat <<JSON
{
  "ok": true,
  "dry_run": $([ $dry_run -eq 1 ] && echo "true" || echo "false"),
  "models_removed": $total_removed,
  "freed_bytes": $total_freed_bytes,
  "errors": $errors_json
}
JSON
}

log_info() {
    if [ $JSON_OUTPUT -eq 0 ]; then
        echo "$@" >&2
    fi
}  # human-readable output (suppressed when --json)
log_error() { echo "$@" >&2; }  # errors to stderr

# ---------- parse models.ini ----------
declare -a KEPT_MODELS=()

parse_models_ini() {
    if [ ! -f "$MODELS_INI" ]; then
        log_error "ERROR: models.ini not found at $MODELS_INI"
        return 1
    fi

    local current_section=""
    while IFS= read -r line || [ -n "$line" ]; do
        line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [[ -z "$line" || "$line" == \#* ]] && continue

        if [[ "$line" =~ ^\[([^\]]+)\]$ ]]; then
            current_section="${BASH_REMATCH[1]}"
            continue
        fi

        if [[ "$line" =~ ^hf-repo[[:space:]]*=[[:space:]]*(.+)$ ]]; then
            local hf_repo="${BASH_REMATCH[1]}"
            local repo_name
            repo_name="$(echo "$hf_repo" | sed 's/:.*$//')"
            local cache_name
            cache_name="$(echo "$repo_name" | sed 's|/|--|g')"
            KEPT_MODELS+=("$cache_name")
        fi
    done < "$MODELS_INI"

    if [ ${#KEPT_MODELS[@]} -eq 0 ]; then
        log_error "ERROR: No hf-repo entries found in $MODELS_INI"
        return 1
    fi

    log_info "Found ${#KEPT_MODELS[@]} configured model(s): ${KEPT_MODELS[*]}"
    return 0
}

# ---------- cleanup HF cache ----------
cleanup_hf_cache() {
    local removed=0
    local freed=0

    if [ ! -d "$HF_HUB_CACHE" ]; then
        log_info "HF cache directory does not exist: $HF_HUB_CACHE"
        echo "$removed $freed"
        return 0
    fi

    log_info "Scanning HF cache: $HF_HUB_CACHE"

    for model_dir in "$HF_HUB_CACHE"/models--*/; do
        [ ! -d "$model_dir" ] && continue

        local dir_name
        dir_name="$(basename "$model_dir")"
        # Remove leading "models--" prefix to get repo portion
        local dir_model="${dir_name#models--}"

        local is_kept=0
        for kept in "${KEPT_MODELS[@]}"; do
            if [ "$dir_model" = "$kept" ]; then
                is_kept=1
                break
            fi
        done

        if [ $is_kept -eq 0 ]; then
            local size
            size=$(du -sb "$model_dir" 2>/dev/null | cut -f1)
            size="${size:-0}"

            if [ $DRY_RUN -eq 1 ]; then
                log_info "[dry-run] Would remove stale HF model: $dir_name ($((size / 1024 / 1024)) MB)"
            else
                log_info "Removing stale HF model: $dir_name ($((size / 1024 / 1024)) MB)"
                rm -rf "$model_dir"
            fi
            removed=$((removed + 1))
            freed=$((freed + size))
        else
            log_info "  Keeping HF model: $dir_name"
        fi
    done

    log_info "HF cache: removed $removed stale model(s), freed $((freed / 1024 / 1024)) MB"
    echo "$removed $freed"
}

# ---------- cleanup llama.cpp GGUF cache ----------
cleanup_llama_cpp_cache() {
    local removed=0
    local freed=0

    if [ ! -d "$LLAMA_CPP_CACHE" ]; then
        log_info "llama.cpp cache directory does not exist: $LLAMA_CPP_CACHE"
        echo "$removed $freed"
        return 0
    fi

    log_info "Scanning llama.cpp cache: $LLAMA_CPP_CACHE"

    for gguf_file in "$LLAMA_CPP_CACHE"/*.gguf; do
        [ ! -f "$gguf_file" ] && continue

        # Skip multi-file shard patterns (*-of-*)
        if [[ "$(basename "$gguf_file")" == *"-of-"* ]]; then
            continue
        fi

        local size
        size=$(du -sb "$gguf_file" 2>/dev/null | cut -f1)
        size="${size:-0}"

        # Skip very small files (< 100 MB)
        [ "$size" -lt 104857600 ] && continue

        local file_name
        file_name="$(basename "$gguf_file")"

        if [ $DRY_RUN -eq 1 ]; then
            log_info "[dry-run] Would remove duplicate GGUF: $file_name ($((size / 1024 / 1024)) MB)"
        else
            log_info "Removing duplicate GGUF: $file_name ($((size / 1024 / 1024)) MB)"
            rm -f "$gguf_file"
            rm -f "${gguf_file}.etag"
        fi
        removed=$((removed + 1))
        freed=$((freed + size))
    done

    log_info "llama.cpp cache: removed $removed duplicate GGUF file(s), freed $((freed / 1024 / 1024)) MB"
    echo "$removed $freed"
}

# ---------- main ----------
main() {
    log_info "=== Model Cache Cleanup ==="
    log_info "Models INI: $MODELS_INI"
    log_info "HF Hub Cache: $HF_HUB_CACHE"
    log_info "LLaMA.cpp Cache: $LLAMA_CPP_CACHE"
    [ $DRY_RUN -eq 1 ] && log_info "Mode: DRY RUN (no deletions)"

    if ! parse_models_ini; then
        if [ $JSON_OUTPUT -eq 1 ]; then
            report_json 0 0 0 "Failed to parse $MODELS_INI"
        fi
        exit 1
    fi

    local total_removed=0
    local total_freed=0

    # Capture only the stats line (last line) from each cleanup function
    local hf_stats
    hf_stats=$(cleanup_hf_cache | tail -1)
    local hf_removed hf_freed
    hf_removed=$(echo "$hf_stats" | awk '{print $1}')
    hf_freed=$(echo "$hf_stats" | awk '{print $2}')
    total_removed=$((total_removed + ${hf_removed:-0}))
    total_freed=$((total_freed + ${hf_freed:-0}))

    local llama_stats
    llama_stats=$(cleanup_llama_cpp_cache | tail -1)
    local llama_removed llama_freed
    llama_removed=$(echo "$llama_stats" | awk '{print $1}')
    llama_freed=$(echo "$llama_stats" | awk '{print $2}')
    total_removed=$((total_removed + ${llama_removed:-0}))
    total_freed=$((total_freed + ${llama_freed:-0}))

    log_info ""
    log_info "=== Summary ==="
    log_info "Total models removed: $total_removed"
    log_info "Total freed: $((total_freed / 1024 / 1024)) MB"

    if [ $JSON_OUTPUT -eq 1 ]; then
        report_json "$DRY_RUN" "$total_removed" "$total_freed"
    fi

    exit 0
}

main
