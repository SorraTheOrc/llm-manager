#!/usr/bin/env bash
# cleanup-all.sh — Run all cleanup scripts in sequence.
#
# Orchestrator that invokes cleanup-model-cache.sh, cleanup-container-images.sh,
# cleanup-stopped-containers.sh, and cleanup-pi-sessions.sh with consistent flags.
# Designed for cron.
#
# Supports --dry-run and --json flags which propagate to all child scripts.
#
# Container cleanup order matters: stopped containers are removed before image
# cleanup because stopped containers hold references to images, preventing their
# removal.
#
# Configuration (all environment variables):
#   CLEANUP_MODEL_CACHE           Path to model cache script (default: scripts/cleanup-model-cache.sh)
#   CLEANUP_CONTAINER_IMAGES      Path to container image script (default: scripts/cleanup-container-images.sh)
#   CLEANUP_STOPPED_CONTAINERS    Path to stopped container script (default: scripts/cleanup-stopped-containers.sh)
#   CLEANUP_PI_SESSIONS           Path to pi sessions script (default: scripts/cleanup-pi-sessions.sh)
#   CLEANUP_ENABLE_MODELS         Set to 0 to skip model cache cleanup (default: 1)
#   CLEANUP_ENABLE_CONTAINERS     Set to 0 to skip container cleanup         (default: 1)
#   CLEANUP_ENABLE_STOPPED_CONTAINERS  Set to 0 to skip stopped container cleanup (default: 1)
#   CLEANUP_ENABLE_PI             Set to 0 to skip pi session cleanup        (default: 1)
#
# Usage:
#   cleanup-all.sh [--dry-run] [--json]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLEANUP_MODEL_CACHE="${CLEANUP_MODEL_CACHE:-${SCRIPT_DIR}/cleanup-model-cache.sh}"
CLEANUP_CONTAINER_IMAGES="${CLEANUP_CONTAINER_IMAGES:-${SCRIPT_DIR}/cleanup-container-images.sh}"
CLEANUP_STOPPED_CONTAINERS="${CLEANUP_STOPPED_CONTAINERS:-${SCRIPT_DIR}/cleanup-stopped-containers.sh}"
CLEANUP_PI_SESSIONS="${CLEANUP_PI_SESSIONS:-${SCRIPT_DIR}/cleanup-pi-sessions.sh}"

CLEANUP_ENABLE_MODELS="${CLEANUP_ENABLE_MODELS:-1}"
CLEANUP_ENABLE_CONTAINERS="${CLEANUP_ENABLE_CONTAINERS:-1}"
CLEANUP_ENABLE_PI="${CLEANUP_ENABLE_PI:-1}"

DRY_RUN=0
JSON_OUTPUT=0

# ---------- flags ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --json)     JSON_OUTPUT=1; shift ;;
        -h|--help)  echo "Usage: $0 [--dry-run] [--json]"; exit 0 ;;
        *)          echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# ---------- helpers ----------
log_info() {
    if [ $JSON_OUTPUT -eq 0 ]; then
        echo "$@"
    fi
}
log_error() { echo "$@" >&2; }

json_escape() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

run_script() {
    local name="$1"
    local script_path="$2"
    local enabled="$3"
    shift 3

    if [ "$enabled" != "1" ]; then
        if [ $JSON_OUTPUT -eq 1 ]; then
            cat <<JSON
{"name":"$(json_escape "$name")","ok":true,"dry_run":$([ $DRY_RUN -eq 1 ] && echo "true" || echo "false"),"skipped":true}
JSON
        fi
        log_info "[SKIP] $name is disabled (CLEANUP_ENABLE flag not set to 1)"
        return 0
    fi

    if [ ! -x "$script_path" ]; then
        if [ $JSON_OUTPUT -eq 1 ]; then
            cat <<JSON
{"name":"$(json_escape "$name")","ok":false,"error":"script not found or not executable: $(json_escape "$script_path")"}
JSON
        fi
        log_error "ERROR: $name script not found or not executable: $script_path"
        return 2
    fi

    log_info "=== Running $name ==="

    local extra_flags=()
    [ $DRY_RUN -eq 1 ] && extra_flags+=("--dry-run")
    [ $JSON_OUTPUT -eq 1 ] && extra_flags+=("--json")

    if [ $JSON_OUTPUT -eq 1 ]; then
        # Capture JSON output
        local output
        output=$("$script_path" "${extra_flags[@]}" 2>/dev/null)
        local rc=$?
        echo "$output"
        return $rc
    else
        "$script_path" "${extra_flags[@]}"
        local rc=$?
        echo ""  # spacing between scripts
        return $rc
    fi
}

# ---------- main ----------
main() {
    local exit_code=0
    local errors=0
    local results_json="["
    local first_result=1

    log_info "==========================================="
    log_info "  LLM Stack Cleanup Orchestrator"
    log_info "==========================================="
    log_info ""
    [ $DRY_RUN -eq 1 ] && log_info "Mode: DRY RUN (no files will be deleted)"
    log_info ""

    # For JSON output, collect results from child scripts
    local json_buffer=""

    # 1. Model cache cleanup
    local model_output model_rc
    model_output=$(run_script "Model Cache Cleanup" "$CLEANUP_MODEL_CACHE" "$CLEANUP_ENABLE_MODELS" 2>/dev/null)
    model_rc=$?
    if [ $model_rc -ne 0 ]; then
        log_error "Model cache cleanup failed"
        errors=$((errors + 1))
        exit_code=1
    fi
    if [ $JSON_OUTPUT -eq 1 ]; then
        json_buffer="${json_buffer}${model_output}"$'\n'
    fi

    # 2. Stopped container cleanup (runs before image cleanup to release
    #    image references held by stopped containers)
    local stopped_output stopped_rc
    stopped_output=$(run_script "Stopped Container Cleanup" "$CLEANUP_STOPPED_CONTAINERS" "$CLEANUP_ENABLE_CONTAINERS" 2>/dev/null)
    stopped_rc=$?
    if [ $stopped_rc -ne 0 ]; then
        log_error "Stopped container cleanup failed"
        errors=$((errors + 1))
        exit_code=1
    fi
    if [ $JSON_OUTPUT -eq 1 ]; then
        json_buffer="${json_buffer}${stopped_output}"$'\n'
    fi

    # 3. Container image cleanup
    local container_output container_rc
    container_output=$(run_script "Container Image Cleanup" "$CLEANUP_CONTAINER_IMAGES" "$CLEANUP_ENABLE_CONTAINERS" 2>/dev/null)
    container_rc=$?
    if [ $container_rc -ne 0 ]; then
        log_error "Container image cleanup failed"
        errors=$((errors + 1))
        exit_code=1
    fi
    if [ $JSON_OUTPUT -eq 1 ]; then
        json_buffer="${json_buffer}${container_output}"$'\n'
    fi

    # 4. Pi session log cleanup
    local pi_output pi_rc
    pi_output=$(run_script "Pi Session Log Cleanup" "$CLEANUP_PI_SESSIONS" "$CLEANUP_ENABLE_PI" 2>/dev/null)
    pi_rc=$?
    if [ $pi_rc -ne 0 ]; then
        log_error "Pi session log cleanup failed"
        errors=$((errors + 1))
        exit_code=1
    fi
    if [ $JSON_OUTPUT -eq 1 ]; then
        json_buffer="${json_buffer}${pi_output}"$'\n'
    fi

    # Collect into a results array for JSON output
    if [ $JSON_OUTPUT -eq 1 ]; then
        local first=1
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            [ $first -eq 0 ] && results_json+=", "
            results_json+="$line"
            first=0
        done <<< "$json_buffer"
        results_json+="]"

        cat <<JSON
{
  "ok": true,
  "dry_run": $([ $DRY_RUN -eq 1 ] && echo "true" || echo "false"),
  "errors": $errors,
  "results": $results_json
}
JSON
    fi

    log_info "==========================================="
    if [ $errors -eq 0 ]; then
        log_info "  All cleanup tasks completed successfully"
    else
        log_info "  $errors cleanup task(s) failed"
    fi
    log_info "==========================================="

    exit $exit_code
}

main
