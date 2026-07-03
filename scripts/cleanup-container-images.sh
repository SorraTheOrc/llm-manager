#!/usr/bin/env bash
# cleanup-container-images.sh — Prune container images not needed by
# the project scripts and Containerfile.
#
# Keeps images referenced by project scripts and Containerfile, plus
# system and cross-project images. Removes dangling <none> images,
# unreferenced build variants, and images not used by the LLM stack.
#
# Supports --dry-run and --json flags. Designed for cron. Idempotent.
#
# Image retention policy:
#   ALWAYS KEPT:
#     - localhost/llm-llama:local          (referenced by podman_start_llama.sh)
#     - rocm/dev-ubuntu-24.04:7.2.4        (base image in Containerfile)
#     - localhost/llama-rocm:gfx1151       (same image, different tag)
#     - fedora-toolbox:latest              (system tool)
#     - alpine:latest                      (system tool)
#
#   KEPT (cross-project, default):
#     - postgres:*   pgvector:*   qdrant:*   milvus:*   debian:*
#     Set LLM_CLEANUP_ALL=1 to include these in removal scope.
#
#   REMOVED:
#     - <none>:<none> (dangling images)
#     - localhost/ampa-template:*
#     - localhost/ampa-dev:*
#     - localhost/llama-rocm:pre-7.2.4
#     - docker.io/kyuz0/amd-strix-halo-toolboxes:*
#
# Usage:
#   cleanup-container-images.sh [--dry-run] [--json] [--mock]
set -uo pipefail

# ---------- flags ----------
DRY_RUN=0
JSON_OUTPUT=0
MOCK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --json)     JSON_OUTPUT=1; shift ;;
        --mock)     MOCK=1; shift ;;  # for testing only
        -h|--help)  echo "Usage: $0 [--dry-run] [--json]"; exit 0 ;;
        *)          echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# ---------- helpers ----------
log_info() {
    if [ $JSON_OUTPUT -eq 0 ]; then
        echo "$@" >&2
    fi
}
log_error() { echo "$@" >&2; }

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
  "images_removed": $total_removed,
  "freed_bytes": $total_freed_bytes,
  "errors": $errors_json
}
JSON
}

# ---------- image classification ----------
# Images that are always kept (LLM stack, system tools)
ALWAYS_KEPT_REPOS=(
    "localhost/llm-llama"
    "docker.io/rocm/dev-ubuntu-24.04"
    "rocm/dev-ubuntu-24.04"
    "localhost/llama-rocm"
    "registry.fedoraproject.org/fedora-toolbox"
    "fedora-toolbox"
    "docker.io/library/alpine"
    "docker.io/library/debian"
)

# Images kept by default (cross-project); removable with LLM_CLEANUP_ALL=1
CROSS_PROJECT_REPOS=(
    "docker.io/library/postgres"
    "docker.io/pgvector"
    "docker.io/qdrant"
    "docker.io/milvusdb"
)

# Image patterns that are always removed (unreferenced build artifacts)
REMOVAL_PATTERNS=(
    "localhost/ampa-template"
    "localhost/ampa-dev"
    "docker.io/kyuz0"
)

# ---------- podman command ----------
run_podman() {
    if [ $MOCK -eq 1 ]; then
        case "$1" in
            images)
                # Return a mock image list for testing
                cat <<'MOCK'
REPOSITORY:TAG IMAGEID SIZE
localhost/llm-llama:local abc123 11.4GB
<none>:<none> def456 1.8GB
localhost/llama-rocm:pre-7.2.4 ghi789 17.8GB
localhost/llama-rocm:gfx1151 f404258 3.92GB
docker.io/rocm/dev-ubuntu-24.04:7.2.4 f404258 3.92GB
docker.io/library/alpine:latest jkl012 8.71MB
registry.fedoraproject.org/fedora-toolbox:latest mno345 2.09GB
localhost/ampa-template:2026-03-26 pqr678 1.83GB
localhost/ampa-dev:latest stu901 1.66GB
docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7rc-rocwmma vwx234 15.4GB
docker.io/library/postgres:15 yza345 412MB
docker.io/pgvector/pgvector:pg15 bcd456 611MB
docker.io/qdrant/qdrant:v1.13.4 efg567 279MB
docker.io/milvusdb/milvus:v2.3.0 hij678 1.83GB
docker.io/library/debian:stable-slim klm789 124MB
MOCK
                ;;
            rmi)
                if [ $DRY_RUN -eq 1 ]; then
                    echo "[mock] would rmi $2"
                else
                    echo "[mock] rmi $2"
                fi
                return 0
                ;;
            *) return 0 ;;
        esac
    fi

    if command -v podman >/dev/null 2>&1; then
        podman "$@"
    elif command -v docker >/dev/null 2>&1; then
        docker "$@"
    else
        log_error "ERROR: Neither podman nor docker found"
        return 1
    fi
}

# ---------- check if an image repository is in a keep list ----------
repo_is_kept() {
    local repo="$1"
    shift
    local list=("$@")
    for kept in "${list[@]}"; do
        if [[ "$repo" == "$kept" ]] || [[ "$repo" == "$kept"/* ]]; then
            return 0
        fi
    done
    return 1
}

# ---------- cleanup container images ----------
cleanup_images() {
    local removed=0
    local freed=0
    declare -a errors_list=()
    declare -a removed_entries=()

    # Determine if we clean cross-project images too
    local include_all="${LLM_CLEANUP_ALL:-0}"

    log_info "=== Container Image Cleanup ==="
    log_info "LLM_CLEANUP_ALL=${include_all}"
    [ $DRY_RUN -eq 1 ] && log_info "Mode: DRY RUN (no deletions)"

    # Get all images with parsing-friendly output format
    local image_data
    if ! image_data=$(run_podman images --all --format "{{.Repository}}:{{.Tag}}|{{.ID}}|{{.Size}}" 2>/dev/null); then
        log_error "ERROR: Failed to list images"
        echo "0 0 'Failed to list images'"
        return 1
    fi

    # Check if we have mock output (not pipe-separated) - handle mock format
    if echo "$image_data" | head -1 | grep -q "^REPOSITORY:TAG"; then
        # Mock format: skip header and use awk
        image_data=$(echo "$image_data" | tail -n +2 | awk '{print $1"|"$2"|"$3}')
    fi

    while IFS='|' read -r repo_tag image_id size_str; do
        [ -z "$repo_tag" ] && continue

        # Skip the header line if it slipped through
        [ "$repo_tag" = "REPOSITORY:TAG" ] && continue

        # Determine the image repository (without tag)
        local repo="${repo_tag%:*}"
        local tag="${repo_tag##*:}"
        # Handle cases where tag might contain colon (e.g., docker.io/rocm/dev-ubuntu-24.04:7.2.4)
        if [ "$repo" = "$repo_tag" ]; then
            # No colon separator means no tag
            repo="$repo_tag"
            tag=""
        fi

        # Check against removal patterns first
        local should_remove=0
        for pattern in "${REMOVAL_PATTERNS[@]}"; do
            if [[ "$repo" == "$pattern" ]] || [[ "$repo" == "$pattern"/* ]]; then
                should_remove=1
                break
            fi
        done

        # Check if it's a dangling image
        if [ "$repo" = "<none>" ]; then
            should_remove=1
        fi

        # Check if it's a kept image
        if repo_is_kept "$repo" "${ALWAYS_KEPT_REPOS[@]}"; then
            should_remove=0
        fi

        # Check cross-project kept list
        if repo_is_kept "$repo" "${CROSS_PROJECT_REPOS[@]}"; then
            if [ "$include_all" -ne 1 ]; then
                should_remove=0  # Keep by default
            else
                should_remove=1  # Remove if LLM_CLEANUP_ALL=1
            fi
        fi

        # Determine size
        local size_bytes=0
        if [[ "$size_str" == *GB ]]; then
            local num
            num=$(echo "$size_str" | sed 's/GB//' | tr -d ' ')
            size_bytes=$(echo "$num * 1073741824" | bc 2>/dev/null | cut -d. -f1)
            [ -z "$size_bytes" ] && size_bytes=0
        elif [[ "$size_str" == *MB ]]; then
            local num
            num=$(echo "$size_str" | sed 's/MB//' | tr -d ' ')
            size_bytes=$(echo "$num * 1048576" | bc 2>/dev/null | cut -d. -f1)
            [ -z "$size_bytes" ] && size_bytes=0
        fi

        # Also compute from du if available and real
        if [ $MOCK -eq 0 ]; then
            local real_size
            real_size=$(run_podman image inspect "$image_id" --format '{{.Size}}' 2>/dev/null || echo "0")
            size_bytes="${real_size:-0}"
        fi

        if [ $should_remove -eq 1 ]; then
            if [ $DRY_RUN -eq 1 ]; then
                log_info "[dry-run] Would remove: ${repo}:${tag} (${size_str})"
            else
                log_info "Removing: ${repo}:${tag} (${size_str})"
                if ! run_podman rmi "$image_id" 2>/dev/null; then
                    log_info "  (could not remove ${repo}:${tag} - may be in use)"
                fi
            fi
            removed=$((removed + 1))
            freed=$((freed + size_bytes))
        else
            log_info "  Keeping: ${repo}:${tag}"
        fi
    done <<< "$image_data"

    log_info ""
    log_info "=== Summary ==="
    log_info "Images removed: $removed"
    log_info "Total freed: $((freed / 1024 / 1024)) MB"

    echo "$removed $freed"
}

# ---------- main ----------
main() {
    local stats
    stats=$(cleanup_images)
    local removed freed
    removed=$(echo "$stats" | tail -1 | awk '{print $1}')
    freed=$(echo "$stats" | tail -1 | awk '{print $2}')

    if [ $JSON_OUTPUT -eq 1 ]; then
        report_json "$DRY_RUN" "${removed:-0}" "${freed:-0}"
    fi

    exit 0
}

main
