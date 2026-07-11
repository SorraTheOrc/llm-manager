#!/usr/bin/env bash
# cleanup-stopped-containers.sh — Remove exited containers to reclaim disk space.
#
# Enumerates all exited containers and removes those that have been stopped
# for longer than a configurable threshold (default: 7 days). Reports freed
# disk space before and after removal.
#
# Container removal policy:
#   - Only containers with status "Exited" are considered for removal.
#   - By default, containers exited for >= 7 days are removed.
#   - Volumes are kept by default (--volumes=false is passed to podman rm)
#     to avoid accidental data loss with orphaned volumes.
#   - The running container (if any) is never touched.
#
# Supports --dry-run and --json flags. Interactive by default (shows report
# then prompts for confirmation). Use --yes for non-interactive / cron use.
#
# Usage:
#   cleanup-stopped-containers.sh [--dry-run] [--json] [--yes] [--mock]
#                                [--max-age DAYS]
set -uo pipefail

# ---------- defaults ----------
DRY_RUN=0
JSON_OUTPUT=0
YES=0
MOCK=0
MAX_AGE_DAYS=7

# ---------- flags ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --json)      JSON_OUTPUT=1; shift ;;
        --yes)       YES=1; shift ;;
        --mock)      MOCK=1; shift ;;  # for testing only
        --max-age)   MAX_AGE_DAYS="$2"; shift 2 ;;
        -h|--help)   echo "Usage: $0 [--dry-run] [--json] [--yes] [--mock] [--max-age DAYS]"; exit 0 ;;
        *)           echo "Unknown option: $1" >&2; exit 2 ;;
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
    local containers_found="$2"
    local containers_removed="$3"
    local freed_bytes="$4"
    shift 4
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
  "containers_found": $containers_found,
  "containers_removed": $containers_removed,
  "freed_bytes": $freed_bytes,
  "errors": $errors_json
}
JSON
}

# ---------- date helpers ----------
# Convert a date string to epoch seconds. Handles both GNU date and BSD/macOS date.
date_to_epoch() {
    local date_str="$1"
    if date --version 2>/dev/null | grep -q GNU; then
        date -d "$date_str" +%s 2>/dev/null || echo 0
    else
        date -j -f "%Y-%m-%d %H:%M:%S" "$(echo "$date_str" | sed 's/\.[0-9]*//; s/ [+-].*//')" +%s 2>/dev/null || echo 0
    fi
}

now_epoch() {
    date +%s
}

# ---------- podman command ----------
run_podman() {
    if [ $MOCK -eq 1 ]; then
        case "$1" in
            ps)
                # Return mock exited container list for testing
                if [ "$2" = "-a" ] || [ "$2" = "--all" ]; then
                    cat <<'MOCK'
CONTAINER ID|NAMES|STATUS|IMAGE|CREATED AT
31816e15fbbd|sourcebase-postgres|Exited (0) 3 months ago|docker.io/pgvector/pgvector:pg15|2026-03-22 21:20:03
8354f117e74f|sourcebase-milvus|Exited (1) 3 months ago|docker.io/milvusdb/milvus:v2.3.0|2026-03-22 21:45:06
4fa27b5fbf21|sourcebase-qdrant|Exited (143) 3 months ago|docker.io/qdrant/qdrant:v1.13.4|2026-03-22 21:52:41
e0727efea564|ampa-template|Exited (143) 3 months ago|localhost/ampa-dev:latest|2026-03-26 14:55:47
2851ed3726e3|ampa-pool-1|Exited (143) 3 months ago|localhost/ampa-template:2026-03-26|2026-03-26 16:24:30
9d0074422796|ampa-pool-2|Exited (143) 3 months ago|localhost/ampa-template:2026-03-26|2026-03-26 16:24:55
795bbeb45d02|ampa-pool-0|Exited (143) 3 months ago|localhost/ampa-template:2026-03-26|2026-03-26 17:27:18
05900ec2d24c|llama|Up 3 hours|localhost/llm-llama:local|2026-06-29 15:12:38
MOCK
                else
                    cat <<'MOCK'
CONTAINER ID|NAMES|STATUS|IMAGE|CREATED AT
05900ec2d24c|llama|Up 3 hours|localhost/llm-llama:local|2026-06-29 15:12:38
MOCK
                fi
                ;;
            rm)
                if [ $DRY_RUN -eq 1 ]; then
                    echo "[mock] would rm $2" >&2
                else
                    echo "[mock] rm $2" >&2
                fi
                return 0
                ;;
            container)
                # podman container inspect --size ...
                if [ "$2" = "inspect" ]; then
                    echo '{"SizeRw":1048576}'
                    return 0
                fi
                return 0
                ;;
            *) return 0 ;;
        esac
        return 0
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

# ---------- get list of exited containers ----------
get_exited_containers() {
    local max_age_days="$1"
    local now_s
    now_s=$(now_epoch)
    local max_age_seconds=$((max_age_days * 86400))

    # List all containers with pipe-separated output
    local all_containers
    all_containers=$(run_podman ps -a --format "{{.ID}}|{{.Names}}|{{.Status}}|{{.Image}}|{{.CreatedAt}}" 2>/dev/null) || return 1

    declare -a candidates=()

    while IFS='|' read -r cid name status image created_at; do
        [ -z "$cid" ] && continue
        [ "$cid" = "CONTAINER ID" ] && continue  # skip header in mock mode

        # Only consider Exited containers
        if echo "$status" | grep -qv "^Exited"; then
            continue
        fi

        # Parse the created date to determine age
        local created_s=0
        # The date format from podman is typically: 2026-03-22 21:20:03 +0000 UTC
        # But in mock it's: 2026-03-22 21:20:03
        # Strip trailing timezone info for parsing
        local clean_date
        clean_date=$(echo "$created_at" | sed 's/ [+-][0-9]\{4\}.*$//; s/\.[0-9]*//' | xargs)
        if [ -n "$clean_date" ]; then
            created_s=$(date_to_epoch "$clean_date")
        fi

        local age_seconds=$((now_s - created_s))
        if [ "$age_seconds" -ge "$max_age_seconds" ] || [ "$created_s" -eq 0 ]; then
            candidates+=("$cid|$name|$image|$created_at|$age_seconds")
        fi
    done <<< "$all_containers"

    # Output candidates
    for c in "${candidates[@]+"${candidates[@]}"}"; do
        echo "$c"
    done
}

# ---------- get container size ----------
get_container_size() {
    local cid="$1"
    local size_str

    if [ $MOCK -eq 1 ]; then
        echo 1048576  # 1 MB mock size
        return 0
    fi

    # Try to get size via podman inspect
    size_str=$(run_podman container inspect --size "$cid" --format '{{.SizeRw}}' 2>/dev/null || echo "0")
    echo "${size_str:-0}"
}

# ---------- format bytes to human-readable ----------
format_bytes() {
    local bytes=$1
    if [ "$bytes" -ge 1073741824 ]; then
        echo "$(echo "scale=2; $bytes / 1073741824" | bc) GB"
    elif [ "$bytes" -ge 1048576 ]; then
        echo "$(echo "scale=2; $bytes / 1048576" | bc) MB"
    elif [ "$bytes" -ge 1024 ]; then
        echo "$(echo "scale=2; $bytes / 1024" | bc) KB"
    else
        echo "${bytes} B"
    fi
}

# ---------- confirm interactively ----------
confirm_removal() {
    # If --yes was passed, no confirmation needed
    if [ $YES -eq 1 ]; then
        return 0
    fi

    # If non-TTY, prompt for confirmation but provide a clear message
    if [ ! -t 0 ]; then
        log_error ""
        log_error "WARNING: stdin is not a terminal. Interactive confirmation requires a TTY."
        log_error "Use --yes to skip confirmation (safe for cron/automation)."
        log_error ""
        return 1
    fi

    echo "" >&2
    read -r -p "Remove these containers? [y/N] " response
    case "$response" in
        [yY]|[yY][eE][sS]) return 0 ;;
        *) return 1 ;;
    esac
}

# ---------- main ----------
main() {
    local exit_code=0
    declare -a errors_list=()
    local containers_found=0
    local containers_removed=0
    local total_freed_bytes=0

    # Report disk usage at start
    log_info "=== Stopped Container Cleanup ==="
    log_info "Max age: ${MAX_AGE_DAYS} days"
    [ $DRY_RUN -eq 1 ] && log_info "Mode: DRY RUN (no removals)"
    [ $YES -eq 1 ] && log_info "Mode: Non-interactive (--yes)"
    log_info ""

    # Gather disk usage before
    local disk_before
    disk_before=$(df -B1 / | tail -1 | awk '{print $4}' 2>/dev/null || echo "0")

    # Get stopped containers
    declare -a exited_containers=()
    while IFS='|' read -r cid name image created_at age; do
        [ -z "$cid" ] && continue
        exited_containers+=("$cid|$name|$image|$created_at|$age")
    done < <(get_exited_containers "$MAX_AGE_DAYS")

    containers_found=${#exited_containers[@]}

    if [ $containers_found -eq 0 ]; then
        log_info "No exited containers older than ${MAX_AGE_DAYS} days found."
        log_info ""

        local disk_after
        disk_after=$(df -B1 / | tail -1 | awk '{print $4}' 2>/dev/null || echo "0")

        if [ $JSON_OUTPUT -eq 1 ]; then
            report_json "$DRY_RUN" 0 0 0
        fi
        exit 0
    fi

    log_info "Found ${containers_found} exited container(s) older than ${MAX_AGE_DAYS} days:"
    log_info ""

    # Show table
    log_info "$(printf "%-20s %-30s %-50s %-12s" "CONTAINER ID" "NAME" "IMAGE" "SIZE")"
    log_info "$(printf "%-20s %-30s %-50s %-12s" "--------------------" "------------------------------" "--------------------------------------------------" "------------")"

    declare -a container_sizes=()
    for c in "${exited_containers[@]}"; do
        local cid cname cimage ccreated cage
        cid=$(echo "$c" | cut -d'|' -f1)
        cname=$(echo "$c" | cut -d'|' -f2)
        cimage=$(echo "$c" | cut -d'|' -f3)
        ccreated=$(echo "$c" | cut -d'|' -f4)
        cage=$(echo "$c" | cut -d'|' -f5)

        local csize
        csize=$(get_container_size "$cid")
        container_sizes+=("$cid|$csize")

        local human_size
        human_size=$(format_bytes "$csize")

        log_info "$(printf "%-20s %-30s %-50s %-12s" "${cid:0:12}" "$cname" "$cimage" "$human_size")"
    done

    log_info ""

    # Compute total reclaimable size
    local total_reclaimable=0
    for cs in "${container_sizes[@]}"; do
        local csize_cid csize_bytes
        csize_cid=$(echo "$cs" | cut -d'|' -f1)
        csize_bytes=$(echo "$cs" | cut -d'|' -f2)
        total_reclaimable=$((total_reclaimable + csize_bytes))
    done

    log_info "Total reclaimable space: $(format_bytes $total_reclaimable)"
    log_info ""

    # In dry-run mode, report what would be removed
    if [ $DRY_RUN -eq 1 ]; then
        log_info "[dry-run] Would remove ${containers_found} container(s) (approximately $(format_bytes $total_reclaimable))."
        log_info "[dry-run] Pass --yes to skip confirmation, or re-run without --dry-run to actually remove."

        if [ $JSON_OUTPUT -eq 1 ]; then
            report_json "$DRY_RUN" "$containers_found" 0 0
        fi
        exit 0
    fi

    # Interactive confirmation (unless --yes was passed)
    if ! confirm_removal; then
        log_info "Removal cancelled."
        if [ $JSON_OUTPUT -eq 1 ]; then
            report_json "$DRY_RUN" "$containers_found" 0 0
        fi
        exit 0
    fi

    # Remove containers
    log_info ""
    log_info "Removing containers..."

    for c in "${exited_containers[@]}"; do
        local cid cname
        cid=$(echo "$c" | cut -d'|' -f1)
        cname=$(echo "$c" | cut -d'|' -f2)

        log_info "  Removing ${cname} (${cid:0:12})..."
        if run_podman rm --volumes=false "$cid" 2>/dev/null; then
            containers_removed=$((containers_removed + 1))
            # Look up the size from our stored array
            for cs in "${container_sizes[@]}"; do
                local cs_cid cs_bytes
                cs_cid=$(echo "$cs" | cut -d'|' -f1)
                cs_bytes=$(echo "$cs" | cut -d'|' -f2)
                if [ "$cs_cid" = "$cid" ]; then
                    total_freed_bytes=$((total_freed_bytes + cs_bytes))
                    break
                fi
            done
        else
            log_error "  ERROR: Failed to remove container ${cname} (${cid:0:12})"
            errors_list+=("Failed to remove container ${cname} (${cid:0:12})")
            exit_code=1
        fi
    done

    # Gather disk usage after
    local disk_after
    disk_after=$(df -B1 / | tail -1 | awk '{print $4}' 2>/dev/null || echo "0")
    local disk_freed=$((disk_after - disk_before))
    if [ $disk_freed -lt 0 ]; then
        disk_freed=0
    fi

    log_info ""
    log_info "=== Summary ==="
    log_info "Containers removed: $containers_removed / $containers_found"
    log_info "Estimated freed: $(format_bytes $total_freed_bytes)"
    log_info "Disk actually freed: $(format_bytes $disk_freed)"

    if [ $JSON_OUTPUT -eq 1 ]; then
        report_json "$DRY_RUN" "$containers_found" "$containers_removed" "$total_freed_bytes"
    fi

    exit $exit_code
}

main
