#!/usr/bin/env bash
# cleanup-pi-sessions.sh — Remove stale pi agent session logs based on
# retention policy (keep last 90 days or last 50 sessions per workspace).
#
# Scans the pi session log directory for workspace subdirectories and
# removes session log files outside the retention window.
#
# Supports --dry-run and --json flags. Designed for cron (logs to stdout
# for mail capture). Idempotent.
#
# Configuration (all environment variables and overridable flags):
#   PI_SESSIONS_DIR   Path to pi session logs (default: ~/.pi/agent/sessions)
#
# Usage:
#   cleanup-pi-sessions.sh [--dry-run] [--json]
#                          [--retention-days <days>] [--keep-sessions <n>]
set -uo pipefail

# ---------- defaults ----------
PI_SESSIONS_DIR="${PI_SESSIONS_DIR:-${HOME}/.pi/agent/sessions}"
RETENTION_DAYS=90
KEEP_SESSIONS=50
DRY_RUN=0
JSON_OUTPUT=0
MOCK_NOW=""  # for testing only

# ---------- flags ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)          DRY_RUN=1; shift ;;
        --json)             JSON_OUTPUT=1; shift ;;
        --retention-days)   RETENTION_DAYS="$2"; shift 2 ;;
        --keep-sessions)    KEEP_SESSIONS="$2"; shift 2 ;;
        --mock-now)         MOCK_NOW="$2"; shift 2 ;;  # testing only
        -h|--help)          echo "Usage: $0 [--dry-run] [--json] [--retention-days <days>] [--keep-sessions <n>]"; exit 0 ;;
        *)                  echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# ---------- helpers ----------
json_escape() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

report_json() {
    local dry_run="$1"
    local total_removed="$2"
    local total_freed_bytes="$3"
    local workspaces_cleaned="$4"
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
  "sessions_removed": $total_removed,
  "freed_bytes": $total_freed_bytes,
  "workspaces_cleaned": $workspaces_cleaned,
  "errors": $errors_json
}
JSON
}

log_info() {
    if [ $JSON_OUTPUT -eq 0 ]; then
        echo "$@" >&2
    fi
}
log_error() { echo "$@" >&2; }

# ---------- date helpers ----------
# Get "now" timestamp in seconds since epoch
get_now_epoch() {
    if [ -n "$MOCK_NOW" ]; then
        # Parse mock date in format YYYY-MM-DD
        local year="${MOCK_NOW%%-*}"
        local rest="${MOCK_NOW#*-}"
        local month="${rest%%-*}"
        local day="${rest#*-}"
        # Use date parsing for portability
        date -d "${year}-${month}-${day}" +%s 2>/dev/null || echo "0"
    else
        date +%s
    fi
}

# Parse a session filename timestamp (ISO-like) into epoch seconds
parse_session_timestamp() {
    local filename="$1"
    local ts

    # Extract timestamp part: 2026-06-29T01-12-21-981Z  -> 2026-06-29 01:12:21
    ts=$(echo "$filename" | sed 's/^\([0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}\)T\([0-9]\{2\}\)-\([0-9]\{2\}\)-\([0-9]\{2\}\).*$/\1 \2:\3:\4/')

    # Check if we got a valid-looking timestamp
    if echo "$ts" | grep -qE '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$'; then
        date -d "$ts" +%s 2>/dev/null || echo "0"
    else
        # Fallback: use file modification time
        date -r "$filename" +%s 2>/dev/null || echo "0"
    fi
}

# ---------- cleanup pi session logs ----------
cleanup_sessions() {
    local total_removed=0
    local total_freed=0
    local workspaces_cleaned=0
    declare -a errors_list=()

    local now_epoch
    now_epoch=$(get_now_epoch)
    local cutoff_epoch=$((now_epoch - RETENTION_DAYS * 86400))

    log_info "=== Pi Session Log Cleanup ==="
    log_info "Session dir: $PI_SESSIONS_DIR"
    log_info "Retention period: $RETENTION_DAYS days"
    log_info "Max sessions per workspace: $KEEP_SESSIONS"
    [ $DRY_RUN -eq 1 ] && log_info "Mode: DRY RUN (no deletions)"

    if [ ! -d "$PI_SESSIONS_DIR" ]; then
        log_info "Session directory does not exist: $PI_SESSIONS_DIR — nothing to clean"
        report_json "$DRY_RUN" 0 0 0
        exit 0
    fi

    # Iterate over workspace directories
    for ws_dir in "$PI_SESSIONS_DIR"/--*/; do
        [ ! -d "$ws_dir" ] && continue

        local ws_name
        ws_name="$(basename "$ws_dir")"
        log_info ""
        log_info "Workspace: $ws_name"

        # Get all session files sorted by name (timestamp order, oldest first)
        local session_files
        session_files=$(find "$ws_dir" -maxdepth 1 -name "*.jsonl" 2>/dev/null | sort)

        if [ -z "$session_files" ]; then
            log_info "  No session files found"
            continue
        fi

        # Phase 1: Remove files outside the retention period (age-based)
        local remaining_files=""
        local removed_by_age=0
        local freed_by_age=0

        while IFS= read -r file; do
            [ -z "$file" ] && continue
            local filename
            filename=$(basename "$file")
            local file_epoch
            file_epoch=$(parse_session_timestamp "$filename")

            if [ "$file_epoch" -le "$cutoff_epoch" ] && [ "$file_epoch" -gt 0 ]; then
                local size
                size=$(stat -c%s "$file" 2>/dev/null || echo "0")

                if [ $DRY_RUN -eq 1 ]; then
                    log_info "  [dry-run] Would remove (age): $filename ($((size / 1024)) KB)"
                else
                    log_info "  Removing (age): $filename ($((size / 1024)) KB)"
                    rm -f "$file"
                fi
                removed_by_age=$((removed_by_age + 1))
                freed_by_age=$((freed_by_age + size))
            else
                remaining_files="$remaining_files$file"$'\n'
            fi
        done <<< "$session_files"

        # Phase 2: If more than KEEP_SESSIONS remain, remove oldest extras
        # Re-sort remaining files (newest first for easy counting)
        local remaining_sorted
        remaining_sorted=$(echo "$remaining_files" | grep -v '^$' | sort -r)

        local count=0
        local removed_by_cap=0
        local freed_by_cap=0

        while IFS= read -r file; do
            [ -z "$file" ] && continue
            count=$((count + 1))

            if [ "$count" -gt "$KEEP_SESSIONS" ]; then
                local size
                size=$(stat -c%s "$file" 2>/dev/null || echo "0")
                local filename
                filename=$(basename "$file")

                if [ $DRY_RUN -eq 1 ]; then
                    log_info "  [dry-run] Would remove (cap): $filename ($((size / 1024)) KB)"
                else
                    log_info "  Removing (cap): $filename ($((size / 1024)) KB)"
                    rm -f "$file"
                fi
                removed_by_cap=$((removed_by_cap + 1))
                freed_by_cap=$((freed_by_cap + size))
            fi
        done <<< "$remaining_sorted"

        local ws_removed=$((removed_by_age + removed_by_cap))
        local ws_freed=$((freed_by_age + freed_by_cap))
        total_removed=$((total_removed + ws_removed))
        total_freed=$((total_freed + ws_freed))

        if [ $ws_removed -gt 0 ]; then
            workspaces_cleaned=$((workspaces_cleaned + 1))
        fi

        log_info "  Removed: $ws_removed session(s), freed $((ws_freed / 1024)) KB"
    done

    log_info ""
    log_info "=== Summary ==="
    log_info "Total sessions removed: $total_removed"
    log_info "Total freed: $((total_freed / 1024 / 1024)) MB"

    if [ $JSON_OUTPUT -eq 1 ]; then
        report_json "$DRY_RUN" "$total_removed" "$total_freed" "$workspaces_cleaned"
    fi

    exit 0
}

cleanup_sessions
