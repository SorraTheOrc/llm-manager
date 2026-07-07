#!/usr/bin/env bash
# install-cron.sh — Install crontab entries for LLM stack cleanup.
#
# Adds daily cron entries for cleanup-all.sh and the slot-cache cleanup
# script (if installed). Can be run multiple times safely (idempotent).
#
# The script reads the project root from the script location or a provided
# path, and installs/updates cron entries using a temporary file.
#
# Supports --dry-run to preview changes, --uninstall to remove entries,
# and --help for usage info.
#
# Configuration (environment variables):
#   PROJECT_ROOT          Override project directory (default: auto-detect)
#   CLEANUP_HOUR          Hour for cleanup cron (default: 3 = 3 AM)
#   CLEANUP_LOG           Log file path (default: /var/log/pi-cleanup.log)
#   CLEANUP_ENABLE_MODELS     0 to skip (default: 1)
#   CLEANUP_ENABLE_CONTAINERS 0 to skip (default: 1)
#   CLEANUP_ENABLE_PI         0 to skip (default: 1)
#
# Usage:
#   install-cron.sh [--dry-run] [--uninstall]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Resolve the real project root (git common dir parent, works in worktrees too)
if [ -z "${PROJECT_ROOT:-}" ]; then
    GIT_COMMON="$(git rev-parse --git-common-dir 2>/dev/null)"
    if [ -n "$GIT_COMMON" ]; then
        PROJECT_ROOT="$(cd "$GIT_COMMON/.." && pwd)"
    else
        PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    fi
fi

CLEANUP_HOUR="${CLEANUP_HOUR:-3}"
CLEANUP_LOG="${CLEANUP_LOG:-/var/log/pi-cleanup.log}"
CRON_TAG="# pi-cleanup-orchestrator"
CRON_TAG_END="# end-pi-cleanup-orchestrator"

DRY_RUN=0
UNINSTALL=0

# ---------- flags ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=1; shift ;;
        --uninstall)  UNINSTALL=1; shift ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--uninstall]"
            echo ""
            echo "Installs crontab entries for daily LLM stack cleanup."
            echo ""
            echo "Environment variables:"
            echo "  PROJECT_ROOT          Project directory (default: auto-detect: $PROJECT_ROOT)"
            echo "  CLEANUP_HOUR          Hour for cleanup cron (default: $CLEANUP_HOUR)"
            echo "  CLEANUP_LOG           Log file path (default: $CLEANUP_LOG)"
            echo "  CLEANUP_ENABLE_MODELS      0 to skip (default: 1)"
            echo "  CLEANUP_ENABLE_CONTAINERS  0 to skip (default: 1)"
            echo "  CLEANUP_ENABLE_PI          0 to skip (default: 1)"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# ---------- helpers ----------
ORCHESTRATOR="$PROJECT_ROOT/scripts/cleanup-all.sh"
SLOT_CACHE_SCRIPT="$PROJECT_ROOT/scripts/cleanup-slot-cache.sh"

info() { echo "$@"; }
warn() { echo "WARNING: $@" >&2; }
error() { echo "ERROR: $@" >&2; }

# ---------- sanity checks ----------
if [ ! -f "$ORCHESTRATOR" ]; then
    warn "cleanup-all.sh not found at $ORCHESTRATOR"
    warn "You can set PROJECT_ROOT to the correct path, e.g.:"
    warn "  PROJECT_ROOT=/home/rgardler/projects/llm $0"
    warn "Continuing with dry-run mode pending script installation."
fi

if [ -f "$ORCHESTRATOR" ] && [ ! -x "$ORCHESTRATOR" ]; then
    error "cleanup-all.sh is not executable at $ORCHESTRATOR"
    exit 1
fi

# Check if log directory is writable (or will be created by cron)
LOG_DIR="$(dirname "$CLEANUP_LOG")"
if [ ! -d "$LOG_DIR" ] && [ ! -w "/" ]; then
    warn "Log directory $LOG_DIR does not exist and may not be writable."
    warn "Cron will attempt to create it. If this fails, set CLEANUP_LOG to a writable path."
fi

# ---------- build cron entries ----------
build_cron_entries() {
    local entries=""

    # Main cleanup orchestrator entry
    entries+="${CRON_TAG}
# Daily LLM stack cleanup orchestrator
# Env toggles: CLEANUP_ENABLE_MODELS, CLEANUP_ENABLE_CONTAINERS, CLEANUP_ENABLE_PI
0 ${CLEANUP_HOUR} * * * cd ${PROJECT_ROOT} && CLEANUP_ENABLE_MODELS=${CLEANUP_ENABLE_MODELS:-1} CLEANUP_ENABLE_CONTAINERS=${CLEANUP_ENABLE_CONTAINERS:-1} CLEANUP_ENABLE_PI=${CLEANUP_ENABLE_PI:-1} ${ORCHESTRATOR} >> ${CLEANUP_LOG} 2>&1
"

    # Slot cache cleanup (if script exists)
    if [ -x "$SLOT_CACHE_SCRIPT" ]; then
        entries+="
# Slot cache cleanup (separate item LP-0MQMC4MNU002QJK4)
# Runs at a different hour to avoid resource contention
0 $((CLEANUP_HOUR + 1)) * * * cd ${PROJECT_ROOT} && ${SLOT_CACHE_SCRIPT} >> ${CLEANUP_LOG} 2>&1
"
    fi

    entries+="${CRON_TAG_END}"
    echo "$entries"
}

# ---------- main ----------
main() {
    info "LLM Stack Cleanup Cron Installer"
    info "Project root: $PROJECT_ROOT"
    info "Orchestrator: $ORCHESTRATOR"
    info "Log file: $CLEANUP_LOG"
    info "Hour: $CLEANUP_HOUR:00"
    info ""

    if [ $UNINSTALL -eq 1 ]; then
        info "Uninstalling cron entries..."
        if [ $DRY_RUN -eq 1 ]; then
            info "[dry-run] Would remove cron entries tagged ${CRON_TAG}"
        else
            crontab -l 2>/dev/null | sed "/${CRON_TAG}/,/${CRON_TAG_END}/d" | crontab -
            info "Cron entries removed."
        fi
        exit 0
    fi

    local new_entries
    new_entries=$(build_cron_entries)

    info "Cron entries to install:"
    info ""
    echo "$new_entries"
    info ""

    if [ $DRY_RUN -eq 1 ]; then
        info "[dry-run] No changes made. Run without --dry-run to install."
        exit 0
    fi

    # Merge with existing crontab
    local current_crontab=""
    if crontab -l 2>/dev/null; then
        current_crontab=$(crontab -l 2>/dev/null)
    fi

    # Remove existing pi cleanup entries (by tag) and add new ones
    local updated_crontab
    updated_crontab=$(echo "$current_crontab" | sed "/${CRON_TAG}/,/${CRON_TAG_END}/d")
    updated_crontab="${updated_crontab}
${new_entries}"

    echo "$updated_crontab" | crontab -

    if [ $? -eq 0 ]; then
        info "Cron entries installed successfully."
        info "Next run: daily at ${CLEANUP_HOUR}:00"
        info "Logs: $CLEANUP_LOG"
        info ""
        info "To uninstall: $0 --uninstall"
    else
        error "Failed to install cron entries."
        exit 1
    fi
}

main
