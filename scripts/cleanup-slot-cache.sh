#!/usr/bin/env bash
# Cleanup stale slot-cache files.
#
# Slot files are named slot_<session-uuid>.bin — one snapshot per session,
# rewritten on every save (mtime tracks last activity). Retention rules:
#
#   1. AGE: delete files older than --max-age-days (default 7).
#   2. SIZE: if the cache still exceeds --max-size-gb (default 20), delete
#      the oldest files until under the cap, but never delete files newer
#      than --min-age-days (default 1) to protect active sessions.
#
# Usage:
#   ./scripts/cleanup-slot-cache.sh                        # defaults
#   ./scripts/cleanup-slot-cache.sh --max-age-days 14      # keep 14 days
#   ./scripts/cleanup-slot-cache.sh --max-size-gb 50       # cap at 50 GB
#   ./scripts/cleanup-slot-cache.sh --min-age-days 2       # protect 2 days
#   ./scripts/cleanup-slot-cache.sh --dry-run              # preview only
#   ./scripts/cleanup-slot-cache.sh --path /custom/path    # custom slot-cache dir
#
# Exit codes:
#   0 - success (no errors or all errors handled gracefully)
#   1 - unexpected error (cannot access directory, missing dependencies)

set -eo pipefail

# Defaults
MAX_AGE_DAYS=7
MAX_SIZE_GB=20
MIN_AGE_DAYS=1
SLOT_CACHE_DIR=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-age-days)
      MAX_AGE_DAYS="$2"; shift 2 ;;
    --max-size-gb)
      MAX_SIZE_GB="$2"; shift 2 ;;
    --min-age-days)
      MIN_AGE_DAYS="$2"; shift 2 ;;
    --path)
      SLOT_CACHE_DIR="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--max-age-days N] [--max-size-gb N] [--min-age-days N] [--path DIR] [--dry-run]"
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# Resolve slot-cache directory
if [[ -z "$SLOT_CACHE_DIR" ]]; then
  repo_root="$(cd "$(dirname "$0")/.." && pwd)"
  config_file="$repo_root/proxy/config.yaml"
  if [[ -f "$config_file" ]]; then
    detected_path=$(grep -E "^[[:space:]]+session_slot_save_path:" "$config_file" | awk '{print $2}' | tr -d '"')
    SLOT_CACHE_DIR="${detected_path:-$repo_root/slot-cache}"
  else
    SLOT_CACHE_DIR="$repo_root/slot-cache"
  fi
fi

if [[ ! -d "$SLOT_CACHE_DIR" ]]; then
  echo "ERROR: Slot-cache directory does not exist: $SLOT_CACHE_DIR" >&2
  exit 1
fi
if [[ ! -r "$SLOT_CACHE_DIR" ]]; then
  echo "ERROR: Cannot read slot-cache directory: $SLOT_CACHE_DIR" >&2
  exit 1
fi

shopt -s nullglob
all_files=("$SLOT_CACHE_DIR"/slot_*.bin)
shopt -u nullglob

if [[ ${#all_files[@]} -eq 0 ]]; then
  echo "No slot files found in $SLOT_CACHE_DIR"
  exit 0
fi

now_epoch=$(date +%s)
max_age_seconds=$((MAX_AGE_DAYS * 86400))
min_age_seconds=$((MIN_AGE_DAYS * 86400))
max_size_bytes=$((MAX_SIZE_GB * 1073741824))

# Collect files with mtime and size, sorted oldest first
entries=()
total_bytes=0
for f in "${all_files[@]}"; do
  mtime=$(stat -c '%Y' "$f" 2>/dev/null || echo "0")
  size=$(stat -c '%s' "$f" 2>/dev/null || echo "0")
  if [[ "$mtime" -eq 0 ]]; then
    echo "WARNING: Cannot read modification time for $f, skipping" >&2
    continue
  fi
  entries+=("$mtime|$size|$f")
  total_bytes=$((total_bytes + size))
done

IFS=$'\n' sorted=($(for e in "${entries[@]}"; do echo "$e"; done | sort -t'|' -k1 -n))
unset IFS

# 1) AGE RULE: delete anything older than max-age-days
deleted=0
retained=0
errors=0
remaining_bytes=$total_bytes
keep_list=()

for e in "${sorted[@]}"; do
  IFS='|' read -r mtime size f <<< "$e"
  unset IFS
  age_seconds=$((now_epoch - mtime))
  if [[ $age_seconds -gt $max_age_seconds ]]; then
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[DRY RUN] Would delete: $f (age: $((age_seconds / 86400)) days, size: $((size / 1048576)) MB)"
      echo "  reason: older than $MAX_AGE_DAYS days"
      remaining_bytes=$((remaining_bytes - size))
      deleted=$((deleted + 1))
    else
      if rm -f "$f" 2>/dev/null; then
        echo "Deleted: $f (age: $((age_seconds / 86400)) days)"
        deleted=$((deleted + 1))
        remaining_bytes=$((remaining_bytes - size))
      else
        echo "WARNING: Failed to delete $f" >&2
        errors=$((errors + 1))
      fi
    fi
  else
    keep_list+=("$mtime|$size|$f")
  fi
done

# 2) SIZE RULE: if still over the cap, delete oldest (skipping files newer
#    than min-age-days to protect active sessions)
if [[ $remaining_bytes -gt $max_size_bytes ]]; then
  over_bytes=$((remaining_bytes - max_size_bytes))
  echo "Cache exceeds --max-size-gb $MAX_SIZE_GB by $((over_bytes / 1073741824)) GB; pruning oldest files (keeping files newer than $MIN_AGE_DAYS day(s))"
  for e in "${keep_list[@]}"; do
    [[ $remaining_bytes -le $max_size_bytes ]] && break
    IFS='|' read -r mtime size f <<< "$e"
    unset IFS
    age_seconds=$((now_epoch - mtime))
    if [[ $age_seconds -lt $min_age_seconds ]]; then
      continue
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[DRY RUN] Would delete: $f (age: $((age_seconds / 86400)) days, size: $((size / 1048576)) MB)"
      echo "  reason: cache over size cap"
      remaining_bytes=$((remaining_bytes - size))
      deleted=$((deleted + 1))
    else
      if rm -f "$f" 2>/dev/null; then
        echo "Deleted: $f (age: $((age_seconds / 86400)) days)"
        deleted=$((deleted + 1))
        remaining_bytes=$((remaining_bytes - size))
      else
        echo "WARNING: Failed to delete $f" >&2
        errors=$((errors + 1))
      fi
    fi
  done
fi

echo ""
if [[ $DRY_RUN -eq 1 ]]; then
  echo "DRY-RUN SUMMARY: $deleted files would be deleted; cache would shrink from $((total_bytes / 1073741824)) GB to ~$((remaining_bytes / 1073741824)) GB"
else
  echo "SUMMARY: $deleted files deleted; cache is now $((remaining_bytes / 1073741824)) GB (was $((total_bytes / 1073741824)) GB)"
fi

if [[ $errors -gt 0 ]]; then
  echo "WARNING: $errors errors encountered during cleanup (see above)" >&2
fi
exit 0
