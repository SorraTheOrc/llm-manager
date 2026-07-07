#!/usr/bin/env bash
# Cleanup stale slot-cache files.
#
# Removes slot snapshot files (slot_*.bin) older than a configurable age,
# retaining a configurable number of the most recently modified files per
# unique slot ID prefix.
#
# Usage:
#   ./scripts/cleanup-slot-cache.sh                     # default: 7 days, keep 3
#   ./scripts/cleanup-slot-cache.sh --max-age-days 14   # keep 14 days
#   ./scripts/cleanup-slot-cache.sh --keep-recent 5      # keep 5 per prefix
#   ./scripts/cleanup-slot-cache.sh --dry-run            # preview only
#   ./scripts/cleanup-slot-cache.sh --path /custom/path  # custom slot-cache dir
#
# Exit codes:
#   0 - success (no errors or all errors handled gracefully)
#   1 - unexpected error (cannot access directory, missing dependencies)

set -eo pipefail

# Defaults
MAX_AGE_DAYS=7
KEEP_RECENT=3
SLOT_CACHE_DIR=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-age-days)
      MAX_AGE_DAYS="$2"; shift 2 ;;
    --keep-recent)
      KEEP_RECENT="$2"; shift 2 ;;
    --path)
      SLOT_CACHE_DIR="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--max-age-days N] [--keep-recent N] [--path DIR] [--dry-run]"
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

# Sort by modification time (oldest first)
sorted_files=()
while IFS= read -r -d '' f; do
  sorted_files+=("$f")
done < <(
  for f in "${all_files[@]}"; do
    mtime=$(stat -c '%Y' "$f" 2>/dev/null || echo "0")
    echo "$mtime|$f"
  done | sort -t'|' -k1 -n | cut -d'|' -f2- | tr '\n' '\0'
)

# Count files per prefix (everything before first '-' after 'slot_')
declare -A prefix_counts
for f in "${sorted_files[@]}"; do
  basename_f="$(basename "$f")"
  prefix="${basename_f%%-*}"
  count=${prefix_counts[$prefix]:-0}
  prefix_counts[$prefix]=$((count + 1))
done

declare -A kept_count
deleted=0
retained=0
errors=0
now_epoch=$(date +%s)
max_age_seconds=$((MAX_AGE_DAYS * 86400))

for f in "${sorted_files[@]}"; do
  basename_f="$(basename "$f")"
  prefix="${basename_f%%-*}"

  total="${prefix_counts[$prefix]:-0}"
  kept="${kept_count[$prefix]:-0}"
  kept=$((kept + 1))
  kept_count[$prefix]=$kept

  remaining=$((total - kept))

  if [[ $remaining -lt $KEEP_RECENT ]]; then
    retained=$((retained + 1))
    continue
  fi

  mtime=$(stat -c '%Y' "$f" 2>/dev/null || echo "0")
  if [[ "$mtime" -eq 0 ]]; then
    echo "WARNING: Cannot read modification time for $f, skipping" >&2
    continue
  fi

  age_seconds=$((now_epoch - mtime))
  if [[ $age_seconds -gt $max_age_seconds ]]; then
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[DRY RUN] Would delete: $f (age: $((age_seconds / 86400)) days)"
    else
      if rm -f "$f" 2>/dev/null; then
        echo "Deleted: $f (age: $((age_seconds / 86400)) days)"
        deleted=$((deleted + 1))
      else
        echo "WARNING: Failed to delete $f" >&2
        errors=$((errors + 1))
      fi
    fi
  fi
done

echo ""
if [[ $DRY_RUN -eq 1 ]]; then
  echo "DRY-RUN SUMMARY: $deleted files would be deleted, $retained recent files retained in $SLOT_CACHE_DIR"
else
  echo "SUMMARY: $deleted files deleted, $retained recent files retained in $SLOT_CACHE_DIR"
fi

if [[ $errors -gt 0 ]]; then
  echo "WARNING: $errors errors encountered during cleanup (see above)" >&2
fi
exit 0
