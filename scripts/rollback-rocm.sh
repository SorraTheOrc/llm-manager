#!/usr/bin/env bash
# Rollback ROCm host APT repository changes made by install-rocm.sh
# Supports --dry-run and --json

set -uo pipefail

DRY_RUN=0
JSON=0
OUTFILE=""
REPO_FILE="/etc/apt/sources.list.d/rocm.list"

usage() {
  cat <<EOF
Usage: $0 [--dry-run] [--json] [--output FILE]
  --dry-run    Do not perform destructive actions
  --json       Emit JSON summary
  --output     Write JSON output to FILE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --json) JSON=1; shift ;;
    --output) OUTFILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

if [[ $DRY_RUN -eq 1 ]]; then
  ok=1
  data=$(cat <<JSON
{
  "ok": ${ok},
  "repo_file": "${REPO_FILE}",
  "planned_steps": ["restore_rocm_repo_to_previous","reinstall_rocm_packages_6.4.2_if_needed"],
  "errors": []
}
JSON
)
  if [[ -n "$OUTFILE" ]]; then printf '%s\n' "$data" > "$OUTFILE"; else printf '%s\n' "$data"; fi
  if [[ $ok -eq 1 ]]; then exit 0; else exit 2; fi
fi

if [[ $(id -u) -ne 0 ]]; then
  echo "Rollback requires root privileges. Run as root or use --dry-run." >&2
  exit 2
fi

# Attempt to restore backup if exists
if [[ -f "${REPO_FILE}.bak" ]]; then
  mv "${REPO_FILE}.bak" "${REPO_FILE}" || { echo "Failed to restore ${REPO_FILE}.bak" >&2; exit 2; }
  echo "Restored ${REPO_FILE} from backup"
else
  echo "No backup found; manual rollback may be required" >&2
  exit 2
fi

# Optionally reinstall packages etc — omitted here for safety

if [[ $JSON -eq 1 ]]; then
  printf '%s\n' "{\"ok\": true, \"repo_file\": \"${REPO_FILE}\" }"
else
  echo "Rollback completed (repo restored)"
fi

exit 0
