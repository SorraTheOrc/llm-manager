#!/usr/bin/env bash
# Install/Upgrade ROCm on host (non-destructive dry-run mode available)
# This script is interactive and requires sudo for non-dry-run mode.

set -uo pipefail

DRY_RUN=0
JSON=0
OUTFILE=""
ROCM_VERSION="7.2.4"
REPO_FILE="/etc/apt/sources.list.d/rocm.list"
GPG_KEY_URL="https://repo.radeon.com/rocm/rocm.gpg.key"

usage() {
  cat <<EOF
Usage: $0 [--version VERSION] [--dry-run] [--json] [--output FILE]
  --version VERSION   ROCm version to install (default: ${ROCM_VERSION})
  --dry-run           Do not perform system changes; emit planned steps
  --json              Emit JSON summary
  --output FILE       Write JSON output to FILE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) ROCM_VERSION="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --json) JSON=1; shift ;;
    --output) OUTFILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

planned_steps=( )
planned_steps+=("update_repo:${ROCM_VERSION}")
planned_steps+=("import_gpg_key:${GPG_KEY_URL}")
planned_steps+=("apt_update_upgrade:rocm-packages")
planned_steps+=("reboot_or_reload_modules")

if [[ $DRY_RUN -eq 1 ]]; then
  ok=1
  data=$(cat <<JSON
{
  "ok": ${ok},
  "rocm_version": "${ROCM_VERSION}",
  "repo_file": "${REPO_FILE}",
  "gpg_key_url": "${GPG_KEY_URL}",
  "planned_steps": ["${planned_steps[*]}"],
  "errors": []
}
JSON
)
  if [[ -n "$OUTFILE" ]]; then printf '%s\n' "$data" > "$OUTFILE"; else printf '%s\n' "$data"; fi
  if [[ $ok -eq 1 ]]; then exit 0; else exit 2; fi
fi

# Non-dry-run: prepare changes (requires sudo)
if [[ $(id -u) -ne 0 ]]; then
  echo "Non-dry-run installation requires sudo/root privileges. Run as root or use --dry-run." >&2
  exit 2
fi

errors=()
# Write repo file
cat > "$REPO_FILE" <<EOF
# ROCm APT repository
deb [trusted=yes] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main
EOF
planned_steps+=("wrote_repo_file")

# Import GPG key
if ! command -v apt-key >/dev/null 2>&1; then
  # Use curl + gpg or apt-key depending on environment
  if command -v curl >/dev/null 2>&1 && command -v gpg >/dev/null 2>&1; then
    curl -fsSL "${GPG_KEY_URL}" | gpg --dearmour -o /usr/share/keyrings/rocm-archive-keyring.gpg || errors+=("Failed to import GPG key via curl|gpg")
  else
    errors+=("No apt-key or curl/gpg available to import GPG key")
  fi
else
  if ! curl -fsSL "${GPG_KEY_URL}" | apt-key add - >/dev/null 2>&1; then errors+=("apt-key add failed"); fi
fi

# Update apt and upgrade rocm packages
if ! apt update -o Dir::Etc::sourcelist="${REPO_FILE}" -o Dir::Etc::sourceparts="-" >/dev/null 2>&1; then errors+=("apt update failed for ROCm repo"); fi
# Note: we intentionally do not run apt upgrade here automatically; the operator should run apt upgrade manually after review

ok=1
if [[ ${#errors[@]} -gt 0 ]]; then ok=0; fi

# Emit JSON summary
PYTHON=$(command -v python3 || command -v python || true)
if [[ -z "$PYTHON" ]]; then echo "No python available to render JSON" >&2; exit 2; fi

export OK="$ok"
export ROCM_VERSION_VAR="$ROCM_VERSION"
export REPO_FILE_VAR="$REPO_FILE"
export GPG_KEY_URL_VAR="$GPG_KEY_URL"
if [[ ${#errors[@]} -gt 0 ]]; then
  export ERRORS="$(IFS='||'; echo "${errors[*]}")"
else
  export ERRORS=""
fi

OUTPUT=$($PYTHON - <<PY
import os,json
out = dict(
  ok = os.environ.get('OK','0') == '1',
  rocm_version = os.environ.get('ROCM_VERSION_VAR',''),
  repo_file = os.environ.get('REPO_FILE_VAR',''),
  gpg_key_url = os.environ.get('GPG_KEY_URL_VAR',''),
  errors = [] if os.environ.get('ERRORS','')=='' else os.environ.get('ERRORS').split('||')
)
print(json.dumps(out))
PY
)

if [[ $JSON -eq 1 ]]; then
  if [[ -n "$OUTFILE" ]]; then printf '%s\n' "$OUTPUT" > "$OUTFILE"; else printf '%s\n' "$OUTPUT"; fi
else
  echo "$OUTPUT"
fi

if [[ $ok -eq 1 ]]; then exit 0; else exit 2; fi
