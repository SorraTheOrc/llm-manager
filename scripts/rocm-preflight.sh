#!/usr/bin/env bash
# Non-destructive ROCm preflight checks
# Outputs a JSON summary when --json is provided.

set -uo pipefail

DRY_RUN=0
JSON=0
OUTFILE=""

usage() {
  cat <<EOF
Usage: $0 [--dry-run] [--json] [--output FILE]
  --dry-run  Perform read-only checks (default)
  --json     Emit machine-readable JSON summary to stdout (or --output FILE)
  --output   Write JSON output to FILE
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

errors=()
repo_present=0
repo_versions=()
repo_lines=""

if [[ -f /etc/apt/sources.list.d/rocm.list ]]; then
  repo_present=1
  repo_lines=$(grep -E 'repo.radeon.com/rocm/apt' /etc/apt/sources.list.d/rocm.list 2>/dev/null || true)
  while IFS= read -r line; do
    ver=$(echo "$line" | grep -oE '/rocm/apt/[0-9]+\.[0-9]+(\.[0-9]+)?' | sed 's#.*/rocm/apt/##' || true)
    if [[ -n "$ver" ]] && [[ " ${repo_versions[*]} " != *" $ver "* ]]; then
      repo_versions+=("$ver")
    fi
  done <<< "$repo_lines"
else
  errors+=("/etc/apt/sources.list.d/rocm.list not found")
fi

# rocm-smi detection
if command -v rocm-smi >/dev/null 2>&1; then
  rocm_smi_present=1
  rocm_smi_version=$(rocm-smi --showtag 2>/dev/null | head -n1 || rocm-smi --version 2>/dev/null || true)
else
  rocm_smi_present=0
  rocm_smi_version=""
fi

# kernel info
uname_out=$(uname -a 2>/dev/null || true)

# GPU detection via lspci
gpu_detected=0
gpu_lines=""
if command -v lspci >/dev/null 2>&1; then
  gpu_lines=$(lspci -nn | grep -i 'amd' || true)
  if [[ -n "$gpu_lines" ]]; then gpu_detected=1; fi
else
  errors+=("lspci not available")
fi

# Basic ok criteria: repo present and GPU detected
ok=1
if [[ $repo_present -eq 0 ]] || [[ $gpu_detected -eq 0 ]]; then ok=0; fi

# Prepare environment for JSON rendering via Python
export OK="$ok"
export REPO_PRESENT="$repo_present"
export REPO_VERSIONS="${repo_versions[*]}"
export ROCM_SMI_PRESENT="$rocm_smi_present"
export ROCM_SMI_VERSION="$rocm_smi_version"
export UNAME="$uname_out"
export GPU_DETECTED="$gpu_detected"
export GPU_LINES="$gpu_lines"
if [[ ${#errors[@]} -gt 0 ]]; then
  export ERRORS="$(IFS='||'; echo "${errors[*]}")"
else
  export ERRORS=""
fi

PYTHON=$(command -v python3 || command -v python || true)
if [[ -z "$PYTHON" ]]; then
  echo "No python interpreter available to render JSON" >&2
  exit 2
fi

OUTPUT_JSON=$($PYTHON - <<'PYCODE'
import os,json
data = dict(
  ok = os.environ.get('OK','0') == '1',
  repo_present = os.environ.get('REPO_PRESENT','0') == '1',
  repo_versions = os.environ.get('REPO_VERSIONS','').split(),
  rocm_smi_present = os.environ.get('ROCM_SMI_PRESENT','0') == '1',
  rocm_smi_version = os.environ.get('ROCM_SMI_VERSION',''),
  uname = os.environ.get('UNAME',''),
  gpu_detected = os.environ.get('GPU_DETECTED','0') == '1',
  gpu_lines = [l for l in os.environ.get('GPU_LINES','').splitlines() if l],
  errors = [] if os.environ.get('ERRORS','')=='' else os.environ.get('ERRORS').split('||')
)
print(json.dumps(data))
PYCODE
)

if [[ $JSON -eq 1 ]]; then
  if [[ -n "$OUTFILE" ]]; then
    printf '%s\n' "$OUTPUT_JSON" > "$OUTFILE"
  else
    printf '%s\n' "$OUTPUT_JSON"
  fi
else
  echo "OK: $ok"
  echo "Repo present: $repo_present"
  echo "Repo versions: ${repo_versions[*]}"
  echo "rocm-smi present: $rocm_smi_present"
  echo "rocm-smi version: $rocm_smi_version"
  echo "uname: $uname_out"
  echo "gpu detected: $gpu_detected"
  echo "gpu lines:"
  printf '%s\n' "$gpu_lines"
  if [[ ${#errors[@]} -gt 0 ]]; then
    echo "Errors:"
    for e in "${errors[@]}"; do echo " - $e"; done
  fi
fi

if [[ $ok -eq 1 ]]; then exit 0; else exit 2; fi
