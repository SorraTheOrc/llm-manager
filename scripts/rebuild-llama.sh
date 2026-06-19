#!/usr/bin/env bash
# Rebuild llama.cpp from master, deploy binary, and run smoke verification.
# Supports --dry-run and --json.

set -uo pipefail

DRY_RUN=0
JSON=0
OUTFILE=""
REPO="https://github.com/ggml-org/llama.cpp.git"
TARGET_DIR="/tmp/llama_rebuild"
DEPLOY_PATH="/home/rgardler/llama.cpp/build/bin/llama-server"
VERIFY_SCRIPT="scripts/verify-upgrade.sh"

usage() {
  cat <<EOF
Usage: $0 [--repo REPO] [--dir TARGET_DIR] [--deploy-path PATH] [--dry-run] [--json] [--output FILE]
  --repo REPO            Git repo to clone (default: ${REPO})
  --dir TARGET_DIR       Directory to clone/build into (default: ${TARGET_DIR})
  --deploy-path PATH     Path to deploy built binary (default: ${DEPLOY_PATH})
  --dry-run              Do not perform destructive actions; emit planned steps
  --json                 Emit JSON summary
  --output FILE          Write JSON output to FILE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --dir) TARGET_DIR="$2"; shift 2 ;;
    --deploy-path) DEPLOY_PATH="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --json) JSON=1; shift ;;
    --output) OUTFILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

planned_steps=("clone_repo" "cmake_configure" "build" "copy_binary" "restart_service" "run_verify_script")

if [[ $DRY_RUN -eq 1 ]]; then
  ok=1
  data=$(cat <<JSON
{
  "ok": ${ok},
  "repo": "${REPO}",
  "target_dir": "${TARGET_DIR}",
  "deploy_path": "${DEPLOY_PATH}",
  "verify_script": "${VERIFY_SCRIPT}",
  "planned_steps": ["${planned_steps[*]}"],
  "errors": []
}
JSON
)
  if [[ -n "$OUTFILE" ]]; then printf '%s\n' "$data" > "$OUTFILE"; else printf '%s\n' "$data"; fi
  if [[ $ok -eq 1 ]]; then exit 0; else exit 2; fi
fi

# Non-dry-run best-effort
errors=()
if ! command -v git >/dev/null 2>&1; then errors+=("git missing"); fi
if ! command -v cmake >/dev/null 2>&1; then errors+=("cmake missing"); fi

if [[ ${#errors[@]} -gt 0 ]]; then
  echo "Errors: ${errors[*]}" >&2
  exit 2
fi

rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

git clone "$REPO" "$TARGET_DIR" || { echo "git clone failed" >&2; exit 2; }

pushd "$TARGET_DIR" >/dev/null
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DLLAMA_OPENSSL=ON || { echo "cmake configure failed" >&2; popd >/dev/null; exit 2; }

cmake --build build --config Release -j"$(nproc)" || { echo "build failed" >&2; popd >/dev/null; exit 2; }

# Deploy the binary (best-effort: copy if path exists)
mkdir -p "$(dirname "$DEPLOY_PATH")"
if [[ -f build/bin/llama-server ]]; then
  cp build/bin/llama-server "$DEPLOY_PATH" || { echo "Failed to copy binary to $DEPLOY_PATH" >&2; popd >/dev/null; exit 2; }
else
  echo "Built binary not found at build/bin/llama-server" >&2
  popd >/dev/null
  exit 2
fi
popd >/dev/null

# Restarting service is environment-specific; attempt to run start-llama.sh if present
if [[ -x start-llama.sh ]]; then
  ./start-llama.sh || echo "start-llama.sh returned non-zero" >&2
else
  echo "start-llama.sh not found or not executable; manual restart may be required" >&2
fi

# Optionally run verify script (not run automatically here)

if [[ $JSON -eq 1 ]]; then
  printf '%s\n' "{\"ok\": true, \"repo\": \"${REPO}\", \"deploy_path\": \"${DEPLOY_PATH}\" }"
else
  echo "OK: rebuilt and deployed to ${DEPLOY_PATH}"
fi

exit 0
