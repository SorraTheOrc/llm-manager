#!/usr/bin/env bash
# Rebuild llama.cpp from master, deploy binary, and run smoke verification.
# Supports --dry-run, --deploy-only and --json.
#
# LP-0MSXXKZOW0038XLK: deploy ordering fix. The previous implementation copied
# the freshly built llama-server over the LIVE running binary, which fails with
# "Text file busy" (ETXTBSY) and aborted the whole MTP rebuild (LP-0MSNI1B68001VE6C).
# Now the deploy step:
#   1. stops running llama-server processes bound to the deploy path (pkill/pgrep
#      poll, up to 30s) BEFORE copying,
#   2. copies the binary AND its sibling shared libs (libggml*.so*, libllama*.so*,
#      libmtmd*.so*) — the new shared-library layout requires them,
#   3. rewrites the RUNPATH on the deployed ELF artifacts to $ORIGIN (via patchelf)
#      so the deployed libs resolve from the deploy dir instead of the temp build,
#   4. sanity-runs `--version` to confirm the binary loads and its linker deps resolve.
#
# --deploy-only re-deploys an ALREADY-BUILT artifact (skips clone/cmake/build); used
# by rebuild-and-restart-mtp.sh to re-attempt the copy AFTER the proxy stack restarts.

set -uo pipefail

DRY_RUN=0
JSON=0
OUTFILE=""
DEPLOY_ONLY=0
REPO="https://github.com/ggml-org/llama.cpp.git"
TARGET_DIR="/tmp/llama_rebuild"
DEPLOY_PATH="/home/rgardler/llama.cpp/build/bin/llama-server"
ROCM_LIBS="/opt/rocm-7.2.4/lib:/opt/rocm/lib"
VERIFY_SCRIPT="scripts/verify-upgrade.sh"

usage() {
  cat <<EOF
Usage: $0 [--repo REPO] [--dir TARGET_DIR] [--deploy-path PATH] [--deploy-only] [--dry-run] [--json] [--output FILE]
  --repo REPO            Git repo to clone (default: ${REPO})
  --dir TARGET_DIR       Directory to clone/build into (default: ${TARGET_DIR})
  --deploy-path PATH     Path to deploy built binary (default: ${DEPLOY_PATH})
  --deploy-only          Skip clone/build; deploy the existing build at \${TARGET_DIR}/build/bin
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
    --deploy-only) DEPLOY_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --json) JSON=1; shift ;;
    --output) OUTFILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

DEPLOY_DIR="$(dirname "$DEPLOY_PATH")"
SRC_BIN="$TARGET_DIR/build/bin/llama-server"
SRC_DIR="$TARGET_DIR/build/bin"

if [[ $DEPLOY_ONLY -eq 1 ]]; then
  planned_steps=("verify_artifact" "stop_old_server" "copy_binary_and_libs" "patch_runpath" "verify_binary")
else
  planned_steps=("clone_repo" "cmake_configure" "build" "stop_old_server" "copy_binary_and_libs" "patch_runpath" "verify_binary" "restart_service")
fi

emit_json() {
  local ok="$1" ver="$2" commit="$3" libs="$4" errs="$5"
  data=$(cat <<JSON
{
  "ok": ${ok},
  "repo": "${REPO}",
  "target_dir": "${TARGET_DIR}",
  "deploy_path": "${DEPLOY_PATH}",
  "git_commit": "${commit}",
  "deployed_libs": [${libs}],
  "version": "${ver}",
  "verify_script": "${VERIFY_SCRIPT}",
  "planned_steps": ["${planned_steps[*]}"],
  "errors": [${errs}]
}
JSON
)
  if [[ -n "$OUTFILE" ]]; then printf '%s\n' "$data" > "$OUTFILE"; else printf '%s\n' "$data"; fi
}

if [[ $DRY_RUN -eq 1 ]]; then
  emit_json 1 "" "" "" ""
  if [[ $JSON -eq 1 ]]; then exit 0; fi
  exit 0
fi

# Non-dry-run checks
errors=()
if ! command -v git >/dev/null 2>&1; then errors+=("git missing"); fi
if ! command -v cmake >/dev/null 2>&1; then errors+=("cmake missing"); fi
if ! command -v patchelf >/dev/null 2>&1; then errors+=("patchelf missing (install with: sudo apt-get install -y patchelf)"); fi

if [[ ${#errors[@]} -gt 0 ]]; then
  echo "Errors: ${errors[*]}" >&2
  emit_json 0 "" "" "" "\"${errors[*]}\""
  exit 2
fi

if [[ $DEPLOY_ONLY -eq 0 ]]; then
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
  git_commit="$(git rev-parse --short HEAD)"
  popd >/dev/null
else
  # --deploy-only: the artifact must already exist (built by a previous run)
  [[ -f "$SRC_BIN" ]] || { echo "Built binary not found at $SRC_BIN (run without --deploy-only first)" >&2; emit_json 0 "" "" "" "\"no artifact at ${SRC_BIN}\""; exit 2; }
  pushd "$TARGET_DIR" >/dev/null
  git_commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  popd >/dev/null
fi

# ---------------------------------------------------------------------------
# Deploy: stop old server -> copy binary + shared libs -> patch RUNPATH -> verify
# ---------------------------------------------------------------------------
[[ -f "$SRC_BIN" ]] || { echo "Built binary not found at $SRC_BIN" >&2; emit_json 0 "" "" "" "\"missing ${SRC_BIN}\""; exit 2; }
mkdir -p "$DEPLOY_DIR"

# 1. Stop running llama-server processes BEFORE copying, otherwise `cp` fails
#    with ETXTBSY ("Text file busy"). Match by process NAME (-x), NOT by the
#    full path: a wrapper shell (e.g. `bash -c '... --deploy-path <path>'`)
#    contains the path in its own command line, so `pkill -f <path>` would kill
#    the invoking shell itself (observed during LP-0MSXXKZOW0038XLK deploy).
if command -v pkill >/dev/null 2>&1 && command -v pgrep >/dev/null 2>&1; then
  pkill -x llama-server 2>/dev/null || true
  for _ in $(seq 1 30); do
    pgrep -x llama-server >/dev/null 2>&1 || break
    sleep 1
  done
else
  echo "WARNING: pkill/pgrep unavailable; copying over a running binary may fail with ETXTBSY" >&2
fi

# 2. Copy the binary and its sibling shared libs, replacing the target ATOMICALLY
#    (cp to a temp name, then rename into place). Once the proxy respawns the
#    router it may exec/mmap the old build within a second — rename() replaces the
#    directory entry without touching the executing inode, so it never hits ETXTBSY.
cp -f "$SRC_BIN" "$DEPLOY_PATH.tmp" || { echo "Failed to stage binary to $DEPLOY_PATH.tmp" >&2; emit_json 0 "" "" "" "\"copy failed for ${DEPLOY_PATH}\""; exit 2; }
mv -f "$DEPLOY_PATH.tmp" "$DEPLOY_PATH" || { echo "Failed to install binary to $DEPLOY_PATH" >&2; emit_json 0 "" "" "" "\"install failed for ${DEPLOY_PATH}\""; exit 2; }
deployed_libs=()
for lib in "$SRC_DIR"/lib*.so*; do
  [[ -e "$lib" ]] || continue
  cp -f "$lib" "$DEPLOY_DIR/$(basename "$lib").tmp" || { echo "Failed to copy library $(basename "$lib") to $DEPLOY_DIR" >&2; }
  mv -f "$DEPLOY_DIR/$(basename "$lib").tmp" "$DEPLOY_DIR/$(basename "$lib")" || { echo "Failed to install library $(basename "$lib") in $DEPLOY_DIR" >&2; }
  deployed_libs+=("$(basename "$lib")")
done

# 3. Rewrite RUNPATH to $ORIGIN so deployed artifacts resolve their sibling libs
#    from the deploy dir (the build dir path in RUNPATH only exists while the
#    temp build survives; next rebuild runs `rm -rf` on it).
patchelf --set-rpath "\$ORIGIN:${ROCM_LIBS}" "$DEPLOY_PATH" || { echo "patchelf failed on $DEPLOY_PATH" >&2; emit_json 0 "" "" "" "\"patchelf failed\""; exit 2; }
for lib in "${deployed_libs[@]}"; do
  [[ "$lib" == *.so.* ]] || continue
  patchelf --set-rpath "\$ORIGIN:${ROCM_LIBS}" "$DEPLOY_DIR/$lib" 2>/dev/null || true
done

# 4. Sanity: the deployed binary must load (linker resolves) and report a version.
version_out="$("$DEPLOY_PATH" --version 2>&1 | grep -E '^(version|built)' | head -2 | tr '\n' ' ' | sed 's/ *$//')"
[[ -n "$version_out" ]] || { echo "WARNING: deployed binary did not report a version (linker deps unresolved?)" >&2; version_out="<no version output>"; }

LIB_JSON=""
for lib in "${deployed_libs[@]}"; do LIB_JSON="${LIB_JSON}\"${lib}\","; done
LIB_JSON="$(echo "${LIB_JSON%,}")"

if [[ $DEPLOY_ONLY -eq 1 ]]; then
  if [[ $JSON -eq 0 ]]; then echo "OK: deployed existing build from ${TARGET_DIR} to ${DEPLOY_DIR} (commit ${git_commit})"; fi
  emit_json 1 "$version_out" "$git_commit" "$LIB_JSON" ""
  exit 0
fi

# Restarting service is environment-specific; attempt to run start-llama.sh if present
if [[ -x start-llama.sh ]]; then
  ./start-llama.sh || echo "start-llama.sh returned non-zero" >&2
else
  echo "start-llama.sh not found or not executable; manual restart may be required" >&2
fi

if [[ $JSON -eq 0 ]]; then echo "OK: rebuilt and deployed to ${DEPLOY_PATH} (commit ${git_commit})"; fi
emit_json 1 "$version_out" "$git_commit" "$LIB_JSON" ""
exit 0