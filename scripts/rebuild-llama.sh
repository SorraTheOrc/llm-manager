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
#
# LP-0MW3K7Y3C003WEXW: CPU-backend build gate. The deployed libggml-cpu builds
# SIGSEGV/GPF inside ggml_cpu_extra_work_size (offset +0x38/+0x41/+0x47), the
# virtual `tensor_traits` dispatch that only exists because GGML_CPU_REPACK
# registers an extra buffer type. Two build facts are therefore checked on every
# artifact this script is about to deploy, and a failing fact aborts the deploy
# BEFORE the live build dir is touched:
#   1. arch-specific quant kernel codegen (GGML_NATIVE/-march=native) — the
#      quant dot products must not fall back to the generic reference loop;
#   2. the CPU repack extra-buffer path must be absent (GGML_CPU_REPACK=OFF, the
#      default here; --with-cpu-repack restores upstream's default ON).
# The verdict comes from one shared classifier, scripts/lib/ggml_cpu_census.py
# (also used by scripts/diagnose_libggml_segfault.sh), and every run writes a
# build provenance record (commit, cmake flags, artifact sha256) — the build
# record LP-0MW3K7Y3C003WEXW AC3 asks for.

set -uo pipefail

DRY_RUN=0
JSON=0
OUTFILE=""
DEPLOY_ONLY=0
BUILD_ONLY=0
JOBS="$(nproc)"
REPO="https://github.com/ggml-org/llama.cpp.git"
TARGET_DIR="/tmp/llama_rebuild"
DEPLOY_PATH="/home/rgardler/llama.cpp/build/bin/llama-server"
ROCM_LIBS="/opt/rocm-7.2.4/lib:/opt/rocm/lib"
VERIFY_SCRIPT="scripts/verify-upgrade.sh"

# CPU-backend build facts (LP-0MW3K7Y3C003WEXW).
CPP_REPACK="OFF"
EXPECT_REPACK=""
SKIP_CENSUS=0
RECORD_FILE=""
RECORD_FILE_OUT=""
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CENSUS_SCRIPT="${GGML_CENSUS_SCRIPT:-${SCRIPT_DIR}/lib/ggml_cpu_census.py}"
WORK_ITEM="LP-0MW3K7Y3C003WEXW"
CENSUS_BUILT="null"
CENSUS_DEPLOYED="null"

usage() {
  cat <<EOF
Usage: $0 [--repo REPO] [--dir TARGET_DIR] [--deploy-path PATH] [--deploy-only] [--dry-run] [--json] [--output FILE]
          [--with-cpu-repack] [--expect-cpu-repack on|off|any] [--skip-census] [--record-file FILE]

  --repo REPO              Git repo to clone (default: ${REPO})
  --dir TARGET_DIR         Directory to clone/build into (default: ${TARGET_DIR})
  --deploy-path PATH       Path to deploy built binary (default: ${DEPLOY_PATH})
  --deploy-only            Skip clone/build; deploy the existing build at \${TARGET_DIR}/build/bin
  --build-only             Build and census the artifact, do NOT deploy (staging a
                           coordinated restart); implies no pkill and no copy
  --jobs N                 Parallel build jobs (default: nproc)
  --dry-run                Do not perform destructive actions; emit planned steps
  --json                   Emit JSON summary
  --output FILE            Write JSON output to FILE
  --with-cpu-repack        Build with GGML_CPU_REPACK=ON (upstream default). NOT
                           recommended: it is the ggml_cpu_extra_work_size crash
                           path (LP-0MW3K7Y3C003WEXW).
  --expect-cpu-repack V    Census expectation for the artifact's repack path
                           (default: on with --with-cpu-repack, else off; 'any' disables)
  --skip-census            Skip the CPU-backend census gate (emergency only)
  --record-file FILE       Build provenance record path (default: \${TARGET_DIR}/build-record.json)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --dir) TARGET_DIR="$2"; shift 2 ;;
    --deploy-path) DEPLOY_PATH="$2"; shift 2 ;;
    --deploy-only) DEPLOY_ONLY=1; shift ;;
    --build-only) BUILD_ONLY=1; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --with-cpu-repack) CPP_REPACK="ON"; shift ;;
    --expect-cpu-repack) EXPECT_REPACK="$2"; shift 2 ;;
    --skip-census) SKIP_CENSUS=1; shift ;;
    --record-file) RECORD_FILE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --json) JSON=1; shift ;;
    --output) OUTFILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "$EXPECT_REPACK" ]]; then
  if [[ "$CPP_REPACK" == "ON" ]]; then EXPECT_REPACK="on"; else EXPECT_REPACK="off"; fi
fi
case "$EXPECT_REPACK" in
  on|off|any) ;;
  *) echo "Unknown --expect-cpu-repack value: $EXPECT_REPACK (want on|off|any)" >&2; exit 2 ;;
esac

DEPLOY_DIR="$(dirname "$DEPLOY_PATH")"
SRC_BIN="$TARGET_DIR/build/bin/llama-server"
SRC_DIR="$TARGET_DIR/build/bin"

CMAKE_FLAGS=(
  -DGGML_HIP=ON
  -DAMDGPU_TARGETS=gfx1151
  -DGGML_HIP_ROCWMMA_FATTN=ON
  -DLLAMA_OPENSSL=ON
  -DGGML_NATIVE=ON
  -DGGML_CPU_REPACK="$CPP_REPACK"
)

if [[ $BUILD_ONLY -eq 1 && $DEPLOY_ONLY -eq 1 ]]; then
  echo "--build-only and --deploy-only are mutually exclusive" >&2
  exit 2
fi

if [[ $BUILD_ONLY -eq 1 ]]; then
  planned_steps=("clone_repo" "cmake_configure" "build" "verify_cpu_backend" "write_build_record")
elif [[ $DEPLOY_ONLY -eq 1 ]]; then
  planned_steps=("verify_artifact" "verify_cpu_backend" "stop_old_server" "copy_binary_and_libs" "patch_runpath" "verify_binary" "write_build_record")
else
  planned_steps=("clone_repo" "cmake_configure" "build" "verify_cpu_backend" "stop_old_server" "copy_binary_and_libs" "patch_runpath" "verify_binary" "verify_cpu_backend_deployed" "write_build_record" "restart_service")
fi

json_or_null() {
  # Compact a census report to a single JSON line, or emit `null`.
  local payload="$1"
  if [[ -z "$payload" ]]; then printf 'null'; return; fi
  printf '%s' "$payload" | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), sort_keys=True))' 2>/dev/null || printf 'null'
}

emit_json() {
  local ok="$1" ver="$2" commit="$3" libs="$4" errs="$5"
  local flags_json
  if command -v python3 >/dev/null 2>&1; then
    flags_json="$(printf '%s\n' "${CMAKE_FLAGS[@]}" | python3 -c 'import json,sys; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))')"
  else
    flags_json="[]"
  fi
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
  "cmake_flags": ${flags_json},
  "cpu_repack": "${CPP_REPACK}",
  "expect_cpu_repack": "${EXPECT_REPACK}",
  "census_skipped": $( [[ $SKIP_CENSUS -eq 1 ]] && echo true || echo false ),
  "cpu_census_built": $(json_or_null "$CENSUS_BUILT"),
  "cpu_census_deployed": $(json_or_null "$CENSUS_DEPLOYED"),
  "build_record": "${RECORD_FILE_OUT}",
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

# Non-dry-run checks. Only the build path needs git/cmake; --deploy-only
# re-uses an existing artifact and never invokes either, so requiring them
# there aborted the script with "cmake missing" on hosts/CI images without a
# compiler (LP-0MUK0KYMV003TIWV). patchelf is needed for both paths.
errors=()
if [[ $DEPLOY_ONLY -eq 0 ]]; then
  if ! command -v git >/dev/null 2>&1; then errors+=("git missing"); fi
  if ! command -v cmake >/dev/null 2>&1; then errors+=("cmake missing"); fi
fi
if ! command -v patchelf >/dev/null 2>&1; then errors+=("patchelf missing (install with: sudo apt-get install -y patchelf)"); fi
if [[ $SKIP_CENSUS -eq 0 ]]; then
  if ! command -v python3 >/dev/null 2>&1; then errors+=("python3 missing (needed for the CPU-backend census gate; bypass with --skip-census)"); fi
  if [[ ! -f "$CENSUS_SCRIPT" ]]; then errors+=("census helper missing at ${CENSUS_SCRIPT}"); fi
fi

if [[ ${#errors[@]} -gt 0 ]]; then
  echo "Errors: ${errors[*]}" >&2
  emit_json 0 "" "" "" "\"${errors[*]}\""
  exit 2
fi

# ---------------------------------------------------------------------------
# CPU-backend census gate (LP-0MW3K7Y3C003WEXW)
# ---------------------------------------------------------------------------
cpu_lib_in() {
  # First libggml-cpu.so* in a directory, or empty.
  local dir="$1" candidate
  for candidate in "$dir"/libggml-cpu.so*; do
    [[ -f "$candidate" ]] || continue
    printf '%s' "$candidate"
    return 0
  done
  return 0
}

run_census() {
  # $1 = label (built|deployed), $2 = directory holding libggml-cpu.so*
  # Sets CENSUS_<LABEL> to the report and returns 0 when the verdict is ok.
  local label="$1" dir="$2" lib output rc
  lib="$(cpu_lib_in "$dir")"
  if [[ -z "$lib" ]]; then
    echo "CPU-backend census (${label}): no libggml-cpu.so* in ${dir}" >&2
    return 1
  fi
  if [[ $SKIP_CENSUS -eq 1 ]]; then
    echo "CPU-backend census (${label}): SKIPPED (--skip-census) for ${lib}" >&2
    return 0
  fi
  # stdout carries the machine-readable report; on a failing verdict the helper
  # also prints the human summary to stderr, which we leave visible.
  output="$(python3 "$CENSUS_SCRIPT" "$lib" --json --expect-repack "$EXPECT_REPACK")"
  rc=$?
  if [[ "$label" == "built" ]]; then CENSUS_BUILT="$output"; else CENSUS_DEPLOYED="$output"; fi
  if [[ $rc -ne 0 ]]; then
    echo "CPU-backend census (${label}) FAILED for ${lib} (rc=${rc})" >&2
    [[ -n "$output" ]] || echo "  (census produced no parseable report)" >&2
    return 1
  fi
  echo "CPU-backend census (${label}): ok — arch-specific codegen present, repack=${EXPECT_REPACK} (${lib})" >&2
  return 0
}

write_build_record() {
  # Provenance record: what was built, from which commit, with which flags.
  local git_commit="$1"
  if [[ -z "$RECORD_FILE" ]]; then
    RECORD_FILE="$TARGET_DIR/build-record.json"
  fi
  RECORD_FILE_OUT="$RECORD_FILE"
  python3 - "$RECORD_FILE" "$WORK_ITEM" "$REPO" "$git_commit" "$CPP_REPACK" \
    "$SRC_DIR" "$DEPLOY_DIR" "$CENSUS_BUILT" "$CENSUS_DEPLOYED" "${CMAKE_FLAGS[@]}" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone

record_path, work_item, repo, commit, repack, src_dir, deploy_dir = sys.argv[1:8]
census_built, census_deployed = sys.argv[8], sys.argv[9]
cmake_flags = sys.argv[10:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

artifacts = []
for directory in (src_dir, deploy_dir):
    if not os.path.isdir(directory):
        continue
    for name in sorted(os.listdir(directory)):
        if not (name == "llama-server" or name.startswith("libggml") or name.startswith("libllama") or name.startswith("libmtmd")):
            continue
        path = os.path.join(directory, name)
        if os.path.islink(path) or not os.path.isfile(path):
            continue
        artifacts.append({
            "path": path,
            "bytes": os.path.getsize(path),
            "sha256": sha256(path),
            "role": "build" if directory == src_dir else "deployed",
        })

def load(raw):
    try:
        return json.loads(raw) if raw and raw != "null" else None
    except json.JSONDecodeError:
        return None

record = {
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "work_item": work_item,
    "repo": repo,
    "llama_cpp_commit": commit,
    "cmake_flags": cmake_flags,
    "cpu_repack": repack,
    "host_cpu": os.uname().machine,
    "artifacts": artifacts,
    "cpu_census_built": load(census_built),
    "cpu_census_deployed": load(census_deployed),
}
os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
with open(record_path, "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(record_path)
PY
}

if [[ $DEPLOY_ONLY -eq 0 ]]; then
  rm -rf "$TARGET_DIR"
  mkdir -p "$TARGET_DIR"

  git clone "$REPO" "$TARGET_DIR" || { echo "git clone failed" >&2; exit 2; }

  pushd "$TARGET_DIR" >/dev/null
  cmake -S . -B build "${CMAKE_FLAGS[@]}" || { echo "cmake configure failed" >&2; popd >/dev/null; exit 2; }

  cmake --build build --config Release -j"${JOBS}" || { echo "build failed" >&2; popd >/dev/null; exit 2; }
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
# Deploy: census the artifact -> stop old server -> copy binary + shared libs
#         -> patch RUNPATH -> verify -> census the deployed copy -> record
# ---------------------------------------------------------------------------
[[ -f "$SRC_BIN" ]] || { echo "Built binary not found at $SRC_BIN" >&2; emit_json 0 "" "" "" "\"missing ${SRC_BIN}\""; exit 2; }
mkdir -p "$DEPLOY_DIR"

# 0. Refuse to touch the live deploy dir when the artifact would reproduce the
#    LP-0MW3K7Y3C003WEXW CPU-backend defect (missing arch kernels, or the
#    GGML_CPU_REPACK extra-buffer path that faults in ggml_cpu_extra_work_size).
if ! run_census built "$SRC_DIR"; then
  echo "Refusing to deploy: the built CPU backend fails the census gate." >&2
  emit_json 0 "" "$git_commit" "" "\"cpu_backend_census_failed\""
  exit 2
fi

# --build-only: stage the artifact and stop here. Nothing in the live deploy dir
# is touched, so a coordinated restart (LP-0MW3K7Y3C003WEXW AC4) can decide when
# to promote it.
if [[ $BUILD_ONLY -eq 1 ]]; then
  RECORD_PATH="$(write_build_record "$git_commit")"
  if [[ -n "$RECORD_PATH" ]]; then
    RECORD_FILE_OUT="$RECORD_PATH"
    echo "OK: build record written to ${RECORD_FILE_OUT}" >&2
  fi
  version_out="$("$SRC_BIN" --version 2>&1 | grep -E '^(version|built)' | head -2 | tr '\n' ' ' | sed 's/ *$//')"
  if [[ $JSON -eq 0 ]]; then echo "OK: built (not deployed) at ${SRC_DIR} (commit ${git_commit})"; fi
  emit_json 1 "$version_out" "$git_commit" "" ""
  exit 0
fi

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
  patchelf --set-rpath "\$ORIGIN:${ROCM_LIBS}" "$DEPLOY_DIR/$lib" 2>/dev/null || true
done

# 4. Sanity: the deployed binary must load (linker resolves) and report a version.
version_out="$("$DEPLOY_PATH" --version 2>&1 | grep -E '^(version|built)' | head -2 | tr '\n' ' ' | sed 's/ *$//')"
[[ -n "$version_out" ]] || { echo "WARNING: deployed binary did not report a version (linker deps unresolved?)" >&2; version_out="<no version output>"; }

# 5. Re-verify the deployed copy (a bad copy/patchelf must not go unnoticed).
if ! run_census deployed "$DEPLOY_DIR"; then
  echo "WARNING: the deployed CPU backend fails the census gate; the artifact is in place but MUST NOT be restarted." >&2
  emit_json 0 "$version_out" "$git_commit" "" "\"deployed_cpu_backend_census_failed\""
  exit 2
fi

LIB_JSON=""
for lib in "${deployed_libs[@]}"; do LIB_JSON="${LIB_JSON}\"${lib}\","; done
LIB_JSON="$(echo "${LIB_JSON%,}")"

# 6. Provenance record: commit, flags, artifact checksums, census verdicts.
RECORD_PATH="$(write_build_record "$git_commit")"
if [[ -n "$RECORD_PATH" ]]; then
  RECORD_FILE_OUT="$RECORD_PATH"
  echo "OK: build record written to ${RECORD_FILE_OUT}" >&2
fi

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
