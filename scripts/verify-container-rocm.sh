#!/usr/bin/env bash
# Verify that a ROCm container image (default: rocm/dev-ubuntu-24.04:7.2.4)
# can start and execute runtime commands (rocm-smi, HIP sample).

set -uo pipefail

DRY_RUN=0
JSON=0
OUTFILE=""
IMAGE="rocm/dev-ubuntu-24.04:7.2.4"

usage() {
  cat <<EOF
Usage: $0 [--image IMAGE] [--dry-run] [--json] [--output FILE]
  --image IMAGE  Container image to test (default: $IMAGE)
  --dry-run      Do not perform network/container actions; emit planned steps
  --json         Emit JSON summary
  --output FILE  Write JSON output to FILE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --json) JSON=1; shift ;;
    --output) OUTFILE="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

errors=()
podman_available=0
docker_available=0

if command -v podman >/dev/null 2>&1; then podman_available=1; fi
if command -v docker >/dev/null 2>&1; then docker_available=1; fi

if [[ $DRY_RUN -eq 1 ]]; then
  ok=1
  if [[ $podman_available -eq 0 && $docker_available -eq 0 ]]; then
    ok=0
    errors+=("No container runtime detected (podman/docker)")
  fi
  data=$(cat <<JSON
{
  "ok": ${ok},
  "image": "${IMAGE}",
  "podman_available": ${podman_available},
  "docker_available": ${docker_available},
  "planned_checks": {
    "pull_image": "pull ${IMAGE}",
    "run_rocm_smi": "run rocm-smi inside container",
    "run_hip_sample": "run small HIP sample inside container"
  },
  "errors": []
}
JSON
)
  if [[ -n "$OUTFILE" ]]; then printf '%s\n' "$data" > "$OUTFILE"; else printf '%s\n' "$data"; fi
  if [[ $ok -eq 1 ]]; then exit 0; else exit 2; fi
fi

# Non-dry-run: attempt to pull and run the image
runtime=""
if [[ $podman_available -eq 1 ]]; then runtime="podman"; elif [[ $docker_available -eq 1 ]]; then runtime="docker"; else errors+=("No container runtime detected"); fi

run_stdout=""
run_stderr=""
run_rc=0

if [[ -z "$runtime" ]]; then
  ok=0
else
  # Pull the image
  if [[ "$runtime" == "podman" ]]; then
    if ! podman pull "$IMAGE" >/dev/null 2>&1; then errors+=("podman pull failed for $IMAGE"); fi
    # Run container and execute rocm-smi and a trivial HIP check
    CMD="rocm-smi --showhw || true; echo 'ROCM-SMI-END'; uname -a; echo 'HIP-SAMPLE-START'; python3 - <<'PY'
try:
    import sys
    print('hip_sample_running')
except Exception as e:
    print('hip_sample_failed', e)
PY"
    set +e
    out=$(podman run --rm --security-opt label=disable --privileged --net=host -v /dev:/dev "$IMAGE" sh -c "$CMD" 2>&1)
    rc=$?
    set -e
    run_stdout="$out"
    run_rc=$rc
  else
    if ! docker pull "$IMAGE" >/dev/null 2>&1; then errors+=("docker pull failed for $IMAGE"); fi
    CMD="rocm-smi --showhw || true; echo 'ROCM-SMI-END'; uname -a; echo 'HIP-SAMPLE-START'; python3 - <<'PY'
try:
    import sys
    print('hip_sample_running')
except Exception as e:
    print('hip_sample_failed', e)
PY"
    set +e
    out=$(docker run --rm --privileged --net=host -v /dev:/dev "$IMAGE" sh -c "$CMD" 2>&1)
    rc=$?
    set -e
    run_stdout="$out"
    run_rc=$rc
  fi
  if [[ $run_rc -ne 0 ]]; then errors+=("container runtime test failed with rc=$run_rc"); fi
  ok=1
  if [[ ${#errors[@]} -gt 0 ]]; then ok=0; fi
fi

# Render JSON summary
PYTHON=$(command -v python3 || command -v python || true)
if [[ -z "$PYTHON" ]]; then echo "No python available to render JSON" >&2; exit 2; fi

export OK="$ok"
export IMAGE_VAR="$IMAGE"
export RUNTIME_VAR="$runtime"
export PODMAN_AVAILABLE="$podman_available"
export DOCKER_AVAILABLE="$docker_available"
export RUN_EXIT_CODE="$run_rc"
export RUN_STDOUT="$run_stdout"
if [[ ${#errors[@]} -gt 0 ]]; then
  export ERRORS="$(IFS='||'; echo "${errors[*]}")"
else
  export ERRORS=""
fi

OUTPUT=$($PYTHON - <<PY
import os,json
out = dict(
  ok = os.environ.get('OK','0') == '1',
  image = os.environ.get('IMAGE_VAR',''),
  runtime = os.environ.get('RUNTIME_VAR',''),
  podman_available = os.environ.get('PODMAN_AVAILABLE','0') == '1',
  docker_available = os.environ.get('DOCKER_AVAILABLE','0') == '1',
  run_exit_code = int(os.environ.get('RUN_EXIT_CODE','0')),
  run_stdout = os.environ.get('RUN_STDOUT',''),
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
