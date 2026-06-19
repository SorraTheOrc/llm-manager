#!/usr/bin/env bash
# Rebuild the llama-server container against a specified ROCm base image.
# Supports --update-containerfile, --build, --tag, --dry-run and --json modes.

set -uo pipefail

DRY_RUN=0
JSON=0
OUTFILE=""
IMAGE="rocm/dev-ubuntu-24.04:7.2.4"
TAG="llm/llama-server:rocm-7.2.4"
UPDATE_CONTAINERFILE=0
BUILD=0
CONTAINERFILE="Containerfile"

usage() {
  cat <<EOF
Usage: $0 [--image IMAGE] [--tag TAG] [--update-containerfile] [--build] [--dry-run] [--json] [--output FILE]
  --image IMAGE            Base image to use (default: $IMAGE)
  --tag TAG                Tag for built image (default: $TAG)
  --update-containerfile   Update ${CONTAINERFILE} FROM line to use IMAGE
  --build                  Build the image using podman/docker
  --dry-run                Do not perform destructive actions; emit planned steps
  --json                   Emit JSON summary
  --output FILE            Write JSON output to FILE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --update-containerfile) UPDATE_CONTAINERFILE=1; shift ;;
    --build) BUILD=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --json) JSON=1; shift ;;
    --output) OUTFILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

planned_steps=( )
planned_steps+=("use_image:${IMAGE}")
planned_steps+=("tag:${TAG}")
planned_steps+=("update_containerfile:${UPDATE_CONTAINERFILE}")
planned_steps+=("build_image:${BUILD}")

if [[ $DRY_RUN -eq 1 ]]; then
  ok=1
  data=$(cat <<JSON
{
  "ok": ${ok},
  "image": "${IMAGE}",
  "tag": "${TAG}",
  "update_containerfile": ${UPDATE_CONTAINERFILE},
  "build": ${BUILD},
  "planned_steps": ["${planned_steps[*]}"],
  "errors": []
}
JSON
)
  if [[ -n "$OUTFILE" ]]; then printf '%s\n' "$data" > "$OUTFILE"; else printf '%s\n' "$data"; fi
  if [[ $ok -eq 1 ]]; then exit 0; else exit 2; fi
fi

# Non-dry-run actions
errors=()
if [[ $UPDATE_CONTAINERFILE -eq 1 ]]; then
  if [[ ! -f "$CONTAINERFILE" ]]; then errors+=("${CONTAINERFILE} not found"); fi
  if [[ ${#errors[@]} -eq 0 ]]; then
    # Backup Containerfile
    cp "$CONTAINERFILE" "${CONTAINERFILE}.bak"
    # Replace FROM line
    awk -v img="$IMAGE" 'BEGIN{updated=0} /^FROM /{ if(!updated){ print "FROM " img; updated=1 } else print $0; next } { print $0 }' "$CONTAINERFILE" > "${CONTAINERFILE}.tmp" && mv "${CONTAINERFILE}.tmp" "$CONTAINERFILE"
    planned_steps+=("containerfile_updated")
  fi
fi

runtime=""
if command -v podman >/dev/null 2>&1; then runtime="podman"; elif command -v docker >/dev/null 2>&1; then runtime="docker"; fi

if [[ $BUILD -eq 1 ]]; then
  if [[ -z "$runtime" ]]; then errors+=("No container runtime (podman/docker) found"); fi
  if [[ -n "$runtime" ]]; then
    if [[ "$runtime" == "podman" ]]; then
      if ! podman build -t "$TAG" -f "$CONTAINERFILE" . ; then errors+=("podman build failed"); fi
    else
      if ! docker build -t "$TAG" -f "$CONTAINERFILE" . ; then errors+=("docker build failed"); fi
    fi
  fi
fi

ok=1
if [[ ${#errors[@]} -gt 0 ]]; then ok=0; fi

# Emit JSON summary
PYTHON=$(command -v python3 || command -v python || true)
if [[ -z "$PYTHON" ]]; then echo "No python available to render JSON" >&2; exit 2; fi

export OK="$ok"
export IMAGE_VAR="$IMAGE"
export TAG_VAR="$TAG"
export UPDATE_CF_VAR="$UPDATE_CONTAINERFILE"
export BUILD_VAR="$BUILD"
export RUNTIME_VAR="$runtime"
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
  tag = os.environ.get('TAG_VAR',''),
  update_containerfile = os.environ.get('UPDATE_CF_VAR','0') == '1',
  build = os.environ.get('BUILD_VAR','0') == '1',
  runtime = os.environ.get('RUNTIME_VAR',''),
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
