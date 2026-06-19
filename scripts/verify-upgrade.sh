#!/usr/bin/env bash
# Verify upgrade by querying llama-server endpoints.
# Supports --base-url, --model, --dry-run, --json

set -uo pipefail

BASE_URL="http://127.0.0.1:8080"
MODEL="qwen3"
DRY_RUN=0
JSON=0
OUTFILE=""
TIMEOUT=10

usage() {
  cat <<EOF
Usage: $0 [--base-url URL] [--model MODEL] [--dry-run] [--json] [--output FILE]
  --base-url URL   Base URL of llama-server (default: $BASE_URL)
  --model MODEL    Model to request (default: $MODEL)
  --dry-run        Do not perform network requests; emit planned steps
  --json           Emit JSON summary
  --output FILE    Write JSON output to FILE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --json) JSON=1; shift ;;
    --output) OUTFILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

planned_steps=("check_health" "send_streaming_request")

if [[ $DRY_RUN -eq 1 ]]; then
  ok=1
  data=$(cat <<JSON
{
  "ok": ${ok},
  "base_url": "${BASE_URL}",
  "model": "${MODEL}",
  "planned_steps": ["${planned_steps[*]}"],
  "errors": []
}
JSON
)
  if [[ -n "$OUTFILE" ]]; then printf '%s\n' "$data" > "$OUTFILE"; else printf '%s\n' "$data"; fi
  if [[ $ok -eq 1 ]]; then exit 0; else exit 2; fi
fi

# Real checks (best-effort): check /health or /v1/health and then perform a completion request
errors=()
health_ok=0
if curl -sS --max-time ${TIMEOUT} "${BASE_URL}/health" >/dev/null 2>&1 || curl -sS --max-time ${TIMEOUT} "${BASE_URL}/v1/health" >/dev/null 2>&1; then
  health_ok=1
else
  errors+=("health endpoint unreachable")
fi

# Prepare a simple completion request
req_body='{"model": "'"${MODEL}"'", "messages": [{"role":"user","content":"Hello"}], "max_tokens":8}'
res=$(curl -sS -X POST "${BASE_URL}/v1/chat/completions" -H "Content-Type: application/json" -d "$req_body" --max-time ${TIMEOUT} 2>&1)
rc=$?

if [[ $rc -ne 0 ]]; then
  errors+=("completion request failed: $res")
fi

# Detect common ROCm crash messages in response
crash_detected=0
if echo "$res" | grep -qi "hipStreamSynchronize\|unspecified launch failure\|hipError"; then crash_detected=1; fi

ok=1
if [[ ${#errors[@]} -gt 0 || $crash_detected -eq 1 || $health_ok -eq 0 ]]; then ok=0; fi

# Emit JSON summary
PYTHON=$(command -v python3 || command -v python || true)
if [[ -z "$PYTHON" ]]; then echo "No python available to render JSON" >&2; exit 2; fi

export OK="$ok"
export BASE_URL_VAR="$BASE_URL"
export MODEL_VAR="$MODEL"
export HEALTH_OK="$health_ok"
export CRASH_DETECTED="$crash_detected"
if [[ ${#errors[@]} -gt 0 ]]; then export ERRORS="$(IFS='||'; echo "${errors[*]}")"; else export ERRORS=""; fi

OUTPUT=$($PYTHON - <<PY
import os,json
out = dict(
  ok = os.environ.get('OK','0') == '1',
  base_url = os.environ.get('BASE_URL_VAR',''),
  model = os.environ.get('MODEL_VAR',''),
  health_ok = os.environ.get('HEALTH_OK','0') == '1',
  crash_detected = os.environ.get('CRASH_DETECTED','0') == '1',
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
