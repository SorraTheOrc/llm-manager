#!/usr/bin/env bash
set -euo pipefail

# Reproducible local fault-injection run for crash-path mitigation validation.
# Captures health/metrics snapshots and log signatures around a forced llama-server crash.

BASE_URL="${BASE_URL:-http://localhost:8000}"
OUT_DIR="${OUT_DIR:-./logs/fault-injection}"
CONCURRENCY="${CONCURRENCY:-8}"
REQUESTS="${REQUESTS:-24}"

mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$OUT_DIR/run-$TS"
mkdir -p "$RUN_DIR"

echo "[fault-injection] output: $RUN_DIR"

echo "[fault-injection] pre-health"
curl -sS "$BASE_URL/health" | tee "$RUN_DIR/health-before.json" >/dev/null
curl -sS "$BASE_URL/admin/metrics" | tee "$RUN_DIR/metrics-before.json" >/dev/null

# Fire concurrent requests in the background to create load.
# We intentionally tolerate 503s so we can observe overload guard behavior.
echo "[fault-injection] firing request burst ($REQUESTS requests, concurrency=$CONCURRENCY)"
seq "$REQUESTS" | xargs -P "$CONCURRENCY" -I{} bash -c '
  curl -sS -X POST "'$BASE_URL'/v1/chat/completions" \
    -H "content-type: application/json" \
    -d "{\"model\":\"qwen3\",\"messages\":[{\"role\":\"user\",\"content\":\"fault-inject {}\"}],\"max_tokens\":8}" \
    -o /dev/null -w "%{http_code}\n" || true
' | tee "$RUN_DIR/request-status-codes.txt" >/dev/null &
BURST_PID=$!

sleep 2

echo "[fault-injection] forcing backend crash (pkill llama-server)"
pkill -9 -f llama-server || true

wait "$BURST_PID" || true

echo "[fault-injection] collecting post-fault snapshots"
for i in $(seq 1 15); do
  curl -sS "$BASE_URL/health" > "$RUN_DIR/health-$i.json" || true
  sleep 1
done
curl -sS "$BASE_URL/admin/metrics" | tee "$RUN_DIR/metrics-after.json" >/dev/null || true

# Capture expected triage signatures from proxy log if present.
LOG_FILE="./logs/proxy.log"
if [[ -f "$LOG_FILE" ]]; then
  grep -E "backend_retry|concurrency_reject|watchdog detected llama-server exit|watchdog router restart" "$LOG_FILE" \
    | tail -n 200 > "$RUN_DIR/signatures.log" || true
fi

echo "[fault-injection] done"
echo "- health snapshots: $RUN_DIR/health-*.json"
echo "- metrics: $RUN_DIR/metrics-before.json, $RUN_DIR/metrics-after.json"
echo "- status codes: $RUN_DIR/request-status-codes.txt"
echo "- signatures: $RUN_DIR/signatures.log"
