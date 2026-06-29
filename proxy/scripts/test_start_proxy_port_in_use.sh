#!/usr/bin/env bash
set -euo pipefail

# This test uses an alternate port to avoid colliding with any already-running
# proxy instance on port 8000 in the developer environment. It verifies the
# start-proxy.sh script exits non-zero and emits a helpful message when the
# target port is already in use.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_SCRIPT="$SCRIPT_DIR/start-proxy.sh"

# Choose a test port (avoid 8000 in case a dev proxy is running)
TEST_PORT=8008

# Start a background listener on 127.0.0.1:$TEST_PORT
python3 - <<PY &
import socket, time, sys
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(('127.0.0.1', $TEST_PORT))
except Exception as e:
    print('Failed to bind test server on port', $TEST_PORT, '->', e)
    sys.exit(2)
s.listen(1)
# Keep the server alive for a short time
time.sleep(30)
PY
server_pid=$!
# Give it a moment to bind
sleep 0.2

# Run start-proxy.sh with a timeout to avoid indefinite blocking
# Pass --port to override the default 8000 so we use the test port
TMP_OUT=$(mktemp)
rc=0
if command -v timeout >/dev/null 2>&1; then
  timeout 8 "$PROXY_SCRIPT" --port $TEST_PORT > "$TMP_OUT" 2>&1 || rc=$?
else
  # Fallback: run and rely on internal exit
  "$PROXY_SCRIPT" --port $TEST_PORT > "$TMP_OUT" 2>&1 || rc=$?
fi

# Clean up background server
kill "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true

echo "-- START-PROXY OUTPUT (rc=$rc) --"
cat "$TMP_OUT"

# Assert exit code non-zero
if [ "$rc" -eq 0 ]; then
  echo "Test failed: expected non-zero exit when port in use"
  rm -f "$TMP_OUT"
  exit 2
fi

# Check output contains helpful message mentioning the tested port or a clear error
if ! (grep -qi "port $TEST_PORT" "$TMP_OUT" || grep -qi "address already in use" "$TMP_OUT" || grep -qi "already in use" "$TMP_OUT"); then
  echo "Test failed: expected helpful port message"
  rm -f "$TMP_OUT"
  exit 3
fi

rm -f "$TMP_OUT"
echo "Test passed"
exit 0
