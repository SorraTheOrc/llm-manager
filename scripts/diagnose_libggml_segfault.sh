#!/bin/sh
# Diagnosis harness for the llama-server SIGSEGV in libggml-cpu (LP-0MVZVZLCS000LQNB).
#
# Modes
# -----
#   (default)  evidence-only, READ-ONLY: collect the kernel segfault lines, the
#              deployed artifact versions, the quant-kernel symbol census, the
#              Q4_K presets in models.ini and the last self-heal lines from
#              proxy.log. Sends no request, starts no process.
#   --replay   the documented bounded reproduction: load the Q4_K preset through
#              the router and issue a bounded number of requests with the fixed
#              prompt shape, then diff kern.log and the llama-server log tail.
#              DESTRUCTIVE — it can crash the live backend. Opt-in only.
#
# Why the replay is opt-in: the host runs the operator's production proxy, and a
# crash aborts any in-flight measurement window. See
# docs/segfault-libggml-cpu-LP-0MVZVZLCS000LQNB.md §5.
#
# Exit codes: 0 = evidence collected (or replay completed without a new
#             segfault), 3 = replay reproduced a new segfault, 2 = usage/error.
set -eu

KERN_LOG="${KERN_LOG:-/var/log/kern.log}"
PROXY_LOG_DIR="${PROXY_LOG_DIR:-/var/log/llama-proxy}"
GGML_LIB="${GGML_LIB:-/usr/local/lib64/libggml-cpu.so.0.9.11}"
LLAMA_BIN="${LLAMA_BIN:-/usr/local/bin/llama-server}"
MODELS_INI="${MODELS_INI:-$(dirname "$0")/../models.ini}"
BASE_URL="${BASE_URL:-http://localhost:8080}"
REPLAY_REQUESTS="${REPLAY_REQUESTS:-3}"
REPLAY_PROMPT_TOKENS="${REPLAY_PROMPT_TOKENS:-16000}"
REPLAY_MODEL="${REPLAY_MODEL:-Qwen3-MTP}"
REPLAY_TIMEOUT="${REPLAY_TIMEOUT:-600}"

usage() {
    cat >&2 <<'EOF'
usage: diagnose_libggml_segfault.sh [--replay]

  (no args)  read-only evidence collection (safe on a live host)
  --replay   bounded destructive reproduction against the live router
EOF
}

section() {
    printf '\n=== %s ===\n' "$1"
}

collect_evidence() {
    section "kernel segfaults (llama-server)"
    if [ -r "$KERN_LOG" ]; then
        grep -i "llama-server.*segfault" "$KERN_LOG" | tail -20 || echo "(none)"
    else
        echo "(cannot read $KERN_LOG)"
    fi

    section "deployed artifacts"
    for f in "$LLAMA_BIN" "$GGML_LIB"; do
        if [ -e "$f" ]; then
            ls -l "$f"
        else
            echo "missing: $f"
        fi
    done

    section "quant dot-product kernel census (generic vs arch-specific)"
    if [ -r "$GGML_LIB" ]; then
        generic=$(nm -C --defined-only "$GGML_LIB" 2>/dev/null | grep -c "_generic" || true)
        arch=$(nm -C --defined-only "$GGML_LIB" 2>/dev/null \
            | grep "ggml_vec_dot" | grep -cE "avx|neon|sve|rvv" || true)
        echo "generic kernels:        ${generic:-0}"
        echo "arch-specific kernels:  ${arch:-0}"
        echo "(0 arch-specific kernels => every quantised dot product runs the"
        echo " compiler-vectorised generic reference code; see the report §2)"
    else
        echo "missing: $GGML_LIB"
    fi

    section "Q4_K presets in models.ini"
    if [ -r "$MODELS_INI" ]; then
        grep -nE "^\[|Q4_K" "$MODELS_INI" | grep -B1 "Q4_K" || echo "(none)"
    else
        echo "missing: $MODELS_INI"
    fi

    section "recent self-heal / restart lines"
    if [ -r "$PROXY_LOG_DIR/proxy.log" ]; then
        grep -E "self-heal|host-first startup|process died|restart_reason" \
            "$PROXY_LOG_DIR/proxy.log" | tail -20 || echo "(none)"
    else
        echo "(cannot read $PROXY_LOG_DIR/proxy.log)"
    fi

    section "distinct llama-server logs in the proxy log dir"
    ls -1t "$PROXY_LOG_DIR"/llama-server*.log 2>/dev/null | wc -l || echo 0
}

replay() {
    echo "REPLAY: destructive reproduction — model=${REPLAY_MODEL} requests=${REPLAY_REQUESTS}" >&2
    echo "REPLAY: prompt shape ~${REPLAY_PROMPT_TOKENS} filler tokens + short instruction," \
         "cache_prompt=false, max_tokens=8" >&2

    before_kern=$(grep -ci "llama-server.*segfault" "$KERN_LOG" 2>/dev/null || echo 0)
    before_logs=$(ls -1t "$PROXY_LOG_DIR"/llama-server*.log 2>/dev/null | head -1 || true)

    payload=$(python3 - "$REPLAY_PROMPT_TOKENS" "$REPLAY_MODEL" <<'PY'
import json
import sys

tokens = int(sys.argv[1])
model = sys.argv[2]
# Deterministic filler: unique enough that the prompt cache cannot mask the
# prefill, shaped like the long-context dispatches that dominate this host.
filler = " ".join(f"w{i % 977}" for i in range(tokens))
body = {
    "model": model,
    "messages": [{"role": "user", "content": filler + "\n\nReply with the single word: ok"}],
    "max_tokens": 8,
    "temperature": 0,
    "cache_prompt": False,
}
print(json.dumps(body))
PY
)

    i=1
    while [ "$i" -le "$REPLAY_REQUESTS" ]; do
        echo "REPLAY: request $i/$REPLAY_REQUESTS" >&2
        curl -sS -m "$REPLAY_TIMEOUT" -o /dev/null -w "  http=%{http_code} time=%{time_total}s\n" \
            -H 'Content-Type: application/json' \
            -X POST "$BASE_URL/v1/chat/completions" \
            -d "$payload" || echo "  request failed (connection reset is the crash signature)" >&2
        i=$((i + 1))
    done

    after_kern=$(grep -ci "llama-server.*segfault" "$KERN_LOG" 2>/dev/null || echo 0)
    section "replay result"
    echo "segfaults before: $before_kern"
    echo "segfaults after:  $after_kern"
    if [ "$after_kern" -gt "$before_kern" ]; then
        echo "REPRODUCED: a new llama-server segfault was recorded."
        grep -i "llama-server.*segfault" "$KERN_LOG" | tail -3
        exit 3
    fi
    echo "NOT REPRODUCED within the bounded attempt (${REPLAY_REQUESTS} requests)."
    if [ -n "$before_logs" ]; then
        echo "--- last llama-server log tail ($before_logs) ---"
        tail -20 "$before_logs" || true
    fi
    exit 0
}

case "${1:-}" in
    "") collect_evidence ;;
    --replay) collect_evidence; replay ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
esac
