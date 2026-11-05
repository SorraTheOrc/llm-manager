#!/bin/sh
# Diagnosis harness for the llama-server SIGSEGV in libggml-cpu (LP-0MVZVZLCS000LQNB,
# re-diagnosis for LP-0MW3K7Y3C003WEXW).
#
# Modes
# -----
#   (default)  evidence-only, READ-ONLY: collect the kernel segfault lines, the
#              deployed artifact versions, the quant-kernel census (via the shared
#              classifier scripts/lib/ggml_cpu_census.py), the enclosing symbol for
#              every crash offset the kernel printed, the Q4_K presets in models.ini
#              and the last self-heal lines from proxy.log. Sends no request,
#              starts no process.
#   --replay   the documented bounded reproduction: load the Q4_K preset through
#              the router and issue a bounded number of requests with the fixed
#              prompt shape, then diff kern.log and the llama-server log tail.
#              DESTRUCTIVE — it can crash the live backend. Opt-in only.
#
# Why the census moved out of this script (LP-0MW3K7Y3C003WEXW): the inline
# census counted symbols matching `ggml_vec_dot.*avx`, which no upstream build
# ever contains (the arch kernel is plain `ggml_vec_dot_q4_K_q8_K`; only the
# fallback carries `_generic`). It reported "0 arch-specific kernels" for a
# perfectly healthy `-march=native` build, and it inspected
# /usr/local/lib64 — not the artifact the running server actually maps. Both
# faults are fixed by delegating to the shared classifier and by resolving the
# artifact from /proc/<pid>/maps.
#
# Why the replay is opt-in: the host runs the operator's production proxy, and a
# crash aborts any in-flight measurement window. See
# docs/segfault-libggml-cpu-LP-0MVZVZLCS000LQNB.md §5 and
# docs/build-record-llama-cpu-LP-0MW3K7Y3C003WEXW.md.
#
# Exit codes: 0 = evidence collected (or replay completed without a new
#             segfault), 3 = replay reproduced a new segfault, 2 = usage/error.
set -eu

KERN_LOG="${KERN_LOG:-/var/log/kern.log}"
PROXY_LOG_DIR="${PROXY_LOG_DIR:-/var/log/llama-proxy}"
MODELS_INI="${MODELS_INI:-$(dirname "$0")/../models.ini}"
BASE_URL="${BASE_URL:-http://localhost:8080}"
REPLAY_REQUESTS="${REPLAY_REQUESTS:-3}"
REPLAY_PROMPT_TOKENS="${REPLAY_PROMPT_TOKENS:-16000}"
REPLAY_MODEL="${REPLAY_MODEL:-Qwen3-MTP}"
REPLAY_TIMEOUT="${REPLAY_TIMEOUT:-600}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CENSUS_SCRIPT="${GGML_CENSUS_SCRIPT:-${SCRIPT_DIR}/lib/ggml_cpu_census.py}"

# Resolve the artifacts of the RUNNING server, not a copy on disk that nothing
# maps: kernel crash offsets are only meaningful against the mapped artifact.
live_artifact() {
    _pattern="$1"
    for _pid in $(pgrep -x llama-server 2>/dev/null || true); do
        _path=$(grep -o "/[^ ]*${_pattern}[^ ]*" "/proc/${_pid}/maps" 2>/dev/null | head -1 || true)
        if [ -n "${_path:-}" ]; then
            printf '%s' "$_path"
            return 0
        fi
    done
    return 1
}

GGML_LIB="${GGML_LIB:-$(live_artifact 'libggml-cpu\.so' || echo /home/rgardler/llama.cpp/build/bin/libggml-cpu.so.0.9.11)}"
LLAMA_BIN="${LLAMA_BIN:-$(live_artifact 'llama-server$' || echo /home/rgardler/llama.cpp/build/bin/llama-server)}"

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

# File offsets the kernel printed for this library, e.g.
#   libggml-cpu.so.0.9.11[51057,ff000]  ->  51057
crash_offsets() {
    [ -r "$KERN_LOG" ] || return 0
    grep -oE 'libggml-cpu[^[]*\[[0-9a-f]+' "$KERN_LOG" 2>/dev/null \
        | grep -oE '[0-9a-f]+$' | sort -u | tail -8 | tr '\n' ' ' || true
}

collect_evidence() {
    section "kernel segfaults (llama-server)"
    if [ -r "$KERN_LOG" ]; then
        grep -iE "llama-server.*(segfault|general protection)" "$KERN_LOG" | tail -20 || echo "(none)"
    else
        echo "(cannot read $KERN_LOG)"
    fi

    section "artifacts of the RUNNING server (from /proc/<pid>/maps)"
    for f in "$LLAMA_BIN" "$GGML_LIB"; do
        if [ -e "$f" ]; then
            ls -l "$f"
        else
            echo "missing: $f"
        fi
    done
    echo "(override with LLAMA_BIN=... GGML_LIB=... to census a staged build)"

    section "quant dot-product kernel census (arch-specific codegen vs generic)"
    if [ -r "$GGML_LIB" ]; then
        # Text mode prints the verdict plus the hot-kernel vector-instruction counts.
        python3 "$CENSUS_SCRIPT" "$GGML_LIB" --no-require-arch-kernels \
            --expect-repack any || true
        echo "(arch_specific=no => every quantised dot product runs the generic"
        echo " reference loop; repack compiled in => the ggml_cpu_extra_work_size"
        echo " dispatch, the LP-0MW3K7Y3C003WEXW crash path, is present)"
    else
        echo "missing: $GGML_LIB"
    fi

    section "crash offset -> enclosing symbol (for THIS artifact)"
    offsets="$(crash_offsets)"
    if [ -z "${offsets// /}" ]; then
        echo "(no libggml-cpu offsets found in $KERN_LOG)"
    elif [ -r "$GGML_LIB" ]; then
        set -- "$GGML_LIB" --no-require-arch-kernels --expect-repack any
        for off in $offsets; do set -- "$@" --locate-offset "0x$off"; done
        python3 "$CENSUS_SCRIPT" "$@" || true
    else
        echo "cannot locate offsets: missing $GGML_LIB"
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

    before_kern=$(grep -ciE "llama-server.*(segfault|general protection)" "$KERN_LOG" 2>/dev/null || echo 0)
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

    after_kern=$(grep -ciE "llama-server.*(segfault|general protection)" "$KERN_LOG" 2>/dev/null || echo 0)
    section "replay result"
    echo "segfaults before: $before_kern"
    echo "segfaults after:  $after_kern"
    if [ "$after_kern" -gt "$before_kern" ]; then
        echo "REPRODUCED: a new llama-server fault was recorded."
        grep -iE "llama-server.*(segfault|general protection)" "$KERN_LOG" | tail -3
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
