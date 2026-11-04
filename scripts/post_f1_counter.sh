#!/bin/sh
# Post-F1 slots_stale counter (LP-0MVVXGI490089OX5, F2 of LP-0MVVCO1N7004TNGY).
#
# The F1 per-phase instrumentation first became live at 2026-11-02 10:40:27
# (proxy restart 10:39:24, first instrumented status_request 10:40:27). This
# counter samples the canonical verification tool every SAMPLE_SECONDS and
# appends one JSONL reading per sample, so the 24 h window that closes at
# 2026-11-03 10:40:27 is measurable even if the proxy log rotates away.
#
# Read-only: it only reads /var/log/llama-proxy and writes under OUT_DIR.
# Self-terminating: it stops once the window has closed (or --until is past).
#
# Usage: post_f1_counter.sh [--until 'YYYY-MM-DD HH:MM:SS']
set -u

SINCE="2026-11-02 10:40:27"
UNTIL="2026-11-03 10:40:27"
SAMPLE_SECONDS="${SAMPLE_SECONDS:-900}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$HOME/proxy-usage-reports/post-f1-counter}"
READINGS="$OUT_DIR/readings.jsonl"

while [ $# -gt 0 ]; do
    case "$1" in
        --until) UNTIL="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUT_DIR" || exit 1

now_epoch() { date +%s; }
until_epoch=$(date -d "$UNTIL" +%s) || exit 2

echo "counter: since='$SINCE' until='$UNTIL' every ${SAMPLE_SECONDS}s -> $READINGS"

while :; do
    now=$(date '+%Y-%m-%d %H:%M:%S')
    if [ "$(now_epoch)" -ge "$until_epoch" ]; then
        echo "counter: window closed at $UNTIL — final sample"
    fi

    reading=$(cd "$REPO_ROOT" && python3 scripts/capacity_signal_verification.py \
        --since "$SINCE" --until "$now" --label post-f1-counter --json 2>/dev/null)

    if [ -n "$reading" ]; then
        printf '%s\n' "$reading" | python3 -c '
import json, sys
d = json.load(sys.stdin)
m = d.get("metrics", {})
pp = m.get("per_phase_read_path", {}) or {}
phases = pp.get("phases", {}) or {}

def band(name, group):
    g = phases.get(name, {}).get(group, {}) or {}
    return {"p50": g.get("p50"), "p90": g.get("p90"), "p99": g.get("p99")}

out = {
    "sampled_at": sys.argv[1],
    "window_since": sys.argv[2],
    "status_polls": m.get("status_polls"),
    "slots_stale": m.get("slots_stale"),
    "by_source": (m.get("slots_stale_attribution") or {}).get("by_source"),
    "by_latency_band": (m.get("slots_stale_attribution") or {}).get("by_latency_band"),
    "instrumented": pp.get("instrumented"),
    "stale": {
        "counts_ms": band("slots_counts_ms", "stale"),
        "detail_ms": band("slots_detail_ms", "stale"),
        "status_query_ms": band("status_query_ms", "stale"),
    },
    "fresh": {
        "counts_ms": band("slots_counts_ms", "fresh"),
        "detail_ms": band("slots_detail_ms", "fresh"),
        "status_query_ms": band("status_query_ms", "fresh"),
    },
    "upstream_calls_stale": (pp.get("upstream_calls") or {}).get("stale"),
    "snapshot_source_stale": (pp.get("snapshot_source") or {}).get("stale"),
}
print(json.dumps(out, sort_keys=True))
' "$now" "$SINCE" >> "$READINGS" || echo "counter: sample at $now failed to render" >&2
        echo "counter: sampled $now -> $(wc -l < "$READINGS") reading(s)"
    else
        echo "counter: sample at $now produced no reading (tool error)" >&2
    fi

    [ "$(now_epoch)" -ge "$until_epoch" ] && break
    sleep "$SAMPLE_SECONDS"
done

echo "counter: done — $READINGS"
