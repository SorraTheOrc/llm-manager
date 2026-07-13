#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# prometheus_snapshot.sh — Poll metrics endpoints during benchmark runs
#
# Collects a snapshot of metrics from:
#   - /admin/metrics (llama-server Prometheus endpoint)
#   - rocminfo (GPU VRAM usage via rocm-smi)
#   - System memory via /proc/meminfo
#
# Usage:
#   ./prometheus_snapshot.sh [--output <file>] [--interval <seconds>]
#
# Options:
#   --output <file>    Write output to the specified file (default: stdout)
#   --interval <sec>   Poll interval in seconds, collects 3 samples (default: 5)
#   --admin-port <p>   llama-server admin port (default: 8080)
#
# Dependencies:
#   - curl (for /admin/metrics)
#   - rocm-smi (optional, for GPU metrics)
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
OUTPUT_FILE=""
POLL_INTERVAL=5
ADMIN_PORT=8080
SAMPLES=3

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --interval)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        --admin-port)
            ADMIN_PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $(basename "$0") [--output <file>] [--interval <sec>] [--admin-port <port>]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

timestamp_ms() {
    date +"%Y-%m-%dT%H:%M:%S.%3N%z"
}

log() {
    local msg="$1"
    if [[ -n "$OUTPUT_FILE" ]]; then
        echo "[$(timestamp_ms)] $msg" >> "$OUTPUT_FILE"
    else
        echo "[$(timestamp_ms)] $msg"
    fi
}

# ---------------------------------------------------------------------------
# Metric collectors
# ---------------------------------------------------------------------------

collect_admin_metrics() {
    local admin_url="http://127.0.0.1:${ADMIN_PORT}/metrics"
    local output

    output=$(curl -s --max-time 5 "$admin_url" 2>/dev/null || true)

    if [[ -z "$output" ]]; then
        log "WARN: /metrics endpoint not reachable at ${admin_url}"
        return
    fi

    # Extract key metrics
    local rss kv_used kv_capacity n_loaded
    rss=$(echo "$output" | grep -E '^llama_process_rss_bytes' | tail -1 | awk '{print $2}' || echo "")
    kv_used=$(echo "$output" | grep -E '^llama_kv_cache_used_bytes' | tail -1 | awk '{print $2}' || echo "")
    kv_capacity=$(echo "$output" | grep -E '^llama_kv_cache_capacity_bytes' | tail -1 | awk '{print $2}' || echo "")
    n_loaded=$(echo "$output" | grep -E '^llama_models_loaded' | tail -1 | awk '{print $2}' || echo "")

    log "=== Admin Metrics Snapshot ==="
    if [[ -n "$rss" ]]; then
        log "llama_process_rss_bytes: $rss"
    fi
    if [[ -n "$kv_used" ]]; then
        log "llama_kv_cache_used_bytes: $kv_used"
    fi
    if [[ -n "$kv_capacity" ]]; then
        log "llama_kv_cache_capacity_bytes: $kv_capacity"
    fi
    if [[ -n "$n_loaded" ]]; then
        log "llama_models_loaded: $n_loaded"
    fi
    log "=== End Admin Metrics ==="
}

collect_gpu_metrics() {
    if ! command -v rocm-smi &>/dev/null; then
        log "INFO: rocm-smi not found — skipping GPU metrics"
        return
    fi

    log "=== GPU Metrics (rocm-smi) ==="

    # VRAM usage
    local vram_output
    vram_output=$(rocm-smi --showmeminfo vram 2>/dev/null || true)
    if [[ -n "$vram_output" ]]; then
        log "VRAM Info:"
        while IFS= read -r line; do
            log "  $line"
        done <<< "$vram_output"
    else
        log "  VRAM info unavailable"
    fi

    # GPU use percentage
    local use_output
    use_output=$(rocm-smi --showuse 2>/dev/null || true)
    if [[ -n "$use_output" ]]; then
        log "GPU Use:"
        while IFS= read -r line; do
            log "  $line"
        done <<< "$use_output"
    fi

    log "=== End GPU Metrics ==="
}

collect_system_memory() {
    log "=== System Memory ==="
    if [[ -f /proc/meminfo ]]; then
        grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo \
            | while IFS= read -r line; do
                log "  $line"
            done
    else
        log "  /proc/meminfo not available"
    fi
    log "=== End System Memory ==="
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

log "=== Prometheus Snapshot Started ==="
log "Admin port: ${ADMIN_PORT}"
log "Samples: ${SAMPLES} at ${POLL_INTERVAL}s intervals"
log ""

for ((i = 1; i <= SAMPLES; i++)); do
    log "=== Sample $i of $SAMPLES ==="
    collect_admin_metrics
    collect_gpu_metrics
    collect_system_memory
    log ""

    if [[ $i -lt $SAMPLES ]]; then
        sleep "$POLL_INTERVAL"
    fi
done

log "=== Prometheus Snapshot Complete ==="
