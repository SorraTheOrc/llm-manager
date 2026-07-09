# GPU Offload Verification Plan

This document defines the executable verification path for router-mode ROCm
GPU offload. It is the output of work item **GPU Offload Verification Harness
(LP-0MRE1ECDR0071U3O)**.

## Overview

The proxy deploys llama-server in router mode (multi-model). The goal is to
offload LLM inference from CPU to the AMD GPU using ROCm/HIP. This document
describes how to verify that offload is actually happening, measure the
improvement, and validate rollback to CPU-only mode.

## Reused Tests & Documentation

The following existing tests and docs are reused without duplication:

| Resource | Purpose | File |
|----------|---------|------|
| Mocked host-flow tests | Validate startup logic (no GPU) | `proxy/tests/test_host_flow_integration.py` |
| Live end-to-end tests | Validate health, chat, embeddings (live) | `proxy/tests/test_host_flow_live_e2e.py` |
| Integration test doc | How to run mocked/live tests | `docs/INTEGRATION.md` |
| Router mode docs | Router operation, models.ini, preloading | `docs/llama-router.md` |
| models.ini config tests | Centralized config validation | `tests/test_models_ini_centralized_config.sh` |
| GPU offload unit tests | Guard tests for ngl propagation | `proxy/tests/test_gpu_offload_verification.py` |
| ROCm upgrade docs | Build and upgrade process | `UPGRADE_ROCM.md` |

### What is NEW in this verification harness

The following are new contributions from this work item:

- **Python unit tests** (`proxy/tests/test_gpu_offload_verification.py`): Guard
  tests that validate models.ini `[global] ngl` parsing, ROCm env var
  propagation, and router-mode command construction. These tests will FAIL if
  the GPU offload settings are accidentally removed.
- **Bash models.ini tests** (`tests/test_models_ini_centralized_config.sh`):
  Extended with `[global] ngl` parsing tests.
- This document: Live verification steps, baseline measurement guidance, and
  rollback validation.

---

## 1. Prototype / Experiment

Before implementing full router-mode offload, confirm whether the current
`./start-llama.sh router` + `--models-preset ./models.ini` causes worker
processes to inherit `[global] ngl = 99` without extra flags.

### 1.1 Check llama-server binary capabilities

```bash
# Verify the built binary was compiled with ROCm/HIP
/home/rgardler/llama.cpp/build/bin/llama-server --version

# Check if HIP is reported in the build features
/home/rgardler/llama.cpp/build/bin/llama-server --help 2>&1 | grep -i "gpu\|hip\|rocm"
```

Look for build flags like `-DGGML_HIP=ON` or `HIP` in the output.

### 1.2 Start router mode

```bash
# Start llama-server in router mode with models.ini
cd /home/rgardler/projects/llm
LLAMA_SERVER_BIN=/home/rgardler/llama.cpp/build/bin/llama-server \
  ./start-llama.sh router &
```

### 1.3 Check worker process flags

```bash
# Find llama-server worker processes
ps aux | grep llama-server

# Check if workers have -ngl flag
ps aux | grep llama-server | grep -E "ngl|gpu-layers"
```

If the workers show `-ngl 99` (or similar), then router mode already honors
`[global] ngl`. If not, the flag must be explicitly forwarded in the router
startup command.

### 1.4 Check GPU memory usage

```bash
# ROCm tool to check GPU memory usage
rocm-smi

# Or check via /sys
cat /sys/class/drm/card*/device/memory_info_vis_vram_total 2>/dev/null || true
cat /sys/class/drm/card*/device/memory_info_vis_vram_used 2>/dev/null || true
```

If the GPU VRAM shows model weights loaded after startup, ROCm offload is
working. If VRAM usage stays near idle, inference is happening on CPU.

---

## 2. Live Verification Steps

These steps require a running proxy and llama-server with GPU access.

### 2.1 Pre-requisites

- GPU with ROCm drivers installed
- `rocm-smi` (part of ROCm tools) available
- llama-server compiled with `-DGGML_HIP=ON`
- Proxy configured (`proxy/config.yaml`) with `llama_router_mode: true`
- `models.ini` has `[global] ngl = 99`

### 2.2 Health Checks

```bash
# Check llama-server health
curl -s http://localhost:8080/health | python3 -m json.tool

# Check proxy health
curl -s http://localhost:8000/health | python3 -m json.tool
```

Expected:
- llama-server: `{"status": "ok"}` (or similar success indicator)
- Proxy: `{"status": "ok", "backend_reachable": true, "ready": true}`

### 2.3 Verify ROCm Usage on a Chat Completion

```bash
# Before the request, check GPU memory baseline
rocm-smi --showmeminfo vram

# Send a chat completion request
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3",
    "messages": [{"role": "user", "content": "Say hello in 5 words"}],
    "max_tokens": 50
  }' | python3 -m json.tool

# After the request, check GPU memory usage
rocm-smi --showmeminfo vram
```

Evidence of GPU offload:
- VRAM usage increases significantly after model load (model weights loaded to GPU)
- Inference completes much faster than CPU-only

### 2.4 Verify Embeddings ROCm Usage

```bash
# Send an embeddings request
curl -s -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embeddings",
    "input": "Hello, world!"
  }' | python3 -m json.tool
```

Expected:
- Response contains `data[0].embedding` with a vector of floats
- Processing is faster with GPU offload

### 2.5 Confirm Inference Source via Logs

Check the proxy logs for model loading and inference info:

```bash
tail -100 /var/log/llama-proxy/llama-server.log
```

Look for evidence that the GPU backend is being used (HIP/ROCm messages).

---

## 3. Baseline Measurement

To measure the CPU reduction from GPU offload, capture these metrics before
and after enabling offload.

### 3.1 CPU Usage Baseline (before offload)

With `ngl = 0` (or commented out) in `[global]` section:

```bash
# Capture CPU usage during inference
# Method 1: top in batch mode
top -b -n 60 -d 1 | grep llama-server > cpu_baseline_before.txt

# Method 2: Use pidstat
pidstat -p $(pgrep -f llama-server | head -1) 1 60 > cpu_baseline_before.csv

# Method 3: mpstat for overall CPU
mpstat 1 60 > cpu_baseline_system_before.csv
```

### 3.2 Throughput Baseline (before offload)

```bash
# Time a standard chat completion request with ~2K story prompt
# Save the prompt to a file
cat > /tmp/test_prompt.json << 'EOF'
{
  "model": "Qwen3",
  "messages": [
    {"role": "user", "content": "Write a short story about a robot learning to paint. Make it about 500 words."}
  ],
  "max_tokens": 1000
}
EOF

# Time it
time curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/test_prompt.json > /dev/null
```

Record:
- Wall clock time
- CPU usage during inference
- GPU VRAM usage (should be minimal)

### 3.3 Measurements After Offload

Enable offload (`ngl = 99` in `[global]`), restart llama-server, and repeat
steps 3.1 and 3.2.

### 3.4 Comparison Checklist

| Metric | Before (CPU) | After (GPU) | Improvement |
|--------|-------------|-------------|-------------|
| Inference time (s) | | | |
| CPU usage (%) | | | |
| GPU VRAM usage (MiB) | | | |
| Tokens/second | | | |

---

## 4. CPU-Only Rollback Validation

### 4.1 Verify Rollback via models.ini

```bash
# Set [global] ngl = 0 in models.ini
# Restart llama-server
LLAMA_SERVER_BIN=/home/rgardler/llama.cpp/build/bin/llama-server \
  ./start-llama.sh router &

# Verify that chat and embeddings still work
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}'

# Check GPU usage (should be minimal with ngl=0)
rocm-smi --showmeminfo vram
```

### 4.2 Rollback to Pre-Offload State

If ROCm instability reappears:

1. Set `ngl = 0` in `[global]` section of `models.ini`
2. Restart the proxy: `systemctl --user restart llama-proxy`
3. Verify health endpoints return 200
4. Verify chat and embeddings requests complete successfully (CPU-only)
5. Confirm GPU memory stays at idle levels

---

## 5. Guard Test Interpretation

The test file `proxy/tests/test_gpu_offload_verification.py` contains two
guard tests that are currently **skipped**:

1. `test_models_ini_global_ngl_is_forwarded_to_router` — Skipped because
   router mode does not yet forward `-ngl`. Once
   **LP-0MRE1ECIQ000B8MU (Qwen3 Router GPU Offload)** is implemented, the
   skip should be removed and the assertion enabled.
2. `test_ngl_zero_rollback_possible_via_env` — Skipped until an env-var
   override for ngl in router mode is implemented.

When these tests change from SKIPPED to PASSING, the offload and rollback
implementation is complete.

## 6. Integration with Existing Monitoring

The following pre-existing monitoring work is reused:

- **GPU VRAM/ROCm metrics**: Prometheus metrics and Grafana dashboards from
  work item LP-0MQMH11AO002IIID. After offload is enabled, monitor
  `rocm_vram_usage_bytes` and `rocm_device_temperature_celsius` to confirm
  GPU utilization.
- **Self-heal / health recovery**: Backend crash recovery from work items
  LP-0MPN8BALG000DBB3 and LP-0MQ0ZHA4H006TUNE. The watchdog will detect
  GPU-related crashes and restart workers automatically.
