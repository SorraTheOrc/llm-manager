#!/bin/bash
set -euo pipefail

model_arg="${1:-}"
model="$(printf '%s' "${model_arg}" | tr '[:upper:]' '[:lower:]')"
router_mode=0

if [[ -z "${model}" || "${model}" == "router" ]]; then
  router_mode=1
fi

##
# Helper: read global ngl (GPU layers) from models.ini [global] section
# Usage: get_global_ngl [models-ini-path]
# Returns the ngl value as a string (e.g., "99"), or empty string if not found.
##
get_global_ngl() {
  local ini_file="${1:-models.ini}"

  if [[ ! -f "$ini_file" ]]; then
    return 1
  fi

  # Use awk to find the [global] section and extract ngl
  awk 'BEGIN { found=0; val="" }
  /^\[/ {
    gsub(/\[|\]/, "")
    if (tolower($0) == "global") {
      found=1
    } else {
      found=0
    }
  }
  found && /^ngl[ \t]*=/ {
    gsub(/.*=/, "")
    gsub(/^[ \t]+|[ \t]+$/, "")
    val=$0
    exit
  }
  END { if (val != "") print val }' "$ini_file"
}

##
# Helper: read ctx-size from models.ini for a given model name
# Usage: get_ctx_size <model-name> [models-ini-path]
##
get_ctx_size() {
  local target_model="$1"
  local ini_file="${2:-models.ini}"

  if [[ ! -f "$ini_file" ]]; then
    return 1
  fi

  # Use awk to find the matching [section] and extract ctx-size
  # Case-insensitive section matching
  awk -v target="$target_model" 'BEGIN { found=0; ctx="" }
  /^\[/ {
    gsub(/\[|\]/, "")
    if (tolower($0) == tolower(target)) {
      found=1
    } else {
      found=0
    }
  }
  found && /^ctx-size/ {
    gsub(/.*=/, "")
    gsub(/^[ \t]+|[ \t]+$/, "")
    ctx=$0
    exit
  }
  END { if (ctx != "") print ctx }' "$ini_file"
}

##
# Helper: read hf-repo from models.ini for a given model name
# Usage: get_hf_repo <model-name> [models-ini-path]
##
get_hf_repo() {
  local target_model="$1"
  local ini_file="${2:-models.ini}"

  if [[ ! -f "$ini_file" ]]; then
    return 1
  fi

  # Use awk to find the matching [section] and extract hf-repo
  # Case-insensitive section matching
  awk -v target="$target_model" 'BEGIN { found=0; repo="" }
  /^\[/ {
    gsub(/\[|\]/, "")
    if (tolower($0) == tolower(target)) {
      found=1
    } else {
      found=0
    }
  }
  found && /^hf-repo/ {
    gsub(/.*=/, "")
    gsub(/^[ \t]+|[ \t]+$/, "")
    repo=$0
    exit
  }
  END { if (repo != "") print repo }' "$ini_file"
}

##
# Configure the server
##
: "${PORT:=8080}"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export ROCM_LLVM_PRE_VEGA=1
export IP_ADDRESS=$(ip addr show | grep "inet " | grep -v 127.0.0.1 | awk 'NR==1 {print $2}' | cut -d'/' -f1)

if [[ "$router_mode" -eq 1 ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  MODELS_INI="${LLAMA_MODELS_PRESET:-${2:-"$SCRIPT_DIR/models.ini"}}"
  MODELS_MAX="${LLAMA_MODELS_MAX:-1}"

  echo "Server Environment"
  echo
  echo "IP_ADDRESS=$IP_ADDRESS"
  echo "PORT=$PORT"
  echo
  echo "ROUTER_MODE=true"
  echo "MODELS_INI=$MODELS_INI"
  echo "MODELS_MAX=$MODELS_MAX"
  echo
  echo
  echo

  # Prefer an explicit LLAMA_SERVER_BIN when provided (proxy exports this env var
  # to use a host-built binary). Fall back
  # to the system `llama-server` on PATH or the literal name.
  if [[ -n "${LLAMA_SERVER_BIN:-}" ]]; then
    LLAMA_BIN="${LLAMA_SERVER_BIN}"
  else
    LLAMA_BIN="$(command -v llama-server 2>/dev/null || true)"
    if [[ -z "$LLAMA_BIN" ]]; then
      LLAMA_BIN="llama-server"
    fi
  fi

  echo "Using llama-server binary: $LLAMA_BIN"

  # Read GPU offload level from models.ini [global] ngl.
  # Supports LLAMA_NGL env var override for CPU-only rollback (LLAMA_NGL=0).
  # Falls back to ngl=0 (CPU-only) when models.ini is absent or lacks [global] ngl.
  GLOBAL_NGL="${LLAMA_NGL:-$(get_global_ngl "$MODELS_INI")}"
  : "${GLOBAL_NGL:=0}"
  echo "GPU layers (ngl): $GLOBAL_NGL"

  # Default parallelism. Can be overridden via LLAMA_PARALLEL env var (set
  # by the proxy lifecycle from server.session_slot_pool_size in config.yaml).
  # Keeping these values aligned is critical: mismatch causes slot exhaustion
  # or restore failures at runtime.
  : "${LLAMA_PARALLEL:=1}"
  LLAMA_CMD=(
    "$LLAMA_BIN"
    --models-preset "$MODELS_INI"
    --models-max "$MODELS_MAX"
    --models-autoload
    --parallel "$LLAMA_PARALLEL"
    --host 0.0.0.0
    --no-mmap
    -ngl "$GLOBAL_NGL"
    --port $PORT
  )

  if [[ -n "${LLAMA_MODELS_DIR:-}" ]]; then
    LLAMA_CMD+=(--models-dir "$LLAMA_MODELS_DIR")
  fi

  if [[ -n "${LLAMA_SLOT_SAVE_PATH:-}" ]]; then
    mkdir -p "$LLAMA_SLOT_SAVE_PATH"
    LLAMA_CMD+=(--slot-save-path "$LLAMA_SLOT_SAVE_PATH")
  fi

  "${LLAMA_CMD[@]}"
  exit 0
fi

##
# Configure the model defaults
##
case "$model" in
  gpt120)
    # See https://github.com/ggml-org/llama.cpp/discussions/15396
    REPOID=Unsloth
    MODEL=gpt-oss-120b-GGUF
    QUANTIZATION=Q5_K_M
    CONTEXT=131072
    BATCH_SIZE=512 # Try values between 256 and 2048 (default)
    # TODO --ubatch-size value between 64 and 512 (default) note batch_size >= ubatch_size
    # TODO --cache-type-k try q4_0, q8_0 and f16 (default)
    # TODO --cache-type-v try q4_0, q8_0 and f16 (default)

    #CHAT_TEMPLATE_FILE="~/projects/llama/templates/gpt-oss-120b-opencode.mustache"
    CHAT_TEMPLATE_KWARGS='{"reasoning_effort": "high"}'
    #REASONING_FORMAT=none

    TEMP=1.0
    TOP_P=1.0
    TOP_K=0
    MIN_P=0

    #EXTRA_CMD_SWITCHES="--jinja --no-context-shift --flash-attn off --reasoning-format=llama3"
    # For large GPT-oss models we recommend disabling mmap if you have
    # sufficient RAM. When tensor overrides place some tensors on CPU,
    # using mmap can cause additional page-fault overhead. `--no-mmap`
    # forces weights to be loaded into memory which often improves throughput.
    EXTRA_CMD_SWITCHES="--gpt-oss-120b-default --no-mmap"
    ;;
  qwen3)
    REPOID=unsloth
    MODEL=unsloth/Qwen3.6-35B-A3B-GGUF
    QUANTIZATION=Q8_0
    CONTEXT=131072 # 128k context window (canonical size; max supported is 262144)
    BATCH_SIZE=4096
    UBATCH_SIZE=256
    CHAT_TEMPLATE_KWARGS=""
    REASONING_FORMAT=deepseek

    TEMP=0.6	
    TOP_P=0.95
    TOP_K=20
    MIN_P=0

    EXTRA_CMD_SWITCHES="--presence-penalty 0.0 --min-p 0.0 --flash-attn on --swa-full --no-mmproj --jinja"
    # recommended switched not included: -sm rows --no-context-shift -fa on -sm rows
    ;;
  mxbai-embed)
    REPOID=magicunicorn
    MODEL=mxbai-embed-large-v1-Q8_0-GGUF
    QUANTIZATION=Q8_0
    CONTEXT=2048
    BATCH_SIZE=512
    CHAT_TEMPLATE_KWARGS=""
    REASONING_FORMAT=deepseek

    TEMP=1.0
    TOP_P=0.95
    TOP_K=40
    MIN_P=0

    EXTRA_CMD_SWITCHES=""
    ;;
  gemma4)
    REPOID=ggml-org
    MODEL=gemma-4-31B-it-GGUF
    QUANTIZATION=Q8_0
    CONTEXT=262144
    BATCH_SIZE=512
    CHAT_TEMPLATE_KWARGS=""
    REASONING_FORMAT=none

    TEMP=1.0
    TOP_P=0.95
    TOP_K=64
    MIN_P=0

    EXTRA_CMD_SWITCHES="--jinja"
    ;;
  *)
    echo "Unrecognized model ('$model'). \nSupported models: GPT120, Qwen3, Qwen2.5, MXBAI-Embed, gemma4, Router"
    exit 1
    ;;
esac

##
# Override CONTEXT from models.ini if available (single source of truth)
##
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_INI_FILE="${LLAMA_MODELS_PRESET:-${SCRIPT_DIR}/models.ini}"
INI_CTX=$(get_ctx_size "$model" "$MODELS_INI_FILE" 2>/dev/null || true)
if [[ -n "$INI_CTX" ]]; then
  echo "Read ctx-size=$INI_CTX from $MODELS_INI_FILE for model '$model' (overriding CONTEXT=$CONTEXT)"
  CONTEXT="$INI_CTX"
else
  echo "No ctx-size found in $MODELS_INI_FILE for model '$model'; using CONTEXT=$CONTEXT"
fi

##
# Override QUANTIZATION from models.ini hf-repo suffix if available (single source of truth)
##
INI_HF_REPO=$(get_hf_repo "$model" "$MODELS_INI_FILE" 2>/dev/null || true)
if [[ -n "$INI_HF_REPO" ]]; then
  INI_QUANT=$(echo "$INI_HF_REPO" | awk -F: '{if (NF>1) print $NF}')
  if [[ -n "$INI_QUANT" ]]; then
    echo "Read quantization=$INI_QUANT from $MODELS_INI_FILE hf-repo suffix for model '$model' (overriding QUANTIZATION=$QUANTIZATION)"
    QUANTIZATION="$INI_QUANT"
  else
    echo "No quantization suffix in $MODELS_INI_FILE hf-repo for model '$model'; using QUANTIZATION=$QUANTIZATION"
  fi
else
  echo "No hf-repo found in $MODELS_INI_FILE for model '$model'; using QUANTIZATION=$QUANTIZATION"
fi

##
# Log the config
##
echo "Server Environment"
echo

echo "IP_ADDRESS=$IP_ADDRESS"
echo "PORT=$PORT"

echo "REPOID=$REPOID"
echo "MODEL=$MODEL"
echo "QUANTIZATION=$QUANTIZATION"
echo "CONTEXT=$CONTEXT"
echo "BATCH_SIZE=$BATCH_SIZE"
echo
echo "TEMP=$TEMP"
echo "TOP_P=$TOP_P"
echo "TOP_K=$TOP_K"
echo "MIN_P=$MIN_P"

echo
echo
echo

##
# Start the Server
##
##
## Determine which llama-server binary to use for single-model invocations
##
if [[ -n "${LLAMA_SERVER_BIN:-}" ]]; then
  LLAMA_BIN="${LLAMA_SERVER_BIN}"
else
  LLAMA_BIN="$(command -v llama-server 2>/dev/null || true)"
  if [[ -z "$LLAMA_BIN" ]]; then
    LLAMA_BIN="llama-server"
  fi
fi

echo "Using llama-server binary: $LLAMA_BIN"

  # Default parallelism. Can be overridden via LLAMA_PARALLEL env var (set
  # by the proxy lifecycle from server.session_slot_pool_size in config.yaml).
  # Keeping these values aligned is critical: mismatch causes slot exhaustion
  # or restore failures at runtime.
  : "${LLAMA_PARALLEL:=1}"
  LLAMA_CMD=(
    "$LLAMA_BIN"
    -hf "$REPOID/$MODEL:$QUANTIZATION"
    --ctx-size "$CONTEXT"
    --batch-size $BATCH_SIZE
    --ubatch-size ${UBATCH_SIZE:-$BATCH_SIZE}
    -np "$LLAMA_PARALLEL"
    -ngl "$GLOBAL_NGL"
    --no-mmap
    --parallel "$LLAMA_PARALLEL"
    $EXTRA_CMD_SWITCHES
    --embeddings
    --pooling mean
    --host 0.0.0.0
    --port $PORT
  )

if [[ -n "${LLAMA_SLOT_SAVE_PATH:-}" ]]; then
  mkdir -p "$LLAMA_SLOT_SAVE_PATH"
  LLAMA_CMD+=(--slot-save-path "$LLAMA_SLOT_SAVE_PATH")
fi

if [[ -n "${CHAT_TEMPLATE_KWARGS:-}" ]]; then
  LLAMA_CMD+=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")
fi

if [[ -n "${CHAT_TEMPLATE_FILE:-}" ]]; then
  LLAMA_CMD+=(--chat-template "$CHAT_TEMPLATE_FILE")
fi

"${LLAMA_CMD[@]}"
