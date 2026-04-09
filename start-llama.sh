#!/bin/bash
set -euo pipefail

model_arg="${1:-}"
model="$(printf '%s' "${model_arg}" | tr '[:upper:]' '[:lower:]')"
router_mode=0

if [[ -z "${model}" || "${model}" == "router" ]]; then
  router_mode=1
fi

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
  MODELS_MAX="${LLAMA_MODELS_MAX:-4}"

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

  LLAMA_CMD=(
    llama-server
    --models-preset "$MODELS_INI"
    --models-max "$MODELS_MAX"
    --models-autoload
    --host 0.0.0.0
    --port $PORT
  )

  if [[ -n "${LLAMA_MODELS_DIR:-}" ]]; then
    LLAMA_CMD+=(--models-dir "$LLAMA_MODELS_DIR")
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
    EXTRA_CMD_SWITCHES="--gpt-oss-120b-default"
    ;;
  qwen3)
    REPOID=Qwen
    MODEL=Qwen3-Coder-Next-GGUF
    QUANTIZATION=Q5_K_M
    CONTEXT=163840 # Max is 262144, max tried so far  131072
    BATCH_SIZE=512
    CHAT_TEMPLATE_KWARGS=""
    REASONING_FORMAT=deepseek

    TEMP=1.0
    TOP_P=0.95
    TOP_K=40
    MIN_P=0

    EXTRA_CMD_SWITCHES="--jinja"
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
  *)
    echo "Unrecognized model ('$model'). \nSupported models: GPT120, Qwen3, Qwen2.5, MXBAI-Embed, Router"
    exit 1
    ;;
esac

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
echo "TEAMP=$TEMP"
echo "TOP_P=$TOP_P"
echo "TOP_K=$TOP_K"
echo "MIN_P=$MIN_P"

echo
echo
echo

##
# Start the Server
##
LLAMA_CMD=(
  llama-server
  -hf "$REPOID/$MODEL:$QUANTIZATION"
  --ctx-size "$CONTEXT"
  --batch-size $BATCH_SIZE
  -np 1
  -ngl 99
  --no-mmap
  --parallel 1
  $EXTRA_CMD_SWITCHES
  --embeddings
  --pooling mean
  --host 0.0.0.0
  --port $PORT
)

if [[ -n "${CHAT_TEMPLATE_KWARGS:-}" ]]; then
  LLAMA_CMD+=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")
fi

if [[ -n "${CHAT_TEMPLATE_FILE:-}" ]]; then
  LLAMA_CMD+=(--chat-template "$CHAT_TEMPLATE_FILE")
fi

"${LLAMA_CMD[@]}"
