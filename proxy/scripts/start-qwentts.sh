#!/usr/bin/env bash
# Start qwentts.cpp TTS server for the LLM Proxy.
# Usage: ./scripts/start-qwentts.sh [options]
#
# Options:
#   --port <port>         Listen port (default: 8081)
#   --model <path>        Talker LM GGUF path
#   --codec <path>        Codec GGUF path
#   --help                Show this help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TT_SERVER="${REPO_ROOT}/qwentts.cpp/tts-server"

# ---- Defaults -----------------------------------------------------------
PORT="${QWTTS_PORT:-8081}"
MODEL="${QWTTS_MODEL:-}"
CODEC="${QWTTS_CODEC:-}"

# ---- Parse arguments ----------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"; shift 2 ;;
    --port=*)
      PORT="${1#*=}"; shift ;;
    --model)
      MODEL="$2"; shift 2 ;;
    --model=*)
      MODEL="${1#*=}"; shift ;;
    --codec)
      CODEC="$2"; shift 2 ;;
    --codec=*)
      CODEC="${1#*=}"; shift ;;
    --help|-h)
      head -15 "$0"
      exit 0 ;;
    *)
      echo "Error: Unknown argument: $1" >&2
      exit 1 ;;
  esac
done

# Resolve model / codec paths if not explicitly set
if [ -z "$MODEL" ]; then
  # Try build output directory first (local dev), then installed models
  for candidate in \
    "${REPO_ROOT}/qwentts.cpp/models/qwen-talker-1.7b-base-Q8_0.gguf" \
    "${REPO_ROOT}/qwentts.cpp/models/qwen-talker-0.6b-base-Q8_0.gguf"; do
    if [ -f "$candidate" ]; then
      MODEL="$candidate"
      break
    fi
  done
fi
if [ -z "$CODEC" ]; then
  for candidate in \
    "${REPO_ROOT}/qwentts.cpp/models/qwen-tokenizer-12hz-Q8_0.gguf" \
    "${REPO_ROOT}/qwentts.cpp/models/qwen-tokenizer-12hz-F32.gguf"; do
    if [ -f "$candidate" ]; then
      CODEC="$candidate"
      break
    fi
  done
fi

# ---- Validation ---------------------------------------------------------
if [ ! -f "$TT_SERVER" ]; then
  echo "Error: tts-server binary not found at: $TT_SERVER" >&2
  echo "Build it: cd qwentts.cpp && mkdir build && cd build && cmake .. -DGGML_HIP=ON && make -j" >&2
  exit 1
fi
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
  echo "Error: TTS model GGUF not found." >&2
  echo "Set --model <path> or install one of:" >&2
  echo "  qwentts.cpp/models/qwen-talker-1.7b-base-Q8_0.gguf" >&2
  echo "  qwentts.cpp/models/qwen-talker-0.6b-base-Q8_0.gguf" >&2
  exit 1
fi
if [ -z "$CODEC" ] || [ ! -f "$CODEC" ]; then
  echo "Error: TTS tokenizer GGUF not found." >&2
  echo "Set --codec <path> or install:" >&2
  echo "  qwentts.cpp/models/qwen-tokenizer-12hz-Q8_0.gguf" >&2
  exit 1
fi

# Check port availability
if command -v ss >/dev/null 2>&1; then
  if ss -ltn | awk '{print $4}' | grep -Eq ":${PORT}$|\.${PORT}$"; then
    echo "Error: port $PORT is already in use." >&2
    exit 1
  fi
elif command -v netstat >/dev/null 2>&1; then
  if netstat -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${PORT}$|\.${PORT}$"; then
    echo "Error: port $PORT is already in use." >&2
    exit 1
  fi
fi

echo "=== Starting qwentts TTS server ==="
echo "Port:  $PORT"
echo "Model: $MODEL"
echo "Codec: $CODEC"

exec "$TT_SERVER" \
  --model "$MODEL" \
  --codec "$CODEC" \
  --port "$PORT"
