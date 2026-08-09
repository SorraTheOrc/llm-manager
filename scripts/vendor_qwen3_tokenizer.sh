#!/usr/bin/env bash
# vendor_qwen3_tokenizer.sh — Download and vendor the Qwen3 tokenizer.json.
#
# Downloads the Qwen3 tokenizer.json from the official Qwen HF repo and
# writes it to proxy/proxy/tokenizer_data/qwen3/tokenizer.json with sha256
# checksum verification. The vendored file is committed to the repo so the
# proxy needs NO runtime network access to count native tokens
# (LP-0MSEQ71IF0003FRT).
#
# Usage:
#   scripts/vendor_qwen3_tokenizer.sh [--force]
#
#   --force   Re-download even if the vendored file already exists
#             (otherwise the script skips a checksum-verified file).
#
# Configuration (environment variables):
#   HF_TOKENIZER_URL   Override the download URL (default: Qwen3.6-35B-A3B
#                      tokenizer.json on huggingface.co).
#   EXPECTED_SHA256    Override the expected sha256 checksum.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve the current checkout root (worktree-aware): --show-toplevel
# returns the worktree root when run inside a worktree, the main checkout
# root otherwise. Falls back to the script's parent dir.
if [ -z "${PROJECT_ROOT:-}" ]; then
    TOPLEVEL="$(git rev-parse --show-toplevel 2>/dev/null)"
    if [ -n "$TOPLEVEL" ]; then
        PROJECT_ROOT="$TOPLEVEL"
    else
        PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    fi
fi

TARGET_DIR="$PROJECT_ROOT/proxy/proxy/tokenizer_data/qwen3"
TARGET_FILE="$TARGET_DIR/tokenizer.json"
HF_TOKENIZER_URL="${HF_TOKENIZER_URL:-https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main/tokenizer.json}"
EXPECTED_SHA256="${EXPECTED_SHA256:-5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42}"

if [ "${1:-}" = "--force" ]; then
    FORCE=1
else
    FORCE=0
fi

if [ -f "$TARGET_FILE" ] && [ "$FORCE" -ne 1 ]; then
    ACTUAL="$(sha256sum "$TARGET_FILE" | awk '{print $1}')"
    if [ "$ACTUAL" = "$EXPECTED_SHA256" ]; then
        echo "Vendored tokenizer already present and checksum-verified: $TARGET_FILE"
        exit 0
    fi
    echo "Existing vendored tokenizer has unexpected checksum ($ACTUAL != $EXPECTED_SHA256); re-downloading" >&2
fi

echo "Downloading Qwen3 tokenizer.json from $HF_TOKENIZER_URL ..."
TMP_FILE="$(mktemp)"
trap 'rm -f "$TMP_FILE"' EXIT

if command -v curl >/dev/null 2>&1; then
    curl -fsSL --max-time 300 -o "$TMP_FILE" "$HF_TOKENIZER_URL"
else
    wget -q -O "$TMP_FILE" "$HF_TOKENIZER_URL"
fi

ACTUAL="$(sha256sum "$TMP_FILE" | awk '{print $1}')"
if [ "$ACTUAL" != "$EXPECTED_SHA256" ]; then
    echo "ERROR: checksum mismatch — expected $EXPECTED_SHA256, got $ACTUAL" >&2
    echo "If the upstream tokenizer legitimately changed, update EXPECTED_SHA256." >&2
    exit 1
fi

mkdir -p "$TARGET_DIR"
mv "$TMP_FILE" "$TARGET_FILE"
trap - EXIT

echo "Vendored Qwen3 tokenizer written to $TARGET_FILE (sha256 $ACTUAL)"
