#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="llama-proxy/llama-server:local"

HERE="$(dirname "$0")"

echo "Building container image: $IMAGE_TAG"
podman build -t "$IMAGE_TAG" -f "$HERE/Containerfile" "$HERE/.."

echo "Built $IMAGE_TAG"
