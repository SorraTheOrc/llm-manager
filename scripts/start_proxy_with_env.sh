#!/usr/bin/env bash
set -euo pipefail

# Start uvicorn for proxy.server with the PYTHONPATH set so the inner
# package 'proxy' (located at /home/rgardler/projects/llm/proxy/proxy)
# is importable as 'proxy.server'.
export PYTHONPATH="/home/rgardler/projects/llm/proxy"
exec /home/rgardler/projects/llm/.venv/bin/python3 -m uvicorn proxy.server:app --host 0.0.0.0 --port 8000
