"""Live end-to-end tests for the host-first startup flow.

These tests are intentionally skipped by default and only run on demand
against a live proxy and llama-server instance started by the operator.

Run manually:

    RUN_LIVE_HOST_FLOW=1 pytest tests/test_host_flow_live_e2e.py -v
"""

import os

import pytest
import requests

pytestmark = [pytest.mark.integration, pytest.mark.live]

if os.getenv("RUN_LIVE_HOST_FLOW", "0") not in ("1", "true", "yes"):
    pytest.skip(
        "live host-flow tests are disabled; set RUN_LIVE_HOST_FLOW=1 to run on demand",
        allow_module_level=True,
    )

LIVE_PROXY_URL = os.environ.get("LIVE_PROXY_BASE_URL", "http://localhost:8000")
LIVE_LLAMA_URL = os.environ.get("LIVE_LLAMA_BASE_URL", "http://localhost:8080")


def test_llama_server_health():
    """llama-server health endpoint is reachable."""
    resp = requests.get(f"{LIVE_LLAMA_URL}/health", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)


def test_proxy_health():
    """Proxy health endpoint is reachable."""
    resp = requests.get(f"{LIVE_PROXY_URL}/health", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)


def test_embedding_request():
    """A simple embedding request returns 200."""
    payload = {
        "model": "embeddings",
        "input": "Hello, world!"
    }
    resp = requests.post(
        f"{LIVE_PROXY_URL}/v1/embeddings",
        json=payload,
        timeout=30,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert len(data["data"]) > 0
    assert "embedding" in data["data"][0]


def test_chat_completion():
    """A simple chat completion request returns 200."""
    payload = {
        "model": "Qwen3",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 50,
    }
    resp = requests.post(
        f"{LIVE_PROXY_URL}/v1/chat/completions",
        json=payload,
        timeout=60,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["choices"]) > 0
    assert data["choices"][0]["message"]["content"] is not None
