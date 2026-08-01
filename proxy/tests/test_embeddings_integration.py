import os
import time

import pytest
import requests
from requests.exceptions import RequestException

# These tests exercise the LIVE proxy (localhost:8000) with real chat and
# embeddings inference. They are opt-in and disabled by default so a routine
# `pytest` run can never crash the running proxy/llama-server (GPU contention)
# or leave it in a bad state (see LP-0MS6R13CP009VO24).
pytestmark = [pytest.mark.integration, pytest.mark.e2e_live]

if os.getenv("RUN_LIVE_PROXY_E2E", "0") != "1":
    pytest.skip(
        "live proxy E2E tests are disabled; set RUN_LIVE_PROXY_E2E=1 to run on demand",
        allow_module_level=True,
    )


def _require_local_proxy(base: str):
    """Skip integration tests when a local proxy instance is not running."""
    try:
        # Generous health timeout: a healthy-but-loaded proxy (e.g. concurrent
        # chat streams contending for the single GPU) may answer /health slowly,
        # and we must not spuriously skip (see LP-0MS9FM27K007NCNE).
        r = requests.get(f"{base}/health", timeout=5)
        if r.status_code != 200:
            pytest.skip(f"local proxy not healthy at {base}/health")
    except RequestException:
        pytest.skip(f"local proxy not reachable at {base}")


def test_router_mode_serves_embeddings_and_chat():
    """Integration smoke test: router-mode serves embeddings + chat concurrently.

    Assumes a local test instance of the proxy is running on http://localhost:8000
    with llama-server in router mode on the configured backend port.
    """
    base = "http://localhost:8000"

    _require_local_proxy(base)

    embeddings_payload = {"model": "embeddings", "input": "hello world"}
    chat_payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5
    }

    # Wait for the proxy/router to be ready to serve the embeddings alias to avoid
    # race conditions with router preload. Poll until a successful embeddings
    # response is returned or we hit the timeout.
    wait_for_embeddings(base, timeout=60)
    wait_for_chat(base, chat_payload, timeout=120)

    embeddings_resp = requests.post(f"{base}/v1/embeddings", json=embeddings_payload, timeout=30)
    chat_resp = requests.post(f"{base}/v1/chat/completions", json=chat_payload, timeout=60)

    assert embeddings_resp.status_code == 200, f"unexpected embeddings status: {embeddings_resp.status_code} {embeddings_resp.text}"
    assert chat_resp.status_code == 200, f"unexpected chat status: {chat_resp.status_code} {chat_resp.text}"

    embeddings_body = embeddings_resp.json()
    assert "data" in embeddings_body and isinstance(embeddings_body["data"], list)
    assert len(embeddings_body["data"]) >= 1
    vec = embeddings_body["data"][0].get("embedding")
    assert vec and isinstance(vec, list)

    chat_body = chat_resp.json()
    assert "choices" in chat_body and isinstance(chat_body["choices"], list)
    assert len(chat_body["choices"]) >= 1
    assert "message" in chat_body["choices"][0]

def test_embeddings_alias_returns_openai_format():
    """Integration test: POST /v1/embeddings with model 'embeddings' returns OpenAI embeddings format.

    Note: This test assumes a local test instance of the proxy is running on http://localhost:8000
    and a local llama-server serving the example model is reachable at the configured backend port.
    """
    url = "http://localhost:8000/v1/embeddings"
    base = "http://localhost:8000"
    _require_local_proxy(base)
    wait_for_embeddings(base, timeout=60)
    payload = {"model": "embeddings", "input": "hello world"}
    # Generous request timeout: embeddings can be slow on a loaded GPU even
    # after wait_for_embeddings() warm-up (see LP-0MS9FM27K007NCNE).
    resp = requests.post(url, json=payload, timeout=30)
    assert resp.status_code == 200, f"unexpected status: {resp.status_code} {resp.text}"
    body = resp.json()
    # Basic OpenAI embeddings response sanity checks
    assert "data" in body and isinstance(body["data"], list)
    assert len(body["data"]) >= 1
    vec = body["data"][0].get("embedding")
    assert vec and isinstance(vec, list)
    # Check vector not all zeros
    assert any(x != 0 for x in vec)


def wait_for_embeddings(base, timeout=30, interval=1.0):
    """Poll the proxy until the embeddings alias is ready.

    Tries GET /health to ensure the proxy is up, then repeatedly POSTs a
    small embeddings request until a 200 response is returned or timeout is
    reached.
    """
    deadline = time.time() + timeout
    health_url = f"{base}/health"
    emb_url = f"{base}/v1/embeddings"
    payload = {"model": "embeddings", "input": "ready?"}

    while time.time() < deadline:
        try:
            # quick health ping (non-blocking)
            h = requests.get(health_url, timeout=2)
            if h.status_code != 200:
                time.sleep(interval)
                continue
        except RequestException:
            time.sleep(interval)
            continue

        try:
            r = requests.post(emb_url, json=payload, timeout=5)
            if r.status_code == 200:
                return
        except RequestException:
            pass

        time.sleep(interval)

    pytest.skip(f"embeddings endpoint not ready after {timeout}s: {emb_url}")


def wait_for_chat(base, payload, timeout=60, interval=1.0):
    """Poll the proxy until the chat endpoint is ready for the model."""
    deadline = time.time() + timeout
    chat_url = f"{base}/v1/chat/completions"

    while time.time() < deadline:
        try:
            r = requests.post(chat_url, json=payload, timeout=15)
            if r.status_code == 200:
                return
        except RequestException:
            pass

        time.sleep(interval)

    pytest.skip(f"chat endpoint not ready after {timeout}s: {chat_url}")
