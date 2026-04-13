import requests
import json
import time
from requests.exceptions import RequestException


def test_router_mode_serves_embeddings_and_chat():
    """Integration smoke test: router-mode serves embeddings + chat concurrently.

    Assumes a local test instance of the proxy is running on http://localhost:8000
    with llama-server in router mode on the configured backend port.
    """
    base = "http://localhost:8000"

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
    wait_for_embeddings(base, timeout=60)
    payload = {"model": "embeddings", "input": "hello world"}
    resp = requests.post(url, json=payload, timeout=10)
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

    raise AssertionError(f"embeddings endpoint not ready after {timeout}s: {emb_url}")
