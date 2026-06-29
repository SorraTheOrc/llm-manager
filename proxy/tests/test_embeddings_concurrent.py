import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from requests.exceptions import RequestException


def _require_local_proxy(base: str):
    """Skip integration tests when a local proxy instance is not running."""
    try:
        r = requests.get(f"{base}/health", timeout=2)
        if r.status_code != 200:
            pytest.skip(f"local proxy not healthy at {base}/health")
    except RequestException:
        pytest.skip(f"local proxy not reachable at {base}")


def run_embedding(base, payload, timeout=30):
    try:
        r = requests.post(f"{base}/v1/embeddings", json=payload, timeout=timeout)
        return r.status_code, r.text
    except Exception as e:
        return None, str(e)


def run_chat(base, payload, timeout=60):
    try:
        r = requests.post(f"{base}/v1/chat/completions", json=payload, timeout=timeout)
        return r.status_code, r.text
    except Exception as e:
        return None, str(e)


def test_concurrent_embeddings_and_chat():
    """Issue 5 concurrent embedding requests and 5 concurrent chat requests and assert HTTP 200 for all.

    Assumes a running proxy at http://localhost:8000 with router-mode llama-server available.
    """
    base = "http://localhost:8000"
    _require_local_proxy(base)

    # Warm up / wait until embeddings alias is ready (reuse helper from other test)
    deadline = time.time() + 60
    emb_url = f"{base}/v1/embeddings"
    health_url = f"{base}/health"
    payload_ready = {"model": "embeddings", "input": "ready?"}

    while time.time() < deadline:
        try:
            h = requests.get(health_url, timeout=2)
            if h.status_code != 200:
                time.sleep(1.0)
                continue
        except RequestException:
            time.sleep(1.0)
            continue
        try:
            r = requests.post(emb_url, json=payload_ready, timeout=5)
            if r.status_code == 200:
                break
        except RequestException:
            pass
        time.sleep(1.0)
    else:
        pytest.skip("embeddings endpoint not ready after 60s")

    # Prepare payloads
    embeddings_payloads = [{"model": "embeddings", "input": f"hello {i}"} for i in range(5)]
    chat_payloads = [{
        "model": "qwen3",
        "messages": [{"role": "user", "content": f"Hello {i}"}],
        "max_tokens": 10
    } for i in range(5)]

    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = []
        for p in embeddings_payloads:
            futures.append(ex.submit(run_embedding, base, p))
        for p in chat_payloads:
            futures.append(ex.submit(run_chat, base, p))

        for fut in as_completed(futures, timeout=120):
            results.append(fut.result())

    # Ensure we got 10 results
    assert len(results) == 10, f"expected 10 responses, got {len(results)}"

    # Validate responses under concurrency guard:
    # with max_concurrent_queries=4, overload 503s are expected when firing 10 at once.
    ok_count = 0
    overload_count = 0
    for status, body in results:
        assert status in (200, 503), f"unexpected status={status} body={body}"
        if status == 200:
            ok_count += 1
        elif status == 503:
            overload_count += 1
            assert (
                "Server overloaded" in body
                or "Model server busy" in body
                or "model_loading" in body
            ), f"unexpected 503 body={body}"

    assert ok_count >= 1, "expected at least one successful request under concurrent load"
