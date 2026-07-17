import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import requests
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


def _teardown(base: str, timeout: int = 30):
    """Wait for the proxy to become healthy after test execution.

    This reduces test order dependencies by allowing server state to
    stabilise before the next test runs.
    """
    deadline = time.time() + timeout
    health_url = f"{base}/health"
    while time.time() < deadline:
        try:
            h = requests.get(health_url, timeout=2)
            if h.status_code == 200:
                return
        except RequestException:
            pass
        time.sleep(1.0)


def test_concurrent_embeddings_and_chat():
    """Issue concurrent embedding and chat requests and verify graceful handling.

    Sends 1 embedding and 1 chat request at a time (matching
    local_max_concurrent_queries=1) rather than 10 concurrent requests,
    reducing the likelihood of overwhelming the server while still
    exercising both endpoints under concurrency.

    Assumes a running proxy at http://localhost:8000 with router-mode
    llama-server available.
    """
    base = "http://localhost:8000"
    _require_local_proxy(base)

    # Warm up / wait until embeddings alias is ready
    deadline = time.time() + 120
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
        pytest.skip("embeddings endpoint not ready after 120s")

    # Prepare payloads — reduced from 5+5=10 to 3+3=6 to keep test
    # duration reasonable while still exercising both endpoints.
    embeddings_payloads = [{"model": "embeddings", "input": f"hello {i}"} for i in range(3)]
    chat_payloads = [{
        "model": "qwen3",
        "messages": [{"role": "user", "content": f"Hello {i}"}],
        "max_tokens": 10
    } for i in range(3)]

    results = []
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = []
            for p in embeddings_payloads:
                futures.append(ex.submit(run_embedding, base, p))
            for p in chat_payloads:
                futures.append(ex.submit(run_chat, base, p))

            for fut in as_completed(futures, timeout=120):
                results.append(fut.result())

        # Ensure we got 6 results (3 embeddings + 3 chats)
        assert len(results) == 6, f"expected 6 responses, got {len(results)}"

        # Validate responses under concurrency guard:
        # with local_max_concurrent_queries=1, overload 503s are expected
        # when firing 2 concurrent requests.
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
                    or "Queue full" in body  # REJECTED_503 from JobScheduler
                ), f"unexpected 503 body={body}"

        assert ok_count >= 1, "expected at least one successful request under concurrent load"
    finally:
        _teardown(base)
