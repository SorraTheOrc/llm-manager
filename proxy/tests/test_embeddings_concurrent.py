import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import RequestException


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
        raise AssertionError("embeddings endpoint not ready after 60s")

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

    # Validate responses: each should be (200, body)
    for status, body in results:
        assert status == 200, f"request failed or returned non-200: status={status} body={body}"
