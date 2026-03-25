import requests
import json

def test_embeddings_alias_returns_openai_format():
    """Integration test: POST /v1/embeddings with model 'embeddings' returns OpenAI embeddings format.

    Note: This test assumes a local test instance of the proxy is running on http://localhost:3000
    and a local llama-server serving the example model is reachable at the configured backend port.
    """
    url = "http://localhost:3000/v1/embeddings"
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
