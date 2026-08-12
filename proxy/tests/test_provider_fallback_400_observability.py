"""
Test-first: 400-body observability in the remote fallback loop.

Red tests for LP-0MSC4UJXW003NW7C (parent LP-0MSC1BNP90017L9K).

Asserts (currently failing — F3 implements):
1. A remote HTTP 400 fallback emits a per-fallback log line containing the
   response body snippet (not only when all providers are exhausted).
2. ``proxy_http_errors_total`` is incremented with status=400 and the fallback
   reason for remote 400 rejections.

Test harness mirrors test_provider_fallback.py: patch
``proxy.server.proxy_to_remote`` with a synthetic httpx.Response.
"""

import logging
from unittest.mock import patch

import httpx
import proxy.metrics as metrics
import proxy.provider as provider
import pytest


class _DummyRequest:
    """Minimal request stub for use in fallback tests."""

    def __init__(self, body: bytes = b'{"model":"test"}'):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


@pytest.fixture(autouse=True)
def reset_cooldown_state():
    """Reset cooldown and failure-count state between tests to avoid cross-test leakage."""
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    yield


@pytest.fixture
def sample_model_config():
    """A model config with an ordered providers list (remote only)."""
    return {
        "providers": [
            {
                "name": "remote-primary",
                "type": "remote",
                "endpoint": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.anthropic.com/v1",
                "api_key_env": "ANTHROPIC_API_KEY",
            },
        ],
        "aliases": ["mimo*"],
    }


def _metric_value(endpoint, status, reason):
    """Return the current counter value for the given labels (0 if never incremented)."""
    try:
        return metrics.proxy_http_errors_total.labels(
            endpoint=endpoint, status=status, reason=reason
        )._value.get()
    except Exception:
        return 0


@pytest.mark.asyncio
async def test_remote_400_fallback_logs_body_snippet_per_fallback(sample_model_config, caplog):
    """A remote 400 fallback must log the response body snippet per-fallback.

    The first provider returns 400 with a body; the second succeeds. The log
    must contain the body snippet at INFO (or DEBUG) level.
    """
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        if provider_cfg["name"] == "remote-primary":
            return httpx.Response(
                status_code=400,
                json={"error": {"message": "missing field `tool_call_id`", "type": "invalid_request_error"}},
            )
        return httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    with caplog.at_level(logging.INFO, logger="llama-proxy.provider"):
        with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
            result = await provider.proxy_with_remote_fallback(
                request, "v1/chat/completions", sample_model_config, cfg
            )

    assert result.status_code == 200
    # A per-fallback log line mentioning the 400 body snippet must exist.
    log_text = caplog.text
    assert "tool_call_id" in log_text, (
        "Expected the 400 response body snippet to appear in fallback logs, "
        f"got:\n{log_text[:2000]}"
    )


@pytest.mark.asyncio
async def test_remote_400_fallback_increments_http_errors_metric(sample_model_config):
    """A remote 400 fallback must increment proxy_http_errors_total{status=400}."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        if provider_cfg["name"] == "remote-primary":
            return httpx.Response(
                status_code=400,
                json={"error": {"message": "bad request"}},
            )
        return httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    before = _metric_value("v1/chat/completions", "400", "HTTP 400")

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert result.status_code == 200
    after = _metric_value("v1/chat/completions", "400", "HTTP 400")
    assert after == before + 1, (
        "Expected proxy_http_errors_total{status=400, reason='HTTP 400'} to be "
        f"incremented by 1 (before={before}, after={after})"
    )


@pytest.mark.asyncio
async def test_remote_500_fallback_does_not_increment_400_metric(sample_model_config):
    """A 500 fallback must not increment the status=400 counter (no cross-talk)."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        if provider_cfg["name"] == "remote-primary":
            return httpx.Response(status_code=502, json={"error": {"message": "bad gateway"}})
        return httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    before = _metric_value("v1/chat/completions", "400", "HTTP 400")

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert result.status_code == 200
    after = _metric_value("v1/chat/completions", "400", "HTTP 400")
    assert after == before, "status=400 counter must not change on a 502 fallback"
