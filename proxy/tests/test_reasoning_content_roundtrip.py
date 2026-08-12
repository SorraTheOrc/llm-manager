"""
Test-first: reasoning_content round-trip 400 must not reach the client raw.

Red tests for LP-0MSGU3JNU0092AFQ (AC3).

Remote thinking-mode providers (Console opencode.ai/zen, Console Go
opencode.ai/zen/go, api.deepseek.com) reject multi-turn requests with HTTP 400
when any assistant message lacks the ``reasoning_content`` field. The sanitizer
repairs the payload before send (AC1/AC2); as a belt-and-braces measure (AC3),
when this specific 400 still occurs and all fallback providers are exhausted,
the proxy must NOT return the raw upstream body to the client — it returns a
synthetic error carrying remediation guidance instead. Other 400 shapes (e.g.
tool-call validation) keep the existing "return first provider error response"
behavior.

Harness mirrors tests/test_provider_fallback_400_observability.py: patch
``proxy.server.proxy_to_remote`` with a synthetic httpx.Response.
"""

import json
from unittest.mock import patch

import httpx
import proxy.provider as provider
import pytest

REASONING_400_BODY_CONSOLE = {
    "error": {
        "param": None,
        "type": "invalid_request_error",
        "code": "invalid_request_error",
        "message": (
            "Error from provider (Console Go): Upstream request failed: "
            "[invalid_request_error] The `reasoning_content` in the thinking "
            "mode must be passed back to the API."
        ),
    }
}

REASONING_400_BODY_DEEPSEEK = {
    "error": {
        "message": "The `reasoning_content` in the thinking mode must be passed back to the API.",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_request_error",
    }
}

TOOLCALL_400_BODY = {
    "error": {
        "message": "messages[1]: missing field 'tool_call_id'",
        "type": "invalid_request_error",
    }
}


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


def _response_body_text(response) -> str:
    """Best-effort text extraction from the returned (starlette/httpx) response."""
    raw = getattr(response, "content", None) or getattr(response, "body", b"")
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


@pytest.mark.asyncio
async def test_reasoning_400_all_providers_exhausted_returns_synthetic_error(sample_model_config):
    """The specific reasoning_content 400 must not reach the client raw.

    Exercises ``proxy_with_fallback`` (the main chat path for local+remote
    chains): when all providers return the reasoning_content 400, the client
    must receive the synthetic remediation error, not the raw upstream body.
    """
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        return httpx.Response(status_code=400, json=REASONING_400_BODY_CONSOLE)

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    body = _response_body_text(result)
    assert "Error from provider (Console Go)" not in body, (
        "raw upstream reasoning_content 400 body must not reach the client"
    )
    assert "reasoning_content" in body, (
        "synthetic error must identify the reasoning_content round-trip issue"
    )
    assert "suggested_action" in body or "retry" in body.lower(), (
        "synthetic error must carry remediation guidance"
    )


@pytest.mark.asyncio
async def test_reasoning_400_deepseek_shape_detected(sample_model_config):
    """The direct-deepseek variant of the error must also be intercepted."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        return httpx.Response(status_code=400, json=REASONING_400_BODY_DEEPSEEK)

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    body = _response_body_text(result)
    assert "must be passed back" not in body or "suggested_action" in body, (
        "deepseek-shaped reasoning 400 must be replaced by the synthetic error"
    )


@pytest.mark.asyncio
async def test_other_400_still_returns_first_provider_error(sample_model_config):
    """Non-reasoning 400s keep the existing 'first provider error' behavior."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        return httpx.Response(status_code=400, json=TOOLCALL_400_BODY)

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert result.status_code == 400
    body = _response_body_text(result)
    assert "tool_call_id" in body, (
        "non-reasoning 400 must keep the raw first-provider error response"
    )


@pytest.mark.asyncio
async def test_remote_only_chain_intercepts_reasoning_400(sample_model_config):
    """Remote-only models (proxy_with_remote_fallback) also get the synthetic error.

    AC3 covers both dispatch paths: remote-only configs go through
    ``proxy_with_remote_fallback`` which otherwise returns an exhausted 503
    whose diagnostics leak the raw upstream body.
    """
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        return httpx.Response(status_code=400, json=REASONING_400_BODY_CONSOLE)

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    body = _response_body_text(result)
    assert "Error from provider (Console Go)" not in body, (
        "raw upstream reasoning_content 400 body must not reach the client"
    )
    assert "suggested_action" in body, (
        "remote-only chain must return the synthetic remediation error"
    )


@pytest.mark.asyncio
async def test_remote_only_chain_keeps_other_400_exhausted_503(sample_model_config):
    """Non-reasoning 400s on the remote-only chain keep the existing behavior."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        return httpx.Response(status_code=400, json=TOOLCALL_400_BODY)

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert result.status_code == 503, (
        "non-reasoning 400 on the remote-only chain keeps the exhausted 503 "
        "(no first-error-response logic in proxy_with_remote_fallback)"
    )


@pytest.mark.asyncio
async def test_reasoning_400_detector_matches_both_variants():
    """Unit-level check: the detector recognises Console and deepseek shapes."""
    console_resp = httpx.Response(status_code=400, json=REASONING_400_BODY_CONSOLE)
    deepseek_resp = httpx.Response(status_code=400, json=REASONING_400_BODY_DEEPSEEK)
    other_resp = httpx.Response(status_code=400, json=TOOLCALL_400_BODY)
    ok_resp = httpx.Response(status_code=200, json={"choices": [{"message": {"content": "ok"}}]})

    assert provider._is_reasoning_content_roundtrip_error(console_resp) is True
    assert provider._is_reasoning_content_roundtrip_error(deepseek_resp) is True
    assert provider._is_reasoning_content_roundtrip_error(other_resp) is False
    assert provider._is_reasoning_content_roundtrip_error(ok_resp) is False
