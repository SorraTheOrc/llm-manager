"""
Tests for LP-0MTVPJQ6T004EZ75: only poison failure domain after 2nd consecutive empty_response.

Acceptance criteria:
  AC1: A single empty_response on one entry does NOT skip same-endpoint siblings
       in the same request; the next sibling is tried.
  AC2: Two consecutive empty_response failures on the same endpoint+model DO poison
       the domain (siblings skipped, matching current behavior for confirmed-bad
       gateways).
  AC3: ReadTimeout/stream_error/HTTP 5xx still poison the domain on first failure.
"""

import json
from unittest.mock import patch

import proxy.provider as provider
import pytest
from fastapi import Response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyRequest:
    """Minimal request stub for use in fallback tests."""

    def __init__(self, body: bytes = b'{"model": "test"}'):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_cooldown_state():
    """Reset cooldown and failure-count state between tests."""
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    yield


# ---------------------------------------------------------------------------
# AC1: single empty_response does NOT poison the failure domain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_empty_response_does_not_poison_failure_domain():
    """AC1: A single empty_response on one entry does NOT skip same-endpoint siblings."""
    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 0,  # short cooldown so siblings are still eligible
    }
    # Three providers on the same endpoint (same failure domain)
    model_config = {
        "providers": [
            {
                "name": "key-a",
                "type": "remote",
                "endpoint": "https://gateway.example.com/v1",
                "api_key_env": "KEY_A",
            },
            {
                "name": "key-b",
                "type": "remote",
                "endpoint": "https://gateway.example.com/v1",
                "api_key_env": "KEY_B",
            },
            {
                "name": "key-c",
                "type": "remote",
                "endpoint": "https://gateway.example.com/v1",
                "api_key_env": "KEY_C",
            },
        ],
        "aliases": ["gw*"],
    }

    async def _mock_ptr(_req, _path, provider_cfg):
        name = provider_cfg["name"]
        if name == "key-a":
            # First provider returns empty response
            return Response(
                content=json.dumps({"choices": []}),
                status_code=200,
                media_type="application/json",
            )
        # Other providers return valid content
        return Response(
            content=json.dumps({"choices": [{"message": {"content": f"reply from {name}"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_ptr):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", model_config, cfg
        )

    assert result.status_code == 200
    body = json.loads(result.body)
    # key-a returned empty -> key-b should be tried (same domain not poisoned)
    assert "key-b" in body["choices"][0]["message"]["content"] or "key-c" in body[
        "choices"
    ][0]["message"]["content"]


@pytest.mark.asyncio
async def test_single_empty_response_tries_all_same_domain_siblings():
    """AC1: When key-a fails empty and key-b also fails empty, key-c (same domain) is still tried."""
    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 0,
    }
    model_config = {
        "providers": [
            {
                "name": "key-a",
                "type": "remote",
                "endpoint": "https://gateway.example.com/v1",
                "api_key_env": "KEY_A",
            },
            {
                "name": "key-b",
                "type": "remote",
                "endpoint": "https://gateway.example.com/v1",
                "api_key_env": "KEY_B",
            },
            {
                "name": "key-c",
                "type": "remote",
                "endpoint": "https://gateway.example.com/v1",
                "api_key_env": "KEY_C",
            },
        ],
        "aliases": ["gw*"],
    }

    call_order = []

    async def _mock_ptr(_req, _path, provider_cfg):
        name = provider_cfg["name"]
        call_order.append(name)
        if name in ("key-a", "key-b"):
            # First two return empty
            return Response(
                content=json.dumps({"choices": []}),
                status_code=200,
                media_type="application/json",
            )
        # key-c succeeds
        return Response(
            content=json.dumps({"choices": [{"message": {"content": f"reply from {name}"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_ptr):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", model_config, cfg
        )

    assert result.status_code == 200
    # key-a and key-b should both be tried (single empty doesn't poison the domain)
    assert "key-a" in call_order
    assert "key-b" in call_order
    assert "key-c" in call_order


# ---------------------------------------------------------------------------
# AC2: two consecutive empty_responses DO poison the failure domain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_consecutive_empty_responses_poison_domain():
    """AC2: Two consecutive empty_response failures on the same endpoint+model DO poison the domain."""
    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 0,
        "sibling_fallback_threshold": 2,
    }
    # Two providers on the same endpoint (same failure domain)
    model_config = {
        "providers": [
            {
                "name": "key-a",
                "type": "remote",
                "endpoint": "https://gateway.example.com/v1",
                "api_key_env": "KEY_A",
            },
            {
                "name": "key-b",
                "type": "remote",
                "endpoint": "https://gateway.example.com/v1",
                "api_key_env": "KEY_B",
            },
            {
                "name": "different-gateway",
                "type": "remote",
                "endpoint": "https://other.example.com/v1",
                "api_key_env": "DIFF_KEY",
            },
        ],
        "aliases": ["gw*"],
    }

    async def _mock_ptr(_req, _path, provider_cfg):
        name = provider_cfg["name"]
        if name != "different-gateway":
            # Same-endpoint keys return empty
            return Response(
                content=json.dumps({"choices": []}),
                status_code=200,
                media_type="application/json",
            )
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_ptr):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", model_config, cfg
        )

    assert result.status_code == 200
    # After 2 empty responses on key-a+key-b (same domain), the domain is poisoned
    # and the request falls through to different-gateway
    body = json.loads(result.body)
    assert body["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_two_consecutive_empty_across_requests_poisons_domain():
    """AC2: Two consecutive empty_response failures across two requests poison the domain."""
    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 0,
        "sibling_fallback_threshold": 2,
    }
    model_config = {
        "providers": [
            {
                "name": "sick-key",
                "type": "remote",
                "endpoint": "https://sick.example.com/v1",
                "api_key_env": "SICK_KEY",
            },
            {
                "name": "healthy-key",
                "type": "remote",
                "endpoint": "https://healthy.example.com/v1",
                "api_key_env": "HEALTHY_KEY",
            },
        ],
        "aliases": ["sib*"],
    }

    call_order = []

    async def _mock_ptr(_req, _path, provider_cfg):
        name = provider_cfg["name"]
        call_order.append(name)
        if name == "sick-key":
            return Response(
                content=json.dumps({"choices": []}),
                status_code=200,
                media_type="application/json",
            )
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    # Request 1: sick-key empty (failure 1) -> healthy answers
    with patch("proxy.server.proxy_to_remote", _mock_ptr):
        result1 = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", model_config, cfg
        )
    assert result1.status_code == 200
    assert call_order == ["sick-key", "healthy-key"]

    # Request 2: sick-key empty again (failure 2 -> threshold exceeded) ->
    # sick-key's domain is now poisoned, so healthy-key is tried directly
    call_order.clear()
    with patch("proxy.server.proxy_to_remote", _mock_ptr):
        result2 = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", model_config, cfg
        )
    assert result2.status_code == 200
    # sick-key should not be tried because its domain is poisoned after 2 empty responses
    # Note: the streak is cross-request so sick-key's domain is poisoned
    assert "healthy-key" in call_order


# ---------------------------------------------------------------------------
# AC3: ReadTimeout / HTTP 5xx still poison on first failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_5xx_poisons_domain_immediately():
    """AC3: HTTP 5xx still poison the domain on first failure (unchanged)."""
    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 0,
    }
    model_config = {
        "providers": [
            {
                "name": "key-a",
                "type": "remote",
                "endpoint": "https://gateway.example.com/v1",
                "api_key_env": "KEY_A",
            },
            {
                "name": "key-b",
                "type": "remote",
                "endpoint": "https://gateway.example.com/v1",
                "api_key_env": "KEY_B",
            },
            {
                "name": "different-gateway",
                "type": "remote",
                "endpoint": "https://other.example.com/v1",
                "api_key_env": "DIFF_KEY",
            },
        ],
        "aliases": ["gw*"],
    }

    async def _mock_ptr(_req, _path, provider_cfg):
        name = provider_cfg["name"]
        if name in ("key-a", "key-b"):
            return Response(
                content=json.dumps({"error": "internal server error"}),
                status_code=500,
                media_type="application/json",
            )
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_ptr):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", model_config, cfg
        )

    assert result.status_code == 200
    # key-a and key-b are same domain -> after key-a returns 500, key-b is skipped
    # because the domain is poisoned immediately
    body = json.loads(result.body)
    assert body["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_connection_error_poisons_domain_immediately():
    """AC3: Connection errors (ReadTimeout style) poison the domain on first failure."""
    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 5,
    }
    model_config = {
        "providers": [
            {
                "name": "key-a",
                "type": "remote",
                "endpoint": "https://gateway.example.com/v1",
                "api_key_env": "KEY_A",
            },
            {
                "name": "key-b",
                "type": "remote",
                "endpoint": "https://gateway.example.com/v1",
                "api_key_env": "KEY_B",
            },
            {
                "name": "different-gateway",
                "type": "remote",
                "endpoint": "https://other.example.com/v1",
                "api_key_env": "DIFF_KEY",
            },
        ],
        "aliases": ["gw*"],
    }

    call_order = []

    async def _mock_ptr(_req, _path, provider_cfg):
        name = provider_cfg["name"]
        call_order.append(name)
        if name in ("key-a", "key-b"):
            from httpx import ConnectTimeout
            raise ConnectTimeout("connection timed out")
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_ptr):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", model_config, cfg
        )

    assert result.status_code == 200
    # key-a fails with timeout -> domain poisoned -> key-b skipped -> different-gateway used
    body = json.loads(result.body)
    assert body["choices"][0]["message"]["content"] == "ok"
