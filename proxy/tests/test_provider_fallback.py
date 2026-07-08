"""
Unit tests for provider fallback: config schema & provider resolution.

Tests for:
- Config parsing with providers list
- resolve_provider() function behaviour
"""

import json
import pytest
import time
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException, Request, Response

import proxy.provider as provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DummyRequest:
    """Minimal request stub for use in fallback tests."""
    def __init__(self, body: bytes = b'{"model":"test"}'):
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
    """Reset cooldown and failure-count state between tests to avoid cross-test leakage."""
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
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


@pytest.fixture
def mixed_model_config():
    """A model config with both local and remote providers."""
    return {
        "providers": [
            {
                "name": "local-llama",
                "type": "local",
                "llama_model": "Qwen3",
            },
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
        ],
        "aliases": ["hybrid*"],
    }


@pytest.fixture
def single_provider_config():
    """A model config with a single provider."""
    return {
        "providers": [
            {
                "name": "sole-provider",
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
                "api_key_env": "EXAMPLE_API_KEY",
            },
        ],
        "aliases": ["sole*"],
    }


# ---------------------------------------------------------------------------
# Config parsing tests
# ---------------------------------------------------------------------------

def test_config_has_providers_list(sample_model_config):
    """Config should contain an ordered providers list."""
    providers_list = sample_model_config.get("providers")
    assert isinstance(providers_list, list)
    assert len(providers_list) == 2


def test_provider_entry_has_required_fields(sample_model_config):
    """Each provider entry must have name, type, and type-specific fields."""
    for entry in sample_model_config["providers"]:
        assert "name" in entry
        assert "type" in entry
        assert entry["type"] in ("local", "remote")

    # Remote provider fields
    remote = sample_model_config["providers"][0]
    assert remote["type"] == "remote"
    assert "endpoint" in remote
    assert "api_key_env" in remote


def test_provider_entry_local_fields(mixed_model_config):
    """Local provider must have llama_model."""
    local = mixed_model_config["providers"][0]
    assert local["type"] == "local"
    assert "llama_model" in local


# ---------------------------------------------------------------------------
# resolve_provider tests
# ---------------------------------------------------------------------------

def test_resolve_provider_returns_first_available(sample_model_config):
    """resolve_provider should return the first provider when no cooldown."""
    result = provider.resolve_provider(sample_model_config)
    assert result is not None
    assert result["name"] == "remote-primary"
    assert result["type"] == "remote"
    assert result["endpoint"] == "https://api.openai.com/v1"


def test_resolve_provider_skips_failed_provider(sample_model_config):
    """resolve_provider should skip the failed_provider and return the next."""
    result = provider.resolve_provider(
        sample_model_config, failed_provider="remote-primary"
    )
    assert result is not None
    assert result["name"] == "remote-fallback"
    assert result["type"] == "remote"


def test_resolve_provider_returns_none_when_all_exhausted(sample_model_config):
    """resolve_provider should return None when all providers are exhausted.

    Exhausted means ALL providers are either in cooldown or have been
    skipped via failed_provider, so none are available.
    """
    # Mark one provider in cooldown and fail the other
    provider.mark_provider_unavailable("remote-primary", 60.0)
    result = provider.resolve_provider(
        sample_model_config, failed_provider="remote-fallback"
    )
    # remote-primary is in cooldown, remote-fallback is the failed_provider
    assert result is None


def test_resolve_provider_returns_none_for_empty_providers_list():
    """resolve_provider should return None for an empty providers list."""
    config = {"providers": [], "aliases": ["empty*"]}
    result = provider.resolve_provider(config)
    assert result is None


def test_resolve_provider_returns_none_for_missing_providers_key():
    """resolve_provider should return None when providers key is missing."""
    config = {"aliases": ["noprov*"]}
    result = provider.resolve_provider(config)
    assert result is None


# ---------------------------------------------------------------------------
# Cooldown / unavailability tests
# ---------------------------------------------------------------------------

def test_resolve_provider_skips_provider_in_cooldown(sample_model_config):
    """resolve_provider should skip providers that are in cooldown."""
    # Mark the first provider as unavailable (in cooldown)
    provider.mark_provider_unavailable("remote-primary", 60.0)

    result = provider.resolve_provider(sample_model_config)
    assert result is not None
    assert result["name"] == "remote-fallback"


def test_resolve_provider_skips_all_in_cooldown(sample_model_config):
    """resolve_provider should return None when all providers are in cooldown."""
    provider.mark_provider_unavailable("remote-primary", 60.0)
    provider.mark_provider_unavailable("remote-fallback", 60.0)

    result = provider.resolve_provider(sample_model_config)
    assert result is None


def test_cooldown_expiry(sample_model_config):
    """resolve_provider should return a provider after its cooldown expires."""
    provider.mark_provider_unavailable("remote-primary", 0.01)  # very short cooldown
    # First call should skip it
    result = provider.resolve_provider(sample_model_config)
    assert result["name"] == "remote-fallback"

    # Wait for cooldown to expire
    time.sleep(0.02)

    # Now the first provider should be available again
    result = provider.resolve_provider(sample_model_config)
    assert result is not None
    assert result["name"] == "remote-primary"


def test_mark_provider_unavailable_stores_timestamp():
    """mark_provider_unavailable should store an expiry timestamp."""
    provider.mark_provider_unavailable("test-provider", 30.0)
    assert "test-provider" in provider._provider_unavailable_until
    expiry = provider._provider_unavailable_until["test-provider"]
    assert expiry > time.time()
    assert expiry <= time.time() + 31.0  # Allow small timing delta


def test_provider_not_in_cooldown_is_available():
    """A provider that was never marked should be available."""
    assert provider._is_provider_unavailable("fresh-provider") is False


def test_provider_in_cooldown_is_unavailable():
    """A provider in cooldown should be reported as unavailable."""
    provider.mark_provider_unavailable("down-provider", 60.0)
    assert provider._is_provider_unavailable("down-provider") is True


def test_expired_cooldown_is_available():
    """A provider whose cooldown expired should be available again."""
    provider.mark_provider_unavailable("recovered-provider", 0.001)
    time.sleep(0.01)
    assert provider._is_provider_unavailable("recovered-provider") is False
    assert "recovered-provider" not in provider._provider_unavailable_until


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_resolve_provider_no_failed_provider_single(single_provider_config):
    """resolve_provider should return the sole provider when available."""
    result = provider.resolve_provider(single_provider_config)
    assert result is not None
    assert result["name"] == "sole-provider"


def test_resolve_provider_failed_single_exhausts(single_provider_config):
    """resolve_provider should return None when the sole provider fails."""
    result = provider.resolve_provider(single_provider_config, failed_provider="sole-provider")
    assert result is None


def test_failed_and_cooldown_both_skip(sample_model_config):
    """A provider is skipped if it's the failed_provider OR in cooldown."""
    # Mark remote-fallback in cooldown
    provider.mark_provider_unavailable("remote-fallback", 60.0)

    # Call with remote-primary as the failed_provider.
    # remote-primary is skipped (failed_provider)
    # remote-fallback is skipped (in cooldown)
    result = provider.resolve_provider(
        sample_model_config, failed_provider="remote-primary"
    )
    assert result is None


# ===================================================================
# Remote provider fallback tests
# ===================================================================

@pytest.mark.asyncio
async def test_remote_fallback_on_connection_error(sample_model_config):
    """Fallback should trigger on connection error and try next provider."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    # Mock proxy_to_remote to raise connection error on first call,
    # then succeed on second call.
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ConnectError("Connection refused")
        # Second call succeeds
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert result.status_code == 200
    assert call_count == 2  # First failed, second succeeded


@pytest.mark.asyncio
async def test_remote_fallback_on_http_4xx(sample_model_config):
    """Fallback should trigger on HTTP 4xx and try next provider."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return Response(status_code=429, content=b"Rate limited")
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert result.status_code == 200
    assert call_count == 2


@pytest.mark.asyncio
async def test_remote_fallback_on_http_5xx(sample_model_config):
    """Fallback should trigger on HTTP 5xx and try next provider."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return Response(status_code=502, content=b"Bad gateway")
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert result.status_code == 200
    assert call_count == 2


@pytest.mark.asyncio
async def test_remote_fallback_tries_providers_in_order(sample_model_config):
    """Fallback should try providers in the order they appear in config."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    attempted = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        attempted.append(provider_cfg.get("name"))
        return Response(status_code=502, content=b"Bad gateway")

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert attempted == ["remote-primary", "remote-fallback"]


@pytest.mark.asyncio
async def test_remote_fallback_all_exhausted_returns_503(sample_model_config):
    """When all providers are exhausted, return 503 with JSON body and retry_after."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        return Response(status_code=502, content=b"Bad gateway")

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert result.status_code == 503
    body = json.loads(result.body)
    assert "retry_after" in body
    assert isinstance(body["retry_after"], (int, float))


@pytest.mark.asyncio
async def test_remote_fallback_cooldown_skips_failed_providers(sample_model_config):
    """After a provider fails, subsequent requests should skip it via cooldown."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            # First two calls fail (both providers fail)
            return Response(status_code=502, content=b"Bad gateway")
        # Third call: remote-fallback is in cooldown, so this
        # should only get called for remote-primary (cooldown expired?)
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        # First request: both fail
        result1 = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )
        assert result1.status_code == 503

        # Second request: both providers should be in cooldown
        result2 = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )
        assert result2.status_code == 503


@pytest.mark.asyncio
async def test_remote_fallback_respects_retry_after(sample_model_config):
    """Retry-After header from upstream response should extend cooldown."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        if provider_cfg.get("name") == "remote-primary":
            return Response(
                status_code=429,
                content=b"Rate limited",
                headers={"Retry-After": "120"},
            )
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert result.status_code == 200
    # remote-primary should be in cooldown with the larger of
    # provider_cooldown_seconds (60) and Retry-After (120)
    assert provider._is_provider_unavailable("remote-primary")
    # Verify the cooldown expiry is approximately now + 120s (max of 60 and 120)
    now = time.time()
    expiry = provider._provider_unavailable_until.get("remote-primary")
    assert expiry is not None, "remote-primary should have a cooldown expiry"
    # Allow 2s tolerance for test execution time
    assert expiry >= now + 118, (
        f"Expected expiry ~now+120s, got {expiry - now:.1f}s from now"
    )
    assert expiry <= now + 125, (
        f"Expiry too far in the future: {expiry - now:.1f}s"
    )


@pytest.mark.asyncio
async def test_remote_fallback_single_provider_fails(single_provider_config):
    """Single provider that fails should return 503."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        return Response(status_code=502, content=b"Bad gateway")

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", single_provider_config, cfg
        )

    assert result.status_code == 503
    body = json.loads(result.body)
    assert "retry_after" in body


@pytest.mark.asyncio
async def test_remote_model_loading_503_does_not_poison_provider_cooldown(single_provider_config):
    """A model_loading 503 should not keep a provider in cooldown across requests.

    Regression: first request can return model_loading while backend spins up,
    but retries after load should still attempt the same provider instead of
    short-circuiting to "All providers exhausted".
    """
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return Response(
                status_code=503,
                content=json.dumps({
                    "error": {
                        "type": "model_loading",
                        "code": "model_loading",
                        "message": "Model Qwen3 is loading, retry shortly",
                    },
                    "status": 503,
                    "retry_after": 30,
                }).encode("utf-8"),
                media_type="application/json",
                headers={"Retry-After": "30"},
            )
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        first = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", single_provider_config, cfg
        )
        assert first.status_code == 503
        first_body = first.body.decode("utf-8")
        assert "model_loading" in first_body or "loading" in first_body.lower()

        # Crucial: transient model-loading should not poison provider health.
        assert not provider._is_provider_unavailable("sole-provider")

        second = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", single_provider_config, cfg
        )

    assert second.status_code == 200
    assert call_count == 2


# ===================================================================
# Local-to-remote fallback tests
# ===================================================================

@pytest.mark.asyncio
async def test_local_fallback_to_remote_on_connection_error(mixed_model_config):
    """Local model connection error should trigger fallback to remote."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}
    call_log = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        call_log.append(("remote", provider_cfg.get("name")))
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    async def _mock_proxy_to_local(_req, _path):
        call_log.append(("local", "local-llama"))
        raise httpx.ConnectError("Connection refused to llama-server")

    with (
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert call_log == [
        ("local", "local-llama"),
        ("remote", "remote-fallback"),
    ]


@pytest.mark.asyncio
async def test_local_streaming_response_returned_as_success_no_remote_fallback(mixed_model_config):
    """A 2xx StreamingResponse from local must be returned as success, not
    treated as an empty response that triggers remote fallback.

    Regression for pi CLI requests with stream=true: proxy_with_fallback used
    to read the (empty) body of the StreamingResponse and fall back to remote.
    """
    from starlette.responses import StreamingResponse as _StreamingResponse

    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    local_called = 0
    remote_called = False

    async def _gen():
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield b'data: [DONE]\n\n'

    async def _mock_proxy_to_local(_req, _path):
        nonlocal local_called
        local_called += 1
        return _StreamingResponse(content=_gen(), status_code=200, media_type="text/event-stream")

    async def _mock_proxy_to_remote(_req, _path, _provider_cfg):
        nonlocal remote_called
        remote_called = True
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok-remote"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with (
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert local_called == 1
    assert not remote_called, "StreamingResponse should not trigger remote fallback"
    assert result.status_code == 200
    assert result.headers.get("X-Provider") == "local-llama"


@pytest.mark.asyncio
async def test_local_slot_exhaustion_uses_short_cooldown_not_provider_cooldown(mixed_model_config):
    """Local slot-exhaustion should use the short slot cooldown (5s) so the
    next request can retry local soon, instead of the full provider cooldown.
    """
    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 60,
        "server": {"slot_unavailable_retry_after": 5},
        "slot_unavailable_retry_after": 5,
    }

    async def _mock_proxy_to_local(_req, _path):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={
                "error": {"type": "server_busy", "code": "no_slots_available",
                          "message": "0/1 slots"},
                "total_slots": 1, "available_slots": 0,
            },
        )

    async def _mock_proxy_to_remote(_req, _path, _provider_cfg):
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200, media_type="application/json",
        )

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
    ):
        await provider.proxy_with_fallback(request, "v1/chat/completions", mixed_model_config, cfg)

    expiry = provider._provider_unavailable_until.get("local-llama")
    assert expiry is not None
    remaining = expiry - time.time()
    # Short cooldown (~5s) not the full provider cooldown (60s).
    assert remaining <= 6.5, f"expected short slot cooldown, got {remaining}s"
    assert remaining >= 3.0, f"expected ~5s cooldown, got {remaining}s"


@pytest.mark.asyncio
async def test_local_flat_slot_exhaustion_format_uses_short_cooldown(mixed_model_config):
    """Llama-server native flat 503 slot-busy format should be parsed as slot
    exhaustion and use the short slot cooldown (not full provider cooldown)."""
    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 60,
        "server": {"slot_unavailable_retry_after": 5},
        "slot_unavailable_retry_after": 5,
    }

    async def _mock_proxy_to_local(_req, _path):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={
                "type": "server_busy",
                "code": "no_slots_available",
                "message": "Model server busy: 0/1 slots available. Please retry later.",
            },
        )

    async def _mock_proxy_to_remote(_req, _path, _provider_cfg):
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
    ):
        await provider.proxy_with_fallback(request, "v1/chat/completions", mixed_model_config, cfg)

    expiry = provider._provider_unavailable_until.get("local-llama")
    assert expiry is not None
    remaining = expiry - time.time()
    assert remaining <= 6.5, f"expected short slot cooldown, got {remaining}s"
    assert remaining >= 3.0, f"expected ~5s cooldown, got {remaining}s"


@pytest.mark.asyncio
async def test_local_fallback_on_slot_exhaustion(mixed_model_config):
    """Slot exhaustion (all slots busy) should trigger fallback to remote."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}
    call_log = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        call_log.append(("remote", provider_cfg.get("name")))
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    async def _mock_proxy_to_local(_req, _path):
        call_log.append(("local", "local-llama"))
        # Simulate slot exhaustion response
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "server_busy",
                    "code": "no_slots_available",
                    "message": "Model server busy: 0/1 slots available. Please retry later.",
                },
                "status": 503,
                "retry_after": 5,
                "total_slots": 1,
                "available_slots": 0,
            },
        )

    with (
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert call_log == [
        ("local", "local-llama"),
        ("remote", "remote-fallback"),
    ]


@pytest.mark.asyncio
async def test_local_slot_exhaustion_retry_prefers_local_before_remote(mixed_model_config):
    """Mimic Pi startup race: first local call reports slot exhaustion,
    then local succeeds shortly after. With local slot retry enabled,
    fallback should stay local and avoid remote.
    """
    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 60,
        "server": {
            "local_slot_exhaustion_retry_attempts": 1,
            "local_slot_exhaustion_retry_delay_seconds": 0,
        },
    }

    local_calls = 0
    remote_called = False

    async def _mock_proxy_to_local(_req, _path):
        nonlocal local_calls
        local_calls += 1
        from fastapi.responses import JSONResponse
        if local_calls == 1:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "type": "server_busy",
                        "code": "no_slots_available",
                        "message": "Model server busy: 0/1 slots available. Please retry later.",
                    },
                    "status": 503,
                    "retry_after": 5,
                    "total_slots": 1,
                    "available_slots": 0,
                },
            )
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok-local"}}]}),
            status_code=200,
            media_type="application/json",
        )

    async def _mock_proxy_to_remote(_req, _path, _provider_cfg):
        nonlocal remote_called
        remote_called = True
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok-remote"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with (
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert result.headers.get("X-Provider") == "local-llama"
    assert local_calls == 2
    assert not remote_called
    assert "local-llama" not in provider._provider_unavailable_until


@pytest.mark.asyncio
async def test_local_503_retry_prefers_local_before_remote(mixed_model_config):
    """Transient local 503 should be retried locally before remote fallback."""
    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 60,
        "server": {
            "local_slot_exhaustion_retry_attempts": 1,
            "local_slot_exhaustion_retry_delay_seconds": 0,
        },
    }

    local_calls = 0
    remote_called = False

    async def _mock_proxy_to_local(_req, _path):
        nonlocal local_calls
        local_calls += 1
        if local_calls == 1:
            return Response(
                content=json.dumps({"error": {"message": "backend busy"}}),
                status_code=503,
                media_type="application/json",
            )
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok-local"}}]}),
            status_code=200,
            media_type="application/json",
        )

    async def _mock_proxy_to_remote(_req, _path, _provider_cfg):
        nonlocal remote_called
        remote_called = True
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok-remote"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with (
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert result.headers.get("X-Provider") == "local-llama"
    assert local_calls == 2
    assert not remote_called


@pytest.mark.asyncio
async def test_local_http_exception_503_retry_prefers_local_before_remote(mixed_model_config):
    """Transient local HTTPException(503) should be retried locally before remote."""
    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 60,
        "server": {
            "local_slot_exhaustion_retry_attempts": 1,
            "local_slot_exhaustion_retry_delay_seconds": 0,
        },
    }

    local_calls = 0
    remote_called = False

    async def _mock_proxy_to_local(_req, _path):
        nonlocal local_calls
        local_calls += 1
        if local_calls == 1:
            raise HTTPException(status_code=503, detail="Backend busy")
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok-local"}}]}),
            status_code=200,
            media_type="application/json",
        )

    async def _mock_proxy_to_remote(_req, _path, _provider_cfg):
        nonlocal remote_called
        remote_called = True
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok-remote"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with (
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert result.headers.get("X-Provider") == "local-llama"
    assert local_calls == 2
    assert not remote_called


@pytest.mark.asyncio
async def test_local_empty_200_retry_prefers_local_before_remote(mixed_model_config):
    """Transient local empty 200 should be retried locally before remote fallback."""
    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 60,
        "server": {
            "local_slot_exhaustion_retry_attempts": 1,
            "local_slot_exhaustion_retry_delay_seconds": 0,
        },
    }

    local_calls = 0
    remote_called = False

    async def _mock_proxy_to_local(_req, _path):
        nonlocal local_calls
        local_calls += 1
        if local_calls == 1:
            return Response(content=b"", status_code=200, media_type="application/json")
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok-local"}}]}),
            status_code=200,
            media_type="application/json",
        )

    async def _mock_proxy_to_remote(_req, _path, _provider_cfg):
        nonlocal remote_called
        remote_called = True
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok-remote"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with (
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert result.headers.get("X-Provider") == "local-llama"
    assert local_calls == 2
    assert not remote_called


@pytest.mark.asyncio
async def test_local_fallback_all_exhausted(mixed_model_config):
    """When all providers (local + remote) fail, return 503."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}
    call_log = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        call_log.append(("remote", provider_cfg.get("name")))
        return Response(status_code=502, content=b"Bad gateway")

    async def _mock_proxy_to_local(_req, _path):
        call_log.append(("local", "local-llama"))
        raise httpx.ConnectError("Connection refused")

    with (
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    # The first provider to return a non-success response is preserved
    # instead of the generic "All providers exhausted" message.
    # Here the remote provider returns 502, so that's what the client sees.
    assert result.status_code == 502
    assert call_log == [
        ("local", "local-llama"),
        ("remote", "remote-fallback"),
    ]
    assert b"Bad gateway" in result.body


@pytest.mark.asyncio
async def test_fallback_preserves_request_semantics(mixed_model_config):
    """When local fallback occurs, the forwarded request to the remote provider
    must preserve method, headers, body, and the path argument passed to the
    proxy. This verifies request semantics are preserved during fallback.
    """
    request = _DummyRequest(body=b'{"prompt":"hello"}')
    # Add some headers to ensure they are forwarded
    request.headers = {
        "Authorization": "Bearer test-token",
        "X-Custom-Header": "custom",
    }

    cfg = {"provider_cooldown_seconds": 60}
    observed = {}

    async def _mock_proxy_to_local(_req, _path):
        # Simulate local failure to force fallback
        raise httpx.ConnectError("Local llama-server unreachable")

    async def _mock_proxy_to_remote(req, path, provider_cfg):
        # Capture the forwarded request semantics
        observed["method"] = req.method
        # Make a shallow copy of headers to avoid framework-specific types
        observed["headers"] = dict(req.headers)
        observed["body"] = await req.body()
        observed["path"] = path
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with (
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    # Verify the remote proxy received the same method
    assert observed.get("method") == request.method
    # Verify headers were forwarded
    assert observed.get("headers", {}).get("Authorization") == "Bearer test-token"
    assert observed.get("headers", {}).get("X-Custom-Header") == "custom"
    # Verify body preserved
    assert observed.get("body") == b'{"prompt":"hello"}'
    # Verify path argument forwarded matches the requested path
    assert observed.get("path") == "v1/chat/completions"


# ===================================================================
# Observability tests (X-Provider header, fallback logging)
# ===================================================================

@pytest.mark.asyncio
async def test_x_provider_header_on_success(sample_model_config):
    """Successful response should include X-Provider header."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert result.status_code == 200
    assert result.headers.get("X-Provider") == "remote-primary"


@pytest.mark.asyncio
async def test_x_provider_header_on_fallback(sample_model_config):
    """On fallback, X-Provider should reflect the successful provider."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return Response(status_code=502, content=b"Bad gateway")
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert result.status_code == 200
    # After fallback from remote-primary, remote-fallback should be in header
    assert result.headers.get("X-Provider") == "remote-fallback"


@pytest.mark.asyncio
async def test_fallback_logging(sample_model_config, caplog):
    """Fallback events should be logged at INFO level."""
    import logging
    caplog.set_level(logging.INFO, logger="llama-proxy.provider")

    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return Response(status_code=502, content=b"Bad gateway")
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert result.status_code == 200
    # Check that a fallback log was emitted
    found = False
    for record in caplog.records:
        if "Fallback triggered" in record.getMessage():
            found = True
            assert record.levelname == "INFO"
            assert "remote-primary" in record.getMessage()
            assert "remote-fallback" in record.getMessage()
            assert "HTTP 502" in record.getMessage()
            break
    assert found, "Expected 'Fallback triggered' log message not found"


@pytest.mark.asyncio
async def test_no_fallback_log_on_success(sample_model_config, caplog):
    """No fallback log should be emitted when first provider succeeds."""
    import logging
    caplog.set_level(logging.INFO, logger="llama-proxy.provider")

    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert result.status_code == 200
    for record in caplog.records:
        assert "Fallback triggered" not in record.getMessage()


# ===================================================================
# Queue bypass tests (LP-0MR5MAJNM005R905)
# ===================================================================


@pytest.mark.asyncio
async def test_local_queue_bypass_when_slot_busy(mixed_model_config):
    """When scheduler has no idle slots and model has fallback providers,
    skip the local provider immediately without marking it unavailable."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}
    call_log = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        call_log.append(("remote", provider_cfg.get("name")))
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    async def _mock_proxy_to_local(_req, _path):
        call_log.append(("local", "local-llama"))
        raise AssertionError("Should not reach local provider when slot is busy")

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_scheduler_has_idle_slot", return_value=False),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    # Local provider should NOT have been called; remote was used directly
    assert call_log == [("remote", "remote-fallback")]
    # Local provider should NOT be marked unavailable (slot busy, not failed)
    assert "local-llama" not in provider._provider_unavailable_until


@pytest.mark.asyncio
async def test_local_queue_bypass_no_fallback_passes_through(mixed_model_config):
    """When scheduler has idle slot, local provider is called normally
    even when fallback providers exist."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}
    call_log = []

    async def _mock_proxy_to_local(_req, _path):
        call_log.append(("local", "local-llama"))
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        call_log.append(("remote", provider_cfg.get("name")))
        raise AssertionError("Should not reach remote when local has idle slot")

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_scheduler_has_idle_slot", return_value=True),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert call_log == [("local", "local-llama")]


@pytest.mark.asyncio
async def test_local_concurrency_limit_fallback(mixed_model_config):
    """When local concurrency limit is reached, skip to next provider
    without marking local as unavailable."""
    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 60,
        "server": {"local_max_concurrent_queries": 1},
    }
    call_log = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        call_log.append(("remote", provider_cfg.get("name")))
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    async def _mock_proxy_to_local(_req, _path):
        call_log.append(("local", "local-llama"))
        raise AssertionError("Should not reach local when at concurrency limit")

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", return_value=(1, 1)),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert call_log == [("remote", "remote-fallback")]
    assert "local-llama" not in provider._provider_unavailable_until


@pytest.mark.asyncio
async def test_local_concurrency_below_limit_calls_local(mixed_model_config):
    """When local concurrency is below limit, local provider is called."""
    request = _DummyRequest()
    cfg = {
        "provider_cooldown_seconds": 60,
        "server": {"local_max_concurrent_queries": 1},
    }
    call_log = []

    async def _mock_proxy_to_local(_req, _path):
        call_log.append(("local", "local-llama"))
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        call_log.append(("remote", provider_cfg.get("name")))
        raise AssertionError("Should not reach remote when local is within concurrency limit")

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.provider._get_local_concurrency_info", return_value=(0, 1)),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert call_log == [("local", "local-llama")]


# ===================================================================
# Integration tests: ui.py request path uses fallback functions
# ===================================================================


@pytest.mark.asyncio
async def test_remote_model_with_providers_calls_fallback(monkeypatch):
    """proxy_openai_api should call proxy_with_remote_fallback for remote models with providers."""
    import proxy.server as server_module
    from unittest.mock import MagicMock, AsyncMock

    # Set up config with a remote model that has providers
    server_module.config = {
        "models": {
            "test-remote": {
                "type": "remote",
                "providers": [
                    {"name": "primary", "type": "remote",
                     "endpoint": "https://primary.test/v1", "api_key_env": "KEY1"},
                    {"name": "backup", "type": "remote",
                     "endpoint": "https://backup.test/v1", "api_key_env": "KEY2"},
                ],
            },
        },
        "server": {"llama_request_timeout": 300},
    }
    server_module.current_model = None
    server_module.llama_process = None
    server_module.backend_ready = True

    # Track that proxy_with_remote_fallback was called
    fallback_called = False
    orig_fallback = provider.proxy_with_remote_fallback

    async def mock_fallback(request, path, model_config, config):
        nonlocal fallback_called
        fallback_called = True
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.provider.proxy_with_remote_fallback", mock_fallback):
        from proxy.ui import proxy_openai_api
        from fastapi import Request as FastAPIRequest

        body = json.dumps({
            "model": "test-remote",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }).encode("utf-8")

        mock_request = MagicMock(spec=FastAPIRequest)
        mock_request.method = "POST"
        mock_request.url = type("U", (), {"path": "/v1/chat/completions"})()
        mock_request.headers = {}
        mock_request._body = body
        async def mock_body():
            return mock_request._body
        mock_request.body = mock_body

        resp = await proxy_openai_api(mock_request, "chat/completions")

    assert fallback_called, \
        "proxy_with_remote_fallback should be called for remote model with providers"
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_remote_model_without_providers_returns_error(monkeypatch):
    """Remote models WITHOUT a providers list should return an error (breaking change)."""
    import proxy.server as server_module
    from unittest.mock import MagicMock
    from fastapi import HTTPException

    # Set up config with a remote model that has NO providers (legacy format)
    server_module.config = {
        "models": {
            "test-remote": {
                "type": "remote",
                "endpoint": "https://test.api.com/v1",
                "api_key_env": "TEST_KEY",
            },
        },
        "server": {"llama_request_timeout": 300},
    }
    server_module.current_model = None
    server_module.llama_process = None
    server_module.backend_ready = True

    from proxy.ui import proxy_openai_api
    from fastapi import Request as FastAPIRequest

    body = json.dumps({
        "model": "test-remote",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }).encode("utf-8")

    mock_request = MagicMock(spec=FastAPIRequest)
    mock_request.method = "POST"
    mock_request.url = type("U", (), {"path": "/v1/chat/completions"})()
    mock_request.headers = {}
    mock_request._body = body
    async def mock_body():
        return mock_request._body
    mock_request.body = mock_body

    with pytest.raises(HTTPException) as exc_info:
        await proxy_openai_api(mock_request, "chat/completions")

    assert exc_info.value.status_code == 500, "Legacy remote model without providers should return error"
    assert "Invalid model configuration" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_local_model_with_providers_calls_fallback(monkeypatch):
    """proxy_openai_api should call proxy_with_fallback for loaded local models with providers."""
    import proxy.server as server_module
    from unittest.mock import MagicMock

    # Set up config with a local model that has providers (local + remote)
    server_module.config = {
        "models": {
            "test-local": {
                "type": "local",
                "llama_model": "test-llama",
                "providers": [
                    {"name": "local-instance", "type": "local", "llama_model": "test-llama"},
                    {"name": "remote-backup", "type": "remote",
                     "endpoint": "https://backup.test/v1", "api_key_env": "KEY"},
                ],
            },
        },
        "server": {"llama_request_timeout": 300},
    }
    # Simulate model is already loaded
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    server_module.llama_process = fake_proc
    server_module.backend_ready = True
    server_module.current_model = "test-llama"

    fallback_called = False

    async def mock_fallback(request, path, model_config, config):
        nonlocal fallback_called
        fallback_called = True
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.provider.proxy_with_fallback", mock_fallback):
        from proxy.ui import proxy_openai_api
        from fastapi import Request as FastAPIRequest

        body = json.dumps({
            "model": "test-local",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }).encode("utf-8")

        mock_request = MagicMock(spec=FastAPIRequest)
        mock_request.method = "POST"
        mock_request.url = type("U", (), {"path": "/v1/chat/completions"})()
        mock_request.headers = {}
        mock_request._body = body
        async def mock_body():
            return mock_request._body
        mock_request.body = mock_body

        resp = await proxy_openai_api(mock_request, "chat/completions")

    assert fallback_called, \
        "proxy_with_fallback should be called for local model with providers"
    assert resp.status_code == 200


# ===================================================================
# Request-header / model-override tests
# ===================================================================


def test_normalize_upstream_request_headers_strips_hop_by_hop_headers():
    from proxy.router_helpers import normalize_upstream_request_headers

    incoming = {
        "Host": "localhost:8000",
        "Content-Length": "123",
        "Connection": "keep-alive, x-drop-me",
        "Keep-Alive": "timeout=5",
        "Transfer-Encoding": "chunked",
        "TE": "trailers",
        "Trailer": "x-trailer",
        "Expect": "100-continue",
        "Proxy-Connection": "keep-alive",
        "X-Drop-Me": "1",
        "Authorization": "Bearer abc",
        "Content-Type": "application/json",
    }

    normalized = normalize_upstream_request_headers(incoming)

    assert "Host" not in normalized
    assert "Content-Length" not in normalized
    assert "Connection" not in normalized
    assert "Keep-Alive" not in normalized
    assert "Transfer-Encoding" not in normalized
    assert "TE" not in normalized
    assert "Trailer" not in normalized
    assert "Expect" not in normalized
    assert "Proxy-Connection" not in normalized
    assert "X-Drop-Me" not in normalized

    assert normalized.get("Authorization") == "Bearer abc"
    assert normalized.get("Content-Type") == "application/json"


@pytest.mark.asyncio
async def test_proxy_to_remote_strips_hop_by_hop_headers_before_forwarding():
    import proxy.server as server_module
    from proxy.proxy_remote import proxy_to_remote
    from unittest.mock import patch as mock_patch

    server_module.config = {
        "server": {"llama_request_timeout": 300},
    }
    server_module.current_model = None

    request = _DummyRequest(body=b'{"model":"plan","messages":[{"role":"user","content":"hi"}],"stream":false}')
    request.headers = {
        "Connection": "keep-alive, x-drop-me",
        "Transfer-Encoding": "chunked",
        "X-Drop-Me": "1",
        "Content-Type": "application/json",
        "X-Custom": "ok",
    }

    provider_cfg = {
        "name": "opencode-deepseek-free",
        "type": "remote",
        "endpoint": "https://opencode.ai/zen",
        "api_key_env": "OPENCODE_API_KEY",
        "model": "deepseek-v4-flash-free",
    }

    observed_headers = None

    async def mock_non_streaming(_req, _url, headers, _body, _model_name, _timeout, **kwargs):
        nonlocal observed_headers
        observed_headers = headers
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with mock_patch("proxy.proxy_remote._handle_remote_non_streaming", mock_non_streaming):
        result = await proxy_to_remote(request, "v1/chat/completions", provider_cfg)

    assert result.status_code == 200
    assert observed_headers is not None
    assert "Connection" not in observed_headers
    assert "Transfer-Encoding" not in observed_headers
    assert "X-Drop-Me" not in observed_headers
    assert observed_headers.get("Content-Type") == "application/json"
    assert observed_headers.get("X-Custom") == "ok"


@pytest.mark.asyncio
async def test_proxy_to_remote_overrides_model_name_with_model_field():
    """When a remote provider config has a `model` field, proxy_to_remote
    should override the model name in the forwarded request body."""
    import proxy.server as server_module
    from proxy.proxy_remote import proxy_to_remote
    from unittest.mock import MagicMock, patch as mock_patch

    # Minimal server config needed for timeout
    server_module.config = {
        "server": {"llama_request_timeout": 300},
    }
    server_module.current_model = None

    request = _DummyRequest(body=b'{"model":"qwen3-fallback","messages":[{"role":"user","content":"hi"}],"stream":false}')
    request.headers = {}

    provider_cfg = {
        "name": "opencode-deepseek-free",
        "type": "remote",
        "endpoint": "https://opencode.ai/zen",
        "api_key_env": "OPENCODE_API_KEY",
        "model": "deepseek-v4-flash-free",
    }

    captured_body = None

    async def mock_non_streaming(_req, _url, _headers, body, _model_name, _timeout, **kwargs):
        nonlocal captured_body
        captured_body = body
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with mock_patch("proxy.proxy_remote._handle_remote_non_streaming", mock_non_streaming):
        result = await proxy_to_remote(request, "v1/chat/completions", provider_cfg)

    assert result.status_code == 200
    assert captured_body is not None, "_handle_remote_non_streaming should have been called"
    captured_json = json.loads(captured_body.decode("utf-8"))
    assert captured_json["model"] == "deepseek-v4-flash-free", \
        f"Expected model override to 'deepseek-v4-flash-free', got '{captured_json.get('model')}'"


@pytest.mark.asyncio
async def test_proxy_to_remote_strips_unknown_chat_fields_for_remote_compatibility():
    """Unknown top-level chat fields should be removed before forwarding to
    remote providers to avoid provider-specific 4xx request-shape failures."""
    import proxy.server as server_module
    from proxy.proxy_remote import proxy_to_remote
    from unittest.mock import patch as mock_patch

    server_module.config = {"server": {"llama_request_timeout": 300}}
    server_module.current_model = None

    request = _DummyRequest(
        body=b'{"model":"plan","messages":[{"role":"user","content":"hi"}],"stream":false,"max_tokens":16,"bogus_field":123,"another_unknown":{"x":1}}'
    )
    request.headers = {}

    provider_cfg = {
        "name": "plain-remote",
        "type": "remote",
        "endpoint": "https://example.com/v1",
        "api_key_env": "SOME_KEY",
    }

    captured_body = None

    async def mock_non_streaming(_req, _url, _headers, body, _model_name, _timeout, **kwargs):
        nonlocal captured_body
        captured_body = body
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with mock_patch("proxy.proxy_remote._handle_remote_non_streaming", mock_non_streaming):
        result = await proxy_to_remote(request, "v1/chat/completions", provider_cfg)

    assert result.status_code == 200
    assert captured_body is not None
    captured_json = json.loads(captured_body.decode("utf-8"))
    assert captured_json["model"] == "plan"
    assert captured_json["stream"] is False
    assert captured_json["max_tokens"] == 16
    assert "bogus_field" not in captured_json
    assert "another_unknown" not in captured_json


@pytest.mark.asyncio
async def test_proxy_to_remote_passes_model_unchanged_without_model_field():
    """When a remote provider config has NO `model` field, proxy_to_remote
    should pass the original model name through unchanged."""
    import proxy.server as server_module
    from proxy.proxy_remote import proxy_to_remote
    from unittest.mock import MagicMock, patch as mock_patch

    server_module.config = {
        "server": {"llama_request_timeout": 300},
    }
    server_module.current_model = None

    request = _DummyRequest(body=b'{"model":"my-original-model","messages":[{"role":"user","content":"hi"}],"stream":false}')
    request.headers = {}

    provider_cfg = {
        "name": "plain-remote",
        "type": "remote",
        "endpoint": "https://example.com/v1",
        "api_key_env": "SOME_KEY",
    }

    captured_body = None

    async def mock_non_streaming(_req, _url, _headers, body, _model_name, _timeout, **kwargs):
        nonlocal captured_body
        captured_body = body
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with mock_patch("proxy.proxy_remote._handle_remote_non_streaming", mock_non_streaming):
        result = await proxy_to_remote(request, "v1/chat/completions", provider_cfg)

    assert result.status_code == 200
    assert captured_body is not None
    captured_json = json.loads(captured_body.decode("utf-8"))
    assert captured_json["model"] == "my-original-model", \
        f"Expected original model 'my-original-model', got '{captured_json.get('model')}'"


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_local_model_not_loaded_remote_fallback_returns_503_returns_model_loading(monkeypatch):
    """When a local model with remote providers is NOT loaded and the remote
    fallback returns a 503 error, proxy_openai_api should return a model_loading
    response, not the 503 from the exhausted remote providers."""
    import proxy.server as server_module
    from unittest.mock import MagicMock

    # Set up config with a local model that has providers (local + remote)
    server_module.config = {
        "models": {
            "test-local": {
                "type": "local",
                "llama_model": "test-llama",
                "providers": [
                    {"name": "local-instance", "type": "local", "llama_model": "test-llama"},
                    {"name": "remote-backup", "type": "remote",
                     "endpoint": "https://backup.test/v1", "api_key_env": "KEY"},
                ],
            },
        },
        "server": {
            "llama_request_timeout": 300,
            "llama_router_mode": False,
        },
    }
    # Simulate model is NOT loaded (current_model is different)
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    server_module.llama_process = fake_proc
    server_module.backend_ready = True
    server_module.current_model = "some-other-model"
    server_module.background_loads = {}
    server_module.logger = MagicMock()
    server_module.schedule_background_load = MagicMock(return_value=True)
    server_module._model_loading_response = MagicMock(
        return_value=Response(
            content=json.dumps({
                "error": {"code": "model_loading", "message": "Model test-llama is loading"},
                "status": 503,
            }),
            status_code=503,
            media_type="application/json",
        )
    )
    server_module.get_local_model_name = lambda m: "test-llama" if m == "test-local" or m == "test-llama" else None

    # Mock proxy_with_remote_fallback to return a 503 (exhausted providers)
    mock_exhausted_resp = Response(
        content=json.dumps({"error": "All providers exhausted", "retry_after": 60}),
        status_code=503,
        media_type="application/json",
    )

    async def mock_remote_fallback(_req, _path, _cfg, _config):
        return mock_exhausted_resp

    with patch("proxy.provider.proxy_with_remote_fallback", mock_remote_fallback):
        # Also mock session/slot detection to skip (no session header)
        with patch("proxy.session._resolve_session_id_header", return_value=(None, None)):
            with patch("proxy.session._build_slot_context", return_value=(None, None, None)):
                from proxy.ui import proxy_openai_api
                from fastapi import Request as FastAPIRequest

                body = json.dumps({
                    "model": "test-local",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False,
                }).encode("utf-8")

                mock_request = MagicMock(spec=FastAPIRequest)
                mock_request.method = "POST"
                mock_request.url = type("U", (), {"path": "/v1/chat/completions"})()
                mock_request.headers = {}
                mock_request._body = body
                async def mock_body():
                    return mock_request._body
                mock_request.body = mock_body

                resp = await proxy_openai_api(mock_request, "chat/completions")

    # Should NOT be the exhausted providers response
    assert resp.status_code == 503
    body_text = resp.body.decode("utf-8") if hasattr(resp, "body") else ""
    # Should be a model_loading response, not "All providers exhausted"
    assert "All providers exhausted" not in body_text, \
        f"Should NOT propagate exhausted provider error: {body_text}"
    assert "model_loading" in body_text or "loading" in body_text.lower(), \
        f"Should return model_loading response: {body_text}"


@pytest.mark.asyncio
async def test_router_transient_not_loaded_then_loaded_prefers_local_before_remote(monkeypatch):
    """When router briefly reports a local model as not loaded, a short grace
    window should allow local fallback path to recover before remote fallback.
    """
    import proxy.server as server_module
    from unittest.mock import MagicMock

    server_module.config = {
        "models": {
            "test-local": {
                "type": "local",
                "llama_model": "test-llama",
                "providers": [
                    {"name": "local-instance", "type": "local", "llama_model": "test-llama"},
                    {"name": "remote-backup", "type": "remote", "endpoint": "https://backup.test/v1", "api_key_env": "KEY"},
                ],
            },
        },
        "server": {
            "llama_request_timeout": 300,
            "llama_router_mode": True,
            "model_loading_local_grace_seconds": 0.2,
        },
    }

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    server_module.llama_process = fake_proc
    server_module.backend_ready = True
    server_module.current_model = "some-other-model"
    server_module.background_loads = {}
    server_module.logger = MagicMock()
    server_module.schedule_background_load = MagicMock(return_value=True)

    # First fast-check says not loaded, grace-window check says loaded.
    check_calls = {"count": 0}

    async def mock_router_is_model_loaded(_model_name):
        check_calls["count"] += 1
        return check_calls["count"] >= 2

    server_module.router_is_model_loaded = mock_router_is_model_loaded
    server_module.get_local_model_name = lambda m: "test-llama" if m in ("test-local", "test-llama") else None

    local_fallback_called = False
    remote_fallback_called = False

    async def mock_with_fallback(_req, _path, _cfg, _config):
        nonlocal local_fallback_called
        local_fallback_called = True
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    async def mock_remote_fallback(_req, _path, _cfg, _config):
        nonlocal remote_fallback_called
        remote_fallback_called = True
        return Response(
            content=json.dumps({"error": "should-not-be-used"}),
            status_code=503,
            media_type="application/json",
        )

    with patch("proxy.provider.proxy_with_fallback", mock_with_fallback):
        with patch("proxy.provider.proxy_with_remote_fallback", mock_remote_fallback):
            with patch("proxy.session._resolve_session_id_header", return_value=(None, None)):
                with patch("proxy.session._build_slot_context", return_value=(None, None, None)):
                    from proxy.ui import proxy_openai_api
                    from fastapi import Request as FastAPIRequest

                    body = json.dumps({
                        "model": "test-local",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": False,
                    }).encode("utf-8")

                    mock_request = MagicMock(spec=FastAPIRequest)
                    mock_request.method = "POST"
                    mock_request.url = type("U", (), {"path": "/v1/chat/completions"})()
                    mock_request.headers = {}
                    mock_request._body = body

                    async def mock_body():
                        return mock_request._body

                    mock_request.body = mock_body

                    resp = await proxy_openai_api(mock_request, "chat/completions")

    assert resp.status_code == 200
    assert local_fallback_called, "Expected local fallback path to be used after grace-window recheck"
    assert not remote_fallback_called, "Remote fallback should not be used when local model becomes available"


@pytest.mark.asyncio
async def test_router_loading_status_but_router_load_model_already_loaded_prefers_local(monkeypatch):
    """If router status lags as 'loading' but router_load_model reports already
    loaded, local path should be preferred before remote fallback.
    """
    import proxy.server as server_module
    from unittest.mock import MagicMock

    server_module.config = {
        "models": {
            "test-local": {
                "type": "local",
                "llama_model": "test-llama",
                "providers": [
                    {"name": "local-instance", "type": "local", "llama_model": "test-llama"},
                    {"name": "remote-backup", "type": "remote", "endpoint": "https://backup.test/v1", "api_key_env": "KEY"},
                ],
            },
        },
        "server": {
            "llama_request_timeout": 300,
            "llama_router_mode": True,
            "model_loading_local_grace_seconds": 0.2,
        },
    }

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    server_module.llama_process = fake_proc
    server_module.backend_ready = True
    server_module.current_model = "some-other-model"
    server_module.background_loads = {}
    server_module.logger = MagicMock()
    server_module.schedule_background_load = MagicMock(return_value=True)

    async def mock_router_is_model_loaded(_model_name):
        return False

    async def mock_router_load_model(_model_name):
        # Simulate router API saying this model is already loaded.
        return True

    server_module.router_is_model_loaded = mock_router_is_model_loaded
    server_module.router_load_model = mock_router_load_model
    server_module.get_local_model_name = lambda m: "test-llama" if m in ("test-local", "test-llama") else None

    local_fallback_called = False
    remote_fallback_called = False

    async def mock_with_fallback(_req, _path, _cfg, _config):
        nonlocal local_fallback_called
        local_fallback_called = True
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    async def mock_remote_fallback(_req, _path, _cfg, _config):
        nonlocal remote_fallback_called
        remote_fallback_called = True
        return Response(
            content=json.dumps({"error": "should-not-be-used"}),
            status_code=503,
            media_type="application/json",
        )

    with patch("proxy.provider.proxy_with_fallback", mock_with_fallback):
        with patch("proxy.provider.proxy_with_remote_fallback", mock_remote_fallback):
            with patch("proxy.session._resolve_session_id_header", return_value=(None, None)):
                with patch("proxy.session._build_slot_context", return_value=(None, None, None)):
                    from proxy.ui import proxy_openai_api
                    from fastapi import Request as FastAPIRequest

                    body = json.dumps({
                        "model": "test-local",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": False,
                    }).encode("utf-8")

                    mock_request = MagicMock(spec=FastAPIRequest)
                    mock_request.method = "POST"
                    mock_request.url = type("U", (), {"path": "/v1/chat/completions"})()
                    mock_request.headers = {}
                    mock_request._body = body

                    async def mock_body():
                        return mock_request._body

                    mock_request.body = mock_body

                    resp = await proxy_openai_api(mock_request, "chat/completions")

    assert resp.status_code == 200
    assert local_fallback_called, "Expected local path after router_load_model confirms availability"
    assert not remote_fallback_called, "Remote fallback should not run when local availability is confirmed"


@pytest.mark.asyncio
async def test_local_400_falls_back_without_local_cooldown(mixed_model_config):
    """A local 4xx should allow same-request fallback but must not put the
    local provider in cooldown for future requests.
    """
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    local_call_count = 0
    call_log = []

    async def _mock_proxy_to_local(_req, _path):
        nonlocal local_call_count
        local_call_count += 1
        call_log.append(("local", local_call_count))
        if local_call_count == 1:
            return Response(
                content=json.dumps({"error": {"message": "invalid_request"}}),
                status_code=400,
                media_type="application/json",
            )
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok-local"}}]}),
            status_code=200,
            media_type="application/json",
        )

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        call_log.append(("remote", provider_cfg.get("name")))
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok-remote"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with (
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
    ):
        first = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

        assert first.status_code == 200
        assert first.headers.get("X-Provider") == "remote-fallback"
        assert "local-llama" not in provider._provider_unavailable_until

        second = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert second.status_code == 200
    assert second.headers.get("X-Provider") == "local-llama"
    assert call_log == [
        ("local", 1),
        ("remote", "remote-fallback"),
        ("local", 2),
    ]


@pytest.mark.asyncio
async def test_local_http_exception_triggers_fallback(mixed_model_config):
    """HTTPException (e.g., 503 backend busy) from a local provider should
    trigger fallback to the next provider."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}
    call_log = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        call_log.append(("remote", provider_cfg.get("name")))
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    async def _mock_proxy_to_local(_req, _path):
        call_log.append(("local", "local-llama"))
        # Simulate a 503 backend-unavailable HTTPException (e.g., concurrency limit)
        raise HTTPException(status_code=503, detail="Backend busy")

    with (
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert call_log == [
        ("local", "local-llama"),
        ("remote", "remote-fallback"),
    ], f"Expected fallback to remote, got call_log={call_log}"


# ===================================================================
# Go tier (opencode-go-deepseek) regression tests
# ===================================================================


@pytest.mark.asyncio
async def test_proxy_to_remote_with_opencode_go_deepseek_model_override():
    """proxy_to_remote with the opencode-go-deepseek provider config must
    override the model name from the incoming request to 'deepseek-v4-flash'."""
    import proxy.server as server_module
    from proxy.proxy_remote import proxy_to_remote
    from unittest.mock import patch as mock_patch

    server_module.config = {
        "server": {"llama_request_timeout": 300},
    }
    server_module.current_model = None

    request = _DummyRequest(body=b'{"model":"plan","messages":[{"role":"user","content":"hi"}],"stream":false}')
    request.headers = {}

    provider_cfg = {
        "name": "opencode-go-deepseek",
        "type": "remote",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": "OPENCODE_API_KEY",
        "model": "deepseek-v4-flash",
    }

    captured_body = None

    async def mock_non_streaming(_req, _url, _headers, body, _model_name, _timeout, **kwargs):
        nonlocal captured_body
        captured_body = body
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with mock_patch("proxy.proxy_remote._handle_remote_non_streaming", mock_non_streaming):
        result = await proxy_to_remote(request, "v1/chat/completions", provider_cfg)

    assert result.status_code == 200
    assert captured_body is not None, "_handle_remote_non_streaming should have been called"
    captured_json = json.loads(captured_body.decode("utf-8"))
    assert captured_json["model"] == "deepseek-v4-flash", (
        f"Expected model override to 'deepseek-v4-flash', got '{captured_json.get('model')}'"
    )


@pytest.mark.asyncio
async def test_proxy_to_remote_with_opencode_go_deepseek_replaces_incoming_authorization_header():
    """Incoming client Authorization header must be replaced (not duplicated)
    with the upstream OPENCODE API key header."""
    import proxy.server as server_module
    from proxy.proxy_remote import proxy_to_remote
    from unittest.mock import patch as mock_patch

    server_module.config = {
        "server": {"llama_request_timeout": 300},
    }
    server_module.current_model = None

    request = _DummyRequest(body=b'{"model":"plan","messages":[{"role":"user","content":"hi"}],"stream":false}')
    # Simulate client auth header to proxy (lower-case variant).
    request.headers = {"authorization": "Bearer LOCAL_PROXY_TOKEN"}

    provider_cfg = {
        "name": "opencode-go-deepseek",
        "type": "remote",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": "OPENCODE_API_KEY",
        "model": "deepseek-v4-flash",
    }

    captured_headers = None

    async def mock_non_streaming(_req, _url, headers, _body, _model_name, _timeout, **kwargs):
        nonlocal captured_headers
        captured_headers = headers
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    test_key = "sk-test-go-tier-key-12345"
    with mock_patch.dict("os.environ", {"OPENCODE_API_KEY": test_key}, clear=False):
        with mock_patch("proxy.proxy_remote._handle_remote_non_streaming", mock_non_streaming):
            result = await proxy_to_remote(request, "v1/chat/completions", provider_cfg)

    assert result.status_code == 200
    assert captured_headers is not None, "_handle_remote_non_streaming should have been called"
    # Must not preserve incoming lowercase auth header.
    assert "authorization" not in captured_headers
    auth_header = captured_headers.get("Authorization")
    assert auth_header == f"Bearer {test_key}", (
        f"Expected Authorization: Bearer {test_key}, got '{auth_header}'"
    )


@pytest.mark.asyncio
async def test_proxy_to_remote_with_opencode_go_deepseek_injects_auth_header():
    """proxy_to_remote with the opencode-go-deepseek provider config must
    inject the Authorization header using the API key from the env var."""
    import proxy.server as server_module
    from proxy.proxy_remote import proxy_to_remote
    from unittest.mock import patch as mock_patch

    server_module.config = {
        "server": {"llama_request_timeout": 300},
    }
    server_module.current_model = None

    request = _DummyRequest(body=b'{"model":"plan","messages":[{"role":"user","content":"hi"}],"stream":false}')
    request.headers = {}

    provider_cfg = {
        "name": "opencode-go-deepseek",
        "type": "remote",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": "OPENCODE_API_KEY",
        "model": "deepseek-v4-flash",
    }

    captured_headers = None

    async def mock_non_streaming(_req, _url, headers, _body, _model_name, _timeout, **kwargs):
        nonlocal captured_headers
        captured_headers = headers
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    test_key = "sk-test-go-tier-key-12345"
    with mock_patch.dict("os.environ", {"OPENCODE_API_KEY": test_key}, clear=False):
        with mock_patch("proxy.proxy_remote._handle_remote_non_streaming", mock_non_streaming):
            result = await proxy_to_remote(request, "v1/chat/completions", provider_cfg)

    assert result.status_code == 200
    assert captured_headers is not None, "_handle_remote_non_streaming should have been called"
    auth_header = captured_headers.get("Authorization")
    assert auth_header is not None, "Authorization header must be present"
    assert auth_header == f"Bearer {test_key}", (
        f"Expected Authorization: Bearer {test_key}, got '{auth_header}'"
    )


@pytest.mark.asyncio
async def test_proxy_to_remote_falls_back_to_auth_json_when_env_var_not_set():
    """When api_key_env is set but the env var is NOT set, proxy_to_remote
    must fall back to resolving the key from ~/.pi/agent/auth.json."""
    import proxy.server as server_module
    from proxy.proxy_remote import proxy_to_remote, _try_pi_auth_json
    from unittest.mock import patch as mock_patch

    server_module.config = {
        "server": {"llama_request_timeout": 300},
    }
    server_module.current_model = None

    request = _DummyRequest(body=b'{"model":"plan","messages":[{"role":"user","content":"hi"}],"stream":false}')
    request.headers = {}

    provider_cfg = {
        "name": "opencode-go-deepseek",
        "type": "remote",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": "OPENCODE_API_KEY",
        "model": "deepseek-v4-flash",
    }

    captured_headers = None

    async def mock_non_streaming(_req, _url, headers, _body, _model_name, _timeout, **kwargs):
        nonlocal captured_headers
        captured_headers = headers
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    # Patch _try_pi_auth_json to return a test key when called with OPENCODE_API_KEY
    with mock_patch("proxy.proxy_remote._try_pi_auth_json", return_value="sk-auth-json-fallback-key"):
        with mock_patch("proxy.proxy_remote._handle_remote_non_streaming", mock_non_streaming):
            result = await proxy_to_remote(request, "v1/chat/completions", provider_cfg)

    assert result.status_code == 200
    assert captured_headers is not None, "_handle_remote_non_streaming should have been called"
    auth_header = captured_headers.get("Authorization")
    assert auth_header is not None, "Authorization header must be present when auth.json fallback resolves"
    assert "sk-auth-json-fallback-key" in auth_header, (
        f"Expected auth.json fallback key in Authorization header, got '{auth_header}'"
    )


# ===================================================================
# _try_pi_auth_json function tests
# ===================================================================


def test_try_pi_auth_json_resolves_opencode_api_key(tmp_path):
    """_try_pi_auth_json must resolve OPENCODE_API_KEY to the opencode-go
    key from auth.json."""
    from unittest.mock import patch as mock_patch
    from proxy.proxy_remote import _try_pi_auth_json, _get_auth_json_path

    auth_data = {
        "opencode-go": {
            "type": "api_key",
            "key": "sk-opencode-go-key",
        },
        "opencode": {
            "type": "api_key",
            "key": "sk-opencode-key",
        },
    }
    auth_file = tmp_path / ".pi" / "agent" / "auth.json"
    auth_file.parent.mkdir(parents=True)
    auth_file.write_text(json.dumps(auth_data), encoding="utf-8")

    with mock_patch("proxy.proxy_remote._get_auth_json_path", return_value=auth_file):
        result = _try_pi_auth_json("OPENCODE_API_KEY")

    assert result == "sk-opencode-go-key", (
        f"Expected opencode-go key, got '{result}'"
    )


def test_try_pi_auth_json_falls_back_to_opencode_when_opencode_go_missing(tmp_path):
    """When opencode-go is not in auth.json, _try_pi_auth_json must fall
    back to opencode entry for OPENCODE_API_KEY."""
    from unittest.mock import patch as mock_patch
    from proxy.proxy_remote import _try_pi_auth_json

    auth_data = {
        "opencode": {
            "type": "api_key",
            "key": "sk-opencode-only-key",
        },
    }
    auth_file = tmp_path / ".pi" / "agent" / "auth.json"
    auth_file.parent.mkdir(parents=True)
    auth_file.write_text(json.dumps(auth_data), encoding="utf-8")

    with mock_patch("proxy.proxy_remote._get_auth_json_path", return_value=auth_file):
        result = _try_pi_auth_json("OPENCODE_API_KEY")

    assert result == "sk-opencode-only-key", (
        f"Expected opencode fallback key, got '{result}'"
    )


def test_try_pi_auth_json_exact_match(tmp_path):
    """_try_pi_auth_json must return the key for an exact lowercase match
    when the auth.json key matches the lookup key directly."""
    from unittest.mock import patch as mock_patch
    from proxy.proxy_remote import _try_pi_auth_json

    auth_data = {
        "openrouter": {
            "type": "api_key",
            "key": "sk-or-test-key",
        },
    }
    auth_file = tmp_path / ".pi" / "agent" / "auth.json"
    auth_file.parent.mkdir(parents=True)
    auth_file.write_text(json.dumps(auth_data), encoding="utf-8")

    with mock_patch("proxy.proxy_remote._get_auth_json_path", return_value=auth_file):
        result = _try_pi_auth_json("OPENROUTER")

    assert result == "sk-or-test-key", (
        f"Expected exact-match key, got '{result}'"
    )


def test_try_pi_auth_json_returns_none_when_file_missing():
    """_try_pi_auth_json must return None when auth.json does not exist."""
    from pathlib import Path
    from unittest.mock import patch as mock_patch
    from proxy.proxy_remote import _try_pi_auth_json

    with mock_patch("proxy.proxy_remote._get_auth_json_path", return_value=Path("/nonexistent/auth.json")):
        result = _try_pi_auth_json("OPENCODE_API_KEY")

    assert result is None, f"Expected None for missing file, got '{result}'"


def test_try_pi_auth_json_returns_none_for_unknown_key(tmp_path):
    """_try_pi_auth_json must return None when the key is not in auth.json
    and no fallback matches."""
    from unittest.mock import patch as mock_patch
    from proxy.proxy_remote import _try_pi_auth_json

    auth_data = {
        "opencode": {
            "type": "api_key",
            "key": "sk-opencode",
        },
    }
    auth_file = tmp_path / ".pi" / "agent" / "auth.json"
    auth_file.parent.mkdir(parents=True)
    auth_file.write_text(json.dumps(auth_data), encoding="utf-8")

    with mock_patch("proxy.proxy_remote._get_auth_json_path", return_value=auth_file):
        result = _try_pi_auth_json("NONEXISTENT_KEY")

    assert result is None, f"Expected None for unknown key, got '{result}'"


def test_try_pi_auth_json_strips_api_key_suffix(tmp_path):
    """_try_pi_auth_json must strip _api_key suffix and look up the stem."""
    from unittest.mock import patch as mock_patch
    from proxy.proxy_remote import _try_pi_auth_json

    auth_data = {
        "openrouter": {
            "type": "api_key",
            "key": "sk-or-test-key",
        },
    }
    auth_file = tmp_path / ".pi" / "agent" / "auth.json"
    auth_file.parent.mkdir(parents=True)
    auth_file.write_text(json.dumps(auth_data), encoding="utf-8")

    with mock_patch("proxy.proxy_remote._get_auth_json_path", return_value=auth_file):
        result = _try_pi_auth_json("OPENROUTER_API_KEY")

    assert result == "sk-or-test-key", (
        f"Expected key from _api_key stripping, got '{result}'"
    )


# ===================================================================
# Go tier fallback chain end-to-end test
# ===================================================================


@pytest.mark.asyncio
async def test_plan_fallback_reaches_go_tier_when_free_tier_rate_limited():
    """When the local provider is exhausted and the free remote tier returns
    429, the fallback chain must reach the Go tier (opencode-go-deepseek)
    and return a successful response."""
    request = _DummyRequest(body=b'{"model":"plan","messages":[{"role":"user","content":"hi"}],"stream":false}')

    plan_model_config = {
        "providers": [
            {
                "name": "local-qwen3",
                "type": "local",
                "llama_model": "Qwen3",
            },
            {
                "name": "opencode-deepseek-free",
                "type": "remote",
                "endpoint": "https://opencode.ai/zen",
                "api_key_env": "OPENCODE_API_KEY",
                "model": "deepseek-v4-flash-free",
            },
            {
                "name": "opencode-go-deepseek",
                "type": "remote",
                "endpoint": "https://opencode.ai/zen/go",
                "api_key_env": "OPENCODE_API_KEY",
                "model": "deepseek-v4-flash",
            },
        ],
        "aliases": ["plan*"],
    }

    cfg = {"provider_cooldown_seconds": 60}
    call_log = []

    async def _mock_proxy_to_local(_req, _path):
        call_log.append("local-qwen3")
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "server_busy",
                    "code": "no_slots_available",
                    "message": "Model server busy: 0/1 slots available.",
                },
                "status": 503,
                "retry_after": 5,
                "total_slots": 1,
                "available_slots": 0,
            },
        )

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        name = provider_cfg.get("name")
        call_log.append(name)
        if name == "opencode-deepseek-free":
            # Free tier rate-limited
            return Response(status_code=429, content=b"Rate limited")
        # Go tier succeeds
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok-from-go-tier"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with (
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", plan_model_config, cfg
        )

    assert result.status_code == 200, f"Expected 200 from Go tier, got {result.status_code}"
    body = json.loads(result.body) if hasattr(result, "body") else {}
    assert "ok-from-go-tier" in str(body), (
        f"Expected Go tier response content, got: {body}"
    )
    assert call_log == [
        "local-qwen3",
        "opencode-deepseek-free",
        "opencode-go-deepseek",
    ], f"Expected fallback chain through all providers, got: {call_log}"
    assert result.headers.get("X-Provider") == "opencode-go-deepseek", (
        f"Expected X-Provider=opencode-go-deepseek, got '{result.headers.get('X-Provider')}'"
    )


@pytest.mark.asyncio
async def test_plan_fallback_all_exhausted_with_go_tier_error():
    """When ALL providers (including Go tier) fail, the fallback must
    return a 503 with appropriate error diagnostics."""
    request = _DummyRequest(body=b'{"model":"plan","messages":[{"role":"user","content":"hi"}],"stream":false}')

    plan_model_config = {
        "providers": [
            {
                "name": "local-qwen3",
                "type": "local",
                "llama_model": "Qwen3",
            },
            {
                "name": "opencode-deepseek-free",
                "type": "remote",
                "endpoint": "https://opencode.ai/zen",
                "api_key_env": "OPENCODE_API_KEY",
                "model": "deepseek-v4-flash-free",
            },
            {
                "name": "opencode-go-deepseek",
                "type": "remote",
                "endpoint": "https://opencode.ai/zen/go",
                "api_key_env": "OPENCODE_API_KEY",
                "model": "deepseek-v4-flash",
            },
        ],
        "aliases": ["plan*"],
    }

    cfg = {"provider_cooldown_seconds": 60}
    call_log = []

    async def _mock_proxy_to_local(_req, _path):
        call_log.append("local-qwen3")
        raise httpx.ConnectError("Connection refused to llama-server")

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        name = provider_cfg.get("name")
        call_log.append(name)
        if name == "opencode-go-deepseek":
            # Go tier also fails
            return Response(status_code=502, content=b"Bad gateway from Go")
        # Free tier fails
        return Response(status_code=502, content=b"Bad gateway from Free")

    with (
        patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote),
        patch("proxy.router.proxy_to_local", _mock_proxy_to_local),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", plan_model_config, cfg
        )

    # Should return the first error response (502 from free tier) since the
    # fallback logic preserves the actual upstream error instead of returning
    # a generic "All providers exhausted" 503.
    assert result.status_code == 502, (
        f"Expected 502 (first upstream error), got {result.status_code}"
    )
    assert b"Bad gateway from Free" in result.body, (
        f"Expected free-tier error body, got: {result.body}"
    )
    assert call_log == [
        "local-qwen3",
        "opencode-deepseek-free",
        "opencode-go-deepseek",
    ], f"Expected all providers tried, got: {call_log}"


# ===================================================================
# Parity tests for shared fallback primitives
# ===================================================================


class TestSharedFallbackPrimitives:
    """Parity tests for the extracted shared fallback primitive functions.

    These functions were extracted from ``proxy_with_remote_fallback()`` and
    ``proxy_with_fallback()`` to eliminate duplicated state-machine logic.
    The tests verify each primitive behaves correctly in isolation.
    """

    def test_record_attempt_stores_all_fields(self):
        """_record_attempt appends a correctly structured dict."""
        attempts = []
        provider._record_attempt(
            attempts,
            provider="test-provider",
            type="remote",
            status="test_status",
            status_code=200,
            body_snippet="ok",
        )
        assert len(attempts) == 1
        entry = attempts[0]
        assert entry["provider"] == "test-provider"
        assert entry["type"] == "remote"
        assert entry["status"] == "test_status"
        assert entry["status_code"] == 200
        assert entry["body_snippet"] == "ok"

    def test_record_attempt_multiple_entries(self):
        """_record_attempt appends entries in order."""
        attempts = []
        provider._record_attempt(attempts, provider="p1", type="local", status="first")
        provider._record_attempt(attempts, provider="p2", type="remote", status="second")
        assert len(attempts) == 2
        assert attempts[0]["provider"] == "p1"
        assert attempts[1]["provider"] == "p2"

    @pytest.mark.asyncio
    async def test_handle_streaming_success_returns_response(self):
        """_handle_streaming_success returns augmented response for streaming 2xx."""
        from fastapi.responses import StreamingResponse
        async def dummy_stream():
            yield b"test"
        response = StreamingResponse(dummy_stream(), status_code=200)
        attempts = []
        result = provider._handle_streaming_success(
            response, "test-provider", "remote", attempts,
            prev_provider=None, fallback_reason=None, path="v1/test",
        )
        assert result is not None
        assert result.headers.get("X-Provider") == "test-provider"
        assert len(attempts) == 1
        assert attempts[0]["status"] == "streaming_success"

    @pytest.mark.asyncio
    async def test_handle_streaming_success_returns_none_for_non_streaming(self):
        """_handle_streaming_success returns None for non-streaming response."""
        response = Response(content=b"plain", status_code=200)
        attempts = []
        result = provider._handle_streaming_success(
            response, "test-provider", "remote", attempts,
            prev_provider=None, fallback_reason=None, path="v1/test",
        )
        assert result is None
        assert len(attempts) == 0

    def test_handle_connection_error_returns_true_and_records(self):
        """_handle_connection_error_in_fallback returns True and records entry."""
        attempts = []
        exc = httpx.ConnectError("Connection refused")
        result = provider._handle_connection_error_in_fallback(
            exc, "test-provider", "remote", 60.0, attempts,
        )
        assert result is True
        assert len(attempts) == 1
        assert attempts[0]["status"] == "connection_error"
        assert provider._is_provider_unavailable("test-provider")

    def test_handle_connection_error_returns_false_for_other_exceptions(self):
        """_handle_connection_error_in_fallback returns False for non-connection errors."""
        attempts = []
        exc = ValueError("Not a connection error")
        result = provider._handle_connection_error_in_fallback(
            exc, "test-provider", "remote", 60.0, attempts,
        )
        assert result is False
        assert len(attempts) == 0

    def test_handle_http_error_with_cooldown_marks_unavailable(self):
        """_handle_http_error_with_cooldown marks provider and records attempt."""
        provider._provider_unavailable_until.clear()
        provider._provider_failure_count.clear()
        attempts = []
        response = Response(status_code=502, content=b"Bad gateway")
        cooldown = provider._handle_http_error_with_cooldown(
            response, "test-provider", "remote", 60.0, attempts, "bad gateway",
        )
        # Remote providers use exponential backoff starting at 1s
        assert cooldown == 1.0
        assert provider._is_provider_unavailable("test-provider")
        assert len(attempts) == 1
        assert attempts[0]["status"] == "http_error"

    def test_handle_empty_response_with_cooldown_marks_unavailable(self):
        """_handle_empty_response_with_cooldown marks provider and records attempt."""
        provider._provider_unavailable_until.clear()
        provider._provider_failure_count.clear()
        attempts = []
        response = Response(status_code=200, content=b"{}")
        cooldown = provider._handle_empty_response_with_cooldown(
            response, "test-provider", "remote", 60.0, attempts, "{}",
        )
        # Remote providers use exponential backoff starting at 1s
        assert cooldown == 1.0
        assert provider._is_provider_unavailable("test-provider")
        assert len(attempts) == 1
        assert attempts[0]["status"] == "empty_response"

    # ------------------------------------------------------------------
    # Comprehensive exponential backoff tests (AC1-AC5)
    # ------------------------------------------------------------------

    def test_connection_error_backoff_exponential_sequence(self):
        """AC1: Connection error backoff doubles each consecutive failure (1s, 2s, 4s...)."""
        provider._provider_failure_count.clear()
        provider._provider_unavailable_until.clear()
        attempts = []
        exc = httpx.ConnectError("Connection refused")
        with patch('time.time', return_value=1000.0):
            for i, expected in enumerate([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]):
                provider._handle_connection_error_in_fallback(
                    exc, "seq-provider", "remote", 60.0, attempts,
                )
                expiry = provider._provider_unavailable_until["seq-provider"]
                assert expiry == 1000.0 + expected, (
                    f"Failure {i+1}: expected {1000.0+expected}s, got {expiry}s"
                )
                assert provider._provider_failure_count["seq-provider"] == i + 1

    def test_connection_error_backoff_capped_at_45s(self):
        """AC1: Connection error backoff is capped at BACKOFF_MAX (45s)."""
        provider._provider_failure_count.clear()
        provider._provider_unavailable_until.clear()
        attempts = []
        exc = httpx.ConnectError("Connection refused")
        with patch('time.time', return_value=1000.0):
            # 6 failures: 1, 2, 4, 8, 16, 32, then 7th = min(64, 45) = 45
            for _ in range(7):
                provider._handle_connection_error_in_fallback(
                    exc, "cap45-provider", "remote", 60.0, attempts,
                )
            expiry = provider._provider_unavailable_until["cap45-provider"]
            # 7th failure: backoff = min(64, 45) = 45, capped by cooldown_seconds=60 -> 45
            # Start at 1000, after 6 increments the remaining cooldown is for the 7th:
            # The 7th call uses count=6: backoff = min(1*64, 45) = 45
            assert expiry == 1045.0, f"Expected cap at 45s, got expiry={expiry}"

    def test_connection_error_backoff_capped_by_cooldown_seconds(self):
        """AC1: Connection error backoff is capped by cooldown_seconds."""
        provider._provider_failure_count.clear()
        provider._provider_unavailable_until.clear()
        attempts = []
        exc = httpx.ConnectError("Connection refused")
        with patch('time.time', return_value=1000.0):
            # 3rd failure: 1*2^2 = 4s, but cooldown_seconds=3 should cap at 3
            for _ in range(3):
                provider._handle_connection_error_in_fallback(
                    exc, "cap-cd-provider", "remote", 3.0, attempts,
                )
            expiry = provider._provider_unavailable_until["cap-cd-provider"]
            # 3rd: count=2, backoff=4, min(backoff, cooldown_seconds)=min(4,3)=3
            assert expiry == 1003.0, f"Expected cap at cooldown_seconds=3, got expiry={expiry}"

    def test_connection_error_local_provider_no_backoff(self):
        """AC5: Local providers do NOT get exponential backoff for connection errors."""
        provider._provider_failure_count.clear()
        provider._provider_unavailable_until.clear()
        attempts = []
        exc = httpx.ConnectError("Connection refused")
        with patch('time.time', return_value=1000.0):
            # Multiple calls with local provider should always use cooldown_seconds
            for i in range(3):
                provider._handle_connection_error_in_fallback(
                    exc, "local-provider", "local", 60.0, attempts,
                )
                expiry = provider._provider_unavailable_until["local-provider"]
                assert expiry == 1000.0 + 60.0, (
                    f"Call {i+1}: local provider should get 60s, got expiry={expiry}"
                )

    def test_http_error_backoff_exponential_sequence(self):
        """AC2: HTTP error backoff doubles each consecutive failure (1s, 2s, 4s...)."""
        provider._provider_failure_count.clear()
        provider._provider_unavailable_until.clear()
        response = Response(status_code=502, content=b"Bad gateway")
        with patch('time.time', return_value=1000.0):
            for i, expected in enumerate([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]):
                attempts = []
                cooldown = provider._handle_http_error_with_cooldown(
                    response, "http-seq-provider", "remote", 60.0, attempts, "bad gateway",
                )
                assert cooldown == expected, (
                    f"Call {i+1}: expected cooldown={expected}, got {cooldown}"
                )

    def test_http_error_backoff_capped_at_45s(self):
        """AC2: HTTP error backoff is capped at BACKOFF_MAX (45s)."""
        provider._provider_failure_count.clear()
        provider._provider_unavailable_until.clear()
        response = Response(status_code=502, content=b"Bad gateway")
        with patch('time.time', return_value=1000.0):
            for _ in range(6):
                provider._handle_http_error_with_cooldown(
                    response, "http-cap45-provider", "remote", 60.0, attempts=[], body_text="bad",
                )
            cooldown = provider._handle_http_error_with_cooldown(
                response, "http-cap45-provider", "remote", 60.0, attempts=[], body_text="bad",
            )
            # 7th: count=6 -> backoff = min(64, 45) = 45
            assert cooldown == 45.0, f"Expected cap at 45s, got {cooldown}"

    def test_http_error_backoff_capped_by_cooldown_seconds(self):
        """AC2: HTTP error backoff is capped by cooldown_seconds."""
        provider._provider_failure_count.clear()
        provider._provider_unavailable_until.clear()
        response = Response(status_code=502, content=b"Bad gateway")
        with patch('time.time', return_value=1000.0):
            for _ in range(2):
                provider._handle_http_error_with_cooldown(
                    response, "http-cap-cd-provider", "remote", 3.0, attempts=[], body_text="bad",
                )
            cooldown = provider._handle_http_error_with_cooldown(
                response, "http-cap-cd-provider", "remote", 3.0, attempts=[], body_text="bad",
            )
            # 3rd: count=2 -> backoff=min(4,45)=4, capped by cooldown_seconds=3 -> 3
            assert cooldown == 3.0, f"Expected cap at cooldown_seconds=3, got {cooldown}"

    def test_http_error_local_provider_no_backoff(self):
        """AC5: Local providers do NOT get exponential backoff for HTTP errors."""
        provider._provider_failure_count.clear()
        provider._provider_unavailable_until.clear()
        response = Response(status_code=502, content=b"Bad gateway")
        with patch('time.time', return_value=1000.0):
            for i in range(3):
                attempts = []
                cooldown = provider._handle_http_error_with_cooldown(
                    response, "http-local-provider", "local", 60.0, attempts, "bad gateway",
                )
                assert cooldown == 60.0, (
                    f"Call {i+1}: local provider should get 60s, got {cooldown}"
                )

    def test_empty_response_backoff_exponential_sequence(self):
        """AC2: Empty response backoff doubles each consecutive failure (1s, 2s, 4s...)."""
        provider._provider_failure_count.clear()
        provider._provider_unavailable_until.clear()
        response = Response(status_code=200, content=b"{}")
        with patch('time.time', return_value=1000.0):
            for i, expected in enumerate([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]):
                attempts = []
                cooldown = provider._handle_empty_response_with_cooldown(
                    response, "empty-seq-provider", "remote", 60.0, attempts, "{}",
                )
                assert cooldown == expected, (
                    f"Call {i+1}: expected cooldown={expected}, got {cooldown}"
                )

    def test_empty_response_backoff_capped_at_45s(self):
        """AC2: Empty response backoff is capped at BACKOFF_MAX (45s)."""
        provider._provider_failure_count.clear()
        provider._provider_unavailable_until.clear()
        response = Response(status_code=200, content=b"{}")
        with patch('time.time', return_value=1000.0):
            for _ in range(6):
                provider._handle_empty_response_with_cooldown(
                    response, "empty-cap45-provider", "remote", 60.0, attempts=[], body_text="{}",
                )
            cooldown = provider._handle_empty_response_with_cooldown(
                response, "empty-cap45-provider", "remote", 60.0, attempts=[], body_text="{}",
            )
            # 7th: count=6 -> backoff = min(64, 45) = 45
            assert cooldown == 45.0, f"Expected cap at 45s, got {cooldown}"

    def test_empty_response_local_provider_no_backoff(self):
        """AC5: Local providers do NOT get exponential backoff for empty responses."""
        provider._provider_failure_count.clear()
        provider._provider_unavailable_until.clear()
        response = Response(status_code=200, content=b"{}")
        with patch('time.time', return_value=1000.0):
            for i in range(3):
                attempts = []
                cooldown = provider._handle_empty_response_with_cooldown(
                    response, "empty-local-provider", "local", 60.0, attempts, "{}",
                )
                assert cooldown == 60.0, (
                    f"Call {i+1}: local provider should get 60s, got {cooldown}"
                )

    def test_success_resets_failure_count_http_error(self):
        """AC3: Success resets failure count, next failure starts at base (1s)."""
        provider._provider_failure_count.clear()
        provider._provider_unavailable_until.clear()
        response = Response(status_code=502, content=b"Bad gateway")
        with patch('time.time', return_value=1000.0):
            # Two failures to push count to 2
            provider._handle_http_error_with_cooldown(
                response, "reset-provider", "remote", 60.0, attempts=[], body_text="bad",
            )
            provider._handle_http_error_with_cooldown(
                response, "reset-provider", "remote", 60.0, attempts=[], body_text="bad",
            )
            assert provider._provider_failure_count["reset-provider"] == 2

            # Reset the failure count
            provider._reset_provider_failure_count("reset-provider")
            assert "reset-provider" not in provider._provider_failure_count

            # Next failure should start at 1s again
            cooldown = provider._handle_http_error_with_cooldown(
                response, "reset-provider", "remote", 60.0, attempts=[], body_text="bad",
            )
            assert cooldown == 1.0, f"After reset, expected 1s, got {cooldown}"
            assert provider._provider_failure_count["reset-provider"] == 1

    def test_success_resets_failure_count_empty_response(self):
        """AC3: Success resets failure count for empty-response path."""
        provider._provider_failure_count.clear()
        provider._provider_unavailable_until.clear()
        response = Response(status_code=200, content=b"{}")
        with patch('time.time', return_value=1000.0):
            # Two failures
            provider._handle_empty_response_with_cooldown(
                response, "reset-empty-provider", "remote", 60.0, attempts=[], body_text="{}",
            )
            provider._handle_empty_response_with_cooldown(
                response, "reset-empty-provider", "remote", 60.0, attempts=[], body_text="{}",
            )
            assert provider._provider_failure_count["reset-empty-provider"] == 2

            provider._reset_provider_failure_count("reset-empty-provider")

            cooldown = provider._handle_empty_response_with_cooldown(
                response, "reset-empty-provider", "remote", 60.0, attempts=[], body_text="{}",
            )
            assert cooldown == 1.0, f"After reset, expected 1s, got {cooldown}"

    def test_success_resets_failure_count_connection_error(self):
        """AC3: Success resets failure count for connection-error path."""
        provider._provider_failure_count.clear()
        provider._provider_unavailable_until.clear()
        exc = httpx.ConnectError("Connection refused")
        with patch('time.time', return_value=1000.0):
            for _ in range(2):
                provider._handle_connection_error_in_fallback(
                    exc, "reset-conn-provider", "remote", 60.0, attempts=[],
                )
            assert provider._provider_failure_count["reset-conn-provider"] == 2

            provider._reset_provider_failure_count("reset-conn-provider")
            assert "reset-conn-provider" not in provider._provider_failure_count

            # Next failure should start at 1s
            provider._handle_connection_error_in_fallback(
                exc, "reset-conn-provider", "remote", 60.0, attempts=[],
            )
            expiry = provider._provider_unavailable_until["reset-conn-provider"]
            assert expiry == 1001.0, f"After reset, expected expiry=1001, got {expiry}"
            assert provider._provider_failure_count["reset-conn-provider"] == 1

    def test_retry_after_respected_alongside_backoff(self):
        """AC4: Retry-After header is respected alongside backoff (whichever is longer)."""
        provider._provider_failure_count.clear()
        provider._provider_unavailable_until.clear()
        with patch('time.time', return_value=1000.0):
            # Backoff would give 1s, but Retry-After=30 should take precedence
            response = Response(status_code=429, content=b"Rate limited", headers={"Retry-After": "30"})
            cooldown = provider._handle_http_error_with_cooldown(
                response, "retry-provider", "remote", 60.0, attempts=[], body_text="rate limited",
            )
            assert cooldown == 30.0, f"Expected max(1, 30)=30, got {cooldown}"

            # With longer backoff (later in sequence), backoff may exceed Retry-After
            provider._handle_http_error_with_cooldown(
                response, "retry-provider", "remote", 60.0, attempts=[], body_text="rate limited",
            )
            # 3rd call: count=2 -> backoff=4, max(4, 30) = 30
            cooldown = provider._handle_http_error_with_cooldown(
                response, "retry-provider", "remote", 60.0, attempts=[], body_text="rate limited",
            )
            assert cooldown == 30.0, f"Expected max(4, 30)=30, got {cooldown}"

            # When backoff exceeds Retry-After, backoff takes precedence
            # Fast-forward: at count=5 (6th call), backoff=32 > retry-after=30
            # Use a separate provider that already has accumulated failures
            provider._provider_failure_count["retry-long-provider"] = 5
            cooldown = provider._handle_http_error_with_cooldown(
                response, "retry-long-provider", "remote", 60.0, attempts=[], body_text="rate limited",
            )
            # backoff = min(1*2^5, 45) = 32, max(32, 30) = 32
            assert cooldown == 32.0, f"Expected max(32, 30)=32, got {cooldown}"

    def test_retry_after_respected_with_empty_response(self):
        """AC4: Retry-After header is respected alongside backoff for empty responses."""
        provider._provider_failure_count.clear()
        provider._provider_unavailable_until.clear()
        with patch('time.time', return_value=1000.0):
            response = Response(status_code=200, content=b"{}", headers={"Retry-After": "15"})
            cooldown = provider._handle_empty_response_with_cooldown(
                response, "retry-empty-provider", "remote", 60.0, attempts=[], body_text="{}",
            )
            # backoff=1, max(1, 15) = 15
            assert cooldown == 15.0, f"Expected max(1, 15)=15, got {cooldown}"

    def test_resolve_reasoning_content_promotion_matches(self):
        """_resolve_reasoning_content_promotion returns response when reasoning_content present."""
        attempts = []
        response = Response(
            content=b'{"choices":[{"message":{"reasoning_content":"thinking..."}}]}',
            status_code=200,
        )
        result = provider._resolve_reasoning_content_promotion(
            response, "test-provider", "remote", attempts,
            prev_provider=None, fallback_reason=None,
            path="v1/test", body_text='{"choices":[{"message":{"reasoning_content":"thinking..."}}]}',
        )
        assert result is not None
        assert result.headers.get("X-Provider") == "test-provider"
        assert len(attempts) == 1
        assert attempts[0]["status"] == "promoted_reasoning"

    def test_resolve_reasoning_content_promotion_no_match(self):
        """_resolve_reasoning_content_promotion returns None without reasoning_content."""
        attempts = []
        response = Response(content=b'{"choices":[{"message":{"content":"hello"}}]}', status_code=200)
        result = provider._resolve_reasoning_content_promotion(
            response, "test-provider", "remote", attempts,
            prev_provider=None, fallback_reason=None,
            path="v1/test", body_text='{"choices":[{"message":{"content":"hello"}}]}',
        )
        assert result is None
        assert len(attempts) == 0

    def test_build_fallback_success_response_adds_header_and_records(self):
        """_build_fallback_success_response adds header and records attempt."""
        attempts = []
        response = Response(content=b'{"choices":[{"message":{"content":"ok"}}]}', status_code=200)
        result = provider._build_fallback_success_response(
            response, "test-provider", "remote", attempts,
            prev_provider=None, fallback_reason=None,
            path="v1/test", body_text='{"choices":[{"message":{"content":"ok"}}]}',
        )
        assert result is not None
        assert result.headers.get("X-Provider") == "test-provider"
        assert len(attempts) == 1
        assert attempts[0]["status"] == "success"

    def test_log_exhausted_providers_returns_empty_for_none(self):
        """_log_exhausted_providers returns empty dict when no providers in cooldown."""
        provider._provider_unavailable_until.clear()
        model_config = {"providers": [{"name": "p1", "type": "remote"}]}
        result = provider._log_exhausted_providers(model_config, "v1/test")
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_log_exhausted_providers_returns_cooldown_info(self):
        """_log_exhausted_providers returns remaining cooldown for providers."""
        provider._provider_unavailable_until.clear()
        provider.mark_provider_unavailable("p1", 60.0)
        model_config = {"providers": [{"name": "p1", "type": "remote"}, {"name": "p2", "type": "remote"}]}
        result = provider._log_exhausted_providers(model_config, "v1/test")
        assert "p1" in result
        assert result["p1"] > 0
        assert "p2" not in result

    def test_build_fallback_success_response_with_status_override(self):
        """_build_fallback_success_response uses status_override when provided."""
        attempts = []
        response = Response(content=b'{}', status_code=200)
        result = provider._build_fallback_success_response(
            response, "test-provider", "remote", attempts,
            prev_provider=None, fallback_reason=None,
            path="v1/test", body_text="{}",
            status_override="success_after_http_exception_retry",
        )
        assert result is not None
        assert len(attempts) == 1
        assert attempts[0]["status"] == "success_after_http_exception_retry"


# ===================================================================
# Cross-session cooldown persistence tests
# ===================================================================


@pytest.mark.asyncio
async def test_cross_session_cooldown_first_session_failure_skipped_by_second(sample_model_config):
    """Cooldown set during one session is respected by a subsequent session.

    Simulates two independent client sessions:
    - Session 1: First provider fails (ReadTimeout), is marked unavailable.
                Second provider succeeds.
    - Session 2: First provider is still in cooldown from session 1, so
                 it is skipped and the second provider is used directly.

    This verifies that cooldown state persists across session boundaries
    (the core requirement of LP-0MRB94JOE0075JNY).
    """
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    # Session 1: first provider fails, second succeeds
    session1_call_count = 0

    async def _session1_mock(_req, _path, provider_cfg):
        nonlocal session1_call_count
        session1_call_count += 1
        if session1_call_count == 1:
            # First provider fails with ReadTimeout
            raise httpx.ReadTimeout("Read timeout", request=None)
        # Second provider succeeds
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _session1_mock):
        result1 = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert result1.status_code == 200
    assert result1.headers.get("X-Provider") == "remote-fallback"
    assert session1_call_count == 2

    # Verify first provider is now in cooldown
    assert provider._is_provider_unavailable("remote-primary")

    # Session 2: first provider is in cooldown, should be skipped
    session2_call_count = 0

    async def _session2_mock(_req, _path, provider_cfg):
        nonlocal session2_call_count
        session2_call_count += 1
        # Should only be called for the second provider since first is in cooldown
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok-session2"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _session2_mock):
        result2 = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert result2.status_code == 200
    # The second session should have skipped the cooldowned provider
    # and gone directly to the second provider
    assert result2.headers.get("X-Provider") == "remote-fallback"
    assert session2_call_count == 1, (
        f"Expected only 1 call (second provider), got {session2_call_count}. "
        "First provider was in cooldown but was not skipped."
    )


@pytest.mark.asyncio
async def test_cross_session_cooldown_all_providers_down(sample_model_config):
    """When all providers are in cooldown from a previous session,
    a new session should get 503 immediately without retrying any provider.

    Simulates:
    - Session 1: Both providers fail, both marked unavailable.
    - Session 2: Both providers in cooldown, returns 503.
    """
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    # Session 1: both providers fail
    async def _session1_mock(_req, _path, provider_cfg):
        raise httpx.ReadTimeout("Read timeout", request=None)

    with patch("proxy.server.proxy_to_remote", _session1_mock):
        result1 = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert result1.status_code == 503
    assert provider._is_provider_unavailable("remote-primary")
    assert provider._is_provider_unavailable("remote-fallback")

    # Session 2: both providers in cooldown
    # If the mock is called, the cooldown bypass has occurred
    session2_mock_called = False

    async def _session2_mock(_req, _path, provider_cfg):
        nonlocal session2_mock_called
        session2_mock_called = True
        return Response(status_code=502, content=b"Bad gateway")

    with patch("proxy.server.proxy_to_remote", _session2_mock):
        result2 = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", sample_model_config, cfg
        )

    assert result2.status_code == 503
    assert not session2_mock_called, (
        "Session 2 should not have called any provider - all are in cooldown. "
        "This indicates cooldown bypass."
    )


@pytest.mark.asyncio
async def test_cross_session_cooldown_does_not_affect_available_providers(mixed_model_config):
    """Cooldown of one provider does not affect other providers.

    Simulates:
    - Session 1: Remote provider fails, marked unavailable.
                 Local provider succeeds.
    - Session 2: Remote provider still in cooldown, local provider available.
                 Should use local provider directly.
    """
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    # Session 1: local succeeds, remote would have failed but isn't reached
    local_call_count = 0
    remote_call_count = 0

    async def _local_mock_s1(_req, _path):
        nonlocal local_call_count
        local_call_count += 1
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok-local"}}]}),
            status_code=200,
            media_type="application/json",
        )

    async def _remote_mock_s1(_req, _path, provider_cfg):
        # This should never be called in session 1 since local succeeds
        nonlocal remote_call_count
        remote_call_count += 1
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok-remote"}}]}),
            status_code=200,
            media_type="application/json",
        )

    # Mark remote as unavailable to simulate it was already in cooldown
    provider.mark_provider_unavailable("remote-fallback", 60.0)

    # Mark local as NOT unavailable
    assert not provider._is_provider_unavailable("local-llama")

    with (
        patch("proxy.router.proxy_to_local", _local_mock_s1),
        patch("proxy.server.proxy_to_remote", _remote_mock_s1),
        patch("proxy.provider._get_scheduler_has_idle_slot", return_value=True),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert result.headers.get("X-Provider") == "local-llama"
    assert local_call_count == 1
    # Remote should NOT have been called (it's in cooldown, and local succeeded)
    assert remote_call_count == 0, (
        "Remote provider should not have been called - it's in cooldown "
        "and local provider succeeded."
    )
