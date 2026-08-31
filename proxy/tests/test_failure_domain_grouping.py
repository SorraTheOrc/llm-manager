"""
Tests for failure-domain grouping in provider fallback (LP-0MSG45I8Q0020N1F).

Failure-domain grouping treats provider entries that share the same
normalized endpoint (e.g. ``https://opencode.ai/zen/go``) as a single
failure domain, so a stall on one entry causes the fallback chain /
mid-stream re-route to skip to the first entry of a *different* domain
instead of hopping to another API-key entry on the same broken gateway.

Covers (parent AC1-AC4 / F1 AC1-AC6):

- ``_failure_domain_key()``: normalized endpoint key derivation (lowercase
  scheme+host, strip trailing slash and fragment, drop default ports, keep
  path case and query strings), brand fallback for local/no-endpoint entries.
- Same-domain skip in ``_resolve_provider_with_exclusions`` after a stall.
- No over-grouping: entries with different endpoints are NOT skipped together.
- ``resolve_provider(failed_provider=...)`` skips entries sharing the failed
  provider's failure domain.
- Integration: mid-stream stall on ``opencode-go-2-deepseek`` re-routes past
  ``opencode-go-deepseek`` (never called) to ``deepseek-v4-flash``.
- Logging: the skip log line includes the failure-domain key and the reason
  "same failure domain as ...".
"""

import json
import logging
from unittest.mock import patch

import proxy.provider as provider
import pytest
from fastapi import Response
from fastapi.responses import StreamingResponse


class _DummyRequest:
    """Minimal request stub (mirrors test_midstream_reroute._DummyRequest)."""

    def __init__(self, body: bytes = b'{"model":"test"}'):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return False


@pytest.fixture(autouse=True)
def reset_cooldown_state():
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    yield


# ---------------------------------------------------------------------------
# Provider chain fixtures (mirrors the real config.yaml pattern)
# ---------------------------------------------------------------------------


@pytest.fixture
def opencode_same_gateway_chain():
    """Two entries on the SAME gateway (opencode.ai/zen/go) followed by a
    different gateway (api.deepseek.com). Mirrors the real ``plan`` model."""
    return {
        "providers": [
            {
                "name": "opencode-go-2-deepseek",
                "type": "remote",
                "provider": "opencode-go",
                "endpoint": "https://opencode.ai/zen/go",
                "api_key_env": "OPENCODE_2_API_KEY",
                "model": "deepseek-v4-flash",
            },
            {
                "name": "opencode-go-deepseek",
                "type": "remote",
                "provider": "opencode-go",
                "endpoint": "https://opencode.ai/zen/go",
                "api_key_env": "OPENCODE_API_KEY",
                "model": "deepseek-v4-flash",
            },
            {
                "name": "deepseek-v4-flash",
                "type": "remote",
                "provider": "deepseek",
                "endpoint": "https://api.deepseek.com",
                "api_key_env": "DEEPSEEK_API_KEY",
                "model": "deepseek-v4-flash",
            },
        ],
        "aliases": ["test*"],
    }


@pytest.fixture
def opencode_distinct_endpoints_chain():
    """Entries on *different* endpoints of the same host (opencode.ai/zen vs
    opencode.ai/zen/go) followed by a third gateway. Used to assert no
    over-grouping (parent AC2)."""
    return {
        "providers": [
            {
                "name": "opencode-deepseek",
                "type": "remote",
                "provider": "opencode",
                "endpoint": "https://opencode.ai/zen",
                "api_key_env": "OPENCODE_API_KEY",
                "model": "deepseek-v4-flash",
            },
            {
                "name": "opencode-go-2-deepseek",
                "type": "remote",
                "provider": "opencode-go",
                "endpoint": "https://opencode.ai/zen/go",
                "api_key_env": "OPENCODE_2_API_KEY",
                "model": "deepseek-v4-flash",
            },
            {
                "name": "deepseek-v4-flash",
                "type": "remote",
                "provider": "deepseek",
                "endpoint": "https://api.deepseek.com",
                "api_key_env": "DEEPSEEK_API_KEY",
                "model": "deepseek-v4-flash",
            },
        ],
        "aliases": ["test*"],
    }


# ---------------------------------------------------------------------------
# SSE chunk helpers (mirrors test_midstream_reroute.py)
# ---------------------------------------------------------------------------


def _sse_data(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _reasoning_chunk(text: str) -> bytes:
    return _sse_data({"choices": [{"delta": {"reasoning_content": text}, "index": 0}]})


def _content_chunk(text: str) -> bytes:
    return _sse_data({"choices": [{"delta": {"content": text}, "index": 0}]})


def _error_event() -> bytes:
    return _sse_data({
        "choices": [
            {"delta": {}, "finish_reason": "error", "index": 0,
             "error": {
                 "type": "stall_after_content",
                 "message": "Upstream idle timeout",
                 "provider": "opencode-go",
                 "model": "deepseek-v4-flash",
                 "entry": "opencode-go-2-deepseek",
                 "suggested_action": "Retry the request with full context, or route to a healthier provider",
             }},
        ]
    })


def _done_marker() -> bytes:
    return b"data: [DONE]\n\n"


def _reasoning_only_stall_stream():
    """Reasoning chunks, then a terminal error. Zero tool_calls, zero content."""
    return [
        _reasoning_chunk("Let me think"),
        _reasoning_chunk(" about this problem"),
        _error_event(),
    ]


def _healthy_stream():
    return [
        _reasoning_chunk("Thinking"),
        _content_chunk("Hello"),
        _content_chunk(" world"),
        _sse_data({"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}),
        _done_marker(),
    ]


def _make_streaming_response(chunks, status=200):
    async def _body():
        for c in chunks:
            yield c

    return StreamingResponse(_body(), status_code=status, media_type="text/event-stream")


def _ok_json_response():
    return Response(
        content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
        status_code=200,
        media_type="application/json",
    )


async def _collect(result: Response) -> str:
    """Collect a StreamingResponse body into a decoded string."""
    body = b"".join([c async for c in result.body_iterator])
    return body.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# F1 AC1: _failure_domain_key normalization
# ---------------------------------------------------------------------------


def test_failure_domain_key_basic_endpoint():
    """A plain remote endpoint yields itself as the failure-domain key."""
    cfg = {"name": "a", "type": "remote", "endpoint": "https://opencode.ai/zen/go"}
    assert provider._failure_domain_key(cfg) == "https://opencode.ai/zen/go"


def test_failure_domain_key_normalizes_scheme_host_case():
    """Scheme and host are lowercased; path case is preserved."""
    cfg = {"name": "a", "type": "remote", "endpoint": "HTTPS://Opencode.AI/Zen/Go"}
    assert provider._failure_domain_key(cfg) == "https://opencode.ai/Zen/Go"


def test_failure_domain_key_strips_trailing_slash():
    """A trailing slash on the path is stripped."""
    cfg = {"name": "a", "type": "remote", "endpoint": "https://opencode.ai/zen/go/"}
    assert provider._failure_domain_key(cfg) == "https://opencode.ai/zen/go"


def test_failure_domain_key_strips_fragment():
    """A URL fragment is dropped from the key."""
    cfg = {"name": "a", "type": "remote", "endpoint": "https://opencode.ai/zen/go#section"}
    assert provider._failure_domain_key(cfg) == "https://opencode.ai/zen/go"


def test_failure_domain_key_drops_default_ports():
    """Default ports (443 for https, 80 for http) are dropped."""
    https_cfg = {"name": "a", "type": "remote", "endpoint": "https://opencode.ai:443/zen/go"}
    assert provider._failure_domain_key(https_cfg) == "https://opencode.ai/zen/go"
    http_cfg = {"name": "a", "type": "remote", "endpoint": "http://opencode.ai:80/zen/go"}
    assert provider._failure_domain_key(http_cfg) == "http://opencode.ai/zen/go"


def test_failure_domain_key_keeps_query_strings():
    """Query strings are retained in the key."""
    cfg = {"name": "a", "type": "remote", "endpoint": "https://opencode.ai/zen/go?api=v1&k=2"}
    assert provider._failure_domain_key(cfg) == "https://opencode.ai/zen/go?api=v1&k=2"


def test_failure_domain_key_falls_back_to_brand_for_local():
    """Local/no-endpoint entries key on the provider brand."""
    cfg = {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3", "provider": "qwen"}
    assert provider._failure_domain_key(cfg) == "qwen"


def test_failure_domain_key_falls_back_to_name_without_brand():
    """An entry with no endpoint and no brand keys on its own name (no
    over-grouping across distinct local entries)."""
    cfg = {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"}
    assert provider._failure_domain_key(cfg) == "local-qwen3"


# ---------------------------------------------------------------------------
# F1 AC2: same-domain skip in _resolve_provider_with_exclusions (parent AC1)
# ---------------------------------------------------------------------------


def test_resolve_with_exclusions_skips_same_endpoint_entry(opencode_same_gateway_chain):
    """After a stall on the first opencode-go entry (domain
    ``https://opencode.ai/zen/go``), resolution skips the second same-endpoint
    entry and returns ``deepseek-v4-flash``."""
    first = opencode_same_gateway_chain["providers"][0]
    domain = provider._failure_domain_key(first)

    result = provider._resolve_provider_with_exclusions(
        opencode_same_gateway_chain,
        excluded_provider_names={"opencode-go-2-deepseek"},
        excluded_domains={domain},
    )
    assert result is not None
    assert result["name"] == "deepseek-v4-flash"


def test_resolve_with_exclusions_does_not_skip_different_endpoint(opencode_distinct_endpoints_chain):
    """No over-grouping: excluding the ``https://opencode.ai/zen`` domain must
    NOT skip the ``https://opencode.ai/zen/go`` entry (parent AC2)."""
    free_entry = opencode_distinct_endpoints_chain["providers"][0]
    domain = provider._failure_domain_key(free_entry)
    assert domain == "https://opencode.ai/zen"

    result = provider._resolve_provider_with_exclusions(
        opencode_distinct_endpoints_chain,
        excluded_provider_names={"opencode-deepseek"},
        excluded_domains={domain},
    )
    assert result is not None
    assert result["name"] == "opencode-go-2-deepseek"


def test_resolve_with_exclusions_default_no_domains(opencode_same_gateway_chain):
    """Backward compatibility: without ``excluded_domains``, the same-endpoint
    sibling remains eligible (only the excluded name is skipped)."""
    result = provider._resolve_provider_with_exclusions(
        opencode_same_gateway_chain,
        excluded_provider_names={"opencode-go-2-deepseek"},
    )
    assert result is not None
    assert result["name"] == "opencode-go-deepseek"


# ---------------------------------------------------------------------------
# F1 AC4: resolve_provider skips the failed provider's failure domain
# ---------------------------------------------------------------------------


def test_resolve_provider_skips_same_domain_after_failure(opencode_same_gateway_chain):
    """``resolve_provider(failed_provider=opencode-go-2-deepseek)`` skips the
    same-endpoint entry and returns the different gateway."""
    result = provider.resolve_provider(
        opencode_same_gateway_chain,
        failed_provider="opencode-go-2-deepseek",
    )
    assert result is not None
    assert result["name"] == "deepseek-v4-flash"


def test_resolve_provider_keeps_different_endpoint_after_failure(opencode_distinct_endpoints_chain):
    """``resolve_provider`` must NOT over-group: failing
    ``opencode-deepseek`` (zen) leaves ``opencode-go-2-deepseek``
    (zen/go) eligible."""
    result = provider.resolve_provider(
        opencode_distinct_endpoints_chain,
        failed_provider="opencode-deepseek",
    )
    assert result is not None
    assert result["name"] == "opencode-go-2-deepseek"


def test_resolve_provider_still_skips_failed_name(opencode_same_gateway_chain):
    """The failed provider name itself is still skipped (existing behavior)."""
    result = provider.resolve_provider(
        opencode_same_gateway_chain,
        failed_provider="opencode-go-deepseek",
    )
    assert result is not None
    assert result["name"] == "deepseek-v4-flash"


# ---------------------------------------------------------------------------
# F1 AC5: integration — mid-stream stall re-routes past same-gateway entry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_midstream_stall_reroutes_past_same_gateway_entry(opencode_same_gateway_chain):
    """A mid-stream stall on ``opencode-go-2-deepseek`` re-routes the SAME
    request past ``opencode-go-deepseek`` (never called — same failure domain)
    straight to ``deepseek-v4-flash`` (parent AC3)."""
    call_order: list[str] = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        name = provider_cfg["name"]
        call_order.append(name)
        if name == "opencode-go-2-deepseek":
            return _make_streaming_response(_reasoning_only_stall_stream())
        if name == "opencode-go-deepseek":
            # Same broken gateway: also stalls (mirrors observed logs).
            return _make_streaming_response(_reasoning_only_stall_stream())
        return _ok_json_response()

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", opencode_same_gateway_chain,
            {"provider_cooldown_seconds": 60},
        )

    assert call_order == ["opencode-go-2-deepseek", "deepseek-v4-flash"], (
        f"Re-route must skip the same-endpoint entry, got {call_order}"
    )
    assert "opencode-go-deepseek" not in call_order
    assert result.status_code == 200
    body = json.loads(result.body.decode()) if isinstance(result.body, bytes) else json.loads(result.body)
    assert body["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_midstream_stall_reroutes_past_same_gateway_in_proxy_with_fallback(opencode_same_gateway_chain):
    """Same behavior through the combined local+remote chain."""
    call_order: list[str] = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        name = provider_cfg["name"]
        call_order.append(name)
        if name == "opencode-go-2-deepseek":
            return _make_streaming_response(_reasoning_only_stall_stream())
        if name == "opencode-go-deepseek":
            return _make_streaming_response(_reasoning_only_stall_stream())
        return _ok_json_response()

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", opencode_same_gateway_chain,
            {"provider_cooldown_seconds": 60},
        )

    assert call_order == ["opencode-go-2-deepseek", "deepseek-v4-flash"], (
        f"Re-route must skip the same-endpoint entry, got {call_order}"
    )
    assert result.status_code == 200
    body = json.loads(result.body.decode()) if isinstance(result.body, bytes) else json.loads(result.body)
    assert body["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_midstream_stall_keeps_different_endpoint_in_chain(opencode_distinct_endpoints_chain):
    """No over-grouping end-to-end: a stall on ``opencode-deepseek``
    (zen) still tries ``opencode-go-2-deepseek`` (zen/go) before reaching
    ``deepseek-v4-flash``."""
    call_order: list[str] = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        name = provider_cfg["name"]
        call_order.append(name)
        if name == "opencode-deepseek":
            return _make_streaming_response(_reasoning_only_stall_stream())
        if name == "opencode-go-2-deepseek":
            return _make_streaming_response(_reasoning_only_stall_stream())
        return _ok_json_response()

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", opencode_distinct_endpoints_chain,
            {"provider_cooldown_seconds": 60},
        )

    assert call_order == ["opencode-deepseek", "opencode-go-2-deepseek", "deepseek-v4-flash"], (
        f"Distinct endpoints must NOT be grouped, got {call_order}"
    )
    assert result.status_code == 200


async def _local_lease_active_response():
    return Response(
        content=json.dumps({
            "error": {
                "type": "server_busy",
                "code": "no_slots_available",
                "reason": "local_lease_active",
                "message": "Local slot reserved for another session",
            },
            "total_slots": 3,
            "available_slots": 0,
        }),
        status_code=503,
        media_type="application/json",
    )


@pytest.mark.asyncio
async def test_lease_active_bypass_does_not_retry_same_domain():
    """The local_lease_active bypass (which ignores cooldown) must still skip
    entries whose failure domain already stalled this request — a same-gateway
    sibling must not be re-selected (LP-0MSG45I8Q0020N1F)."""
    config = {
        "providers": [
            {"name": "local", "type": "local", "llama_model": "Qwen3"},
            {"name": "opencode-go-2-deepseek", "type": "remote", "provider": "opencode-go",
             "endpoint": "https://opencode.ai/zen/go", "api_key_env": "K1"},
            {"name": "opencode-go-deepseek", "type": "remote", "provider": "opencode-go",
             "endpoint": "https://opencode.ai/zen/go", "api_key_env": "K2"},
            {"name": "deepseek-v4-flash", "type": "remote", "provider": "deepseek",
             "endpoint": "https://api.deepseek.com", "api_key_env": "K3"},
        ],
        "aliases": ["test*"],
    }
    call_order: list[str] = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        name = provider_cfg["name"]
        call_order.append(name)
        if name == "opencode-go-2-deepseek":
            return _make_streaming_response(_reasoning_only_stall_stream())
        if name == "opencode-go-deepseek":
            return _make_streaming_response(_reasoning_only_stall_stream())
        return _ok_json_response()

    async def _mock_proxy_to_local(_req, _path):
        call_order.append("local")
        return await _local_lease_active_response()

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote), patch(
        "proxy.router.proxy_to_local", _mock_proxy_to_local
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", config,
            {
                "provider_cooldown_seconds": 60,
                # Disable the large-context smart-routing bypass so the local
                # provider is actually attempted in this test.
                "local_large_context_cold_cache_threshold": 0,
                "local_large_context_warm_cache_threshold": 0,
                "session_slot_pool_size": 4,
                "local_slot_retry_attempts": 0,
                "local_slot_retry_delay_seconds": 0,
            },
        )

    # local lease-active -> opencode-go-2 stalls -> same-domain sibling skipped
    # -> deepseek-v4-flash succeeds. The lease-active bypass must NOT resurrect
    # opencode-go-deepseek.
    assert call_order == ["local", "opencode-go-2-deepseek", "deepseek-v4-flash"], (
        f"Lease-active bypass must not retry same-domain sibling, got {call_order}"
    )
    assert result.status_code == 200
    body = json.loads(result.body.decode()) if isinstance(result.body, bytes) else json.loads(result.body)
    assert body["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_reroute_sse_comment_present_when_skipping_same_gateway(opencode_same_gateway_chain):
    """The SSE re-route comment still marks the switch, now directly
    primary->deepseek (no intermediate same-gateway hop)."""
    call_order: list[str] = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        name = provider_cfg["name"]
        call_order.append(name)
        if name == "opencode-go-2-deepseek":
            return _make_streaming_response(_reasoning_only_stall_stream())
        if name == "opencode-go-deepseek":
            return _make_streaming_response(_reasoning_only_stall_stream())
        return _make_streaming_response(_healthy_stream())

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", opencode_same_gateway_chain,
            {"provider_cooldown_seconds": 60},
        )

    collected = await _collect(result)
    assert ": re-route provider=opencode-go-2-deepseek->deepseek-v4-flash" in collected, (
        f"Expected direct re-route comment, got: {collected!r}"
    )
    assert "stall_after_reasoning" in collected
    assert "Hello" in collected


# ---------------------------------------------------------------------------
# F1 AC6: logging — skip reason includes failure-domain key
# ---------------------------------------------------------------------------


def test_same_domain_skip_logs_failure_domain_key(opencode_same_gateway_chain, caplog):
    """The skip log line includes the failure-domain key and the reason
    'same failure domain as ...' (parent AC4)."""
    first = opencode_same_gateway_chain["providers"][0]
    domain = provider._failure_domain_key(first)

    with caplog.at_level(logging.INFO, logger="llama-proxy.provider"):
        provider._resolve_provider_with_exclusions(
            opencode_same_gateway_chain,
            excluded_provider_names={"opencode-go-2-deepseek"},
            excluded_domains={domain},
        )

    assert any(
        "same failure domain as" in record.getMessage()
        and domain in record.getMessage()
        for record in caplog.records
    ), f"Expected 'same failure domain as' log with domain={domain}, got: {caplog.text}"
