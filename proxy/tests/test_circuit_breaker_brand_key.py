"""
Tests for the stall circuit breaker brand/entry key mismatch fix
(LP-0MSG45LOO007K236).

The Tier-3 stall circuit breaker marks the provider BRAND (e.g.
``opencode-go``) unavailable via ``mark_provider_unavailable()``, but the
fallback resolvers previously only checked the ENTRY name
(``opencode-go-2-deepseek``). The brand key was never consulted, so the
breaker cooldown never blocked the entries pointing at the broken gateway.

These tests verify the resolvers skip an entry when EITHER its entry name
OR its provider brand is in cooldown, so a tripped breaker actually
quarantines every entry pointing at the broken gateway.

Covers (child AC1-AC6):
- ``_resolve_provider_with_exclusions`` skips entries sharing a brand that
  is in cooldown (AC1).
- Cooldown expiry restores eligibility (AC2).
- ``resolve_provider`` (non-exclusions variant) honors brand cooldown (AC3).
- Integration: a stall trip on the brand makes the NEXT request skip both
  opencode-go entries and be served by deepseek-v4-flash (AC4).
- No regression: per-entry cooldowns and local providers unaffected (AC5).
- Diagnostics: skip log line includes the cooldown key and remaining
  seconds (AC6).
"""

import json
import logging
import time
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


class _TimeStepper:
    """Helper to control time.time() in tests via mock."""

    def __init__(self, start_time=1000.0):
        self._now = start_time

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def __call__(self) -> float:
        return self._now


@pytest.fixture(autouse=True)
def reset_cooldown_state():
    """Reset cooldown state and the stall circuit breaker singleton between
    tests to avoid cross-test leakage."""
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    import proxy.stall_circuit_breaker as scb
    scb._initialized = False
    scb.stall_circuit_breaker = scb.StallCircuitBreaker()
    yield


# ---------------------------------------------------------------------------
# Provider chain fixture (mirrors the real config.yaml ``plan`` model)
# ---------------------------------------------------------------------------


@pytest.fixture
def opencode_same_gateway_chain():
    """Two entries on the SAME gateway (opencode.ai/zen/go, same brand
    ``opencode-go``) followed by a different gateway (api.deepseek.com)."""
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
def mixed_chain_with_local():
    """Local provider first (no brand), then two same-brand remote entries."""
    return {
        "providers": [
            {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
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


def _ok_json_response():
    return Response(
        content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
        status_code=200,
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# AC1: _resolve_provider_with_exclusions skips entries sharing a brand in
# cooldown
# ---------------------------------------------------------------------------


def test_resolve_with_exclusions_skips_brand_cooldown_entries(
    opencode_same_gateway_chain,
):
    """After mark_provider_unavailable("opencode-go", 180), the resolver
    returns the non-opencode-go provider: both opencode-go entries are
    skipped because their BRAND is in cooldown (AC1)."""
    provider.mark_provider_unavailable("opencode-go", 180)

    result = provider._resolve_provider_with_exclusions(
        opencode_same_gateway_chain,
        excluded_provider_names=set(),
    )

    assert result is not None
    assert result["name"] == "deepseek-v4-flash", (
        f"Both opencode-go entries must be skipped (brand in cooldown), "
        f"got {result['name']}"
    )


def test_resolve_with_exclusions_returns_none_when_all_entries_share_brand():
    """When EVERY entry shares the cooling-down brand, no provider is left
    (AC1, exhaustive case)."""
    provider.mark_provider_unavailable("opencode-go", 180)
    chain = {
        "providers": [
            {
                "name": "opencode-go-2-deepseek",
                "type": "remote",
                "provider": "opencode-go",
                "endpoint": "https://opencode.ai/zen/go",
            },
            {
                "name": "opencode-go-deepseek",
                "type": "remote",
                "provider": "opencode-go",
                "endpoint": "https://opencode.ai/zen/go",
            },
        ],
        "aliases": ["test*"],
    }

    result = provider._resolve_provider_with_exclusions(
        chain, excluded_provider_names=set()
    )

    assert result is None


def test_resolve_with_exclusions_keeps_different_brand_entries(
    opencode_same_gateway_chain,
):
    """Entries with a DIFFERENT brand (deepseek) are NOT skipped when only
    opencode-go is in cooldown — no over-blocking (AC5 regression guard)."""
    provider.mark_provider_unavailable("opencode-go", 180)

    result = provider._resolve_provider_with_exclusions(
        opencode_same_gateway_chain,
        excluded_provider_names=set(),
    )

    assert result is not None
    assert result["name"] == "deepseek-v4-flash"


# ---------------------------------------------------------------------------
# AC2: Cooldown expiry restores eligibility
# ---------------------------------------------------------------------------


def test_brand_cooldown_expiry_restores_eligibility(
    monkeypatch, opencode_same_gateway_chain
):
    """After the brand cooldown expires, the opencode-go entries are eligible
    again (AC2)."""
    stepper = _TimeStepper()
    monkeypatch.setattr(time, "time", stepper)

    provider.mark_provider_unavailable("opencode-go", 180)
    # Within cooldown: brand entries skipped
    result = provider._resolve_provider_with_exclusions(
        opencode_same_gateway_chain,
        excluded_provider_names=set(),
    )
    assert result["name"] == "deepseek-v4-flash"

    # Advance past expiry
    stepper.advance(181)

    result = provider._resolve_provider_with_exclusions(
        opencode_same_gateway_chain,
        excluded_provider_names=set(),
    )
    assert result is not None
    assert result["name"] == "opencode-go-2-deepseek", (
        "Entries must be eligible again after brand cooldown expiry"
    )


# ---------------------------------------------------------------------------
# AC3: resolve_provider (non-exclusions variant) honors brand cooldown
# ---------------------------------------------------------------------------


def test_resolve_provider_skips_brand_cooldown_entries(opencode_same_gateway_chain):
    """resolve_provider also honors brand-level unavailability (AC3)."""
    provider.mark_provider_unavailable("opencode-go", 180)

    result = provider.resolve_provider(opencode_same_gateway_chain)

    assert result is not None
    assert result["name"] == "deepseek-v4-flash", (
        f"resolve_provider must skip brand entries in cooldown, got {result['name']}"
    )


def test_resolve_provider_brand_cooldown_expiry(monkeypatch, opencode_same_gateway_chain):
    """resolve_provider restores brand entries after cooldown expiry (AC3/AC2)."""
    stepper = _TimeStepper()
    monkeypatch.setattr(time, "time", stepper)

    provider.mark_provider_unavailable("opencode-go", 180)
    assert provider.resolve_provider(opencode_same_gateway_chain)["name"] == "deepseek-v4-flash"

    stepper.advance(181)
    result = provider.resolve_provider(opencode_same_gateway_chain)
    assert result["name"] == "opencode-go-2-deepseek"


# ---------------------------------------------------------------------------
# AC5: No regression — per-entry cooldowns and local providers unaffected
# ---------------------------------------------------------------------------


def test_per_entry_cooldown_still_blocks_only_that_entry(
    opencode_same_gateway_chain,
):
    """A cooldown keyed on the ENTRY name (e.g. FreeUsageLimitError 3h
    cooldown) still blocks only that entry — the same-brand sibling remains
    eligible because the brand itself is NOT in cooldown (AC5)."""
    provider.mark_provider_unavailable("opencode-go-2-deepseek", 180)

    result = provider._resolve_provider_with_exclusions(
        opencode_same_gateway_chain,
        excluded_provider_names=set(),
    )

    assert result is not None
    assert result["name"] == "opencode-go-deepseek", (
        "Per-entry cooldown must not skip the same-brand sibling when the "
        "brand itself is not in cooldown"
    )


def test_local_provider_unaffected_by_brand_cooldown(mixed_chain_with_local):
    """Local providers (no brand field) are unaffected by a remote brand
    cooldown — entry-name check only (AC5)."""
    provider.mark_provider_unavailable("opencode-go", 180)

    result = provider._resolve_provider_with_exclusions(
        mixed_chain_with_local,
        excluded_provider_names=set(),
    )

    assert result is not None
    assert result["name"] == "local-qwen3"


# ---------------------------------------------------------------------------
# AC6: Diagnostics — skip log line includes cooldown key and remaining seconds
# ---------------------------------------------------------------------------


def test_brand_cooldown_skip_logs_key_and_remaining(
    opencode_same_gateway_chain, caplog
):
    """The skip log line includes the brand cooldown key and the remaining
    cooldown seconds (AC6)."""
    provider.mark_provider_unavailable("opencode-go", 180)

    with caplog.at_level(logging.INFO, logger="llama-proxy.provider"):
        provider._resolve_provider_with_exclusions(
            opencode_same_gateway_chain,
            excluded_provider_names=set(),
        )

    assert any(
        "Skipping provider=opencode-go-2-deepseek" in record.getMessage()
        and "opencode-go" in record.getMessage()
        and "cooldown" in record.getMessage()
        and "remaining" in record.getMessage()
        for record in caplog.records
    ), f"Expected brand-cooldown skip log with key + remaining, got: {caplog.text}"


def test_entry_cooldown_skip_logs_key_and_remaining(
    opencode_same_gateway_chain, caplog
):
    """Entry-name cooldown skips also log the key + remaining seconds (AC6)."""
    provider.mark_provider_unavailable("opencode-go-2-deepseek", 180)

    with caplog.at_level(logging.INFO, logger="llama-proxy.provider"):
        provider._resolve_provider_with_exclusions(
            opencode_same_gateway_chain,
            excluded_provider_names=set(),
        )

    assert any(
        "Skipping provider=opencode-go-2-deepseek" in record.getMessage()
        and "opencode-go-2-deepseek in cooldown" in record.getMessage()
        and "remaining" in record.getMessage()
        for record in caplog.records
    ), f"Expected entry-cooldown skip log with key + remaining, got: {caplog.text}"


# ---------------------------------------------------------------------------
# AC4: Integration — stall trip on the brand makes the NEXT request skip both
# opencode-go entries and be served by deepseek-v4-flash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stall_breaker_trip_then_next_request_skips_both_entries(
    opencode_same_gateway_chain,
):
    """Trigger the stall circuit breaker on the ``opencode-go`` brand (3
    stalls, as proxy_remote does on stall-exhausted), then verify the NEXT
    request skips both opencode-go entries and completes via
    deepseek-v4-flash (AC4, mock upstreams)."""
    from proxy.stall_circuit_breaker import _check_stall_circuit_breaker

    config = {
        "server": {
            "upstream_stall_window_seconds": 300,
            "upstream_stall_threshold": 3,
            "upstream_stall_cooldown_seconds": 180,
        }
    }

    # Simulate 3 stalls recorded against the brand (as proxy_remote does via
    # _check_stall_circuit_breaker(provider or "remote", config)).
    for _ in range(3):
        _check_stall_circuit_breaker("opencode-go", config)

    # The circuit breaker fired: the brand is now in cooldown.
    assert provider._is_provider_unavailable("opencode-go"), (
        "Stall circuit breaker should have marked the brand unavailable"
    )

    # NEXT request: both opencode-go entries must be skipped.
    call_order: list[str] = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        name = provider_cfg["name"]
        call_order.append(name)
        return _ok_json_response()

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(),
            "v1/chat/completions",
            opencode_same_gateway_chain,
            {"provider_cooldown_seconds": 60},
        )

    assert call_order == ["deepseek-v4-flash"], (
        f"Next request must skip both opencode-go entries (brand in cooldown), "
        f"got {call_order}"
    )
    assert result.status_code == 200
    body = json.loads(result.body.decode()) if isinstance(result.body, bytes) else json.loads(result.body)
    assert body["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_stall_breaker_trip_does_not_block_other_brand(
    opencode_same_gateway_chain,
):
    """A stall trip on a DIFFERENT brand (deepseek) must not block the
    opencode-go entries — no cross-brand over-blocking (AC5 regression
    guard)."""
    from proxy.stall_circuit_breaker import _check_stall_circuit_breaker

    config = {
        "server": {
            "upstream_stall_window_seconds": 300,
            "upstream_stall_threshold": 3,
            "upstream_stall_cooldown_seconds": 180,
        }
    }

    for _ in range(3):
        _check_stall_circuit_breaker("deepseek", config)

    assert provider._is_provider_unavailable("deepseek")
    assert not provider._is_provider_unavailable("opencode-go")

    # The first provider (opencode-go brand) must still be selected.
    result = provider._resolve_provider_with_exclusions(
        opencode_same_gateway_chain,
        excluded_provider_names=set(),
    )
    assert result is not None
    assert result["name"] == "opencode-go-2-deepseek"
