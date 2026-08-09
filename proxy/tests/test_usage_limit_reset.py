"""
Tests for usage-limit reset handling (LP-0MSLJPOCC0001ROJ).

When the opencode upstream returns HTTP 429 with error type
``GoUsageLimitError`` (a usage-limit variant distinct from the already-handled
``FreeUsageLimitError``), the proxy must:

- Recognize it as a usage-limit event (not a generic rate-limit event).
- Compute the exact reset time from the provider message (e.g.
  ``Resets in 22hr 43min`` → now + 22h43m) plus a 2-minute safety margin.
- Quarantine the failing provider's API-key ACCOUNT until the computed
  reset time + margin passes, logging ``usage_limit_reset_pending`` for
  routing decisions during the block. Entries using a different
  ``api_key_env`` on the same gateway (e.g. opencode-go vs opencode-go-2)
  have independent limits and are NOT quarantined together
  (LP-0MSMBWB23009XYPW).
- Resume routing automatically once the reset time arrives.
- Keep the existing ``FreeUsageLimitError`` 3-hour cooldown unchanged.

Covers parent AC1-AC6:

- AC1: mocked 429 ``GoUsageLimitError`` + ``Resets in 22hr 43min`` → reset =
  now + 22h43m + 2min, provider skipped until then.
- AC2: Daily / Weekly / Monthly variants (message and/or ``limitName``) each
  produce the correct reset computation.
- AC3: entries sharing the failing entry's API-key account are blocked;
  entries with a different ``api_key_env`` on the same gateway are NOT
  (LP-0MSMBWB23009XYPW).
- AC4: after the reset time passes, routing resumes automatically.
- AC5: routing during the block logs ``usage_limit_reset_pending`` with the
  reset time and does not contact the upstream.
- AC6: existing ``FreeUsageLimitError`` 3-hour cooldown is unchanged.
"""

import json
import time
from unittest.mock import patch

import proxy.provider as provider
import pytest
from fastapi import Response


class _DummyRequest:
    """Minimal request stub (mirrors test_failure_domain_grouping._DummyRequest)."""

    def __init__(self, body: bytes = b'{"model":"test"}'):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


@pytest.fixture(autouse=True)
def reset_cooldown_state():
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    yield


@pytest.fixture
def opencode_usage_chain():
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


def _gousage_429(message: str, limit_name: str = "weekly") -> Response:
    """Build the observed opencode 429 GoUsageLimitError response body."""
    body = json.dumps({
        "type": "error",
        "error": {
            "type": "GoUsageLimitError",
            "message": message,
        },
        "metadata": {"limitName": limit_name},
    })
    return Response(status_code=429, content=body.encode("utf-8"))


# ---------------------------------------------------------------------------
# Message parsing (unit)
# ---------------------------------------------------------------------------


def test_parse_resets_in_variants():
    """_parse_resets_in should handle hr/min, days, and seconds formats."""
    assert provider._parse_resets_in("Weekly usage limit reached. Resets in 22hr 43min.") == pytest.approx(
        22 * 3600 + 43 * 60
    )
    assert provider._parse_resets_in("Resets in 23hr 17min.") == pytest.approx(
        23 * 3600 + 17 * 60
    )
    assert provider._parse_resets_in("Daily usage limit reached. Resets in 5 hours 30 minutes.") == pytest.approx(
        5 * 3600 + 30 * 60
    )
    assert provider._parse_resets_in("Monthly usage limit reached. Resets in 2 days.") == pytest.approx(
        2 * 86400
    )
    assert provider._parse_resets_in("Rate limit reached. Resets in 45s.") == pytest.approx(45)
    # No reset info -> None
    assert provider._parse_resets_in("Usage limit reached.") is None
    assert provider._parse_resets_in("") is None


# ---------------------------------------------------------------------------
# Detection + reset computation (unit)
# ---------------------------------------------------------------------------


def test_usage_limit_reset_seconds_gousage_with_message():
    """AC1: GoUsageLimitError + 'Resets in 22hr 43min' -> 22h43m + 2m margin."""
    resp = _gousage_429("Weekly usage limit reached. Resets in 22hr 43min.", "weekly")
    seconds = provider._usage_limit_reset_seconds(resp, resp.body.decode())
    assert seconds == pytest.approx(22 * 3600 + 43 * 60 + 120)


def test_usage_limit_reset_seconds_limit_name_fallback():
    """AC2: message without reset time falls back to metadata.limitName."""
    # daily -> 24h + 2m
    resp = _gousage_429("Daily usage limit reached.", "daily")
    assert provider._usage_limit_reset_seconds(resp, resp.body.decode()) == pytest.approx(
        24 * 3600 + 120
    )
    # weekly -> 7d + 2m
    resp = _gousage_429("Weekly usage limit reached.", "weekly")
    assert provider._usage_limit_reset_seconds(resp, resp.body.decode()) == pytest.approx(
        7 * 24 * 3600 + 120
    )
    # monthly -> 30d + 2m
    resp = _gousage_429("Monthly usage limit reached.", "monthly")
    assert provider._usage_limit_reset_seconds(resp, resp.body.decode()) == pytest.approx(
        30 * 24 * 3600 + 120
    )


def test_usage_limit_reset_seconds_requires_429_and_usage_error_type():
    """Non-429 status or non-usage-limit error types must not match."""
    body = json.dumps({
        "type": "error",
        "error": {
            "type": "GoUsageLimitError",
            "message": "Weekly usage limit reached. Resets in 22hr 43min.",
        },
        "metadata": {"limitName": "weekly"},
    })
    # 503 is not a 429
    resp_503 = Response(status_code=503, content=body.encode())
    assert provider._usage_limit_reset_seconds(resp_503, body) is None
    # 429 with a different error type
    body_other = json.dumps({"error": {"type": "rate_limit_error", "message": "Resets in 1h."}})
    resp_other = Response(status_code=429, content=body_other.encode())
    assert provider._usage_limit_reset_seconds(resp_other, body_other) is None
    # 429 usage-limit error with NO reset info anywhere
    body_none = json.dumps({
        "error": {"type": "GoUsageLimitError", "message": "Quota exhausted."},
        "metadata": {},
    })
    resp_none = Response(status_code=429, content=body_none.encode())
    assert provider._usage_limit_reset_seconds(resp_none, body_none) is None


def test_usage_limit_reset_seconds_free_usage_with_reset_time():
    """A FreeUsageLimitError variant that carries a reset time is handled too."""
    body = json.dumps({
        "error": {
            "type": "FreeUsageLimitError",
            "message": "Free tier limit reached. Resets in 1 hour.",
        },
        "metadata": {},
    })
    resp = Response(status_code=429, content=body.encode())
    assert provider._usage_limit_reset_seconds(resp, body) == pytest.approx(3600 + 120)


def test_usage_limit_reset_seconds_case_insensitive():
    """Error type matching is case-insensitive."""
    body = json.dumps({
        "error": {"type": "gousagelimiterror", "message": "Resets in 10 minutes."},
        "metadata": {"limitName": "weekly"},
    })
    resp = Response(status_code=429, content=body.encode())
    assert provider._usage_limit_reset_seconds(resp, body) == pytest.approx(600 + 120)


def test_usage_reset_remaining_cleanup():
    """_usage_reset_remaining returns 0 and cleans up expired entries."""
    domain = "https://opencode.ai/zen/go"
    provider._usage_reset_at[domain] = time.time() + 100
    assert provider._usage_reset_remaining(domain) == pytest.approx(100, abs=2)
    # Expired entry is lazily removed
    provider._usage_reset_at[domain] = time.time() - 1
    assert provider._usage_reset_remaining(domain) == 0
    assert domain not in provider._usage_reset_at
    # Unknown domain -> 0
    assert provider._usage_reset_remaining("https://other.example.com") == 0


# ---------------------------------------------------------------------------
# Fallback integration (AC1/AC3/AC4/AC5/AC6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remote_fallback_usage_limit_reset_quarantines_account_not_gateway(
    opencode_usage_chain, caplog
):
    """AC1+AC3+AC5: a 429 GoUsageLimitError quarantines only the failing
    API-key ACCOUNT; the same-gateway sibling with a different api_key_env has
    its own independent limit and is still tried (LP-0MSMBWB23009XYPW)."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}
    call_count = 0
    contacted = []

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        name = provider_cfg.get("name")
        contacted.append(name)
        if name == "opencode-go-2-deepseek":
            # First entry returns the observed GoUsageLimitError
            return _gousage_429("Weekly usage limit reached. Resets in 22hr 43min.", "weekly")
        # opencode-go-deepseek (different account) must still be contacted.
        assert name == "opencode-go-deepseek", (
            "deepseek-v4-flash must not be reached while a sibling account "
            "with its own limit is available"
        )
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        with caplog.at_level("INFO", logger="llama-proxy.provider"):
            result = await provider.proxy_with_remote_fallback(
                request, "v1/chat/completions", opencode_usage_chain, cfg
            )

    assert result.status_code == 200
    # Entry 1 (GoUsageLimitError) + entry 2 (opencode-go-deepseek); the
    # different-account sibling was contacted and deepseek was not needed.
    assert call_count == 2
    assert contacted == ["opencode-go-2-deepseek", "opencode-go-deepseek"]

    # Only the FAILING account is quarantined until reset + 2m margin.
    go2_key = provider._usage_limit_account_key(opencode_usage_chain["providers"][0])
    go_key = provider._usage_limit_account_key(opencode_usage_chain["providers"][1])
    assert go2_key != go_key
    expiry = provider._usage_reset_at.get(go2_key)
    assert expiry is not None, "failing account should have a usage reset expiry"
    remaining = expiry - time.time()
    assert remaining == pytest.approx(22 * 3600 + 43 * 60 + 120, abs=5)
    # The sibling account is NOT quarantined.
    assert go_key not in provider._usage_reset_at

    # A FRESH routing decision skips only the quarantined account and resolves
    # to the sibling (its own limit is untouched).
    with caplog.at_level("INFO", logger="llama-proxy.provider"):
        fresh = provider._resolve_provider_with_exclusions(opencode_usage_chain, set())
    assert fresh is not None
    assert fresh.get("name") == "opencode-go-deepseek"
    assert any(
        "usage_limit_reset_pending" in rec.message
        and "reset_at=" in rec.message
        and "reset_in=" in rec.message
        for rec in caplog.records
    )
    assert any(
        "usage_limit_reset_pending" in rec.message and "opencode-go-2-deepseek" in rec.message
        for rec in caplog.records
    )
    assert not any(
        "usage_limit_reset_pending" in rec.message and "opencode-go-deepseek" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_remote_fallback_usage_limit_reset_blocks_shared_account():
    """AC3: entries sharing the SAME api_key_env as the failing entry ARE
    quarantined together (one account, one limit)."""
    chain = {
        "providers": [
            {
                "name": "opencode-go-2-a",
                "type": "remote",
                "provider": "opencode-go",
                "endpoint": "https://opencode.ai/zen/go",
                "api_key_env": "OPENCODE_2_API_KEY",
                "model": "deepseek-v4-flash",
            },
            {
                "name": "opencode-go-2-b",
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
    }
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        name = provider_cfg.get("name")
        if name == "opencode-go-2-a":
            return _gousage_429("Weekly usage limit reached. Resets in 22hr 43min.", "weekly")
        # Same-account sibling must never be contacted; hop to the other account.
        assert name == "opencode-go-deepseek", (
            "opencode-go-2-b (same account) must never be contacted"
        )
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", chain, cfg
        )

    assert result.status_code == 200
    # go-2-a (429) + opencode-go-deepseek (ok); the same-account entry
    # opencode-go-2-b was skipped without a network call.
    assert call_count == 2


@pytest.mark.asyncio
async def test_remote_fallback_usage_limit_reset_daily_variant(opencode_usage_chain):
    """AC2: daily limit message produces the correct reset window end-to-end."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        if provider_cfg.get("name") == "opencode-go-2-deepseek":
            return _gousage_429("Daily usage limit reached. Resets in 1 hour.", "daily")
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", opencode_usage_chain, cfg
        )

    assert result.status_code == 200
    # The sibling account was tried and succeeded; only the failing account
    # carries the reset window.
    assert call_count == 2
    go2_key = provider._usage_limit_account_key(opencode_usage_chain["providers"][0])
    remaining = provider._usage_reset_at[go2_key] - time.time()
    assert remaining == pytest.approx(3600 + 120, abs=5)


@pytest.mark.asyncio
async def test_usage_limit_reset_expires_and_routing_resumes(opencode_usage_chain):
    """AC4: after the reset time passes, routing to the provider resumes."""
    # Quarantine the failing account with a reset already in the past.
    go2_key = provider._usage_limit_account_key(opencode_usage_chain["providers"][0])
    provider._usage_reset_at[go2_key] = time.time() - 1

    # The first provider of the chain is now eligible again.
    resolved = provider._resolve_provider_with_exclusions(opencode_usage_chain, set())
    assert resolved is not None
    assert resolved.get("name") == "opencode-go-2-deepseek"
    assert provider._usage_reset_remaining(go2_key) == 0


@pytest.mark.asyncio
async def test_resolve_provider_respects_usage_limit_reset(opencode_usage_chain, caplog):
    """AC5: resolve_provider skips accounts with a pending usage reset but
    still offers a different-account sibling on the same gateway."""
    go2_key = provider._usage_limit_account_key(opencode_usage_chain["providers"][0])
    provider._usage_reset_at[go2_key] = time.time() + 3600

    with caplog.at_level("INFO", logger="llama-proxy.provider"):
        resolved = provider.resolve_provider(opencode_usage_chain)

    assert resolved is not None
    assert resolved.get("name") == "opencode-go-deepseek"
    assert any(
        "usage_limit_reset_pending" in rec.message
        and "opencode-go-2-deepseek" in rec.message
        for rec in caplog.records
    )
    assert not any(
        "usage_limit_reset_pending" in rec.message
        and "opencode-go-deepseek" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_free_usage_limit_3h_cooldown_regression(sample_model_config):
    """AC6: FreeUsageLimitError without a reset time keeps the 3h cooldown."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return Response(
                status_code=429,
                content=json.dumps({
                    "type": "error",
                    "error": {
                        "type": "FreeUsageLimitError",
                        "message": "Rate limit exceeded. Please try again later.",
                    },
                    "metadata": {},
                }).encode("utf-8"),
                media_type="application/json",
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
    assert call_count == 2
    # 3-hour cooldown still applies (no usage reset entry for this domain).
    now = time.time()
    expiry = provider._provider_unavailable_until.get("remote-primary")
    assert expiry is not None
    cooldown_seconds = expiry - now
    assert 10700 <= cooldown_seconds <= 10900
    assert provider._usage_reset_at == {}


@pytest.mark.asyncio
async def test_usage_limit_reset_in_proxy_with_fallback(opencode_usage_chain, caplog):
    """GoUsageLimitError handling also applies in proxy_with_fallback."""
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        if provider_cfg.get("name") == "opencode-go-2-deepseek":
            return _gousage_429("Weekly usage limit reached. Resets in 22hr 43min.", "weekly")
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
            status_code=200,
            media_type="application/json",
        )

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        with caplog.at_level("INFO", logger="llama-proxy.provider"):
            result = await provider.proxy_with_fallback(
                request, "v1/chat/completions", opencode_usage_chain, cfg
            )

    assert result.status_code == 200
    # go-2 (429) + sibling account (ok); deepseek not needed.
    assert call_count == 2
    go2_key = provider._usage_limit_account_key(opencode_usage_chain["providers"][0])
    remaining = provider._usage_reset_at[go2_key] - time.time()
    assert remaining == pytest.approx(22 * 3600 + 43 * 60 + 120, abs=5)
    # AC5: a fresh routing decision after the block resolves to the sibling
    # account, not deepseek.
    with caplog.at_level("INFO", logger="llama-proxy.provider"):
        fresh = provider._resolve_provider_with_exclusions(opencode_usage_chain, set())
    assert fresh is not None
    assert fresh.get("name") == "opencode-go-deepseek"
    assert any("usage_limit_reset_pending" in rec.message for rec in caplog.records)
