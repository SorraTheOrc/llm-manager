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
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import proxy.provider as provider
import proxy.quota_state as quota_state
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
    provider._usage_reset_guessed.clear()
    provider._usage_reset_window.clear()
    provider._usage_limit_last_probe.clear()
    quota_state.reset_quota_state()
    yield
    quota_state.reset_quota_state()


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


def opencode_usage_chain_entry() -> dict:
    """A single ``opencode-*`` entry, mirroring the chain fixture's first one.

    Unit-level reset tests only need one entry (plus a seeded snapshot) rather
    than the full three-provider chain.
    """
    return {
        "name": "opencode-go-2-deepseek",
        "type": "remote",
        "provider": "opencode-go",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": "OPENCODE_2_API_KEY",
        "model": "deepseek-v4-flash",
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
# Authoritative quota-snapshot reset (LP-0MVF7MPZN009GOJB)
# ---------------------------------------------------------------------------

#: Frozen wall clock the quota module reads through ``quota_state.set_clock``.
_NOW = datetime(2026, 10, 19, 12, 0, 0, tzinfo=UTC)

#: The monthly reset offset used by seeded snapshots (seconds from ``_NOW``).
_SNAPSHOT_RESET_SECONDS = 5000


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _usage_body(monthly_reset_seconds: float) -> dict:
    """A complete ``/v1/usage`` body with a known monthly reset offset."""
    return {
        "usage": {
            "rolling": {
                "status": "ok",
                "percent": 0.0,
                "resetsAt": _iso(_NOW + timedelta(hours=5)),
            },
            "weekly": {
                "status": "ok",
                "percent": 0.0,
                "resetsAt": _iso(_NOW + timedelta(days=2)),
            },
            "monthly": {
                "status": "ok",
                "percent": 50.0,
                "resetsAt": _iso(_NOW + timedelta(seconds=monthly_reset_seconds)),
            },
        }
    }


class _UsageTransport:
    """Injectable ``/v1/usage`` transport keyed by account (test seam)."""

    def __init__(self, bodies: dict[str, dict]):
        self._bodies = dict(bodies)

    async def __call__(self, provider_cfg: dict) -> dict:
        return self._bodies[quota_state.account_key(provider_cfg)]


async def _seed_snapshot(
    provider_cfg: dict, config: dict | None, body: dict
) -> None:
    """Populate the quota cache for one account through the public poller."""
    account = quota_state.account_key(provider_cfg)
    quota_state.set_quota_transport(_UsageTransport({account: body}))
    await quota_state.refresh_quota_accounts(
        {"providers": [provider_cfg]}, config, force=True
    )


@pytest.mark.asyncio
async def test_quota_snapshot_reset_seconds_uses_fresh_snapshot():
    """Unit: a fresh ``opencode-*`` snapshot returns its monthly reset."""
    entry = {
        "name": "opencode-go-2-deepseek",
        "provider": "opencode-go",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": "OPENCODE_2_API_KEY",
    }
    config: dict = {}
    quota_state.set_clock(lambda: _NOW)
    await _seed_snapshot(entry, config, _usage_body(_SNAPSHOT_RESET_SECONDS))

    assert provider._quota_snapshot_reset_seconds(entry, config) == pytest.approx(
        _SNAPSHOT_RESET_SECONDS, abs=1
    )


@pytest.mark.asyncio
async def test_quota_snapshot_reset_seconds_none_when_stale():
    """Unit: a stale snapshot is not authoritative (falls back to the guess)."""
    entry = {
        "name": "opencode-go-2-deepseek",
        "provider": "opencode-go",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": "OPENCODE_2_API_KEY",
    }
    config: dict = {}
    quota_state.set_clock(lambda: _NOW)
    # Monthly reset far in the future, but the reading itself ages out.
    await _seed_snapshot(entry, config, _usage_body(15 * 86400))
    quota_state.set_clock(lambda: _NOW + timedelta(hours=2))

    assert provider._quota_snapshot_reset_seconds(entry, config) is None


@pytest.mark.asyncio
async def test_quota_snapshot_reset_seconds_none_when_absent():
    """Unit: no snapshot at all yields ``None`` (guess path unchanged)."""
    entry = {
        "name": "opencode-go-2-deepseek",
        "provider": "opencode-go",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": "OPENCODE_2_API_KEY",
    }
    quota_state.set_clock(lambda: _NOW)
    assert provider._quota_snapshot_reset_seconds(entry, {}) is None


@pytest.mark.asyncio
async def test_quota_snapshot_reset_seconds_ignores_non_opencode():
    """Unit: a non-``opencode-*`` entry never uses quota state, even when a
    snapshot exists under its shared account key."""
    opencode_entry = {
        "name": "opencode-go-2-deepseek",
        "provider": "opencode-go",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": "SHARED_KEY",
    }
    deepseek_entry = {
        "name": "deepseek-v4-flash",
        "provider": "deepseek",
        "endpoint": "https://api.deepseek.com",
        "api_key_env": "SHARED_KEY",
    }
    config: dict = {}
    quota_state.set_clock(lambda: _NOW)
    await _seed_snapshot(
        opencode_entry, config, _usage_body(_SNAPSHOT_RESET_SECONDS)
    )
    # The snapshot is keyed on the shared api_key_env, so it *is* visible to
    # the deepseek entry — the provider-type gate is what keeps it out.
    assert quota_state.get_quota_snapshot(deepseek_entry, config) is not None
    assert provider._quota_snapshot_reset_seconds(deepseek_entry, config) is None


@pytest.mark.asyncio
async def test_quota_snapshot_reset_seconds_respects_kill_switch_and_dry_run():
    """Unit: the kill switch and dry-run both keep pre-feature behaviour."""
    entry = {
        "name": "opencode-go-2-deepseek",
        "provider": "opencode-go",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": "OPENCODE_2_API_KEY",
    }
    quota_state.set_clock(lambda: _NOW)
    await _seed_snapshot(entry, {}, _usage_body(_SNAPSHOT_RESET_SECONDS))

    assert (
        provider._quota_snapshot_reset_seconds(
            entry, {"server": {"quota_surge_enabled": False}}
        )
        is None
    )
    assert (
        provider._quota_snapshot_reset_seconds(
            entry, {"server": {"quota_surge_dry_run": True}}
        )
        is None
    )
    # Sanity: default (enabled) config does use it.
    assert provider._quota_snapshot_reset_seconds(entry, {}) is not None


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


def test_usage_limit_reset_seconds_limit_name_fallback_is_capped():
    """AC2 (updated for LP-0MUQTCMW2001VXH8): a message without an explicit
    reset time falls back to ``metadata.limitName``, but the guessed period is
    capped at the 24h guess cap instead of the full 7d/30d period."""
    cap = provider._USAGE_LIMIT_GUESS_MAX_SECONDS
    # daily (24h) is naturally within the cap
    resp = _gousage_429("Daily usage limit reached.", "daily")
    assert provider._usage_limit_reset_seconds(resp, resp.body.decode()) == pytest.approx(
        min(24 * 3600, cap) + 120
    )
    # weekly (7d) is capped at 24h
    resp = _gousage_429("Weekly usage limit reached.", "weekly")
    assert provider._usage_limit_reset_seconds(resp, resp.body.decode()) == pytest.approx(
        cap + 120
    )
    # monthly (30d) is capped at 24h, never the full period
    resp = _gousage_429("Monthly usage limit reached.", "monthly")
    assert provider._usage_limit_reset_seconds(resp, resp.body.decode()) == pytest.approx(
        cap + 120
    )
    assert provider._usage_limit_reset_seconds(resp, resp.body.decode()) < 30 * 24 * 3600


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


# ---------------------------------------------------------------------------
# AC1/AC2: authoritative snapshot reset vs. fallback (LP-0MVF7MPZN009GOJB)
# ---------------------------------------------------------------------------


def _ok_response() -> Response:
    return Response(
        content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
        status_code=200,
        media_type="application/json",
    )


@pytest.mark.asyncio
async def test_gousagelimit_fresh_snapshot_reset_wins_for_opencode(
    opencode_usage_chain, caplog
):
    """AC1: a ``GoUsageLimitError`` on an ``opencode-*`` account with a fresh
    snapshot quarantines for exactly the snapshot's window ``seconds_to_reset``
    (+ the configured 2-minute margin), even when the message advertises a
    different explicit ``Resets in ...`` duration."""
    cfg = {"provider_cooldown_seconds": 60}
    entry = opencode_usage_chain["providers"][0]
    quota_state.set_clock(lambda: _NOW)
    await _seed_snapshot(entry, cfg, _usage_body(_SNAPSHOT_RESET_SECONDS))

    request = _DummyRequest()
    call_count = 0

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        nonlocal call_count
        call_count += 1
        if provider_cfg.get("name") == "opencode-go-2-deepseek":
            # The message claims ~22h43m; the snapshot must still win.
            return _gousage_429(
                "Monthly usage limit reached. Resets in 22hr 43min.", "monthly"
            )
        return _ok_response()

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        with caplog.at_level("INFO", logger="llama-proxy.provider"):
            result = await provider.proxy_with_remote_fallback(
                request, "v1/chat/completions", opencode_usage_chain, cfg
            )

    assert result.status_code == 200
    go2_key = provider._usage_limit_account_key(entry)
    remaining = provider._usage_reset_at[go2_key] - time.time()
    # Exactly the snapshot's monthly window reset + 2-minute margin — not 22h43m.
    assert remaining == pytest.approx(_SNAPSHOT_RESET_SECONDS + 120, abs=5)
    assert go2_key not in provider._usage_reset_guessed
    assert provider._usage_reset_window[go2_key] == "monthly"
    # The behaviour change is observable in the log.
    assert any(
        "source=snapshot" in rec.message and "opencode-go-2-deepseek" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_gousagelimit_stale_snapshot_falls_back_to_message(
    opencode_usage_chain, caplog
):
    """AC2: a stale snapshot falls back to the existing message-parsing path
    unchanged (explicit ``Resets in ...`` duration)."""
    cfg = {"provider_cooldown_seconds": 60}
    entry = opencode_usage_chain["providers"][0]
    quota_state.set_clock(lambda: _NOW)
    # Monthly reset far in the future, but the reading ages past the default
    # 3600 s staleness horizon.
    await _seed_snapshot(entry, cfg, _usage_body(15 * 86400))
    quota_state.set_clock(lambda: _NOW + timedelta(hours=2))

    request = _DummyRequest()

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        if provider_cfg.get("name") == "opencode-go-2-deepseek":
            return _gousage_429(
                "Weekly usage limit reached. Resets in 22hr 43min.", "weekly"
            )
        return _ok_response()

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        with caplog.at_level("INFO", logger="llama-proxy.provider"):
            result = await provider.proxy_with_remote_fallback(
                request, "v1/chat/completions", opencode_usage_chain, cfg
            )

    assert result.status_code == 200
    go2_key = provider._usage_limit_account_key(entry)
    remaining = provider._usage_reset_at[go2_key] - time.time()
    assert remaining == pytest.approx(22 * 3600 + 43 * 60 + 120, abs=5)
    assert any("source=message" in rec.message for rec in caplog.records)
    assert not any("source=snapshot" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_gousagelimit_stale_snapshot_falls_back_to_guess(
    opencode_usage_chain, caplog
):
    """AC2: a stale snapshot with a duration-less message uses the capped
    ``metadata.limitName`` guess unchanged."""
    cfg = {"provider_cooldown_seconds": 60}
    entry = opencode_usage_chain["providers"][0]
    quota_state.set_clock(lambda: _NOW)
    await _seed_snapshot(entry, cfg, _usage_body(15 * 86400))
    quota_state.set_clock(lambda: _NOW + timedelta(hours=2))

    request = _DummyRequest()

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        if provider_cfg.get("name") == "opencode-go-2-deepseek":
            return _gousage_429("Weekly usage limit reached.", "weekly")
        return _ok_response()

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        with caplog.at_level("INFO", logger="llama-proxy.provider"):
            result = await provider.proxy_with_remote_fallback(
                request, "v1/chat/completions", opencode_usage_chain, cfg
            )

    assert result.status_code == 200
    go2_key = provider._usage_limit_account_key(entry)
    remaining = provider._usage_reset_at[go2_key] - time.time()
    cap = provider._USAGE_LIMIT_GUESS_MAX_SECONDS
    assert remaining == pytest.approx(cap + 120, abs=5)
    assert go2_key in provider._usage_reset_guessed
    assert any("source=guess" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_gousagelimit_absent_snapshot_falls_back_to_guess(
    opencode_usage_chain,
):
    """AC2: with no snapshot at all the existing guess path is used unchanged."""
    cfg = {"provider_cooldown_seconds": 60}
    request = _DummyRequest()

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        if provider_cfg.get("name") == "opencode-go-2-deepseek":
            return _gousage_429("Weekly usage limit reached.", "weekly")
        return _ok_response()

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", opencode_usage_chain, cfg
        )

    assert result.status_code == 200
    go2_key = provider._usage_limit_account_key(opencode_usage_chain["providers"][0])
    remaining = provider._usage_reset_at[go2_key] - time.time()
    cap = provider._USAGE_LIMIT_GUESS_MAX_SECONDS
    assert remaining == pytest.approx(cap + 120, abs=5)


@pytest.mark.asyncio
async def test_gousagelimit_non_opencode_ignores_snapshot():
    """A non-``opencode-*`` entry keeps the existing message/guess path even
    when a quota snapshot exists under its shared account key."""
    cfg = {"provider_cooldown_seconds": 60}
    chain = {
        "providers": [
            {
                "name": "deepseek-v4-flash",
                "type": "remote",
                "provider": "deepseek",
                "endpoint": "https://api.deepseek.com",
                "api_key_env": "SHARED_KEY",
                "model": "deepseek-v4-flash",
            },
            {
                "name": "opencode-go-2-deepseek",
                "type": "remote",
                "provider": "opencode-go",
                "endpoint": "https://opencode.ai/zen/go",
                "api_key_env": "SHARED_KEY",
                "model": "deepseek-v4-flash",
            },
        ],
    }
    quota_state.set_clock(lambda: _NOW)
    await _seed_snapshot(
        chain["providers"][1], cfg, _usage_body(_SNAPSHOT_RESET_SECONDS)
    )
    assert (
        quota_state.get_quota_snapshot(chain["providers"][0], cfg) is not None
    )

    request = _DummyRequest()

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        if provider_cfg.get("name") == "deepseek-v4-flash":
            return _gousage_429(
                "Weekly usage limit reached. Resets in 22hr 43min.", "weekly"
            )
        return _ok_response()

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        result = await provider.proxy_with_remote_fallback(
            request, "v1/chat/completions", chain, cfg
        )

    assert result.status_code == 200
    deepseek_key = provider._usage_limit_account_key(chain["providers"][0])
    remaining = provider._usage_reset_at[deepseek_key] - time.time()
    # The non-opencode entry used the message duration, not the snapshot.
    assert remaining == pytest.approx(22 * 3600 + 43 * 60 + 120, abs=5)


# ---------------------------------------------------------------------------
# Window-aware reset selection (LP-0MVPAE77Z002VGIA AC1-AC5, AC8, AC9)
#
# The 429 payload names WHICH quota window was exceeded
# (``metadata.limitName``). The quarantine must follow that window's own
# reset — before this fix it always used the monthly account-level reset, so
# a weekly-limited account stayed locked out for up to a month
# (the 2026-10-24 incident).
# ---------------------------------------------------------------------------

#: The incident's window geometry: weekly limit, monthly reset 24 days out.
_INCIDENT_ROLLING = (0.0, 5 * 3600)
_INCIDENT_WEEKLY = (99.0, 3 * 86400)
_INCIDENT_MONTHLY = (54.0, 24 * 86400)


def _window(spec: tuple[float, float | None], status: str = "ok") -> dict:
    percent, offset = spec
    return {
        "status": status,
        "percent": percent,
        "resetsAt": (
            _iso(_NOW + timedelta(seconds=offset)) if offset is not None else None
        ),
    }


def _windowed_usage_body(
    *,
    rolling: tuple[float, float | None] = (0.0, 5 * 3600),
    weekly: tuple[float, float | None] = (0.0, 7 * 86400),
    monthly: tuple[float, float | None] = (50.0, 24 * 86400),
    rolling_status: str = "ok",
    weekly_status: str = "ok",
    monthly_status: str = "ok",
) -> dict:
    """A ``/v1/usage`` body with an independent percent + reset per window."""
    return {
        "usage": {
            "rolling": _window(rolling, rolling_status),
            "weekly": _window(weekly, weekly_status),
            "monthly": _window(monthly, monthly_status),
        }
    }


def _reset_info(
    limit_name: str | None, provider_cfg: dict, config: dict | None = None,
    message: str = "Usage limit reached.",
) -> provider._UsageLimitReset | None:
    """Resolve a duration-less 429 for *provider_cfg* through the unit seam."""
    body = json.dumps({
        "type": "error",
        "error": {"type": "GoUsageLimitError", "message": message},
        "metadata": {} if limit_name is None else {"limitName": limit_name},
    })
    response = Response(status_code=429, content=body.encode("utf-8"))
    return provider._usage_limit_reset_info(
        response, body, config, provider_cfg=provider_cfg
    )


@pytest.mark.asyncio
async def test_weekly_limit_uses_the_weekly_window_reset():
    """AC1: ``limitName=weekly`` quarantines for the weekly reset, not monthly."""
    entry = opencode_usage_chain_entry()
    cfg: dict = {}
    quota_state.set_clock(lambda: _NOW)
    weekly_seconds = 3 * 86400
    monthly_seconds = 24 * 86400
    await _seed_snapshot(
        entry,
        cfg,
        _windowed_usage_body(
            rolling=_INCIDENT_ROLLING,
            weekly=(99.0, weekly_seconds),
            monthly=_INCIDENT_MONTHLY,
        ),
    )

    info = _reset_info("weekly", entry, cfg)

    assert info is not None
    assert info.window == "weekly"
    assert info.explicit is True
    margin = provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
    assert info.seconds == pytest.approx(weekly_seconds + margin, abs=5)
    assert info.seconds < monthly_seconds


@pytest.mark.asyncio
async def test_rolling_limit_uses_the_rolling_window_reset():
    """AC2: ``limitName=rolling`` quarantines for the rolling reset."""
    entry = opencode_usage_chain_entry()
    cfg: dict = {}
    quota_state.set_clock(lambda: _NOW)
    rolling_seconds = 4 * 3600
    await _seed_snapshot(
        entry,
        cfg,
        _windowed_usage_body(
            rolling=(99.0, rolling_seconds),
            weekly=(0.0, 6 * 86400),
            monthly=(10.0, 20 * 86400),
        ),
    )

    info = _reset_info("rolling", entry, cfg)

    assert info is not None
    assert info.window == "rolling"
    margin = provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
    assert info.seconds == pytest.approx(rolling_seconds + margin, abs=5)


@pytest.mark.asyncio
async def test_monthly_limit_uses_the_monthly_window_reset():
    """AC3: ``limitName=monthly`` is unchanged — the monthly reset wins."""
    entry = opencode_usage_chain_entry()
    cfg: dict = {}
    quota_state.set_clock(lambda: _NOW)
    monthly_seconds = 11 * 86400
    await _seed_snapshot(
        entry,
        cfg,
        _windowed_usage_body(
            rolling=(0.0, 2 * 3600),
            weekly=(0.0, 3 * 86400),
            monthly=(99.0, monthly_seconds),
        ),
    )

    info = _reset_info("monthly", entry, cfg)

    assert info is not None
    assert info.window == "monthly"
    margin = provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
    assert info.seconds == pytest.approx(monthly_seconds + margin, abs=5)


@pytest.mark.asyncio
async def test_unknown_limit_name_is_bounded_by_the_soonest_usable_window():
    """AC4: absent/empty/unknown ``limitName`` never yields the monthly value.

    With every window still usable the quarantine is bounded by the soonest
    one (here the rolling window) and recorded as *guessed*, so the
    self-healing probe revalidates it and it is never persisted.
    """
    entry = opencode_usage_chain_entry()
    cfg: dict = {}
    quota_state.set_clock(lambda: _NOW)
    rolling_seconds = 4 * 3600
    await _seed_snapshot(
        entry,
        cfg,
        _windowed_usage_body(
            rolling=(10.0, rolling_seconds),
            weekly=(20.0, 5 * 86400),
            monthly=(30.0, 21 * 86400),
        ),
    )
    margin = provider._USAGE_LIMIT_RESET_MARGIN_SECONDS

    for limit_name in (None, "", "daily", "yearly"):
        info = _reset_info(limit_name, entry, cfg)

        assert info is not None, limit_name
        assert info.window == "rolling", limit_name
        assert info.explicit is False, limit_name
        assert info.seconds == pytest.approx(rolling_seconds + margin, abs=5), limit_name
        # Never the monthly window's 21-day value.
        assert info.seconds < 21 * 86400, limit_name


@pytest.mark.asyncio
async def test_unknown_limit_name_skips_blocking_windows_when_bounding():
    """AC4: a blocking (unhealthy) window is not a usable bound."""
    entry = opencode_usage_chain_entry()
    cfg: dict = {}
    quota_state.set_clock(lambda: _NOW)
    weekly_seconds = 6 * 86400
    await _seed_snapshot(
        entry,
        cfg,
        _windowed_usage_body(
            rolling=(0.0, 3600),
            weekly=(40.0, weekly_seconds),
            monthly=(20.0, 18 * 86400),
            rolling_status="unavailable",
        ),
    )

    info = _reset_info(None, entry, cfg)

    assert info is not None
    assert info.window == "weekly"
    assert info.explicit is False
    margin = provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
    assert info.seconds == pytest.approx(weekly_seconds + margin, abs=5)


@pytest.mark.asyncio
async def test_no_usable_window_retains_the_account_level_reset():
    """AC4 tail: with no usable window the monthly account-level path applies."""
    entry = opencode_usage_chain_entry()
    cfg: dict = {}
    quota_state.set_clock(lambda: _NOW)
    monthly_seconds = 9 * 86400
    await _seed_snapshot(
        entry,
        cfg,
        _windowed_usage_body(
            rolling=(0.0, 0),
            weekly=(100.0, 0),
            monthly=(99.0, monthly_seconds),
            # Every window reports an unhealthy status: no window is *usable*,
            # so the account-level monthly path is the only evidence left.
            rolling_status="unavailable",
            weekly_status="unavailable",
            monthly_status="unavailable",
        ),
    )

    info = _reset_info(None, entry, cfg)

    assert info is not None
    assert info.explicit is True
    assert info.window is None
    margin = provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
    assert info.seconds == pytest.approx(monthly_seconds + margin, abs=5)


@pytest.mark.asyncio
async def test_named_window_with_stale_reset_is_bounded_and_guessed():
    """AC8: contradictory evidence for the named window is bounded, not explicit.

    The snapshot says the weekly window has already reset (a non-positive
    countdown) while upstream still returns a usage-limit 429 for it. The
    quarantine must be capped at the guess cap and marked *guessed* so the
    self-healing probe revalidates it — never a multi-week explicit lockout.
    """
    entry = opencode_usage_chain_entry()
    cfg: dict = {}
    quota_state.set_clock(lambda: _NOW)
    await _seed_snapshot(
        entry,
        cfg,
        _windowed_usage_body(
            rolling=(0.0, 4 * 3600),
            weekly=(99.0, -60),  # reset one minute ago — contradictory
            monthly=(54.0, 24 * 86400),
        ),
    )

    info = _reset_info("weekly", entry, cfg)

    assert info is not None
    assert info.explicit is False
    assert info.window == "weekly"
    margin = provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
    assert info.seconds == pytest.approx(
        provider._USAGE_LIMIT_GUESS_MAX_SECONDS + margin, abs=5
    )
    assert info.seconds < 24 * 86400


@pytest.mark.asyncio
async def test_incident_2026_10_24_weekly_limit_never_waits_for_monthly(
    opencode_usage_chain, caplog
):
    """AC5: reproduce the incident — weekly at 99%, monthly reset 24 days out.

    The quarantine must be bounded by the weekly window (≤ 7 days) and must
    never use the monthly reset.
    """
    cfg = {"provider_cooldown_seconds": 60}
    entry = opencode_usage_chain["providers"][0]
    quota_state.set_clock(lambda: _NOW)
    await _seed_snapshot(
        entry,
        cfg,
        _windowed_usage_body(
            rolling=_INCIDENT_ROLLING,
            weekly=_INCIDENT_WEEKLY,
            monthly=_INCIDENT_MONTHLY,
        ),
    )

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        if provider_cfg.get("name") == "opencode-go-2-deepseek":
            return _gousage_429("Weekly usage limit reached.", "weekly")
        return _ok_response()

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        with caplog.at_level("INFO", logger="llama-proxy.provider"):
            result = await provider.proxy_with_remote_fallback(
                _DummyRequest(), "v1/chat/completions", opencode_usage_chain, cfg
            )

    assert result.status_code == 200
    account = provider._usage_limit_account_key(entry)
    remaining = provider._usage_reset_at[account] - time.time()
    assert remaining <= 7 * 86400
    assert remaining == pytest.approx(_INCIDENT_WEEKLY[1] + 120, abs=5)
    # Not the 24-day monthly reset that caused the incident.
    assert remaining < _INCIDENT_MONTHLY[1]
    # The window that bounded it is recorded for revalidation.
    assert provider._usage_reset_window[account] == "weekly"
    assert account not in provider._usage_reset_guessed
    assert any(
        "window=weekly" in rec.message and "source=snapshot" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_reset_log_line_names_the_window_and_value(opencode_usage_chain, caplog):
    """AC9: the reset log line identifies the window and the value used."""
    cfg = {"provider_cooldown_seconds": 60}
    entry = opencode_usage_chain["providers"][0]
    quota_state.set_clock(lambda: _NOW)
    weekly_seconds = 3 * 86400
    await _seed_snapshot(
        entry, cfg, _windowed_usage_body(weekly=(99.0, weekly_seconds))
    )

    async def _mock_proxy_to_remote(_req, _path, provider_cfg):
        if provider_cfg.get("name") == "opencode-go-2-deepseek":
            return _gousage_429("Weekly usage limit reached.", "WEEKLY")
        return _ok_response()

    with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
        with caplog.at_level("INFO", logger="llama-proxy.provider"):
            await provider.proxy_with_remote_fallback(
                _DummyRequest(), "v1/chat/completions", opencode_usage_chain, cfg
            )

    reset_lines = [
        rec.message for rec in caplog.records if "usage-limit reset source=" in rec.message
    ]
    assert reset_lines
    line = reset_lines[0]
    assert "window=weekly" in line
    assert f"seconds_to_reset={weekly_seconds + 120:.0f}" in line


def test_skip_log_line_names_the_quarantine_window(opencode_usage_chain, caplog):
    """AC9: the ``usage_limit_reset_pending`` skip line reports the window."""
    entry = opencode_usage_chain["providers"][0]
    account = provider._usage_limit_account_key(entry)
    provider._usage_reset_at[account] = time.time() + 3600
    provider._usage_reset_window[account] = "weekly"

    with caplog.at_level("INFO", logger="llama-proxy.provider"):
        resolved = provider.resolve_provider(opencode_usage_chain)

    assert resolved is not None
    assert resolved.get("name") == "opencode-go-deepseek"
    pending = [
        rec.message
        for rec in caplog.records
        if "usage_limit_reset_pending" in rec.message
    ]
    assert pending
    assert "window=weekly" in pending[0]


# ===========================================================================
# Absolute "… It resets at <timestamp> UTC" daily limit (LP-0MVT8SAZE00570J5)
#
# MunitoAI's free tier answers HTTP 429 with an ABSOLUTE daily-reset instant
# (`error.type = "insufficient_quota"`, `error.code =
# "free_daily_limit_exceeded"`, "… It resets at 2026-10-30 00:00 UTC."). The
# proxy recognized neither the markers nor the timestamp shape, so it fell
# through to the generic per-entry cooldown and re-probed a dead account every
# ~60 s for the rest of the UTC day. These tests pin the detection, the
# absolute-timestamp parsing (fixed clock), the period bound and the
# past/unparseable/no-reset fallbacks.
# ===========================================================================

#: Frozen clock for the absolute-timestamp parser: the 2026-10-29 09:39 log
#: line that opened the incident.
_RESET_NOW = datetime(2026, 10, 29, 9, 40, 0, tzinfo=UTC)

#: Seconds from ``_RESET_NOW`` to the advertised ``2026-10-30 00:00 UTC``.
_RESET_AT_SECONDS = 14 * 3600 + 20 * 60


class _FrozenResetDateTime(datetime):
    """``datetime`` with a frozen ``now`` (the parser's injectable clock)."""

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return _RESET_NOW
        return _RESET_NOW.astimezone(tz)


def _munitoai_free_entry() -> dict:
    """The free-tier MunitoAI entry the incident was observed on."""
    return {
        "name": "munitoai-free",
        "type": "remote",
        "provider": "MunitoAI",
        "endpoint": "https://api.munito.ai/inference",
        "api_key_env": "MUNITOAI_FREE_API_KEY",
        "model": "deepseek-ai/deepseek-v4-flash",
        "respect_ratelimit_headers": True,
    }


def _munitoai_daily_limit_body(reset_at: str = "2026-10-30 00:00 UTC") -> str:
    """The verbatim MunitoAI free-tier daily-limit 429 body (2026-10-29 log).

    ``reset_at`` is substituted so a test can place the advertised instant in
    the past or the far future.
    """
    return json.dumps({
        "error": {
            "message": (
                "The daily free limit of deepseek-ai/deepseek-v4-flash is "
                f"reached. It resets at {reset_at}. Add credits to continue "
                "now."
            ),
            "type": "insufficient_quota",
            "param": None,
            "code": "free_daily_limit_exceeded",
        }
    })


def _daily_limit_response(body: str) -> Response:
    return Response(status_code=429, content=body.encode("utf-8"))


def _daily_reset_info(body: str, provider_cfg: dict | None = None, config=None):
    """``_usage_limit_reset_info`` under the frozen clock."""
    with patch("proxy.provider.datetime", _FrozenResetDateTime):
        return provider._usage_limit_reset_info(
            _daily_limit_response(body),
            body,
            config,
            provider_cfg=provider_cfg,
        )


def test_parse_resets_at_timestamp_formats():
    """AC2: all four supported absolute-timestamp shapes parse to the same
    seconds-until-reset against a fixed clock."""
    cases = {
        "It resets at 2026-10-30 00:00 UTC.": _RESET_AT_SECONDS,
        "It resets at 2026-10-30 00:00:30 UTC.": _RESET_AT_SECONDS + 30,
        "It resets at 2026-10-30T00:00:00Z.": _RESET_AT_SECONDS,
        "It resets at 2026-10-30T02:00:00+02:00.": _RESET_AT_SECONDS,
        "It resets at 2026-10-30T03:00:00+0300.": _RESET_AT_SECONDS,
    }
    for message, expected in cases.items():
        assert provider._parse_resets_at_timestamp(
            message, now=_RESET_NOW
        ) == pytest.approx(expected), message


def test_parse_resets_at_timestamp_rejects_unusable_values():
    """AC4: past instants and timestamps without timezone evidence are not
    usable evidence (the caller then falls back to the guessed period)."""
    for message in (
        "It resets at 2026-10-29 09:00 UTC.",   # in the past
        "It resets at 2026-10-29 09:40 UTC.",   # exactly now
        "It resets at 2026-10-30 00:00.",       # no timezone evidence
        "It resets at tomorrow midnight.",
        "",
    ):
        assert (
            provider._parse_resets_at_timestamp(message, now=_RESET_NOW) is None
        ), message


def test_munitoai_daily_limit_429_uses_the_absolute_timestamp(caplog):
    """AC1+AC2+AC3: the verbatim MunitoAI body takes the usage-limit reset
    path with the timestamp + the 2-minute margin, logged as
    ``source=timestamp``."""
    body = _munitoai_daily_limit_body()
    entry = _munitoai_free_entry()
    with caplog.at_level("INFO", logger="llama-proxy.provider"):
        info = _daily_reset_info(body, provider_cfg=entry)

    assert info is not None
    assert info.explicit is True
    assert info.limit_name == "daily"
    assert info.window is None
    assert info.seconds == pytest.approx(
        _RESET_AT_SECONDS + provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
    )
    with patch("proxy.provider.datetime", _FrozenResetDateTime):
        assert provider._usage_limit_reset_seconds(
            _daily_limit_response(body), body
        ) == pytest.approx(info.seconds)

    reset_lines = [
        rec.message for rec in caplog.records if "usage-limit reset source=" in rec.message
    ]
    assert reset_lines, "the reset source must be logged"
    assert "source=timestamp" in reset_lines[0]
    assert "provider=munitoai-free" in reset_lines[0]
    assert f"seconds_to_reset={info.seconds:.0f}" in reset_lines[0]
    assert "limit=daily" in reset_lines[0]


def test_insufficient_quota_markers_are_detected_top_level_and_nested():
    """AC1: the markers are read from the top level AND from a nested ``error``
    dict, and matching is case-insensitive — the code alone is enough for
    gateways that omit ``type``."""
    reset_at = "2026-10-30 00:00 UTC"
    payloads = [
        # top-level type (observed shape family)
        {
            "type": "insufficient_quota",
            "message": f"Daily limit reached. It resets at {reset_at}.",
        },
        # nested type, upper-case
        {
            "error": {
                "type": "INSUFFICIENT_QUOTA",
                "message": f"Daily limit reached. It resets at {reset_at}.",
            }
        },
        # code only, nested and upper-case (no type at all)
        {
            "error": {
                "code": "FREE_DAILY_LIMIT_EXCEEDED",
                "message": f"Daily limit reached. It resets at {reset_at}.",
            }
        },
        # code only, top-level, with a generic wrapper type
        {
            "type": "error",
            "code": "free_daily_limit_exceeded",
            "message": f"Daily limit reached. It resets at {reset_at}.",
        },
    ]
    for payload in payloads:
        body = json.dumps(payload)
        info = _daily_reset_info(body)
        assert info is not None, payload
        assert info.explicit is True, payload
        assert info.seconds == pytest.approx(
            _RESET_AT_SECONDS + provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
        ), payload


def test_explicit_duration_wins_over_the_absolute_timestamp():
    """Constraint: an explicit ``Resets in …`` duration keeps precedence over
    an absolute timestamp in the same message."""
    body = json.dumps({
        "error": {
            "type": "insufficient_quota",
            "message": (
                "Daily free limit reached. Resets in 1 hour. "
                "It resets at 2026-10-30 00:00 UTC."
            ),
        }
    })
    info = _daily_reset_info(body)
    assert info is not None
    assert info.seconds == pytest.approx(
        3600 + provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
    )


def test_absolute_timestamp_is_bounded_to_the_named_daily_period():
    """AC4: hard evidence is bounded by the period the payload names, so a
    far-future timestamp cannot lock the account out for weeks."""
    body = _munitoai_daily_limit_body("2026-11-30 00:00 UTC")  # ~32 days out
    info = _daily_reset_info(body)
    assert info is not None
    assert info.explicit is True
    assert info.limit_name == "daily"
    assert info.seconds == pytest.approx(
        provider._PERIOD_DEFAULT_SECONDS["daily"]
        + provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
    )
    assert info.seconds < 31 * 24 * 3600


def test_past_timestamp_falls_back_to_the_guessed_daily_period():
    """AC4: a timestamp in the past is not evidence — the quarantine becomes
    the bounded GUESS (probe-revalidated, never persisted)."""
    body = _munitoai_daily_limit_body("2026-10-29 09:00 UTC")
    info = _daily_reset_info(body)
    assert info is not None
    assert info.explicit is False
    assert info.limit_name == "daily"
    assert info.seconds == pytest.approx(
        min(
            provider._PERIOD_DEFAULT_SECONDS["daily"],
            provider._USAGE_LIMIT_GUESS_MAX_SECONDS,
        )
        + provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
    )

    account = provider._usage_limit_account_key(_munitoai_free_entry())
    provider._set_usage_limit_quarantine(account, info)
    assert account in provider._usage_reset_guessed


def test_unparseable_timestamp_falls_back_to_the_guessed_daily_period():
    """AC4: an unparseable timestamp falls back to the guessed period."""
    body = _munitoai_daily_limit_body("tomorrow at midnight")
    info = _daily_reset_info(body)
    assert info is not None
    assert info.explicit is False
    assert info.limit_name == "daily"
    assert info.seconds == pytest.approx(
        provider._PERIOD_DEFAULT_SECONDS["daily"]
        + provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
    )


def test_insufficient_quota_without_any_reset_information_is_guessed():
    """AC4: an out-of-credit body with no reset information at all is
    quarantined as a bounded guess rather than re-probed every ~60 s."""
    body = json.dumps({
        "error": {
            "type": "insufficient_quota",
            "message": "You exceeded your current quota.",
        }
    })
    info = _daily_reset_info(body)
    assert info is not None
    assert info.explicit is False
    assert info.seconds == pytest.approx(
        provider._usage_limit_guess_cap_seconds({})
        + provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
    )

    account = provider._usage_limit_account_key(_munitoai_free_entry())
    provider._set_usage_limit_quarantine(account, info)
    assert account in provider._usage_reset_guessed


def test_non_quota_responses_stay_out_of_the_usage_limit_path():
    """AC6: a non-429 carrying ``insufficient_quota`` and a 429 carrying only
    ``rate_limit_exceeded`` keep their pre-change handling."""
    body = _munitoai_daily_limit_body()
    # 200 with the quota body is not an error at all.
    assert provider._usage_limit_reset_info(
        Response(status_code=200, content=body.encode()), body
    ) is None
    # 429 with only the per-minute rate-limit marker is not a daily reset.
    rate_body = json.dumps({
        "error": {
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
            "message": "Too many requests. It resets at 2026-10-30 00:00 UTC.",
        }
    })
    assert provider._usage_limit_reset_info(
        Response(status_code=429, content=rate_body.encode()), rate_body
    ) is None


def test_absolute_timestamp_quarantine_survives_a_state_restore(tmp_path, monkeypatch):
    """AC3: the reset is recorded through ``_set_usage_limit_quarantine`` and
    survives a ``provider-state.json`` round trip as an explicit entry."""
    monkeypatch.setenv(
        provider._PROVIDER_STATE_FILE_ENV, str(tmp_path / "provider-state.json")
    )
    info = _daily_reset_info(_munitoai_daily_limit_body(), provider_cfg=_munitoai_free_entry())
    assert info is not None
    account = provider._usage_limit_account_key(_munitoai_free_entry())
    provider._set_usage_limit_quarantine(account, info)

    state_file = tmp_path / "provider-state.json"
    assert state_file.exists(), "the quarantine must be persisted"

    provider._usage_reset_at.clear()
    provider._usage_reset_guessed.clear()
    provider._usage_reset_window.clear()
    restored_cooldowns, restored_quarantine = provider.load_provider_state(state_file)

    assert restored_cooldowns == 0
    assert restored_quarantine == 1
    assert provider._usage_reset_remaining(account) == pytest.approx(
        info.seconds, abs=5
    )
    assert account not in provider._usage_reset_guessed
    assert account not in provider._usage_reset_window
