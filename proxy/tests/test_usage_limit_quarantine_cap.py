"""Tests for bounded, self-healing usage-limit quarantine
(LP-0MUQTCMW2001VXH8).

The gateway can return HTTP 429 ``GoUsageLimitError`` bodies that carry only
``metadata.limitName`` (e.g. ``monthly``) and no explicit ``Resets in ...``
duration. Before this fix the proxy fell back to the full period — up to 30
days — and persisted it, silently removing healthy accounts from the routing
chain across restarts.

This module covers:

- AC1: a period-name-only (guessed) quarantine is capped at the configured
  short window (≤ 24 h by default), never the full 30-day period.
- AC2: an explicit ``Resets in ...`` duration is still honoured in full.
- AC3: a guessed quarantine is not persisted as a long-lived expiry and is
  re-validated by a probe, so a recovered account is reintroduced without a
  proxy restart.
- AC4: ``POST /admin/clear-usage-limit`` clears a pending quarantine at
  runtime.
- AC5: once the probe succeeds, the recovered account is routed to within the
  same request cycle.
"""

import json
import time
from unittest.mock import patch

import proxy.provider as provider
import proxy.server as server
import pytest
from fastapi import Response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyRequest:
    """Minimal request stub (mirrors the other provider fallback tests)."""

    def __init__(self, body: bytes = b'{"model":"test"}'):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


class _JsonRequest(_DummyRequest):
    """Request stub whose ``json()`` returns a fixed payload."""

    def __init__(self, payload=None):
        super().__init__()
        self._payload = payload if payload is not None else {}

    async def json(self):
        return self._payload


def _gousage_429(message: str, limit_name: str = "monthly") -> Response:
    """Build the observed duration-less opencode 429 GoUsageLimitError body."""
    body = json.dumps({
        "type": "error",
        "error": {"type": "GoUsageLimitError", "message": message},
        "metadata": {"limitName": limit_name},
    })
    return Response(
        status_code=429,
        content=body.encode("utf-8"),
        media_type="application/json",
    )


def _ok() -> Response:
    return Response(
        content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
        status_code=200,
        media_type="application/json",
    )


def _account(provider_cfg: dict) -> str:
    return provider._usage_limit_account_key(provider_cfg)


_TWO_GO_CHAIN = {
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
            "name": "opencode-go-3-deepseek",
            "type": "remote",
            "provider": "opencode-go",
            "endpoint": "https://opencode.ai/zen/go",
            "api_key_env": "OPENCODE_3_API_KEY",
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
    "aliases": ["plan*"],
}


@pytest.fixture(autouse=True)
def _reset_probe_transport():
    """Ensure no test leaks a probe transport stub into the next."""
    provider._usage_limit_probe_transport = None
    provider._usage_limit_last_probe.clear()
    provider._usage_reset_guessed.clear()
    yield
    provider._usage_limit_probe_transport = None
    provider._usage_limit_last_probe.clear()
    provider._usage_reset_guessed.clear()


# ---------------------------------------------------------------------------
# AC1 / AC2 — cap guessed durations, honour explicit ones
# ---------------------------------------------------------------------------


class TestResetInfoBounding:
    def test_guessed_monthly_is_capped_at_default(self):
        resp = _gousage_429("Go usage limit exceeded", "monthly")

        info = provider._usage_limit_reset_info(resp, resp.body.decode())

        assert info is not None
        assert info.explicit is False
        assert info.limit_name == "monthly"
        # Capped at the 24h default, not the 30-day period.
        assert info.seconds == pytest.approx(
            provider._USAGE_LIMIT_GUESS_MAX_SECONDS
            + provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
        )
        assert info.seconds < 30 * 24 * 3600

    def test_guessed_weekly_is_capped_at_default(self):
        resp = _gousage_429("Go usage limit exceeded", "weekly")

        info = provider._usage_limit_reset_info(resp, resp.body.decode())

        assert info is not None
        assert info.explicit is False
        assert info.seconds == pytest.approx(
            provider._USAGE_LIMIT_GUESS_MAX_SECONDS
            + provider._USAGE_LIMIT_RESET_MARGIN_SECONDS
        )

    def test_guessed_daily_is_not_extended(self):
        """A daily guess (24h) is naturally within the 24h cap."""
        resp = _gousage_429("Go usage limit exceeded", "daily")

        info = provider._usage_limit_reset_info(resp, resp.body.decode())

        assert info is not None
        assert info.seconds == pytest.approx(24 * 3600 + 120)

    def test_guessed_cap_is_configurable(self):
        resp = _gousage_429("Go usage limit exceeded", "monthly")

        info = provider._usage_limit_reset_info(
            resp,
            resp.body.decode(),
            {"server": {"usage_limit_guess_cap_seconds": 3600}},
        )

        assert info is not None
        assert info.seconds == pytest.approx(3600 + 120)

    def test_guessed_cap_env_override(self, monkeypatch):
        monkeypatch.setenv(provider._USAGE_LIMIT_GUESS_CAP_ENV, "1800")
        resp = _gousage_429("Go usage limit exceeded", "weekly")

        info = provider._usage_limit_reset_info(resp, resp.body.decode())

        assert info is not None
        assert info.seconds == pytest.approx(1800 + 120)

    def test_explicit_duration_is_honoured_in_full(self):
        resp = _gousage_429(
            "Weekly usage limit reached. Resets in 22hr 43min.", "weekly"
        )

        info = provider._usage_limit_reset_info(resp, resp.body.decode())

        assert info is not None
        assert info.explicit is True
        assert info.seconds == pytest.approx(22 * 3600 + 43 * 60 + 120)

    def test_explicit_duration_not_capped_by_guess_cap(self):
        """The cap applies only to guesses; an explicit 30d is preserved."""
        resp = _gousage_429("Monthly usage limit reached. Resets in 30 days.", "monthly")

        info = provider._usage_limit_reset_info(resp, resp.body.decode())

        assert info is not None
        assert info.explicit is True
        assert info.seconds == pytest.approx(30 * 24 * 3600 + 120)

    def test_backwards_compatible_seconds_wrapper(self):
        resp = _gousage_429("Go usage limit exceeded", "monthly")

        seconds = provider._usage_limit_reset_seconds(resp, resp.body.decode())

        assert seconds == pytest.approx(
            provider._USAGE_LIMIT_GUESS_MAX_SECONDS + 120
        )

    def test_no_duration_and_no_period_is_none(self):
        body = json.dumps({
            "error": {"type": "GoUsageLimitError", "message": "Quota exhausted."},
            "metadata": {},
        })
        resp = Response(status_code=429, content=body.encode())

        assert provider._usage_limit_reset_info(resp, body) is None


# ---------------------------------------------------------------------------
# AC1 — end-to-end bounded quarantine + guessed marker
# ---------------------------------------------------------------------------


class TestGuessedQuarantineEndToEnd:
    @pytest.mark.asyncio
    async def test_durationless_429_quarantines_for_capped_window(self):
        call_count = 0

        async def _mock_proxy_to_remote(_req, _path, provider_cfg):
            nonlocal call_count
            call_count += 1
            if provider_cfg.get("name") == "opencode-go-2-deepseek":
                return _gousage_429("Go usage limit exceeded", "monthly")
            return _ok()

        with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
            result = await provider.proxy_with_remote_fallback(
                _DummyRequest(),
                "v1/chat/completions",
                _TWO_GO_CHAIN,
                {"provider_cooldown_seconds": 60},
            )

        assert result.status_code == 200
        account = _account(_TWO_GO_CHAIN["providers"][0])
        remaining = provider._usage_reset_at[account] - time.time()
        assert remaining == pytest.approx(
            provider._USAGE_LIMIT_GUESS_MAX_SECONDS + 120, abs=5
        )
        # The account is marked as a guess (soft, re-checkable).
        assert account in provider._usage_reset_guessed

    @pytest.mark.asyncio
    async def test_explicit_429_quarantines_for_full_window(self):
        async def _mock_proxy_to_remote(_req, _path, provider_cfg):
            if provider_cfg.get("name") == "opencode-go-2-deepseek":
                return _gousage_429(
                    "Weekly usage limit reached. Resets in 22hr 43min.", "weekly"
                )
            return _ok()

        with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
            result = await provider.proxy_with_remote_fallback(
                _DummyRequest(),
                "v1/chat/completions",
                _TWO_GO_CHAIN,
                {"provider_cooldown_seconds": 60},
            )

        assert result.status_code == 200
        account = _account(_TWO_GO_CHAIN["providers"][0])
        remaining = provider._usage_reset_at[account] - time.time()
        assert remaining == pytest.approx(22 * 3600 + 43 * 60 + 120, abs=5)
        # An explicit duration is hard evidence, not a guess.
        assert account not in provider._usage_reset_guessed


# ---------------------------------------------------------------------------
# AC3 — guessed quarantines are not persisted; expiration clears the marker
# ---------------------------------------------------------------------------


class TestGuessedQuarantinePersistence:
    def test_guessed_quarantine_is_not_persisted(self):
        account = "OPENCODE_2_API_KEY@https://opencode.ai/zen/go"
        provider._usage_reset_at[account] = time.time() + 3600
        provider._usage_reset_guessed.add(account)

        provider.save_provider_state()

        raw = json.loads(
            provider.default_provider_state_file().read_text(encoding="utf-8")
        )
        assert account not in raw["usage_reset_at"]

    def test_explicit_quarantine_is_persisted(self):
        account = "OPENCODE_2_API_KEY@https://opencode.ai/zen/go"
        provider._usage_reset_at[account] = time.time() + 3600
        # Not present in _usage_reset_guessed -> treated as explicit/hard.

        provider.save_provider_state()

        raw = json.loads(
            provider.default_provider_state_file().read_text(encoding="utf-8")
        )
        assert account in raw["usage_reset_at"]

    def test_load_clears_stale_guessed_markers(self):
        provider._usage_reset_guessed.add("leftover@domain")
        provider._usage_reset_at["leftover@domain"] = time.time() + 3600
        save_path = provider.default_provider_state_file()

        provider.save_provider_state(save_path)
        # Wipe in-memory and reload: leaked guessed markers must not survive.
        provider._usage_reset_at.clear()
        provider.load_provider_state(save_path)

        assert provider._usage_reset_guessed == set()
        assert provider._usage_reset_at == {}

    def test_expired_guessed_entry_drops_marker(self):
        account = "OPENCODE_2_API_KEY@https://opencode.ai/zen/go"
        provider._usage_reset_at[account] = time.time() - 1
        provider._usage_reset_guessed.add(account)

        assert provider._usage_reset_remaining(account) == 0

        assert account not in provider._usage_reset_at
        assert account not in provider._usage_reset_guessed


# ---------------------------------------------------------------------------
# AC3 / AC5 — self-healing probe clears quarantine and routes same cycle
# ---------------------------------------------------------------------------


class TestSelfHealingProbe:
    @pytest.mark.asyncio
    async def test_probe_success_reintroduces_account_same_cycle(self):
        chain = _TWO_GO_CHAIN
        for p in chain["providers"][:2]:
            acct = _account(p)
            provider._usage_reset_at[acct] = time.time() + 3600
            provider._usage_reset_guessed.add(acct)

        probed: list[str] = []

        async def _probe_ok(provider_cfg):
            probed.append(provider_cfg["name"])
            return True

        provider._usage_limit_probe_transport = _probe_ok

        contacted: list[str] = []

        async def _mock_proxy_to_remote(_req, _path, provider_cfg):
            contacted.append(provider_cfg.get("name"))
            return _ok()

        with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
            result = await provider.proxy_with_remote_fallback(
                _DummyRequest(),
                "v1/chat/completions",
                chain,
                {"provider_cooldown_seconds": 60},
            )

        assert result.status_code == 200
        # Both soft-quarantined accounts were probed and cleared, and the
        # first (opencode-go-2) was routed to within this one request cycle.
        assert set(probed) == {"opencode-go-2-deepseek", "opencode-go-3-deepseek"}
        assert contacted == ["opencode-go-2-deepseek"]
        assert provider._usage_reset_at == {}
        assert provider._usage_reset_guessed == set()

    @pytest.mark.asyncio
    async def test_probe_failure_keeps_quarantine_and_falls_through(self):
        chain = _TWO_GO_CHAIN
        acct = _account(chain["providers"][0])
        provider._usage_reset_at[acct] = time.time() + 3600
        provider._usage_reset_guessed.add(acct)

        async def _probe_fail(_provider_cfg):
            return False

        provider._usage_limit_probe_transport = _probe_fail

        contacted: list[str] = []

        async def _mock_proxy_to_remote(_req, _path, provider_cfg):
            contacted.append(provider_cfg.get("name"))
            return _ok()

        with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
            result = await provider.proxy_with_remote_fallback(
                _DummyRequest(),
                "v1/chat/completions",
                chain,
                {"provider_cooldown_seconds": 60},
            )

        assert result.status_code == 200
        # go-2 kept its quarantine, so the chain routed to go-3.
        assert contacted == ["opencode-go-3-deepseek"]
        assert acct in provider._usage_reset_at
        assert acct in provider._usage_reset_guessed

    @pytest.mark.asyncio
    async def test_explicit_quarantine_is_not_probed(self):
        """Only guessed (soft) quarantines are re-validated by a probe."""
        chain = _TWO_GO_CHAIN
        acct = _account(chain["providers"][0])
        provider._usage_reset_at[acct] = time.time() + 3600
        # Not in _usage_reset_guessed -> explicit/hard.

        probed: list[str] = []

        async def _probe_ok(provider_cfg):
            probed.append(provider_cfg["name"])
            return True

        provider._usage_limit_probe_transport = _probe_ok

        async def _mock_proxy_to_remote(_req, _path, provider_cfg):
            return _ok()

        with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
            await provider.proxy_with_remote_fallback(
                _DummyRequest(),
                "v1/chat/completions",
                chain,
                {"provider_cooldown_seconds": 60},
            )

        assert probed == []

    @pytest.mark.asyncio
    async def test_probe_interval_gates_repeated_probes(self):
        chain = _TWO_GO_CHAIN
        acct = _account(chain["providers"][0])
        provider._usage_reset_at[acct] = time.time() + 3600
        provider._usage_reset_guessed.add(acct)

        probed = 0

        async def _probe_fail(_provider_cfg):
            nonlocal probed
            probed += 1
            return False

        provider._usage_limit_probe_transport = _probe_fail

        async def _mock_proxy_to_remote(_req, _path, provider_cfg):
            return _ok()

        with patch("proxy.server.proxy_to_remote", _mock_proxy_to_remote):
            await provider.proxy_with_remote_fallback(
                _DummyRequest(), "v1/chat/completions", chain,
                {"provider_cooldown_seconds": 60},
            )
            await provider.proxy_with_remote_fallback(
                _DummyRequest(), "v1/chat/completions", chain,
                {"provider_cooldown_seconds": 60},
            )

        # Second request is within the probe interval -> no extra probe.
        assert probed == 1


# ---------------------------------------------------------------------------
# AC4 — runtime admin endpoint
# ---------------------------------------------------------------------------


class TestAdminClearUsageLimit:
    @pytest.mark.asyncio
    async def test_clears_all_entries_by_default(self):
        from proxy.handlers import admin_clear_usage_limit

        provider._usage_reset_at["a@x"] = time.time() + 3600
        provider._usage_reset_guessed.add("a@x")
        provider._usage_reset_at["b@y"] = time.time() + 3600

        result = await admin_clear_usage_limit(_JsonRequest({}))

        assert result["status"] == "success"
        assert set(result["cleared"]) == {"a@x", "b@y"}
        assert result["count"] == 2
        assert provider._usage_reset_at == {}
        assert provider._usage_reset_guessed == set()

    @pytest.mark.asyncio
    async def test_clears_only_named_account(self):
        from proxy.handlers import admin_clear_usage_limit

        provider._usage_reset_at["a@x"] = time.time() + 3600
        provider._usage_reset_guessed.add("a@x")
        provider._usage_reset_at["b@y"] = time.time() + 3600

        result = await admin_clear_usage_limit(_JsonRequest({"account": "a@x"}))

        assert result["cleared"] == ["a@x"]
        assert "b@y" in provider._usage_reset_at
        assert "a@x" not in provider._usage_reset_at
        assert "a@x" not in provider._usage_reset_guessed

    @pytest.mark.asyncio
    async def test_clear_unknown_account_is_a_noop(self):
        from proxy.handlers import admin_clear_usage_limit

        provider._usage_reset_at["b@y"] = time.time() + 3600

        result = await admin_clear_usage_limit(_JsonRequest({"account": "nope"}))

        assert result["cleared"] == []
        assert "b@y" in provider._usage_reset_at

    @pytest.mark.asyncio
    async def test_clear_persists_state(self):
        from proxy.handlers import admin_clear_usage_limit

        account = "OPENCODE_2_API_KEY@https://opencode.ai/zen/go"
        provider._usage_reset_at[account] = time.time() + 3600
        provider._usage_reset_guessed.add(account)

        await admin_clear_usage_limit(_JsonRequest({}))

        raw = json.loads(
            provider.default_provider_state_file().read_text(encoding="utf-8")
        )
        assert account not in raw["usage_reset_at"]

    def test_route_is_registered(self):
        paths = {getattr(route, "path", None) for route in server.app.routes}
        assert "/admin/clear-usage-limit" in paths
