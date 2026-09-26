"""Tests for exhaustion quarantine diagnostics (LP-0MU56ZFVD001LP0H).

Covers the child work item's acceptance criteria:

- AC1: a usage-limit quarantined account is surfaced in
  ``unavailable_providers`` and in ``diagnostics`` with a
  ``usage_limit_reset`` status plus its remaining reset seconds.
- AC2: a provider that is both quarantined and outside its
  ``available_times`` window is reported once, with ``usage_limit_reset``
  (not ``outside_time_window``) as the reason.
- AC3: ``retry_after`` is > 0 and derived from the real availability
  (quarantine included).
- AC4: quarantine-only, quarantine+window and quarantine+cooldown mixes.
"""

import json
import time
from datetime import UTC, datetime
from unittest.mock import patch

import proxy.provider as provider
import pytest
from fastapi import Response


class _FixedDateTime(datetime):
    """``datetime`` subclass with a patchable class-level ``now``.

    Keeps ``fromtimestamp`` (and the ISO formatting of reset times) real
    while making ``available_times`` window checks deterministic.
    """

    fixed_now: datetime = datetime(2026, 1, 1, 20, 0, 0, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):
        return cls.fixed_now


class _DummyRequest:
    """Minimal request stub (mirrors the other provider fallback tests)."""

    def __init__(self, body: bytes = b'{"model":"test"}'):
        self._body = body
        self.headers = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


@pytest.fixture(autouse=True)
def _reset_provider_state():
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    yield
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()


def _provider(
    name: str,
    *,
    brand: str = "acme",
    endpoint: str = "https://acme.example/v1",
    api_key_env: str | None = None,
    available_times=None,
) -> dict:
    cfg: dict = {
        "name": name,
        "type": "remote",
        "provider": brand,
        "endpoint": endpoint,
        "model": "m1",
    }
    if api_key_env:
        cfg["api_key_env"] = api_key_env
    if available_times is not None:
        cfg["available_times"] = available_times
    return cfg


def _quarantine(cfg: dict, seconds: float = 3600.0) -> str:
    """Quarantine *cfg*'s account for *seconds* and return the account key."""
    key = provider._usage_limit_account_key(cfg)
    provider._usage_reset_at[key] = time.time() + seconds
    return key


async def _exhaust(model: dict) -> Response:
    """Run the remote fallback chain to exhaustion with a mocked upstream."""
    async def _never_called(_req, _path, _cfg):  # pragma: no cover - guard
        raise AssertionError("upstream must not be contacted for blocked providers")

    cfg = {"provider_cooldown_seconds": 60}
    with patch("proxy.server.proxy_to_remote", _never_called):
        return await provider.proxy_with_remote_fallback(
            _DummyRequest(), "v1/chat/completions", model, cfg
        )


def _statuses(body: dict) -> dict:
    return {d.get("provider"): d.get("status") for d in body.get("diagnostics", [])}


# ---------------------------------------------------------------------------
# Unit: _log_exhausted_providers
# ---------------------------------------------------------------------------


class TestLogExhaustedProvidersQuarantine:
    def test_includes_quarantined_account_and_reset_seconds(self):
        p1 = _provider("q1", api_key_env="K1")
        p2 = _provider("p2", api_key_env="K2", endpoint="https://beta.example/v1")
        _quarantine(p1, 900)

        result = provider._log_exhausted_providers({"providers": [p1, p2]}, "v1/test")

        assert "q1" in result
        assert 880 <= result["q1"] <= 900
        assert "p2" not in result

    def test_expired_quarantine_not_included(self):
        p1 = _provider("q1", api_key_env="K1")
        provider._usage_reset_at[provider._usage_limit_account_key(p1)] = time.time() - 5

        result = provider._log_exhausted_providers({"providers": [p1]}, "v1/test")

        assert "q1" not in result


# ---------------------------------------------------------------------------
# Unit: _providers_outside_window
# ---------------------------------------------------------------------------


class TestProvidersOutsideWindowQuarantine:
    def test_quarantined_provider_excluded_from_window_skips(self):
        with patch("proxy.provider.datetime", _FixedDateTime):
            p1 = _provider("q1", api_key_env="K1", available_times=["09:00-17:00"])
            p2 = _provider(
                "w2", api_key_env="K2", endpoint="https://beta.example/v1",
                available_times=["09:00-17:00"],
            )
            _quarantine(p1, 900)

            result = provider._providers_outside_window({"providers": [p1, p2]})

        assert {r["name"] for r in result} == {"w2"}


# ---------------------------------------------------------------------------
# Unit: _compute_retry_after
# ---------------------------------------------------------------------------


class TestComputeRetryAfterQuarantine:
    def test_quarantine_only_candidate_is_surfaced(self):
        p1 = _provider("q1", api_key_env="K1")
        _quarantine(p1, 4321)

        result = provider._compute_retry_after({}, model_config={"providers": [p1]})

        assert result > 0
        assert 4300 <= result <= 4321


# ---------------------------------------------------------------------------
# Integration: exhaustion payloads
# ---------------------------------------------------------------------------


class TestExhaustionPayloadQuarantine:
    @pytest.mark.asyncio
    async def test_quarantine_only_surfaces_account_and_status(self):
        p1 = _provider("q1", api_key_env="K1")
        p2 = _provider("q2", api_key_env="K2", endpoint="https://beta.example/v1")
        _quarantine(p1, 3600)
        _quarantine(p2, 7200)

        result = await _exhaust({"providers": [p1, p2]})

        assert result.status_code == 503
        body = json.loads(result.body)
        assert body["error"] == "All providers exhausted"
        assert body["unavailable_providers"]["q1"] >= 3500
        assert body["unavailable_providers"]["q2"] >= 7100
        statuses = _statuses(body)
        assert statuses["q1"] == "usage_limit_reset"
        assert statuses["q2"] == "usage_limit_reset"
        q1_diag = next(d for d in body["diagnostics"] if d["provider"] == "q1")
        assert q1_diag["reset_in"] >= 3500
        assert body["retry_after"] > 0
        assert result.headers.get("retry-after") == str(body["retry_after"])

    @pytest.mark.asyncio
    async def test_quarantine_plus_window_reports_quarantine_once(self):
        with patch("proxy.provider.datetime", _FixedDateTime):
            p1 = _provider("q1", api_key_env="K1", available_times=["09:00-17:00"])
            p2 = _provider(
                "w2", api_key_env="K2", endpoint="https://beta.example/v1",
                available_times=["09:00-17:00"],
            )
            _quarantine(p1, 3600)

            result = await _exhaust({"providers": [p1, p2]})

        body = json.loads(result.body)
        # Quarantine (not the window) is the reason recorded for q1, exactly once.
        statuses = _statuses(body)
        assert statuses["q1"] == "usage_limit_reset"
        assert statuses["w2"] == "outside_time_window"
        assert sum(1 for d in body["diagnostics"] if d["provider"] == "q1") == 1
        assert "q1" in body["unavailable_providers"]
        # A quarantined provider makes this a generic exhaustion, not a
        # "scheduled time window" exhaustion.
        assert body["error"] == "All providers exhausted"

    @pytest.mark.asyncio
    async def test_quarantine_plus_cooldown_reports_both(self):
        p1 = _provider("q1", api_key_env="K1")
        p2 = _provider("c2", api_key_env="K2", endpoint="https://beta.example/v1")
        _quarantine(p1, 3600)
        provider.mark_provider_unavailable("c2", 120.0)

        result = await _exhaust({"providers": [p1, p2]})

        body = json.loads(result.body)
        assert body["unavailable_providers"]["q1"] >= 3500
        assert body["unavailable_providers"]["c2"] >= 100
        statuses = _statuses(body)
        assert statuses["q1"] == "usage_limit_reset"

    @pytest.mark.asyncio
    async def test_window_only_without_quarantine_keeps_window_message(self):
        """Regression: the window-specific message still fires when the
        window is the sole blocker (no quarantine/cooldown)."""
        with patch("proxy.provider.datetime", _FixedDateTime):
            p1 = _provider("w1", api_key_env="K1", available_times=["09:00-17:00"])
            p2 = _provider(
                "w2", api_key_env="K2", endpoint="https://beta.example/v1",
                available_times=["09:00-17:00"],
            )

            result = await _exhaust({"providers": [p1, p2]})

        body = json.loads(result.body)
        assert "scheduled time window" in body["error"]
        assert _statuses(body)["w1"] == "outside_time_window"
