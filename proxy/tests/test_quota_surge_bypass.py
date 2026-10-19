"""Tests for the final-week ``available_times`` surge bypass
(LP-0MVF7KJN2000ASE8, epic LP-0MVF4FIP1001RD8P feature F4).

Goal: during the final ``server.quota_surge_days`` (default 7) before an
``opencode-*`` account's monthly reset, that account's ``available_times``
restriction is lifted so its remaining allowance can be spent at any UTC hour —
while bypassing **nothing else**.

Every test seeds quota state through the public poller seam
(``quota_state.set_quota_transport`` + ``refresh_quota_accounts``) with a frozen
clock, freezes ``proxy.provider.datetime`` at 02:00 UTC (outside the configured
``10:00-00:00`` window), and asserts observable behaviour through the public
routing API — eligibility, resolution, the real fallback dispatch cycle, the
retry-after derivation and the log lines.

Covered acceptance criteria:

- AC5: with ``resetsAt`` 6 days away and monthly 50%, an entry configured with
  ``available_times: ["10:00-00:00"]`` is eligible at 02:00 UTC (previously
  ineligible) and the eligibility is logged with ``reason=quota_surge``; with
  ``resetsAt`` 8 days away the same entry stays ineligible.
- AC5b: the bypass clears when the entry becomes exhausted or its reset passes,
  and ``_seconds_until_next_window`` / ``_compute_retry_after`` stay coherent
  while surge is active (no phantom window delay).
- AC5c: the bypass is per account (two entries with different ``resetsAt``
  bypass independently) and the transition log fires exactly once per state
  change (no per-request INFO spam).
- AC6: the bypass is scoped to ``available_times`` only — rate-limited,
  circuit-broken and usage-limit-quarantined entries are still skipped.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import proxy.provider as provider
import proxy.provider_budget as provider_budget
import proxy.quota_state as quota_state
import pytest
from fastapi import Response
from proxy.provider import (
    _compute_retry_after,
    _has_next_provider,
    _is_within_allowed_window,
    _provider_in_allowed_window,
    _quota_surge_bypass_active,
    _resolve_provider_with_exclusions,
    _seconds_until_next_window,
    resolve_provider,
)
from proxy.quota_state import reset_quota_state
from proxy.rate_limiter import get_rate_limiter

#: 02:00 UTC is outside the configured ``10:00-00:00`` window.
NOW = datetime(2026, 10, 19, 2, 0, 0, tzinfo=UTC)

_PROVIDER_LOGGER = "llama-proxy.provider"
_OK_BODY = json.dumps({"choices": [{"message": {"content": "ok"}}]})


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _FakeDatetime:
    """Stand-in for ``proxy.provider.datetime`` with a frozen wall clock."""

    _now = NOW

    @classmethod
    def now(cls, tz=None):
        return cls._now

    @classmethod
    def fromtimestamp(cls, ts, tz=None):
        return datetime.fromtimestamp(ts, tz)


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    """Fresh quota cache, surge-transition log and provider module state."""
    reset_quota_state()
    provider.reset_surge_bypass_state()
    quota_state.set_clock(lambda: NOW)
    monkeypatch.setattr(provider, "datetime", _FakeDatetime)
    # Deterministic priority so the cycle never caps to ``low`` inside the
    # cheap window; both bindings matter (see test_provider_context_limit).
    monkeypatch.setattr(provider, "request_priority", lambda *a, **k: "medium")
    monkeypatch.setattr("proxy.priority.request_priority", lambda *a, **k: "medium")

    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    provider_budget._affinity.clear()
    yield
    reset_quota_state()
    provider.reset_surge_bypass_state()
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    provider_budget._affinity.clear()


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _usage_body(
    *,
    monthly_percent: float = 50.0,
    monthly_reset_days: float = 6.0,
    rolling_percent: float = 0.0,
    weekly_percent: float = 0.0,
) -> dict:
    """A ``/v1/usage`` body whose monthly reset sits a configurable distance out."""
    return {
        "usage": {
            "rolling": {
                "status": "ok",
                "percent": rolling_percent,
                "resetsAt": _iso(NOW + timedelta(hours=5)),
            },
            "weekly": {
                "status": "ok",
                "percent": weekly_percent,
                "resetsAt": _iso(NOW + timedelta(days=2)),
            },
            "monthly": {
                "status": "ok",
                "percent": monthly_percent,
                "resetsAt": _iso(NOW + timedelta(days=monthly_reset_days)),
            },
        }
    }


def _provider(
    name: str = "opencode-go",
    api_key_env: str = "OPENCODE_KEY_A",
    *,
    available_times: tuple[str, ...] = ("10:00-00:00",),
    **extra,
) -> dict:
    entry = {
        "name": name,
        "type": "remote",
        "provider": "opencode-go",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": api_key_env,
        "model": "deepseek-v4-flash",
        "available_times": list(available_times),
    }
    entry.update(extra)
    return entry


def _config(**server: object) -> dict:
    config: dict = {"server": dict(server), "provider_cooldown_seconds": 60}
    return config


class _Transport:
    """Injectable ``/v1/usage`` transport keyed by account."""

    def __init__(self, bodies: dict[str, dict]):
        self.bodies = dict(bodies)
        self.calls: list[str] = []

    async def __call__(self, provider_cfg: dict) -> dict:
        key = quota_state.account_key(provider_cfg)
        self.calls.append(key)
        return self.bodies[key]


async def _seed(model_config: dict, config: dict, bodies: dict[str, dict]) -> None:
    """Populate the quota cache through the public poller seam."""
    quota_state.set_quota_transport(_Transport(bodies))
    await quota_state.refresh_quota_accounts(model_config, config, force=True)


def _surge_logs(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
        and "quota_surge_bypass" in record.getMessage()
    ]


# ---------------------------------------------------------------------------
# Remote fallback cycle harness (real dispatch path)
# ---------------------------------------------------------------------------


class _Request:
    """Minimal request stub (body + session header) for the fallback cycle."""

    def __init__(self, session_id: str | None = "surge-session"):
        self._body = json.dumps(
            {"model": "surge-test", "messages": [{"role": "user", "content": "hi"}]}
        ).encode()
        self.headers = {"content-type": "application/json"}
        if session_id:
            self.headers["x-session-id"] = session_id
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


def _patch_remote_cycle(monkeypatch, providers: list[dict]) -> list[str]:
    """Patch the remote dispatch pointer; returns the dispatched entry names."""
    calls: list[str] = []

    async def _ptr(_request, _path, provider_cfg):
        calls.append(provider_cfg["name"])
        return Response(content=_OK_BODY, status_code=200, media_type="application/json")

    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: _ptr)
    monkeypatch.setattr(provider, "_get_tokenizer_for_model", MagicMock(return_value=(None, 1.0)))
    monkeypatch.setattr(provider, "_maybe_probe_quarantined_accounts", _noop_async)
    return calls


async def _noop_async(*args, **kwargs):
    return None


def _single_entry_model() -> dict:
    return {"name": "surge-test", "providers": [_provider()]}


async def _drive_cycle(model_config: dict, config: dict):
    return await provider._proxy_with_remote_fallback_cycle(
        _Request(), "v1/chat/completions", model_config, config
    )


# ---------------------------------------------------------------------------
# AC5 — 6 days out bypasses, 8 days out does not
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_six_days_out_bypasses_window_at_0200(caplog):
    """``resetsAt`` 6 days away at 02:00 UTC: eligible, logged with the reason."""
    model_config = _single_entry_model()
    config = _config()
    entry = model_config["providers"][0]
    await _seed(model_config, config, {"OPENCODE_KEY_A": _usage_body()})

    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        # Precondition: the window gate alone excludes the entry at 02:00 UTC.
        assert _is_within_allowed_window(entry, now_utc=NOW) is False
        assert _quota_surge_bypass_active(entry, config) is True
        assert _provider_in_allowed_window(entry, config, now_utc=NOW) is True
        assert resolve_provider(model_config, config=config)["name"] == "opencode-go"
        assert (
            _resolve_provider_with_exclusions(model_config, set(), set(), config=config)[
                "name"
            ]
            == "opencode-go"
        )
        assert _has_next_provider(model_config, set(), set(), config=config) is True

    surge_lines = _surge_logs(caplog)
    assert len(surge_lines) == 1, surge_lines
    assert "entered" in surge_lines[0]
    assert "reason=quota_surge" in surge_lines[0]
    assert "account=OPENCODE_KEY_A" in surge_lines[0]
    assert "resets_at=2026-10-25T02:00:00Z" in surge_lines[0]


@pytest.mark.asyncio
async def test_eight_days_out_remains_ineligible(caplog):
    """``resetsAt`` 8 days away: unchanged behaviour (window still applies)."""
    model_config = _single_entry_model()
    config = _config()
    entry = model_config["providers"][0]
    await _seed(
        model_config, config, {"OPENCODE_KEY_A": _usage_body(monthly_reset_days=8.0)}
    )

    assert _quota_surge_bypass_active(entry, config) is False
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        assert _provider_in_allowed_window(entry, config, now_utc=NOW) is False
        assert resolve_provider(model_config, config=config) is None
    assert _surge_logs(caplog) == []


@pytest.mark.asyncio
async def test_surge_dispatch_reaches_the_real_fallback_cycle(monkeypatch):
    """The bypass works end-to-end: the real chain dispatches at 02:00 UTC."""
    model_config = _single_entry_model()
    config = _config()
    await _seed(model_config, config, {"OPENCODE_KEY_A": _usage_body()})
    calls = _patch_remote_cycle(monkeypatch, model_config["providers"])

    response = await _drive_cycle(model_config, config)

    assert response.status_code == 200
    assert calls == ["opencode-go"]


@pytest.mark.asyncio
async def test_kill_switch_and_dry_run_keep_the_window(monkeypatch):
    """The kill switch and dry-run restore the pre-surge window behaviour."""
    model_config = _single_entry_model()
    entry = model_config["providers"][0]
    for config in (
        _config(quota_surge_enabled=False),
        _config(quota_surge_dry_run=True),
    ):
        await _seed(model_config, config, {"OPENCODE_KEY_A": _usage_body()})
        assert _quota_surge_bypass_active(entry, config) is False
        assert _provider_in_allowed_window(entry, config, now_utc=NOW) is False
        assert resolve_provider(model_config, config=config) is None


# ---------------------------------------------------------------------------
# AC6 — the bypass is scoped to available_times only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limited_entry_still_skipped_under_surge(monkeypatch):
    """Surge + an exhausted ``rate_limit_rpm`` window: still skipped."""
    model_config = _single_entry_model()
    model_config["providers"][0]["rate_limit_rpm"] = 1
    config = _config()
    entry = model_config["providers"][0]
    await _seed(model_config, config, {"OPENCODE_KEY_A": _usage_body()})
    assert _quota_surge_bypass_active(entry, config) is True

    limiter = get_rate_limiter()
    limiter.reset("opencode-go")
    assert await limiter.check_and_increment("opencode-go", 1, 60) is True
    calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
    try:
        with pytest.raises(provider.ChainExhaustedError) as exc_info:
            await _drive_cycle(model_config, config)
    finally:
        limiter.reset("opencode-go")

    assert calls == [], "a rate-limited entry must not be dispatched"
    payload = json.loads(exc_info.value.response.body)
    assert any(a.get("status") == "rate_limited" for a in payload.get("diagnostics", []))


@pytest.mark.asyncio
async def test_circuit_broken_entry_still_skipped_under_surge(monkeypatch):
    """Surge + an open sibling-failure circuit breaker: still skipped."""
    model_config = _single_entry_model()
    config = _config()
    entry = model_config["providers"][0]
    await _seed(model_config, config, {"OPENCODE_KEY_A": _usage_body()})
    assert _quota_surge_bypass_active(entry, config) is True

    # Trip the sibling-failure breaker (default threshold 2) on the entry.
    tripped = False
    for _ in range(5):
        tripped = provider._record_sibling_failure(
            "opencode-go",
            config,
            entry.get("provider"),
            domain_key=provider._failure_domain_key(entry),
            provider_type="remote",
        )
        if tripped:
            break
    assert tripped is True, "the sibling-failure breaker must be open"

    calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
    with pytest.raises(provider.ChainExhaustedError):
        await _drive_cycle(model_config, config)
    assert calls == [], "a circuit-broken entry must not be dispatched"


@pytest.mark.asyncio
async def test_usage_limit_quarantined_entry_still_skipped_under_surge(monkeypatch):
    """Surge + a pending usage-limit quarantine: still skipped."""
    model_config = _single_entry_model()
    config = _config()
    entry = model_config["providers"][0]
    await _seed(model_config, config, {"OPENCODE_KEY_A": _usage_body()})
    assert _quota_surge_bypass_active(entry, config) is True

    account = provider._usage_limit_account_key(entry)
    provider._usage_reset_at[account] = time.time() + 600

    calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
    with pytest.raises(provider.ChainExhaustedError) as exc_info:
        await _drive_cycle(model_config, config)
    assert calls == [], "a quarantined entry must not be dispatched"
    payload = json.loads(exc_info.value.response.body)
    assert any(
        a.get("status") == "usage_limit_reset" for a in payload.get("diagnostics", [])
    )


@pytest.mark.asyncio
async def test_failure_domain_sibling_still_excluded_under_surge():
    """Surge never re-admits an entry sharing a failed entry's domain."""
    model_config = {
        "providers": [
            _provider("opencode-go", "OPENCODE_KEY_A"),
            _provider("opencode-go-2", "OPENCODE_KEY_B"),
        ]
    }
    config = _config()
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(),
            "OPENCODE_KEY_B": _usage_body(),
        },
    )
    domain = provider._failure_domain_key(model_config["providers"][1])

    # Sanity: surge really is active for both entries at 02:00 UTC.
    assert resolve_provider(model_config, config=config)["name"] == "opencode-go"
    assert _has_next_provider(model_config, set(), {domain}, config=config) is False
    assert (
        _resolve_provider_with_exclusions(
            model_config, set(), {domain}, config=config
        )
        is None
    )


# ---------------------------------------------------------------------------
# AC5b — the bypass clears automatically; retry-after stays coherent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exhausted_entry_is_not_bypassed():
    """An exhausted account loses the bypass even inside the final week."""
    model_config = _single_entry_model()
    config = _config()
    entry = model_config["providers"][0]
    await _seed(
        model_config, config, {"OPENCODE_KEY_A": _usage_body(monthly_percent=100.0)}
    )

    snapshot = quota_state.get_quota_snapshot(entry, config)
    assert snapshot is not None and snapshot.exhausted is True
    assert _quota_surge_bypass_active(entry, config) is False
    assert resolve_provider(model_config, config=config) is None


@pytest.mark.asyncio
async def test_bypass_clears_once_the_reset_passes():
    """After ``resetsAt`` the horizon is not positive: no bypass."""
    model_config = _single_entry_model()
    config = _config()
    entry = model_config["providers"][0]
    await _seed(
        model_config, config, {"OPENCODE_KEY_A": _usage_body(monthly_reset_days=-0.5)}
    )

    snapshot = quota_state.get_quota_snapshot(entry, config)
    assert snapshot is not None and snapshot.seconds_to_reset == 0.0
    assert _quota_surge_bypass_active(entry, config) is False
    assert resolve_provider(model_config, config=config) is None


@pytest.mark.asyncio
async def test_no_phantom_window_delay_for_a_surged_entry():
    """A surged entry contributes no ``next window`` / ``retry_after`` delay."""
    model_config = _single_entry_model()
    config = _config()
    entry = model_config["providers"][0]
    await _seed(model_config, config, {"OPENCODE_KEY_A": _usage_body()})

    # Surged: usable now, so no wait at all.
    assert _seconds_until_next_window(entry, now_utc=NOW, config=config) is None
    assert (
        _compute_retry_after({}, model_config=model_config, now_utc=NOW, config=config)
        == 0
    )

    # Outside the surge window the honest window-edge delay is unchanged.
    quota_state.set_quota_transport(
        _Transport({"OPENCODE_KEY_A": _usage_body(monthly_reset_days=8.0)})
    )
    await quota_state.refresh_quota_accounts(model_config, config, force=True)

    secs = _seconds_until_next_window(entry, now_utc=NOW, config=config)
    assert secs == 8 * 3600 + 0  # 02:00 -> 10:00 UTC
    assert (
        _compute_retry_after({}, model_config=model_config, now_utc=NOW, config=config)
        == 8 * 3600
    )


# ---------------------------------------------------------------------------
# AC5c — per-account windows and once-per-transition logging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bypass_window_is_per_account(caplog):
    """Two keys with different ``resetsAt`` bypass independently."""
    model_config = {
        "providers": [
            _provider("opencode-go", "OPENCODE_KEY_A"),
            _provider("opencode-go-2", "OPENCODE_KEY_B"),
        ]
    }
    config = _config()
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(monthly_reset_days=6.0),
            "OPENCODE_KEY_B": _usage_body(monthly_reset_days=10.0),
        },
    )
    first, second = model_config["providers"]

    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        assert _provider_in_allowed_window(first, config, now_utc=NOW) is True
        assert _provider_in_allowed_window(second, config, now_utc=NOW) is False
        assert resolve_provider(model_config, config=config)["name"] == "opencode-go"
        assert (
            _has_next_provider(model_config, {"opencode-go"}, set(), config=config)
            is False
        )

    lines = _surge_logs(caplog)
    assert len(lines) == 1
    assert "account=OPENCODE_KEY_A" in lines[0]
    assert "OPENCODE_KEY_B" not in lines[0]


@pytest.mark.asyncio
async def test_transition_logs_once_on_enter_and_once_on_leave(caplog):
    """Repeated eligibility checks log one line per state change only."""
    entry = _provider()
    config = _config()
    await _seed({"providers": [entry]}, config, {"OPENCODE_KEY_A": _usage_body()})

    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        for _ in range(5):
            assert _provider_in_allowed_window(entry, config, now_utc=NOW) is True
    entered = _surge_logs(caplog)
    assert len(entered) == 1
    assert "entered" in entered[0]
    assert "reason=quota_surge" in entered[0]

    caplog.clear()
    await _seed(
        {"providers": [entry]},
        config,
        {"OPENCODE_KEY_A": _usage_body(monthly_reset_days=-0.5)},
    )
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        for _ in range(5):
            assert _provider_in_allowed_window(entry, config, now_utc=NOW) is False
    left = _surge_logs(caplog)
    assert len(left) == 1
    assert "left" in left[0]
    assert "reason=quota_surge" in left[0]


def test_non_quota_providers_never_log_or_bypass(caplog):
    """A non-``opencode-*`` entry keeps its restriction and logs nothing."""
    entry = {
        "name": "deepseek",
        "type": "remote",
        "provider": "deepseek",
        "endpoint": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_KEY",
        "available_times": ["10:00-00:00"],
    }
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        assert _provider_in_allowed_window(entry, _config(), now_utc=NOW) is False
    assert _surge_logs(caplog) == []
