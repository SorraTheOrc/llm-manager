"""Tests for the quota hard stop / cost guard (LP-0MVF7KO8C005NB59, epic
LP-0MVF4FIP1001RD8P feature F3).

Goal: surge must never dispatch to an account at or near its monthly cap, so
the proxy can never spend real pay-as-you-go money on an account with the
console "Use balance" option enabled. An entry whose monthly consumption is at
or above ``server.quota_exhaustion_percent`` (default 99.5) **or** whose
``effective_headroom`` is at or below ``100 - threshold`` (a spent rolling or
weekly window) is skipped before any upstream request is attempted and logged
at INFO with ``reason=quota_exhausted``.

Every test seeds quota state through the public poller seam
(``quota_state.set_quota_transport`` + ``refresh_quota_accounts``) with a
frozen clock and asserts observable behaviour through the public routing API:
the skip decision, the real fallback dispatch cycle (zero upstream calls when
skipped), the mid-surge hard stop and the interaction with the reactive
usage-limit quarantine (LP-0MSLJPOCC0001ROJ) — cumulative, never
alternatives.

Covered acceptance criteria:

- AC4: monthly ``percent >= 99.5`` (or effective headroom below the equivalent
  threshold) → skipped without any upstream request, logged with the reason,
  repeated over a table of window combinations.
- AC10: mid-surge crossing to ``monthly.percent = 99.6`` hard-stops on the
  very next dispatch decision (zero further requests), and the
  ``GoUsageLimitError`` quarantine path still applies.
- AC4b: the skip is evaluated on ``effective_headroom`` — a spent rolling or
  weekly window is skipped even when the monthly percent is low.
- AC4c: absent / invalid / out-of-range ``server.quota_exhaustion_percent``
  values fail safe (clamped, one WARN), and the default is used when the key
  is absent.
- AC10b: a quota skip never clears, shortens or overrides an active
  usage-limit quarantine, and the two reasons appear distinctly in the log.
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
    _has_next_provider,
    _provider_in_allowed_window,
    _quota_exhausted,
    _quota_surge_bypass_active,
    _resolve_provider_with_exclusions,
    resolve_provider,
)
from proxy.quota_state import (
    DEFAULT_QUOTA_EXHAUSTION_PERCENT,
    MAX_QUOTA_EXHAUSTION_PERCENT,
    MIN_QUOTA_EXHAUSTION_PERCENT,
    quota_exhaustion_percent,
    reset_quota_state,
)

#: 02:00 UTC is outside the configured ``10:00-00:00`` window — the surge
#: bypass is the only thing that makes an entry eligible at this hour, so a
#: hard-stopped entry provably cannot be re-enabled by surge.
NOW = datetime(2026, 10, 19, 2, 0, 0, tzinfo=UTC)

_PROVIDER_LOGGER = "llama-proxy.provider"
_QUOTA_STATE_LOGGER = "llama-proxy.quota_state"
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
    monkeypatch.setattr(provider, "request_priority", lambda *a, **k: "medium")
    monkeypatch.setattr("proxy.priority.request_priority", lambda *a, **k: "medium")

    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._usage_reset_guessed.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    provider_budget._affinity.clear()
    yield
    reset_quota_state()
    provider.reset_surge_bypass_state()
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._usage_reset_guessed.clear()
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
    """A ``/v1/usage`` body whose monthly reset sits 6 days out by default."""
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


def _skip_logs(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
        and "Skipping provider=" in record.getMessage()
    ]


# ---------------------------------------------------------------------------
# Remote fallback cycle harness (real dispatch path)
# ---------------------------------------------------------------------------


class _Request:
    """Minimal request stub (body + session header) for the fallback cycle."""

    def __init__(self, session_id: str | None = "hard-stop-session"):
        self._body = json.dumps(
            {"model": "hard-stop-test", "messages": [{"role": "user", "content": "hi"}]}
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
    monkeypatch.setattr(
        provider, "_get_tokenizer_for_model", MagicMock(return_value=(None, 1.0))
    )
    monkeypatch.setattr(provider, "_maybe_probe_quarantined_accounts", _noop_async)
    return calls


async def _noop_async(*args, **kwargs):
    return None


def _single_entry_model() -> dict:
    return {"name": "hard-stop-test", "providers": [_provider()]}


async def _drive_cycle(model_config: dict, config: dict):
    return await provider._proxy_with_remote_fallback_cycle(
        _Request(), "v1/chat/completions", model_config, config
    )


# ---------------------------------------------------------------------------
# AC4 — threshold table across window combinations, zero upstream calls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "monthly,rolling,weekly,skip",
    [
        # Below the threshold in every window: eligible.
        (50.0, 0.0, 0.0, False),
        # Just below (99.4 => headroom 0.6 > 0.5): still eligible — boundary
        # row proving the guard does not over-trigger.
        (99.4, 0.0, 0.0, False),
        (99.4, 99.4, 99.4, False),
        # At / above the threshold in the monthly window: hard stop.
        (99.5, 0.0, 0.0, True),
        (99.6, 0.0, 0.0, True),
        (100.0, 0.0, 0.0, True),
        # AC4b: monthly percent low but a rolling/weekly window is spent —
        # effective headroom is below the equivalent threshold.
        (10.0, 99.6, 0.0, True),
        (10.0, 0.0, 99.6, True),
        # AC4b boundary: rolling/weekly at 99.4 leave effective headroom 0.6,
        # so a low monthly percent does not trigger the hard stop.
        (10.0, 99.4, 99.4, False),
    ],
)
@pytest.mark.asyncio
async def test_quota_skip_table(monkeypatch, caplog, monthly, rolling, weekly, skip):
    """The hard stop skips at/above the threshold across window combos.

    Each row drives the REAL fallback dispatch cycle and asserts zero upstream
    requests when the entry is exhausted, plus the INFO skip reason.
    """
    model_config = _single_entry_model()
    config = _config()
    entry = model_config["providers"][0]
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(
                monthly_percent=monthly,
                rolling_percent=rolling,
                weekly_percent=weekly,
            )
        },
    )
    snapshot = quota_state.get_quota_snapshot(entry, config)
    assert snapshot is not None
    exhausted, reason = _quota_exhausted(entry, config)

    if skip:
        assert exhausted is True
        assert reason.startswith("reason=quota_exhausted")
        assert "account=OPENCODE_KEY_A" in reason
        # The monthly-percent path carries the observed percent; the
        # headroom path carries the effective headroom.
        assert "threshold=99.5" in reason
        with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
            assert resolve_provider(model_config, config=config) is None
            assert (
                _resolve_provider_with_exclusions(
                    model_config, set(), set(), config=config
                )
                is None
            )
            assert (
                _has_next_provider(model_config, set(), set(), config=config)
                is False
            )
            calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
            with pytest.raises(provider.ChainExhaustedError):
                await _drive_cycle(model_config, config)
        assert calls == [], "a hard-stopped entry must never be dispatched"
        skip_lines = _skip_logs(caplog)
        assert any(
            "quota_exhausted" in line and "OPENCODE_KEY_A" in line
            for line in skip_lines
        ), skip_lines
        if not snapshot.exhausted:
            # The hard stop's own INFO line carries the log-safe reason, the
            # observed monthly percent and the threshold (AC4). A strictly
            # exhausted snapshot is already caught by the earlier exhausted
            # gate, whose line reports the account and headroom instead.
            assert any(
                "reason=quota_exhausted" in line
                and "monthly_percent=" in line
                and "threshold=99.5" in line
                for line in skip_lines
            ), skip_lines
    else:
        assert exhausted is False
        assert reason == ""
        with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
            calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
            response = await _drive_cycle(model_config, config)
        assert response.status_code == 200
        assert calls == ["opencode-go"]
        assert all(
            "reason=quota_exhausted" not in line for line in _skip_logs(caplog)
        )


@pytest.mark.asyncio
async def test_skip_applies_when_surge_disabled_or_dry_run(monkeypatch):
    """The guard is independent of the surge kill switch / dry-run."""
    for config in (
        _config(quota_surge_enabled=False),
        _config(quota_surge_dry_run=True),
    ):
        model_config = _single_entry_model()
        await _seed(
            model_config,
            config,
            {"OPENCODE_KEY_A": _usage_body(monthly_percent=99.6)},
        )
        assert resolve_provider(model_config, config=config) is None
        calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
        with pytest.raises(provider.ChainExhaustedError):
            await _drive_cycle(model_config, config)
        assert calls == [], "surge must never re-enable a hard-stopped entry"


@pytest.mark.asyncio
async def test_fail_open_without_snapshot_or_non_quota_entry():
    """No authoritative quota state => no hard stop (the guard never guesses)."""
    # An opencode entry that has never been polled resolves normally when the
    # window gate is absent.
    no_window_entry = _provider()
    no_window_entry["available_times"] = []
    model_config = {"name": "m", "providers": [no_window_entry]}
    config = _config()
    assert _quota_exhausted(no_window_entry, config) == (False, "")
    assert resolve_provider(model_config, config=config)["name"] == "opencode-go"

    # A non-quota (non-opencode) entry is outside the guard's scope entirely,
    # even with an aggressive threshold.
    other = {
        "name": "deepseek",
        "type": "remote",
        "provider": "deepseek",
        "endpoint": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_KEY",
    }
    model_config = {"name": "m", "providers": [other]}
    assert _quota_exhausted(other, _config(quota_exhaustion_percent=1)) == (False, "")
    assert resolve_provider(model_config, config=config)["name"] == "deepseek"


# ---------------------------------------------------------------------------
# AC4c — config fail-safe (default, clamp, single WARN)
# ---------------------------------------------------------------------------


def test_absent_key_uses_default_threshold():
    assert quota_exhaustion_percent(None) == DEFAULT_QUOTA_EXHAUSTION_PERCENT
    assert quota_exhaustion_percent({}) == DEFAULT_QUOTA_EXHAUSTION_PERCENT
    assert (
        quota_exhaustion_percent({"server": {}}) == DEFAULT_QUOTA_EXHAUSTION_PERCENT
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0, MIN_QUOTA_EXHAUSTION_PERCENT),
        (-5, MIN_QUOTA_EXHAUSTION_PERCENT),
        (0.5, MIN_QUOTA_EXHAUSTION_PERCENT),
        (101, MAX_QUOTA_EXHAUSTION_PERCENT),
        (1000, MAX_QUOTA_EXHAUSTION_PERCENT),
        ("abc", DEFAULT_QUOTA_EXHAUSTION_PERCENT),
        (None, DEFAULT_QUOTA_EXHAUSTION_PERCENT),
        (float("nan"), DEFAULT_QUOTA_EXHAUSTION_PERCENT),
    ],
)
def test_invalid_values_are_clamped(raw, expected):
    """Out-of-range values clamp into [1, 100]; non-numeric default."""
    assert (
        quota_exhaustion_percent({"server": {"quota_exhaustion_percent": raw}})
        == expected
    )
    # Flat (non ``server.*``) placement is accepted too.
    assert (
        quota_exhaustion_percent({"quota_exhaustion_percent": raw}) == expected
    )


def test_invalid_threshold_warns_exactly_once(caplog):
    """Multiple invalid values produce exactly one WARN (per process)."""
    with caplog.at_level(logging.WARNING, logger=_QUOTA_STATE_LOGGER):
        for raw in (0, -5, 101, "abc", float("nan")):
            quota_exhaustion_percent({"server": {"quota_exhaustion_percent": raw}})
        # A subsequent VALID call must not warn nor change the result.
        assert (
            quota_exhaustion_percent({"server": {"quota_exhaustion_percent": 99.0}})
            == 99.0
        )
    warns = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "quota_exhaustion_percent" in record.getMessage()
    ]
    assert len(warns) == 1, warns
    assert "0" in warns[0]  # names the offending value

    # The one-shot flag resets cleanly for the next process/test.
    quota_state.reset_quota_exhaustion_warning()
    with caplog.at_level(logging.WARNING, logger=_QUOTA_STATE_LOGGER):
        quota_exhaustion_percent({"server": {"quota_exhaustion_percent": 101}})
    warns = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "quota_exhaustion_percent" in record.getMessage()
    ]
    assert len(warns) == 2, warns


@pytest.mark.asyncio
async def test_clamped_threshold_changes_the_decision(monkeypatch):
    """The clamped value is actually applied, not just logged."""
    # "abc" -> default 99.5: a 99.6-monthly entry is hard-stopped.
    model_config = _single_entry_model()
    config = _config(quota_exhaustion_percent="abc")
    await _seed(
        model_config, config, {"OPENCODE_KEY_A": _usage_body(monthly_percent=99.6)}
    )
    assert resolve_provider(model_config, config=config) is None

    # 101 -> clamped to 100: the same 99.6-monthly entry stays eligible
    # (99.6 < 100 and headroom 0.4 > 0.0), so the clamp is honoured.
    model_config = _single_entry_model()
    config = _config(quota_exhaustion_percent=101)
    await _seed(
        model_config, config, {"OPENCODE_KEY_A": _usage_body(monthly_percent=99.6)}
    )
    entry = model_config["providers"][0]
    assert _quota_exhausted(entry, config) == (False, "")
    calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
    response = await _drive_cycle(model_config, config)
    assert response.status_code == 200
    assert calls == ["opencode-go"]


# ---------------------------------------------------------------------------
# AC10 — mid-surge hard stop (no sticky surge state)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mid_surge_crossing_hard_stops_next_dispatch(monkeypatch, caplog):
    """90% dispatches; a poll crossing to 99.6% stops the NEXT decision.

    The surge window stays active throughout (the snapshot is not 'exhausted'
    in the strict sense), so it is the hard stop — not the exhausted gate and
    not a sticky surge state — that forbids the dispatch.
    """
    model_config = _single_entry_model()
    config = _config()
    entry = model_config["providers"][0]
    await _seed(
        model_config, config, {"OPENCODE_KEY_A": _usage_body(monthly_percent=90.0)}
    )
    snapshot = quota_state.get_quota_snapshot(entry, config)
    assert snapshot is not None and snapshot.effective_headroom == 10.0
    assert _quota_surge_bypass_active(entry, config) is True

    # First dispatch while inside the surge window: allowed.
    calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
    response = await _drive_cycle(model_config, config)
    assert response.status_code == 200
    assert calls == ["opencode-go"]

    # A poll shows the account crossing the threshold mid-surge.
    await _seed(
        model_config, config, {"OPENCODE_KEY_A": _usage_body(monthly_percent=99.6)}
    )
    snapshot = quota_state.get_quota_snapshot(entry, config)
    assert snapshot is not None
    assert snapshot.effective_headroom == pytest.approx(0.4, abs=1e-9)
    # Surge would still lift the available_times window…
    assert _quota_surge_bypass_active(entry, config) is True
    assert _provider_in_allowed_window(entry, config, now_utc=NOW) is True

    # …but the very next dispatch decision must not select the entry.
    assert resolve_provider(model_config, config=config) is None
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        with pytest.raises(provider.ChainExhaustedError):
            await _drive_cycle(model_config, config)
    assert calls == ["opencode-go"], "zero further requests after the crossing"
    assert any(
        "reason=quota_exhausted" in line
        and "monthly_percent=99.6" in line
        for line in _skip_logs(caplog)
    )


@pytest.mark.asyncio
async def test_single_dispatch_followed_by_crossing_is_the_only_request(monkeypatch):
    """Even the pre-crossing allowed dispatch stays a one-off within the test."""
    model_config = _single_entry_model()
    config = _config()
    await _seed(
        model_config, config, {"OPENCODE_KEY_A": _usage_body(monthly_percent=90.0)}
    )
    calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
    assert (await _drive_cycle(model_config, config)).status_code == 200
    assert calls == ["opencode-go"]

    await _seed(
        model_config, config, {"OPENCODE_KEY_A": _usage_body(monthly_percent=99.6)}
    )
    with pytest.raises(provider.ChainExhaustedError):
        await _drive_cycle(model_config, config)
    with pytest.raises(provider.ChainExhaustedError):
        await _drive_cycle(model_config, config)
    assert calls == ["opencode-go"], "only the pre-crossing dispatch reached upstream"


# ---------------------------------------------------------------------------
# AC10 / AC10b — cumulative with the reactive usage-limit quarantine
# ---------------------------------------------------------------------------


def _gousage_429(message: str) -> Response:
    """The observed opencode 429 GoUsageLimitError body."""
    body = json.dumps(
        {
            "type": "error",
            "error": {"type": "GoUsageLimitError", "message": message},
        }
    )
    return Response(
        status_code=429,
        content=body.encode("utf-8"),
        media_type="application/json",
    )


@pytest.mark.asyncio
async def test_gousagelimiterror_quarantine_and_hard_stop_are_cumulative(
    monkeypatch, caplog
):
    """A GoUsageLimitError quarantines; the quota skip neither clears nor
    shortens it, and the two reasons appear distinctly across the log."""
    model_config = _single_entry_model()
    config = _config()
    entry = model_config["providers"][0]
    await _seed(
        model_config, config, {"OPENCODE_KEY_A": _usage_body(monthly_percent=99.6)}
    )

    # Reactive path: a real GoUsageLimitError 429 is parsed into a quarantine.
    info = provider._usage_limit_reset_info(
        _gousage_429("Weekly usage limit reached. Resets in 1hr."),
        json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "GoUsageLimitError",
                    "message": "Weekly usage limit reached. Resets in 1hr.",
                },
            }
        ),
        config,
    )
    assert info is not None and info.explicit
    account = provider._usage_limit_account_key(entry)
    provider._usage_reset_at[account] = time.time() + info.seconds
    expiry_before = provider._usage_reset_at[account]

    # Quarantine takes precedence: only its reason appears for this entry.
    calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        assert resolve_provider(model_config, config=config) is None
        with pytest.raises(provider.ChainExhaustedError):
            await _drive_cycle(model_config, config)
    assert calls == []
    quarantined_lines = _skip_logs(caplog)
    assert any("usage_limit_reset_pending" in line for line in quarantined_lines)
    assert not any(
        "reason=quota_exhausted" in line for line in quarantined_lines
    ), "quarantine keeps precedence over the quota skip reason"

    # A fresh quota poll (any snapshot change) never clears/shortens it.
    await _seed(
        model_config, config, {"OPENCODE_KEY_A": _usage_body(monthly_percent=99.8)}
    )
    assert provider._usage_reset_at[account] == expiry_before

    # Clearing the quarantine does NOT clear the quota skip: the entry is
    # still hard-stopped, now with its own distinct reason.
    cleared = provider.clear_usage_limit_quarantine(account)
    assert cleared == [account]
    caplog.clear()
    calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        assert resolve_provider(model_config, config=config) is None
        with pytest.raises(provider.ChainExhaustedError):
            await _drive_cycle(model_config, config)
    assert calls == []
    quota_lines = _skip_logs(caplog)
    assert any("reason=quota_exhausted" in line for line in quota_lines)
    assert not any("usage_limit_reset_pending" in line for line in quota_lines)
    # The two reasons have appeared in the log, distinctly.
    all_reasons = quarantined_lines + quota_lines
    assert any("usage_limit_reset_pending" in line for line in all_reasons)
    assert any("reason=quota_exhausted" in line for line in all_reasons)


@pytest.mark.asyncio
async def test_quota_skip_never_overrides_an_active_quarantine(monkeypatch, caplog):
    """Two entries, one quarantined and one hard-stopped: both reasons appear
    distinctly in a single dispatch decision."""
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
            "OPENCODE_KEY_A": _usage_body(monthly_percent=99.6),
            "OPENCODE_KEY_B": _usage_body(monthly_percent=99.6),
        },
    )
    # Quarantine only account A via the reactive path.
    info = provider._usage_limit_reset_info(
        _gousage_429("Monthly usage limit reached. Resets in 2hr."),
        json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "GoUsageLimitError",
                    "message": "Monthly usage limit reached. Resets in 2hr.",
                },
            }
        ),
        config,
    )
    assert info is not None
    quarantined = provider._usage_limit_account_key(model_config["providers"][0])
    provider._usage_reset_at[quarantined] = time.time() + info.seconds

    calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        assert resolve_provider(model_config, config=config) is None
        with pytest.raises(provider.ChainExhaustedError):
            await _drive_cycle(model_config, config)
    assert calls == []
    lines = _skip_logs(caplog)
    assert any("usage_limit_reset_pending" in line and "OPENCODE_KEY_A@" in line
               for line in lines)
    assert any("reason=quota_exhausted" in line and "account=OPENCODE_KEY_B" in line
               for line in lines)
