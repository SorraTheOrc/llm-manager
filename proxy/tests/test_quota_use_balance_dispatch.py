"""Tests for the plan-vs-balance dispatch override (LP-0MW46PBD200249WK,
epic LP-0MW46P7QU0064LM3).

Goal: when an OpenCode Go account declares ``use_balance: true`` and **only**
its monthly plan window is spent (rolling and weekly still hold real
headroom), the proxy dispatches to it and the upstream bills the over-plan
spend to prepaid credit, instead of skipping it as ``quota_exhausted`` for
the whole monthly reset horizon.

The decision is a pure cache read, exactly like the hard stop it softens:
``_dispatch_on_balance`` never polls, never blocks and never raises, and the
``available_times`` / surge / cooldown / usage-limit gates keep their exact
precedence — only the hard-stop predicate changes (AC7, scope).

Every test seeds quota state through the public poller seam
(``quota_state.set_quota_transport`` + ``refresh_quota_accounts``) with a
frozen clock and asserts observable behaviour through the public routing API.

Covered acceptance criteria:

- AC1: monthly blocking + rolling/weekly ok + ``use_balance: true`` →
  eligible (no skip, request attempted); monthly blocking +
  ``use_balance: false`` → skipped (current behaviour).
- AC2: rolling or weekly blocking (or their headroom at/below the threshold
  margin) + ``use_balance: true`` → skipped (genuine exhaustion is not
  overridden).
- AC3: no snapshot / disabled / stale-suspended → byte-identical fail-open or
  hard-stop behaviour; the override is inert there.
- AC4: the decision is logged with a distinct, greppable reason
  (``dispatch_on_balance (... use_balance=true)``) that stays apart from the
  applied ``reason=quota_exhausted`` skips.
- AC5 / AC10: an end-to-end integration test proves a ``plan`` request
  reaches ``opencode-go`` when the plan is exhausted and ``use_balance: true``
  and does not reach it when false.
- AC11: ``use_balance: false`` or absent keeps the current hard stop,
  byte-identical (regression).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import proxy.provider as provider
import proxy.provider_budget as provider_budget
import proxy.quota_state as quota_state
import pytest
from fastapi import Response
from proxy.provider import (
    _quota_exhausted,
    _quota_unavailable_reason,
    _resolve_provider_with_exclusions,
    resolve_provider,
)
from proxy.quota_state import reset_quota_state

#: 02:00 UTC is outside the default ``10:00-00:00`` window, so tests that
#: expect a dispatch widen the window explicitly — the ``available_times``
#: gate is deliberately NOT touched by the balance override.
NOW = datetime(2026, 10, 19, 2, 0, 0, tzinfo=UTC)

_PROVIDER_LOGGER = "llama-proxy.provider"
_OK_BODY = json.dumps({"choices": [{"message": {"content": "ok"}}]})

#: A window that covers NOW, so an exhausted-but-eligible entry can actually
#: be dispatched in the integration tests (surge bypass is inactive for an
#: exhausted snapshot).
_ALWAYS_WINDOW = ("00:00-23:59",)

#: The shipped config's plan/author/code chains - ``opencode-go`` opts into
#: balance use, its siblings stay at the false default (config.yaml).
_PLAN_SPENT_STATUS = "rate-limited"


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
    provider._usage_reset_window.clear()
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
    provider._usage_reset_window.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    provider_budget._affinity.clear()


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _usage_body(
    *,
    monthly_percent: float = 50.0,
    monthly_status: str = "ok",
    monthly_reset_days: float = 6.0,
    rolling_percent: float = 0.0,
    rolling_status: str = "ok",
    weekly_percent: float = 0.0,
    weekly_status: str = "ok",
) -> dict:
    """A ``/v1/usage`` body with per-window status/percent control."""
    return {
        "usage": {
            "rolling": {
                "status": rolling_status,
                "percent": rolling_percent,
                "resetsAt": _iso(NOW + timedelta(hours=5)),
            },
            "weekly": {
                "status": weekly_status,
                "percent": weekly_percent,
                "resetsAt": _iso(NOW + timedelta(days=2)),
            },
            "monthly": {
                "status": monthly_status,
                "percent": monthly_percent,
                "resetsAt": _iso(NOW + timedelta(days=monthly_reset_days)),
            },
        }
    }


def _provider(
    name: str = "opencode-go",
    api_key_env: str = "OPENCODE_KEY_A",
    *,
    available_times: tuple[str, ...] = _ALWAYS_WINDOW,
    use_balance: bool | None = None,
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
    # ``None`` means "key absent" (the shipped default) — an explicit key is
    # only added for a real boolean, mirroring the config contract.
    if use_balance is not None:
        entry["use_balance"] = use_balance
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
        if record.levelno == logging.INFO and "Skipping provider=" in record.getMessage()
    ]


def _dispatch_on_balance_logs(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
        and "dispatch_on_balance" in record.getMessage()
    ]


class _Request:
    """Minimal request stub (body + session header) for the fallback cycle."""

    def __init__(self, session_id: str | None = "balance-session"):
        self._body = json.dumps(
            {"model": "balance-test", "messages": [{"role": "user", "content": "hi"}]}
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


def _single_entry_model(entry: dict | None = None) -> dict:
    return {"name": "balance-test", "providers": [entry if entry is not None else _provider()]}


async def _drive_cycle(model_config: dict, config: dict):
    return await provider._proxy_with_remote_fallback_cycle(
        _Request(), "v1/chat/completions", model_config, config
    )


# ---------------------------------------------------------------------------
# AC1 / AC10 / AC11 — the monthly-plan-only override decision table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "monthly_status,monthly_percent,use_balance,skip",
    [
        # AC1: monthly plan spent (blocking, status rate-limited) + opt-in
        # -> eligible; without the opt-in -> skipped (current behaviour).
        (_PLAN_SPENT_STATUS, 100.0, True, False),
        (_PLAN_SPENT_STATUS, 100.0, False, True),
        (_PLAN_SPENT_STATUS, 100.0, None, True),
        # Plan at the cap with a healthy status -- the same override applies.
        ("ok", 100.0, True, False),
        ("ok", 100.0, False, True),
        # AC10: at/above the threshold (99.6) with the opt-in -> eligible.
        ("ok", 99.6, True, False),
        ("ok", 99.6, False, True),
        # Boundary: below the threshold stays eligible regardless of the
        # opt-in (the override is never the reason an entry is unblocked).
        ("ok", 99.4, True, False),
        ("ok", 99.4, False, False),
        # AC11 regression: use_balance=False keeps the exact pre-feature
        # decision (the 99.6-False row above already covers the payload).
    ],
)
@pytest.mark.asyncio
async def test_monthly_plan_only_decision_table(
    monkeypatch, caplog, monthly_status, monthly_percent, use_balance, skip
):
    """The hard stop softens only for monthly-plan-only + balance opt-in.

    Each row drives the REAL fallback dispatch cycle: with the opt-in the
    request reaches opencode-go (billed to balance); without it the entry is
    skipped with the unchanged ``reason=quota_exhausted`` and zero upstream
    calls.
    """
    entry = _provider(use_balance=use_balance)
    model_config = _single_entry_model(entry)
    config = _config()
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(
                monthly_percent=monthly_percent,
                monthly_status=monthly_status,
            )
        },
    )
    snapshot = quota_state.get_quota_snapshot(entry, config)
    assert snapshot is not None
    exhausted, reason = _quota_exhausted(entry, config)

    if skip:
        assert exhausted is True
        if use_balance is False:
            # AC11: byte-identical hard-stop reason payload (unchanged). The
            # effective headroom is the minimum across the three windows (0
            # when a window blocks), so it varies per row.
            effective = min(
                100.0,  # rolling at 0%
                100.0,  # weekly at 0%
                0.0 if monthly_status != "ok" else max(0.0, 100.0 - monthly_percent),
            )
            assert reason == (
                "reason=quota_exhausted (account=OPENCODE_KEY_A, "
                f"monthly_percent={monthly_percent:.1f}%, "
                f"effective_headroom={effective:.1f}%, "
                "threshold=99.5%)"
            )
        else:
            assert reason.startswith("reason=quota_exhausted")
            assert "account=OPENCODE_KEY_A" in reason
        with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
            assert resolve_provider(model_config, config=config) is None
            calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
            with pytest.raises(provider.ChainExhaustedError):
                await _drive_cycle(model_config, config)
        assert calls == [], "a skipped entry must never be dispatched"
        assert _dispatch_on_balance_logs(caplog) == []
        assert any(
            "quota_exhausted" in line and "OPENCODE_KEY_A" in line
            for line in _skip_logs(caplog)
        ), _skip_logs(caplog)
        if not snapshot.exhausted:
            # A strictly-exhausted snapshot is caught by the earlier exhausted
            # gate; only the hard stop's own line carries the log-safe reason
            # with the observed monthly percent and the threshold.
            assert any(
                "reason=quota_exhausted" in line
                and "monthly_percent=" in line
                and "threshold=99.5" in line
                for line in _skip_logs(caplog)
            ), _skip_logs(caplog)
    else:
        assert exhausted is False
        assert reason == ""
        # The proactive quota gate must not flag the entry as unavailable
        # either — both hard-stop predecessors agree (AC5).
        assert _quota_unavailable_reason(entry, config) is None
        with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
            assert resolve_provider(model_config, config=config)["name"] == "opencode-go"
            calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
            response = await _drive_cycle(model_config, config)
        assert response.status_code == 200
        assert calls == ["opencode-go"], "the request must reach opencode-go"
        assert all(
            "reason=quota_exhausted" not in line for line in _skip_logs(caplog)
        )


# ---------------------------------------------------------------------------
# AC2 — rolling/weekly exhaustion is never overridden
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rolling_status,rolling_percent,weekly_status,weekly_percent,monthly_percent,skip",
    [
        # Rolling window blocking (rate-limited): genuine exhaustion.
        (_PLAN_SPENT_STATUS, 100.0, "ok", 0.0, 100.0, True),
        # Weekly window blocking (rate-limited): genuine exhaustion.
        ("ok", 0.0, _PLAN_SPENT_STATUS, 100.0, 100.0, True),
        # Rolling headroom at/below the threshold margin (99.6 => 0.4 <= 0.5)
        # with an ok status is genuine exhaustion too.
        ("ok", 99.6, "ok", 0.0, 100.0, True),
        ("ok", 0.0, "ok", 99.6, 100.0, True),
        # Control: rolling and weekly at 99.4 (headroom 0.6 > 0.5) with a
        # spent monthly window stays eligible on balance.
        ("ok", 99.4, "ok", 99.4, 100.0, False),
    ],
)
@pytest.mark.asyncio
async def test_rolling_weekly_exhaustion_never_overridden(
    monkeypatch, caplog, rolling_status, rolling_percent, weekly_status,
    weekly_percent, monthly_percent, skip,
):
    """A spent rolling/weekly window keeps the hard stop even with the opt-in."""
    entry = _provider(use_balance=True)
    model_config = _single_entry_model(entry)
    config = _config()
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(
                monthly_percent=monthly_percent,
                rolling_percent=rolling_percent,
                rolling_status=rolling_status,
                weekly_percent=weekly_percent,
                weekly_status=weekly_status,
            )
        },
    )
    exhausted, reason = _quota_exhausted(entry, config)
    assert exhausted is skip
    if skip:
        assert reason.startswith("reason=quota_exhausted")
        with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
            assert resolve_provider(model_config, config=config) is None
            calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
            with pytest.raises(provider.ChainExhaustedError):
                await _drive_cycle(model_config, config)
        assert calls == []
        assert _dispatch_on_balance_logs(caplog) == []
        assert any(
            "quota_exhausted" in line for line in _skip_logs(caplog)
        ), _skip_logs(caplog)
    else:
        assert reason == ""
        calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
        response = await _drive_cycle(model_config, config)
        assert response.status_code == 200
        assert calls == ["opencode-go"]


# ---------------------------------------------------------------------------
# AC3 — fail-open / unchanged regimes (override is inert)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_snapshot_and_non_quota_entry_unchanged():
    """No authoritative quota state => no hard stop, exactly as today."""
    entry = _provider(use_balance=True)
    entry["available_times"] = []
    model_config = _single_entry_model(entry)
    config = _config()
    assert _quota_exhausted(entry, config) == (False, "")
    assert _quota_unavailable_reason(entry, config) is None
    assert resolve_provider(model_config, config=config)["name"] == "opencode-go"

    # Non-quota entry with the flag set: ignored entirely (the flag is only
    # meaningful for opencode-* entries), no hard stop.
    other = {
        "name": "deepseek",
        "type": "remote",
        "provider": "deepseek",
        "endpoint": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_KEY",
        "use_balance": True,
    }
    model_config = {"name": "m", "providers": [other]}
    assert _quota_exhausted(other, _config()) == (False, "")
    assert resolve_provider(model_config, config=_config())["name"] == "deepseek"


@pytest.mark.asyncio
async def test_stale_suspended_keeps_the_hard_stop(monkeypatch, caplog):
    """A stale 'plan spent' reading must never bill balance (AC3).

    The money guard keeps blocking when quota behaviour is stale-suspended —
    despite the balance opt-in — so a stale reading that predates a plan
    reset can never spend prepaid credit. ``_quota_unavailable_reason``
    fails open exactly as today.
    """
    entry = _provider(use_balance=True)
    model_config = _single_entry_model(entry)
    config = _config(quota_state_max_age_seconds=3600)
    await _seed(
        model_config,
        config,
        {"OPENCODE_KEY_A": _usage_body(monthly_percent=100.0, monthly_status=_PLAN_SPENT_STATUS)},
    )
    # Advance past the freshness horizon: the snapshot is now stale.
    quota_state.set_clock(lambda: NOW + timedelta(seconds=7200))
    assert quota_state.quota_behaviour(config).suspended is True
    assert _quota_unavailable_reason(entry, config) is None  # fail open, today
    exhausted, reason = _quota_exhausted(entry, config)  # money guard keeps blocking
    assert exhausted is True
    assert reason.startswith("reason=quota_exhausted")
    assert resolve_provider(model_config, config=config) is None
    calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
    with pytest.raises(provider.ChainExhaustedError):
        await _drive_cycle(model_config, config)
    assert calls == []
    assert _dispatch_on_balance_logs(caplog) == []


@pytest.mark.asyncio
async def test_disabled_kill_switch_keeps_the_hard_stop(monkeypatch, caplog):
    """With the surge kill switch off the override is inert (AC3)."""
    entry = _provider(use_balance=True)
    model_config = _single_entry_model(entry)
    config = _config(quota_surge_enabled=False)
    await _seed(
        model_config,
        config,
        {"OPENCODE_KEY_A": _usage_body(monthly_percent=100.0, monthly_status=_PLAN_SPENT_STATUS)},
    )
    assert quota_state.quota_behaviour(config).disabled is True
    exhausted, reason = _quota_exhausted(entry, config)
    assert exhausted is True
    assert reason.startswith("reason=quota_exhausted")
    assert _quota_unavailable_reason(entry, config) is None  # fail open, today
    calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        assert resolve_provider(model_config, config=config) is None
        with pytest.raises(provider.ChainExhaustedError):
            await _drive_cycle(model_config, config)
    assert calls == []
    assert _dispatch_on_balance_logs(caplog) == []


# ---------------------------------------------------------------------------
# AC4 — distinct greppable log reason, once per account
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decision_logged_distinctly_and_once_per_account(caplog):
    """The balance decision is a dedicated INFO line, not a skip reason.

    The skip path logs ``reason=quota_exhausted``; the balance path logs
    ``dispatch_on_balance (... use_balance=true)`` — and only once per
    account (mirroring the dry-run logging discipline).
    """
    entry = _provider(use_balance=True)
    model_config = _single_entry_model(entry)
    config = _config()
    await _seed(
        model_config,
        config,
        {"OPENCODE_KEY_A": _usage_body(monthly_percent=100.0, monthly_status=_PLAN_SPENT_STATUS)},
    )
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        # Two hard-stop evaluations and two full resolutions.
        _quota_exhausted(entry, config)
        resolve_provider(model_config, config=config)
        _quota_exhausted(entry, config)
        resolve_provider(model_config, config=config)

    balance_lines = _dispatch_on_balance_logs(caplog)
    assert len(balance_lines) == 1, balance_lines
    assert balance_lines[0] == (
        "dispatch_on_balance (account=OPENCODE_KEY_A, "
        "plan_monthly=100.0%, use_balance=true)"
    )
    # The two vocabularies never mix: no skip line for this decision.
    assert _skip_logs(caplog) == []


@pytest.mark.asyncio
async def test_skip_reason_vocabulary_unchanged_for_non_optin(caplog):
    """Without the opt-in the skip keeps its own reason payload (AC11/AC4)."""
    entry = _provider()  # no use_balance key
    model_config = _single_entry_model(entry)
    config = _config()
    await _seed(
        model_config,
        config,
        {"OPENCODE_KEY_A": _usage_body(monthly_percent=100.0, monthly_status=_PLAN_SPENT_STATUS)},
    )
    with caplog.at_level(logging.INFO, logger=_PROVIDER_LOGGER):
        assert resolve_provider(model_config, config=config) is None
    assert _dispatch_on_balance_logs(caplog) == []
    # The strictly-exhausted snapshot is caught by the earlier exhausted
    # gate, whose line reports the account and effective headroom.
    assert any(
        "quota_exhausted" in line
        and "account=OPENCODE_KEY_A" in line
        and "effective_headroom=0.0%" in line
        for line in _skip_logs(caplog)
    ), _skip_logs(caplog)


# ---------------------------------------------------------------------------
# AC5 — chain resolution honours the balance dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chain_resolution_dispatch_on_balance(monkeypatch):
    """The preferences chain resolver picks the balance-eligible entry.

    Two entries with the same spent plan; only the opt-in one is eligible, so
    the chain hops straight to it (skip machinery for the exhausted sibling
    stays intact).
    """
    opted_in = _provider("opencode-go", "OPENCODE_KEY_A", use_balance=True)
    default = _provider("opencode-go-2", "OPENCODE_KEY_B")
    model_config = {"name": "balance-test", "providers": [default, opted_in]}
    config = _config()
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(
                monthly_percent=100.0, monthly_status=_PLAN_SPENT_STATUS
            ),
            "OPENCODE_KEY_B": _usage_body(
                monthly_percent=100.0, monthly_status=_PLAN_SPENT_STATUS
            ),
        },
    )
    # The first (default) entry is hard-stopped; the opt-in sibling wins.
    chosen = _resolve_provider_with_exclusions(
        model_config, set(), set(), config=config
    )
    assert chosen is not None
    assert chosen["name"] == "opencode-go"
    calls = _patch_remote_cycle(monkeypatch, model_config["providers"])
    response = await _drive_cycle(model_config, config)
    assert response.status_code == 200
    assert calls == ["opencode-go"]


@pytest.mark.asyncio
async def test_balance_entry_rejected_when_not_opted_in_chain():
    """Without the opt-in the chain refuses the whole model (regression)."""
    default = _provider()  # no use_balance key
    model_config = _single_entry_model(default)
    config = _config()
    await _seed(
        model_config,
        config,
        {
            "OPENCODE_KEY_A": _usage_body(
                monthly_percent=100.0, monthly_status=_PLAN_SPENT_STATUS
            )
        },
    )
    assert _resolve_provider_with_exclusions(model_config, set(), set(), config=config) is None
