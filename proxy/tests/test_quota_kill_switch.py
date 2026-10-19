"""Tests for the quota-surge kill switch, dry run and stale-state fail-safe.

Covers LP-0MVF7KPB9009U7CG (epic LP-0MVF4FIP1001RD8P feature F6):

- AC8 — ``server.quota_surge_enabled: false`` makes routing decisions
  byte-identical to today's behaviour across a frozen ten-row decision table
  (mixed windows, quota states and chains); dry-run computes and logs the
  would-be decisions but leaves routing unchanged.
- AC8b — dry-run leaves the selected provider, the eligibility decision and
  the ``retry_after`` / next-window values unchanged, logs the decision it
  would have taken (``reason=quota_surge_dry_run``), and is honoured per
  request (no cached decision).
- AC9 — a snapshot older than ``server.quota_state_max_age_seconds``
  (default 3600 s) suspends surge: no bypass, no quota-driven re-ordering or
  skip, one WARN per account per suspension.
- AC9b — missing / never-polled state never enables surge and never blocks
  routing, and a stale snapshot never re-ranks or strands a candidate.

Every test seeds quota state through the public poller seam
(``quota_state.set_quota_transport`` + ``refresh_quota_accounts``) with a
frozen clock and asserts observable behaviour through the public routing API
(``resolve_provider`` / ``_has_next_provider`` / the retry-after helpers) plus
the documented log lines. The cost guard (``quota_exhaustion_percent``) is a
money guard and deliberately stays active regardless of the kill switch
(LP-0MVF7KO8C005NB59) — the table pins that documented difference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import proxy.provider as provider
import proxy.quota_state as quota_state
import pytest
from proxy.provider import (
    _compute_retry_after,
    _has_next_provider,
    _provider_in_allowed_window,
    _quota_ranked_providers,
    _quota_surge_bypass_active,
    _quota_unavailable_reason,
    _seconds_until_next_window,
    resolve_provider,
)
from proxy.quota_state import (
    DEFAULT_QUOTA_STATE_MAX_AGE_SECONDS,
    MODE_DISABLED,
    MODE_DRY_RUN,
    MODE_ENABLED,
    MODE_SUSPENDED,
    QuotaBehaviour,
    quota_behaviour,
    quota_state_max_age_seconds,
    reset_quota_state,
)

#: 12:00 UTC is inside the configured ``10:00-00:00`` window.
T_IN = datetime(2026, 10, 20, 12, 0, 0, tzinfo=UTC)
#: 02:00 UTC is outside the configured ``10:00-00:00`` window.
T_OUT = datetime(2026, 10, 20, 2, 0, 0, tzinfo=UTC)

_PROVIDER_LOGGER = "llama-proxy.provider"
_QUOTA_LOGGER = "llama-proxy.quota_state"

#: The default ``available_times`` window used by the table entries.
_WINDOW = ("10:00-00:00",)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _FakeDatetime:
    """Stand-in for ``proxy.provider.datetime`` with a settable wall clock."""

    _now = T_IN

    @classmethod
    def now(cls, tz=None):
        return cls._now

    @classmethod
    def fromtimestamp(cls, ts, tz=None):
        return datetime.fromtimestamp(ts, tz)


def _set_clock(now: datetime) -> None:
    """Freeze both clocks (provider window checks and the quota staleness)."""
    _FakeDatetime._now = now
    quota_state.set_clock(lambda: now)


def _advance_clock(seconds: float) -> None:
    _set_clock(_FakeDatetime._now + timedelta(seconds=seconds))


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    """Fresh quota cache, transition state and provider module state."""
    reset_quota_state()
    provider.reset_surge_bypass_state()
    _set_clock(T_IN)
    monkeypatch.setattr(provider, "datetime", _FakeDatetime)

    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    yield
    reset_quota_state()
    provider.reset_surge_bypass_state()
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _usage(
    now: datetime,
    *,
    monthly: float = 50.0,
    reset_days: float = 6.0,
    rolling: float = 0.0,
    weekly: float = 0.0,
) -> dict:
    """A ``/v1/usage`` body anchored to *now* with a configurable reset."""
    return {
        "usage": {
            "rolling": {
                "status": "ok",
                "percent": rolling,
                "resetsAt": _iso(now + timedelta(hours=5)),
            },
            "weekly": {
                "status": "ok",
                "percent": weekly,
                "resetsAt": _iso(now + timedelta(days=2)),
            },
            "monthly": {
                "status": "ok",
                "percent": monthly,
                "resetsAt": _iso(now + timedelta(days=reset_days)),
            },
        }
    }


def _opencode(
    name: str = "opencode-go",
    api_key_env: str = "OPENCODE_KEY_A",
    *,
    available_times: tuple[str, ...] = _WINDOW,
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


def _deepseek() -> dict:
    return {
        "name": "deepseek",
        "type": "remote",
        "provider": "deepseek",
        "endpoint": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_KEY",
        "model": "deepseek-chat",
    }


def _config(**server: object) -> dict:
    return {"server": dict(server)}


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


def _names(providers: list[dict] | None) -> list[str]:
    return [str(p.get("name")) for p in providers or []]


def _chosen_name(provider_cfg: dict | None) -> str | None:
    return str(provider_cfg.get("name")) if provider_cfg else None


def _log_lines(caplog, *, level: int, needle: str) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == level and needle in record.getMessage()
    ]


# ---------------------------------------------------------------------------
# AC8 — the frozen decision table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    """One representative request in the frozen decision table."""

    label: str
    now: datetime
    providers: tuple[dict, ...]
    bodies: dict[str, dict] = field(default_factory=dict)
    #: Frozen routing decision with the surge killed (and with no quota state).
    expected: str | None = None
    #: The cost guard is a money guard: with authoritative state it still
    #: skips even when the surge kill switch is off (F5 contract).
    hard_stop: bool = False


def _decision_table() -> list[_Row]:
    """Ten representative requests: windows, chains and quota states mixed."""
    two_accounts = (
        _opencode("opencode-go", "OPENCODE_KEY_A"),
        _opencode("opencode-go-3", "OPENCODE_KEY_B"),
    )
    return [
        _Row(
            "single_entry_inside_window",
            T_IN,
            (_opencode(),),
            {"OPENCODE_KEY_A": _usage(T_IN, monthly=50, reset_days=6)},
            "opencode-go",
        ),
        _Row(
            "single_entry_outside_window_in_surge_week",
            T_OUT,
            (_opencode(),),
            {"OPENCODE_KEY_A": _usage(T_OUT, monthly=50, reset_days=6)},
            None,
        ),
        _Row(
            "single_entry_outside_window_far_reset",
            T_OUT,
            (_opencode(),),
            {"OPENCODE_KEY_A": _usage(T_OUT, monthly=50, reset_days=10)},
            None,
        ),
        _Row(
            "two_accounts_headroom_inside_window",
            T_IN,
            two_accounts,
            {
                "OPENCODE_KEY_A": _usage(T_IN, monthly=88, reset_days=3),
                "OPENCODE_KEY_B": _usage(T_IN, monthly=5, reset_days=3),
            },
            "opencode-go",
        ),
        _Row(
            "two_accounts_headroom_outside_window",
            T_OUT,
            two_accounts,
            {
                "OPENCODE_KEY_A": _usage(T_OUT, monthly=88, reset_days=3),
                "OPENCODE_KEY_B": _usage(T_OUT, monthly=5, reset_days=3),
            },
            None,
        ),
        _Row(
            "two_accounts_equal_headroom_tie",
            T_IN,
            two_accounts,
            {
                "OPENCODE_KEY_A": _usage(T_IN, monthly=30, reset_days=3),
                "OPENCODE_KEY_B": _usage(T_IN, monthly=30, reset_days=3),
            },
            "opencode-go",
        ),
        _Row(
            "mixed_chain_non_quota_first",
            T_OUT,
            (_deepseek(), _opencode()),
            {"OPENCODE_KEY_A": _usage(T_OUT, monthly=50, reset_days=6)},
            "deepseek",
        ),
        _Row(
            "mixed_chain_non_quota_second",
            T_IN,
            (_opencode(), _deepseek()),
            {"OPENCODE_KEY_A": _usage(T_IN, monthly=88, reset_days=6)},
            "opencode-go",
        ),
        _Row(
            "no_window_restriction",
            T_OUT,
            (_opencode(available_times=()),),
            {"OPENCODE_KEY_A": _usage(T_OUT, monthly=50, reset_days=6)},
            "opencode-go",
        ),
        _Row(
            "hard_stop_at_monthly_cap",
            T_IN,
            (_opencode(),),
            {"OPENCODE_KEY_A": _usage(T_IN, monthly=99.6, reset_days=3)},
            "opencode-go",
            hard_stop=True,
        ),
    ]


async def _resolve_row(row: _Row, config: dict, *, seed: bool) -> str | None:
    """Resolve *row* with (``seed=True``) or without quota state present."""
    _set_clock(row.now)
    if not seed:
        # Start from an empty cache so "missing state" really has no snapshot.
        reset_quota_state()
        _set_clock(row.now)
    model_config = {
        "name": "quota-kill-switch",
        "providers": [dict(entry) for entry in row.providers],
    }
    if seed:
        await _seed(model_config, config, row.bodies)
    return _chosen_name(resolve_provider(model_config, config=config))


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _decision_table(), ids=lambda row: row.label)
async def test_kill_switch_is_byte_identical_to_today(row):
    """``quota_surge_enabled: false`` never changes the frozen decision."""
    disabled = await _resolve_row(row, _config(quota_surge_enabled=False), seed=True)
    missing = await _resolve_row(row, _config(), seed=False)

    if row.hard_stop:
        # The cost guard is independent of the surge kill switch: with
        # authoritative state the entry is still hard-stopped, while with no
        # state at all the guard fails open (F5 contract).
        assert disabled is None
        assert missing == row.expected
        return

    assert disabled == row.expected, f"kill switch changed {row.label!r}"
    # Missing / never-polled state never enables surge and never blocks
    # routing: it resolves exactly like the killed switch (child AC9b).
    assert missing == row.expected, f"missing state changed {row.label!r}"


@pytest.mark.asyncio
async def test_without_the_kill_switch_the_table_would_route_differently():
    """Control: the byte-identity assertion above is not vacuous."""
    table = {row.label: row for row in _decision_table()}

    bypass_row = table["single_entry_outside_window_in_surge_week"]
    assert await _resolve_row(bypass_row, _config(), seed=True) == "opencode-go"

    ordering_row = table["two_accounts_headroom_inside_window"]
    assert await _resolve_row(ordering_row, _config(), seed=True) == "opencode-go-3"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [row for row in _decision_table() if not row.hard_stop],
    ids=lambda row: row.label,
)
async def test_dry_run_leaves_routing_unchanged(row):
    """Dry-run still returns exactly today's decision for every table row."""
    dry = await _resolve_row(row, _config(quota_surge_dry_run=True), seed=True)
    assert dry == row.expected


# ---------------------------------------------------------------------------
# AC8b — dry-run logs the decision it would have taken
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_logs_the_bypass_it_would_apply(caplog):
    """A would-be bypass is logged with ``reason=quota_surge_dry_run``."""
    row = {r.label: r for r in _decision_table()}[
        "single_entry_outside_window_in_surge_week"
    ]
    with caplog.at_level(logging.INFO):
        assert await _resolve_row(row, _config(quota_surge_dry_run=True), seed=True) is None

    lines = _log_lines(caplog, level=logging.INFO, needle="reason=quota_surge_dry_run")
    bypass = [line for line in lines if "would_bypass=true" in line]
    assert len(bypass) == 1
    assert "account=OPENCODE_KEY_A" in bypass[0]
    assert "provider=opencode-go" in bypass[0]


@pytest.mark.asyncio
async def test_dry_run_logs_the_rerank_it_would_apply(caplog):
    """The computed ranking is logged, marked unapplied and dry-run."""
    row = {r.label: r for r in _decision_table()}[
        "two_accounts_headroom_inside_window"
    ]
    with caplog.at_level(logging.DEBUG, logger=_PROVIDER_LOGGER):
        chosen = await _resolve_row(row, _config(quota_surge_dry_run=True), seed=True)

    assert chosen == "opencode-go", "dry-run must keep config order"
    decisions = _log_lines(caplog, level=logging.DEBUG, needle="quota_ordering")
    assert len(decisions) == 2
    assert all(
        "applied=False" in line
        and "behaviour=dry_run" in line
        and "reason=quota_surge_dry_run" in line
        for line in decisions
    )
    # The would-be ranking names the account that would be preferred.
    assert "rank=1" in decisions[0]
    assert "provider=opencode-go-3" in decisions[0]


@pytest.mark.asyncio
async def test_dry_run_logs_the_skip_it_would_apply(caplog):
    """A would-be quota skip is logged but eligibility is left untouched."""
    entry = _opencode(available_times=())
    model_config = {"name": "m", "providers": [entry]}
    # A blocking rolling window makes the snapshot exhausted without touching
    # the monthly percent the cost guard keys on.
    body = _usage(T_OUT, monthly=30, reset_days=6)
    body["usage"]["rolling"]["status"] = "error"
    config = _config(quota_surge_dry_run=True)
    await _seed(model_config, config, {"OPENCODE_KEY_A": body})

    with caplog.at_level(logging.INFO):
        assert _quota_unavailable_reason(entry, config) is None
    skips = [
        line
        for line in _log_lines(
            caplog, level=logging.INFO, needle="reason=quota_surge_dry_run"
        )
        if "would_skip=true" in line
    ]
    assert len(skips) == 1
    assert "account=OPENCODE_KEY_A" in skips[0]

    # Control: with the feature live the same snapshot *does* skip the entry.
    assert _quota_unavailable_reason(entry, _config()) is not None


@pytest.mark.asyncio
async def test_dry_run_keeps_eligibility_and_retry_after_unchanged():
    """Dry-run leaves eligibility, next-window and retry_after at today's values."""
    row = {r.label: r for r in _decision_table()}[
        "single_entry_outside_window_in_surge_week"
    ]
    _set_clock(row.now)
    model_config = {
        "name": "quota-kill-switch",
        "providers": [dict(entry) for entry in row.providers],
    }
    config = _config(quota_surge_dry_run=True)
    await _seed(model_config, config, row.bodies)
    entry = model_config["providers"][0]

    # Not eligible: the configured window still applies (02:00 -> 10:00 = 8 h).
    assert _provider_in_allowed_window(entry, config, now_utc=row.now) is False
    assert _seconds_until_next_window(entry, now_utc=row.now, config=config) == 8 * 3600
    assert (
        _compute_retry_after(
            {}, model_config=model_config, now_utc=row.now, config=config
        )
        == 8 * 3600
    )

    # Control: the live feature would lift the window and remove the delay.
    live = _config()
    assert _provider_in_allowed_window(entry, live, now_utc=row.now) is True
    assert _seconds_until_next_window(entry, now_utc=row.now, config=live) is None
    assert (
        _compute_retry_after({}, model_config=model_config, now_utc=row.now, config=live)
        == 0
    )


@pytest.mark.asyncio
async def test_dry_run_flag_is_honoured_per_request():
    """The flag is re-read per request: no decision is cached."""
    row = {r.label: r for r in _decision_table()}[
        "two_accounts_headroom_inside_window"
    ]
    _set_clock(row.now)
    model_config = {
        "name": "quota-kill-switch",
        "providers": [dict(entry) for entry in row.providers],
    }
    await _seed(model_config, _config(), row.bodies)

    dry = _config(quota_surge_dry_run=True)
    live = _config()
    assert _chosen_name(resolve_provider(model_config, config=dry)) == "opencode-go"
    assert _chosen_name(resolve_provider(model_config, config=live)) == "opencode-go-3"
    assert _chosen_name(resolve_provider(model_config, config=dry)) == "opencode-go"


# ---------------------------------------------------------------------------
# AC9 / AC9b — stale state suspends surge (fail-safe)
# ---------------------------------------------------------------------------


def _stale_rows() -> dict[str, _Row]:
    return {row.label: row for row in _decision_table()}


@pytest.mark.asyncio
async def test_stale_state_suspends_the_bypass_and_warns_once(caplog):
    """A stale snapshot suspends the bypass with one WARN per account."""
    row = _stale_rows()["single_entry_outside_window_in_surge_week"]
    _set_clock(row.now)
    model_config = {
        "name": "quota-kill-switch",
        "providers": [dict(entry) for entry in row.providers],
    }
    config = _config()
    await _seed(model_config, config, row.bodies)
    entry = model_config["providers"][0]

    # Control: a fresh snapshot inside the final week lifts the window.
    assert _quota_surge_bypass_active(entry, config) is True

    _advance_clock(7200)  # twice the default 3600 s staleness horizon

    with caplog.at_level(logging.WARNING, logger=_QUOTA_LOGGER):
        for _ in range(3):
            assert _quota_surge_bypass_active(entry, config) is False
            assert (
                _provider_in_allowed_window(entry, config, now_utc=row.now) is False
            )
            assert resolve_provider(model_config, config=config) is None

    warnings = _log_lines(
        caplog, level=logging.WARNING, needle="reason=quota_surge_suspended"
    )
    assert len(warnings) == 1, warnings
    assert "account=OPENCODE_KEY_A" in warnings[0]
    assert "max_age=3600s" in warnings[0]
    assert "last_poll_age=7200s" in warnings[0]


@pytest.mark.asyncio
async def test_stale_state_reverts_ordering_to_config_order(caplog):
    """Stale snapshots suspend the whole ranking: plain config order."""
    row = _stale_rows()["two_accounts_headroom_inside_window"]
    _set_clock(row.now)
    model_config = {
        "name": "quota-kill-switch",
        "providers": [dict(entry) for entry in row.providers],
    }
    config = _config()
    await _seed(model_config, config, row.bodies)

    # Control: fresh state prefers the account with more headroom.
    assert _chosen_name(resolve_provider(model_config, config=config)) == "opencode-go-3"

    _advance_clock(7200)

    with caplog.at_level(logging.WARNING, logger=_QUOTA_LOGGER):
        for _ in range(3):
            assert _names(_quota_ranked_providers(model_config, config)) == [
                "opencode-go",
                "opencode-go-3",
            ]
            assert _chosen_name(resolve_provider(model_config, config=config)) == "opencode-go"

    warnings = _log_lines(
        caplog, level=logging.WARNING, needle="reason=quota_surge_suspended"
    )
    assert len(warnings) == 2, warnings
    assert sorted(line.split("account=")[1].split()[0] for line in warnings) == [
        "OPENCODE_KEY_A",
        "OPENCODE_KEY_B",
    ]


@pytest.mark.asyncio
async def test_stale_snapshot_suspends_ranking_instead_of_stranding(caplog):
    """A stale snapshot is not consulted: no skip, no re-rank from old data."""
    row = _stale_rows()["two_accounts_headroom_inside_window"]
    _set_clock(row.now)
    model_config = {
        "name": "quota-kill-switch",
        "providers": [dict(entry) for entry in row.providers],
    }
    config = _config()
    await _seed(model_config, config, row.bodies)
    first, second = model_config["providers"]

    # Advance past the staleness horizon and re-poll only the SECOND account,
    # so the first is stale and the second fresh.
    _advance_clock(7200)
    await _seed(
        {"providers": [second]},
        config,
        {"OPENCODE_KEY_B": _usage(_FakeDatetime._now, monthly=5, reset_days=2)},
    )

    with caplog.at_level(logging.WARNING, logger=_QUOTA_LOGGER):
        # The stale account is not skipped (no ordering-driven skip from old
        # data) and is still routable: nobody is stranded.
        assert _quota_unavailable_reason(first, config) is None
        assert (
            _has_next_provider(model_config, {second["name"]}, set(), config=config)
            is True
        )
        assert (
            _has_next_provider(model_config, {first["name"]}, set(), config=config)
            is True
        )
        # Ranking is suspended globally, so the stale reading cannot re-rank
        # the chain: config order applies (the stale snapshot said the first
        # account had little headroom left, which a fresh ranking would have
        # punished).
        assert _names(_quota_ranked_providers(model_config, config)) == [
            "opencode-go",
            "opencode-go-3",
        ]
        assert _chosen_name(resolve_provider(model_config, config=config)) == first["name"]

    warnings = _log_lines(
        caplog, level=logging.WARNING, needle="reason=quota_surge_suspended"
    )
    assert len(warnings) == 1
    assert "account=OPENCODE_KEY_A" in warnings[0]


@pytest.mark.asyncio
async def test_surge_recovers_after_a_fresh_poll(caplog):
    """A fresh poll lifts the suspension and re-logs the bypass entry."""
    row = _stale_rows()["single_entry_outside_window_in_surge_week"]
    _set_clock(row.now)
    model_config = {
        "name": "quota-kill-switch",
        "providers": [dict(entry) for entry in row.providers],
    }
    config = _config()
    await _seed(model_config, config, row.bodies)
    entry = model_config["providers"][0]

    _advance_clock(7200)
    assert _quota_surge_bypass_active(entry, config) is False
    assert quota_behaviour(config).mode == MODE_SUSPENDED

    await _seed(
        model_config, config, {"OPENCODE_KEY_A": _usage(_FakeDatetime._now, monthly=50, reset_days=6)}
    )
    assert quota_behaviour(config).mode == MODE_ENABLED
    assert _quota_surge_bypass_active(entry, config) is True


def test_missing_state_never_suspends():
    """No snapshot at all is not stale: nothing is consulted, nothing blocks."""
    entries = [_opencode(available_times=()), _opencode("opencode-go-3", "OPENCODE_KEY_B", available_times=())]
    config = _config(quota_state_max_age_seconds=1)
    assert quota_behaviour(config).mode == MODE_ENABLED
    for entry in entries:
        assert _quota_surge_bypass_active(entry, config) is False
        assert _quota_unavailable_reason(entry, config) is None
    assert _names(_quota_ranked_providers({"providers": entries}, config)) == [
        "opencode-go",
        "opencode-go-3",
    ]


# ---------------------------------------------------------------------------
# The central predicate itself
# ---------------------------------------------------------------------------


def test_behaviour_modes_and_precedence():
    """The predicate reports enabled / dry_run / suspended / disabled."""
    assert quota_behaviour(_config()).mode == MODE_ENABLED
    assert quota_behaviour(_config(quota_surge_dry_run=True)).mode == MODE_DRY_RUN
    assert quota_behaviour(_config(quota_surge_enabled=False)).mode == MODE_DISABLED

    enabled = quota_behaviour(_config())
    assert enabled.applies and enabled.computes and not enabled.dry_run
    dry = quota_behaviour(_config(quota_surge_dry_run=True))
    assert not dry.applies and dry.computes and dry.log_reason() == "quota_surge_dry_run"
    killed = quota_behaviour(_config(quota_surge_enabled=False))
    assert not killed.applies and not killed.computes and killed.disabled


@pytest.mark.asyncio
async def test_kill_switch_beats_dry_run_and_suspension(caplog):
    """The kill switch wins over both dry-run and a stale snapshot."""
    entry = _opencode()
    config = _config(quota_surge_enabled=False, quota_surge_dry_run=True)
    await _seed(
        {"providers": [entry]},
        config,
        {"OPENCODE_KEY_A": _usage(T_OUT, monthly=50, reset_days=6)},
    )
    _advance_clock(7200)

    with caplog.at_level(logging.WARNING, logger=_QUOTA_LOGGER):
        behaviour = quota_behaviour(config)
    assert behaviour.mode == MODE_DISABLED
    assert _log_lines(
        caplog, level=logging.WARNING, needle="quota surge suspended"
    ) == [], "a killed feature must not warn about staleness"


@pytest.mark.asyncio
async def test_stale_suspension_beats_dry_run():
    """Fail-safe ordering: a stale snapshot suspends even in dry-run."""
    entry = _opencode()
    config = _config(quota_surge_dry_run=True)
    await _seed(
        {"providers": [entry]},
        config,
        {"OPENCODE_KEY_A": _usage(T_OUT, monthly=50, reset_days=6)},
    )
    _advance_clock(7200)
    assert quota_behaviour(config).mode == MODE_SUSPENDED


def test_state_max_age_config_parsing():
    """``quota_state_max_age_seconds`` defaults and fails safe."""
    assert quota_state_max_age_seconds(None) == DEFAULT_QUOTA_STATE_MAX_AGE_SECONDS
    assert quota_state_max_age_seconds({}) == DEFAULT_QUOTA_STATE_MAX_AGE_SECONDS
    assert quota_state_max_age_seconds({"server": {"quota_state_max_age_seconds": 7200}}) == 7200.0
    assert quota_state_max_age_seconds({"server": {"quota_state_max_age_seconds": "90"}}) == 90.0
    # Invalid values fail safe: fall back to the default, or clamp negatives
    # to "always stale" (0 s) — never to an unbounded trust in stale data.
    assert (
        quota_state_max_age_seconds({"server": {"quota_state_max_age_seconds": "abc"}})
        == DEFAULT_QUOTA_STATE_MAX_AGE_SECONDS
    )
    assert quota_state_max_age_seconds({"server": {"quota_state_max_age_seconds": -5}}) == 0.0


@pytest.mark.asyncio
async def test_every_quota_decision_site_consults_the_predicate(monkeypatch):
    """Disabling the predicate switches ordering, bypass and skip off at once."""
    row = _stale_rows()["two_accounts_headroom_outside_window"]
    _set_clock(row.now)
    model_config = {
        "name": "quota-kill-switch",
        "providers": [dict(entry) for entry in row.providers],
    }
    config = _config()
    await _seed(model_config, config, row.bodies)
    entry = model_config["providers"][0]

    exhausted_entry = _opencode(api_key_env="OPENCODE_KEY_C", available_times=())
    body = _usage(row.now, monthly=30, reset_days=6)
    body["usage"]["rolling"]["status"] = "error"
    await _seed(
        {"providers": [exhausted_entry]},
        config,
        {"OPENCODE_KEY_C": body},
    )

    # Sanity: with the real predicate all three decisions act.
    assert _quota_surge_bypass_active(entry, config) is True
    assert _names(_quota_ranked_providers(model_config, config)) == [
        "opencode-go-3",
        "opencode-go",
    ]
    assert _quota_unavailable_reason(exhausted_entry, config) is not None

    monkeypatch.setattr(
        provider,
        "_quota_behaviour",
        lambda *args, **kwargs: QuotaBehaviour(MODE_DISABLED),
    )

    assert _quota_surge_bypass_active(entry, config) is False
    assert _quota_unavailable_reason(exhausted_entry, config) is None
    assert _names(_quota_ranked_providers(model_config, config)) == [
        "opencode-go",
        "opencode-go-3",
    ]
