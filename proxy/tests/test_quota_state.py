"""Tests for proactive per-account quota state (LP-0MVF7KEG7007C8MF).

These tests exercise the tracer-bullet quota poller end to end through its
public API: the injectable ``/v1/usage`` transport (the shared harness later
quota-surge features build on), the derivation model, the interval/single-
flight coalescing and the fail-open contract. No test touches the network or
the wall clock.

Covered acceptance criteria:

- AC1: at most one request per account per ``quota_poll_interval_seconds``,
  and concurrent refreshes coalesce into a single in-flight request.
- AC1b: 500 / timeout / malformed body keeps the previous values, logs at
  WARN and never raises (so a client request is never failed or delayed).
- AC1c: two entries sharing one ``api_key_env`` share one snapshot and one
  poll.
- AC2a: derived rolling/weekly/monthly percent + ``resetsAt`` +
  ``seconds_to_reset`` + ``effective_headroom``, with effective headroom the
  minimum of the three windows.
- AC2b: a non-``ok`` status is conservatively blocking/exhausted, and a
  missing / invalid ``resetsAt`` yields ``seconds_to_reset <= 0``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone

import httpx
import proxy.quota_state as quota_state
import pytest
from proxy.quota_state import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    account_key,
    get_quota_snapshot,
    is_quota_provider,
    parse_usage_payload,
    poll_interval_seconds,
    refresh_quota_accounts,
    reset_quota_state,
    snapshot_from_usage,
    surge_accounts,
)

NOW = datetime(2026, 10, 19, 12, 0, 0, tzinfo=UTC)

_RATELIMIT_LOGGER = "llama-proxy.quota_state"

# Default reset offsets (seconds) used by the payload builder.
_ROLLING_RESET = 5 * 3600
_WEEKLY_RESET = 7 * 86400
_MONTHLY_RESET = 14 * 86400


class _Clock:
    """Deterministic wall clock the poller reads through ``set_clock``."""

    def __init__(self, start: datetime = NOW):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


@pytest.fixture(autouse=True)
def clock():
    """Fresh module state and a frozen clock for every test."""
    reset_quota_state()
    frozen = _Clock()
    quota_state.set_clock(frozen)
    yield frozen
    reset_quota_state()


# ---------------------------------------------------------------------------
# Fixtures / payload builders
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _window(percent, status, resets_at) -> dict:
    return {"status": status, "percent": percent, "resetsAt": resets_at}


def _usage_body(
    rolling: float = 0,
    weekly: float = 0,
    monthly: float = 0,
    *,
    rolling_status: str = "ok",
    weekly_status: str = "ok",
    monthly_status: str = "ok",
    rolling_resets: float | None = _ROLLING_RESET,
    weekly_resets: float | None = _WEEKLY_RESET,
    monthly_resets: float | None = _MONTHLY_RESET,
) -> dict:
    def resets(offset):
        return _iso(NOW + timedelta(seconds=offset)) if offset is not None else None

    return {
        "usage": {
            "rolling": _window(rolling, rolling_status, resets(rolling_resets)),
            "weekly": _window(weekly, weekly_status, resets(weekly_resets)),
            "monthly": _window(monthly, monthly_status, resets(monthly_resets)),
        }
    }


def _provider(
    name: str = "opencode-go",
    api_key_env: str = "OPENCODE_API_KEY",
    model: str = "deepseek-v4-flash",
    **extra,
) -> dict:
    return {
        "name": name,
        "type": "remote",
        "provider": "opencode-go",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": api_key_env,
        "model": model,
        **extra,
    }


def _model_config(*providers: dict) -> dict:
    return {"providers": list(providers)}


def _config(interval: float = 900, **server_extra) -> dict:
    return {"server": {"quota_poll_interval_seconds": interval, **server_extra}}


class _FakeTransport:
    """Injectable transport recording call and account counts."""

    def __init__(self, body: dict | None = None, exc: Exception | None = None, delay: float = 0.0):
        self.body = body
        self.exc = exc
        self.delay = delay
        self.calls = 0
        self.accounts: list[str] = []

    async def __call__(self, provider_cfg: dict) -> dict:
        self.calls += 1
        self.accounts.append(account_key(provider_cfg))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return self.body


async def _refresh(transport: _FakeTransport, model_config: dict, config: dict):
    quota_state.set_quota_transport(transport)
    return await refresh_quota_accounts(model_config, config)


# ---------------------------------------------------------------------------
# Cache reads and config helpers
# ---------------------------------------------------------------------------


def test_snapshot_is_none_before_first_poll(clock):
    """A cache read for an unknown account is ``None`` and never blocks."""
    assert get_quota_snapshot(_provider(), _config()) is None


def test_account_key_uses_api_key_env_not_model():
    """Quota is per API key: the model must not split the account key (AC1c)."""
    a = _provider(name="opencode-go", model="deepseek-v4-flash")
    b = _provider(name="opencode-go-2", model="deepseek-v4-pro")
    assert account_key(a) == account_key(b) == "OPENCODE_API_KEY"


def test_config_defaults_and_overrides():
    assert poll_interval_seconds({}) == DEFAULT_POLL_INTERVAL_SECONDS
    assert poll_interval_seconds({"server": {"quota_poll_interval_seconds": 60}}) == 60
    # malformed and non-positive values fall back to the default / floor
    assert poll_interval_seconds({"server": {"quota_poll_interval_seconds": "nope"}}) == 900
    assert poll_interval_seconds({"server": {"quota_poll_interval_seconds": 0}}) == 1
    assert surge_accounts({}) is None
    assert surge_accounts({"server": {"quota_surge_accounts": ["opencode-go"]}}) == ["opencode-go"]
    assert surge_accounts({"server": {"quota_surge_accounts": "opencode-go, opencode-go-2"}}) == [
        "opencode-go",
        "opencode-go-2",
    ]


def test_is_quota_provider_scopes_to_opencode():
    assert is_quota_provider(_provider())
    assert not is_quota_provider({"name": "deepseek", "provider": "deepseek", "api_key_env": "X"})
    # an allow-list restricts by entry name or api_key_env
    config = {"server": {"quota_surge_accounts": ["opencode-go"]}}
    assert is_quota_provider(_provider(name="opencode-go"), config)
    assert not is_quota_provider(_provider(name="opencode-go-2", api_key_env="OPENCODE_2_API_KEY"), config)


# ---------------------------------------------------------------------------
# AC1 — interval gating and single-flight coalescing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interval_gating_allows_one_poll_per_interval(clock):
    """AC1: many refreshes inside one interval produce exactly one request."""
    transport = _FakeTransport(_usage_body())
    model_config = _model_config(_provider())
    config = _config(interval=900)

    for _ in range(10):
        await _refresh(transport, model_config, config)

    assert transport.calls == 1
    snapshot = get_quota_snapshot(model_config["providers"][0], config)
    assert snapshot is not None
    assert snapshot.monthly.percent == 0

    # Once the interval elapses exactly one further poll is allowed.
    clock.advance(901)
    await _refresh(transport, model_config, config)
    assert transport.calls == 2


@pytest.mark.asyncio
async def test_concurrent_refreshes_coalesce_into_one_request(clock):
    """AC1: concurrent refreshes share a single in-flight request per account."""
    transport = _FakeTransport(_usage_body(monthly=10), delay=0.05)
    model_config = _model_config(_provider())
    config = _config()
    quota_state.set_quota_transport(transport)
    await asyncio.gather(*(refresh_quota_accounts(model_config, config) for _ in range(5)))

    assert transport.calls == 1
    assert transport.accounts == ["OPENCODE_API_KEY"]


@pytest.mark.asyncio
async def test_cache_read_does_not_wait_for_inflight_poll(clock):
    """AC1b: a client-facing cache read ignores an in-flight poll."""
    transport = _FakeTransport(_usage_body(), delay=0.2)
    model_config = _model_config(_provider())
    config = _config()
    quota_state.set_quota_transport(transport)

    poll = asyncio.ensure_future(refresh_quota_accounts(model_config, config))
    await asyncio.sleep(0.01)
    # While the poll is still running, the read returns immediately (no snapshot).
    assert get_quota_snapshot(model_config["providers"][0], config) is None
    assert not poll.done()
    await poll
    assert get_quota_snapshot(model_config["providers"][0], config) is not None


# ---------------------------------------------------------------------------
# AC1b — fail open on poll errors
# ---------------------------------------------------------------------------


def _http_500() -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://opencode.ai/zen/go/v1/usage")
    response = httpx.Response(500, request=request)
    return httpx.HTTPStatusError("500", request=request, response=response)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http_500", "timeout", "malformed_body"])
async def test_poll_failure_keeps_previous_values_and_warns(clock, caplog, failure):
    """AC1b: a failed poll keeps the last good values and logs at WARN."""
    transport = _FakeTransport(_usage_body(monthly=40))
    model_config = _model_config(_provider())
    config = _config()
    await _refresh(transport, model_config, config)
    before = get_quota_snapshot(model_config["providers"][0], config)
    assert before.monthly.percent == 40

    clock.advance(1000)
    if failure == "http_500":
        transport.exc = _http_500()
    elif failure == "timeout":
        transport.exc = TimeoutError()
    else:
        transport.body = {"usage": "not-a-mapping"}

    with caplog.at_level("WARNING", logger=_RATELIMIT_LOGGER):
        result = await refresh_quota_accounts(model_config, config)

    after = get_quota_snapshot(model_config["providers"][0], config)
    assert after is not None
    assert after.monthly.percent == 40
    assert after.effective_headroom == before.effective_headroom
    assert after.last_success_at == before.last_success_at
    assert after.last_status != "ok"
    assert result["OPENCODE_API_KEY"].last_status != "ok"
    assert any(record.levelname == "WARNING" for record in caplog.records)


@pytest.mark.asyncio
async def test_poll_failure_without_prior_success_does_not_raise(clock):
    """AC1b: a first-ever poll failure fails open and caches nothing."""
    transport = _FakeTransport(exc=RuntimeError("boom"))
    model_config = _model_config(_provider())
    config = _config()

    result = await _refresh(transport, model_config, config)

    assert result == {}
    assert get_quota_snapshot(model_config["providers"][0], config) is None


# ---------------------------------------------------------------------------
# AC2a — derivation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derivation_reports_windows_and_minimum_headroom(clock):
    """AC2a: rolling=95, weekly=10, monthly=40 → effective headroom 5."""
    transport = _FakeTransport(_usage_body(rolling=95, weekly=10, monthly=40))
    model_config = _model_config(_provider())
    config = _config()

    await _refresh(transport, model_config, config)
    snapshot = get_quota_snapshot(model_config["providers"][0], config)

    assert snapshot is not None
    assert snapshot.rolling.percent == 95
    assert snapshot.weekly.percent == 10
    assert snapshot.monthly.percent == 40
    assert snapshot.rolling_headroom == pytest.approx(5)
    assert snapshot.weekly_headroom == pytest.approx(90)
    assert snapshot.monthly_headroom == pytest.approx(60)
    assert snapshot.effective_headroom == pytest.approx(5)
    assert snapshot.exhausted is False
    assert snapshot.monthly.resets_at == NOW + timedelta(seconds=_MONTHLY_RESET)
    assert snapshot.seconds_to_reset == pytest.approx(_MONTHLY_RESET)
    assert snapshot.last_status == "ok"
    assert snapshot.last_success_at == NOW


def test_snapshot_from_usage_parses_iso_z_and_offsets():
    body = _usage_body(monthly=12)
    body["usage"]["monthly"]["resetsAt"] = "2026-11-02T12:00:00+00:00"
    snapshot = snapshot_from_usage("acct", body, fetched_at=NOW)
    assert snapshot.monthly.resets_at == datetime(2026, 11, 2, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# AC2b — conservative status handling and reset parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["rate_limited", "exceeded", "exhausted", "anything"])
async def test_non_ok_window_status_is_blocking(clock, status):
    """AC2b: any status other than ``ok`` is conservatively exhausted."""
    transport = _FakeTransport(
        _usage_body(rolling=10, weekly=10, monthly=10, rolling_status=status)
    )
    model_config = _model_config(_provider())
    config = _config()

    await _refresh(transport, model_config, config)
    snapshot = get_quota_snapshot(model_config["providers"][0], config)

    assert snapshot.rolling.blocking is True
    assert snapshot.rolling.headroom == 0
    assert snapshot.effective_headroom == 0
    assert snapshot.exhausted is True


@pytest.mark.asyncio
async def test_missing_or_invalid_resets_at_yields_zero(clock):
    """AC2b: a missing / unparseable resetsAt is never guessed."""
    transport = _FakeTransport(_usage_body(monthly=50, monthly_resets=None))
    model_config = _model_config(_provider())
    config = _config()
    await _refresh(transport, model_config, config)
    snapshot = get_quota_snapshot(model_config["providers"][0], config)
    assert snapshot.monthly.seconds_to_reset <= 0
    assert snapshot.seconds_to_reset <= 0

    # An unparseable string behaves the same way.
    clock.advance(1000)
    transport.body = _usage_body(monthly=50)
    transport.body["usage"]["monthly"]["resetsAt"] = "not-a-date"
    await _refresh(transport, model_config, config)
    snapshot = get_quota_snapshot(model_config["providers"][0], config)
    assert snapshot.monthly.seconds_to_reset <= 0
    assert snapshot.seconds_to_reset <= 0


def test_missing_window_is_treated_as_blocking():
    """AC2b: a window absent from the payload is conservatively spent."""
    body = _usage_body(rolling=0, weekly=0, monthly=0)
    del body["usage"]["monthly"]
    windows = parse_usage_payload(body)
    assert windows["monthly"].present is False
    assert windows["monthly"].blocking is True
    snapshot = snapshot_from_usage("acct", body, fetched_at=NOW)
    assert snapshot.exhausted is True
    assert snapshot.effective_headroom == 0
    assert snapshot.seconds_to_reset <= 0


# ---------------------------------------------------------------------------
# AC1c — shared account de-duplication
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shared_api_key_env_shares_one_snapshot_and_poll(clock):
    """AC1c: two entries sharing one api_key_env → one snapshot, one poll."""
    transport = _FakeTransport(_usage_body(monthly=50))
    primary = _provider(name="opencode-go", model="deepseek-v4-flash")
    sibling = _provider(name="opencode-go-2", model="deepseek-v4-pro")
    model_config = _model_config(primary, sibling)
    config = _config()

    await _refresh(transport, model_config, config)

    assert transport.calls == 1
    first = get_quota_snapshot(primary, config)
    second = get_quota_snapshot(sibling, config)
    assert first is not None
    assert first is second
    assert first.account == "OPENCODE_API_KEY"


@pytest.mark.asyncio
async def test_distinct_api_keys_poll_once_each(clock):
    """Distinct accounts are polled independently (one request each)."""
    transport = _FakeTransport(_usage_body(monthly=10))
    model_config = _model_config(
        _provider(name="opencode-go", api_key_env="OPENCODE_API_KEY"),
        _provider(name="opencode-go-2", api_key_env="OPENCODE_2_API_KEY"),
    )

    await _refresh(transport, model_config, _config())

    assert transport.calls == 2
    assert sorted(transport.accounts) == ["OPENCODE_2_API_KEY", "OPENCODE_API_KEY"]


@pytest.mark.asyncio
async def test_non_opencode_entries_are_not_polled(clock):
    """The poller only touches ``opencode-*`` entries by default."""
    transport = _FakeTransport(_usage_body())
    model_config = _model_config(
        {"name": "deepseek", "provider": "deepseek", "api_key_env": "DEEPSEEK_API_KEY"}
    )

    await _refresh(transport, model_config, _config())

    assert transport.calls == 0


# ---------------------------------------------------------------------------
# Background task wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_and_stop_quota_poller(clock):
    """The background poller runs one cycle and stops cleanly on shutdown."""
    transport = _FakeTransport(_usage_body(monthly=33))
    quota_state.set_quota_transport(transport)
    config = {
        "server": {"quota_poll_interval_seconds": 900},
        "providers": [_provider()],
    }

    task = quota_state.start_quota_poller(config)
    assert task is not None
    await asyncio.sleep(0.02)

    await quota_state.stop_quota_poller()

    assert transport.calls >= 1
    assert task.done()
    snapshot = get_quota_snapshot(_provider(), config)
    assert snapshot is not None
    assert snapshot.monthly.percent == 33


@pytest.mark.asyncio
async def test_quota_poll_loop_survives_cycle_errors(clock, caplog):
    """A failed poll cycle is logged and the loop keeps running."""
    transport = _FakeTransport(exc=RuntimeError("endpoint down"))
    quota_state.set_quota_transport(transport)
    config = {"server": {"quota_poll_interval_seconds": 900}, "providers": [_provider()]}

    with caplog.at_level("WARNING", logger=_RATELIMIT_LOGGER):
        task = quota_state.start_quota_poller(config)
        await asyncio.sleep(0.02)
        assert not task.done()
        await quota_state.stop_quota_poller()

    assert any(record.levelname == "WARNING" for record in caplog.records)
