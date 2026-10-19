"""Tests for the quota observability surface (LP-0MVF7KQED005TG9Z, epic
LP-0MVF4FIP1001RD8P feature F6).

Goal: let the operator answer "how much is left, when does it reset, is surge
active" from the existing status/metrics surfaces, without grepping logs.

Every test seeds quota state through the public poller seam
(``quota_state.set_quota_transport`` + ``refresh_quota_accounts``) with a
frozen clock, then asserts the **exposed** values through the public surface
that renders them (the ``/events`` payload builder, the home-page row builder
and the Prometheus metrics API).

Covered acceptance criteria:

- AC2: the derived per-account state (rolling/weekly/monthly percent,
  ``resetsAt``, ``seconds_to_reset``, ``effective_headroom``, ``surge_active``,
  ``exhausted``) is exposed — with ``effective_headroom`` the **minimum** of the
  three windows (the documented ``rolling=95, weekly=10, monthly=40`` feed
  yields ``5``).
- AC2b: the ``/events`` status payload and the home-page rows carry the same
  per-account fields and degrade cleanly (``quota: null`` / no quota cell) when
  quota state is absent, without raising.
- AC2c: Prometheus export works when ``prometheus_client`` is installed and is a
  no-op (no exception, plaintext fallback) when it is not; no API key **value**
  ever appears in an exposed field — only the account key derived from
  ``api_key_env``.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import proxy.metrics as metrics
import proxy.quota_state as quota_state
import pytest
from proxy.quota_state import (
    format_quota_status,
    quota_status_payload,
    refresh_quota_accounts,
    reset_quota_state,
)

NOW = datetime(2026, 10, 19, 12, 0, 0, tzinfo=UTC)

#: Monthly reset used by most tests (14 days out — outside the surge window).
_MONTHLY_RESET = 14 * 86400
_ROLLING_RESET = 5 * 3600
_WEEKLY_RESET = 7 * 86400

#: A key **value** that must never be exposed on any surface.
_SECRET = "sk-live-do-not-leak-0123456789"


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
    _clear_quota_gauge_stores()
    yield frozen
    reset_quota_state()


#: The repo-root ``prometheus_client.py`` test stub lets a labelled gauge be
#: set only once (after ``set`` its per-label slot holds the plain value, so a
#: second ``labels().set()`` raises AttributeError). Clearing the per-label
#: store before every test keeps each test observing only its own writes; the
#: real client allows repeated ``set`` and needs no reset.
_QUOTA_GAUGES = (
    metrics.quota_rolling_percent,
    metrics.quota_weekly_percent,
    metrics.quota_monthly_percent,
    metrics.quota_effective_headroom,
    metrics.quota_seconds_to_reset,
    metrics.quota_resets_at_timestamp_seconds,
    metrics.quota_surge_active,
    metrics.quota_exhausted,
    metrics.quota_last_poll_timestamp_seconds,
)


def _clear_quota_gauge_stores() -> None:
    for gauge in _QUOTA_GAUGES:
        store = getattr(gauge, "_store", None)
        if isinstance(store, dict):
            store.clear()


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
    monthly_resets: float | None = _MONTHLY_RESET,
) -> dict:
    def resets(offset):
        return _iso(NOW + timedelta(seconds=offset)) if offset is not None else None

    return {
        "usage": {
            "rolling": _window(rolling, "ok", resets(_ROLLING_RESET)),
            "weekly": _window(weekly, "ok", resets(_WEEKLY_RESET)),
            "monthly": _window(monthly, "ok", resets(monthly_resets)),
        }
    }


def _provider(
    name: str = "opencode-go",
    api_key_env: str = "OPENCODE_API_KEY",
    **extra,
) -> dict:
    return {
        "name": name,
        "type": "remote",
        "provider": "opencode-go",
        "endpoint": "https://opencode.ai/zen/go",
        "api_key_env": api_key_env,
        "model": "deepseek-v4-flash",
        **extra,
    }


def _config(providers: list[dict] | None = None, **server_extra) -> dict:
    config: dict = {"server": {"quota_poll_interval_seconds": 900, **server_extra}}
    if providers is not None:
        config["providers"] = list(providers)
    return config


class _FakeTransport:
    """Injectable ``/v1/usage`` transport (never touches the network)."""

    def __init__(self, body: dict | None = None, exc: Exception | None = None):
        self.body = body
        self.exc = exc
        self.calls = 0

    async def __call__(self, provider_cfg: dict) -> dict:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.body


async def _poll(providers, config, body=None, *, exc=None, force=False):
    transport = _FakeTransport(body=body, exc=exc)
    quota_state.set_quota_transport(transport)
    return await refresh_quota_accounts(
        {"providers": list(providers)}, config, force=force
    )


def _poll_sync(providers, config, body=None, *, exc=None, force=False):
    return asyncio.run(_poll(providers, config, body, exc=exc, force=force))


# ---------------------------------------------------------------------------
# Metric reader helpers (works with the real client and the repo-root stub)
# ---------------------------------------------------------------------------


def _gauge_value(gauge, **labels):
    """Read a labelled gauge value from either prometheus_client build.

    The repo-root ``prometheus_client.py`` test stub returns the stored int
    directly from ``labels()`` (after ``set`` the mapping slot holds the value)
    instead of a labelled child, so the reader adapts to whichever
    implementation the environment loaded.
    """
    child = gauge.labels(**labels)
    if isinstance(child, (int, float)):
        return child
    value = getattr(child, "_value", None)
    if value is not None and hasattr(value, "get"):
        return value.get()
    store = getattr(child, "_store", None)
    key = getattr(child, "_key", None)
    assert store is not None and key in store, "unreadable gauge implementation"
    return store[key]


def _counter_value(counter, **labels) -> int:
    return int(counter.labels(**labels)._value.get())


LABELS = {"account": "OPENCODE_API_KEY", "entry": "opencode-go"}


# ---------------------------------------------------------------------------
# AC2 — derived state is exposed (metrics + status payload)
# ---------------------------------------------------------------------------


def test_status_payload_exposes_derived_state():
    """AC2: the documented feed exposes every derived field per account.

    ``rolling=95, weekly=10, monthly=40`` → effective headroom is the
    **minimum** of the three window headrooms (5), not the monthly one (60).
    """
    provider = _provider()
    config = _config([provider])
    _poll_sync([provider], config, _usage_body(rolling=95, weekly=10, monthly=40))

    payload = quota_status_payload(config)
    assert payload is not None
    assert payload["mode"] == "enabled"
    assert payload["enabled"] is True
    assert payload["poll_interval_seconds"] == 900

    (account,) = payload["accounts"]
    assert account["account"] == "OPENCODE_API_KEY"
    assert account["entry"] == "opencode-go"
    assert account["rolling"]["percent"] == 95
    assert account["weekly"]["percent"] == 10
    assert account["monthly"]["percent"] == 40
    assert account["effective_headroom"] == 5
    assert account["exhausted"] is False
    assert account["seconds_to_reset"] == _MONTHLY_RESET
    assert account["resetsAt"] == _iso(NOW + timedelta(seconds=_MONTHLY_RESET))
    assert account["rolling"]["resetsAt"] == _iso(NOW + timedelta(seconds=_ROLLING_RESET))
    assert account["last_status"] == "ok"
    assert account["stale"] is False


def test_status_payload_reports_surge_active_in_final_week():
    """AC2: ``surge_active`` is True inside the final-week window, False outside."""
    provider = _provider()
    config = _config([provider])

    _poll_sync([provider], config, _usage_body(monthly=40, monthly_resets=6 * 86400))
    (account,) = quota_status_payload(config)["accounts"]
    assert account["surge_active"] is True

    reset_quota_state()
    quota_state.set_clock(_Clock())
    _poll_sync([provider], config, _usage_body(monthly=40, monthly_resets=_MONTHLY_RESET))
    (account,) = quota_status_payload(config)["accounts"]
    assert account["surge_active"] is False


def test_surge_active_follows_the_central_behaviour_predicate():
    """AC2: dry-run and the kill switch mean surge is *not* active."""
    provider = _provider()
    in_window = _usage_body(monthly=40, monthly_resets=6 * 86400)

    dry_run = _config([provider], quota_surge_dry_run=True)
    _poll_sync([provider], dry_run, in_window)
    (account,) = quota_status_payload(dry_run)["accounts"]
    assert account["surge_active"] is False

    reset_quota_state()
    quota_state.set_clock(_Clock())
    disabled = _config([provider], quota_surge_enabled=False)
    _poll_sync([provider], disabled, in_window)
    payload = quota_status_payload(disabled)
    assert payload["mode"] == "disabled"
    assert payload["enabled"] is False
    assert payload["accounts"][0]["surge_active"] is False


def test_exhausted_account_is_exposed():
    """AC2: a blocking window marks the account exhausted with zero headroom."""
    provider = _provider()
    config = _config([provider])
    body = _usage_body(monthly=40, monthly_resets=6 * 86400)
    body["usage"]["weekly"]["status"] = "exhausted"
    _poll_sync([provider], config, body)

    (account,) = quota_status_payload(config)["accounts"]
    assert account["exhausted"] is True
    assert account["effective_headroom"] == 0
    assert account["surge_active"] is False


def test_metrics_gauges_carry_the_exposed_values():
    """AC2: every documented Prometheus gauge carries the exposed value."""
    assert metrics._enabled, "prometheus_client must be available in the test env"
    provider = _provider()
    config = _config([provider])
    _poll_sync([provider], config, _usage_body(rolling=95, weekly=10, monthly=40))

    assert _gauge_value(metrics.quota_rolling_percent, **LABELS) == 95
    assert _gauge_value(metrics.quota_weekly_percent, **LABELS) == 10
    assert _gauge_value(metrics.quota_monthly_percent, **LABELS) == 40
    assert _gauge_value(metrics.quota_effective_headroom, **LABELS) == 5
    assert _gauge_value(metrics.quota_seconds_to_reset, **LABELS) == _MONTHLY_RESET
    assert int(_gauge_value(metrics.quota_resets_at_timestamp_seconds, **LABELS)) == int(
        (NOW + timedelta(seconds=_MONTHLY_RESET)).timestamp()
    )
    assert _gauge_value(metrics.quota_surge_active, **LABELS) == 0
    assert _gauge_value(metrics.quota_exhausted, **LABELS) == 0
    assert int(_gauge_value(metrics.quota_last_poll_timestamp_seconds, **LABELS)) == int(
        NOW.timestamp()
    )


def test_metrics_surge_and_exhausted_flags_track_state():
    """AC2: the boolean gauges flip with the exposed state (1 = true)."""
    provider = _provider()
    config = _config([provider])
    _poll_sync(
        [provider], config, _usage_body(monthly=100, monthly_resets=6 * 86400)
    )

    assert _gauge_value(metrics.quota_exhausted, **LABELS) == 1
    assert _gauge_value(metrics.quota_effective_headroom, **LABELS) == 0
    # An exhausted account is never surge-eligible.
    assert _gauge_value(metrics.quota_surge_active, **LABELS) == 0


def test_metrics_surge_flag_rises_inside_the_final_week():
    """AC2: the surge gauge is 1 only when surge actually applies."""
    provider = _provider()
    config = _config([provider])
    _poll_sync(
        [provider], config, _usage_body(monthly=40, monthly_resets=6 * 86400)
    )

    assert _gauge_value(metrics.quota_surge_active, **LABELS) == 1
    assert _gauge_value(metrics.quota_exhausted, **LABELS) == 0


def test_poll_failure_increments_the_failure_counter_and_keeps_last_known():
    """AC2: a failed poll is counted by reason; previous values are kept."""
    provider = _provider()
    config = _config([provider])
    _poll_sync([provider], config, _usage_body(monthly=40))

    before = _counter_value(
        metrics.quota_poll_failures_total, account="OPENCODE_API_KEY", reason="QuotaPollError"
    )
    _poll_sync(
        [provider], config, exc=quota_state.QuotaPollError("boom"), force=True
    )

    after = _counter_value(
        metrics.quota_poll_failures_total, account="OPENCODE_API_KEY", reason="QuotaPollError"
    )
    assert after == before + 1
    # Fail-open: the last known-good values survive the failed poll.
    (account,) = quota_status_payload(config)["accounts"]
    assert account["monthly"]["percent"] == 40
    assert account["last_status"].startswith("poll_error")


# ---------------------------------------------------------------------------
# AC2b — /events status payload and home-page rows
# ---------------------------------------------------------------------------


def _mock_srv(config: dict) -> MagicMock:
    srv = MagicMock()
    srv.per_model_queries = {}
    srv.per_model_queries_lock = MagicMock()
    srv.sse_clients = set()
    srv.current_model = "Qwen3"
    srv.token_counts = {"total_sent": 100, "total_recv": 200}
    srv.config = config
    srv.query_llama_status = AsyncMock(
        return_value={
            "llama_server_running": False,
            "n_ctx": None,
            "kv_cache_tokens": None,
            "router_mode": False,
        }
    )
    srv.router_list_models = AsyncMock(return_value=[])
    return srv


async def _first_status_event(srv) -> dict:
    from proxy.ui import status_events

    with patch("proxy.ui._srv", return_value=srv):
        response = await status_events()
    chunk = await asyncio.wait_for(response.body_iterator.__anext__(), timeout=3.0)
    raw = chunk if isinstance(chunk, str) else chunk.decode("utf-8")
    for line in raw.splitlines():
        if line.startswith("data:"):
            return json.loads(line[len("data:"):].strip())
    raise AssertionError(f"no SSE data line in payload: {raw!r}")


@pytest.mark.asyncio
async def test_status_events_payload_carries_the_quota_section():
    """AC2b: ``/events`` exposes the same per-account fields."""
    provider = _provider()
    config = _config([provider])
    await _poll([provider], config, _usage_body(rolling=95, weekly=10, monthly=40))

    payload = await _first_status_event(_mock_srv(config))

    quota = payload["quota"]
    assert quota["mode"] == "enabled"
    (account,) = quota["accounts"]
    assert account["account"] == "OPENCODE_API_KEY"
    assert account["rolling"]["percent"] == 95
    assert account["weekly"]["percent"] == 10
    assert account["monthly"]["percent"] == 40
    assert account["effective_headroom"] == 5
    assert account["seconds_to_reset"] == _MONTHLY_RESET
    assert account["resetsAt"] == _iso(NOW + timedelta(seconds=_MONTHLY_RESET))
    assert account["surge_active"] is False
    assert account["exhausted"] is False


@pytest.mark.asyncio
async def test_status_events_reports_null_quota_when_state_is_absent():
    """AC2b: no poll yet → ``quota: null`` and no exception."""
    payload = await _first_status_event(_mock_srv(_config()))
    assert "quota" in payload
    assert payload["quota"] is None


@pytest.mark.asyncio
async def test_status_events_degrades_when_quota_lookup_raises():
    """AC2b: a broken quota surface never breaks the SSE status stream."""
    srv = _mock_srv(_config())
    with patch(
        "proxy.ui.quota_state.quota_status_payload",
        side_effect=RuntimeError("boom"),
    ):
        payload = await _first_status_event(srv)
    assert payload["quota"] is None
    assert payload["type"] == "status"


def test_home_page_rows_show_quota_fields():
    """AC2b: the home-page row shows allowance, reset countdown and surge."""
    from proxy.ui import _build_home_model_rows

    provider = _provider()
    config = {
        "server": {},
        "models": {"m": {"type": "remote", "providers": [provider]}},
    }
    _poll_sync(
        [provider], config, _usage_body(monthly=95, monthly_resets=6 * 86400)
    )

    html = _build_home_model_rows(_mock_srv(config)).replace("&gt;", ">")
    assert 'class="provider-quota"' in html
    assert "5% eff" in html
    assert "monthly 95%" in html
    assert "resets 6d" in html
    assert "surge" in html


def test_home_page_rows_omit_quota_without_state():
    """AC2b: no quota state → no quota cell, and no exception."""
    from proxy.ui import _build_home_model_rows

    config = {"server": {}, "models": {"m": {"providers": [_provider()]}}}
    html = _build_home_model_rows(_mock_srv(config))
    assert "provider-quota" not in html
    assert format_quota_status(_provider(), config) == ""


def test_format_quota_status_marks_stale_state():
    """AC2b: stale state is visible instead of silently trusted."""
    provider = _provider()
    config = _config([provider])
    _poll_sync([provider], config, _usage_body(monthly=40))
    quota_state.set_clock(_Clock(NOW + timedelta(days=30)))

    text = format_quota_status(provider, config)
    assert "stale" in text
    (account,) = quota_status_payload(config)["accounts"]
    assert account["stale"] is True
    assert account["surge_active"] is False


# ---------------------------------------------------------------------------
# AC2c — Prometheus export, absence fallback and no key leakage
# ---------------------------------------------------------------------------


def test_quota_metric_objects_exist():
    """AC2c: the documented gauges/counter exist with the documented labels."""
    assert metrics._enabled
    for name in (
        "quota_monthly_percent",
        "quota_weekly_percent",
        "quota_rolling_percent",
        "quota_effective_headroom",
        "quota_seconds_to_reset",
        "quota_resets_at_timestamp_seconds",
        "quota_surge_active",
        "quota_exhausted",
        "quota_last_poll_timestamp_seconds",
    ):
        gauge = getattr(metrics, name)
        assert gauge is not None, f"{name} missing"
        # The label order list matters for the exposition; the real client
        # stores a tuple, the repo-root stub a list.
        assert tuple(gauge._labelnames) == ("account", "entry"), name
    assert metrics.quota_poll_failures_total is not None
    assert tuple(metrics.quota_poll_failures_total._labelnames) == ("account", "reason")


def test_export_payload_is_bytes_when_client_is_available():
    """AC2c: ``generate_metrics_payload()`` returns an exposition, never raises."""
    _poll_sync([_provider()], _config(), _usage_body(monthly=40))
    body, content_type = metrics.generate_metrics_payload()
    assert isinstance(body, bytes) and body
    assert "text/plain" in content_type


def _real_prometheus_client_dir() -> Path | None:
    """Locate the installed ``prometheus_client`` package, skipping the stub."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    for entry in sys.path:
        if not entry:
            continue
        if Path(entry).resolve() == repo_root:
            continue
        candidate = Path(entry) / "prometheus_client"
        if (candidate / "__init__.py").is_file():
            return candidate
    return None


@contextmanager
def _load_metrics_against_real_client():
    """Load ``proxy/metrics.py`` against the real prometheus_client.

    The repo root ships a lightweight ``prometheus_client.py`` stub that
    shadows the installed package on the test path, so this loads the module
    fresh (registered under a private name) with the real package first on
    ``sys.path``. Everything is restored afterwards, so each run starts from a
    fresh collector registry.
    """
    from unittest.mock import patch as _patch

    real_dir = _real_prometheus_client_dir()
    if real_dir is None:
        pytest.skip("the real prometheus_client package is not installed")
    metrics_path = Path(metrics.__file__)
    search = [str(real_dir.parent), *[p for p in sys.path if p]]
    with _patch.object(sys, "path", search):
        with _patch.dict(sys.modules):
            sys.modules.pop("prometheus_client", None)
            real_pc = importlib.import_module("prometheus_client")
            if not hasattr(real_pc, "Histogram"):  # pragma: no cover - env guard
                pytest.skip("could not import the real prometheus_client")
            spec = importlib.util.spec_from_file_location(
                "proxy._metrics_real_client", metrics_path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            yield module, real_pc


def _active_prometheus_is_stub() -> bool:
    """Whether the active ``prometheus_client`` is the repo-root test stub.

    ``pytest`` (console script) runs without the repo root on ``sys.path``, so
    the real package wins; ``python3 -m pytest`` (and the proxy process) put
    the repo root first, so the lightweight ``prometheus_client.py`` stub wins.
    """
    import prometheus_client as active

    stub = Path(__file__).resolve().parent.parent.parent / "prometheus_client.py"
    return Path(active.__file__).resolve() == stub.resolve()


_REAL_EXPORT_LABELS = {"account": "OPENCODE_REAL_CLIENT", "entry": "real-client"}


def _quota_observation_for_export() -> dict:
    return {
        **{k: v for k, v in _REAL_EXPORT_LABELS.items()},
        "rolling": {"percent": 95.0},
        "weekly": {"percent": 10.0},
        "monthly": {"percent": 40.0},
        "effective_headroom": 5.0,
        "seconds_to_reset": float(_MONTHLY_RESET),
        "resets_at_timestamp_seconds": 1792000000.0,
        "surge_active": True,
        "exhausted": False,
        "last_poll_timestamp_seconds": 1792416000.0,
    }


def test_export_includes_quota_series_with_the_real_client():
    """AC2c: with the real client installed the quota series are exported.

    Two environments: "python3 -m pytest" (repo-root stub active, so a fresh
    module is loaded against the real package) and the test-skill runner
    (real package already active, so the module is exercised in place).
    """
    if not _active_prometheus_is_stub():
        assert metrics._enabled, "metrics must enable with the real client"
        metrics.record_quota_state(_quota_observation_for_export())
        body, _ = metrics.generate_metrics_payload()
    else:
        with _load_metrics_against_real_client() as (module, _real_pc):
            assert module._enabled, "metrics must enable with the real client"
            module.record_quota_state(_quota_observation_for_export())
            body, _ = module.generate_metrics_payload()

    text = body.decode("utf-8")
    assert 'quota_effective_headroom{account="OPENCODE_REAL_CLIENT",entry="real-client"} 5.0' in text
    assert 'quota_monthly_percent{account="OPENCODE_REAL_CLIENT",entry="real-client"} 40.0' in text
    assert 'quota_surge_active{account="OPENCODE_REAL_CLIENT",entry="real-client"} 1.0' in text


def test_metrics_are_a_noop_without_prometheus_client():
    """AC2c: absent library → plaintext fallback, recorders never raise."""
    module = _load_metrics_without_prometheus_client()
    assert module._enabled is False
    body, content_type = module.generate_metrics_payload()
    assert body == b"Prometheus client library not installed; metrics disabled\n"
    assert "text/plain" in content_type

    # Every recorder is a silent no-op rather than an exception.
    module.record_quota_state({"account": "a", "entry": "e"})
    module.record_quota_poll_failure("a", "QuotaPollError")


def _load_metrics_without_prometheus_client():
    """Load ``proxy/metrics.py`` with ``prometheus_client`` unavailable."""
    from unittest.mock import patch as _patch

    metrics_path = Path(metrics.__file__)
    with _patch.dict(sys.modules, {"prometheus_client": None}):
        spec = importlib.util.spec_from_file_location(
            "proxy._metrics_no_client", metrics_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def test_no_api_key_value_is_exposed_on_any_surface(monkeypatch):
    """AC2c: only the ``api_key_env`` name is exposed, never the key value."""
    from proxy.ui import _build_home_model_rows

    monkeypatch.setenv("OPENCODE_API_KEY", _SECRET)
    provider = _provider()
    config = {
        "server": {"quota_surge_enabled": True},
        "providers": [provider],
        "models": {"m": {"providers": [provider]}},
    }
    _poll_sync([provider], config, _usage_body(monthly=40))

    payload_json = json.dumps(quota_status_payload(config))
    home_html = _build_home_model_rows(_mock_srv(config))
    metrics_body, _ = metrics.generate_metrics_payload()

    assert _SECRET not in payload_json
    assert _SECRET not in home_html
    assert _SECRET not in metrics_body.decode("utf-8", errors="replace")
    # The account key is the env *name*.
    assert "OPENCODE_API_KEY" in payload_json


# ---------------------------------------------------------------------------
# AC2 — poller wiring and INFO/DEBUG log levels (epic R8)
# ---------------------------------------------------------------------------


def test_poller_records_metrics_on_every_successful_poll(monkeypatch):
    """AC2: the poller itself feeds the metrics surface."""
    provider = _provider()
    config = _config([provider])
    recorded: list[dict] = []
    monkeypatch.setattr(
        metrics, "record_quota_state", lambda obs: recorded.append(obs)
    )

    _poll_sync([provider], config, _usage_body(monthly=40))

    assert len(recorded) == 1
    assert recorded[0]["account"] == "OPENCODE_API_KEY"
    assert recorded[0]["entry"] == "opencode-go"
    assert recorded[0]["effective_headroom"] == 60


def test_poll_logs_at_info(caplog):
    """AC2 (R8): a poll result is visible at INFO (not DEBUG)."""
    provider = _provider()

    with caplog.at_level("INFO", logger="llama-proxy.quota_state"):
        _poll_sync([provider], _config([provider]), _usage_body(monthly=40))

    poll_lines = [r for r in caplog.records if "quota poll ok" in r.getMessage()]
    assert len(poll_lines) == 1
    assert poll_lines[0].levelno == 20


def test_per_request_ordering_stays_at_debug(caplog):
    """AC2 (R8): the per-request ordering line is DEBUG, not INFO."""
    from proxy.provider import resolve_provider

    cheap = _provider(name="opencode-go", api_key_env="OPENCODE_API_KEY")
    spent = _provider(name="opencode-go-2", api_key_env="OPENCODE_API_KEY_2")
    config = _config([cheap, spent])
    _poll_sync([cheap], config, _usage_body(monthly=10, monthly_resets=6 * 86400))
    _poll_sync([spent], config, _usage_body(monthly=90, monthly_resets=6 * 86400))

    with caplog.at_level("DEBUG", logger="llama-proxy.provider"):
        chosen = resolve_provider({"providers": [spent, cheap]}, config=config)

    assert chosen["name"] == "opencode-go"  # more headroom first
    ordering = [r for r in caplog.records if "quota_ordering" in r.getMessage()]
    assert ordering, "ordering decisions must be logged for debugging"
    assert all(r.levelno == 10 for r in ordering)
    assert not [r for r in ordering if r.levelno >= 20]
