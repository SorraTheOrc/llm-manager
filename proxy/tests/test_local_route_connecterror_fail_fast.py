"""Local-route ConnectError retry storm — fail fast + pre-dispatch skip
(LP-0MW42T6IJ006EQA3).

The 2026-11-05 21:36 incident: a dead llama-server (killed ~15:13) was still
advertised available, so every request dispatched local and burned ~28 s on
``backend_retry_attempts=8`` x chain-hold cycles of deterministic
``ConnectError`` (connection refused — nothing inside the backoff horizon can
fix a refused connect) before falling to a remote provider, then poisoned the
session with a cooldown. The fail-fast shipped in LP-0MW0KZR7A003UFQK covers
*provably dead remote chains*, not a connection-refused local backend, and no
pre-dispatch check consulted the reachability evidence ``/health`` already
had.

The contracts pinned here:

- AC1 — a deterministic ``ConnectError`` (connection refused) on a local
  dispatch fails the request over to the next provider after a **single
  attempt** (no 8x backoff retries, no chain-hold re-dispatch), with a log
  line naming the fail-fast reason (``backend_fail_fast …
  reason=connection_refused``).
- AC2 — the local route is skipped **before dispatch** when the backend is
  known unreachable: ``backend_reachable=false`` (the ``/health`` probe
  writes through to the routing skip state) or a slots snapshot older than
  the configured staleness bound ⇒ the chain moves to the next provider
  without attempting a connection (``routing_skip_local …
  reason=backend_unreachable``).
- AC3 — the skip state is honoured across turns but is TTL-bounded and
  cleared by a successful dispatch / fresh probe, so a recovered backend is
  used again immediately (no permanent quarantine).
- AC4 — regression rows: (i) dead backend → single-attempt fallthrough to
  remote; (ii) recovered backend → local route restored; (iii) routing skips
  local pre-dispatch when ``/health`` shows unreachable.

Hermetic: stub request objects, patched ``_get_proxy_to_local`` /
``_get_proxy_to_remote`` dispatch surfaces, and module-state resets per test.
"""

from __future__ import annotations

import asyncio
import errno
import json
import time

import httpx
import proxy.observability as observability
import proxy.provider as provider
import proxy.server as server
import pytest
from fastapi import Response


class _StubRequest:
    """Minimal request stub for the provider fallback cycle."""

    def __init__(self, body: bytes = b'{"model":"qwen3"}'):
        self._body = body
        self.headers: dict[str, str] = {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return False


def _refused_connect_error(url: str = "http://localhost:8080/v1/chat/completions"):
    """A deterministic connection-refused ``httpx.ConnectError``.

    Mirrors what httpx raises on ``ECONNREFUSED``: a ``ConnectError`` whose
    cause chain carries the refused socket error.
    """
    request = httpx.Request("POST", url)
    exc = httpx.ConnectError("Connection refused", request=request)
    exc.__cause__ = ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")
    return exc


def _plain_connect_error(url: str = "http://localhost:8080/v1/chat/completions"):
    """A *retryable* connect error with no refused cause (e.g. DNS failure)."""
    request = httpx.Request("POST", url)
    return httpx.ConnectError("connect failed", request=request)


def _success_response(body: dict | None = None) -> Response:
    if body is None:
        body = {"choices": [{"message": {"content": "ok"}}]}
    return Response(
        content=json.dumps(body).encode("utf-8"),
        status_code=200,
        media_type="application/json",
    )


def _mixed_chain() -> dict:
    """A local+remote provider chain (the local-qwen3 -> munitoai shape)."""
    return {
        "providers": [
            {"name": "local-qwen3", "type": "local", "llama_model": "qwen3"},
            {
                "name": "munitoai",
                "type": "remote",
                "endpoint": "https://api.munito.ai/inference",
            },
        ]
    }


def _base_config(**server_overrides) -> dict:
    cfg = {
        "provider_cooldown_seconds": 0,
        "local_backend_skip_seconds": 60,
        "server": {
            "backend_retry_attempts": 8,
            "backend_retry_base_delay_seconds": 0,
            "backend_retry_max_delay_seconds": 0,
            "backend_retry_jitter_ratio": 0,
            "llama_server_port": 8080,
            **server_overrides,
        },
    }
    return cfg


@pytest.fixture(autouse=True)
def _reset_local_reachability_state(monkeypatch):
    """Reset the routing-skip state and provider cooldowns for every test."""
    import proxy.server as srv_mod

    monkeypatch.setattr(srv_mod, "local_backend_unreachable_until", 0.0)
    monkeypatch.setattr(srv_mod, "backend_ready", False)
    monkeypatch.setattr(
        srv_mod,
        "backend_signal_counts",
        {
            **observability.backend_signal_counts,
            "connect_failures": 0,
            "read_failures": 0,
            "timeout_failures": 0,
            "other_failures": 0,
            "concurrency_rejects": 0,
        },
    )
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    monkeypatch.setattr(
        srv_mod,
        "config",
        {"server": {"backend_retry_attempts": 3, "llama_server_port": 8080}},
    )
    yield


# ======================================================================
# Deterministic connection-refused detection
# ======================================================================


def test_is_connection_refused_detects_refused_cause():
    """A ConnectError carrying a refused socket error is deterministic."""
    assert observability._is_connection_refused(_refused_connect_error()) is True


def test_is_connection_refused_detects_refused_errno_oserror():
    """An OSError cause with errno ECONNREFUSED (not a ConnectionRefusedError
    subclass) is still refused."""
    exc = _plain_connect_error()
    exc.__cause__ = OSError(errno.ECONNREFUSED, "Connection refused")
    assert observability._is_connection_refused(exc) is True


def test_is_connection_refused_false_for_retryable_errors():
    """Non-refused connect errors and non-Connect errors stay retryable."""
    assert observability._is_connection_refused(_plain_connect_error()) is False
    timeout = httpx.ConnectTimeout(
        "timed out", request=httpx.Request("POST", "http://localhost:8080")
    )
    assert observability._is_connection_refused(timeout) is False
    read = httpx.ReadError(
        "read failed", request=httpx.Request("POST", "http://localhost:8080")
    )
    assert observability._is_connection_refused(read) is False


# ======================================================================
# AC1 — fail fast inside the backend retry loop (no 8x backoff, log line)
# ======================================================================


@pytest.mark.asyncio
async def test_backend_retry_loop_fails_fast_on_connection_refused(monkeypatch, caplog):
    """AC1 row: a refused local dispatch is a single attempt — no retries, no
    backoff sleep — and the fail-fast reason is logged."""
    import logging

    server.config = {
        "server": {
            "backend_retry_attempts": 8,
            "backend_retry_base_delay_seconds": 0.25,
            "backend_retry_max_delay_seconds": 2.0,
            "backend_retry_jitter_ratio": 0.25,
        }
    }
    attempts = {"n": 0}

    async def _panic_sleep(_delay):
        raise AssertionError("fail-fast must not sleep between attempts")

    monkeypatch.setattr(asyncio, "sleep", _panic_sleep)

    async def call_factory():
        attempts["n"] += 1
        raise _refused_connect_error()

    with caplog.at_level(logging.WARNING):
        with pytest.raises(httpx.ConnectError):
            await server._call_with_backend_retries(
                call_factory, path="v1/chat/completions", stream=True
            )

    assert attempts["n"] == 1
    assert server.backend_signal_counts["connect_failures"] == 1
    assert server.backend_ready is False
    # The routing skip is armed (AC2/AC3 state) for subsequent turns.
    assert server.local_backend_unreachable_until > time.monotonic()
    assert any(
        "backend_fail_fast" in r.message and "reason=connection_refused" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_backend_retry_loop_still_retries_non_refused_connect_errors(monkeypatch):
    """Regression: only *refused* connects fail fast; other connect errors
    keep the configured retry count."""
    server.config = {
        "server": {
            "backend_retry_attempts": 3,
            "backend_retry_base_delay_seconds": 0,
            "backend_retry_max_delay_seconds": 0,
            "backend_retry_jitter_ratio": 0,
        }
    }
    attempts = {"n": 0}

    async def fake_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def call_factory():
        attempts["n"] += 1
        raise _plain_connect_error()

    with pytest.raises(httpx.ConnectError):
        await server._call_with_backend_retries(
            call_factory, path="v1/chat/completions", stream=False
        )

    assert attempts["n"] == 3
    assert server.backend_signal_counts["connect_failures"] == 3
    # No skip is armed by a non-refused (potentially transient) error.
    assert server.local_backend_unreachable_until == 0.0


@pytest.mark.asyncio
async def test_successful_dispatch_clears_routing_skip(monkeypatch):
    """AC3: a backend that answers a dispatch proves reachability and clears
    the skip state immediately."""
    server.local_backend_unreachable_until = time.monotonic() + 1000

    async def call_factory():
        return "ok"

    result = await server._call_with_backend_retries(
        call_factory, path="v1/chat/completions", stream=True
    )

    assert result == "ok"
    assert server.local_backend_unreachable_until == 0.0


def test_refused_connect_arms_skip_and_publishes_unreachable_evidence():
    """AC4: a refused local connect still fail-fasts, arms the skip
    immediately, and leaves the unreachable evidence the staleness
    corroboration consumes (LP-0MWCOMI160011F6C)."""
    exc = _refused_connect_error()

    with pytest.raises(httpx.ConnectError):
        observability._record_connection_refused_fail_fast(
            "v1/chat/completions", False, 1, 8, exc
        )

    assert server.local_backend_unreachable_until > time.monotonic()
    assert observability.last_local_backend_reachability(_base_config()) is False


# ======================================================================
# AC2 — the routing gate (remaining seconds / staleness read)
# ======================================================================


def test_routing_gate_reads_armed_skip_and_expiry():
    server.local_backend_unreachable_until = time.monotonic() + 30
    remaining = observability._local_backend_unreachable_remaining_seconds(_base_config())
    assert 0 < remaining <= 30

    server.local_backend_unreachable_until = time.monotonic() - 1
    assert observability._local_backend_unreachable_remaining_seconds(_base_config()) == 0.0

    server.local_backend_unreachable_until = 0.0
    assert observability._local_backend_unreachable_remaining_seconds(_base_config()) == 0.0


def test_routing_gate_skips_on_stale_snapshot_plus_unreachable_probe():
    """AC2: a stale snapshot *corroborated* by a recent failed reachability
    probe skips local, and the skip stays bounded by the TTL."""
    observability._last_slot_counts_cache = (1, 2, time.monotonic() - 10_000)
    observability.record_local_backend_reachability(False)
    cfg = _base_config(local_slots_max_snapshot_age_seconds=60)
    remaining = observability._local_backend_unreachable_remaining_seconds(cfg)
    assert remaining > 0
    assert remaining <= 60


def test_routing_gate_fails_open_on_stale_snapshot_alone(monkeypatch):
    """AC1: a stale slot snapshot on its own is never ``backend_unreachable``.

    Reproduces the 2026-11-11 starvation: the counts cache is frozen far past
    ``local_slots_max_snapshot_age_seconds`` while the backend is healthy and
    the ``/health`` probe passes. The gate must return 0 (dispatch local)
    instead of skipping, and must not consult ``/slots`` to decide.
    """
    observability._last_slot_counts_cache = (1, 2, time.monotonic() - 5566)
    observability.record_local_backend_reachability(True)
    cfg = _base_config(local_slots_max_snapshot_age_seconds=300)
    assert observability._local_backend_unreachable_remaining_seconds(cfg) == 0.0


def test_routing_gate_fails_open_when_probe_evidence_is_stale():
    """AC2: the staleness skip is bounded by the probe interval — evidence
    older than ``local_reachability_probe_max_age_seconds`` is no evidence at
    all, so the gate fails open and a recovered backend is re-admitted
    without waiting for a lucky ``/slots`` success."""
    observability._last_slot_counts_cache = (1, 2, time.monotonic() - 10_000)
    observability._local_reachability_probe = (False, time.monotonic() - 120)
    cfg = _base_config(
        local_slots_max_snapshot_age_seconds=60,
        local_reachability_probe_max_age_seconds=60,
    )
    assert observability._local_backend_unreachable_remaining_seconds(cfg) == 0.0


def test_routing_gate_does_not_skip_on_fresh_slots_snapshot():
    observability._last_slot_counts_cache = (1, 2, time.monotonic())
    cfg = _base_config(local_slots_max_snapshot_age_seconds=60)
    assert observability._local_backend_unreachable_remaining_seconds(cfg) == 0.0


# ======================================================================
# AC4(i) — dead backend -> single-attempt fallthrough to remote
# ======================================================================


@pytest.mark.asyncio
async def test_dead_backend_single_attempt_fallthrough_to_remote(monkeypatch, caplog):
    """AC4(i): a connection-refused local dispatch is one local call, then the
    chain serves from the next provider."""
    import logging

    local_calls = {"n": 0}
    remote_calls = {"n": 0}

    async def _dead_local(_req, _path, _endpoint=None):
        local_calls["n"] += 1
        raise _refused_connect_error()

    async def _healthy_remote(_req, _path, _provider_cfg):
        remote_calls["n"] += 1
        return _success_response()

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: _dead_local)
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: _healthy_remote)

    cfg = _base_config(chain_hold_seconds=0.01, chain_hold_max_cycles=3)
    with caplog.at_level(logging.INFO, logger="llama-proxy.provider"):
        result = await provider.proxy_with_fallback(
            _StubRequest(), "v1/chat/completions", _mixed_chain(), cfg
        )

    assert result.status_code == 200
    assert local_calls["n"] == 1
    assert remote_calls["n"] == 1
    # The fail-fast path arms the skip so later turns never re-enter local.
    assert server.local_backend_unreachable_until > time.monotonic()
    assert any(
        "backend_fail_fast" in r.message and "reason=connection_refused" in r.message
        for r in caplog.records
    )


# ======================================================================
# AC1/AC2 — no chain-hold re-dispatch while the skip is armed
# ======================================================================


@pytest.mark.asyncio
async def test_chain_hold_cycle_skips_local_not_redispatch(monkeypatch, caplog):
    """AC1/AC3: after the fail-fast arms the skip, a chain-hold cycle re-runs
    from the FIRST provider but local is skipped pre-dispatch — the retry
    storm (8 attempts x N cycles) cannot recur."""
    import logging

    local_calls = {"n": 0}
    remote_calls = {"n": 0}

    async def _dead_local(_req, _path, _endpoint=None):
        local_calls["n"] += 1
        raise _refused_connect_error()

    async def _remote_fail_then_succeed(_req, _path, _provider_cfg):
        remote_calls["n"] += 1
        if remote_calls["n"] == 1:
            return Response(status_code=502, content=b"bad gateway")
        return _success_response()

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: _dead_local)
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: _remote_fail_then_succeed)

    cfg = _base_config(chain_hold_seconds=0.01, chain_hold_max_cycles=3)
    with caplog.at_level(logging.INFO, logger="llama-proxy.provider"):
        result = await provider.proxy_with_fallback(
            _StubRequest(), "v1/chat/completions", _mixed_chain(), cfg
        )

    assert result.status_code == 200
    # Cycle 0 dispatches local once; cycle 1 skips local entirely.
    assert local_calls["n"] == 1
    assert remote_calls["n"] == 2
    assert any(
        "routing_skip_local" in r.message and "reason=backend_unreachable" in r.message
        for r in caplog.records
    )


# ======================================================================
# AC4(iii) — routing skips local pre-dispatch when the backend is unreachable
# ======================================================================


@pytest.mark.asyncio
async def test_routing_skips_local_predispatch_when_backend_unreachable(monkeypatch, caplog):
    """AC4(iii)-armed: an armed skip (e.g. from a failed /health probe) keeps
    local out of the chain without any connection attempt."""
    import logging

    server.local_backend_unreachable_until = time.monotonic() + 300
    remote_calls = {"n": 0}

    async def _local_must_not_run(_req, _path, _endpoint=None):
        raise AssertionError("local must not be dispatched while the skip is armed")

    async def _healthy_remote(_req, _path, _provider_cfg):
        remote_calls["n"] += 1
        return _success_response()

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: _local_must_not_run)
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: _healthy_remote)

    cfg = _base_config()
    with caplog.at_level(logging.INFO, logger="llama-proxy.provider"):
        result = await provider.proxy_with_fallback(
            _StubRequest(), "v1/chat/completions", _mixed_chain(), cfg
        )

    assert result.status_code == 200
    assert remote_calls["n"] == 1
    assert any(
        "routing_skip_local" in r.message and "reason=backend_unreachable" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_routing_dispatches_local_on_stale_snapshot_when_probe_passes(monkeypatch, caplog):
    """AC1: a stale snapshot plus a passing reachability probe dispatches
    local — the healthy-but-idle backend is no longer starved."""
    import logging

    observability._last_slot_counts_cache = (2, 2, time.monotonic() - 4_477)
    observability.record_local_backend_reachability(True)
    local_calls = {"n": 0}

    async def _healthy_local(_req, _path, _endpoint=None):
        local_calls["n"] += 1
        return _success_response()

    async def _remote_must_not_run(_req, _path, _provider_cfg):
        raise AssertionError("local must be dispatched on a stale-but-reachable backend")

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: _healthy_local)
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: _remote_must_not_run)

    cfg = _base_config(local_slots_max_snapshot_age_seconds=300)
    with caplog.at_level(logging.INFO, logger="llama-proxy.provider"):
        result = await provider.proxy_with_fallback(
            _StubRequest(), "v1/chat/completions", _mixed_chain(), cfg
        )

    assert result.status_code == 200
    assert local_calls["n"] == 1
    assert not any(
        "routing_skip_local" in r.message and "reason=backend_unreachable" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_routing_skips_local_on_stale_snapshot_with_unreachable_probe(monkeypatch, caplog):
    """AC2: a stale snapshot corroborated by a recent failed probe still skips
    local pre-dispatch (the dead-backend case the branch exists for)."""
    import logging

    observability._last_slot_counts_cache = (1, 2, time.monotonic() - 10_000)
    observability.record_local_backend_reachability(False)
    remote_calls = {"n": 0}

    async def _local_must_not_run(_req, _path, _endpoint=None):
        raise AssertionError("local must not be dispatched without a reachable probe")

    async def _healthy_remote(_req, _path, _provider_cfg):
        remote_calls["n"] += 1
        return _success_response()

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: _local_must_not_run)
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: _healthy_remote)

    cfg = _base_config(local_slots_max_snapshot_age_seconds=60)
    with caplog.at_level(logging.INFO, logger="llama-proxy.provider"):
        result = await provider.proxy_with_fallback(
            _StubRequest(), "v1/chat/completions", _mixed_chain(), cfg
        )

    assert result.status_code == 200
    assert remote_calls["n"] == 1
    assert any(
        "routing_skip_local" in r.message and "reason=backend_unreachable" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_reachability_probe_publishes_evidence(monkeypatch):
    """AC2: every reachability probe publishes its result, so the gate's
    staleness corroboration is fed by ``/health`` polling rather than by a
    ``/slots`` success."""
    import proxy.lifecycle as lifecycle
    import proxy.server as srv_mod

    class _Resp:
        def __init__(self, status_code: int):
            self.status_code = status_code

    class _Client:
        def __init__(self, status_code: int):
            self.status_code = status_code

        async def get(self, _url, timeout=None):
            return _Resp(self.status_code)

    obs = observability
    obs.reset_local_backend_reachability()
    monkeypatch.setattr(srv_mod, "_http_client", _Client(200))
    assert await lifecycle._probe_backend_reachable(8080) is True
    assert obs.last_local_backend_reachability(_base_config()) is True

    monkeypatch.setattr(srv_mod, "_http_client", _Client(503))
    assert await lifecycle._probe_backend_reachable(8080) is False
    assert obs.last_local_backend_reachability(_base_config()) is False


# ======================================================================
# AC4(iii) — /health probe writes through to the routing skip state
# ======================================================================


@pytest.mark.asyncio
async def test_health_probe_unreachable_arms_skip_and_reachable_clears(monkeypatch):
    """AC4(iii): /health showing ``backend_reachable=false`` arms the routing
    skip; a healthy probe clears it (AC3 recovery via health polling)."""
    import proxy.server as srv_mod

    class Proc:
        def poll(self):
            return None

    monkeypatch.setattr(srv_mod, "llama_process", Proc())
    monkeypatch.setattr(srv_mod, "backend_ready", True)
    monkeypatch.setattr(srv_mod, "current_model", "qwen3")
    monkeypatch.setattr(
        srv_mod,
        "config",
        {"server": {"llama_router_mode": False, "llama_server_port": 8080}},
    )

    async def _unreachable(_port):
        return False

    monkeypatch.setattr(srv_mod, "_probe_backend_reachable", _unreachable)
    health = await srv_mod.health_check()
    assert health["backend_reachable"] is False
    assert srv_mod.local_backend_unreachable_until > time.monotonic()

    async def _reachable(_port):
        return True

    monkeypatch.setattr(srv_mod, "_probe_backend_reachable", _reachable)
    health = await srv_mod.health_check()
    assert health["backend_reachable"] is True
    assert srv_mod.local_backend_unreachable_until == 0.0


# ======================================================================
# AC4(ii) — recovered backend -> local route restored
# ======================================================================


@pytest.mark.asyncio
async def test_recovered_backend_restores_local_route_after_skip_expiry(monkeypatch):
    """AC4(ii): once the TTL-bounded skip expires (no permanent quarantine),
    a healthy local backend is dispatched again."""
    server.local_backend_unreachable_until = time.monotonic() - 1
    local_calls = {"n": 0}
    remote_calls = {"n": 0}

    async def _healthy_local(_req, _path, _endpoint=None):
        local_calls["n"] += 1
        return _success_response()

    async def _remote_must_not_run(_req, _path, _provider_cfg):
        remote_calls["n"] += 1
        raise AssertionError("a recovered local backend must serve the request")

    monkeypatch.setattr(provider, "_get_proxy_to_local", lambda: _healthy_local)
    monkeypatch.setattr(provider, "_get_proxy_to_remote", lambda: _remote_must_not_run)

    cfg = _base_config()
    result = await provider.proxy_with_fallback(
        _StubRequest(), "v1/chat/completions", _mixed_chain(), cfg
    )

    assert result.status_code == 200
    assert local_calls["n"] == 1
    assert remote_calls["n"] == 0
