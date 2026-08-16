"""
Hermetic tests for the prefill-progress dispatch-lease extension
(LP-0MSE05J53004C6EL).

Context
-------
The dispatch lease refresh fires only on stream data chunks
(LP-0MRDKV44T003FRBP). The prefill phase of a large-context request emits
no chunks, so a prefill longer than the lease (base 60s, or the adaptive
token-estimate value capped at 1500s) can outlive its lease; the ~10s
cleanup loop then orphan-cleans the active record mid-prefill and the
slot is handed to another session.

This item extends the lease during the prefill phase for *explicit*
sessions based on **observed prefill progress** (per-slot ``n_past`` /
``n_prompt_tokens_processed`` from llama-server ``/slots``, falling back to
aggregate ``kv_cache_tokens`` from ``query_llama_status()``): while the
reported progress is advancing, ``expires_at`` is pushed out to
``now + safety_buffer``. Extension stops when the first actual data chunk
arrives (the existing chunk-refresh path takes over) or the stream ends.
When progress is unobservable, the lease keeps the adaptive token-estimate
value applied at acquisition (fallback) rather than being dropped.

These tests verify hermetically (fake server state, no live llama-server):

1. An explicit-session request whose prefill emits no data for longer than
   the base lease keeps its dispatch lease for the full prefill duration
   (parent AC1).
2. Extension is driven by observed progress + safety buffer, stops once
   prefill completes (first data chunk), and small prompts never trigger it
   — the base lease is unchanged (parent AC3).
3. When progress is unobservable the lease falls back to the adaptive
   token-estimate value rather than being dropped (parent AC2 fallback).
4. Repeated cleanup-loop runs never orphan-clean an active prefill lease
   (parent AC4).
"""

import asyncio
import copy
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import proxy.server as server
import pytest

BASE_SERVER_CONFIG = {
    "server": {
        "llama_router_mode": False,
        "llama_server_port": 8080,
        "max_concurrent_queries": 4,
        "local_max_concurrent_queries": 1,
        "llama_request_timeout": 30,
        # Short base lease so a "long prefill" fits in test time. In
        # production the base lease is 60s (LP-0MRHV4UYE0013F6P).
        "local_dispatch_lease_timeout_seconds": 0.3,
        "local_dispatch_lease_per_token_seconds": 0.0,
        "local_dispatch_lease_max_seconds": 1500,
        "local_dispatch_lease_prefill_poll_seconds": 0.05,
        "local_dispatch_lease_prefill_buffer_seconds": 0.5,
        "session_single_flight_mode": "bypass",
        "disconnect_cleanup_timeout": 1,
        "stream_heartbeat_interval_seconds": 0.05,
        "stream_idle_timeout_seconds": 0.3,
        "session_guardrail_max_runtime_seconds": 3600,
        "session_guardrail_max_completion_tokens": 4096,
        "session_guardrail_repetition_min_pattern_chars": 100,
        "session_guardrail_repetition_min_repeats": 3,
        "session_guardrail_invalidate_on_cutoff": False,
        "session_guardrail_invalidate_on_repetition": False,
        "session_guardrail_max_token_rate": 0,
        "session_guardrail_token_rate_window_seconds": 60,
    }
}

DEFAULT_LEASE_CONFIG = {
    "server": {"local_dispatch_lease_timeout_seconds": 60},
}


def _make_srv(records: dict | None = None, config: dict | None = None) -> SimpleNamespace:
    """Build a fake server object with dispatch-tracking state."""
    return SimpleNamespace(
        config=config if config is not None else copy.deepcopy(BASE_SERVER_CONFIG),
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records=records if records is not None else {},
        local_dispatch_records_lock=asyncio.Lock(),
        logger=MagicMock(),
    )


def _dummy_request(body: dict, stream: bool = False):
    """Build a minimal dummy Request that proxy_to_local can consume."""
    payload = {**body}
    if stream:
        payload["stream"] = True
    body_bytes = json.dumps(payload).encode("utf-8")

    class DummyRequest:
        headers = {"host": "localhost"}
        method = "POST"
        url = type("U", (), {"path": "/v1/chat/completions"})()

        async def body(self):
            return body_bytes

        async def is_disconnected(self):
            return False

    return DummyRequest()


def _make_mock_cm(aiter_func):
    """Create (cm, response) matching _call_with_backend_retries' contract."""

    async def _aiter():
        async for chunk in aiter_func():
            yield chunk

    mock_resp = type("MockStreamResponse", (), {
        "status_code": 200,
        "headers": {"content-type": "text/event-stream"},
        "aiter_bytes": staticmethod(aiter_func),
        "aread": AsyncMock(return_value=b""),
    })

    class _MockCM:
        async def __aenter__(self):
            return mock_resp()

        async def __aexit__(self, *args):
            pass

    return _MockCM(), mock_resp()


def _make_advancing_status(session_id: str, snapshots: list, progress: list):
    """Return a query_llama_status fake reporting advancing kv_cache_tokens.

    Each call bumps the reported progress and snapshots the session's
    dispatch record state (expires_at) *before* the lease extension for
    that poll is applied.
    """

    async def _status():
        progress[0] += 1000
        record = server.local_dispatch_records.get(session_id)
        if record is not None:
            snapshots.append({
                "t": time.monotonic(),
                "started_at": record.get("started_at"),
                "expires_at": record["expires_at"],
                "active": record.get("active"),
            })
        return {
            "llama_server_running": True,
            "n_ctx": 32768,
            "kv_cache_tokens": progress[0],
            "router_mode": False,
        }

    return _status


@pytest.fixture(autouse=True)
def _reset_server_state(monkeypatch):
    """Reset server-level state before each test."""
    monkeypatch.setattr(server, "config", copy.deepcopy(BASE_SERVER_CONFIG))
    monkeypatch.setattr(server, "active_queries", 0)
    monkeypatch.setattr(server, "local_active_queries", 0)
    monkeypatch.setattr(server, "local_dispatch_records", {})
    monkeypatch.setattr(server, "local_dispatch_records_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "llama_process", MagicMock(poll=lambda: None, pid=1))
    monkeypatch.setattr(server, "current_model", "test-model")
    monkeypatch.setattr(server, "session_manager", MagicMock())
    monkeypatch.setattr(server, "logger", MagicMock())

    # Disable self-healing
    monkeypatch.setattr("proxy.router._is_self_healing_active", lambda: False)

    # Mock slot save/restore
    monkeypatch.setattr("proxy.router._restore_slot_snapshot", AsyncMock(return_value=False))
    monkeypatch.setattr("proxy.router._save_slot_snapshot", AsyncMock(return_value=False))
    monkeypatch.setattr("proxy.router._build_slot_context", MagicMock(return_value=(None, None, 3.0)))

    # Mock session handlers
    monkeypatch.setattr("proxy.router._handle_session", AsyncMock(return_value={
        "session_id": "test-session-id",
        "session_id_header": "test-session-id",
        "session_explicit": True,
        "session_created": True,
        "is_delta_request": False,
        "session_fallback_reason": None,
        "delta_messages": [],
        "original_message_count": 1,
        "body_override": None,
        "body_json": None,
    }))

    # Mock log resolvers
    monkeypatch.setattr("proxy.session._resolve_log_path", MagicMock(return_value=MagicMock(
        exists=lambda: False,
        stat=lambda: MagicMock(st_size=0),
    )))

    # Mock slot availability
    monkeypatch.setattr("proxy.router._check_slot_availability", AsyncMock(return_value=None))

    # Hermetic progress sources: default to no per-slot data and a fake
    # aggregate status so no real HTTP is ever attempted.
    try:
        monkeypatch.setattr("proxy.observability._query_slots_progress", AsyncMock(return_value={}))
    except AttributeError:
        pass  # helper not yet implemented (red phase)
    monkeypatch.setattr("proxy.observability.query_llama_status", AsyncMock(return_value={
        "llama_server_running": True,
        "n_ctx": 32768,
        "kv_cache_tokens": None,
        "router_mode": False,
    }))


async def _collect_streamed_chunks(resp):
    collected = b""
    async for chunk in resp.body_iterator:
        collected += chunk
    return collected


def _info_log_lines(logger) -> list[str]:
    return [str(call) for call in logger.info.call_args_list]


# ═══════════════════════════════════════════════════════════════════════════════
# Config helpers
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_prefill_lease_config_defaults():
    """Defaults: poll cadence 10s, safety buffer 30s (production config)."""
    from proxy.router_helpers import _get_prefill_lease_config

    srv = _make_srv(config=DEFAULT_LEASE_CONFIG)
    poll_seconds, buffer_seconds = _get_prefill_lease_config(srv)
    assert poll_seconds == 10.0, f"Default poll cadence should be 10s, got {poll_seconds}"
    assert buffer_seconds == 30.0, f"Default safety buffer should be 30s, got {buffer_seconds}"


@pytest.mark.asyncio
async def test_prefill_lease_config_explicit_values():
    """Explicit config values are honoured."""
    from proxy.router_helpers import _get_prefill_lease_config

    srv = _make_srv(config={"server": {
        "local_dispatch_lease_prefill_poll_seconds": 15,
        "local_dispatch_lease_prefill_buffer_seconds": 45,
    }})
    poll_seconds, buffer_seconds = _get_prefill_lease_config(srv)
    assert poll_seconds == 15.0
    assert buffer_seconds == 45.0


# ═══════════════════════════════════════════════════════════════════════════════
# _query_prefill_progress: progress-source selection
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_query_prefill_progress_uses_per_slot_slots_data(monkeypatch):
    """Per-slot /slots state is preferred when slot_id is known (progress
    returned alongside the liveness flag)."""
    from proxy.router_helpers import _query_prefill_progress

    monkeypatch.setattr(
        "proxy.observability._query_slots_progress",
        AsyncMock(return_value={
            3: {"progress": 50000, "processing": True},
            5: {"progress": 100, "processing": False},
        }),
    )
    srv = _make_srv()
    assert await _query_prefill_progress(srv, 8080, model_name="qwen", slot_id=3) == (50000, True)
    assert await _query_prefill_progress(srv, 8080, model_name="qwen", slot_id=5) == (100, False)


@pytest.mark.asyncio
async def test_query_prefill_progress_liveness_when_no_numeric_progress(monkeypatch):
    """llama.cpp b8782 failure mode: /slots exposes no n_past/
    n_prompt_tokens_processed, but the slot's is_processing flag reports
    the request is alive — (None, alive=True) must be returned so the
    lease can be extended on liveness (LP-0MSUO5Z0K007HBSS)."""
    from proxy.router_helpers import _query_prefill_progress

    monkeypatch.setattr(
        "proxy.observability._query_slots_progress",
        AsyncMock(return_value={
            3: {"progress": None, "processing": True},
            5: {"progress": None, "processing": False},
        }),
    )
    monkeypatch.setattr("proxy.observability.query_llama_status", AsyncMock(return_value={
        "llama_server_running": True,
        "n_ctx": 32768,
        "kv_cache_tokens": None,
        "router_mode": False,
    }))
    srv = _make_srv()
    assert await _query_prefill_progress(srv, 8080, model_name="qwen", slot_id=3) == (None, True)
    assert await _query_prefill_progress(srv, 8080, model_name="qwen", slot_id=5) == (None, False)


@pytest.mark.asyncio
async def test_query_prefill_progress_falls_back_to_aggregate_status(monkeypatch):
    """Falls back to aggregate query_llama_status kv_cache_tokens when the
    per-slot query yields nothing for the slot."""
    from proxy.router_helpers import _query_prefill_progress

    monkeypatch.setattr("proxy.observability._query_slots_progress", AsyncMock(return_value={}))
    monkeypatch.setattr("proxy.observability.query_llama_status", AsyncMock(return_value={
        "llama_server_running": True,
        "n_ctx": 32768,
        "kv_cache_tokens": 4321,
        "router_mode": False,
    }))
    srv = _make_srv()
    assert await _query_prefill_progress(srv, 8080, slot_id=3) == (4321, True)


@pytest.mark.asyncio
async def test_query_prefill_progress_none_when_unobservable(monkeypatch):
    """(None, False) when neither source reports progress or liveness."""
    from proxy.router_helpers import _query_prefill_progress

    monkeypatch.setattr("proxy.observability._query_slots_progress", AsyncMock(return_value={}))
    monkeypatch.setattr("proxy.observability.query_llama_status", AsyncMock(return_value={
        "llama_server_running": True,
        "n_ctx": 32768,
        "kv_cache_tokens": None,
        "router_mode": False,
    }))
    srv = _make_srv()
    assert await _query_prefill_progress(srv, 8080, slot_id=3) == (None, False)


@pytest.mark.asyncio
async def test_query_prefill_progress_warns_when_unobservable(monkeypatch):
    """AC2: silent query failures are surfaced with a throttled warning —
    the prefill extension must log when progress cannot be observed, and
    the warning is rate-limited to one per interval (LP-0MSUO5Z0K007HBSS)."""
    import proxy.router_helpers as rh

    monkeypatch.setattr("proxy.observability._query_slots_progress", AsyncMock(return_value={}))
    monkeypatch.setattr("proxy.observability.query_llama_status", AsyncMock(return_value={
        "llama_server_running": True,
        "n_ctx": 32768,
        "kv_cache_tokens": None,
        "router_mode": False,
    }))
    srv = _make_srv()
    # Reset the throttle so this test observes the first warning.
    monkeypatch.setattr(rh, "_last_prefill_progress_warn_ts", 0.0)

    assert await rh._query_prefill_progress(srv, 8080, slot_id=3) == (None, False)
    warn_lines = [str(call) for call in srv.logger.warning.call_args_list]
    assert any("prefill_progress_unobservable" in line for line in warn_lines), (
        "Expected a warning when prefill progress is unobservable"
    )

    # Throttled: a second unobservable poll within the interval does not log again.
    srv.logger.warning.reset_mock()
    assert await rh._query_prefill_progress(srv, 8080, slot_id=3) == (None, False)
    assert not any(
        "prefill_progress_unobservable" in str(call)
        for call in srv.logger.warning.call_args_list
    ), "Throttled warning must not repeat within the interval"


@pytest.mark.asyncio
async def test_query_prefill_progress_uses_status_when_slot_unknown(monkeypatch):
    """Aggregate status is used directly when no slot_id is known."""
    from proxy.router_helpers import _query_prefill_progress

    monkeypatch.setattr("proxy.observability.query_llama_status", AsyncMock(return_value={
        "llama_server_running": True,
        "n_ctx": 32768,
        "kv_cache_tokens": 777,
        "router_mode": False,
    }))
    srv = _make_srv()
    assert await _query_prefill_progress(srv, 8080) == (777, True)


# ═══════════════════════════════════════════════════════════════════════════════
# _extend_lease_during_prefill: lease semantics
# ═══════════════════════════════════════════════════════════════════════════════


def _install_fake_progress(monkeypatch, values):
    """Install a _query_prefill_progress fake returning (progress, alive)
    tuples in sequence."""
    seq = iter(values)

    async def _fake_progress(*args, **kwargs):
        return next(seq)

    monkeypatch.setattr("proxy.router_helpers._query_prefill_progress", _fake_progress)
    return _fake_progress


@pytest.mark.asyncio
async def test_extend_lease_while_progress_advances(monkeypatch):
    """AC1: expires_at is pushed out by the safety buffer while progress
    advances, and a lease_extended_during_prefill event is logged."""
    from proxy.router_helpers import _extend_lease_during_prefill

    now = time.monotonic()
    srv = _make_srv(records={
        "sess-1": {"backend": "local", "started_at": now, "active": True, "expires_at": now + 0.3},
    })
    _install_fake_progress(monkeypatch, [(1000, True), (2000, True), (3000, True)])

    last_progress = 0
    for expected in (1000, 2000, 3000):
        last_progress, extended = await _extend_lease_during_prefill(
            srv, "sess-1", llama_port=8080, slot_id=None, last_progress=last_progress
        )
        assert extended is True, f"expected extension for progress {expected}"
        assert last_progress == expected

    record = srv.local_dispatch_records["sess-1"]
    remaining = record["expires_at"] - time.monotonic()
    assert remaining >= 0.45, (
        f"Lease should have ~0.5s buffer remaining after extension, got {remaining:.2f}s"
    )
    assert any("lease_extended_during_prefill" in line for line in _info_log_lines(srv.logger)), (
        "Expected lease_extended_during_prefill log event"
    )


@pytest.mark.asyncio
async def test_extend_lease_does_not_extend_when_progress_stalls(monkeypatch):
    """AC3: extension stops when progress stops advancing."""
    from proxy.router_helpers import _extend_lease_during_prefill

    now = time.monotonic()
    original_expiry = now + 10.0
    srv = _make_srv(records={
        "sess-1": {"backend": "local", "started_at": now, "active": True, "expires_at": original_expiry},
    })
    _install_fake_progress(monkeypatch, [(5000, False)])

    last_progress, extended = await _extend_lease_during_prefill(
        srv, "sess-1", llama_port=8080, slot_id=None, last_progress=5000
    )
    assert extended is False
    assert last_progress == 5000
    assert srv.local_dispatch_records["sess-1"]["expires_at"] == original_expiry
    assert not any("lease_extended_during_prefill" in line for line in _info_log_lines(srv.logger))


@pytest.mark.asyncio
async def test_extend_lease_on_liveness_when_numeric_progress_absent(monkeypatch):
    """llama.cpp b8782 failure mode: no numeric progress is reported, but the
    slot is observed processing — the lease is still extended on liveness so
    a long prefill never loses its lease mid-flight (LP-0MSUO5Z0K007HBSS AC2)."""
    from proxy.router_helpers import _extend_lease_during_prefill

    now = time.monotonic()
    srv = _make_srv(records={
        "sess-1": {"backend": "local", "started_at": now, "active": True, "expires_at": now + 0.3},
    })
    _install_fake_progress(monkeypatch, [(None, True)])

    last_progress, extended = await _extend_lease_during_prefill(
        srv, "sess-1", llama_port=8080, slot_id=None, last_progress=0
    )
    assert extended is True, "Lease must be extended on liveness when progress is unobservable"
    assert last_progress == 0
    record = srv.local_dispatch_records["sess-1"]
    assert record["expires_at"] - time.monotonic() >= 0.45
    assert any("lease_extended_during_prefill" in line for line in _info_log_lines(srv.logger))


@pytest.mark.asyncio
async def test_extend_lease_noop_when_unobservable_and_not_alive(monkeypatch):
    """No extension when progress is unobservable AND the slot is not alive
    (build exposes neither numeric progress nor is_processing)."""
    from proxy.router_helpers import _extend_lease_during_prefill

    now = time.monotonic()
    original_expiry = now + 0.3
    srv = _make_srv(records={
        "sess-1": {"backend": "local", "started_at": now, "active": True, "expires_at": original_expiry},
    })
    _install_fake_progress(monkeypatch, [(None, False)])

    last_progress, extended = await _extend_lease_during_prefill(
        srv, "sess-1", llama_port=8080, slot_id=None, last_progress=0
    )
    assert extended is False
    assert last_progress == 0
    assert srv.local_dispatch_records["sess-1"]["expires_at"] == original_expiry
    assert not any("lease_extended_during_prefill" in line for line in _info_log_lines(srv.logger))


@pytest.mark.asyncio
async def test_extend_lease_noop_when_progress_unobservable(monkeypatch):
    """Fallback: when progress is unobservable the lease is not dropped —
    it keeps the adaptive token-estimate expiry from acquisition."""
    from proxy.router_helpers import _extend_lease_during_prefill

    now = time.monotonic()
    original_expiry = now + 810.0  # adaptive estimate from acquisition
    srv = _make_srv(records={
        "sess-1": {"backend": "local", "started_at": now, "active": True, "expires_at": original_expiry},
    })
    _install_fake_progress(monkeypatch, [(None, False)])

    last_progress, extended = await _extend_lease_during_prefill(
        srv, "sess-1", llama_port=8080, slot_id=None, last_progress=0
    )
    assert extended is False
    assert last_progress == 0
    assert srv.local_dispatch_records["sess-1"]["expires_at"] == original_expiry
    assert not any("lease_extended_during_prefill" in line for line in _info_log_lines(srv.logger))


@pytest.mark.asyncio
async def test_extend_lease_disabled_when_poll_cadence_zero(monkeypatch):
    """poll cadence 0 disables progress-based extension entirely."""
    from proxy.router_helpers import _extend_lease_during_prefill

    config = copy.deepcopy(BASE_SERVER_CONFIG)
    config["server"]["local_dispatch_lease_prefill_poll_seconds"] = 0
    now = time.monotonic()
    original_expiry = now + 0.3
    srv = _make_srv(
        records={"sess-1": {"backend": "local", "started_at": now, "active": True, "expires_at": original_expiry}},
        config=config,
    )
    calls = []

    async def _fake_progress(*args, **kwargs):
        calls.append(1)
        return (1000, True)

    monkeypatch.setattr("proxy.router_helpers._query_prefill_progress", _fake_progress)

    last_progress, extended = await _extend_lease_during_prefill(
        srv, "sess-1", llama_port=8080, slot_id=None, last_progress=0
    )
    assert not calls, "No status polling should occur when poll cadence is 0"
    assert extended is False
    assert srv.local_dispatch_records["sess-1"]["expires_at"] == original_expiry


@pytest.mark.asyncio
async def test_extend_lease_skips_inactive_record(monkeypatch):
    """Inactive (post-request) records are never extended."""
    from proxy.router_helpers import _extend_lease_during_prefill

    now = time.monotonic()
    original_expiry = now + 60.0
    srv = _make_srv(records={
        "sess-1": {"backend": "local", "started_at": now, "active": False, "expires_at": original_expiry},
    })
    _install_fake_progress(monkeypatch, [(5000, True)])

    last_progress, extended = await _extend_lease_during_prefill(
        srv, "sess-1", llama_port=8080, slot_id=None, last_progress=0
    )
    assert extended is False
    assert srv.local_dispatch_records["sess-1"]["expires_at"] == original_expiry


@pytest.mark.asyncio
async def test_extend_lease_uses_configured_buffer(monkeypatch):
    """The safety buffer is configurable."""
    from proxy.router_helpers import _extend_lease_during_prefill

    config = copy.deepcopy(BASE_SERVER_CONFIG)
    config["server"]["local_dispatch_lease_prefill_buffer_seconds"] = 60
    now = time.monotonic()
    srv = _make_srv(
        records={"sess-1": {"backend": "local", "started_at": now, "active": True, "expires_at": now + 0.3}},
        config=config,
    )
    _install_fake_progress(monkeypatch, [(1000, True)])

    _, extended = await _extend_lease_during_prefill(
        srv, "sess-1", llama_port=8080, slot_id=None, last_progress=0
    )
    assert extended is True
    remaining = srv.local_dispatch_records["sess-1"]["expires_at"] - time.monotonic()
    assert 59.0 <= remaining <= 60.0, (
        f"Lease should be extended by the configured 60s buffer, got {remaining:.1f}s remaining"
    )


@pytest.mark.asyncio
async def test_cleanup_loop_preserves_extended_prefill_lease(monkeypatch):
    """AC4: >=5 repeated cleanup-loop runs never orphan-clean an active
    prefill whose lease was progress-extended past the base timeout."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
        _extend_lease_during_prefill,
        _increment_local_active_queries,
    )

    srv = _make_srv()
    # A large-prompt explicit-session request acquires an adaptive lease.
    body = {"model": "test", "messages": [{"role": "user", "content": "a" * 200_000}]}
    await _increment_local_active_queries(
        srv, session_key="sess-prefill", backend="local", body_json=body
    )

    # Simulate the prefill already running past the base 60s lease: shift
    # started_at back; expires_at is still valid because the prefill has
    # been observed making progress.
    record = srv.local_dispatch_records["sess-prefill"]
    record["started_at"] -= 120.0

    _install_fake_progress(monkeypatch, [(12345, True)])
    _, extended = await _extend_lease_during_prefill(
        srv, "sess-prefill", llama_port=8080, slot_id=None, last_progress=0
    )
    assert extended is True
    assert srv.local_dispatch_records["sess-prefill"]["expires_at"] > time.monotonic()

    for iteration in range(5):
        removed = await _cleanup_stale_local_dispatch(srv)
        assert removed == 0, (
            f"Cleanup iteration {iteration} removed {removed} record(s); an active "
            f"prefill lease with future expires_at must be preserved"
        )
        assert "sess-prefill" in srv.local_dispatch_records, (
            f"Cleanup iteration {iteration} orphan-cleaned the active prefill"
        )

    assert srv.local_dispatch_records["sess-prefill"]["expires_at"] > time.monotonic()

    orphan_warnings = [
        call for call in srv.logger.warning.call_args_list
        if "reason=orphan_cleanup" in str(call)
    ]
    assert not orphan_warnings, (
        f"Expected no orphan_cleanup warnings, got {len(orphan_warnings)}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Stream-loop integration: explicit-session prefill keeps its lease
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_stream_loop_keeps_lease_during_long_prefill(monkeypatch):
    """AC1+AC3: An explicit-session request whose prefill emits no data
    chunks for longer than the base lease keeps its dispatch lease for the
    full prefill duration (progress-based extension), then the lease is
    released once the stream completes — with no orphan_cleanup."""
    from proxy.router import proxy_to_local

    snapshots = []
    progress = [0]
    monkeypatch.setattr(
        "proxy.observability.query_llama_status",
        _make_advancing_status("test-session-id", snapshots, progress),
    )

    first_chunk_ts = [None]

    async def _delayed_first_chunk():
        await asyncio.sleep(0.6)  # prefill: 2x the 0.3s base lease
        first_chunk_ts[0] = time.monotonic()
        yield b'data: {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}\n\n'
        yield b'data: {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}\n\n'
        yield b"data: [DONE]\n\n"

    cm, resp = _make_mock_cm(_delayed_first_chunk)
    monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, resp)))
    monkeypatch.setattr(
        "proxy.router._call_with_empty_retry",
        AsyncMock(return_value=resp),
    )
    monkeypatch.setattr("proxy.router._update_session_and_slot", AsyncMock(return_value=None))

    response = await proxy_to_local(
        _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hi"}]}, stream=True),
        "v1/chat/completions",
    )
    collected = await _collect_streamed_chunks(response)

    assert b"Hello" in collected, "Streamed response should contain the chunk"

    record = server.local_dispatch_records.get("test-session-id")
    assert record is not None, "Dispatch record must exist for the explicit session"
    assert record["active"] is False, "Lease should be released (inactive) after stream completes"

    # Progress was polled multiple times during the prefill phase.
    assert len(snapshots) >= 4, (
        f"Expected multiple prefill-progress polls during a 0.6s prefill at 0.05s "
        f"cadence, got {len(snapshots)}"
    )

    # The lease was extended well beyond the 0.3s base lease while prefill
    # was advancing: snapshots capture expires_at before each poll's
    # extension, so the last snapshot reflects the previous extension.
    base_lease = BASE_SERVER_CONFIG["server"]["local_dispatch_lease_timeout_seconds"]
    max_expiry = max(s["expires_at"] for s in snapshots)
    started_at = record["started_at"]
    assert max_expiry - started_at > base_lease + 0.2, (
        f"Lease (max expires_at {max_expiry - started_at:.2f}s after start) should "
        f"exceed the base lease {base_lease}s + buffer margin during prefill"
    )

    # Extension/polling stopped once the first data chunk arrived (prefill
    # complete): every poll happened before the first chunk.
    assert first_chunk_ts[0] is not None
    last_poll_t = max(s["t"] for s in snapshots)
    assert last_poll_t <= first_chunk_ts[0] + 0.1, (
        f"Last prefill poll ({last_poll_t - started_at:.2f}s) should occur before "
        f"the first chunk ({first_chunk_ts[0] - started_at:.2f}s)"
    )

    # lease_extended_during_prefill was logged during the prefill phase.
    assert any("lease_extended_during_prefill" in line for line in _info_log_lines(server.logger)), (
        "Expected lease_extended_during_prefill log events"
    )

    # No orphan_cleanup of the active prefill.
    orphan_warnings = [
        call for call in server.logger.warning.call_args_list
        if "reason=orphan_cleanup" in str(call)
    ]
    assert not orphan_warnings, f"Expected no orphan_cleanup warnings, got {len(orphan_warnings)}"


@pytest.mark.asyncio
async def test_stream_loop_small_prompt_no_extension(monkeypatch):
    """AC3: A small prompt with an immediate first chunk never triggers
    prefill lease extension — the base lease is unchanged."""
    from proxy.router import proxy_to_local

    monkeypatch.setattr(
        "proxy.observability.query_llama_status",
        AsyncMock(return_value={
            "llama_server_running": True,
            "n_ctx": 32768,
            "kv_cache_tokens": 0,  # never advancing → never extends
            "router_mode": False,
        }),
    )

    async def _immediate_chunks():
        yield b'data: {"choices": [{"delta": {"content": "Hi"}, "index": 0}]}\n\n'
        yield b'data: {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}\n\n'
        yield b"data: [DONE]\n\n"

    cm, resp = _make_mock_cm(_immediate_chunks)
    monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, resp)))
    monkeypatch.setattr(
        "proxy.router._call_with_empty_retry",
        AsyncMock(return_value=resp),
    )
    monkeypatch.setattr("proxy.router._update_session_and_slot", AsyncMock(return_value=None))

    response = await proxy_to_local(
        _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hello"}]}, stream=True),
        "v1/chat/completions",
    )
    collected = await _collect_streamed_chunks(response)
    assert b"Hi" in collected

    # No prefill extension events for a small prompt.
    assert not any(
        "lease_extended_during_prefill" in line for line in _info_log_lines(server.logger)
    ), "Small prompts must not trigger prefill lease extension"

    record = server.local_dispatch_records.get("test-session-id")
    assert record is not None and record["active"] is False, (
        "Lease should be released after the small prompt completes"
    )


@pytest.mark.asyncio
async def test_stream_loop_anonymous_session_not_extended(monkeypatch):
    """Explicit-session gate: anonymous/non-explicit sessions never get
    prefill progress polling or lease extension (owned by LP-0MSEHMMBK0062ZPI)."""
    from proxy.router import proxy_to_local

    status_calls = []

    async def _status():
        status_calls.append(1)
        return {"llama_server_running": True, "n_ctx": 32768, "kv_cache_tokens": 9000, "router_mode": False}

    monkeypatch.setattr("proxy.observability.query_llama_status", _status)
    monkeypatch.setattr(
        "proxy.router._handle_session",
        AsyncMock(return_value={
            "session_id": "anon-session",
            "session_id_header": "anon-session",
            "session_explicit": False,
            "session_created": True,
            "is_delta_request": False,
            "session_fallback_reason": None,
            "delta_messages": [],
            "original_message_count": 1,
            "body_override": None,
            "body_json": None,
        }),
    )

    async def _delayed_first_chunk():
        await asyncio.sleep(0.15)
        yield b'data: {"choices": [{"delta": {"content": "Hi"}, "index": 0}]}\n\n'
        yield b'data: {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}\n\n'
        yield b"data: [DONE]\n\n"

    cm, resp = _make_mock_cm(_delayed_first_chunk)
    monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, resp)))
    monkeypatch.setattr(
        "proxy.router._call_with_empty_retry",
        AsyncMock(return_value=resp),
    )
    monkeypatch.setattr("proxy.router._update_session_and_slot", AsyncMock(return_value=None))

    response = await proxy_to_local(
        _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hello"}]}, stream=True),
        "v1/chat/completions",
    )
    collected = await _collect_streamed_chunks(response)
    assert b"Hi" in collected

    assert not status_calls, "Anonymous sessions must not poll prefill progress"
    assert not any(
        "lease_extended_during_prefill" in line for line in _info_log_lines(server.logger)
    ), "Anonymous sessions must not extend the lease during prefill"


@pytest.mark.asyncio
async def test_stream_loop_adaptive_fallback_when_status_unobservable(monkeypatch):
    """Fallback: when progress is unobservable the explicit-session lease is
    not dropped — it keeps its adaptive acquisition-time expiry and the
    stream completes normally."""
    from proxy.router import proxy_to_local

    snapshots = []

    async def _status():
        record = server.local_dispatch_records.get("test-session-id")
        if record is not None:
            snapshots.append(record["expires_at"])
        return {"llama_server_running": True, "n_ctx": 32768, "kv_cache_tokens": None, "router_mode": False}

    monkeypatch.setattr("proxy.observability.query_llama_status", _status)

    async def _delayed_first_chunk():
        await asyncio.sleep(0.15)  # shorter than the adaptive lease
        yield b'data: {"choices": [{"delta": {"content": "Hi"}, "index": 0}]}\n\n'
        yield b'data: {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}\n\n'
        yield b"data: [DONE]\n\n"

    cm, resp = _make_mock_cm(_delayed_first_chunk)
    monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, resp)))
    monkeypatch.setattr(
        "proxy.router._call_with_empty_retry",
        AsyncMock(return_value=resp),
    )
    monkeypatch.setattr("proxy.router._update_session_and_slot", AsyncMock(return_value=None))

    response = await proxy_to_local(
        _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hello"}]}, stream=True),
        "v1/chat/completions",
    )
    collected = await _collect_streamed_chunks(response)
    assert b"Hi" in collected

    # The lease was polled but never extended (unobservable), yet the
    # record persisted with a future expires_at throughout the prefill —
    # the adaptive estimate from acquisition was not dropped.
    assert len(snapshots) >= 2, "Expected progress polls during prefill"
    assert all(exp > time.monotonic() - 2.0 for exp in snapshots), (
        "Lease should remain valid (future expires_at) throughout the prefill"
    )
    assert not any(
        "lease_extended_during_prefill" in line for line in _info_log_lines(server.logger)
    ), "Unobservable progress must not log extension events"
