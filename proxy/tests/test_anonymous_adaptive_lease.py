"""
Hermetic tests for the adaptive dispatch lease applied to anonymous/non-explicit
sessions during large prefills (LP-0MSEHMMBK0062ZPI).

Context
-------
The adaptive lease timeout (base 60s + est_tokens x per_token_seconds, capped at
1500s) was previously applied only to *explicit* sessions via
``_try_acquire_local_dispatch`` (router.py gates it on ``session_id and
session_explicit``). Anonymous/non-explicit requests (no X-Session-Id header,
e.g. curl, integrations, benchmarks) create their dispatch record via
``_increment_local_active_queries`` with the plain base 60s lease — insufficient
for >2-minute cache prefills where no stream chunks arrive to refresh the lease.
The 10s cleanup loop then orphan-cleaned the active lease mid-prefill
(``reason=orphan_cleanup``) and the prefill restarted.

These tests verify hermetically (fake server state, no live llama-server):

1. A large-prompt anonymous session acquires an adaptive lease
   (>= base + est_tokens x per_token_seconds, <= max cap 1500).
2. A small-prompt anonymous session keeps ~base 60s lease (cross-session
   blocking not increased).
3. Repeated cleanup-loop runs never orphan-clean an active prefill lease
   whose expires_at is still in the future.
4. The router wires ``body_json`` through the anonymous call site so a
   non-streaming large-prompt request holds an adaptive lease mid-prefill.
"""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import proxy.server as server
import pytest

BASE_SERVER_CONFIG = {
    "server": {
        "llama_router_mode": False,
        "llama_server_port": 8080,
        "max_concurrent_queries": 4,
        "local_max_concurrent_queries": 1,
        "llama_request_timeout": 30,
        "local_dispatch_lease_timeout_seconds": 60,
        "local_dispatch_lease_per_token_seconds": 0.015,
        "local_dispatch_lease_max_seconds": 1500,
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

# Documented heuristic (proxy/lifecycle.py:_estimate_prompt_tokens): ~4 bytes
# per token. "a" * 200_000 chars -> 50_000 est tokens -> 60 + 50_000*0.015
# = 810s adaptive lease (below the 1500s cap).
_LARGE_CONTENT = "a" * 200_000
_EXPECTED_LARGE_TOKENS = len(_LARGE_CONTENT) // 4  # 50_000
_EXPECTED_LARGE_LEASE = 60 + _EXPECTED_LARGE_TOKENS * 0.015  # 810.0


def _anon_body(content: str) -> dict:
    return {"model": "test", "messages": [{"role": "user", "content": content}]}


def _make_srv(records: dict | None = None) -> SimpleNamespace:
    """Build a fake server object with dispatch-tracking state."""
    return SimpleNamespace(
        config=dict(BASE_SERVER_CONFIG),
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records=records if records is not None else {},
        local_dispatch_records_lock=asyncio.Lock(),
        logger=MagicMock(),
    )


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


async def _collect_streamed_chunks(resp):
    collected = b""
    async for chunk in resp.body_iterator:
        collected += chunk
    return collected


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests: _increment_local_active_queries + _cleanup_stale_local_dispatch
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_anonymous_large_prompt_adaptive_lease():
    """AC1: Large-prompt anonymous session gets an adaptive lease.

    ``_increment_local_active_queries`` (the path anonymous requests take)
    must apply the adaptive lease timeout when a large ``body_json`` is
    provided: lease duration >= base + est_tokens x per_token_seconds and
    <= the max cap (1500s).
    """
    from proxy.router_helpers import _increment_local_active_queries

    srv = _make_srv()

    await _increment_local_active_queries(
        srv,
        session_key="anon-large",
        backend="local",
        body_json=_anon_body(_LARGE_CONTENT),
    )

    assert "anon-large" in srv.local_dispatch_records, (
        "Anonymous dispatch record should be created"
    )
    record = srv.local_dispatch_records["anon-large"]
    assert record["active"] is True

    lease = record["expires_at"] - record["started_at"]
    assert lease >= _EXPECTED_LARGE_LEASE - 1, (
        f"Anonymous large-prompt lease ({lease:.0f}s) should be >= adaptive "
        f"value ({_EXPECTED_LARGE_LEASE:.0f}s = base 60 + "
        f"{_EXPECTED_LARGE_TOKENS} tokens x 0.015)"
    )
    assert lease <= 1500, (
        f"Anonymous large-prompt lease ({lease:.0f}s) must not exceed the "
        f"1500s max cap"
    )


@pytest.mark.asyncio
async def test_anonymous_large_prompt_lease_capped_at_max():
    """AC1 (cap branch): Very large prompts are capped at the max lease.

    A body large enough that base + est_tokens x per_token_seconds exceeds
    the configured cap must yield a lease equal to the cap (1500s), not
    blow past it.
    """
    from proxy.router_helpers import _increment_local_active_queries

    srv = _make_srv()

    # 500_000 chars -> 125_000 est tokens -> 60 + 125_000*0.015 = 1935s
    # (above the 1500s cap, so the lease must be pinned at 1500).
    huge_body = _anon_body("a" * 500_000)
    await _increment_local_active_queries(
        srv,
        session_key="anon-huge",
        backend="local",
        body_json=huge_body,
    )

    record = srv.local_dispatch_records["anon-huge"]
    lease = record["expires_at"] - record["started_at"]
    assert lease <= 1500 + 1, (
        f"Capped anonymous lease ({lease:.0f}s) must not exceed 1500s"
    )
    assert lease >= 1499, (
        f"Capped anonymous lease ({lease:.0f}s) should be pinned near the "
        f"1500s cap for a huge prompt"
    )


@pytest.mark.asyncio
async def test_anonymous_small_prompt_base_lease():
    """AC2: Small-prompt anonymous session keeps ~base 60s lease.

    Cross-session blocking must not increase for small prompts: the lease
    stays approximately at the base 60s (adaptive extension is negligible).
    """
    from proxy.router_helpers import _increment_local_active_queries

    srv = _make_srv()

    await _increment_local_active_queries(
        srv,
        session_key="anon-small",
        backend="local",
        body_json=_anon_body("Hello, how are you?"),
    )

    assert "anon-small" in srv.local_dispatch_records
    record = srv.local_dispatch_records["anon-small"]
    lease = record["expires_at"] - record["started_at"]

    base_timeout = 60.0
    assert abs(lease - base_timeout) <= 5, (
        f"Anonymous small-prompt lease ({lease:.0f}s) should be approximately "
        f"the base 60s, got a {lease - base_timeout:+.0f}s deviation"
    )


@pytest.mark.asyncio
async def test_cleanup_loop_preserves_active_prefill_lease():
    """AC3: Repeated cleanup-loop runs never orphan-clean an active prefill.

    Simulates the F3 hazard: an anonymous large-prompt prefill has already
    run past the base 60s lease (120s of prefill, no chunks) while holding
    an adaptive lease. Running the cleanup-loop logic >= 3 times (the real
    loop fires every 10s) must not remove the record, and no
    ``reason=orphan_cleanup`` warning may be emitted.
    """
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
        _increment_local_active_queries,
    )

    srv = _make_srv()

    # Acquire the adaptive lease exactly as an anonymous request would.
    await _increment_local_active_queries(
        srv,
        session_key="anon-prefill",
        backend="local",
        body_json=_anon_body(_LARGE_CONTENT),
    )

    record = srv.local_dispatch_records["anon-prefill"]
    adaptive_lease = record["expires_at"] - record["started_at"]
    assert adaptive_lease > 60, (
        f"Adaptive lease ({adaptive_lease:.0f}s) must exceed the base 60s "
        f"lease so prefills longer than 60s are covered"
    )

    # Simulate the prefill already running for 120s (> base 60s lease) with
    # no stream chunks arriving. expires_at stays in the future because the
    # lease was adaptively extended at acquisition.
    elapsed_prefill = 120.0
    record["started_at"] -= elapsed_prefill

    for iteration in range(5):  # >= 3 repeated cleanup-loop runs
        removed = await _cleanup_stale_local_dispatch(srv)
        assert removed == 0, (
            f"Cleanup iteration {iteration} removed {removed} record(s); an "
            f"active prefill lease with future expires_at must be preserved"
        )
        assert "anon-prefill" in srv.local_dispatch_records, (
            f"Cleanup iteration {iteration} orphan-cleaned the active prefill"
        )

    # The lease must still be in the future after all iterations.
    assert srv.local_dispatch_records["anon-prefill"]["expires_at"] > time.monotonic()

    # No orphan_cleanup WARNING may have been emitted for this session.
    orphan_warnings = [
        call for call in srv.logger.warning.call_args_list
        if "reason=orphan_cleanup" in str(call)
    ]
    assert not orphan_warnings, (
        f"Expected no orphan_cleanup warnings, got {len(orphan_warnings)}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Router-path test: non-streaming request from an anonymous session
# ═══════════════════════════════════════════════════════════════════════════════


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


@pytest.fixture(autouse=True)
def _reset_server_state(monkeypatch):
    """Reset server-level state before each test."""
    monkeypatch.setattr(server, "config", dict(BASE_SERVER_CONFIG))
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


@pytest.mark.asyncio
async def test_router_non_streaming_anonymous_adaptive_lease(monkeypatch):
    """AC1 via router: non-streaming large-prompt anonymous request holds an
    adaptive lease mid-prefill.

    The anonymous call site in proxy_to_local must pass ``body_json`` through
    to ``_increment_local_active_queries``. We assert the dispatch record's
    lease is adaptive at the moment the backend is about to start the prefill
    (i.e., while no chunks have arrived yet).
    """
    from proxy.router import proxy_to_local

    # Anonymous session: no explicit header, auto-generated session id.
    monkeypatch.setattr(
        "proxy.router._handle_session",
        AsyncMock(return_value={
            "session_id": "anon-adaptive-test",
            "session_id_header": "anon-adaptive-test",
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

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.content = (
        b'{"id":"test","choices":[{"message":{"role":"assistant",'
        b'"content":"ok"},"finish_reason":"stop"}]}'
    )

    # The router catches exceptions from the backend call and converts them
    # to a 503, which would swallow assertion messages. So we record the
    # dispatch-record state at the exact prefill moment (inside the backend
    # call, before any chunks arrive) and assert on it after the response.
    observed = {}

    async def _backend_call(_send_once, path=None, stream=None):
        # This runs after _increment_local_active_queries created the record
        # and before any chunks arrive — the prefill phase.
        record = server.local_dispatch_records.get("anon-adaptive-test")
        if record is not None:
            observed["record_exists"] = True
            observed["active"] = record.get("active")
            observed["lease"] = record["expires_at"] - record["started_at"]
        return mock_resp

    monkeypatch.setattr(
        "proxy.router._call_with_backend_retries", _backend_call
    )
    monkeypatch.setattr(
        "proxy.router._call_with_empty_retry",
        AsyncMock(return_value=mock_resp),
    )
    monkeypatch.setattr(
        "proxy.router._update_session_and_slot", AsyncMock(return_value=None)
    )

    response = await proxy_to_local(
        _dummy_request(_anon_body(_LARGE_CONTENT), stream=False),
        "v1/chat/completions",
    )

    assert response.status_code == 200
    assert observed.get("record_exists"), (
        "Anonymous dispatch record must exist during the prefill phase"
    )
    assert observed["active"] is True
    assert observed["lease"] >= _EXPECTED_LARGE_LEASE - 1, (
        f"Router-path anonymous lease ({observed['lease']:.0f}s) should be "
        f"adaptive (>= {_EXPECTED_LARGE_LEASE:.0f}s = base 60 + "
        f"{_EXPECTED_LARGE_TOKENS} tokens x 0.015), got base 60s behaviour"
    )
    assert observed["lease"] <= 1500


# ═══════════════════════════════════════════════════════════════════════════════
# Chunk-refresh lease path for non-explicit sessions (LP-0MSUO6HLX0089MNQ)
# ═══════════════════════════════════════════════════════════════════════════════


def _chunk_refresh_config() -> dict:
    """Config with a short base lease so a multi-chunk stream outlives it."""
    cfg = dict(BASE_SERVER_CONFIG)
    cfg["server"]["local_dispatch_lease_timeout_seconds"] = 0.3
    cfg["server"]["local_dispatch_lease_per_token_seconds"] = 0.0
    cfg["server"]["local_dispatch_lease_chunk_refresh_buffer_seconds"] = 1.0
    return cfg


@pytest.mark.asyncio
async def test_chunk_refresh_buffer_defaults():
    """Default chunk-refresh safety buffer is 30s (production config)."""
    from proxy.router_helpers import _get_chunk_refresh_buffer_seconds

    srv = _make_srv()
    assert _get_chunk_refresh_buffer_seconds(srv) == 30.0, (
        "Default chunk-refresh buffer should be 30s"
    )


@pytest.mark.asyncio
async def test_chunk_refresh_buffer_explicit_config():
    """Explicit chunk-refresh buffer config is honoured."""
    from proxy.router_helpers import _get_chunk_refresh_buffer_seconds

    srv = _make_srv()
    srv.config["server"]["local_dispatch_lease_chunk_refresh_buffer_seconds"] = 12
    assert _get_chunk_refresh_buffer_seconds(srv) == 12.0


@pytest.mark.asyncio
async def test_router_streaming_anonymous_chunk_refresh(monkeypatch):
    """AC1: a non-explicit (anonymous) session with an active stream gets its
    dispatch lease refreshed on data-chunk arrival — the lease survives a
    stream longer than the 0.3s base lease with no orphan_cleanup.

    This mirrors the explicit-session chunk-refresh path but exercises the
    anonymous branch (``_has_actual_data and session_id`` without
    ``session_explicit``), which uses the dedicated chunk-refresh buffer
    (LP-0MSUO6HLX0089MNQ).
    """
    from proxy.router import proxy_to_local

    monkeypatch.setattr(server, "config", _chunk_refresh_config())

    # Anonymous session: no explicit header.
    monkeypatch.setattr(
        "proxy.router._handle_session",
        AsyncMock(return_value={
            "session_id": "anon-stream",
            "session_id_header": "anon-stream",
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

    # A stream whose total duration (0.8s) exceeds the 0.3s base lease;
    # chunks arrive every 0.25s, so without chunk-refresh the record would
    # be orphan-cleaned mid-stream (cleanup loop cadence ~0.1s in tests).
    async def _multi_chunk_stream():
        await asyncio.sleep(0.25)
        yield b'data: {"choices": [{"delta": {"content": "one"}, "index": 0}]}\n\n'
        await asyncio.sleep(0.25)
        yield b'data: {"choices": [{"delta": {"content": "two"}, "index": 0}]}\n\n'
        await asyncio.sleep(0.25)
        yield b'data: {"choices": [{"delta": {"content": "three"}, "index": 0}]}\n\n'
        yield b'data: {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}\n\n'
        yield b"data: [DONE]\n\n"

    cm, resp = _make_mock_cm(_multi_chunk_stream)
    monkeypatch.setattr("proxy.router._call_with_backend_retries", AsyncMock(return_value=(cm, resp)))
    monkeypatch.setattr(
        "proxy.router._call_with_empty_retry",
        AsyncMock(return_value=resp),
    )
    monkeypatch.setattr("proxy.router._update_session_and_slot", AsyncMock(return_value=None))

    # Snapshot the dispatch record's expires_at as chunks arrive: capture it
    # from the stream generator between yields (the router refreshes after
    # consuming each chunk, before requesting the next).
    snapshots = []
    _orig_aiter = resp.aiter_bytes

    async def _observing_aiter():
        async for chunk in _orig_aiter():
            record = server.local_dispatch_records.get("anon-stream")
            if record is not None:
                snapshots.append({
                    "t": time.monotonic(),
                    "expires_at": record["expires_at"],
                    "active": record.get("active"),
                })
            yield chunk

    resp.aiter_bytes = _observing_aiter

    response = await proxy_to_local(
        _dummy_request({"model": "test", "messages": [{"role": "user", "content": "hello"}]}, stream=True),
        "v1/chat/completions",
    )
    collected = await _collect_streamed_chunks(response)
    assert b"one" in collected and b"three" in collected, (
        "Streamed response should contain all chunks"
    )

    assert len(snapshots) >= 3, (
        f"Expected to observe the dispatch record across >=3 chunks, got {len(snapshots)}"
    )
    # Every chunk-refresh pushes expires_at forward: the lease remaining at
    # each observation stays above the 0.3s base lease horizon (buffer 1.0s
    # minus the 0.25s chunk cadence leaves ~0.75s steady-state headroom).
    # The first snapshot may precede the first chunk's refresh (acquisition
    # lease only), so it is excluded from the headroom assertion.
    for snap in snapshots[1:]:
        assert snap["active"] is True
        remaining = snap["expires_at"] - snap["t"]
        assert remaining > 0.3, (
            f"Lease remaining ({remaining:.2f}s) must exceed the base 0.3s "
            f"lease while the stream is active — chunk-refresh not applied"
        )

    # The lease must be seen advancing as chunks arrive (refreshed, not
    # just initially long): later expiries exceed earlier ones.
    expiries = [s["expires_at"] for s in snapshots]
    assert max(expiries) > snapshots[0]["expires_at"] + 0.2, (
        f"Chunk-refresh must push expires_at forward (first {snapshots[0]['expires_at']:.2f}, "
        f"max {max(expiries):.2f})"
    )

    # No orphan_cleanup may have been emitted for the anonymous stream.
    orphan_warnings = [
        call for call in server.logger.warning.call_args_list
        if "reason=orphan_cleanup" in str(call)
    ]
    assert not orphan_warnings, (
        f"Expected no orphan_cleanup warnings, got {len(orphan_warnings)}"
    )


@pytest.mark.asyncio
async def test_cleanup_loop_preserves_anonymous_chunk_refreshed_lease(monkeypatch):
    """AC2: after chunk-refresh extends an anonymous lease, repeated cleanup
    loop runs never orphan-clean it — the record survives while active."""
    from proxy.router_helpers import (
        _cleanup_stale_local_dispatch,
        _increment_local_active_queries,
    )

    srv = _make_srv()
    srv.config["server"]["local_dispatch_lease_timeout_seconds"] = 0.3
    srv.config["server"]["local_dispatch_lease_chunk_refresh_buffer_seconds"] = 0.5

    await _increment_local_active_queries(
        srv,
        session_key="anon-refresh",
        backend="local",
        body_json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
    )
    record = srv.local_dispatch_records["anon-refresh"]
    # Simulate the stream having run past the base lease; a chunk just
    # arrived and refreshed expires_at to now + 0.5s buffer.
    record["started_at"] -= 0.4
    record["expires_at"] = time.monotonic() + 0.5

    for iteration in range(5):
        removed = await _cleanup_stale_local_dispatch(srv)
        assert removed == 0, (
            f"Cleanup iteration {iteration} removed {removed} record(s); an "
            f"active chunk-refreshed lease must be preserved"
        )
        assert "anon-refresh" in srv.local_dispatch_records, (
            f"Cleanup iteration {iteration} orphan-cleaned the refreshed lease"
        )

    assert srv.local_dispatch_records["anon-refresh"]["expires_at"] > time.monotonic()
    orphan_warnings = [
        call for call in srv.logger.warning.call_args_list
        if "reason=orphan_cleanup" in str(call)
    ]
    assert not orphan_warnings
