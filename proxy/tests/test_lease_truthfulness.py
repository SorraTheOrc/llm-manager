"""Truthful lease and in-flight accounting (LP-0MVKUKFSO008GI6O).

Child C5 of the wedged prefill lease loop investigation
(LP-0MVKN9IRP003XJH6). On 2026-10-23 the local dispatch gate was held for
~4 h by a session whose prefill never produced a byte:

``/var/log/llama-proxy`` evidence::

    lease_extended_during_prefill progress=None alive=False in_flight=True   (every 10 s)
    prefill_deadline_exceeded ... -> detour
    lease_renewed session=01a16636... timeout=30s                           (lease retained)
    local_admission_guard ... busy_slots=1 in_flight_prefills=1 available_slots=0
    local_dispatch_denied owner=<wedged>                                    (2,056 times)
    lease_released session=... reason=session_evicted                        (~4 h later)

Three accounting defects compose the wedge, each covered here:

- AC1 — a dispatch cancelled by the prefill deadline retained its lease
  (``lease_renewed``) and left its ``local_prefill_in_flight`` marker behind.
  It must delete the record and pop the marker, logging
  ``lease_released reason=prefill_deadline``.
- AC2 — the prefill-lease extension must be scoped to the record's own
  session: another session's in-flight request must never refresh this
  record's ``expires_at`` / ``last_progress_ts``.
- AC3 — the admission gate's ``busy_slots`` must count only in-flight entries
  backed by a live dispatch record, so an orphan marker cannot report
  ``available_slots=0`` for every other session.
- AC4 — the 180 s no-progress watchdog must still fire for a genuinely
  progress-less but in-flight record: an open request is not progress, so the
  in-flight signal may extend ``expires_at`` but must not refresh
  ``last_progress_ts`` (LP-0MUXQ5QL40027VZ4 liveness behaviour is unchanged).

The router-level AC1 test drives ``proxy_to_local`` with a stubbed upstream,
so the deadline branch's release wiring is exercised end to end rather than
by inspecting source.
"""

from __future__ import annotations

import asyncio
import copy
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import proxy.provider as provider
import proxy.router_helpers as helpers
import proxy.server as server
import pytest

SERVER_CONFIG = {
    "server": {
        "llama_router_mode": False,
        "llama_server_port": 8080,
        "max_concurrent_queries": 4,
        "session_slot_pool_size": 1,
        "local_dispatch_lease_timeout_seconds": 0.3,
        "local_dispatch_lease_per_token_seconds": 0.0,
        "local_dispatch_lease_max_seconds": 1500,
        "local_dispatch_lease_prefill_poll_seconds": 0.05,
        "local_dispatch_lease_prefill_buffer_seconds": 0.5,
        "local_dispatch_no_progress_timeout_seconds": 180,
        "local_dispatch_max_prefill_seconds": 900,
        "local_prefill_deadline_seconds": 0.05,
        "local_dispatch_first_byte_timeout_seconds": 840,
        "local_dispatch_prefill_observability_warn_seconds": 0,
        "session_single_flight_mode": "bypass",
        "disconnect_cleanup_timeout": 1,
        "stream_heartbeat_interval_seconds": 0.05,
        "stream_idle_timeout_seconds": 0.3,
        "local_max_concurrent_prefills": 1,
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


def _make_srv(records=None, markers=None) -> SimpleNamespace:
    """Fake server with dispatch-record and prefill-marker state."""
    return SimpleNamespace(
        config=copy.deepcopy(SERVER_CONFIG),
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records=records if records is not None else {},
        local_dispatch_records_lock=asyncio.Lock(),
        local_prefill_in_flight=markers if markers is not None else {},
        local_prefill_in_flight_lock=asyncio.Lock(),
        logger=MagicMock(),
    )


def _record(now: float, **overrides) -> dict:
    record = {
        "backend": "local",
        "started_at": now,
        "active": True,
        "expires_at": now + 0.3,
        "model_name": "test-model",
        "last_progress": 0,
        "last_progress_ts": now,
    }
    record.update(overrides)
    return record


def _log_lines(logger, level="info") -> list[str]:
    return [str(call) for call in getattr(logger, level).call_args_list]


# ═══════════════════════════════════════════════════════════════════════════════
# AC1: deadline cancellation releases the lease and the in-flight marker
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_deadline_release_deletes_record_and_pops_marker():
    """AC1: the deadline release removes the record, pops the marker and emits
    ``lease_released ... reason=prefill_deadline``."""
    now = time.monotonic()
    srv = _make_srv(
        records={"sess-1": _record(now)},
        markers={"sess-1": True},
    )

    removed = await helpers._release_prefill_deadline_dispatch(srv, "sess-1")

    assert removed is True
    assert "sess-1" not in srv.local_dispatch_records, (
        "the prefill-deadline lease must not be retained (pre-fix: inactive "
        "record with a future expires_at, 'lease_renewed')"
    )
    assert "sess-1" not in srv.local_prefill_in_flight, (
        "the in-flight marker must be popped, not left to inflate occupancy"
    )
    lines = _log_lines(srv.logger)
    assert any(
        "lease_released" in line and "reason=prefill_deadline" in line
        for line in lines
    ), f"expected lease_released reason=prefill_deadline, got: {lines}"


@pytest.mark.asyncio
async def test_deadline_release_is_session_scoped_and_idempotent():
    """AC1: only the cancelled session's record/marker are released; a repeat
    call is a harmless no-op (the deadline path must never raise)."""
    now = time.monotonic()
    srv = _make_srv(
        records={
            "sess-1": _record(now),
            "sess-2": _record(now, last_progress_ts=now - 5.0),
        },
        markers={"sess-1": True, "sess-2": True},
    )
    other_before = dict(srv.local_dispatch_records["sess-2"])

    assert await helpers._release_prefill_deadline_dispatch(srv, "sess-1") is True
    assert srv.local_dispatch_records["sess-2"] == other_before, (
        "another session's lease must be untouched by a deadline release"
    )
    assert "sess-2" in srv.local_prefill_in_flight

    # Idempotent: nothing left to release, no raise, no extra log noise.
    srv.logger.info.reset_mock()
    assert await helpers._release_prefill_deadline_dispatch(srv, "sess-1") is False
    assert not any("reason=prefill_deadline" in line for line in _log_lines(srv.logger))


@pytest.mark.asyncio
async def test_deadline_release_pops_marker_without_a_record():
    """AC1: an orphaned marker (record already gone) is still popped — the
    non-explicit cleanup path deletes the record but left the marker behind."""
    srv = _make_srv(records={}, markers={"anon-session": True})

    assert await helpers._release_prefill_deadline_dispatch(srv, "anon-session") is False
    assert "anon-session" not in srv.local_prefill_in_flight


@pytest.mark.asyncio
async def test_deadline_release_handles_endpoint_scoped_keys():
    """AC1: per-endpoint (tuple) record keys are released too."""
    now = time.monotonic()
    endpoint = "http://127.0.0.1:8081"
    srv = _make_srv(
        records={(endpoint, "sess-1"): _record(now)},
        markers={"sess-1": True},
    )

    removed = await helpers._release_prefill_deadline_dispatch(
        srv, "sess-1", endpoint=endpoint
    )
    assert removed is True
    assert not srv.local_dispatch_records
    assert "sess-1" not in srv.local_prefill_in_flight


@pytest.mark.asyncio
async def test_router_deadline_detour_releases_lease_and_marker(monkeypatch):
    """AC1 (end-to-end): when ``prefill_deadline_exceeded`` fires inside
    ``proxy_to_local``, the dispatch record and in-flight marker are released
    before the request is detoured — the pre-fix path logged ``lease_renewed``
    and kept both."""
    from proxy.router import proxy_to_local

    monkeypatch.setattr(server, "config", copy.deepcopy(SERVER_CONFIG))
    monkeypatch.setattr(server, "active_queries", 0)
    monkeypatch.setattr(server, "active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_active_queries", 0)
    monkeypatch.setattr(server, "local_active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_generating_queries", 0)
    monkeypatch.setattr(server, "local_generating_queries_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_generating_sessions", set())
    monkeypatch.setattr(server, "local_prefill_in_flight", {})
    monkeypatch.setattr(server, "local_prefill_in_flight_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_dispatch_records", {})
    monkeypatch.setattr(server, "local_dispatch_records_lock", asyncio.Lock())
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "llama_process", MagicMock(poll=lambda: None, pid=1))
    monkeypatch.setattr(server, "current_model", "test-model")
    monkeypatch.setattr(server, "session_manager", MagicMock())
    monkeypatch.setattr(server, "logger", MagicMock())
    helpers._reset_prefill_thrash_cooldowns()

    monkeypatch.setattr("proxy.router._is_self_healing_active", lambda: False)
    monkeypatch.setattr(
        "proxy.router._handle_session",
        AsyncMock(
            return_value={
                "session_id": "deadline-session",
                "session_id_header": "deadline-session",
                "session_explicit": True,
                "session_created": False,
                "is_delta_request": False,
                "session_fallback_reason": None,
                "delta_messages": [],
                "original_message_count": 1,
                "body_override": None,
                "body_json": None,
            }
        ),
    )
    monkeypatch.setattr(
        "proxy.router._build_slot_context", MagicMock(return_value=(None, None, 3.0))
    )
    monkeypatch.setattr(
        "proxy.router._check_slot_availability", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "proxy.router._restore_slot_snapshot", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        "proxy.session._resolve_log_path",
        MagicMock(
            return_value=MagicMock(exists=lambda: False, stat=lambda: MagicMock(st_size=0))
        ),
    )
    # Prefill progress unobservable (the incident shape: /slots stale).
    monkeypatch.setattr(
        "proxy.router_helpers._query_prefill_progress",
        AsyncMock(return_value=(None, False)),
    )

    async def _never_returns(*_args, **_kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr(
        "proxy.router._call_with_backend_retries", _never_returns
    )
    monkeypatch.setattr(
        "proxy.router._call_with_empty_retry", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "proxy.router._update_session_and_slot", AsyncMock(return_value=None)
    )

    body = b'{"model":"test","messages":[{"role":"user","content":"hi"}],"stream":true}'

    class _Req:
        headers = {"host": "localhost", "x-session-id": "deadline-session"}
        method = "POST"
        url = type("U", (), {"path": "/v1/chat/completions"})()

        async def body(self):
            return body

        async def is_disconnected(self):
            return False

    # The dispatch acquires the lease and marks the prefill hold ...
    record_created = asyncio.Event()

    _orig_extend = helpers._extend_lease_during_prefill

    async def _spy_extend(srv, session_key, **kwargs):
        if server.local_dispatch_records.get(session_key):
            record_created.set()
        return await _orig_extend(srv, session_key, **kwargs)

    monkeypatch.setattr("proxy.router_helpers._extend_lease_during_prefill", _spy_extend)
    monkeypatch.setattr("proxy.router._extend_lease_during_prefill", _spy_extend)

    response = await proxy_to_local(_Req(), "v1/chat/completions")
    await asyncio.sleep(0)

    assert record_created.is_set(), "the dispatch must have acquired a lease"
    assert response.status_code >= 400, (
        "a deadline cancellation must detour with a retryable error response"
    )
    record = server.local_dispatch_records.get("deadline-session")
    assert record is None, (
        "AC1: the deadline-detoured dispatch must not retain a lease "
        f"(found active={record and record.get('active')})"
    )
    assert "deadline-session" not in server.local_prefill_in_flight, (
        "AC1: the deadline-detoured dispatch must not retain its in-flight marker"
    )
    lines = _log_lines(server.logger)
    assert any("reason=prefill_deadline" in line for line in lines), (
        f"expected a lease_released reason=prefill_deadline event, got: {lines[-10:]}"
    )
    # Cooldown (C4) is armed by the same branch.
    assert helpers._prefill_thrash_cooldown_remaining(
        "deadline-session", SERVER_CONFIG
    ) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# AC2: the prefill-lease extension is scoped to the record's own session
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_two_session_extension_never_touches_the_other_lease(monkeypatch):
    """AC2: extending session A refreshes only A's record; B's ``expires_at``
    and ``last_progress_ts`` are byte-for-byte unchanged, and B's lease is
    never extended on A's in-flight marker."""
    now = time.monotonic()
    srv = _make_srv(
        records={
            "sess-A": _record(now, expires_at=now + 0.3, last_progress_ts=now - 500.0),
            "sess-B": _record(now, expires_at=now + 0.3, last_progress_ts=now - 400.0),
        },
        markers={"sess-A": True},
    )
    b_before = copy.deepcopy(srv.local_dispatch_records["sess-B"])

    async def _unobservable(*_args, **_kwargs):
        return (None, False)

    monkeypatch.setattr(helpers, "_query_prefill_progress", _unobservable)

    # A's own in-flight marker extends A.
    _, extended_a = await helpers._extend_lease_during_prefill(
        srv, "sess-A", llama_port=8080, slot_id=0, last_progress=0,
        prefill_in_flight=True,
    )
    assert extended_a is True
    assert srv.local_dispatch_records["sess-A"]["expires_at"] > now + 0.3

    # B has no in-flight marker of its own: A's marker must not extend B,
    # even when the extension is invoked for B with ``prefill_in_flight``.
    _, extended_b = await helpers._extend_lease_during_prefill(
        srv, "sess-B", llama_port=8080, slot_id=0, last_progress=0,
        prefill_in_flight=True,
    )
    assert extended_b is False, (
        "another session's in-flight marker must not extend this lease"
    )
    assert srv.local_dispatch_records["sess-B"] == b_before, (
        "session B's expires_at / last_progress_ts must be unchanged"
    )


@pytest.mark.asyncio
async def test_own_marker_extends_only_when_it_belongs_to_the_session(monkeypatch):
    """AC2: with both markers present, each session extends only its own
    record and never the sibling's."""
    now = time.monotonic()
    endpoint = "http://127.0.0.1:8080"
    srv = _make_srv(
        records={
            (endpoint, "sess-A"): _record(now, last_progress_ts=now - 500.0),
            (endpoint, "sess-B"): _record(now, last_progress_ts=now - 400.0),
        },
        markers={"sess-A": True, "sess-B": True},
    )

    async def _unobservable(*_args, **_kwargs):
        return (None, False)

    monkeypatch.setattr(helpers, "_query_prefill_progress", _unobservable)

    _, extended_a = await helpers._extend_lease_during_prefill(
        srv, "sess-A", endpoint=endpoint, llama_port=8080, slot_id=0,
        last_progress=0, prefill_in_flight=True,
    )
    assert extended_a is True
    b_record = srv.local_dispatch_records[(endpoint, "sess-B")]
    assert b_record["expires_at"] == now + 0.3
    assert b_record["last_progress_ts"] == now - 400.0

    _, extended_b = await helpers._extend_lease_during_prefill(
        srv, "sess-B", endpoint=endpoint, llama_port=8080, slot_id=0,
        last_progress=0, prefill_in_flight=True,
    )
    assert extended_b is True
    a_record = srv.local_dispatch_records[(endpoint, "sess-A")]
    assert a_record["last_progress_ts"] == now - 500.0


# ═══════════════════════════════════════════════════════════════════════════════
# AC3: orphan in-flight markers do not raise busy_slots
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_orphan_in_flight_marker_does_not_raise_busy_slots(monkeypatch):
    """AC3: an in-flight entry with no live dispatch record is not occupancy —
    the incident's ``busy_slots=1 in_flight_prefills=1 available_slots=0``."""
    monkeypatch.setattr(
        server, "local_prefill_in_flight", {"wedged-session": True}, raising=False
    )
    monkeypatch.setattr(server, "local_dispatch_records", {}, raising=False)
    monkeypatch.setattr(
        provider, "_get_local_concurrency_info", lambda config, endpoint=None: (0, 1)
    )

    busy, in_flight = provider._local_gate_occupancy(SERVER_CONFIG, 1)

    assert in_flight == 0, (
        "an orphan marker (no live dispatch record) must not be counted as an "
        "in-flight prefill"
    )
    assert busy == 0, "an orphan marker must not report the gate as busy"


@pytest.mark.asyncio
async def test_live_in_flight_marker_still_counts_as_occupancy(monkeypatch):
    """AC3 counter-case: a marker backed by a live record still occupies the
    gate (an in-flight prefill must never read as 'free soon')."""
    now = time.monotonic()
    monkeypatch.setattr(
        server, "local_prefill_in_flight", {"sess-1": True}, raising=False
    )
    monkeypatch.setattr(
        server,
        "local_dispatch_records",
        {"sess-1": _record(now)},
        raising=False,
    )
    monkeypatch.setattr(
        provider, "_get_local_concurrency_info", lambda config, endpoint=None: (0, 1)
    )

    busy, in_flight = provider._local_gate_occupancy(SERVER_CONFIG, 1)

    assert in_flight == 1
    assert busy == 1


@pytest.mark.asyncio
async def test_retained_inactive_record_does_not_back_the_marker(monkeypatch):
    """AC3: the pre-fix retention shape — marker plus an inactive record with a
    future ``expires_at`` (``lease_renewed``) — is not an in-flight prefill.

    The dispatch is finished (no first byte was ever produced); only the lease
    was being retained for the owning session.
    """
    now = time.monotonic()
    monkeypatch.setattr(
        server, "local_prefill_in_flight", {"wedged": True}, raising=False
    )
    monkeypatch.setattr(
        server,
        "local_dispatch_records",
        {"wedged": _record(now, active=False, expires_at=now + 60.0)},
        raising=False,
    )
    monkeypatch.setattr(
        provider, "_get_local_concurrency_info", lambda config, endpoint=None: (0, 1)
    )

    busy, in_flight = provider._local_gate_occupancy(SERVER_CONFIG, 1)

    assert in_flight == 0
    assert busy == 0


@pytest.mark.asyncio
async def test_released_record_stops_counting_immediately(monkeypatch):
    """AC3: releasing the lease (AC1) drops the occupancy in the same step —
    the gate recovers as soon as the cancelled prefill is released."""
    now = time.monotonic()
    srv = _make_srv(records={"wedged": _record(now)}, markers={"wedged": True})
    monkeypatch.setattr(server, "local_prefill_in_flight", srv.local_prefill_in_flight, raising=False)
    monkeypatch.setattr(server, "local_dispatch_records", srv.local_dispatch_records, raising=False)
    monkeypatch.setattr(
        provider, "_get_local_concurrency_info", lambda config, endpoint=None: (0, 1)
    )
    assert provider._local_gate_occupancy(SERVER_CONFIG, 1) == (1, 1)

    await helpers._release_prefill_deadline_dispatch(srv, "wedged")

    assert provider._local_gate_occupancy(SERVER_CONFIG, 1) == (0, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# AC4: the no-progress watchdog still fires for a progress-less in-flight record
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_watchdog_fires_for_progress_less_in_flight_record(monkeypatch):
    """AC4: ``progress=None alive=False in_flight=True`` keeps the lease valid
    but must not stop the 180 s no-progress watchdog (the wedge mechanism)."""
    now = time.monotonic()
    srv = _make_srv(
        records={
            "sess-1": _record(
                now,
                started_at=now - 200.0,
                expires_at=now + 0.3,
                last_progress_ts=now - 200.0,
            )
        },
        markers={"sess-1": True},
    )

    async def _unobservable(*_args, **_kwargs):
        return (None, False)

    monkeypatch.setattr(helpers, "_query_prefill_progress", _unobservable)

    _, extended = await helpers._extend_lease_during_prefill(
        srv, "sess-1", llama_port=8080, slot_id=0, last_progress=0,
        prefill_in_flight=True,
    )
    assert extended is True, "the in-flight request must keep the lease valid"
    record = srv.local_dispatch_records["sess-1"]
    assert record["last_progress_ts"] == now - 200.0, (
        "an open request is not progress: last_progress_ts must not be refreshed"
    )
    assert helpers._watchdog_no_progress(srv, "sess-1", record) is True, (
        "the 180s no-progress watchdog must still release a progress-less "
        "in-flight record"
    )


@pytest.mark.asyncio
async def test_liveness_extension_still_refreshes_the_watchdog_clock(monkeypatch):
    """AC4 (no regression to LP-0MUXQ5QL40027VZ4): observed slot liveness does
    refresh the no-progress clock, so a genuinely running prefill is not
    killed by the watchdog."""
    now = time.monotonic()
    srv = _make_srv(
        records={
            "sess-1": _record(
                now,
                started_at=now - 200.0,
                expires_at=now + 0.3,
                last_progress_ts=now - 200.0,
            )
        },
        markers={"sess-1": True},
    )

    async def _alive(*_args, **_kwargs):
        return (None, True)

    monkeypatch.setattr(helpers, "_query_prefill_progress", _alive)

    _, extended = await helpers._extend_lease_during_prefill(
        srv, "sess-1", llama_port=8080, slot_id=0, last_progress=0,
        prefill_in_flight=True,
    )
    assert extended is True
    record = srv.local_dispatch_records["sess-1"]
    assert record["last_progress_ts"] > now - 1.0, (
        "observed liveness IS progress evidence and must refresh the clock"
    )
    assert helpers._watchdog_no_progress(srv, "sess-1", record) is False


@pytest.mark.asyncio
async def test_numeric_progress_still_extends_and_refreshes(monkeypatch):
    """AC4 (no regression to LP-0MSE05J53004C6EL): a numeric progress advance
    extends the lease and refreshes the no-progress clock."""
    now = time.monotonic()
    srv = _make_srv(
        records={"sess-1": _record(now, last_progress_ts=now - 200.0)},
        markers={"sess-1": True},
    )

    async def _advancing(*_args, **_kwargs):
        return (5000, True)

    monkeypatch.setattr(helpers, "_query_prefill_progress", _advancing)

    last_progress, extended = await helpers._extend_lease_during_prefill(
        srv, "sess-1", llama_port=8080, slot_id=0, last_progress=0,
    )
    assert extended is True
    assert last_progress == 5000
    record = srv.local_dispatch_records["sess-1"]
    assert record["last_progress"] == 5000
    assert record["last_progress_ts"] > now - 1.0
    assert helpers._watchdog_no_progress(srv, "sess-1", record) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Client-dead mid-prefill release (LP-0MVLFVY6K003VYEL)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_client_disconnect_release_frees_every_hold(monkeypatch):
    """AC1/AC2: the client-disconnect release deletes the record, pops the
    marker, frees the slot assignment, logs a distinct reason and cancels the
    captured upstream stream."""
    now = time.monotonic()
    stream_cm = MagicMock()
    stream_cm.__aexit__ = AsyncMock(return_value=None)
    srv = _make_srv(
        records={"sess-dead": _record(now, stream_cm=stream_cm)},
        markers={"sess-dead": True},
    )
    wake = AsyncMock(return_value=None)
    monkeypatch.setattr("proxy.contention_queue.wake", wake)
    freed: list[tuple[str, object]] = []
    monkeypatch.setattr(
        "proxy.session._free_slot_assignment",
        lambda sid, endpoint=None: freed.append((sid, endpoint)),
    )

    released = helpers._release_client_disconnect_dispatch(srv, "sess-dead")

    assert released is True
    assert "sess-dead" not in srv.local_dispatch_records
    assert "sess-dead" not in srv.local_prefill_in_flight
    assert freed == [("sess-dead", None)], "the slot assignment must be freed"
    lines = _log_lines(srv.logger)
    assert any(
        "lease_released" in line
        and "reason=client_disconnect" in line
        and "sess-dead" in line
        for line in lines
    ), f"expected lease_released reason=client_disconnect, got: {lines}"

    # The upstream stream close and the contention wake are detached tasks
    # (the release must not await — AC3); give them one loop turn.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    stream_cm.__aexit__.assert_awaited()
    assert wake.await_count == 1, "the contention queue must be woken once"


@pytest.mark.asyncio
async def test_client_disconnect_release_is_scoped_and_idempotent():
    """AC1: only the dead client's session is released; a repeat call is a
    no-op that logs nothing (it races the deadline/watchdog/orphan paths)."""
    now = time.monotonic()
    srv = _make_srv(
        records={
            "sess-dead": _record(now),
            "sess-live": _record(now),
        },
        markers={"sess-dead": True, "sess-live": True},
    )
    live_before = dict(srv.local_dispatch_records["sess-live"])

    assert helpers._release_client_disconnect_dispatch(srv, "sess-dead") is True
    assert srv.local_dispatch_records["sess-live"] == live_before
    assert "sess-live" in srv.local_prefill_in_flight

    srv.logger.info.reset_mock()
    assert helpers._release_client_disconnect_dispatch(srv, "sess-dead") is False
    assert not any(
        "reason=client_disconnect" in line for line in _log_lines(srv.logger)
    ), "a second release must not re-log the event"


@pytest.mark.asyncio
async def test_client_disconnect_release_handles_endpoint_scoped_keys():
    """AC1: per-endpoint (tuple) records and an orphaned marker are released."""
    now = time.monotonic()
    endpoint = "http://127.0.0.1:8081"
    srv = _make_srv(
        records={(endpoint, "sess-dead"): _record(now)},
        markers={"sess-dead": True},
    )

    assert (
        helpers._release_client_disconnect_dispatch(
            srv, "sess-dead", endpoint=endpoint
        )
        is True
    )
    assert not srv.local_dispatch_records
    assert "sess-dead" not in srv.local_prefill_in_flight

    # Marker-only orphan (record already cleaned): still released.
    srv.local_prefill_in_flight["orphan"] = True
    assert helpers._release_client_disconnect_dispatch(srv, "orphan") is True
    assert "orphan" not in srv.local_prefill_in_flight


@pytest.mark.asyncio
async def test_client_disconnect_release_is_cancellation_proof():
    """AC3: a pending ``CancelledError`` cannot interrupt the release.

    The production hazard is a reaper cancellation landing inside the
    ``finally`` cleanup: an awaiting release (lock acquisition, stream
    ``__aexit__``) can be skipped by a second cancel. The release therefore
    contains no ``await`` — a cancellation requested *before* it runs is
    delivered only at the next suspension point, so the record and marker are
    already gone.
    """
    now = time.monotonic()
    srv = _make_srv(
        records={"sess-dead": _record(now)},
        markers={"sess-dead": True},
    )

    async def _release_with_pending_cancel():
        task = asyncio.current_task()
        assert task is not None
        task.cancel()  # delivered at the next await, i.e. after the release
        released = helpers._release_client_disconnect_dispatch(srv, "sess-dead")
        await asyncio.sleep(0)  # cancellation lands here
        return released

    task = asyncio.ensure_future(_release_with_pending_cancel())
    with pytest.raises(asyncio.CancelledError):
        await task

    assert "sess-dead" not in srv.local_dispatch_records, (
        "the record must be removed even when the task is being cancelled"
    )
    assert "sess-dead" not in srv.local_prefill_in_flight, (
        "the marker must be removed even when the task is being cancelled"
    )


@pytest.mark.asyncio
async def test_monitor_releases_on_confirmed_disconnect(monkeypatch):
    """AC1/AC2: the first-byte monitor polls ``is_disconnected()`` on its
    existing cadence, releases the dispatch and raises
    ``ClientDisconnectedDuringPrefillError`` — and cancels the pending upstream
    first-byte wait so no prefill is left running for a dead client."""
    now = time.monotonic()
    srv = _make_srv(
        records={"sess-dead": _record(now)},
        markers={"sess-dead": True},
    )
    start = time.monotonic()

    class _DeadClient:
        scope: dict = {}

        def __init__(self):
            self.calls = 0

        async def is_disconnected(self):
            self.calls += 1
            return True

    request = _DeadClient()
    upstream_cancelled = asyncio.Event()
    upstream_task: list[asyncio.Task] = []

    async def _never_first_byte():
        current = asyncio.current_task()
        assert current is not None
        upstream_task.append(current)
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            upstream_cancelled.set()
            raise

    with pytest.raises(helpers.ClientDisconnectedDuringPrefillError):
        await helpers._await_first_byte_with_prefill_monitor(
            srv,
            _never_first_byte(),
            timeout=300.0,
            poll_seconds=0.05,
            warn_seconds=0,
            session_id="sess-dead",
            llama_port=8080,
            slot_id=1,
            request=request,
            dispatch_session_id="sess-dead",
        )

    elapsed = time.monotonic() - start
    assert elapsed < 300.0, "AC1: the release must be well inside the 300 s bound"
    assert request.calls >= 1, "the monitor must probe the client connection"
    assert "sess-dead" not in srv.local_dispatch_records
    assert "sess-dead" not in srv.local_prefill_in_flight
    # ``cancel()`` only requests cancellation; it is delivered at the next
    # loop turn.
    for _ in range(5):
        await asyncio.sleep(0)
    assert upstream_task and upstream_task[0].cancelled(), (
        "the pending upstream first-byte wait must be cancelled"
    )
    assert upstream_cancelled.is_set()
    assert any(
        "reason=client_disconnect" in line for line in _log_lines(srv.logger)
    )


@pytest.mark.asyncio
async def test_monitor_keeps_a_slow_but_live_prefill(monkeypatch):
    """Constraint (no false positives): a live client whose prefill simply
    takes long is never released by the disconnect path — only the existing
    first-byte ceiling/timeout may abort it."""
    now = time.monotonic()
    srv = _make_srv(
        records={"sess-slow": _record(now)},
        markers={"sess-slow": True},
    )

    class _LiveSlowClient:
        scope: dict = {}

        async def is_disconnected(self):
            return False

    async def _unobservable(*_args, **_kwargs):
        return (None, False)

    monkeypatch.setattr(helpers, "_query_prefill_progress", _unobservable)

    async def _never_first_byte():
        await asyncio.sleep(60)

    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await helpers._await_first_byte_with_prefill_monitor(
            srv,
            _never_first_byte(),
            timeout=0.1,
            poll_seconds=0.05,
            warn_seconds=0,
            session_id="sess-slow",
            llama_port=8080,
            slot_id=1,
            request=_LiveSlowClient(),
            dispatch_session_id="sess-slow",
        )

    assert "sess-slow" in srv.local_dispatch_records, (
        "a live client's slow prefill must keep its lease"
    )
    assert "sess-slow" in srv.local_prefill_in_flight
    assert not any(
        "reason=client_disconnect" in line for line in _log_lines(srv.logger)
    )


CLIENT_DEAD_CONFIG = copy.deepcopy(SERVER_CONFIG)
CLIENT_DEAD_CONFIG["server"].update(
    {
        # Disable the deadline so the *disconnect* path is what releases the
        # dispatch (the deadline release is covered by the AC1 tests above).
        "local_prefill_deadline_seconds": 0,
        "local_dispatch_first_byte_timeout_seconds": 840,
        "local_dispatch_lease_prefill_poll_seconds": 0.05,
        "local_dispatch_lease_prefill_buffer_seconds": 0.5,
        "local_dispatch_prefill_observability_warn_seconds": 0,
    }
)


def _patch_router_server(monkeypatch, session_id: str) -> None:
    """Wire ``proxy.server`` (used by ``proxy_to_local``) to a clean state."""
    monkeypatch.setattr(server, "config", copy.deepcopy(CLIENT_DEAD_CONFIG))
    monkeypatch.setattr(server, "active_queries", 0)
    monkeypatch.setattr(server, "active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_active_queries", 0)
    monkeypatch.setattr(server, "local_active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_generating_queries", 0)
    monkeypatch.setattr(server, "local_generating_queries_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_generating_sessions", set())
    monkeypatch.setattr(server, "local_prefill_in_flight", {})
    monkeypatch.setattr(server, "local_prefill_in_flight_lock", asyncio.Lock())
    monkeypatch.setattr(server, "local_dispatch_records", {})
    monkeypatch.setattr(server, "local_dispatch_records_lock", asyncio.Lock())
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "llama_process", MagicMock(poll=lambda: None, pid=1))
    monkeypatch.setattr(server, "current_model", "test-model")
    monkeypatch.setattr(server, "session_manager", MagicMock())
    monkeypatch.setattr(server, "logger", MagicMock())
    helpers._reset_prefill_thrash_cooldowns()

    monkeypatch.setattr("proxy.router._is_self_healing_active", lambda: False)
    monkeypatch.setattr(
        "proxy.router._handle_session",
        AsyncMock(
            return_value={
                "session_id": session_id,
                "session_id_header": session_id,
                "session_explicit": True,
                "session_created": False,
                "is_delta_request": False,
                "session_fallback_reason": None,
                "delta_messages": [],
                "original_message_count": 1,
                "body_override": None,
                "body_json": None,
            }
        ),
    )
    monkeypatch.setattr(
        "proxy.router._build_slot_context", MagicMock(return_value=(None, None, 3.0))
    )
    monkeypatch.setattr(
        "proxy.router._check_slot_availability", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "proxy.router._restore_slot_snapshot", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        "proxy.router._call_with_empty_retry", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "proxy.router._update_session_and_slot", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "proxy.router_helpers._query_prefill_progress",
        AsyncMock(return_value=(None, False)),
    )

    async def _never_returns(*_args, **_kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr("proxy.router._call_with_backend_retries", _never_returns)


class _ClientDeadRequest:
    """Minimal Starlette-Request stand-in with a shared ASGI scope."""

    def __init__(self, session_id: str, disconnect: bool):
        self.headers = {"host": "localhost", "x-session-id": session_id}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()
        self.scope: dict = {"type": "http"}
        self._disconnect = disconnect
        self._body = (
            b'{"model":"test","messages":[{"role":"user","content":"hi"}],'
            b'"stream":true}'
        )

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return self._disconnect


@pytest.mark.asyncio
async def test_router_prefill_client_disconnect_releases_and_admits_another(monkeypatch):
    """AC1/AC2 (end-to-end): a client that dies mid-prefill releases its
    dispatch and marker, logs ``reason=client_disconnect``, and the next local
    request from another session is admitted immediately instead of being
    denied with ``reason=local_lease_active``."""
    from proxy.router import proxy_to_local

    session_id = "client-dead-session"
    _patch_router_server(monkeypatch, session_id)
    request = _ClientDeadRequest(session_id, disconnect=True)

    response = await proxy_to_local(request, "v1/chat/completions")

    assert response.status_code == 499, (
        "a client that vanished gets a 499 (its send is a no-op), not a "
        "provider detour or a backend cooldown"
    )
    assert session_id not in server.local_dispatch_records, (
        "the dead client's dispatch record must be released"
    )
    assert session_id not in server.local_prefill_in_flight, (
        "the dead client's prefill marker must be released"
    )
    assert server.local_active_queries == 0, (
        "the per-request local counter must be decremented exactly once"
    )
    lines = _log_lines(server.logger)
    assert any(
        "lease_released" in line and "reason=client_disconnect" in line
        for line in lines
    ), f"expected lease_released reason=client_disconnect, got: {lines[-10:]}"

    # AC1: another session is admitted afterwards, same tick, no waiting.
    acquired, owner, _active, _retry = await helpers._try_acquire_local_dispatch(
        server, 1, "other-session", "local"
    )
    assert acquired is True, (
        f"another session must be admitted after the client-dead release "
        f"(owner={owner})"
    )
    assert "other-session" in server.local_dispatch_records


@pytest.mark.asyncio
async def test_router_prefill_reaper_cancel_releases_on_confirmed_verdict(monkeypatch):
    """AC3 (end-to-end): when the disconnect reaper cancels the in-flight
    request task during prefill, its confirmed verdict releases the dispatch
    synchronously and the counter cleanup still runs — the cancelled task does
    not skip the release."""
    from proxy.disconnect_reaper import mark_client_disconnected
    from proxy.router import proxy_to_local

    session_id = "reaper-dead-session"
    _patch_router_server(monkeypatch, session_id)
    request = _ClientDeadRequest(session_id, disconnect=False)

    task = asyncio.ensure_future(proxy_to_local(request, "v1/chat/completions"))
    for _ in range(100):
        if session_id in server.local_dispatch_records:
            break
        await asyncio.sleep(0.01)
    assert session_id in server.local_dispatch_records, (
        "the dispatch must have acquired its lease before the reaper fires"
    )

    # The reaper marks the confirmed verdict on the ASGI scope, then cancels.
    mark_client_disconnected(request)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert session_id not in server.local_dispatch_records, (
        "a reaper cancellation during prefill must release the dispatch record"
    )
    assert session_id not in server.local_prefill_in_flight, (
        "a reaper cancellation during prefill must release the prefill marker"
    )

    # The counter cleanup is scheduled as a detached task (the cancelled task
    # cannot await it) — let it run.
    for _ in range(10):
        await asyncio.sleep(0)
    assert server.local_active_queries == 0, (
        "the counter cleanup must still run after the cancellation"
    )
    assert any(
        "reason=client_disconnect" in line for line in _log_lines(server.logger)
    )
