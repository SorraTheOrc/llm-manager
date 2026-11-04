"""Local dispatch-lease occupancy: stale records, denial evidence, divergence.

LP-0MW0KZTWC004HXQ5. On 2026-11-03 the proxy refused local dispatches while
llama-server reported free slots, and the refusal could not be diagnosed from
the log::

    08:49:31.690 local_dispatch_denied session=audit-RISE-... owner=herdr-... active=1
    08:49:31.034 /llama/local/status -> available_slots=2 (session_slot_pool_size=2)

Four contracts are pinned here:

- AC1 — the occupancy check prunes records that are inactive *and* expired
  before it counts them, and an ``active`` record past its expiry is live work
  (never pruned, never silently reclaimed).
- AC2 — a refusal logs the whole occupancy evidence: record keys with their
  active/expiry state, the active counter, ``max_local``, the generating and
  prefill-in-flight counts with the effective prefill cap, and llama-server's
  last known slot counts.
- AC3 — a refusal that the occupancy view cannot explain is logged as a
  ``local_occupancy_divergence`` warning rather than silently served.
- AC4 / AC5 — the 08:49:31 shape (stale record + one live holder on a two-slot
  pool) admits the next session and emits **no** ``local_dispatch_denied``,
  while a genuinely full pool reports ``available_slots == 0`` and the denial
  names the real owner.

Every test is hermetic: throwaway ``SimpleNamespace`` servers with fresh
``asyncio.Lock``s, or the ASGI app with the ``/slots`` query patched. No live
proxy, backend or shared file.
"""

from __future__ import annotations

import asyncio
import copy
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import proxy.router_helpers as helpers
import proxy.server as server
import pytest

SERVER_CONFIG = {
    "server": {
        "llama_router_mode": False,
        "llama_server_port": 8080,
        "max_concurrent_queries": 4,
        # The incident pool: two slots.
        "session_slot_pool_size": 2,
        "local_dispatch_lease_timeout_seconds": 180,
        "local_dispatch_lease_per_token_seconds": 0.0,
        "local_dispatch_lease_max_seconds": 1500,
        "local_dispatch_lease_prefill_poll_seconds": 0.05,
        "local_dispatch_lease_prefill_buffer_seconds": 0.5,
        "local_dispatch_no_progress_timeout_seconds": 180,
        "local_dispatch_max_prefill_seconds": 900,
        "local_prefill_deadline_seconds": 0,
        "local_dispatch_first_byte_timeout_seconds": 840,
        "local_dispatch_prefill_observability_warn_seconds": 0,
        "local_max_concurrent_prefills": 1,
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


def _record(*, active: bool, expires_in: float, started_ago: float = 0.0) -> dict:
    now = time.monotonic()
    return {
        "backend": "local",
        "started_at": now - started_ago,
        "active": active,
        "expires_at": now + expires_in,
        "model_name": "test-model",
        "last_progress": 0,
        "last_progress_ts": now,
    }


def _make_srv(records: dict | None = None, markers: dict | None = None) -> SimpleNamespace:
    """Fake server exercising the real (generating-aware) admission path."""
    return SimpleNamespace(
        config=copy.deepcopy(SERVER_CONFIG),
        local_active_queries=0,
        local_active_queries_lock=asyncio.Lock(),
        local_dispatch_records=records if records is not None else {},
        local_dispatch_records_lock=asyncio.Lock(),
        local_prefill_in_flight=markers if markers is not None else {},
        local_prefill_in_flight_lock=asyncio.Lock(),
        local_generating_queries=0,
        local_generating_queries_lock=asyncio.Lock(),
        local_generating_sessions=set(),
        logger=MagicMock(),
    )


def _log_lines(logger, level: str = "info") -> list[str]:
    """Render ``logger.<level>`` calls back into their formatted lines."""
    lines: list[str] = []
    for call in getattr(logger, level).call_args_list:
        template = str(call.args[0]) if call.args else ""
        args = call.args[1:]
        if args and "%" in template:
            try:
                lines.append(template % args)
                continue
            except (TypeError, ValueError):
                pass
        lines.append(template + (" " + " ".join(str(a) for a in args) if args else ""))
    return lines


# ======================================================================
# AC1: pruning stale records before counting
# ======================================================================


@pytest.mark.asyncio
async def test_prune_removes_only_inactive_expired_records():
    """AC1: a record is stale only when it is inactive *and* past expiry."""
    srv = _make_srv(
        records={
            "stale": _record(active=False, expires_in=-5.0),
            "cooldown": _record(active=False, expires_in=60.0),
            "live": _record(active=True, expires_in=60.0),
            # An active record past its expiry is live work whose lease lapsed
            # (the no-progress watchdog's concern): never reclaimed here.
            "lapsed-live": _record(active=True, expires_in=-5.0),
        }
    )

    removed = helpers._prune_stale_dispatch_records(srv, time.monotonic())

    assert removed == ["stale"], f"only the inactive expired record is stale: {removed}"
    assert set(srv.local_dispatch_records) == {"cooldown", "live", "lapsed-live"}


@pytest.mark.asyncio
async def test_stale_record_never_blocks_a_live_dispatch():
    """AC1/AC4: a stale record plus one live holder on a two-slot pool admits
    the next session, and the stale record is gone afterwards."""
    srv = _make_srv(
        records={
            "stale-session": _record(active=False, expires_in=-5.0),
            "owner-session": _record(active=True, expires_in=120.0),
        }
    )
    srv.local_generating_queries = 1  # the live holder is decoding
    srv.local_generating_sessions = {"owner-session"}

    acquired, owner, _active, _retry = await helpers._try_acquire_local_dispatch(
        srv, 2, "newcomer-session", "local"
    )

    assert acquired is True, f"stale records must not block a live dispatch (owner={owner})"
    assert "newcomer-session" in srv.local_dispatch_records
    assert "stale-session" not in srv.local_dispatch_records, (
        "the prune that runs before the occupancy count must drop the stale record"
    )


@pytest.mark.asyncio
async def test_legacy_mode_stale_record_does_not_count_towards_occupancy():
    """AC1 (legacy path): with no generating-only state the occupancy count
    ignores stale records, so the second of two slots stays admissible."""
    srv = _make_srv(
        records={
            "stale-session": _record(active=False, expires_in=-5.0),
            "owner-session": _record(active=True, expires_in=120.0),
        }
    )
    # Legacy single-server mode: no generating-only backing state.
    del srv.local_generating_queries
    del srv.local_generating_sessions
    del srv.local_generating_queries_lock

    acquired, owner, _active, _retry = await helpers._try_acquire_local_dispatch(
        srv, 2, "newcomer-session", "local"
    )

    assert acquired is True, f"a stale record must not occupy a slot (owner={owner})"


# ======================================================================
# AC2: the denial line carries the occupancy evidence
# ======================================================================


@pytest.mark.asyncio
async def test_occupancy_evidence_reports_every_gate_input():
    """AC2: the snapshot names each record's state plus the aggregate counters."""
    srv = _make_srv(
        records={
            "owner-session": _record(active=True, expires_in=120.0),
            "stale-session": _record(active=False, expires_in=-5.0),
        },
        markers={"owner-session": True},
    )
    srv.local_active_queries = 1
    srv.local_generating_queries = 2

    evidence = await helpers._local_dispatch_occupancy_evidence(srv, max_local=2)

    assert evidence["max_local"] == 2
    assert evidence["records_total"] == 2
    assert evidence["records_active"] == 1
    assert evidence["records_unexpired"] == 1
    assert evidence["records_stale"] == 1
    assert evidence["active_counter"] == 1
    assert evidence["generating"] == 2
    assert evidence["prefill_in_flight"] == 1
    assert evidence["prefill_cap"] == 1  # local_max_concurrent_prefills
    by_key = {r["key"]: r for r in evidence["records"]}
    assert by_key["owner-session"]["active"] is True
    assert by_key["owner-session"]["expired"] is False
    assert by_key["owner-session"]["expires_in_s"] > 0
    assert by_key["stale-session"]["active"] is False
    assert by_key["stale-session"]["expired"] is True


@pytest.mark.asyncio
async def test_occupancy_evidence_is_read_only():
    """AC2: taking the evidence never mutates the registry it inspects."""
    srv = _make_srv(records={"stale-session": _record(active=False, expires_in=-5.0)})

    await helpers._local_dispatch_occupancy_evidence(srv, max_local=2)

    assert "stale-session" in srv.local_dispatch_records, (
        "diagnostics must not prune: the prune belongs to the admission path"
    )


def test_formatted_evidence_carries_every_published_field():
    """AC2: the rendered suffix contains every field the contract names."""
    evidence = {field: 0 for field in helpers.OCCUPANCY_EVIDENCE_FIELDS}
    evidence["records"] = [
        {"key": "ep|sess-a", "active": True, "expired": False, "expires_in_s": 12.5},
        {"key": "sess-b", "active": False, "expired": True, "expires_in_s": 0.0},
    ]

    rendered = helpers._format_occupancy_evidence(evidence)

    for field in helpers.OCCUPANCY_EVIDENCE_FIELDS:
        assert f"{field}=" in rendered, f"missing {field} in: {rendered}"
    assert "records=ep|sess-a:active:expires_in=12.5,sess-b:idle:expired" in rendered


def test_formatted_evidence_marks_an_empty_registry():
    """AC2: an empty registry reads ``records=none``, never an empty suffix."""
    evidence = {field: 0 for field in helpers.OCCUPANCY_EVIDENCE_FIELDS}
    evidence["records"] = []

    assert helpers._format_occupancy_evidence(evidence).endswith("records=none")


# ======================================================================
# AC3: divergence verdicts
# ======================================================================


def test_divergence_is_not_inferred_from_missing_data():
    """AC3: an absent llama count is not a disagreement."""
    evidence = {field: None for field in helpers.OCCUPANCY_EVIDENCE_FIELDS}
    evidence["records"] = []
    evidence["max_local"] = 2
    evidence["prefill_cap"] = 1
    evidence["generating"] = 0
    evidence["prefill_in_flight"] = 0
    evidence["records_unexpired"] = 0

    # llama_available_slots unknown: the unexplained shape still applies...
    assert (
        helpers._occupancy_divergence_reason(evidence)
        == helpers.LOCAL_OCCUPANCY_DIVERGENCE_UNEXPLAINED
    )
    # ...but with no max_local at all there is nothing to call a divergence.
    evidence["max_local"] = None
    assert helpers._occupancy_divergence_reason(evidence) is None


def test_divergence_names_llama_free_slots_first():
    """AC3: llama reporting a free slot outranks the unexplained verdict."""
    evidence = {field: None for field in helpers.OCCUPANCY_EVIDENCE_FIELDS}
    evidence.update(
        {
            "records": [],
            "max_local": 2,
            "prefill_cap": 1,
            "generating": 2,  # a full generating pool: no unexplained shape
            "prefill_in_flight": 1,
            "records_unexpired": 2,
            "llama_available_slots": 2,
            "llama_total_slots": 2,
        }
    )

    assert (
        helpers._occupancy_divergence_reason(evidence)
        == helpers.LOCAL_OCCUPANCY_DIVERGENCE_LLAMA_FREE_SLOTS
    )


def test_a_fully_explained_refusal_is_not_a_divergence():
    """AC3: a genuinely full pool matches llama, so nothing is logged."""
    evidence = {field: None for field in helpers.OCCUPANCY_EVIDENCE_FIELDS}
    evidence.update(
        {
            "records": [],
            "max_local": 2,
            "prefill_cap": 1,
            "generating": 2,
            "prefill_in_flight": 1,
            "records_unexpired": 2,
            "llama_available_slots": 0,
            "llama_total_slots": 2,
        }
    )

    assert helpers._occupancy_divergence_reason(evidence) is None


# ======================================================================
# AC2/AC3/AC5 end-to-end: the router's denial line
# ======================================================================


class _Request:
    """Minimal Starlette-Request stand-in with a shared ASGI scope."""

    def __init__(self, session_id: str):
        self.headers = {"host": "localhost", "x-session-id": session_id}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()
        self.scope: dict = {"type": "http"}
        self._body = (
            b'{"model":"test","messages":[{"role":"user","content":"hi"}],'
            b'"stream":true}'
        )

    async def body(self):
        return self._body

    async def is_disconnected(self):
        return False


def _patch_router_server(monkeypatch, session_id: str) -> None:
    """Wire ``proxy.server`` (used by ``proxy_to_local``) to a clean state."""
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


async def _capture_admission_then_hang(monkeypatch, captured: dict) -> None:
    """Stand in for the upstream call: snapshot admission state, then block.

    Taking the snapshot *after* the lease was granted (and before the request
    is cancelled) makes the admission assertions independent of whatever the
    teardown path does with the record.
    """

    async def _wait(*_args, **_kwargs):
        captured["denied"] = [
            line
            for line in _log_lines(server.logger)
            if line.startswith("local_dispatch_denied")
        ]
        captured["records"] = dict(server.local_dispatch_records)
        await asyncio.sleep(30)

    monkeypatch.setattr("proxy.router._call_with_backend_retries", _wait)


async def _drive_denial(monkeypatch, *, generating: int, llama_counts) -> list[str]:
    """Drive ``proxy_to_local`` into a lease denial and return the log lines."""
    from proxy.router import proxy_to_local

    session_id = "audit-RISE-denied"
    _patch_router_server(monkeypatch, session_id)
    server.local_dispatch_records.update(
        {"owner-session": _record(active=True, expires_in=120.0)}
    )
    server.local_generating_queries = generating
    server.local_generating_sessions = {"owner-session"}
    server.local_prefill_in_flight = {"owner-session": True}

    with patch(
        "proxy.observability.last_known_slot_counts", return_value=llama_counts
    ):
        response = await proxy_to_local(_Request(session_id), "v1/chat/completions")

    assert response.status_code == 503, f"expected a lease denial, got {response.status_code}"
    return _log_lines(server.logger)


@pytest.mark.asyncio
async def test_denial_line_carries_occupancy_evidence_and_names_the_owner(monkeypatch):
    """AC2/AC3/AC5: the refusal is diagnosable from the log alone.

    The pool is full (two generating sessions on a two-slot pool) while
    llama-server's last known counts still say two slots are free — the
    2026-11-03 shape — so the refusal must name the real owner, carry the full
    occupancy evidence, and log the divergence.
    """
    lines = await _drive_denial(monkeypatch, generating=2, llama_counts=(2, 2))

    denied = [line for line in lines if line.startswith("local_dispatch_denied")]
    assert denied, f"expected a local_dispatch_denied line, got: {lines}"
    line = denied[0]
    assert "owner=owner-session" in line, f"the denial must name the real owner: {line}"
    for field in helpers.OCCUPANCY_EVIDENCE_FIELDS:
        assert f"{field}=" in line, f"missing {field} in: {line}"
    assert "records=owner-session:active:expires_in=" in line
    assert "max_local=2" in line
    assert "llama_available_slots=2" in line

    divergences = [l for l in _log_lines(server.logger, "info")
                   if l.startswith("local_occupancy_divergence")]
    assert divergences, "a refusal llama contradicts must be logged as a divergence"
    assert (
        f"reason={helpers.LOCAL_OCCUPANCY_DIVERGENCE_LLAMA_FREE_SLOTS}" in divergences[0]
    )
    assert "owner=owner-session" in divergences[0]
    assert not [
        l for l in _log_lines(server.logger, "warning")
        if l.startswith("local_occupancy_divergence")
    ], (
        "a llama-count disagreement is an expected capacity-view split, so it is "
        "logged at INFO beside the denial rather than into the WARNING stream"
    )


@pytest.mark.asyncio
async def test_full_pool_matching_llama_logs_no_divergence(monkeypatch):
    """AC3/AC5: a refusal llama agrees with is not a divergence."""
    lines = await _drive_denial(monkeypatch, generating=2, llama_counts=(0, 2))

    assert [l for l in lines if l.startswith("local_dispatch_denied")]
    assert "llama_available_slots=0" in [
        l for l in lines if l.startswith("local_dispatch_denied")
    ][0]
    assert not [
        l
        for l in _log_lines(server.logger, "info")
        if l.startswith("local_occupancy_divergence")
    ], "a full pool that llama agrees with is not a divergence"


@pytest.mark.asyncio
async def test_incident_pairing_admits_the_second_session(monkeypatch):
    """AC4: the 08:49:31 shape — one live holder plus a stale record on a
    two-slot pool — admits the next session, with no denial line at all."""
    from proxy.router import proxy_to_local

    session_id = "audit-RISE-admitted"
    _patch_router_server(monkeypatch, session_id)
    server.local_dispatch_records.update(
        {
            "stale-session": _record(active=False, expires_in=-30.0),
            "owner-session": _record(active=True, expires_in=120.0),
        }
    )
    server.local_generating_queries = 1
    server.local_generating_sessions = {"owner-session"}
    captured: dict = {}
    await _capture_admission_then_hang(monkeypatch, captured)

    # The dispatch is admitted and then blocks upstream; a short timeout is
    # enough — the assertions are about the admission decision, not the stream.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            proxy_to_local(_Request(session_id), "v1/chat/completions"), timeout=1.0
        )

    assert captured.get("records") is not None, (
        "the request must have reached the upstream call, i.e. been admitted"
    )
    assert not captured["denied"], (
        f"the second session must not be denied: {captured['denied']}"
    )
    assert session_id in captured["records"], (
        "the admitted session must hold a lease"
    )
    assert "stale-session" not in captured["records"], (
        "the admission path must prune the stale record before counting"
    )


# ======================================================================
# AC5: the status endpoint's engine-capacity view
# ======================================================================


class _FakeLock:
    def locked(self):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_all_slots_busy_reports_zero_available_slots():
    """AC5: when llama reports every slot processing, the endpoint says 0.

    ``available_slots``/``total_slots`` come from one source — llama-server's
    ``/slots`` — so a busy pool can never be reported as free capacity just
    because the lease registry happens to be empty.
    """
    import httpx

    from proxy.server import app

    async def fake_query():
        return {"llama_server_running": True}

    transport = httpx.ASGITransport(app=app)
    slots_mock = AsyncMock(return_value=(0, 2))
    detail_mock = AsyncMock(return_value=[])
    with patch("proxy.server.query_llama_status", side_effect=fake_query):
        with patch.object(server, "current_model", "test-model"):
            with patch.object(server, "model_switch_refcount", 0):
                with patch.object(server, "model_switch_lock", _FakeLock()):
                    with patch.object(server, "background_loads", {}):
                        with patch.object(server, "local_active_queries", 0):
                            with patch.object(server, "local_dispatch_records", {}):
                                with patch.object(
                                    server, "local_dispatch_records_lock", _FakeLock()
                                ):
                                    with patch.object(
                                        server,
                                        "config",
                                        {
                                            "server": {
                                                "session_slot_pool_size": 2,
                                                "llama_server_port": 8080,
                                            }
                                        },
                                    ):
                                        with patch(
                                            "proxy.observability._query_slots",
                                            slots_mock,
                                        ):
                                            with patch(
                                                "proxy.observability._query_slots_detail",
                                                detail_mock,
                                            ):
                                                # No fossil per-slot detail may
                                                # leak in from another module
                                                # and trigger the stale-detail
                                                # reconciliation.
                                                with patch(
                                                    "proxy.observability._last_slot_details_cache",
                                                    None,
                                                ):
                                                    async with httpx.AsyncClient(
                                                        transport=transport,
                                                        base_url="http://test",
                                                    ) as ac:
                                                        resp = await ac.get(
                                                            "/llama/local/status"
                                                        )

    assert resp.status_code == 200
    payload = resp.json()
    slots_mock.assert_awaited_once()
    assert payload["total_slots"] == 2
    assert payload["available_slots"] == 0, (
        "a pool with every slot processing must report zero available slots"
    )
