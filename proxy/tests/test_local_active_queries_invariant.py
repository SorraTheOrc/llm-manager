"""Exactly-once ``local_active_queries`` accounting under local contention.

Red-first concurrency invariant for parent LP-0MVWF78QK0060TZH **AC4**: the
in-flight local counter must equal the number of *active* dispatch records at
every observable step of a mixed-outcome local workload, so the self-healing
clamp (``_recover_stuck_local_active_queries``) never has anything to repair.

Why this test exists
--------------------

The counter is the proxy's in-flight local-request count. It is read by
``/llama/local/status`` (``available_slots`` / ``capacity_hint``), by
admission pricing and by the prefill saturation gate, so an over-count
reports a busy GPU that is idle and starves the herdr downtime dispatcher
(LP-0MVKSXR4P001V0CM). The measured recurrence — 76 recoveries in 13.2 h on
2026-10-31, then 95 recoveries in ~12.2 h post-v0.1.27 (≥1 leaked increment
per ~13 local streams) — is quantified on the parent; the release-path table
and the leaking-path verdicts are on the AC2/AC8 audit child
(LP-0MVYWK964006YDQ8).

The audit's primary mechanism is reproduced here: a streaming dispatch is
incremented (``_try_acquire_local_dispatch``) *before* the session
single-flight guard runs, so a duplicate of an in-flight prompt — a per-design
rejection, ``session_single_flight_max_queue_depth`` defaults to 1 — takes the
``_cleanup_after_request(decrement_local=False)`` detour and deletes the
dispatch record it just re-acquired. One increment is never decremented, and
the clamp resets it on the next tick. The buffered twin of that path does
decrement, which is why this test drives streaming dispatches.

Acceptance-criterion map
------------------------

- **AC1 — mixed-parallelism invariant.** ``TestMixedOutcomeInvariant`` runs
  three concurrent local streams with distinct outcomes (success, measured
  cache-deficit abort/eviction, client cancel) and asserts
  ``srv.local_active_queries == _count_active_local_dispatch_records(srv)`` at
  every observable step: admission, first byte, per chunk, the abort/eviction
  release, the client-cancel release and completion. The observation is taken
  under the production ``local_active_queries_lock`` → ``dispatch_records_lock``
  order (the order ``_try_acquire_local_dispatch`` uses), so the check itself
  cannot race a counter mutator.
- **AC2 — zero-clamp assertion.** Every checkpoint then runs the *real*
  self-healing clamp and asserts that the ``local_active_queries counter
  recovered`` WARNING never appears.
- **AC3 — RED on the current tree.** The duplicate-fan-out scenario below is
  the audited leak, so the invariant and the clamp assertions failed on ``dev``
  before the exactly-once fix. The assertions were marked
  ``xfail(strict=True)`` — the repository's red-first encoding (see
  ``test_abandoned_local_dispatch.py``) — so the test-only item could land with
  a green suite while the assertion stayed pinned. The implementation child
  (LP-0MVYWLBV70018ZZB) removed the markers as it turned them green: the
  single-flight duplicate now re-uses the in-flight lease's credit instead of
  taking a second one, so the counter and the active-record count stay equal at
  every step and the clamp has nothing to repair. ``strict`` meant a marker
  could not be forgotten, and the row now runs unmarked.
- **AC4 — hermetic and deterministic.** No network and no llama-server: the
  upstream is a per-stream fake (paced SSE chunks / a blocking prefill / a
  measured ``prompt_progress`` stream), the connection gates are stubbed, and
  the interleaving is driven by explicit barriers rather than sleeps. The
  invariant is only sampled at *quiescent* checkpoints (every live stream is
  parked awaiting its next upstream chunk, so no production task is
  mid-transition), which is what makes an equality assertion deterministic
  instead of a sampler race.

Harness idioms are reused from ``test_measured_cache_abort_wiring.py`` (the
``proxy_to_local`` integration harness and its b8782 stream fixtures) and
``test_lease_truthfulness.py`` (router-level dispatch accounting).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import proxy.router as router
import proxy.router_helpers as helpers
import proxy.server as server
import pytest
from fastapi.responses import StreamingResponse

from proxy import prefill_eta

#: The implementation child that turns the pinned assertions green
#: (LP-0MVYWL4A700211I0 covers the per-path release matrix; this one the
#: contention invariant). Named on the markers so the removal is attributable.
IMPLEMENTATION_ITEM = "LP-0MVYWLBV70018ZZB"

MODEL = "Qwen3"
PROMPT_TOKENS = 56_000
PRICED_NEW_TOKENS = 150

SESSION_SUCCESS = "sess-invariant-success"
SESSION_ABORT = "sess-invariant-abort"
SESSION_CANCEL = "sess-invariant-cancel"


def _shipped_server_section() -> dict:
    """The shipped ``server:`` section, so budgets are DERIVED not frozen."""
    import yaml

    config_path = Path(__file__).resolve().parents[2] / "proxy" / "config.yaml"
    return (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})["server"]


SERVER_CONFIG = {
    "server": {
        "llama_router_mode": False,
        "llama_server_port": 8080,
        # Four local streams run at once, so the global cap and the local slot
        # pool must both accommodate them; the prefill cap is what would
        # otherwise serialise (and falsely deny) the second admission.
        "max_concurrent_queries": 8,
        "session_slot_pool_size": 4,
        "local_max_concurrent_prefills": 4,
        "local_dispatch_lease_timeout_seconds": 30,
        "local_dispatch_lease_per_token_seconds": 0.0,
        "local_dispatch_lease_max_seconds": 1500,
        "local_dispatch_lease_prefill_poll_seconds": 0.05,
        "local_dispatch_lease_prefill_buffer_seconds": 0.5,
        "local_dispatch_no_progress_timeout_seconds": 180,
        "local_dispatch_max_prefill_seconds": 900,
        # Long enough that the 120 s deadline detour never fires inside this
        # test; the measured-deficit abort's benefit gate still prices against
        # it (see ``_shipped_server_section`` usage below).
        "local_prefill_deadline_seconds": _shipped_server_section()[
            "local_prefill_deadline_seconds"
        ],
        "local_admission_guard_budget_safety_fraction": 0.95,
        "local_dispatch_first_byte_timeout_seconds": 840,
        "local_dispatch_prefill_observability_warn_seconds": 0,
        # Queue mode: a duplicate of an in-flight prompt takes the 409 detour
        # (the audited leaking path), a different prompt would queue. Depth 1
        # is the shipped default and is what makes the fan-out a rejection
        # rather than a wait.
        "session_single_flight_mode": "queue",
        "session_single_flight_max_queue_depth": 1,
        "session_single_flight_queue_timeout_seconds": 120,
        "disconnect_cleanup_timeout": 1,
        # Generous idle/heartbeat bounds: the paced streams park between
        # chunks while the test samples, and a park must never be mistaken for
        # a stalled upstream (that would synthesise a termination).
        "stream_heartbeat_interval_seconds": 0.05,
        "stream_idle_timeout_seconds": 60,
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


# ---------------------------------------------------------------------------
# b8782 stream fixtures (shapes reused from test_measured_cache_abort_wiring)
# ---------------------------------------------------------------------------

_CONTENT_CHUNK = (
    b'data: {"choices":[{"delta":{"content":"hi"},"index":0}],'
    b'"created":0,"id":"chatcmpl-1","model":"Qwen3",'
    b'"object":"chat.completion.chunk"}\n\n'
)
_CONTENT_CHUNK_2 = (
    b'data: {"choices":[{"delta":{"content":" there"},"index":0}],'
    b'"created":0,"id":"chatcmpl-2","model":"Qwen3",'
    b'"object":"chat.completion.chunk"}\n\n'
)
_FINISH_CHUNK = (
    b'data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}],'
    b'"created":0,"id":"chatcmpl-finish","model":"Qwen3",'
    b'"object":"chat.completion.chunk"}\n\n'
)
_DONE = b"data: [DONE]\n\n"


def _fake_warm_progress() -> bytes:
    """The measured fake-warm signature: 56k prompt, 200-token reused prefix.

    Same shape the live F1 measurement verified (LP-0MVR7IN0O000DPXN): the
    deficit is ~99.6 % of the prompt, far over the priced ``new_tokens``, so
    the early-abort decision fires deterministically.
    """
    payload = {
        "choices": [
            {
                "finish_reason": None,
                "index": 0,
                "delta": {"role": "assistant", "content": None},
            }
        ],
        "created": 0,
        "id": "chatcmpl-progress",
        "model": MODEL,
        "system_fingerprint": "b8782-e97492369",
        "object": "chat.completion.chunk",
        "prompt_progress": {"total": PROMPT_TOKENS, "cache": 200, "processed": 0},
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


# ---------------------------------------------------------------------------
# Fake upstream (paced, blocking, and abort shapes)
# ---------------------------------------------------------------------------


class _CM:
    """Async context manager returning the wrapped fake response."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *_exc):
        return False


class _PacedUpstream:
    """A fake SSE upstream whose chunks are released by the test.

    ``parked`` is set whenever the consumer is waiting for the next chunk and
    cleared when one is handed over, so the test can await "everything this
    stream had is fully consumed and forwarded" without sleeping. That is what
    makes a checkpoint quiescent: every live stream is parked here, holding no
    proxy lock, so an equality assertion cannot race a transition.
    """

    status_code = 200
    headers = {"content-type": "text/event-stream"}

    def __init__(self, chunks: tuple[bytes, ...] = ()) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self.parked = asyncio.Event()
        self.entered = asyncio.Event()
        for chunk in chunks:
            self._queue.put_nowait(chunk)

    def cm(self) -> _CM:
        return _CM(self)

    def push(self, chunk: bytes) -> None:
        self._queue.put_nowait(chunk)

    def end(self) -> None:
        self._queue.put_nowait(None)

    async def aiter_bytes(self):
        self.entered.set()
        while True:
            self.parked.set()
            item = await self._queue.get()
            if item is None:
                return
            self.parked.clear()
            yield item

    async def aread(self) -> bytes:
        return b""


class _BlockingUpstream:
    """A fake upstream that never produces first-byte headers (a prefill).

    Models a local dispatch that is accepted and prefilling, which is exactly
    the state the pre-first-byte disconnect monitor has to release.
    """

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    def cm(self):
        return self

    async def __aenter__(self):
        self.entered.set()
        await asyncio.Event().wait()

    async def __aexit__(self, *_exc):
        return False


class _RoutingClient:
    """``httpx.AsyncClient`` stand-in routing each dispatch to its own plan.

    The router builds one client per request; the plan is selected from the
    upstream body, whose message content carries the stream's marker.
    """

    def __init__(self, plans: dict[str, object], sent: list[tuple[str, dict]]):
        self._plans = plans
        self._sent = sent

    def stream(self, _method, _url, headers=None, content=None):
        try:
            body = json.loads(content or b"{}")
        except Exception:  # pragma: no cover - defensive
            body = {}
        rendered = json.dumps(body)
        for marker, plan in self._plans.items():
            if marker in rendered:
                self._sent.append((marker, body))
                return plan.cm()
        raise AssertionError(
            "upstream request matched no plan: " + rendered[:200]
        )

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Request stand-in
# ---------------------------------------------------------------------------


class _Request:
    """Minimal FastAPI ``Request`` stand-in for a streaming local dispatch."""

    method = "POST"
    url = type("U", (), {"path": "/v1/chat/completions"})()

    def __init__(
        self,
        session_id: str,
        marker: str,
        *,
        admission_evidence: dict | None = None,
    ) -> None:
        self.headers = {"host": "localhost", "x-session-id": session_id}
        self.scope = {"type": "http"}
        self._session_id = session_id
        self._marker = marker
        self.client_gone = False
        # LP-0MVODJ3P6000AIUM: the admission-time pricing snapshot the abort
        # decision prices against (verbatim, never recomputed).
        self._local_admission_evidence = admission_evidence
        self._body = json.dumps(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": marker}],
                "stream": True,
            }
        ).encode()

    async def body(self) -> bytes:
        return self._body

    async def is_disconnected(self) -> bool:
        return self.client_gone


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _deterministic_prefill_eta():
    """Start every row from the uncalibrated shipped prefill-ETA fit.

    The measured-deficit benefit gate prices through the calibrated model and
    every served dispatch records its own first-byte sample
    (``record_dispatch_sample``). This scenario's streams serve in
    microseconds, so a prior row could poison the ``(Qwen3, ~56k)`` bucket and
    silently stop the abort rows aborting.
    """
    prefill_eta.reset()
    yield
    prefill_eta.reset()


def _install_server(monkeypatch) -> None:
    """Install the ``proxy.server`` state the real lease bookkeeping touches."""
    config = json.loads(json.dumps(SERVER_CONFIG))
    monkeypatch.setattr(server, "config", config)
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
    monkeypatch.setattr(server, "per_model_queries", {})
    monkeypatch.setattr(server, "per_model_queries_lock", asyncio.Lock())
    monkeypatch.setattr(server, "backend_ready", True)
    monkeypatch.setattr(server, "llama_process", MagicMock(poll=lambda: None, pid=1))
    monkeypatch.setattr(server, "current_model", MODEL)
    monkeypatch.setattr(server, "session_manager", MagicMock())
    monkeypatch.setattr(server, "logger", MagicMock())


def _patch_router(monkeypatch, plans: dict, sent: list) -> None:
    """Patch everything around the dispatch so the real router code runs."""

    async def _session(_srv, body_json, _config, headers):
        session_id = headers.get("x-session-id") or "sess-anon"
        return {
            "session_id": session_id,
            "session_id_header": session_id,
            "session_created": False,
            "is_delta_request": False,
            "session_fallback_reason": None,
            "delta_messages": [],
            "body_override": None,
            "body_json": body_json,
            "original_message_count": 1,
            "session_explicit": True,
        }

    async def _retry(factory, **_kwargs):
        return await factory()

    monkeypatch.setattr(router, "_is_self_healing_active", lambda: False)
    monkeypatch.setattr(router, "_check_slot_availability", AsyncMock(return_value=None))
    monkeypatch.setattr(router, "_handle_session", _session)
    monkeypatch.setattr(
        router, "_build_slot_context", MagicMock(return_value=(None, None, 3.0))
    )
    monkeypatch.setattr(
        router.httpx,
        "AsyncClient",
        lambda *_args, **_kwargs: _RoutingClient(plans, sent),
    )
    monkeypatch.setattr(router, "_schedule_token_increment", AsyncMock())
    monkeypatch.setattr(router, "_schedule_recv_token_increment", AsyncMock())
    monkeypatch.setattr(router, "_schedule_traffic_recording", MagicMock())
    monkeypatch.setattr(router, "count_text_tokens", MagicMock(return_value=5))
    monkeypatch.setattr(
        router, "_extract_delta_text_from_sse_chunk", MagicMock(return_value="")
    )
    monkeypatch.setattr(router, "evaluate_stream_guardrail", MagicMock(return_value=None))
    monkeypatch.setattr(router, "log_request", MagicMock())
    monkeypatch.setattr(router, "log_response_chunk", MagicMock())
    monkeypatch.setattr(router, "_call_with_backend_retries", _retry)
    monkeypatch.setattr(router, "_update_session_and_slot", AsyncMock())
    monkeypatch.setattr(router, "_restore_slot_snapshot", AsyncMock(return_value=False))
    monkeypatch.setattr(
        router, "_detect_restore_signal_from_llama_log", MagicMock(return_value=False)
    )
    monkeypatch.setattr(
        "proxy.session._resolve_log_path",
        MagicMock(return_value=Path("/nonexistent")),
    )
    # Hermetic: never probe a real llama-server for prefill progress (and never
    # extend a lease from a real /slots read).
    monkeypatch.setattr(
        helpers, "_query_prefill_progress", AsyncMock(return_value=(None, False))
    )
    monkeypatch.setattr(
        "proxy.provider._estimate_effective_prompt_tokens_for_routing",
        AsyncMock(return_value=PROMPT_TOKENS),
    )


async def _drive(request: _Request):
    """Run one streaming dispatch and drain its body if it is a stream."""
    result = await router.proxy_to_local(request, "v1/chat/completions")
    received = b""
    if isinstance(result, StreamingResponse):
        async for chunk in result.body_iterator:
            received += chunk
    return result, received


async def _wait(event: asyncio.Event, label: str, timeout: float = 10.0) -> None:
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
    except TimeoutError:  # pragma: no cover - harness failure, not a proxy bug
        raise AssertionError(f"barrier {label!r} never reached") from None


# ---------------------------------------------------------------------------
# The invariant observation
# ---------------------------------------------------------------------------


def _warning_lines(logger: MagicMock) -> list[str]:
    """Render ``logger.warning`` calls back into their formatted lines."""
    lines: list[str] = []
    for call in getattr(logger, "warning").call_args_list:
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


def _active_record_dump(srv) -> str:
    return repr(
        {
            str(key): {
                "active": rec.get("active"),
                "expires_at": rec.get("expires_at"),
                "session": rec.get("session"),
            }
            for key, rec in (getattr(srv, "local_dispatch_records", {}) or {}).items()
        }
    )


async def _snapshot(srv) -> tuple[int, int]:
    """Read the counter and the active-record count atomically.

    Acquires the locks in the order ``_try_acquire_local_dispatch`` uses
    (``local_active_queries_lock`` → ``local_dispatch_records_lock``), so the
    observation cannot interleave with any counter mutator — the assertion is
    not the race it is testing for. Only valid at a quiescent checkpoint.
    """
    async with srv.local_active_queries_lock:
        async with srv.local_dispatch_records_lock:
            counter = int(srv.local_active_queries)
            active = helpers._count_active_local_dispatch_records(srv)
    return counter, int(active or 0)


class _Observations:
    """Collected per-checkpoint observations plus the clamp detections."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, int, int]] = []
        self.clamped: list[tuple[str, list[str]]] = []

    def summary(self) -> str:
        return ", ".join(f"{label}:{counter}/{active}" for label, counter, active in self.rows)


async def _checkpoint(srv, observations: _Observations, label: str) -> None:
    """Assert the AC4 invariant at one quiescent step, then run the clamp."""
    counter, active = await _snapshot(srv)
    observations.rows.append((label, counter, active))
    assert counter == active, (
        f"[{label}] local_active_queries ({counter}) != active dispatch records "
        f"({active}) — the counter and its ground truth diverged "
        f"(records={_active_record_dump(srv)}); observed so far: "
        f"{observations.summary()}"
    )

    # AC2: the real self-healing clamp is the detector. At a quiescent step it
    # must find nothing to repair — a repair here is a leaked increment.
    await helpers._recover_stuck_local_active_queries(srv)
    recovered = [
        line for line in _warning_lines(srv.logger) if "counter recovered" in line
    ]
    if recovered:
        observations.clamped.append((label, recovered))
    assert not recovered, (
        f"[{label}] the self-healing clamp fired ({len(recovered)} warning(s)) — "
        f"a release path leaked an increment: {recovered[-3:]}"
    )


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


async def _run_mixed_scenario(
    monkeypatch,
    observations: _Observations,
    *,
    fan_out_duplicate: bool,
) -> dict:
    """Drive the mixed-outcome local workload and sample the invariant.

    Three concurrent local streams with distinct outcomes:

    * ``success`` — a paced SSE stream that runs to ``[DONE]``;
    * ``abort`` — a measured cache-deficit stream (fake-warm ``prompt_progress``)
      that is evicted early and detoured with a retryable 503;
    * ``cancel`` — a prefill that never produces a first byte, whose client
      disappears (the pre-first-byte disconnect release → 499).

    With *fan_out_duplicate* a fourth streaming dispatch repeats the success
    stream's prompt on the same session while it is still in flight: the
    single-flight guard rejects it as a duplicate (409) after the dispatch
    lease was already re-acquired — the audited leak
    (``router.py`` streaming duplicate detour, ``decrement_local=False``).
    """
    helpers._reset_prefill_thrash_cooldowns()
    _install_server(monkeypatch)

    success = _PacedUpstream()
    abort = _PacedUpstream([_fake_warm_progress(), _CONTENT_CHUNK, _FINISH_CHUNK, _DONE])
    cancel = _BlockingUpstream()
    plans = {
        "stream-success": success,
        "stream-abort": abort,
        "stream-cancel": cancel,
    }
    sent: list[tuple[str, dict]] = []
    _patch_router(monkeypatch, plans, sent)

    req_success = _Request(SESSION_SUCCESS, "stream-success")
    req_abort = _Request(
        SESSION_ABORT,
        "stream-abort",
        admission_evidence={
            "estimated_tokens": PROMPT_TOKENS,
            "cached_ratio": 0.0,
            "new_tokens": PRICED_NEW_TOKENS,
        },
    )
    req_cancel = _Request(SESSION_CANCEL, "stream-cancel")

    task_success = asyncio.create_task(_drive(req_success))
    task_abort = asyncio.create_task(_drive(req_abort))
    task_cancel = asyncio.create_task(_drive(req_cancel))

    # ── Admission: the success stream is peeking its first chunk, the cancel
    # stream is awaiting first-byte headers, the abort stream has already been
    # evicted by its measured deficit. Every live stream is parked.
    await _wait(success.parked, "success first-byte park")
    await _wait(cancel.entered, "cancel prefill entered")
    abort_result, _abort_received = await asyncio.wait_for(task_abort, timeout=10)
    assert abort_result.status_code == 503, (
        "the measured cache-deficit stream must be evicted early with a "
        f"retryable 503, got {abort_result.status_code}"
    )

    duplicate = None
    if fan_out_duplicate:
        # Same session, same messages ⇒ duplicate-in-flight (409). Dispatched
        # while the original holds the single-flight lock, after every stream
        # has been admitted.
        req_duplicate = _Request(SESSION_SUCCESS, "stream-success")
        duplicate = await asyncio.wait_for(_drive(req_duplicate), timeout=10)
        assert duplicate[0].status_code == 409, (
            "the fan-out must take the duplicate-in-flight detour (409), the "
            f"audited leaking path, got {duplicate[0].status_code}"
        )

    await _checkpoint(server, observations, "admission")

    # ── First byte: the success stream's first client-visible chunk is
    # forwarded (and the probe reveals no deficit, so it keeps streaming).
    success.push(_CONTENT_CHUNK)
    success.parked.clear()
    await _wait(success.parked, "success first chunk consumed")
    await _checkpoint(server, observations, "first_byte")

    # ── Per chunk: a second chunk on the same in-flight stream.
    success.push(_CONTENT_CHUNK_2)
    success.parked.clear()
    await _wait(success.parked, "success second chunk consumed")
    await _checkpoint(server, observations, "per_chunk")

    # ── Client cancel: the prefill's client disappears; the monitor confirms
    # the disconnect and releases the dispatch (499 to a client that is gone).
    req_cancel.client_gone = True
    cancel_result, _cancel_received = await asyncio.wait_for(task_cancel, timeout=10)
    assert cancel_result.status_code == 499, (
        "a confirmed pre-first-byte client disconnect must release the "
        f"dispatch and end with 499, got {cancel_result.status_code}"
    )
    await asyncio.sleep(0)
    await _checkpoint(server, observations, "client_cancel")

    # ── Completion: the success stream finishes normally.
    success.push(_FINISH_CHUNK)
    success.push(_DONE)
    success.end()
    success_result, success_received = await asyncio.wait_for(task_success, timeout=10)
    assert success_result.status_code == 200
    assert _CONTENT_CHUNK in success_received, (
        "the success stream's client-visible bytes must be forwarded; got "
        f"{success_received[:120]!r}"
    )
    await _checkpoint(server, observations, "completion")

    # ── Settled: nothing is left in flight, so the counter must be zero and
    # the dispatch registry empty (a record left behind is a lease leak).
    await asyncio.sleep(0)
    await _checkpoint(server, observations, "settled")
    assert server.local_active_queries == 0, (
        "local_active_queries must return to zero once every stream settled; "
        f"counter={server.local_active_queries}, "
        f"records={_active_record_dump(server)}"
    )
    assert not [
        rec
        for rec in server.local_dispatch_records.values()
        if isinstance(rec, dict) and rec.get("active")
    ], f"no active dispatch record may outlive the workload: {_active_record_dump(server)}"

    return {
        "sent_markers": [marker for marker, _body in sent],
        "duplicate": duplicate,
        "observations": observations,
    }


# ---------------------------------------------------------------------------
# AC1 / AC2 / AC4 — control: the three pure mixed-outcome paths are exact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_three_mixed_streams_keep_counter_equal_to_active_records(monkeypatch):
    """Control: success + deficit-abort + client-cancel are each exactly-once.

    This row must pass on today's code and after the fix. It proves the
    harness reaches every outcome the AC names (so the red row below is
    failing on the leak, not on a broken fixture) and that the clamp has
    nothing to repair for the paths that are already correct.
    """
    observations = _Observations()
    report = await _run_mixed_scenario(
        monkeypatch, observations, fan_out_duplicate=False
    )

    assert [label for label, _c, _a in observations.rows] == [
        "admission",
        "first_byte",
        "per_chunk",
        "client_cancel",
        "completion",
        "settled",
    ], f"every observable step must be sampled; got {observations.summary()}"
    assert not observations.clamped, (
        "the clamp must never fire on correctly-accounted release paths"
    )
    # The three outcome paths really were exercised (not a fixture no-op).
    assert set(report["sent_markers"]) == {
        "stream-success",
        "stream-abort",
        "stream-cancel",
    }


# ---------------------------------------------------------------------------
# AC1 / AC2 / AC3 — the audited duplicate fan-out no longer leaks an increment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_fan_out_keeps_counter_equal_to_active_records(monkeypatch):
    """AC1/AC3: mixed-outcome concurrency *plus* a rejected duplicate stream.

    A client that fans out the same prompt on one session while the original
    is prefillng is the audited leak: the counter and the active dispatch
    records diverge at admission and stay divergent, and the clamp repairs the
    difference on its next tick. The invariant must hold with that stream in
    the mix, at every step.
    """
    observations = _Observations()
    report = await _run_mixed_scenario(monkeypatch, observations, fan_out_duplicate=True)

    assert report["duplicate"][0].status_code == 409
    assert observations.rows[-1] == ("settled", 0, 0), (
        "after the workload settles the counter and the record set must both be "
        f"empty; got {observations.summary()}"
    )
    assert not observations.clamped


@pytest.mark.asyncio
async def test_duplicate_fan_out_never_triggers_the_recovery_clamp(monkeypatch):
    """AC2/AC3: the clamp's WARNING must never appear for a contended workload.

    Runs the same mixed workload with the duplicate fan-out and asserts that
    the *real* ``_recover_stuck_local_active_queries`` finds nothing to repair
    at any checkpoint — i.e. zero ``local_active_queries counter recovered``
    lines across the whole run.
    """
    observations = _Observations()
    await _run_mixed_scenario(monkeypatch, observations, fan_out_duplicate=True)

    recovered = [
        line for line in _warning_lines(server.logger) if "counter recovered" in line
    ]
    assert not recovered, (
        "the recovery clamp fired, so a release path leaked an increment: "
        f"{recovered}"
    )
