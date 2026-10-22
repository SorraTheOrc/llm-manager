"""Regression tests for the ``/events`` SSE first-frame bound.

Context (LP-0MVJML600003CQ6Y, AC4)
----------------------------------
2026-10-22: an operator reported "the proxy Web UI is not responding" while
every non-SSE endpoint answered in milliseconds. A *fresh* ``GET /events``
worked, but the operator's already-open tab received **nothing at all** — not
even the 30 s keepalive — while the TCP connection stayed up, so the browser
``EventSource`` never reconnected and never surfaced an error.

``proxy.ui.status_events()`` assembles its initial payload **before** entering
the keepalive loop::

    llama_status = await srv.query_llama_status()      # unbounded
    ... router_list_models(), _get_per_model_queries(), _query_slots_detail()
    yield f"data: {initial_status}\\n\\n"
    while True:                                        # keepalive loop
        ...

so a single stalled backend call means the generator yields **zero bytes
forever**. The invariant these tests pin: an SSE client must always receive a
frame — a ``data:`` status payload *or* a ``: keepalive`` comment — within a
bounded time, and the stream must then keep producing frames.

Test deadlines
--------------
``_FIRST_FRAME_DEADLINE`` (10 s) is deliberately **shorter than the 30 s
keepalive interval** the loop uses, so a passing assertion proves the frame is
a real bounded emission rather than an ordinary keepalive; and it leaves 2x
headroom over the 5 s budget candidate recorded on the fix item
LP-0MVK1YDW900757Q9 (which reuses the existing ``/slots`` 5 s detail timeout as
the single worst case).

TDD history
-----------
The two liveness assertions were marked ``xfail(strict=True)`` while ``dev``
stayed green during the RED phase; the verified pre-fix RED output is recorded
on LP-0MVK1YD81004DPYZ. The markers were removed by the fix that bounded the
initial payload (LP-0MVK1YDW900757Q9), so every test here is now an enforced
assertion — a regression re-opens the original defect loudly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Test budgets ────────────────────────────────────────────────────────────
# AC1: a stalled backend must still produce a first frame inside this bound.
_FIRST_FRAME_DEADLINE = 10.0
# AC2: after the first frame the generator must resume and serve the queue.
_FOLLOW_UP_FRAME_DEADLINE = 3.0
# AC3: the prompt path must not pay the (slow-path) budget.
_PROMPT_FRAME_DEADLINE = 3.0

# "far beyond the payload budget": longer than both the fix candidate (5 s) and
# the in-test deadline, so only a *bounded* implementation can pass.
_STALL_SECONDS = 60.0

_FIX_ITEM = "LP-0MVK1YDW900757Q9"

# Keys the prompt-path payload must keep (LP-0MVF7KQED005TG9Z added ``quota``;
# LP-0MVECWYWK000U3E1 added ``slots``). AC3: bounding the payload must not
# change the shape a healthy backend produces.
_EXPECTED_PAYLOAD_KEYS = frozenset(
    {
        "type",
        "current_model",
        "loaded_models",
        "llama_server_running",
        "n_ctx",
        "kv_cache_tokens",
        "total_sent",
        "total_recv",
        "per_model_queries",
        "slots",
        "quota",
    }
)

_PROMPT_STATUS = {
    "llama_server_running": False,
    "n_ctx": 32768,
    "kv_cache_tokens": 1024,
    "router_mode": False,
}

_BROADCAST_MARKER = "broadcast-after-stalled-first-frame"

# The disconnect test waits this long for the generator to park on the stalled
# backend; it is a liveness guard, not a timing assertion.
_STARTED_DEADLINE = 2.0


def _mock_srv() -> MagicMock:
    """A MagicMock server matching ``test_slot_sse.py``'s ``_srv`` pattern.

    ``llama_server_running=False`` keeps ``_query_slots_detail()`` out of the
    path so the tests observe the payload-assembly await only, and
    ``sse_clients`` is a **real** set because the endpoint registers the
    client queue on it before the generator starts.
    """
    srv = MagicMock()
    srv.per_model_queries = {}
    srv.per_model_queries_lock = asyncio.Lock()
    srv.sse_clients = set()
    srv.current_model = "Qwen3"
    srv.token_counts = {"total_sent": 100, "total_recv": 200}
    srv.config = {"server": {}}
    srv.query_llama_status = AsyncMock(return_value=dict(_PROMPT_STATUS))
    srv.router_list_models = AsyncMock(return_value=[])
    return srv


def _stalling_srv(stall_seconds: float = _STALL_SECONDS) -> MagicMock:
    """A server whose ``query_llama_status()`` stalls far past the budget.

    Models the real trigger: llama-server answers ``/slots`` in ~23 s while the
    local model generates, and ``query_llama_status()`` fans out to it.
    """
    srv = _mock_srv()

    async def _stall() -> dict:
        await asyncio.sleep(stall_seconds)
        return dict(_PROMPT_STATUS)

    srv.query_llama_status = AsyncMock(side_effect=_stall)
    return srv


async def _open_stream(srv: MagicMock):
    """Open ``/events`` and return its ``StreamingResponse``."""
    from proxy.ui import status_events

    with patch("proxy.ui._srv", return_value=srv):
        return await status_events()


async def _read_frame(iterator, timeout: float) -> str:
    chunk = await asyncio.wait_for(iterator.__anext__(), timeout=timeout)
    return chunk if isinstance(chunk, str) else chunk.decode("utf-8")


async def _first_frame(iterator, deadline: float) -> str:
    """Read the first frame, converting a hang into a descriptive failure."""
    try:
        return await _read_frame(iterator, deadline)
    except TimeoutError:
        raise AssertionError(
            f"/events produced no first frame within {deadline:.1f}s; a stalled "
            "backend must still yield a 'data:' status payload or a "
            "': keepalive' comment so the browser EventSource stays alive "
            f"({_FIX_ITEM})"
        ) from None


async def _close(iterator) -> None:
    """Close the SSE generator so its ``finally`` discards the client queue."""
    with contextlib.suppress(BaseException):
        await iterator.aclose()


def _assert_liveness_frame(raw: str) -> str:
    """Assert *raw* is a valid first frame; return ``payload``/``keepalive``."""
    if raw.startswith(":"):
        assert raw.strip() == ": keepalive", f"unexpected SSE comment frame: {raw!r}"
        return "keepalive"
    assert raw.startswith("data:"), (
        f"first SSE frame is neither a 'data:' payload nor a ': keepalive' "
        f"comment: {raw!r}"
    )
    payload = json.loads(raw[len("data:"):].strip())
    assert payload.get("type") == "status", (
        f"degraded first frame must still carry the status contract "
        f"(type == 'status'): {payload!r}"
    )
    return "payload"


@pytest.mark.asyncio
async def test_first_frame_arrives_when_status_query_stalls():
    """AC1: a stalled ``query_llama_status()`` cannot leave the stream silent.

    The backend call sleeps far beyond both the payload budget and the in-test
    deadline, so only a bounded initial payload can produce a frame here.
    """
    srv = _stalling_srv()
    response = await _open_stream(srv)
    iterator = response.body_iterator
    try:
        raw = await _first_frame(iterator, _FIRST_FRAME_DEADLINE)
        assert _assert_liveness_frame(raw) in {"payload", "keepalive"}
    finally:
        await _close(iterator)


@pytest.mark.asyncio
async def test_stream_keeps_producing_frames_after_a_stalled_first_frame():
    """AC2: the generator resumes into its broadcast loop after the first frame.

    A message queued on the client's own ``sse_clients`` queue (the path
    ``_periodic_broadcast_loop()`` uses to push updates) must reach the stream
    inside a bounded time — i.e. the generator is not wedged on the stalled
    payload assembly.
    """
    srv = _stalling_srv()
    response = await _open_stream(srv)
    iterator = response.body_iterator
    try:
        await _first_frame(iterator, _FIRST_FRAME_DEADLINE)

        # The endpoint registers the client queue on ``sse_clients`` before the
        # generator body runs, so it is available without awaiting a frame.
        queues = list(srv.sse_clients)
        assert len(queues) == 1, (
            f"expected exactly one registered SSE client queue, got {queues!r}"
        )
        queues[0].put_nowait(f'data: {{"type": "status", "marker": "{_BROADCAST_MARKER}"}}\n\n')

        loop = asyncio.get_running_loop()
        end = loop.time() + _FOLLOW_UP_FRAME_DEADLINE
        frames: list[str] = []
        while True:
            remaining = end - loop.time()
            if remaining <= 0:
                raise AssertionError(
                    "the SSE generator produced no frame carrying the queued "
                    f"broadcast within {_FOLLOW_UP_FRAME_DEADLINE:.1f}s of the "
                    f"first frame; frames seen: {frames!r}"
                )
            try:
                frame = await _read_frame(iterator, remaining)
            except TimeoutError:
                raise AssertionError(
                    "the SSE generator stalled after its first frame; no second "
                    f"frame within {_FOLLOW_UP_FRAME_DEADLINE:.1f}s "
                    f"(frames seen: {frames!r})"
                ) from None
            frames.append(frame)
            _assert_liveness_frame(frame)
            if _BROADCAST_MARKER in frame:
                break
    finally:
        await _close(iterator)


@pytest.mark.asyncio
async def test_prompt_path_payload_keeps_its_full_contract():
    """AC3: a prompt backend keeps the unchanged payload, without the budget.

    Guards both halves of AC3: the payload shape is untouched (every
    pre-existing key, including ``slots`` and ``quota``) *and* the happy path
    does not pay the slow-path budget — a frame that only arrives after the
    bounded wait would fail ``_PROMPT_FRAME_DEADLINE``.
    """
    srv = _mock_srv()
    response = await _open_stream(srv)
    iterator = response.body_iterator
    try:
        raw = await _first_frame(iterator, _PROMPT_FRAME_DEADLINE)
        assert _assert_liveness_frame(raw) == "payload"

        payload = json.loads(raw[len("data:"):].strip())
        missing = _EXPECTED_PAYLOAD_KEYS - set(payload)
        assert not missing, f"prompt-path payload lost keys: {sorted(missing)}"
        assert payload["current_model"] == "Qwen3"
        assert payload["llama_server_running"] is False
        assert payload["n_ctx"] == 32768
        assert payload["kv_cache_tokens"] == 1024
        assert payload["total_sent"] == 100
        assert payload["total_recv"] == 200
        assert payload["per_model_queries"] == {}
        assert payload["slots"] == []
        # Quota is absent (nothing polled) → the documented null degradation.
        assert payload["quota"] is None
    finally:
        await _close(iterator)


@pytest.mark.asyncio
async def test_client_disconnect_while_initial_payload_stalls_releases_queue():
    """AC4: a disconnect during the bounded initial query must not leak a queue.

    Starlette's ``StreamingResponse`` cancels the streaming task when the peer
    goes away, which raises ``CancelledError`` at the generator's await point.
    The endpoint's ``finally`` must still discard the client queue, otherwise a
    client that disconnects while the initial payload is slow leaves a dead
    queue registered and every later broadcast pays for it.
    """
    started = asyncio.Event()

    async def _stall() -> dict:
        started.set()
        await asyncio.sleep(_STALL_SECONDS)
        return dict(_PROMPT_STATUS)

    srv = _mock_srv()
    srv.query_llama_status = AsyncMock(side_effect=_stall)
    response = await _open_stream(srv)
    iterator = response.body_iterator
    assert len(srv.sse_clients) == 1, (
        "status_events() must register the client queue before streaming"
    )

    pending = asyncio.ensure_future(iterator.__anext__())
    try:
        await asyncio.wait_for(started.wait(), timeout=_STARTED_DEADLINE)
        assert not pending.done(), (
            "the generator must still be inside the bounded initial payload"
        )

        # Peer disconnect: cancel the pending read, which cancels the
        # generator at its await point (the ASGI disconnect path).
        pending.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pending

        assert srv.sse_clients == set(), (
            "a client disconnect during the initial payload must discard the "
            f"client queue; leaked: {srv.sse_clients!r}"
        )
    finally:
        if not pending.done():
            pending.cancel()
            with contextlib.suppress(BaseException):
                await pending
        await _close(iterator)


@pytest.mark.asyncio
async def test_queue_released_after_a_timed_out_initial_payload():
    """AC4: the degraded (timed-out) path keeps no queue after a disconnect.

    A timed-out payload must leave the client *connected* — the queue stays
    registered while the stream is live — and must still release it on close,
    so a timeout never strands a queue on ``sse_clients``.
    """
    srv = _stalling_srv()
    response = await _open_stream(srv)
    iterator = response.body_iterator
    try:
        raw = await _first_frame(iterator, _FIRST_FRAME_DEADLINE)
        _assert_liveness_frame(raw)
        assert len(srv.sse_clients) == 1, (
            "a degraded stream must keep its client queue while connected"
        )
    finally:
        await _close(iterator)

    assert srv.sse_clients == set(), (
        "closing a degraded stream must discard the client queue; "
        f"leaked: {srv.sse_clients!r}"
    )
