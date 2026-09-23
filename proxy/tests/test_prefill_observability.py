"""
Hermetic tests for prefill-progress observability during the initial
upstream response wait (LP-0MUCEFC0T009V7ZY).

Context
-------
``stream_cm.__aenter__()`` blocks until llama-server returns the SSE
response headers, which for this build arrive with the first token. The
post-headers prefill-progress poll in the streaming loop therefore never
runs during a real prefill: 0 ``lease_extended_during_prefill`` events and
0 ``prefill_progress_unobservable`` warnings were emitted even for
15-minute first-byte waits.

``_await_first_byte_with_prefill_monitor`` races the first-byte wait against
a periodic prefill-progress poll so that:

- Progress observed before the first byte extends the dispatch lease.
- A ``prefill_wait_exceeds_threshold`` warning fires once the initial wait
  exceeds a threshold (making the pre-header wait visible in production).
- The total wait is bounded (``local_dispatch_max_prefill_seconds``) and
  raises ``asyncio.TimeoutError`` when exceeded, with the pending upstream
  task cancelled.

These tests verify hermetically (no live llama-server).
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _make_srv() -> SimpleNamespace:
    """Build a fake server object with the attributes the helper touches."""
    return SimpleNamespace(
        config={"server": {}},
        logger=MagicMock(),
    )


async def _slow_open(delay: float, result=("cm", "resp")):
    """Coroutine that resolves to *result* after *delay* seconds."""
    await asyncio.sleep(delay)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Progress observability before first byte
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_progress_polled_before_first_byte(monkeypatch):
    """AC1: progress/liveness is observed before first byte during a wait
    longer than the poll cadence."""
    from proxy.router_helpers import _await_first_byte_with_prefill_monitor

    srv = _make_srv()
    poll_calls = []

    async def fake_extend(srv_arg, session_key, **kwargs):
        poll_calls.append(kwargs.get("last_progress"))
        return (100 * len(poll_calls), True)

    monkeypatch.setattr(
        "proxy.router_helpers._extend_lease_during_prefill", fake_extend
    )

    result = await _await_first_byte_with_prefill_monitor(
        srv,
        _slow_open(0.2),
        timeout=5.0,
        poll_seconds=0.05,
        warn_seconds=0.0,
        session_id="sess-1",
        llama_port=8080,
    )

    assert result == ("cm", "resp")
    assert len(poll_calls) >= 3, (
        f"Expected multiple prefill polls during a 0.2s wait at 0.05s "
        f"cadence, got {len(poll_calls)}"
    )


@pytest.mark.asyncio
async def test_prefill_wait_exceeds_threshold_warning(monkeypatch):
    """AC2: an observable warning is emitted once the initial wait exceeds
    the configured threshold (previously this path was completely silent)."""
    from proxy.router_helpers import _await_first_byte_with_prefill_monitor

    srv = _make_srv()

    async def fake_extend(srv_arg, session_key, **kwargs):
        return (0, False)

    monkeypatch.setattr(
        "proxy.router_helpers._extend_lease_during_prefill", fake_extend
    )

    await _await_first_byte_with_prefill_monitor(
        srv,
        _slow_open(0.2),
        timeout=5.0,
        poll_seconds=0.05,
        warn_seconds=0.1,
        session_id="sess-1",
        llama_port=8080,
    )

    warning_lines = [str(call) for call in srv.logger.warning.call_args_list]
    assert any("prefill_wait_exceeds_threshold" in line for line in warning_lines), (
        f"Expected prefill_wait_exceeds_threshold warning, got: {warning_lines}"
    )
    # Warning is emitted at most once.
    threshold_warnings = [
        line for line in warning_lines if "prefill_wait_exceeds_threshold" in line
    ]
    assert len(threshold_warnings) == 1, (
        f"Threshold warning must fire at most once, got {len(threshold_warnings)}"
    )


@pytest.mark.asyncio
async def test_no_warning_before_threshold(monkeypatch):
    """A fast first byte never emits the threshold warning."""
    from proxy.router_helpers import _await_first_byte_with_prefill_monitor

    srv = _make_srv()

    await _await_first_byte_with_prefill_monitor(
        srv,
        _slow_open(0.01),
        timeout=5.0,
        poll_seconds=0.05,
        warn_seconds=30.0,
        session_id="sess-1",
        llama_port=8080,
    )

    warning_lines = [str(call) for call in srv.logger.warning.call_args_list]
    assert not any("prefill_wait_exceeds_threshold" in line for line in warning_lines), (
        "No threshold warning should fire for an immediate first byte"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Bounded wait + cancellation
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_timeout_raises_and_cancels_open_task(monkeypatch):
    """AC: exceeding the ceiling raises asyncio.TimeoutError and cancels the
    pending upstream open task (no leaked stream)."""
    from proxy.router_helpers import _await_first_byte_with_prefill_monitor

    srv = _make_srv()
    cancelled = asyncio.Event()

    async def _never_returns():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(asyncio.TimeoutError):
        await _await_first_byte_with_prefill_monitor(
            srv,
            _never_returns(),
            timeout=0.1,
            poll_seconds=0.02,
            warn_seconds=0.0,
            session_id=None,
        )

    # Give the cancellation a chance to propagate.
    for _ in range(20):
        if cancelled.is_set():
            break
        await asyncio.sleep(0.01)
    assert cancelled.is_set(), "Pending upstream open task must be cancelled on timeout"


@pytest.mark.asyncio
async def test_open_exception_propagates(monkeypatch):
    """An exception from the first-byte coroutine propagates unchanged."""
    from proxy.router_helpers import _await_first_byte_with_prefill_monitor

    srv = _make_srv()

    async def _fails():
        await asyncio.sleep(0.01)
        raise ValueError("backend connection error")

    with pytest.raises(ValueError, match="backend connection error"):
        await _await_first_byte_with_prefill_monitor(
            srv,
            _fails(),
            timeout=5.0,
            poll_seconds=0.05,
            warn_seconds=0.0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# No-regression branches
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_direct_await_when_monitoring_disabled():
    """No polling and no timeout: the coroutine is awaited directly."""
    from proxy.router_helpers import _await_first_byte_with_prefill_monitor

    srv = _make_srv()
    result = await _await_first_byte_with_prefill_monitor(
        srv,
        _slow_open(0.01),
        timeout=0.0,
        poll_seconds=0.0,
        warn_seconds=30.0,
        session_id="sess-1",
    )
    assert result == ("cm", "resp")


@pytest.mark.asyncio
async def test_anonymous_session_does_not_poll(monkeypatch):
    """session_id=None (anonymous/non-explicit) never polls prefill progress."""
    from proxy.router_helpers import _await_first_byte_with_prefill_monitor

    srv = _make_srv()
    poll_calls = []

    async def fake_extend(*args, **kwargs):
        poll_calls.append(1)
        return (0, False)

    monkeypatch.setattr(
        "proxy.router_helpers._extend_lease_during_prefill", fake_extend
    )

    result = await _await_first_byte_with_prefill_monitor(
        srv,
        _slow_open(0.2),
        timeout=5.0,
        poll_seconds=0.05,
        warn_seconds=0.0,
        session_id=None,
    )
    assert result == ("cm", "resp")
    assert not poll_calls, "Anonymous sessions must not poll prefill progress"


@pytest.mark.asyncio
async def test_poll_failure_does_not_break_first_byte(monkeypatch):
    """A failing progress poll is swallowed; the first-byte wait still wins."""
    from proxy.router_helpers import _await_first_byte_with_prefill_monitor

    srv = _make_srv()

    async def failing_extend(*args, **kwargs):
        raise RuntimeError("status query exploded")

    monkeypatch.setattr(
        "proxy.router_helpers._extend_lease_during_prefill", failing_extend
    )

    result = await _await_first_byte_with_prefill_monitor(
        srv,
        _slow_open(0.15),
        timeout=5.0,
        poll_seconds=0.02,
        warn_seconds=0.0,
        session_id="sess-1",
    )
    assert result == ("cm", "resp")


@pytest.mark.asyncio
async def test_fast_first_byte_returns_without_poll(monkeypatch):
    """No regression: an immediate first byte returns with no poll delay."""
    from proxy.router_helpers import _await_first_byte_with_prefill_monitor

    srv = _make_srv()
    poll_calls = []

    async def fake_extend(*args, **kwargs):
        poll_calls.append(1)
        return (0, False)

    monkeypatch.setattr(
        "proxy.router_helpers._extend_lease_during_prefill", fake_extend
    )

    start = time.monotonic()
    result = await _await_first_byte_with_prefill_monitor(
        srv,
        _slow_open(0.0),
        timeout=5.0,
        poll_seconds=1.0,  # longer than the wait
        warn_seconds=0.0,
        session_id="sess-1",
    )
    elapsed = time.monotonic() - start
    assert result == ("cm", "resp")
    assert not poll_calls, "A first byte arriving before the poll cadence must not poll"
    assert elapsed < 0.5, f"Fast first byte should not be delayed, took {elapsed:.2f}s"
