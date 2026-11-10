"""Contention-queue Prometheus metrics (LP-0MVWPJW2700966ZU).

Covers the three series published at the queue's own admit / wait / drop
points (AC1-AC3):

- ``proxy_contention_queue_depth`` (gauge) — waiters currently queued
- ``proxy_contention_queue_wait_seconds`` (histogram) — completed waits
- ``proxy_contention_queue_loss_total{reason}`` (counter) — dropped requests by
  canonical fall-through reason

The tests drive the *public* queue API (``wait_for_local_slot``) rather than
poking module internals, so they pin the real admission/wait/loss paths. The
metric objects are process-global singletons, so every assertion is a
before/after delta and the autouse fixture resets the queue (and therefore the
depth gauge) between cases.
"""

import asyncio

import pytest

from proxy import contention_queue

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _fresh_queue():
    """Start and end every case with an empty queue and a fresh condition."""
    contention_queue.reset()
    yield
    contention_queue.reset()


def _depth() -> float:
    """Read the depth gauge under either prometheus_client flavour.

    The real client exposes ``_value.get()``; the repo-root test stub
    (``prometheus_client.py``) stores an unlabelled gauge in ``_store[()]``.
    """
    from proxy.metrics import proxy_contention_queue_depth

    value = getattr(proxy_contention_queue_depth, "_value", None)
    if value is not None:
        return float(value.get())
    return float(proxy_contention_queue_depth._store.get((), 0))


def _wait_count() -> float:
    from proxy.metrics import proxy_contention_queue_wait_seconds

    return float(proxy_contention_queue_wait_seconds._count.get())


def _wait_sum() -> float:
    from proxy.metrics import proxy_contention_queue_wait_seconds

    return float(proxy_contention_queue_wait_seconds._sum.get())


def _loss(reason: str) -> float:
    from proxy.metrics import proxy_contention_queue_loss_total

    return float(
        proxy_contention_queue_loss_total.labels(reason=reason)._value.get()
    )


async def _settle(loops: int = 5) -> None:
    """Let queued coroutines reach their ``await`` point."""
    for _ in range(loops):
        await asyncio.sleep(0)


def test_series_are_defined_with_expected_shape():
    """The three series exist with the documented types and labels.

    AC1: a depth gauge, a wait histogram and a loss counter carrying the
    canonical reason label. This is a behaviour test of the exposition
    surface, not a type-level tautology: a missing/renamed metric would break
    every scrape and alert built on it.
    """
    from prometheus_client import Counter, Gauge, Histogram
    from proxy import metrics

    assert isinstance(metrics.proxy_contention_queue_depth, Gauge)
    assert isinstance(metrics.proxy_contention_queue_wait_seconds, Histogram)
    assert isinstance(metrics.proxy_contention_queue_loss_total, Counter)

    assert list(metrics.proxy_contention_queue_depth._labelnames) == []
    assert list(metrics.proxy_contention_queue_wait_seconds._labelnames) == []
    assert list(metrics.proxy_contention_queue_loss_total._labelnames) == ["reason"]


async def test_depth_gauge_tracks_admit_and_dequeue():
    """Depth rises on admission and returns to zero when the waiter leaves.

    AC2: the value is published at the admission and dequeue decisions, so a
    scrape never reads a depth stale by more than one decision.
    """
    assert _depth() == 0

    free = False

    def _slot_free() -> bool:
        return free

    task = asyncio.create_task(
        contention_queue.wait_for_local_slot(
            max_wait_seconds=5.0,
            max_depth=4,
            slot_free_check=_slot_free,
            session_id="depth-session",
        )
    )
    await _settle()

    assert contention_queue.queue_depth() == 1
    assert _depth() == 1

    free = True
    await contention_queue.wake(1)
    assert await asyncio.wait_for(task, timeout=2) is not None

    assert contention_queue.queue_depth() == 0
    assert _depth() == 0


async def test_admitted_wait_is_observed_on_the_histogram():
    """A wait that wins is observed as elapsed seconds (AC2/AC3)."""
    before_count = _wait_count()
    before_sum = _wait_sum()
    free = False

    def _slot_free() -> bool:
        return free

    task = asyncio.create_task(
        contention_queue.wait_for_local_slot(
            max_wait_seconds=5.0,
            max_depth=4,
            slot_free_check=_slot_free,
            session_id="win-session",
        )
    )
    await _settle()
    free = True
    await contention_queue.wake(1)
    elapsed = await asyncio.wait_for(task, timeout=2)

    assert elapsed is not None and elapsed >= 0.0
    assert _wait_count() == before_count + 1
    assert _wait_sum() >= before_sum + elapsed


async def test_loss_counter_records_depth_cap_with_default_reason():
    """A request refused at the depth cap is counted as a queue loss.

    AC3: the counter moves on a loss and carries the canonical
    ``fallback_after_queue`` reason the router records.
    """
    free = False

    def _slot_free() -> bool:
        return free

    # Fill the queue to its depth cap.
    block = [
        asyncio.create_task(
            contention_queue.wait_for_local_slot(
                max_wait_seconds=5.0,
                max_depth=1,
                slot_free_check=_slot_free,
                session_id="cap-0",
            )
        )
    ]
    await _settle()
    assert contention_queue.queue_depth() == 1
    assert _depth() == 1

    before = _loss(contention_queue.DEFAULT_LOSS_REASON)
    result = await contention_queue.wait_for_local_slot(
        max_wait_seconds=5.0,
        max_depth=1,
        slot_free_check=_slot_free,
        session_id="cap-overflow",
    )

    assert result is None
    assert _loss(contention_queue.DEFAULT_LOSS_REASON) == before + 1
    assert _depth() == 1  # the refused request never enqueued

    # Drain the blocked waiter so the module state is clean.
    free = True
    await contention_queue.wake(1)
    await asyncio.wait_for(block[0], timeout=2)


async def test_loss_counter_records_custom_canonical_reason():
    """The lease-protected call site's reason reaches the counter label."""
    before = _loss("local_lease_active")

    result = await contention_queue.wait_for_local_slot(
        max_wait_seconds=5.0,
        max_depth=0,  # cap 0 refuses before enqueueing anything
        slot_free_check=lambda: False,
        session_id="lease-session",
        loss_reason="local_lease_active",
    )

    assert result is None
    assert _loss("local_lease_active") == before + 1


async def test_loss_counter_and_histogram_record_wait_expiry():
    """A wait that burns its budget is a loss *and* an observed wait.

    AC2/AC3: the timed-out wait is in the histogram (so the loss latency is
    visible) and the loss counter is labelled with the canonical reason.
    """
    before_count = _wait_count()
    before_loss = _loss(contention_queue.DEFAULT_LOSS_REASON)

    result = await contention_queue.wait_for_local_slot(
        max_wait_seconds=0.05,
        max_depth=4,
        slot_free_check=lambda: False,
        session_id="timeout-session",
    )

    assert result is None
    assert _wait_count() == before_count + 1
    assert _loss(contention_queue.DEFAULT_LOSS_REASON) == before_loss + 1
    assert _depth() == 0


async def test_recorders_are_noops_when_metrics_disabled(monkeypatch):
    """Without prometheus_client the queue still waits and drops silently.

    AC5: observability must never change queue behaviour — the recorders are
    best-effort and a disabled metrics module cannot raise into the queue.
    """
    from proxy import metrics

    monkeypatch.setattr(metrics, "_enabled", False)

    # A depth-cap loss must still return None (behaviour) without raising.
    result = await contention_queue.wait_for_local_slot(
        max_wait_seconds=5.0,
        max_depth=0,
        slot_free_check=lambda: False,
        session_id="disabled-session",
    )
    assert result is None

    # A successful wait must still return elapsed seconds.
    free = False
    task = asyncio.create_task(
        contention_queue.wait_for_local_slot(
            max_wait_seconds=5.0,
            max_depth=4,
            slot_free_check=lambda: free,
            session_id="disabled-win",
        )
    )
    await _settle()
    free = True
    await contention_queue.wake(1)
    assert await asyncio.wait_for(task, timeout=2) is not None
