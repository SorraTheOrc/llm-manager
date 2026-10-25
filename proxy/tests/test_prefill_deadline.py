"""Prefill deadline: bounded first-byte detour for local dispatches.

LP-0MVIQ0G2M0006QF6 AC3 — a *hard backstop independent of ETA-estimator
quality*: if a local dispatch has produced no first byte by
``local_prefill_deadline_seconds`` (default 90 s), the pending upstream
request is cancelled and the local dispatch aborts with the retryable 503 the
first-byte ceiling already produces, so the chain detours to the next
provider instead of leaving the client on a silent socket for minutes
(baseline: 48 of 606 local dispatches had a first byte after >120 s, max
592 s).

The deadline is deliberately:
  * ABOVE the admission budget (``contention_queue_max_wait_seconds``, 60 s)
    so a request legitimately admitted inside its budget is never cut off;
  * BELOW ``local_dispatch_first_byte_timeout_seconds`` (840 s) so it is the
    bound that actually fires;
  * bounded above by ``local_dispatch_max_prefill_seconds`` (lifetime wedge
    bound, unchanged).
"""

from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _srv(**server_overrides) -> SimpleNamespace:
    server = {
        "local_prefill_deadline_seconds": 90,
        "local_dispatch_first_byte_timeout_seconds": 840,
        "local_dispatch_max_prefill_seconds": 900,
    }
    server.update(server_overrides)
    return SimpleNamespace(config={"server": server}, logger=MagicMock())


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------

def test_prefill_deadline_defaults_to_90():
    from proxy.router_helpers import _get_prefill_deadline_seconds

    assert _get_prefill_deadline_seconds(_srv(local_prefill_deadline_seconds=None)) == 90.0
    assert _get_prefill_deadline_seconds(_srv(local_prefill_deadline_seconds=45)) == 45.0


def test_prefill_deadline_malformed_value_uses_safe_default():
    """AC4: a malformed value falls back to the safe default, never raises."""
    from proxy.router_helpers import _get_prefill_deadline_seconds

    srv = _srv()
    srv.config["server"]["local_prefill_deadline_seconds"] = "not-a-number"
    assert _get_prefill_deadline_seconds(srv) == 90.0


def test_prefill_deadline_is_strictly_below_the_wedge_ceiling():
    """AC4: the deadline is the bound that fires, not the 840 s wedge bound."""
    from proxy.router_helpers import _get_first_byte_timeout_seconds

    srv = _srv()
    ceiling = _get_first_byte_timeout_seconds(srv)
    assert ceiling == 90.0, (
        "the effective first-byte ceiling must be tightened to the deadline"
    )
    assert ceiling < 840.0
    assert ceiling < 900.0  # lifetime wedge bound unchanged, not exceeded


def test_prefill_deadline_does_not_lengthen_a_tighter_explicit_ceiling():
    """A first-byte ceiling tighter than the deadline must be preserved."""
    from proxy.router_helpers import _get_first_byte_timeout_seconds

    srv = _srv(local_dispatch_first_byte_timeout_seconds=30)
    assert _get_first_byte_timeout_seconds(srv) == 30.0


def test_disabled_deadline_keeps_the_original_ceiling():
    """0 disables the deadline: the previous ceiling behaviour is unchanged."""
    from proxy.router_helpers import _get_first_byte_timeout_seconds

    srv = _srv(local_prefill_deadline_seconds=0)
    assert _get_first_byte_timeout_seconds(srv) == 840.0


def test_disabled_first_byte_ceiling_still_honours_the_deadline():
    """A disabled wedge ceiling (0) must not disable the deadline backstop."""
    from proxy.router_helpers import _get_first_byte_timeout_seconds

    srv = _srv(local_dispatch_first_byte_timeout_seconds=0)
    assert _get_first_byte_timeout_seconds(srv) == 90.0


# ---------------------------------------------------------------------------
# Behaviour: a stalled upstream is cancelled at the deadline
# ---------------------------------------------------------------------------

def _monitor_srv() -> SimpleNamespace:
    return SimpleNamespace(
        config={
            "server": {
                "local_prefill_deadline_seconds": 90,
                "local_dispatch_first_byte_timeout_seconds": 840,
                "local_dispatch_max_prefill_seconds": 900,
                "local_dispatch_no_progress_timeout_seconds": 180,
                "local_dispatch_lease_timeout_seconds": 30,
                "local_dispatch_lease_chunk_refresh_buffer_seconds": 30,
            }
        },
        logger=MagicMock(),
        local_dispatch_records={},
        local_dispatch_records_lock=asyncio.Lock(),
        local_prefill_in_flight={},
        local_active_queries=1,
    )


@pytest.mark.asyncio
async def test_stalled_upstream_is_cancelled_when_the_deadline_fires(monkeypatch):
    """AC3: no first byte by the deadline ⇒ the pending upstream request is
    cancelled (aborted), so no orphaned lease/slot survives the detour."""
    from proxy.router_helpers import _await_first_byte_with_prefill_monitor

    srv = _monitor_srv()
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
            timeout=0.12,  # stands in for the 90 s prefill deadline
            poll_seconds=0.0,
            warn_seconds=0.0,
            session_id=None,
        )

    for _ in range(20):
        if cancelled.is_set():
            break
        await asyncio.sleep(0.01)
    assert cancelled.is_set(), "the stalled upstream request must be cancelled"


@pytest.mark.asyncio
async def test_first_byte_before_the_deadline_is_returned_unchanged(monkeypatch):
    """AC3 (no false positive): a dispatch that produces a first byte inside
    the deadline is untouched — the deadline must not fire for healthy local
    work."""
    from proxy.router_helpers import _await_first_byte_with_prefill_monitor

    srv = _monitor_srv()

    async def _fast():
        await asyncio.sleep(0.01)
        return ("cm", "response")

    result = await _await_first_byte_with_prefill_monitor(
        srv,
        _fast(),
        timeout=5.0,
        poll_seconds=0.02,
        warn_seconds=0.0,
        session_id=None,
    )
    assert result == ("cm", "response")
    assert not srv.logger.warning.called
