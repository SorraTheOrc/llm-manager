"""Pin the abandoned-drain starvation fix (AC3 second half, hermetic).

Parent: LP-0MVVEQW5Y00962T2 (AC3) — the test-first half
(LP-0MVW8S820004P0BI, this item); sibling implementation
LP-0MVW8S8DH008E4W4.

The mechanism the AC1 decision record (`docs/dev/abandoned-prefill-release-cancel.md`)
chose is **mechanism (c)** — bound the wait against the observed drain: the
retry must not be admitted into a slot that is still draining work the proxy
itself abandoned, and the drain is bounded (p90 6.6 s = one ubatch at the
measured rate), so the refusal carries the remaining drain as its retry-after.

This module pins the **fixed** contract. It was red-first (every contract
assertion carried ``xfail(strict=True)`` until the sibling implementation
LP-0MVW8S8DH008E4W4 landed); the implementation item removed the markers so the
assertions now run for real. Tests that must hold with or without the fix (the
harness/state contract and the warm-path guard) are not and never were xfail.

No live ``llama-server``, no network, no wall-clock sleeps longer than an
injected fake second: the abandoned-slot registry runs under the injected
clock the sibling tests use.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

#: The implementation item that landed the fix and removed the
#: ``xfail(strict=True)`` markers (kept for provenance: the markers were
#: removed by LP-0MVW8S8DH008E4W4).
IMPL_ITEM = "LP-0MVW8S8DH008E4W4"

#: The gate reason string the fix publishes on the refusal (asserted by
#: field, see AC3). Chosen to be machine-parseable and distinct from the
#: pre-existing gate reasons.
DRAIN_REASON = "abandoned_slot_drain"


class _InjectedClock:
    """Monotonic clock that only advances when the test drives it."""

    def __init__(self) -> None:
        self._now = 0.0

    @property
    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class _TimeShim:
    """Replace ``router_helpers.time`` so only ``monotonic`` is injected."""

    def __init__(self, clock: _InjectedClock) -> None:
        self._clock = clock

    def monotonic(self) -> float:
        return self._clock.now

    def __getattr__(self, name: str):
        return getattr(__import__("time"), name)


@pytest.fixture
def clock() -> _InjectedClock:
    return _InjectedClock()


@pytest.fixture(autouse=True)
def _isolate_abandoned_state(monkeypatch, clock: _InjectedClock):
    """Isolate the abandoned-slot registry and the injected clock."""
    from proxy import router_helpers as helpers

    monkeypatch.setattr(helpers, "time", _TimeShim(clock))
    reset = getattr(helpers, "_reset_abandoned_slots", None)
    if reset is not None:
        reset()
    yield
    if reset is not None:
        reset()


def _server_stub(records: dict | None = None, config: dict | None = None):
    """Minimal server stub with the dispatch-record state admission touches."""
    import asyncio as _asyncio

    return SimpleNamespace(
        config=config
        or {
            "server": {
                "session_slot_pool_size": 2,
                "local_dispatch_lease_timeout_seconds": 60,
                "local_dispatch_no_progress_timeout_seconds": 180,
                "local_dispatch_max_prefill_seconds": 900,
            }
        },
        local_active_queries=0,
        local_active_queries_lock=_asyncio.Lock(),
        local_dispatch_records=records if records is not None else {},
        local_dispatch_records_lock=_asyncio.Lock(),
        local_prefill_in_flight={},
        local_prefill_in_flight_lock=_asyncio.Lock(),
        logger=MagicMock(),
    )


def _bind_slot(session_id: str, slot_id: int) -> None:
    """Bind *session_id* to llama-server slot *slot_id* (session pinned)."""
    from proxy.session import _slot_registry_key

    patch.dict(
        "proxy.session._slot_owners",
        {_slot_registry_key(None, slot_id): session_id},
        clear=True,
    ).start()


def _record_drain(slot_id: int, session_id: str, reason: str, clock) -> dict:
    """Record a live abandonment entry (the drain state the fix must see)."""
    from proxy.router_helpers import _record_abandoned_slot

    return _record_abandoned_slot(
        slot_id,
        session_id=session_id,
        reason=reason,
        now=clock.now,
    )


def _gate_refuses(
    srv,
    session_key: str,
    config: dict | None = None,
) -> tuple[bool, str, float]:
    """Run the real dispatch-gate introduction and return its verdict."""
    from proxy.router_helpers import _local_dispatch_gate

    return _local_dispatch_gate(
        srv,
        session_key,
        config=config if config is not None else srv.config,
        entry="lease_acquire",
    )


def _rendered_log_calls(mock, needle: str) -> list[str]:
    """Render the ``%``-formatted calls a mocked logger received."""
    rendered: list[str] = []
    for logged in getattr(mock, "call_args_list", []):
        args = logged.args
        if not args or not isinstance(args[0], str) or needle not in args[0]:
            continue
        try:
            rendered.append(args[0] % tuple(args[1:]))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            rendered.append(args[0])
    return rendered


# ---------------------------------------------------------------------------
# AC0 — state contract that must hold with today's code (not xfail)
# ---------------------------------------------------------------------------


def test_live_drain_entry_is_visible_to_the_published_annotation(clock):
    """The drain state the fix consumes is the published annotation: a real
    slot id records a live entry under the injected clock, readable through
    ``_abandoned_slot_for``. Locks the harness, not the fix."""
    from proxy.router_helpers import _abandoned_slot_for

    entry = _record_drain(0, "sess-S", "prefill_deadline", clock)
    assert entry["slot_id"] == 0
    assert entry["session_id"] == "sess-S"
    assert _abandoned_slot_for(0, now=clock.now) is not None
    # An equal slot id must stay readable and then expire with the clock.
    clock.advance(900.0 + 1.0)
    assert _abandoned_slot_for(0, now=clock.now) is None


# ---------------------------------------------------------------------------
# AC1 / AC2 — the starvation shape and the fixed behaviour
# ---------------------------------------------------------------------------


def test_retry_is_not_admitted_into_own_draining_slot_deadline_variant(clock):
    """AC1/AC2a: session S pinned to slot 0 releases its dispatch at the
    prefill deadline; the drain entry is live; S's retry (same session, same
    slot) must NOT be admitted while the slot drains — the gate refuses with
    the drain reason instead of accepting the retry into zero-progress work."""
    from proxy.session import _slot_registry_key

    srv = _server_stub()
    with patch.dict(
        "proxy.session._slot_owners",
        {_slot_registry_key(None, 0): "sess-S"},
        clear=True,
    ):
        _record_drain(0, "sess-S", "prefill_deadline", clock)
        refused, reason, remaining = _gate_refuses(srv, "sess-S")

    assert refused is True, (
        "retry into a slot still draining its own abandoned work must be "
        "refused before acceptance (mechanism (c))"
    )
    assert reason == DRAIN_REASON
    assert remaining > 0.0


def test_retry_is_not_admitted_into_own_draining_slot_measured_deficit_variant(
    clock,
):
    """AC1/AC2a, Option D variant: same shape, released by the measured
    cache-deficit early abort. The fix must treat both abandonment reasons the
    same (the parent AC names measured_cache_deficit explicitly)."""
    srv = _server_stub()
    _record_drain(0, "sess-S", "measured_cache_deficit", clock)
    refused, reason, remaining = _gate_refuses(srv, "sess-S")

    assert refused is True
    assert reason == DRAIN_REASON
    assert remaining > 0.0


def test_drain_refusal_bounds_the_wait_to_the_drain_ttl(clock):
    """AC2b: the refusal is a bound, not a quarantine — once the drain ends
    (the recorded window expires) the same session is admitted again with the
    ordinary warm-path outcome. Measuring the wait as the remaining drain keeps
    the bound honest instead of a fixed sleep."""
    from proxy.session import _slot_registry_key

    srv = _server_stub()
    with patch.dict(
        "proxy.session._slot_owners",
        {_slot_registry_key(None, 0): "sess-S"},
        clear=True,
    ):
        _record_drain(0, "sess-S", "prefill_deadline", clock)
        refused, _reason, remaining = _gate_refuses(srv, "sess-S")
        assert refused is True
        assert remaining > 0.0

        # Drain ends: the recorded window expires (bounded TTL)...
        clock.advance(remaining + 1.0)

        # ... and the retry is admitted exactly as if no drain had occurred.
        from proxy.router_helpers import _try_acquire_local_dispatch

        acquired, _owner, _count, _retry = asyncio.run(
            _try_acquire_local_dispatch(
                srv, max_local=2, session_key="sess-S", backend="local"
            )
        )
        assert acquired is True


# ---------------------------------------------------------------------------
# AC3 — the decision is observable (attributable log field)
# ---------------------------------------------------------------------------


def test_drain_refusal_emits_an_attributable_log_line(clock):
    """AC3: the refusal is observable in the log by field, not guessed —
    ``routing_skip_local … reason=abandoned_slot_drain … entry=…`` so the
    post-change window can count drain refusals without re-parsing."""
    srv = _server_stub()
    _record_drain(0, "sess-S", "prefill_deadline", clock)
    _gate_refuses(srv, "sess-S")

    lines = _rendered_log_calls(srv.logger.info, "routing_skip_local")
    assert lines, "the drain refusal must be logged on the dispatch side"
    drain_lines = [ln for ln in lines if f"reason={DRAIN_REASON}" in ln]
    assert len(drain_lines) == 1, drain_lines
    assert "session=sess-S" in drain_lines[0]
    assert "entry=lease_acquire" in drain_lines[0]
    # The remaining-drain bound is present so an operator can see the wait.
    import re

    assert re.search(r"cooldown_remaining=[0-9.]+s", drain_lines[0])


# ---------------------------------------------------------------------------
# AC4 — warm path untouched when no drain is present
# ---------------------------------------------------------------------------


def test_no_drain_warm_path_is_unchanged(clock):
    """AC4: with no abandoned entry on the session's slot, admission is exactly
    today's behaviour: no refusal, no drain log line, session keeps its slot.
    (Not xfail: the warm path must already hold and stay green.)"""
    from proxy.session import _slot_registry_key
    from proxy.router_helpers import _try_acquire_local_dispatch

    srv = _server_stub()
    with patch.dict(
        "proxy.session._slot_owners",
        {_slot_registry_key(None, 0): "sess-S"},
        clear=True,
    ):
        refused, _reason, _remaining = _gate_refuses(srv, "sess-S")
        assert refused is False

        acquired, _owner, _count, _retry = asyncio.run(
            _try_acquire_local_dispatch(
                srv, max_local=2, session_key="sess-S", backend="local"
            )
        )
        assert acquired is True
        assert srv.local_dispatch_records["sess-S"]["slot_id"] == 0

    drain_lines = _rendered_log_calls(srv.logger.info, DRAIN_REASON)
    assert not drain_lines, "no drain present → no drain refusal may be logged"


# ---------------------------------------------------------------------------
# AC5 — no duplication of the annotation work
# ---------------------------------------------------------------------------


def test_test_shim_uses_only_the_published_annotation(clock):
    """AC5: this module consumes the published abandoned-slot registry
    (``_record_abandoned_slot`` / ``_abandoned_slot_for``) exactly as today's
    code writes it; it never records slot identity itself and passes whether
    ``abandoned_slot_id`` is populated by this tree or by
    LP-0MVUVHAQP001IRKK."""
    from proxy.router_helpers import (
        _abandoned_slot_for,
        _abandoned_slot_lookup,
    )

    # The lookup path the fix may use (session → slot) is read-only and must
    # not mutate the registry.
    _record_drain(0, "sess-S", "prefill_deadline", clock)
    assert _abandoned_slot_lookup("sess-S", endpoint="local") in (0, None)
    assert _abandoned_slot_for(0, now=clock.now) is not None
