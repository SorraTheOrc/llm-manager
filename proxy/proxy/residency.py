"""Read-only residency projection over the cached ``/slots`` snapshot.

Local admission needs to know, per slot, *which session owns the prefix that
is currently resident and how much of it is reusable*. Two in-process sources
already answer that:

- ``observability._last_slot_details_cache`` — the last successful ``/slots``
  detail snapshot, refreshed ~1 s by ``_periodic_broadcast_loop()``
  (TTL-coalesced, 5 s failure backoff), stamped with
  ``observability._last_slot_details_at``;
- ``session._slot_owners`` — the authoritative ``(endpoint, slot_id) ->
  session_id`` ownership registry.

This module joins them into one small immutable projection so the
resident-prefix gate (AC1), the tenancy bound (AC2), live-prefix pricing
(AC3), the admission logging (AC5) and ``admission_reserve`` all reason about
*the same* prefix facts (LP-0MVN1IQTF0037WTG).

Hard constraints it exists to honour:

- **No upstream call on the admission hot path.** ``/slots`` is held for more
  than 15 s while llama-server processes a prompt batch (measured 8.7 s then
  >15 s against a 1 s status timeout, ``LP-0MUXTW4MD003TVIC``), so admission
  may only read what has already been cached — never issue a synchronous
  ``/slots`` query.
- **Fail-open on unknown state.** A missing snapshot, a snapshot older than
  :data:`RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS`, or a slot whose build no longer
  reports ``context_tokens`` is reported as *unknown* (``known is False`` /
  ``context_tokens is None``), never as an exception and never as a
  fabricated context size: the caller falls back to the session-owns-slot
  check and then to the predicted-ratio decision.
- **No lock inversion.** Both sources are copied into the view up front, so a
  concurrent snapshot refresh or ownership change cannot mutate a view that
  is mid-decision and no lock is held across an ``await``.

The projection is intentionally dependency-free: no new poller, no new task,
no new config knob.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

#: Maximum age (seconds) of the cached ``/slots`` snapshot for its measurements
#: to be trusted. The snapshot is refreshed about once a second; a few seconds
#: of slack keeps decisions honest without turning a transiently stale cache
#: into a hard local outage (fail-open: the view reports ``known is False``).
RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS: float = 5.0


@dataclass(frozen=True)
class SlotResidency:
    """One slot's residency facts, as last observed."""

    endpoint: str
    slot_id: int
    owner_session: str | None
    context_tokens: int | None
    is_processing: bool

    @property
    def reusable(self) -> int | None:
        """Alias making the AC1/AC5 field name explicit at call sites."""
        return self.context_tokens


@dataclass(frozen=True)
class ResidencyView:
    """Immutable projection of the local pool's residency at one instant.

    ``known`` is the fail-open signal: it is ``True`` only when a snapshot
    exists *and* is fresh. Consumers must treat an unknown view as "no
    measurement available" (fall back one rung of the admission ladder) rather
    than as "nothing is resident".
    """

    slots: tuple[SlotResidency, ...] = ()
    snapshot_age: float | None = None
    known: bool = False

    # -- per-session -------------------------------------------------------
    def slot_for_session(
        self, session_id: str | None, endpoint: str | None = None
    ) -> SlotResidency | None:
        """Return the slot *session_id* owns, preferring *endpoint*.

        Returns ``None`` when the session owns no slot (a non-resident
        session) or *session_id* is empty.
        """
        if not session_id:
            return None
        ep = endpoint or ""
        owned = [s for s in self.slots if s.owner_session == session_id]
        for slot in owned:
            if slot.endpoint == ep:
                return slot
        return owned[0] if owned else None

    def context_tokens_for_session(
        self, session_id: str | None, endpoint: str | None = None
    ) -> int | None:
        """Return the session's measured reusable prefix, or ``None``.

        ``None`` means *unknown*: either the session owns no slot or the
        owning slot did not report ``context_tokens`` (llama.cpp b8782 dropped
        ``n_past``), which the admission ladder must not read as zero.
        """
        slot = self.slot_for_session(session_id, endpoint=endpoint)
        return None if slot is None else slot.context_tokens

    # -- per-slot ----------------------------------------------------------
    def owner_of_slot(self, slot_id: int, endpoint: str | None = None) -> str | None:
        """Return the session id that owns *slot_id* on *endpoint*."""
        ep = endpoint or ""
        for slot in self.slots:
            if slot.slot_id != slot_id:
                continue
            if endpoint is None or slot.endpoint == ep:
                return slot.owner_session
        return None

    # -- pool-wide ---------------------------------------------------------
    def occupied_sessions(self, endpoint: str | None = None) -> tuple[str, ...]:
        """Return the distinct session ids that own at least one slot."""
        ep = endpoint or ""
        seen: list[str] = []
        for slot in self.slots:
            if endpoint is not None and slot.endpoint != ep:
                continue
            if slot.owner_session and slot.owner_session not in seen:
                seen.append(slot.owner_session)
        return tuple(seen)

    def processing_sessions(self, endpoint: str | None = None) -> tuple[str, ...]:
        """Return the distinct session ids whose slot is actively working.

        A slot that is ``is_processing`` is holding its resident prefix for a
        live turn, which is what the tenancy bound (AC2) counts.
        """
        ep = endpoint or ""
        seen: list[str] = []
        for slot in self.slots:
            if endpoint is not None and slot.endpoint != ep:
                continue
            if (
                slot.is_processing
                and slot.owner_session
                and slot.owner_session not in seen
            ):
                seen.append(slot.owner_session)
        return tuple(seen)


def _endpoint_for_slot(owners: dict, slot_id: int, preferred: str | None) -> str:
    """Resolve the endpoint a cached slot id belongs to.

    The cached snapshot carries no endpoint (it comes from the current local
    child), so the endpoint is taken from the ownership registry: the
    preferred endpoint when it owns that slot id, otherwise the first
    endpoint that does. Legacy integer keys are treated as the empty
    endpoint, matching ``session._slot_key_endpoint``.
    """
    ep = preferred or ""
    if (ep, slot_id) in owners:
        return ep
    for key in owners:
        if isinstance(key, tuple) and len(key) == 2 and key[1] == slot_id:
            return key[0] or ""
    return ep


def get_residency_view(
    now: float | None = None, *, endpoint: str | None = None
) -> ResidencyView:
    """Project the cached slot snapshot + ownership registry into a view.

    Read-only and synchronous: no ``/slots`` query, no new polling task, no
    lock held across an ``await``. The snapshot and the owner map are copied
    up front, so the returned view is stable for the duration of a decision
    even if a refresh lands concurrently.

    Args:
        now: Monotonic timestamp to age the snapshot against (tests pass a
            frozen clock); defaults to ``time.monotonic()``.
        endpoint: Endpoint whose ownership should be preferred when several
            local llama-server instances are in play.

    Returns:
        A :class:`ResidencyView`. ``known`` is ``False`` when the snapshot is
        missing or older than :data:`RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS`.
    """
    try:
        from proxy import observability
        from proxy import session as session_registry

        snapshot = list(observability._last_slot_details_cache)
        taken_at = float(getattr(observability, "_last_slot_details_at", 0.0) or 0.0)
        owners = dict(session_registry._slot_owners)
    except Exception:  # pragma: no cover - defensive: never break admission
        return ResidencyView()

    if not snapshot or taken_at <= 0.0:
        return ResidencyView()

    current = time.monotonic() if now is None else float(now)
    age = max(0.0, current - taken_at)

    slots: list[SlotResidency] = []
    for entry in snapshot:
        if not isinstance(entry, dict):
            continue
        slot_id = entry.get("slot_id")
        if not isinstance(slot_id, int):
            continue
        ep = _endpoint_for_slot(owners, slot_id, endpoint)
        raw_ctx = entry.get("context_tokens")
        context_tokens = (
            int(raw_ctx)
            if isinstance(raw_ctx, (int, float)) and raw_ctx >= 0
            else None
        )
        slots.append(
            SlotResidency(
                endpoint=ep,
                slot_id=slot_id,
                owner_session=owners.get((ep, slot_id)),
                context_tokens=context_tokens,
                is_processing=bool(entry.get("is_processing", False)),
            )
        )

    return ResidencyView(
        slots=tuple(slots),
        snapshot_age=age,
        known=age <= RESIDENCY_SNAPSHOT_MAX_AGE_SECONDS,
    )
