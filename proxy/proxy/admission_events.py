"""Shared local-admission eviction event.

A local admission decision that ends up *evicting* a resident session (its
dispatch lease being reclaimed, its slot assignment freed, or its resident
prefix being replaced by another tenant) is an operator-visible event: the
2026-10-24 incident was invisible precisely because evictions were only
implied by the absence of reuse. This module is the **single** sink for that
event (``local_eviction``), so the residency-aware admission work
(``LP-0MVN1IQTF0037WTG`` AC6) and the post-acceptance eviction reporting
(``LP-0MVN3KMGI009BZ0D`` AC7) share one convention instead of growing two.

Contract:

- always ``ERROR`` level when an admission was accepted and later evicted
  (post-acceptance eviction is a defect signal, not noise);
- one stable prefix (``local_eviction``) plus a structured ``reason`` from
  :data:`EVICTION_REASONS`, so log consumers can count evictions by cause;
- the same fields the residency decision logs (``slot_owner_session``,
  ``slot``, ``endpoint``, tenant count) plus ``snapshot_planned`` (whether the
  evicted session's KV snapshot is being persisted so a returning client
  restores warm);
- never raises: an eviction path is already an exceptional path.
"""

from __future__ import annotations

import logging

#: A reclaimed dispatch lease (client disconnect, deadline cancel, orphan).
EVICTION_LEASE_EVICTED = "lease_evicted"

#: A slot assignment released after the session's tenancy ended normally.
EVICTION_SLOT_ASSIGNMENT_FREED = "slot_assignment_freed"

#: Another tenant took the pool and the resident prefix is no longer reusable.
EVICTION_RESIDENT_PREFIX_EVICTED = "resident_prefix_evicted_by_tenant"

#: A tenancy idle-lapsed (or its watchdog fired) without observed activity.
EVICTION_TENANCY_LEASE_EXPIRED = "tenancy_lease_expired"

#: The closed vocabulary of eviction reasons (AC6/AC7).
EVICTION_REASONS: tuple[str, ...] = (
    EVICTION_LEASE_EVICTED,
    EVICTION_SLOT_ASSIGNMENT_FREED,
    EVICTION_RESIDENT_PREFIX_EVICTED,
    EVICTION_TENANCY_LEASE_EXPIRED,
)


def log_local_eviction(
    srv,
    session_id: str | None,
    reason: str,
    *,
    endpoint: str | None = None,
    slot_id: int | None = None,
    slot_owner_session: str | None = None,
    tenant_count: int | None = None,
    snapshot_planned: bool | None = None,
    level: int = logging.ERROR,
    extra: dict | None = None,
) -> None:
    """Emit the shared ``local_eviction`` event (never raises).

    Args:
        srv: Server (or any object exposing ``logger``); ``None`` falls back to
            the module logger.
        session_id: The evicted session.
        reason: One of :data:`EVICTION_REASONS` (unknown values are still
            logged so a caller bug is visible rather than swallowed).
        endpoint: llama-server endpoint scope of the eviction.
        slot_id: Slot released by the eviction, when known.
        slot_owner_session: Session that owned the slot before the eviction.
        tenant_count: Distinct sessions holding the pool at eviction time.
        snapshot_planned: Whether the evicted session's KV snapshot is being
            persisted (so the returning client restores warm).
        level: Log level; defaults to ``ERROR`` (post-acceptance eviction).
        extra: Additional structured fields merged into ``extra``.
    """
    logger = getattr(srv, "logger", None) or logging.getLogger("llama-proxy")
    payload = {
        "event": "local_eviction",
        "session_id": session_id,
        "reason": reason,
        "endpoint": endpoint,
        "slot_id": slot_id,
        "slot_owner_session": slot_owner_session,
        "tenants": tenant_count,
        "snapshot_planned": snapshot_planned,
    }
    if extra:
        payload.update(extra)
    try:
        logger.log(
            level,
            "local_eviction session=%s reason=%s endpoint=%s slot=%s "
            "slot_owner_session=%s tenants=%s snapshot_planned=%s",
            session_id or "unknown",
            reason,
            endpoint or "default",
            slot_id if slot_id is not None else "none",
            slot_owner_session or "none",
            tenant_count if tenant_count is not None else "none",
            snapshot_planned if snapshot_planned is not None else "unknown",
            extra=payload,
        )
    except Exception:
        pass
