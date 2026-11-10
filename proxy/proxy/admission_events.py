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
- the pricing evidence (``estimated_tokens`` / ``new_tokens`` /
  ``cached_ratio`` / ``eta_prefill_s``) is the **admission-time snapshot** the
  routing decision stamped on the request (LP-0MVODJ3P6000AIUM AC1): the
  dispatch record stores the decision's numbers verbatim, so an eviction event
  never re-prices the prompt from drifted state. A record created without a
  snapshot carries explicit ``None`` markers rendered as ``0``/``unknown``
  (AC4) — never a silent recompute.
- the identity evidence (``slot_id`` / ``llama_task_id``) associates the
  evicted dispatch with the llama-server slot and task it used
  (LP-0MW959RRO008AAH9, parent AC10). The slot is the candidate pinned on the
  record at admission; the task id is read from the already-cached ``/slots``
  snapshot (never a new upstream call) and re-resolved at release, because the
  slot is usually still idle when the record is created. A record with a known
  slot never reports a bare ``llama_task_id=unknown``: when the id cannot be
  resolved the line carries ``llama_task_id_reason=`` naming why (AC10c).
- never raises: an eviction path is already an exceptional path.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

#: A reclaimed dispatch lease (client disconnect, deadline cancel, orphan).
EVICTION_LEASE_EVICTED = "lease_evicted"

#: A slot assignment released after the session's tenancy ended normally.
EVICTION_SLOT_ASSIGNMENT_FREED = "slot_assignment_freed"

#: Another tenant took the pool and the resident prefix is no longer reusable.
EVICTION_RESIDENT_PREFIX_EVICTED = "resident_prefix_evicted_by_tenant"

#: A tenancy idle-lapsed (or its watchdog fired) without observed activity.
EVICTION_TENANCY_LEASE_EXPIRED = "tenancy_lease_expired"

#: The prefill deadline cancelled an accepted dispatch (parent AC7).
EVICTION_PREFILL_DEADLINE = "prefill_deadline"

#: The dispatch was released because the *measured* ``prompt_progress`` showed
#: it re-prefilling far more than admission priced (Option D early abort,
#: LP-0MVR7IN0O000DPXN). Distinct from :data:`EVICTION_PREFILL_DEADLINE` so the
#: post-deploy window can count early detours separately from 120 s ones.
EVICTION_MEASURED_CACHE_DEFICIT = "measured_cache_deficit"

#: The no-progress watchdog released an accepted dispatch.
EVICTION_NO_PROGRESS_WATCHDOG = "no_progress_watchdog"

#: An inactive dispatch record expired past ``expires_at``.
EVICTION_IDLE_TIMEOUT = "idle_timeout"

#: A deliberate backend restart (crash recovery, self-heal, model switch) threw
#: away the prefills that were in flight (LP-0MVZVZLCS000LQNB AC3). Without
#: this reason a restart looks like a capacity problem: the abandoned
#: dispatches simply vanish from the accounting while the cache is cold.
EVICTION_BACKEND_RESTART = "backend_restart"

#: The orphan-cleanup sweep released a dispatch with no live backend slot.
EVICTION_ORPHAN_CLEANUP = "orphan_cleanup"

#: The reasons an ACCEPTED-and-unserved dispatch can be released for
#: (LP-0MVN4LCED00661MX AC1, extended by LP-0MVZVZLCS000LQNB AC3). Every one of
#: these is an ERROR when it happens before the first byte.
EVICTION_REASONS_AFTER_ACCEPTANCE: tuple[str, ...] = (
    EVICTION_PREFILL_DEADLINE,
    EVICTION_MEASURED_CACHE_DEFICIT,
    EVICTION_NO_PROGRESS_WATCHDOG,
    EVICTION_IDLE_TIMEOUT,
    EVICTION_ORPHAN_CLEANUP,
    EVICTION_LEASE_EVICTED,
    EVICTION_SLOT_ASSIGNMENT_FREED,
    EVICTION_BACKEND_RESTART,
)

#: The closed vocabulary of eviction reasons (AC6/AC7).
EVICTION_REASONS: tuple[str, ...] = (
    EVICTION_LEASE_EVICTED,
    EVICTION_SLOT_ASSIGNMENT_FREED,
    EVICTION_RESIDENT_PREFIX_EVICTED,
    EVICTION_TENANCY_LEASE_EXPIRED,
    EVICTION_PREFILL_DEADLINE,
    EVICTION_MEASURED_CACHE_DEFICIT,
    EVICTION_NO_PROGRESS_WATCHDOG,
    EVICTION_IDLE_TIMEOUT,
    EVICTION_ORPHAN_CLEANUP,
    EVICTION_BACKEND_RESTART,
)


# ---------------------------------------------------------------------------
# Post-acceptance eviction (LP-0MVN4LCED00661MX, parent AC7)
# ---------------------------------------------------------------------------

#: Monotonic per-reason counters (reset only on process restart, AC5).
_after_acceptance_counts: dict[str, int] = {}


def record_eviction_after_acceptance(
    srv,
    session_id: str | None,
    reason: str,
    record: dict | None = None,
    *,
    endpoint: str | None = None,
    elapsed_s: float | None = None,
    extra_fields: Mapping[str, object] | None = None,
) -> bool:
    """Record that an ACCEPTED local dispatch was evicted before its first byte.

    This is the AC7 defect signal: an accepted dispatch (admitted, dispatch
    record created) that is released before producing a first byte is not
    expected churn — it means the admission/pricing model admitted something
    the GPU could not serve, so it is logged at **ERROR** with machine-parseable
    evidence and counted.

    Classification is by the record's ``first_byte_seen`` flag, never by the
    release path (AC3): a lease that expires *after* a served response is
    normal churn and stays at its current level, so it is not an error and not
    counted.

    Idempotent per record: a second release path racing the first cannot emit
    the event twice (the flag lives on the record).

    Evidence semantics (LP-0MVODJ3P6000AIUM AC1/AC4): the pricing fields are
    the admission-time values the routing decision used and the request's
    dispatch record stored verbatim — the event must never claim an over-
    budget admission that the decision never priced. Records without a
    threaded snapshot render ``0``/``unknown`` explicitly; nothing here is
    recomputed from the request body.

    Args:
        srv: Server object (its ``logger`` when present); may be ``None``.
        session_id: The evicted session.
        reason: One of :data:`EVICTION_REASONS_AFTER_ACCEPTANCE`.
        record: The dispatch record being released, when the caller has it.
            ``None`` (a path without a record) is treated as post-acceptance
            and is not de-duplicated — every caller that has a record must
            pass it.
        endpoint: llama-server endpoint scope of the eviction.
        elapsed_s: Seconds from dispatch acceptance to the eviction; computed
            from the record's ``started_at`` (monotonic) when omitted.
        extra_fields: Optional caller-specific ``key=value`` evidence appended
            to the line (e.g. the measured/priced numbers behind a
            ``measured_cache_deficit`` release). Omitted (``None``) leaves the
            line byte-for-byte as before, so every pre-existing consumer and
            test regex is unaffected.

    Returns:
        True when an event was emitted (and counted), False when suppressed
        (already reported, or the dispatch had already served a first byte).
    """
    if record is not None:
        if record.get("eviction_logged"):
            return False
        if record.get("first_byte_seen"):
            return False
        record["eviction_logged"] = True

    if elapsed_s is None:
        try:
            import time as _time

            started = float((record or {}).get("started_at") or 0.0)
            elapsed_s = max(0.0, _time.monotonic() - started) if started else 0.0
        except Exception:
            elapsed_s = 0.0

    estimated_tokens = _int_field(record, "estimated_tokens")
    new_tokens = _int_field(record, "new_tokens")
    eta_prefill_s = _float_field(record, "eta_prefill_s")
    slot_id = (record or {}).get("slot_id")
    llama_task_id = (record or {}).get("llama_task_id")
    endpoint = endpoint or (record or {}).get("endpoint")

    # LP-0MW959RRO008AAH9 (AC10b/AC10c): attribute the eviction to the backend
    # task this dispatch used. The id is resolved from the already-cached
    # ``/slots`` snapshot — no new upstream call — and re-resolved here rather
    # than trusted from record-creation time: the slot is usually still idle
    # when the record is created (the request has not reached llama-server yet),
    # while at release it is running this request. A record whose slot is
    # unknown keeps its explicit ``unknown`` markers (records created outside
    # the guarded routing path); a record with a **known** slot reports why the
    # id is unavailable instead of a bare ``unknown/unknown``.
    task_id_reason = None
    if slot_id is not None and llama_task_id is None:
        try:
            # Lazy import: this module is the single eviction sink (imported by
            # ``router_helpers``), so importing the observability snapshot
            # eagerly would couple the two import graphs. The resolver reads the
            # already-cached ``/slots`` snapshot only — never a new upstream
            # call (AC10b).
            from proxy import observability as _observability

            task_id, task_id_reason = _observability.slot_task_id_from_cache(slot_id)
        except Exception:
            task_id, task_id_reason = None, None
        if task_id is not None:
            llama_task_id = task_id
            task_id_reason = None
            # The association the line reports is persisted on the record, so
            # a later reader sees what this request was attributed to.
            if isinstance(record, dict):
                record["llama_task_id"] = task_id
                record["llama_task_id_reason"] = None
        elif isinstance(record, dict):
            record["llama_task_id_reason"] = task_id_reason

    _after_acceptance_counts[reason] = _after_acceptance_counts.get(reason, 0) + 1

    try:
        from proxy import metrics as _metrics

        _metrics.record_local_eviction_after_acceptance(reason)
    except Exception:
        pass

    logger = getattr(srv, "logger", None) or logging.getLogger("llama-proxy")
    # One stable, machine-parseable line (AC1/AC7). Unknown values are emitted
    # as ``unknown``/``0`` rather than omitted so the line is always parseable.
    # The association evidence is APPENDED (AC10e): ``llama_task_id_reason=``
    # only when a known slot could not be attributed, then any caller evidence,
    # so every pre-existing key keeps its position for deployed parsers.
    _suffix = _format_extra_fields(extra_fields)
    _reason_suffix = (
        f" llama_task_id_reason={task_id_reason}" if task_id_reason else ""
    )
    try:
        logger.error(
            "local_eviction_after_acceptance session=%s reason=%s "
            "estimated_tokens=%d new_tokens=%d eta_prefill_s=%s "
            "elapsed_s=%.1f slot_id=%s llama_task_id=%s endpoint=%s%s%s",
            session_id or "unknown",
            reason,
            estimated_tokens,
            new_tokens,
            "unknown" if eta_prefill_s is None else f"{eta_prefill_s:.1f}",
            elapsed_s or 0.0,
            "unknown" if slot_id is None else slot_id,
            "unknown" if llama_task_id is None else llama_task_id,
            endpoint or "default",
            _reason_suffix,
            _suffix,
        )
    except Exception:
        pass
    return True


def _format_extra_fields(extra_fields: Mapping[str, object] | None) -> str:
    """Render caller evidence as a trailing `` key=value`` run (never raises).

    ``None``/empty renders the empty string so lines emitted without extra
    evidence stay byte-identical to the pre-change format.
    """
    if not extra_fields:
        return ""
    parts: list[str] = []
    try:
        for key, value in extra_fields.items():
            if value is None:
                rendered = "unknown"
            elif isinstance(value, bool):
                rendered = "true" if value else "false"
            elif isinstance(value, int):
                rendered = str(value)
            elif isinstance(value, float):
                rendered = f"{value:.6g}"
            else:
                rendered = str(value)
            parts.append(f"{key}={rendered}")
    except Exception:
        return ""
    return " " + " ".join(parts)


def record_backend_restart_evictions(srv, *, restart_reason: str) -> int:
    """Record the dispatches a backend restart abandons (LP-0MVZVZLCS000LQNB AC3).

    A restart discards the in-flight prefills: the GPU work is gone and the
    prompt cache is cold again. Left unrecorded, that churn is indistinguishable
    from a capacity problem (the 2026-11-02 window measured 61 post-acceptance
    evictions across six backend restarts without a single one attributed to
    the restarts themselves). Every active dispatch that has not yet served its
    first byte is therefore recorded as an
    :data:`EVICTION_BACKEND_RESTART` eviction, carrying the reason the backend
    was restarted.

    Idempotent per dispatch record (the shared sink de-duplicates through the
    record's ``eviction_logged`` flag), so a restart path that races another
    release path cannot double-count. Never raises: a restart is already an
    exceptional path.

    Returns the number of evictions emitted.
    """
    records = getattr(srv, "local_dispatch_records", None)
    if not isinstance(records, dict):
        return 0
    try:
        snapshot = list(records.items())
    except Exception:
        return 0

    emitted = 0
    for session_id, record in snapshot:
        if not isinstance(record, dict):
            continue
        if not record.get("active"):
            continue
        if record.get("first_byte_seen"):
            continue
        try:
            recorded = record_eviction_after_acceptance(
                srv,
                session_id,
                EVICTION_BACKEND_RESTART,
                record,
                extra_fields={"restart_reason": restart_reason},
            )
        except Exception:
            continue
        if recorded:
            emitted += 1
    return emitted


def eviction_after_acceptance_counters() -> dict:
    """Monotonic counter snapshot: ``{"total": n, "by_reason": {...}}``."""
    return {
        "total": sum(_after_acceptance_counts.values()),
        "by_reason": dict(_after_acceptance_counts),
    }


def reset_eviction_after_acceptance_counters() -> None:
    """Clear the counters (test helper; production resets on process restart)."""
    _after_acceptance_counts.clear()


def _int_field(record: dict | None, key: str) -> int:
    try:
        return max(0, int((record or {}).get(key) or 0))
    except (TypeError, ValueError):
        return 0


def _float_field(record: dict | None, key: str) -> float | None:
    raw = (record or {}).get(key)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value == value else None  # reject NaN


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
