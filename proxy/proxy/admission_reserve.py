
# <!-- REFACTOR-LP-0MVIR51KL0096QOB
# smell: unused_import
# severity: critical
# description: Local variable `_expired` referenced before assignment
# -->
"""Atomic local-admission reservations (LP-0MVIKPCSG002B0NL).

A client that wants to keep its work local can *claim* local capacity for a
specific request before sending it.  ``POST /llama/local/reserve`` evaluates
the request with the **same** occupancy-aware estimator the admission guard
uses (``prefill_eta.evaluate_admission``) and, when the projected wait fits the
caller's budget, returns a short-TTL, single-use ticket. A follow-up request
carrying ``X-Local-Reservation: <id>`` then dispatches without re-checking or
queueing.

Why a handshake: polling ``/llama/local/status`` and then sending is racy —
N clients can observe the same idle window and all queue behind the same
serialised prefill gate (queue depth reached 7 while the status payload still
reported 2 available slots). A reservation is a serialised decision, i.e. a
gate ticket, taken *before* the work is sent.

Design constraints (from the work item):

- The verdict comes from ``prefill_eta`` — this module never duplicates ETA
  maths. ``busy_slots`` / ``in_flight_prefills`` are supplied by the caller so
  the reserve verdict and the guard verdict are identical for identical inputs.
- Fail-open: an inert (uncalibrated) guard admits. A refusal requires a
  calibrated verdict against a real budget.
- Tickets are TTL-bound (default 10 s, capped at 60 s), single-use and
  cancellable. A stale/missing ticket falls back to the ordinary path — never
  a hard failure.
- Ticket IDs are unpredictable (24 random bytes).
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from proxy.prefill_eta import (
    evaluate_admission,
    suggested_retry_after_seconds,
)

logger = logging.getLogger("llama-proxy")

#: Default and maximum ticket lifetime (seconds).
DEFAULT_RESERVE_TTL_SECONDS = 10
MAX_RESERVE_TTL_SECONDS = 60

#: ``X-Local-Preference`` values (AC2 of the work item).
PREF_LOCAL_ONLY = "local-only"
PREF_PREFER_LOCAL = "prefer-local"
PREF_REMOTE_OK = "remote-ok"
VALID_PREFERENCES = (PREF_LOCAL_ONLY, PREF_PREFER_LOCAL, PREF_REMOTE_OK)
DEFAULT_PREFERENCE = PREF_PREFER_LOCAL

HEADER_PREFERENCE = "x-local-preference"
HEADER_RESERVATION = "x-local-reservation"
HEADER_ROUTE = "X-Local-Route"
HEADER_FALLBACK_REASON = "X-Local-Fallback-Reason"


@dataclass(frozen=True)
class Reservation:
    """One claimed local-admission ticket."""

    ticket_id: str
    estimated_tokens: int
    cached_ratio: float
    priority: str | None
    provider: str | None
    session_id: str | None
    created_at: float = field(default_factory=time.monotonic)
    ttl_seconds: float = DEFAULT_RESERVE_TTL_SECONDS

    @property
    def expires_at(self) -> float:
        return self.created_at + self.ttl_seconds

    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at


#: ``ticket_id -> Reservation``
_reservations: dict[str, Reservation] = {}
#: ``session_id -> ticket_id`` of that session's most recent claimed ticket.
_session_reservation: dict[str, str] = {}
#: Observed-but-unused tickets (AC monitoring: "reserved-but-unused").
_issued = 0
_used = 0
_cancelled = 0
_expired = 0


def reset() -> None:
    """Clear all reservation state (test helper)."""
    global _issued, _used, _cancelled, _expired
    _reservations.clear()
    _session_reservation.clear()
    _issued = _used = _cancelled = _expired = 0


def stats() -> dict[str, int]:
    """Reservation counters for the status payload / monitoring."""
    return {
        "reserved_outstanding": len(_reservations),
        "reserved_issued": _issued,
        "reserved_used": _used,
        "reserved_cancelled": _cancelled,
        "reserved_expired": _expired,
    }


def _prune_expired(now: float | None = None) -> None:
    """Drop expired tickets (and their session pointers)."""
    global _expired
    now = time.monotonic() if now is None else now
    for ticket_id, reservation in list(_reservations.items()):
        if now > reservation.expires_at:
            del _reservations[ticket_id]
            _expired += 1
            sid = reservation.session_id
            if sid and _session_reservation.get(sid) == ticket_id:
                del _session_reservation[sid]


def _new_ticket_id() -> str:
    """Unpredictable ticket ID (security constraint: never guessable)."""
    return secrets.token_hex(24)


# ---------------------------------------------------------------------------
# POST /llama/local/reserve
# ---------------------------------------------------------------------------

def reserve(
    body: dict[str, Any],
    config: dict,
    *,
    model: str | None = None,
    busy_slots: int = 0,
    in_flight_prefills: int = 0,
    occupant_prefill_seconds: float | None = None,
) -> tuple[int, dict[str, Any]]:
    """Evaluate and (unless refused / dry-run) claim a reservation ticket.

    *busy_slots* / *in_flight_prefills* must come from the caller's occupancy
    probe so the verdict is identical to the admission guard's (AC7).
    *occupant_prefill_seconds* is the same measured remaining occupant service
    the guard prices (LP-0MVOFVCA2008DQVE) so the two verdicts cannot drift
    for identical live state.

    Returns ``(status_code, payload)``.
    """
    _prune_expired()
    if not isinstance(body, dict):
        return 400, {"error": "invalid body", "admit": False}

    raw_tokens = body.get("estimated_tokens")
    if raw_tokens is None:
        return 400, {"error": "estimated_tokens is required", "admit": False}
    try:
        estimated_tokens = int(raw_tokens)
    except (TypeError, ValueError):
        return 400, {
            "error": "estimated_tokens must be a positive integer",
            "admit": False,
        }
    if estimated_tokens <= 0:
        return 400, {
            "error": "estimated_tokens must be a positive integer",
            "admit": False,
        }

    try:
        cached_ratio = min(1.0, max(0.0, float(body.get("cached_ratio") or 0.0)))
    except (TypeError, ValueError):
        cached_ratio = 0.0
    session_id = body.get("session_id") or None
    # Live-prefix agreement with the admission guard (LP-0MVN1IQTF0037WTG
    # AC1/AC3): when the residency view carries a *measured* reusable prefix
    # for this session it wins over the caller's prediction, so
    # ``POST /llama/local/reserve`` and ``_should_skip_local`` price the same
    # prefill from the same facts. Fail-open: no fresh view / no measurement
    # leaves today's body-supplied ratio untouched.
    cached_ratio, cached_ratio_source = _resolve_cached_ratio(
        cached_ratio, session_id, estimated_tokens
    )
    priority = body.get("priority") or None
    try:
        ttl = int(body.get("ttl_seconds", DEFAULT_RESERVE_TTL_SECONDS))
    except (TypeError, ValueError):
        ttl = DEFAULT_RESERVE_TTL_SECONDS
    ttl = max(1, min(MAX_RESERVE_TTL_SECONDS, ttl))
    dry_run = body.get("dry_run") is True

    decision = evaluate_admission(
        config,
        model,
        estimated_tokens,
        cached_ratio,
        priority,
        busy_slots=busy_slots,
        in_flight_prefills=in_flight_prefills,
        occupant_prefill_seconds=occupant_prefill_seconds,
    )

    projected = (
        decision.projected_wait_seconds
        if decision.projected_wait_seconds is not None
        else float(decision.eta_prefill_seconds or 0.0)
    )

    if not decision.admit:
        retry_after = suggested_retry_after_seconds(
            decision.eta_prefill_seconds, decision.budget_seconds
        )
        logger.info(
            "local_admission_reserve admit=false reason=%s projected_wait_s=%.1f "
            "budget_s=%.1f retry_after_s=%d busy_slots=%d session=%s",
            decision.reason,
            projected,
            float(decision.budget_seconds or 0.0),
            retry_after,
            decision.busy_slots,
            session_id or "unknown",
        )
        return 429, {
            "admit": False,
            "reason": decision.reason,
            "queue_eta_s": round(projected, 1),
            "budget_s": round(float(decision.budget_seconds or 0.0), 1),
            "retry_after_s": retry_after,
            "cached_ratio": round(cached_ratio, 4),
            "cached_ratio_source": cached_ratio_source,
        }

    if dry_run:
        return 200, {
            "admit": True,
            "dry_run": True,
            "reason": decision.reason,
            "projected_wait_s": round(projected, 1),
            "budget_s": (
                round(float(decision.budget_seconds or 0.0), 1)
                if decision.budget_seconds is not None
                else None
            ),
            "busy_slots": decision.busy_slots,
            "cached_ratio": round(cached_ratio, 4),
            "cached_ratio_source": cached_ratio_source,
        }

    ticket_id = _new_ticket_id()
    reservation = Reservation(
        ticket_id=ticket_id,
        estimated_tokens=estimated_tokens,
        cached_ratio=cached_ratio,
        priority=priority,
        provider=model,
        session_id=session_id,
        ttl_seconds=float(ttl),
    )
    _reservations[ticket_id] = reservation
    if session_id:
        _session_reservation[session_id] = ticket_id
    global _issued
    _issued += 1
    logger.info(
        "local_admission_reserve admit=true ticket=%s projected_wait_s=%.1f "
        "ttl_s=%d tokens=%d busy_slots=%d session=%s",
        ticket_id,
        projected,
        ttl,
        estimated_tokens,
        decision.busy_slots,
        session_id or "unknown",
    )
    return 200, {
        "admit": True,
        "reservation_id": ticket_id,
        "projected_wait_s": round(projected, 1),
        "ttl_s": ttl,
        "cached_ratio": round(cached_ratio, 4),
        "cached_ratio_source": cached_ratio_source,
    }


def _resolve_cached_ratio(
    body_ratio: float, session_id: str | None, estimated_tokens: int
) -> tuple[float, str]:
    """Price the reusable prefix from the live residency view when possible.

    Returns ``(cached_ratio, source)`` where *source* is ``"live"`` when the
    cached ``/slots`` snapshot measured a reusable prefix for *session_id*,
    else ``"body"`` (the caller's own ``cached_ratio``) or ``"none"`` when
    the body carried no ratio at all.

    Fail-open by construction: an unknown/stale view, a session that owns no
    slot, or a slot that did not report ``context_tokens`` all leave the
    caller's ratio in place (LP-0MVN1IQTF0037WTG AC1 fallback ladder).
    """
    if not session_id or estimated_tokens <= 0:
        return body_ratio, ("body" if body_ratio > 0.0 else "none")
    try:
        from proxy.residency import get_residency_view

        view = get_residency_view()
        if not view.known:
            return body_ratio, ("body" if body_ratio > 0.0 else "none")
        context_tokens = view.context_tokens_for_session(session_id)
    except Exception:
        return body_ratio, ("body" if body_ratio > 0.0 else "none")
    if context_tokens is None:
        return body_ratio, ("body" if body_ratio > 0.0 else "none")
    reusable = min(int(context_tokens), int(estimated_tokens))
    return min(1.0, max(0.0, reusable / float(estimated_tokens))), "live"


# ---------------------------------------------------------------------------
# Ticket lookup (follow-up request) and cancellation
# ---------------------------------------------------------------------------

def peek(ticket_id: str | None) -> Reservation | None:
    """Non-consuming lookup of *ticket_id* (expired tickets pruned).

    Lets the caller decide whether a ticket belongs to this gate before
    single-use consumption. Returns the reservation or ``None``.
    """
    if not ticket_id:
        return None
    _prune_expired()
    reservation = _reservations.get(ticket_id)
    if reservation is None:
        return None
    if reservation.is_expired():
        return None
    return reservation


def claim(ticket_id: str | None) -> Reservation | None:
    """Consume *ticket_id* (single-use).

    Returns the reservation when it exists and has not expired, removing it
    from the store; otherwise ``None`` (the caller falls back to today's
    ordinary path — never a hard failure).
    """
    global _used, _expired
    if not ticket_id:
        return None
    _prune_expired()
    reservation = _reservations.pop(ticket_id, None)
    if reservation is None:
        return None
    if reservation.is_expired():
        _expired += 1
        return None
    _used += 1
    sid = reservation.session_id
    if sid and _session_reservation.get(sid) == ticket_id:
        del _session_reservation[sid]
    return reservation


def cancel(ticket_id: str) -> tuple[int, dict[str, Any]]:
    """Cancel an outstanding ticket, releasing its gate claim."""
    global _cancelled
    _prune_expired()
    reservation = _reservations.pop(ticket_id, None)
    if reservation is None:
        return 404, {
            "error": "reservation_not_found",
            "reservation_id": ticket_id,
            "cancelled": False,
        }
    _cancelled += 1
    sid = reservation.session_id
    if sid and _session_reservation.get(sid) == ticket_id:
        del _session_reservation[sid]
    logger.info(
        "local_admission_reserve cancelled ticket=%s session=%s",
        ticket_id,
        sid or "unknown",
    )
    return 200, {"cancelled": True, "reservation_id": ticket_id}


# ---------------------------------------------------------------------------
# X-Local-Preference
# ---------------------------------------------------------------------------

def reservation_id_from_headers(headers: Any) -> str | None:
    """Return the ``X-Local-Reservation`` value (case-tolerant).

    Starlette's ``Headers`` is case-insensitive; plain dicts (tests, direct
    callers) are not, so both spellings are accepted.
    """
    if not headers:
        return None
    for key in (HEADER_RESERVATION, "X-Local-Reservation"):
        try:
            value = headers.get(key)
        except Exception:
            value = None
        if value:
            return str(value).strip() or None
    return None


def parse_preference(headers: Any) -> str:
    """Parse ``X-Local-Preference`` (case-insensitive header names).

    Missing or unknown values fall back to ``prefer-local`` — today's
    behaviour — so the header is purely additive (AC8).
    """
    if not headers:
        return DEFAULT_PREFERENCE
    raw = None
    try:
        raw = headers.get(HEADER_PREFERENCE)
        if raw is None:
            raw = headers.get("X-Local-Preference")
    except Exception:
        raw = None
    if not raw:
        return DEFAULT_PREFERENCE
    value = str(raw).strip().lower()
    if value in VALID_PREFERENCES:
        return value
    logger.debug(
        "unknown X-Local-Preference=%r; using %r", raw, DEFAULT_PREFERENCE
    )
    return DEFAULT_PREFERENCE


def headers_for_route(route: str, reason: str | None = None) -> dict[str, str]:
    """Response headers describing where the request was served."""
    headers = {HEADER_ROUTE: route}
    if route in ("remote", "refused"):
        headers[HEADER_FALLBACK_REASON] = reason or "unknown"
    return headers


# ---------------------------------------------------------------------------
# Response stamping (single choke point for AC6)
# ---------------------------------------------------------------------------

def _provider_is_local(model_config: dict, provider_name: str | None) -> bool:
    if not provider_name:
        return False
    for provider in (model_config or {}).get("providers") or []:
        if isinstance(provider, dict) and provider.get("name") == provider_name:
            return provider.get("type") == "local"
    return False


def stamp_route_headers(
    response: Any,
    model_config: dict,
    *,
    fallback_reason: str | None = None,
) -> Any:
    """Add ``X-Local-Route`` / ``X-Local-Fallback-Reason`` to *response*.

    Applied at the top-level entry point so every path (local success, remote
    success, refusal, exhaustion) is covered. A response that already carries
    ``X-Local-Route`` (e.g. a local-only refusal) is left untouched.
    """
    try:
        headers = getattr(response, "headers", None)
        if headers is None:
            return response
        if headers.get(HEADER_ROUTE):
            return response
        served_by = headers.get("X-Provider")
        if _provider_is_local(model_config, served_by):
            headers[HEADER_ROUTE] = "local"
        else:
            headers[HEADER_ROUTE] = "remote"
            headers[HEADER_FALLBACK_REASON] = fallback_reason or "no_local_provider"
    except Exception:  # pragma: no cover - headers are best-effort
        logger.debug("stamp_route_headers failed", exc_info=True)
    return response
