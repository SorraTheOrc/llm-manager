
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
    AdmissionDecision,
    evaluate_admission,
    suggested_retry_after_seconds,
)

logger = logging.getLogger("llama-proxy")

#: Default and maximum ticket lifetime (seconds).
DEFAULT_RESERVE_TTL_SECONDS = 10
MAX_RESERVE_TTL_SECONDS = 60

#: Canonical ``X-Local-Preference`` values (AC2 of the work item).
#:
#: ``local-preferred`` is the **canonical pin-local value**: the client asks
#: that its request never be moved to a remote provider while local can serve
#: it (the deferral/timing semantics live outside this vocabulary module).
#: ``prefer-local`` (the default) may fall through; ``remote-ok`` keeps the
#: local route and never waits for it.
#:
#: ``remote-ok`` is a **historic name** (LP-0MVUH9OYZ007GMWF AC1): it means
#: *"local allowed, remote acceptable"* — the local entry stays in the chain
#: and is attempted exactly as a headerless request attempts it, and a request
#: that cannot be served locally *right now* steps straight to the next
#: provider without queueing. It does **not** mean "remote only"; no
#: preference strips the local route from a chain any more (only a
#: ``CAPACITY_TOO_LARGE`` verdict does).
PREF_LOCAL_PREFERRED = "local-preferred"
PREF_PREFER_LOCAL = "prefer-local"
PREF_REMOTE_OK = "remote-ok"

#: Deployed wire value accepted as a **deprecated alias** of
#: :data:`PREF_LOCAL_PREFERRED` (LP-0MVSHWYVE005MS16). It stays a first-class
#: constant — never re-spelled at a call site — so the canonical set and the
#: alias table cannot drift, and so an un-migrated client is normalised rather
#: than demoted to ``prefer-local`` (which would let its request go remote).
PREF_LOCAL_ONLY = "local-only"

#: ``alias -> canonical value``. :func:`parse_preference` normalises through
#: this table and logs one deprecation warning per request.
LEGACY_PREFERENCE_ALIASES = {PREF_LOCAL_ONLY: PREF_LOCAL_PREFERRED}

VALID_PREFERENCES = (PREF_LOCAL_PREFERRED, PREF_PREFER_LOCAL, PREF_REMOTE_OK)
DEFAULT_PREFERENCE = PREF_PREFER_LOCAL

HEADER_PREFERENCE = "x-local-preference"
HEADER_RESERVATION = "x-local-reservation"
HEADER_ROUTE = "X-Local-Route"
HEADER_FALLBACK_REASON = "X-Local-Fallback-Reason"

# ---------------------------------------------------------------------------
# Canonical local-route-exit reason tokens (LP-0MVRMC1W6003MV3Y AC1)
#
# One table is the single source of truth for the ``X-Local-Fallback-Reason``
# header, the ``reason=`` fields of the routing log lines and the refusal body
# ``reason``: an exit that records anything else (or a new exit that records
# nothing) fails loudly through :func:`canonical_local_fallback_reason`
# instead of silently drifting between the three observability surfaces.
#
# The 2026-10-25 incident was exactly that drift: chain-resolution skips
# (cooldown / quota / usage-limit reset / ``available_times`` window) and the
# lease-active ``continue`` left no reason on the request, so a pin-local
# refusal (the header was then spelled ``local-only``) reported the generic
# ``local_route_unavailable`` instead of why local was unavailable.
# ---------------------------------------------------------------------------

#: Tokens a local-route exit may stamp on a request. Chain-resolution skips
#: record the *local* token (a remote entry's cooldown must never mint a local
#: fallback reason); the lease token names expected local contention; the last
#: two are the structural refusals used when there was no local route at all
#: or an exit left no reason behind.
LOCAL_FALLBACK_REASON_TOKENS: frozenset[str] = frozenset(
    {
        # Chain resolution (``_resolve_provider_with_exclusions``).
        "local_provider_cooldown",
        "quota_hard_stop",
        "usage_limit_reset_pending",
        "outside_available_window",
        # Lease-active continue: expected contention, not a provider failure.
        "local_lease_active",
        # Structural refusals.
        "local_only_no_local_route",
        "local_route_unavailable",
    }
)


def canonical_local_fallback_reason(token: str) -> str:
    """Return *token* when it belongs to the canonical reason-token table.

    Raises:
        ValueError: *token* is not a canonical reason token. Callers that add
            a new local-route exit must canonicalise its token in
            :data:`LOCAL_FALLBACK_REASON_TOKENS` first, so the header, the
            ``reason=`` log field and the refusal body can never disagree.
    """
    if token not in LOCAL_FALLBACK_REASON_TOKENS:
        raise ValueError(
            f"unknown local-fallback reason token {token!r}; add it to "
            "admission_reserve.LOCAL_FALLBACK_REASON_TOKENS to canonicalise it"
        )
    return token


# ---------------------------------------------------------------------------
# Local-capacity verdict: ``deferrable`` vs ``too_large`` (LP-0MVSHX47F003XEIZ)
#
# One authoritative classification of *why* the local route cannot serve a
# request right now, so chain selection can distinguish "wait, a slot is
# coming" from "this prompt will never fit local".
#
# **Single-implementation contract.** Both dispatch cycles and the pin-local
# refusal builder call :func:`local_capacity_verdict`; ``provider.py`` must
# never re-derive this split. A second copy in the dispatcher would be a
# defect, not a local optimisation.
#
# The inputs are *already computed* by the caller — the
# :class:`proxy.prefill_eta.AdmissionDecision` and/or the raw values below.
# The helper never re-prices a prompt: no ETA maths, no token estimation and
# no second sizing model (parent assumption A1). The reason tokens themselves
# stay owned by ``prefill_eta`` and ``provider``; this module only classifies
# them (naming them locally would otherwise need a ``provider`` import, which
# would create an import cycle).
# ---------------------------------------------------------------------------

#: Verdict vocabulary — exactly these two values.
CAPACITY_DEFERRABLE = "deferrable"
CAPACITY_TOO_LARGE = "too_large"

CAPACITY_VERDICTS: frozenset[str] = frozenset(
    {CAPACITY_DEFERRABLE, CAPACITY_TOO_LARGE}
)

#: Reason tokens that describe a property of the request size: waiting cannot
#: help, so the caller may compact or fall through to a remote provider.
CAPACITY_TOO_LARGE_REASONS: frozenset[str] = frozenset(
    {
        # ``prefill_eta`` cold-prefill cap (``new_tokens > budget_tokens``).
        "prefill_budget_exceeded",
        # Local context/size skip (router/priority and provider hard cap).
        "context_too_large",
    }
)


@dataclass(frozen=True)
class CapacityVerdict:
    """Why the local route cannot serve a request right now.

    Attributes:
        verdict: :data:`CAPACITY_DEFERRABLE` when the blocking reason is
            temporal (waiting can help, stay pinned to local) or
            :data:`CAPACITY_TOO_LARGE` when the request exceeds the local
            serving envelope (waiting cannot help).
        reason: The underlying reason token, preserved verbatim so the
            refusal body and log line can report *why*.
    """

    verdict: str
    reason: str | None = None


def _as_int(value: Any) -> int | None:
    """Return *value* as an ``int``, or ``None`` when it is not numeric."""
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _exceeds(value: Any, limit: Any) -> bool:
    """True when *value* is greater than an enabled (positive) *limit*.

    A missing value or a non-positive limit means the bound is disabled
    (``local_cold_prefill_budget_tokens: 0`` / unset context limit), so the
    comparison cannot fire and the verdict stays fail-safe.
    """
    value_int = _as_int(value)
    limit_int = _as_int(limit)
    if value_int is None or limit_int is None or limit_int <= 0:
        return False
    return value_int > limit_int


def local_capacity_verdict(
    decision: AdmissionDecision | None = None,
    *,
    reason: str | None = None,
    new_tokens: int | None = None,
    budget_tokens: int | None = None,
    estimated_tokens: int | None = None,
    context_limit: int | None = None,
) -> CapacityVerdict:
    """Classify why the local route cannot serve this request right now.

    Classification rule table:

    =================== =====================================================
    Verdict             Trigger
    =================== =====================================================
    ``too_large``       The reason is ``prefill_budget_exceeded`` (the guard's
                        cold-prefill cap) or ``context_too_large`` (the local
                        context/size skip).
    ``too_large``       ``new_tokens > budget_tokens`` with ``budget_tokens >
                        0`` — the raw form of the cold-prefill cap.
    ``too_large``       ``estimated_tokens > context_limit`` with
                        ``context_limit > 0`` — the raw form of the size skip.
    ``deferrable``      A temporal reason: ``eta_exceeds_budget``,
                        ``eta_exceeds_budget_direct``,
                        ``prefill_eta_exceeds_deadline``,
                        ``prefill_eta_exceeds_safety_budget``,
                        ``local_lease_active``, ``local_provider_cooldown``,
                        occupancy/busy-slot (``slot_exhaustion``), queue-expiry
                        (``fallback_after_queue``), ``model_loading`` or
                        stall/cooldown (``prefill_thrash_cooldown``). Waiting
                        can help, so the request stays pinned to local.
    ``deferrable``      Fail safe: an unknown or absent reason, or missing
                        inputs. Never promotes a fits-local request.
    =================== =====================================================

    Args:
        decision: The guard decision already computed by
            :func:`proxy.prefill_eta.evaluate_admission` (read-only input).
        reason: A reason token for exits that produced no guard decision
            (non-guard temporal exits and routing/context skips). Overrides
            ``decision.reason`` when supplied.
        new_tokens: New (uncached) tokens priced by the caller. Overrides
            ``decision.new_tokens`` when supplied.
        budget_tokens: The configured cold-prefill budget
            (``server.local_cold_prefill_budget_tokens``). Overrides
            ``decision.budget_tokens`` when supplied.
        estimated_tokens: The caller's prompt estimate for the local context
            bound (the same number the size skip compared).
        context_limit: The local model's context limit; ``0``/``None``
            disables the bound.

    Returns:
        :class:`CapacityVerdict` carrying the verdict and the reason token
        (verbatim, or ``None`` when no reason was supplied).
    """
    resolved_reason = reason
    if resolved_reason is None and decision is not None:
        resolved_reason = decision.reason

    resolved_new_tokens = new_tokens
    if resolved_new_tokens is None and decision is not None:
        resolved_new_tokens = decision.new_tokens

    resolved_budget_tokens = budget_tokens
    if resolved_budget_tokens is None and decision is not None:
        resolved_budget_tokens = decision.budget_tokens

    too_large = (
        resolved_reason in CAPACITY_TOO_LARGE_REASONS
        or _exceeds(resolved_new_tokens, resolved_budget_tokens)
        or _exceeds(estimated_tokens, context_limit)
    )

    return CapacityVerdict(
        verdict=CAPACITY_TOO_LARGE if too_large else CAPACITY_DEFERRABLE,
        reason=resolved_reason,
    )


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

    The legacy wire value ``local-only`` is normalised to the canonical
    ``local-preferred`` through :data:`LEGACY_PREFERENCE_ALIASES`, so a
    deployed client keeps pin-local semantics instead of being demoted to
    ``prefer-local`` (LP-0MVSHWYVE005MS16 AC1/AC5). Each normalisation logs one
    ``WARNING`` naming both values; because both call sites cache the parsed
    value on the request (``request._local_preference``), that is once per
    request, not once per routing decision (AC2).
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
    requested = str(raw).strip().lower()
    value = LEGACY_PREFERENCE_ALIASES.get(requested, requested)
    if value in VALID_PREFERENCES:
        if value != requested:
            logger.warning(
                "deprecated X-Local-Preference=%r; treating it as %r — clients "
                "should send %r",
                requested,
                value,
                PREF_LOCAL_PREFERRED,
            )
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
    ``X-Local-Route`` (e.g. a pin-local refusal) is left untouched.
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
