# <!-- REFACTOR-LP-0MW95FYMH006XYNK
# Parents: LP-0MW0KZUJP0019SLS (Serialise audits at the shared local resource).
# -->
"""Proxy-owned cross-host audit admission budget and live lease registry.

LP-0MW95FYMH006XYNK (parent epic LP-0MW0KZUJP0019SLS AC1): the audit cap and
the live audit-lease set live with the proxy — the one component every client
host talks to — instead of per-host ``~/.herdr/downtime/budget.json``. Two
ContextHub instances on different servers could each believe they were under
cap while together they oversubscribed the single local ``llama-server`` GPU;
a single proxy-side registry serialises audits across hosts by construction.

The gate is a **long-TTL audit-tier lease**:

- ``POST /llama/local/audit/lease`` acquires (or, for the same session,
  idempotently renews) a lease. At most ``audit_admission.max_audits``
  (default 1) live leases can be held at once; a second **different** audit
  session is refused with the stable token ``audit-budget-saturated`` plus a
  bounded ``retry_after_s`` hint — the audit skill waits and retries instead
  of competing with the running audit.
- ``POST /llama/local/audit/lease/renew`` extends the holder's lease.
- ``DELETE /llama/local/audit/lease/<session_id>`` releases it (clean exit).
- ``GET /llama/local/audit/availability`` publishes ``{cap, live, available,
  leases}`` with the vocabulary the herdr dispatcher logs, so the dispatcher's
  admission decision and the proxy's refusal can never disagree (parent AC3).

Design constraints (from the work item):

- **Proxy owns the cap.** ``server.audit_admission.max_audits`` (default 1)
  is authoritative. A client may still send a requested ``cap``, which is
  clamped to the proxy's configured maximum — the proxy never relies on a
  client-supplied cap.
- **Single mutation path.** Acquire/renew/release are the only way to change
  the live lease set. A refused acquire performs no partial state change.
- **Staleness and crash recovery.** A lease older than its TTL is reclaimed
  and the reclamation is logged (``audit_lease_reclaimed reason=expired``),
  so a crashed holder cannot wedge the budget. Clock comparisons use the
  proxy's own monotonic clock only (a skewed host cannot hold a stale lease).
- **Fail-safe.** An internal registry error never blocks every audit
  indefinitely: acquire returns a bounded ``retry_after_s`` with the stable
  token ``audit-budget-unavailable`` and the availability read reports
  ``error: registry_unavailable`` instead of asserting capacity; the audit
  skill's documented fallback is a bounded wait then legacy per-host
  behaviour.
- **Backwards compatible.** The existing per-host ``budget.json`` is no
  longer authoritative for audit admission; the proxy logs a WARNING at
  startup when it detects a conflicting client-side cap (see
  :func:`warn_on_conflicting_client_cap`) and the migration is documented in
  ``docs/routing.md``.

Audit-tier identification: audit sessions are named
``audit-<issue>-<context>-<uuid8>`` (the ``X-Session-Id`` prefix). The gate
records the session identity it used in every log line; see
:func:`is_audit_session`.
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("llama-proxy")

#: Default cap: at most one audit-tier session across ALL client hosts
#: (parent AC1; ``budget.json`` ``maxAudit`` default 1 preserved).
DEFAULT_MAX_AUDITS = 1
#: Default lease lifetime (seconds). Long on purpose: a whole audit
#: (multi-phase, dozens of Pi calls) can last tens of minutes, so a
#: per-request TTL would reclaim a live holder mid-run. The holder renews
#: periodically; a crashed holder loses the lease this long after its last
#: renew (bounded, never wedged).
DEFAULT_LEASE_TTL_SECONDS = 3600.0
#: Smallest and largest accepted TTL (seconds).
MIN_LEASE_TTL_SECONDS = 60.0
MAX_LEASE_TTL_SECONDS = 86400.0

#: Stable reason token for a refused acquire: the budget is saturated.
AUDIT_BUDGET_SATURATED = "audit-budget-saturated"
#: Stable reason token for an internal registry failure (fail-safe path).
AUDIT_BUDGET_UNAVAILABLE = "audit-budget-unavailable"
#: Stable error marker in the availability read when the registry is broken.
REGISTRY_UNAVAILABLE = "registry_unavailable"

#: Audit-session id marker (parent risk note): audit sessions are named
#: ``audit-<issue>-<context>-<uuid8>``.
AUDIT_SESSION_PREFIX = "audit-"

#: Bounded wait hint on the fail-safe path (seconds): the client falls back
#: to legacy per-host behaviour after waiting this long.
FAILSAFE_RETRY_AFTER_SECONDS = 10


@dataclass(frozen=True)
class AuditLease:
    """One held audit-admission lease."""

    lease_id: str
    session_id: str
    client_ip: str
    acquired_at: float = field(default_factory=time.monotonic)
    ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS
    renewed_at: float = field(default_factory=time.monotonic)

    @property
    def expires_at(self) -> float:
        """Monotonic expiry instant (proxy's clock only)."""
        return self.renewed_at + self.ttl_seconds

    def is_expired(self, now: float | None = None) -> bool:
        if now is None:
            now = time.monotonic()
        return now > self.expires_at


#: ``session_id -> AuditLease`` (one lease per audit session).
_leases: dict[str, AuditLease] = {}
#: Lifetime counters (observability).
_issued = 0
_reclaimed = 0
_released = 0
_refused_saturated = 0
_refused_unavailable = 0


def reset() -> None:
    """Clear all audit-budget state (test helper)."""
    global _issued, _reclaimed, _released, _refused_saturated, _refused_unavailable
    _leases.clear()
    _issued = _reclaimed = _released = _refused_saturated = _refused_unavailable = 0


def stats() -> dict[str, int]:
    """Audit-budget counters for the status payload (fail-open)."""
    return {
        "audit_leases_live": len(_leases),
        "audit_leases_issued": _issued,
        "audit_leases_reclaimed": _reclaimed,
        "audit_leases_released": _released,
        "audit_refused_saturated": _refused_saturated,
        "audit_refused_unavailable": _refused_unavailable,
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def audit_admission_config(config: dict | None) -> tuple[int, float]:
    """Resolve ``(max_audits, lease_ttl_seconds)`` from the proxy config.

    Reads ``server.audit_admission.max_audits`` (default 1) and
    ``server.audit_admission.lease_ttl_seconds`` (default 3600). A
    ``max_audits`` of 0 disables the gate entirely (mirrors the ContextHub
    ``budget.json`` ``maxAudit: 0`` kill switch, LP-0MW95FYMH006XYNK AC5).
    """
    server_cfg = config.get("server", {}) if isinstance(config, dict) else {}
    section = server_cfg.get("audit_admission", {}) if isinstance(server_cfg, dict) else {}
    if not isinstance(section, dict):
        section = {}
    try:
        cap = int(section.get("max_audits", DEFAULT_MAX_AUDITS))
    except (TypeError, ValueError):
        cap = DEFAULT_MAX_AUDITS
    if cap < 0:
        cap = DEFAULT_MAX_AUDITS
    try:
        ttl = float(section.get("lease_ttl_seconds", DEFAULT_LEASE_TTL_SECONDS))
    except (TypeError, ValueError):
        ttl = DEFAULT_LEASE_TTL_SECONDS
    ttl = max(MIN_LEASE_TTL_SECONDS, min(MAX_LEASE_TTL_SECONDS, ttl))
    return cap, ttl


def is_audit_session(session_id: str | None) -> bool:
    """True when *session_id* names an audit-tier session (``audit-`` prefix).

    The parent risk note requires the gate to log the identity it used: a
    non-audit session that happens to carry the ``audit-`` prefix would be
    misclassified, so every acquire/renew/release log line records whether the
    session actually matches the prefix (and the raw value).
    """
    return bool(session_id and str(session_id).startswith(AUDIT_SESSION_PREFIX))


# ---------------------------------------------------------------------------
# Registry internals
# ---------------------------------------------------------------------------

def _prune_expired(now: float | None = None) -> int:
    """Reclaim expired leases; log each reclamation.

    Returns the number of reclaimed leases. A crashed holder's lease expires
    here ``ttl`` after its last renew — never wedging the budget.
    """
    global _reclaimed
    now = time.monotonic() if now is None else now
    reclaimed: list[AuditLease] = [
        lease for lease in _leases.values() if now > lease.expires_at
    ]
    for lease in reclaimed:
        del _leases[lease.session_id]
        _reclaimed += 1
        logger.info(
            "audit_lease_reclaimed reason=expired lease_id=%s session=%s "
            "audit_tier=%s client_ip=%s age_s=%.1f",
            lease.lease_id,
            lease.session_id,
            is_audit_session(lease.session_id),
            lease.client_ip,
            now - lease.acquired_at,
        )
    return len(reclaimed)


def _new_lease_id() -> str:
    """Unpredictable lease id (never guessable)."""
    return secrets.token_hex(16)


def _lease_payload(lease: AuditLease, now: float) -> dict[str, Any]:
    """Wire payload for a single lease (the dispatcher's vocabulary).

    ``acquired_at`` / ``expires_at`` are published as ISO-8601 UTC so a
    dispatcher on another host can read them; the proxy's own expiry decisions
    still use its monotonic clock, and ``expires_in_s`` is the machine-readable
    countdown for the admission decision.
    """
    return {
        "session_id": lease.session_id,
        "client_ip": lease.client_ip,
        "acquired_at": _iso(lease.acquired_at),
        "expires_at": _iso(lease.expires_at),
        "expires_in_s": round(max(0.0, lease.expires_at - now), 1),
    }


def _iso(monotonic: float) -> str:
    """ISO-8601 UTC form of a monotonic instant (for the wire only).

    The proxy compares against the live monotonic clock; publishing a readable
    timestamp lets an operator/dispatcher attach wall-clock meaning to a lease
    without trusting its own clock for the *decision*.
    """
    import datetime as _datetime

    return _datetime.datetime.fromtimestamp(
        time.time() - (time.monotonic() - monotonic),
        tz=_datetime.UTC,
    ).isoformat()


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def availability(config: dict | None = None) -> dict[str, Any]:
    """Publish the audit budget: ``{cap, live, available, leases}``.

    The read never raises and never asserts capacity on an internal error: a
    broken registry reports ``error: registry_unavailable`` (the dispatcher's
    documented fallback is a bounded wait) instead of a fabricated capacity.
    """
    cap, _ttl = audit_admission_config(config)
    try:
        _prune_expired()
        live = len(_leases)
        leases = [
            _lease_payload(lease, time.monotonic())
            for lease in sorted(
                _leases.values(), key=lambda item: item.acquired_at
            )
        ]
        return {
            "cap": cap,
            "live": live,
            "available": max(0, cap - live),
            "leases": leases,
            "max_audits": cap,  # legacy alias for the dispatch-budget reader
        }
    except Exception as exc:  # pragma: no cover - defensive, never crash the read
        logger.error("audit_budget_unavailable read: %s", exc)
        return {
            "error": REGISTRY_UNAVAILABLE,
            "cap": None,
            "live": None,
            "available": None,
            "leases": [],
        }


def acquire(
    config: dict | None,
    session_id: str | None,
    client_ip: str,
    requested_cap: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """Acquire (or idempotently renew) an audit-tier lease for *session_id*.

    Returns ``(status_code, payload)``:

    - ``200`` — lease acquired (or the session's existing lease renewed).
    - ``429`` — the budget is saturated by a **different** session, with
      ``reason: audit-budget-saturated`` and a bounded ``retry_after_s``. No
      partial state change (AC3).
    - ``503`` — internal registry failure, fail-safe token
      ``audit-budget-unavailable`` with a bounded retry hint.
    """
    global _issued, _refused_saturated, _refused_unavailable
    cap, ttl = audit_admission_config(config)
    if cap <= 0:
        # Kill switch (max_audits: 0): refuse everything with the saturated
        # token; the operator chose to block audits entirely.
        _refused_saturated += 1
        return 429, {
            "admit": False,
            "reason": AUDIT_BUDGET_SATURATED,
            "cap": cap,
            "live": 0,
            "available": 0,
            "retry_after_s": None,
        }
    if not session_id:
        return 400, {"admit": False, "error": "session_id is required"}
    audit_tier = is_audit_session(session_id)
    try:
        _prune_expired()
        now = time.monotonic()

        # Idempotent re-acquire: the same holder re-acquiring is not a
        # competing audit — renew instead of refusing its own lease.
        existing = _leases.get(session_id)
        if existing is not None and not existing.is_expired(now):
            renewed = AuditLease(
                lease_id=existing.lease_id,
                session_id=session_id,
                client_ip=client_ip or existing.client_ip,
                acquired_at=existing.acquired_at,
                ttl_seconds=ttl,
                renewed_at=now,
            )
            _leases[session_id] = renewed
            logger.info(
                "audit_lease_renewed lease_id=%s session=%s audit_tier=%s "
                "client_ip=%s ttl_s=%.0f live=%d cap=%d",
                renewed.lease_id,
                session_id,
                audit_tier,
                client_ip,
                ttl,
                len(_leases),
                cap,
            )
            return 200, {
                "admit": True,
                "renewed": True,
                "lease_id": renewed.lease_id,
                "session_id": session_id,
                "cap": cap,
                "live": len(_leases),
                "available": max(0, cap - len(_leases)),
                "ttl_s": ttl,
                "expires_at": _iso(renewed.expires_at),
                "expires_in_s": round(max(0.0, renewed.expires_at - now), 1),
            }

        live = len(_leases)
        if live >= cap:
            # Clamp a requested cap (never honoured above the proxy's own).
            effective_cap = cap
            if requested_cap is not None:
                try:
                    requested_cap = int(requested_cap)
                except (TypeError, ValueError):
                    requested_cap = None
                if requested_cap is not None:
                    effective_cap = max(0, min(requested_cap, cap))
            if live >= max(1, effective_cap):
                _refused_saturated += 1
                logger.info(
                    "audit_lease_refused reason=%s session=%s audit_tier=%s "
                    "client_ip=%s cap=%d live=%d requested_cap=%s",
                    AUDIT_BUDGET_SATURATED,
                    session_id,
                    audit_tier,
                    client_ip,
                    cap,
                    live,
                    requested_cap,
                )
                return 429, {
                    "admit": False,
                    "reason": AUDIT_BUDGET_SATURATED,
                    "cap": cap,
                    "live": live,
                    "available": 0,
                    "retry_after_s": _retry_hint(ttl),
                }

        lease = AuditLease(
            lease_id=_new_lease_id(),
            session_id=session_id,
            client_ip=client_ip or "unknown",
            ttl_seconds=ttl,
        )
        _leases[session_id] = lease
        _issued += 1
        logger.info(
            "audit_lease_acquired lease_id=%s session=%s audit_tier=%s "
            "client_ip=%s ttl_s=%.0f live=%d cap=%d available=%d",
            lease.lease_id,
            session_id,
            audit_tier,
            client_ip,
            ttl,
            len(_leases),
            cap,
            max(0, cap - len(_leases)),
        )
        return 200, {
            "admit": True,
            "lease_id": lease.lease_id,
            "session_id": session_id,
            "cap": cap,
            "live": len(_leases),
            "available": max(0, cap - len(_leases)),
            "ttl_s": ttl,
            "expires_at": _iso(lease.expires_at),
            "expires_in_s": round(max(0.0, lease.expires_at - now), 1),
        }
    except Exception as exc:  # pragma: no cover - defensive, fail-safe
        _refused_unavailable += 1
        logger.error(
            "audit_budget_unavailable acquire session=%s audit_tier=%s: %s",
            session_id,
            audit_tier,
            exc,
        )
        return 503, {
            "admit": False,
            "reason": AUDIT_BUDGET_UNAVAILABLE,
            "retry_after_s": FAILSAFE_RETRY_AFTER_SECONDS,
        }


def renew(
    config: dict | None,
    session_id: str | None,
    client_ip: str = "unknown",
) -> tuple[int, dict[str, Any]]:
    """Extend *session_id*'s lease (no-op when no live lease exists).

    Returns ``200`` when renewed or when there is nothing to renew (the
    caller's next acquire starts fresh), ``503`` on internal error.
    """
    global _refused_unavailable
    cap, ttl = audit_admission_config(config)
    if not session_id:
        return 400, {"admit": False, "error": "session_id is required"}
    audit_tier = is_audit_session(session_id)
    try:
        _prune_expired()
        now = time.monotonic()
        existing = _leases.get(session_id)
        if existing is None or existing.is_expired(now):
            return 200, {"renewed": False, "live": len(_leases), "cap": cap}
        renewed = AuditLease(
            lease_id=existing.lease_id,
            session_id=session_id,
            client_ip=client_ip or existing.client_ip,
            acquired_at=existing.acquired_at,
            ttl_seconds=ttl,
            renewed_at=now,
        )
        _leases[session_id] = renewed
        logger.info(
            "audit_lease_renewed lease_id=%s session=%s audit_tier=%s "
            "client_ip=%s ttl_s=%.0f live=%d cap=%d",
            renewed.lease_id,
            session_id,
            audit_tier,
            client_ip,
            ttl,
            len(_leases),
            cap,
        )
        return 200, {
            "renewed": True,
            "lease_id": renewed.lease_id,
            "live": len(_leases),
            "cap": cap,
            "expires_at": _iso(renewed.expires_at),
            "expires_in_s": round(max(0.0, renewed.expires_at - now), 1),
        }
    except Exception as exc:  # pragma: no cover - defensive, fail-safe
        _refused_unavailable += 1
        logger.error(
            "audit_budget_unavailable renew session=%s audit_tier=%s: %s",
            session_id,
            audit_tier,
            exc,
        )
        return 503, {
            "admit": False,
            "reason": AUDIT_BUDGET_UNAVAILABLE,
            "retry_after_s": FAILSAFE_RETRY_AFTER_SECONDS,
        }


def release(session_id: str | None) -> tuple[int, dict[str, Any]]:
    """Release *session_id*'s lease (clean run exit).

    A missing lease is a reported no-op (``200``, ``released: false``) so the
    audit skill's release-after-crash path degrades gracefully.
    """
    global _released, _refused_unavailable
    if not session_id:
        return 400, {"released": False, "error": "session_id is required"}
    audit_tier = is_audit_session(session_id)
    try:
        lease = _leases.pop(session_id, None)
        if lease is None:
            return 200, {"released": False, "live": len(_leases)}
        _released += 1
        logger.info(
            "audit_lease_released lease_id=%s session=%s audit_tier=%s "
            "client_ip=%s live=%d",
            lease.lease_id,
            session_id,
            audit_tier,
            lease.client_ip,
            len(_leases),
        )
        return 200, {"released": True, "live": len(_leases)}
    except Exception as exc:  # pragma: no cover - defensive, fail-safe
        _refused_unavailable += 1
        logger.error(
            "audit_budget_unavailable release session=%s audit_tier=%s: %s",
            session_id,
            audit_tier,
            exc,
        )
        return 503, {
            "released": False,
            "reason": AUDIT_BUDGET_UNAVAILABLE,
        }


def _retry_hint(ttl_seconds: float) -> int:
    """Bounded retry hint for a saturated refusal.

    The refusal never blocks forever: the hint is capped well below the lease
    TTL, so a client's wait is bounded by how long a holder can hold the lease
    (a crashed holder is reclaimed within ``ttl`` by :func:`_prune_expired`).
    """
    return max(1, min(int(ttl_seconds), FAILSAFE_RETRY_AFTER_SECONDS * 60))


def warn_on_conflicting_client_cap(config: dict | None) -> None:
    """Log a WARNING when a client-side ``budget.json`` cap conflicts.

    AC5 (backwards compatible): the per-host ``~/.herdr/downtime/budget.json``
    ``maxAudit`` value is no longer authoritative for audit admission. When it
    exists and differs from the proxy's ``audit_admission.max_audits`` the
    startup logs a WARNING naming both values and which one wins. Best-effort:
    a missing/unreadable file (the common case — most proxy hosts do not run
    ContextHub) logs nothing and never blocks startup.
    """
    from pathlib import Path

    proxy_cap, _ttl = audit_admission_config(config)
    try:
        budget_path = Path.home() / ".herdr" / "downtime" / "budget.json"
        if not budget_path.exists():
            return
        import json

        budget = json.loads(budget_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except Exception:
        return
    if not isinstance(budget, dict):
        return
    client_cap = budget.get("maxAudit")
    if client_cap is None:
        return
    try:
        client_cap = int(client_cap)
    except (TypeError, ValueError):
        return
    if client_cap == proxy_cap:
        logger.info(
            "audit budget: client-side budget.json maxAudit=%d matches the "
            "proxy cap audit_admission.max_audits=%d",
            client_cap,
            proxy_cap,
        )
        return
    if proxy_cap == 0:
        logger.warning(
            "audit budget migration: client-side ~/.herdr/downtime/budget.json "
            "maxAudit=%d CONFLICTS with the proxy audit_admission.max_audits=%d "
            "(kill switch). The proxy cap is authoritative for audit admission; "
            "remove or update budget.json to stop this warning. See "
            "proxy/docs/routing.md (audit admission budget section).",
            client_cap,
            proxy_cap,
        )
        return
    logger.warning(
        "audit budget migration: client-side ~/.herdr/downtime/budget.json "
        "maxAudit=%d CONFLICTS with the proxy audit_admission.max_audits=%d. "
        "The proxy cap is authoritative for audit admission; update or remove "
        "budget.json to stop this warning. See proxy/docs/routing.md (audit "
        "admission budget section).",
        client_cap,
        proxy_cap,
    )
