"""Per-request priority routing for the LLM proxy (LP-0MVDT94F0003VHUA).

The proxy no longer has a global fast/cheap operating mode. Instead, every
request carries a **priority** (``X-Priority: low|medium|high``, default
``medium``) and the per-request routing knobs — provider-chain selection,
contention-queue entry, economic-bypass recovery — are keyed to the request's
**effective priority**.

Effective priority
------------------

A request's effective priority is its header value, unless the proxy is inside
the **cheap window** (``server.priority_window``, derived from the retired
mode schedule; default local time 01:00–10:00). During the cheap window every
request routes as ``low`` regardless of its header — the header itself is
never modified and no 503/Retry-After deferral is applied (AC1).

Priority semantics (AC3)
------------------------

- ``high`` — never queue; prefer immediate remote fallback when local slots
  are busy; economic cold-cache bypass routes remote (fast-mode semantics).
- ``medium`` — today's behaviour: config-driven queue policy and
  economic-bypass recovery from the merged base config (parity baseline).
- ``low`` — prefer local and queue (bounded by the configured contention-queue
  caps), then the free tier, over paid remote fallback; economic cold-cache
  bypass is recovered locally (cheap-mode semantics in both windows).
  **Off-peak exception (LP-0MVF17VP70053JXX):** after local and the free tier
  are exhausted, the ``low`` chain continues into the ``opencode-go`` entries
  while they are inside their ``available_times`` (UTC) windows — see
  :func:`priority_routing_profile` for the full rationale and caveats.

Provider chains (AC2)
---------------------

Each model config may declare ``priority_routes`` — a map from
``low|medium|high`` to an ordered provider list (either full provider config
dicts or provider **names** resolved against the model's ``providers`` list).
The chain used for a request is selected by its effective priority; a missing
entry (or a config without ``priority_routes``) falls back to the model's
``providers`` list — the existing base config is therefore 100% backward
compatible (``medium`` semantics).

Observability (AC6)
-------------------

Every request is stamped with its effective priority:
``request.state.priority``; decision points log ``priority=<value>`` and the
top-level endpoint adds the ``X-Priority-Applied`` response header.
"""

import logging
from datetime import datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any

logger = logging.getLogger("llama-proxy")

# ---------------------------------------------------------------------------
# Priority vocabulary
# ---------------------------------------------------------------------------

PRIORITY_LOW = "low"
PRIORITY_MEDIUM = "medium"
PRIORITY_HIGH = "high"
VALID_PRIORITIES = (PRIORITY_LOW, PRIORITY_MEDIUM, PRIORITY_HIGH)
PRIORITY_RANK = {PRIORITY_LOW: 0, PRIORITY_MEDIUM: 1, PRIORITY_HIGH: 2}
DEFAULT_PRIORITY = PRIORITY_MEDIUM

# Request header carrying the client-declared priority (in v1 there is no
# trust gate per decision d in the intake: any reachable client may claim
# ``high``; hardening is tracked separately in LP-0MVDTE0CI0088QOR).
PRIORITY_HEADER = "X-Priority"
# Response header recording the effective (post cheap-window cap) priority
# for observability (AC6).
PRIORITY_APPLIED_HEADER = "X-Priority-Applied"

# ---------------------------------------------------------------------------
# Cheap window (default derived from the retired mode schedule 01:00–10:00)
# ---------------------------------------------------------------------------

DEFAULT_CHEAP_WINDOW_START = "01:00"
DEFAULT_CHEAP_WINDOW_END = "10:00"


def proxy_dir() -> Path:
    """Return the proxy directory (parent of the ``proxy`` package)."""
    return Path(__file__).resolve().parent.parent


def parse_priority(value: Any) -> str:
    """Parse a raw ``X-Priority`` header value into a valid priority.

    Absent, empty or invalid values default to ``medium``. Whitespace and
    case are tolerated (``HIGH`` → ``high``); any value outside
    ``low|medium|high`` logs a warning and resolves to ``medium`` (AC1).
    """
    if value is None:
        return DEFAULT_PRIORITY
    text = str(value).strip().lower()
    if not text:
        return DEFAULT_PRIORITY
    if text not in VALID_PRIORITIES:
        logger.warning(
            "Invalid X-Priority value %r; defaulting to %s", value, DEFAULT_PRIORITY
        )
        return DEFAULT_PRIORITY
    return text


def is_valid_priority(value: Any) -> bool:
    """True when *value* is one of ``low`` / ``medium`` / ``high``."""
    return str(value).strip().lower() in VALID_PRIORITIES


def _parse_hhmm(text: str) -> dt_time | None:
    """Parse an ``HH:MM`` string into a time, or None when invalid."""
    parts = str(text).strip().split(":")
    if len(parts) != 2:
        return None
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return dt_time(hour, minute)


def cheap_window_config(server_config: dict | None = None) -> dict:
    """Resolve the ``server.priority_window`` config with defaults.

    Key names mirror the retired ``server.mode_switch_drain`` style:

    - ``enabled`` (default True) — master switch; ``enabled: false`` disables
      the cheap-window low-capping entirely (every request keeps its header
      priority).
    - ``start`` / ``end`` (default ``01:00`` / ``10:00``, local time) — the
      half-open window during which every request routes as ``low``. A window
      where ``start > end`` wraps across midnight. Invalid times are skipped
      with a warning and reset to the defaults (a broken window never crashes
      routing and never blocks requests).

    Returns:
        A dict with ``enabled`` (bool), ``start`` (dt_time), ``end`` (dt_time).
    """
    section = {}
    if isinstance(server_config, dict):
        raw = server_config.get("priority_window")
        if isinstance(raw, dict):
            section = raw
    enabled = bool(section.get("enabled", True))
    start = _parse_hhmm(str(section.get("start", DEFAULT_CHEAP_WINDOW_START)))
    end = _parse_hhmm(str(section.get("end", DEFAULT_CHEAP_WINDOW_END)))
    if start is None:
        logger.warning(
            "priority_window: invalid start=%r; using default %s",
            section.get("start"),
            DEFAULT_CHEAP_WINDOW_START,
        )
        start = _parse_hhmm(DEFAULT_CHEAP_WINDOW_START)
    if end is None:
        logger.warning(
            "priority_window: invalid end=%r; using default %s",
            section.get("end"),
            DEFAULT_CHEAP_WINDOW_END,
        )
        end = _parse_hhmm(DEFAULT_CHEAP_WINDOW_END)
    return {"enabled": enabled, "start": start, "end": end}


def in_cheap_window(
    server_config: dict | None = None,
    now: datetime | None = None,
) -> bool:
    """Whether *now* (default: current local time) falls inside the cheap window.

    The window is half-open ``[start, end)`` and wraps across midnight when
    ``start > end`` (e.g. ``22:00–06:00``). Returns False when the window is
    disabled or malformed (a broken window never blocks or re-routes).
    """
    cfg = cheap_window_config(server_config)
    if not cfg["enabled"]:
        return False
    current = (now if now is not None else datetime.now()).time()
    start, end = cfg["start"], cfg["end"]
    if start <= end:
        return start <= current < end
    # Overnight wrap: at/after start, or before end.
    return current >= start or current < end


def effective_priority(
    header_value: Any,
    server_config: dict | None = None,
    now: datetime | None = None,
) -> str:
    """Compute the effective priority for a request (AC1).

    The parsed header priority, capped to ``low`` while the cheap window is
    active — the header value itself is unmodified and no deferral is
    applied (a spoofed ``high`` at 03:00 still routes as ``low``).
    """
    priority = parse_priority(header_value)
    if in_cheap_window(server_config, now=now):
        return PRIORITY_LOW
    return priority


def priority_from_headers(
    headers,
    server_config: dict | None = None,
    now: datetime | None = None,
) -> str:
    """Effective priority read from a request's header mapping.

    *headers* may be a Starlette ``Headers`` object, a plain dict, or an
    object exposing ``.get`` — the lookup is case-insensitive for dict-like
    headers (Starlette already is; a plain dict is normalised).
    """
    value = None
    try:
        if hasattr(headers, "get"):
            value = headers.get(PRIORITY_HEADER)
            if value is None:
                # Plain dicts are case-sensitive; normalise the key.
                lower = {str(k).lower(): v for k, v in headers.items()}
                value = lower.get(PRIORITY_HEADER.lower())
    except Exception:
        logger.debug("priority: unable to read X-Priority header", exc_info=True)
        value = None
    return effective_priority(value, server_config, now=now)


def set_request_priority(request, priority: str) -> None:
    """Stamp the effective priority on the request (``request.state``).

    Decision points read it via :func:`request_priority` so the cheap-window
    cap computed at the top level is honoured consistently across the whole
    request cycle (never recomputed with a different clock).
    """
    try:
        if not hasattr(request, "state"):
            return
        request.state.priority = priority
    except Exception:
        logger.debug("priority: unable to stamp request state", exc_info=True)


def request_priority(request, server_config: dict | None = None) -> str:
    """Effective priority for *request* (stamped value, else recomputed).

    Falls back to recomputing from the request headers when no stamp exists
    (e.g. a dispatch path that did not go through the top-level endpoint).
    """
    try:
        stamped = getattr(request.state, "priority", None)
        if stamped in VALID_PRIORITIES:
            return stamped
    except Exception:
        pass
    try:
        return priority_from_headers(getattr(request, "headers", None), server_config)
    except Exception:
        return DEFAULT_PRIORITY


# ---------------------------------------------------------------------------
# Priority routing profile (AC3)
# ---------------------------------------------------------------------------


def priority_routing_profile(
    priority: str,
    server_config: dict | None = None,
) -> dict:
    """Resolve the per-priority routing knobs (AC3).

    Returns a dict with:

    - ``queue`` (bool) — whether the request enters the bounded contention
      queue when local slots are busy:
        * ``high``: never queues (immediate remote fallback).
        * ``low``: always queues (bounded by the configured caps), preferring
          a local slot over a paid remote call.
        * ``medium``: today's config-driven behaviour (``contention_queue_
          policy == "queue"`` in the merged base config).
    - ``economic_bypass_serves_local`` (bool) — whether an economic cold-cache
      ``large_context_bypass`` is recovered locally (served local / queued)
      instead of skipped:
        * ``high``: False (fast-mode semantics — bypass routes remote).
        * ``low``: True (cheap-mode semantics — recover locally).
        * ``medium``: the merged base config flag
          (``local_large_context_economic_bypass_serves_local``).

    Paid-tier invariant for ``low`` (LP-0MVF17VP70053JXX): ``low`` never
    *prefers* a paid remote tier — local is preferred, then the free tier —
    but it is no longer forbidden from spending at all. Once the free tier is
    exhausted the chain continues into the ``opencode-go`` entries, which are
    only eligible inside their own ``available_times`` (UTC) windows, so paid
    spend stays confined to off-peak hours. The cheap priority window is
    local ``Europe/Dublin`` 01:00-10:00 (= UTC 00:00-09:00) while the opencode
    windows are UTC 00:00-01:00, 04:00-06:00 and 10:00-00:00, so the exception
    adds capacity in ~3 of the 9 cheap hours; the UTC 01:00-04:00 and
    06:00-10:00 gaps are deliberate paid-spend avoidance (the windows are
    never widened to close them).

    ``low`` failover capacity vs resiliency:

    - rate-limit (429) failures are quarantined per API-key account
      (``_usage_limit_account_key`` = ``api_key_env@endpoint``), so the three
      ``opencode-go`` entries carry independent quota and give 3x rate-limit
      capacity;
    - 5xx/stall failures are quarantined per failure domain
      (``_failure_domain_key`` = ``endpoint:model``), so the three entries
      share ONE domain and do **not** give 3x outage resilience — the chain
      skips straight past them during an opencode outage.

    Unknown priorities fall back to medium (defence in depth).
    """
    priority = parse_priority(priority)
    if priority == PRIORITY_HIGH:
        return {"queue": False, "economic_bypass_serves_local": False}
    if priority == PRIORITY_LOW:
        return {"queue": True, "economic_bypass_serves_local": True}
    server_config = server_config or {}
    policy = str(
        server_config.get("contention_queue_policy", "fallback") or "fallback"
    ).strip().lower()
    economic = bool(
        server_config.get("local_large_context_economic_bypass_serves_local", False)
    )
    return {"queue": policy == "queue", "economic_bypass_serves_local": economic}


# ---------------------------------------------------------------------------
# Per-model priority provider chains (AC2)
# ---------------------------------------------------------------------------


def _resolve_chain_entries(
    model_config: dict, chain: list[Any]
) -> list[dict]:
    """Resolve a ``priority_routes`` entry into provider config dicts.

    Each entry is either a full provider config dict (kept as-is) or a
    provider **name** (resolved against the model's ``providers`` list; a
    name that cannot be resolved is skipped with a warning — a bad entry
    must never crash routing).
    """
    providers = list(model_config.get("providers") or [])
    by_name = {
        str(p.get("name", "")): p
        for p in providers
        if isinstance(p, dict)
    }
    resolved: list[dict] = []
    for entry in chain or []:
        if isinstance(entry, dict):
            resolved.append(entry)
            continue
        name = str(entry or "").strip()
        if not name:
            continue
        match = by_name.get(name)
        if match is not None:
            resolved.append(match)
        else:
            logger.warning(
                "priority_routes: unknown provider name %r (model has %d "
                "providers); skipping entry",
                name,
                len(providers),
            )
    return resolved


def provider_chain_for_priority(
    model_config: dict, priority: str
) -> list[dict]:
    """Provider chain selected by *priority* for *model_config* (AC2).

    Precedence:

    1. ``model_config["priority_routes"][priority]`` when present (resolved
       names / dicts).
    2. The model's full ``providers`` list (backward-compatible default,
       today's ``medium`` semantics).

    A malformed ``priority_routes`` falls back to ``providers`` (never an
    empty or broken chain unless the model itself is empty).
    """
    routes = model_config.get("priority_routes")
    if isinstance(routes, dict):
        chain = routes.get(priority)
        resolved = _resolve_chain_entries(model_config, chain) if chain is not None else None
        if resolved:
            return resolved
    return list(model_config.get("providers") or [])


def local_preference_for_request(request) -> str:
    """Return the request's ``X-Local-Preference`` (default ``prefer-local``).

    LP-0MVIKPCSG002B0NL: the header is additive — missing or unknown values
    keep today's behaviour (AC8). The parsed value is cached on the request so
    every routing decision in the same request agrees.
    """
    cached = getattr(request, "_local_preference", None)
    if cached:
        return cached
    try:
        from proxy.admission_reserve import parse_preference

        preference = parse_preference(getattr(request, "headers", {}) or {})
    except Exception:
        preference = "prefer-local"
    try:
        setattr(request, "_local_preference", preference)
    except Exception:
        pass
    return preference


def apply_local_preference(chain: list[dict], preference: str) -> list[dict]:
    """Filter *chain* by the client's ``X-Local-Preference``.

    Only ``remote-ok`` changes the chain: it removes every ``type: local``
    entry so the local route is never attempted (LP-0MVIKPCSG002B0NL AC5).
    ``local-only`` and ``prefer-local`` keep the chain intact — their
    difference is enforced at the admission decision (a ``local-only`` refusal
    returns 429 instead of falling through).

    A chain that becomes empty is returned as-is (empty) so the caller's
    exhaustion path reports the truth rather than silently re-adding local.
    """
    if preference != "remote-ok":
        return chain
    return [
        entry
        for entry in chain
        if not (isinstance(entry, dict) and entry.get("type") == "local")
    ]


def effective_chain_for_request(
    request, model_config: dict, server_config: dict | None = None
) -> list[dict]:
    """Provider chain for *request*: effective priority → ``providers``.

    Convenience used by the dispatch cycles: computes the effective priority
    (cheap-window capped), stamps it on the request, and returns the
    priority-selected provider chain. The client's ``X-Local-Preference`` is
    applied last so ``remote-ok`` never attempts the local route
    (LP-0MVIKPCSG002B0NL AC5).
    """
    priority = request_priority(request, server_config)
    set_request_priority(request, priority)
    chain = provider_chain_for_priority(model_config, priority)
    if not chain:
        chain = list(model_config.get("providers") or [])
    return apply_local_preference(chain, local_preference_for_request(request))
