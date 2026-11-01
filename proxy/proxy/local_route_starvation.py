"""Self-throttled local-route starvation warning (LP-0MVWFKIVO007EGDU, parent AC11).

Why this exists
---------------

The 2026-10-31 measurement (parent ``LP-0MVWEZCB0006WPBQ``) found the local
entry armed for **217 windows in 5.5 h** (median gap 67 s ≈ the 60 s TTL —
arm → expire → re-fail → re-arm) while ``/llama/local/status`` kept reporting
``available_slots: 2``, ``capacity_hint: idle`` and ``llama_server_running:
true``. Local share for that window was **2.8 %** (165 local vs 5 707 remote
streams). Nothing said "the local route is armed while the engine is idle";
the evidence existed only as hundreds of separate log lines, so the episode was
diagnosable only by post-hoc archaeology.

This module is the standing detector that measurement asked for. It is
**observability only**: it records arm events that the routing code already
produces, and emits one ``local_route_starved`` WARNING when the armed state
covers a sustained fraction of a rolling window while the engine is healthy.

Design constraints (parent AC11)

- **State-only, no new polling.** The tracker is fed from the existing arm
  sites (``provider._handle_http_error_with_cooldown``,
  ``provider._handle_connection_error_in_fallback`` and
  ``router_helpers._record_prefill_thrash_cooldown``) and is evaluated *on
  those events*. There is no timer, no task and no HTTP call: engine health is
  read from in-process state that ``/llama/local/status`` already publishes
  (``llama_process`` liveness and the existing coalesced ``/slots`` count
  cache, ``observability.last_known_slot_counts``).
- **Bounded output.** At most one WARNING per
  :data:`DEFAULT_THROTTLE_SECONDS` (10 min).
- **No false positive on a single fault.** One arm per window is normal, so a
  warning needs at least :data:`DEFAULT_MIN_ARMS` (2) arms **and** the union of
  their armed intervals to cover at least half the window.
- **Quiet during warm-up.** A local deferral inside the cold-start grace window
  is expected behaviour (``cold_start.is_cold``), so the detector never fires
  then.
- **Never a routing input.** The routing decision never reads this module; the
  tracker cannot change a cooldown, a skip or a chain. :func:`starvation_snapshot`
  exists so observability code and tests can read the state back, and no
  ``/llama/local/status`` field is added beyond the parent's AC10 surface.

The emitted line is deliberately greppable and carries per-reason counts in the
same canonical tokens the routing docs use (``local_provider_cooldown``,
``measured_cache_deficit``, ``prefill_thrash_cooldown``)::

    local_route_starved armed_s=302 window_s=600 armed_fraction=0.50 arms=3 \
        available_slots=2 llama_server_running=true \
        reasons=local_provider_cooldown:2,measured_cache_deficit:1
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping

logger = logging.getLogger("llama-proxy.local_route_starvation")

#: Stable log token. Everything about the warning is greppable from this word.
WARNING_TOKEN = "local_route_starved"

#: Reason tokens are **canonical** routing vocabulary, spelled here as plain
#: literals rather than imported constants so this module keeps the import graph
#: free of a new edge (a re-exported import is stripped as unused by the repo's
#: ``ruff --fix`` pass, which silently broke the arm-site import once).
#: ``test_local_route_starvation_detector.py`` pins each literal against its
#: owning definition — ``observability.LOCAL_ROUTE_REASON_PROVIDER_COOLDOWN``,
#: ``measured_cache_abort.MEASURED_CACHE_DEFICIT_REASON`` and
#: ``admission_reserve.LOCAL_FALLBACK_REASON_TOKENS`` — so a renamed token fails
#: the tests instead of drifting quietly.

#: The local provider entry is inside a cooldown window (entry-wide arm).
LOCAL_ROUTE_REASON_PROVIDER_COOLDOWN = "local_provider_cooldown"

#: The measured cache-deficit guard released a dispatch, arming the per-session /
#: per-prompt thrash window.
LOCAL_ROUTE_REASON_MEASURED_CACHE_DEFICIT = "measured_cache_deficit"

#: Per-session/prompt post-deadline (or post-deficit) cooldown token. It is the
#: ``reason=`` field of ``routing_skip_local``; the routing-docs table spells it
#: the same way, so the detector counts it verbatim rather than inventing a
#: second spelling (LP-0MVRMC1W6003MV3Y).
LOCAL_ROUTE_REASON_PREFILL_THRASH = "prefill_thrash_cooldown"

#: Rolling window the armed fraction is measured over (parent AC11: 10 min).
DEFAULT_WINDOW_SECONDS = 600.0

#: Fraction of the window that must be armed before a warning is warranted.
#: The measured fingerprint (60 s arms, median 67 s gap) sits at ~0.9, while a
#: single genuine 60 s fault sits at 0.1.
DEFAULT_MIN_ARMED_FRACTION = 0.5

#: Minimum number of arm events in the window. A single arm never fires the
#: detector, however long its own cooldown is (parent AC11d).
DEFAULT_MIN_ARMS = 2

#: At most one warning per 10 minutes (parent AC11).
DEFAULT_THROTTLE_SECONDS = 600.0

#: Hard bound on retained events, so an arm storm cannot grow memory without
#: limit. Pruning normally keeps the list far below this.
MAX_RETAINED_ARMS = 2048


class _ArmEvent:
    """One observed arm: ``[start, end]`` (monotonic) plus its reason token."""

    __slots__ = ("start", "end", "reason")

    def __init__(self, start: float, end: float, reason: str) -> None:
        self.start = start
        self.end = end
        self.reason = reason


#: Arm events inside the current window (monotonic timestamps).
_arm_events: list[_ArmEvent] = []

#: Monotonic timestamp of the last emitted warning (the throttle anchor).
_last_warning_at: float | None = None


def reset_local_route_starvation_tracker() -> None:
    """Drop all tracker state (tests, and a deliberate operator reset).

    Never called from routing code: the tracker is observability only.
    """
    _arm_events.clear()
    global _last_warning_at
    _last_warning_at = None


def _server_section(config: object) -> Mapping:
    """Return the ``server`` config section of *config* (flat configs too)."""
    if not isinstance(config, Mapping):
        return {}
    server = config.get("server")
    if isinstance(server, Mapping):
        return server
    return config


def _positive_float(raw: object, default: float) -> float:
    """Parse a positive float, falling back to *default* on anything else.

    A malformed value must never disable a detector, so typos fall back to the
    documented default; ``0`` or a negative value disables the knob it belongs
    to and is therefore returned as ``0.0``.
    """
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if value < 0:
        return default
    return value


def _min_arms(raw: object, default: int) -> int:
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return value if value >= 1 else default


def _as_bool(raw: object, default: bool = True) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in ("1", "true", "yes", "on"):
            return True
        if value in ("0", "false", "no", "off"):
            return False
    return default


def starved_warning_settings(config: object = None) -> dict:
    """Resolve the detector knobs from the ``server`` config section.

    Every key is optional and documented in ``proxy/config.yaml``; absent keys
    use the module defaults. ``local_route_starvation_enabled: 0`` is the
    rollback (a value change, no code change).
    """
    section = _server_section(config)
    window = _positive_float(
        section.get("local_route_starvation_window_seconds"),
        DEFAULT_WINDOW_SECONDS,
    )
    if window <= 0:
        window = DEFAULT_WINDOW_SECONDS
    fraction_raw = _positive_float(
        section.get("local_route_starvation_min_armed_fraction"),
        DEFAULT_MIN_ARMED_FRACTION,
    )
    fraction = min(1.0, max(0.0, fraction_raw))
    throttle = _positive_float(
        section.get("local_route_starvation_throttle_seconds"),
        DEFAULT_THROTTLE_SECONDS,
    )
    if throttle <= 0:
        throttle = DEFAULT_THROTTLE_SECONDS
    return {
        "enabled": _as_bool(section.get("local_route_starvation_enabled", True)),
        "window_seconds": window,
        "min_armed_fraction": fraction,
        "min_arms": _min_arms(
            section.get("local_route_starvation_min_arms"), DEFAULT_MIN_ARMS
        ),
        "throttle_seconds": throttle,
    }


def default_engine_health(config: object = None) -> dict:
    """Read engine health from state the status surface already publishes.

    Both halves are in-process and cheap — no HTTP, no new poll:

    - ``llama_server_running`` — ``server.llama_process`` liveness, exactly the
      predicate ``query_llama_status`` reports;
    - ``available_slots`` — the existing coalesced ``/slots`` sample cache
      (``observability.last_known_slot_counts``), ``None`` when no *fresh*
      sample exists (never a fabricated ``0`` or a stale ``2``);
    - ``cold_start`` — ``cold_start.is_cold``, the documented warm-up boundary.

    Never raises: an unreadable probe reports the engine as unknown, and the
    detector then stays silent (a warning is observability, so silence is the
    safe failure).
    """
    running = False
    slots: int | None = None
    cold = False
    try:
        from proxy import server as _srv

        if config is None:
            config = getattr(_srv, "config", None)
        process = getattr(_srv, "llama_process", None)
        running = process is not None and process.poll() is None
    except Exception:
        running = False
    try:
        from proxy.observability import last_known_slot_counts

        counts = last_known_slot_counts()
        if counts is not None:
            slots = int(counts[0])
    except Exception:
        slots = None
    try:
        from proxy.cold_start import is_cold

        cold = bool(is_cold(config))
    except Exception:
        cold = False
    return {
        "llama_server_running": running,
        "available_slots": slots,
        "cold_start": cold,
    }


def _engine_is_healthy(health: Mapping | None) -> bool:
    """Whether *health* satisfies the parent's healthy-engine condition.

    ``llama_server_running`` must be true and ``available_slots`` must be a known
    positive count; an unknown count (no fresh ``/slots`` sample) does not
    qualify — the warning claims "the engine is idle and the route is armed",
    which is not assertable without the count.
    """
    if not isinstance(health, Mapping):
        return False
    if health.get("cold_start"):
        return False
    if health.get("llama_server_running") is not True:
        return False
    slots = health.get("available_slots")
    if isinstance(slots, bool) or not isinstance(slots, int):
        return False
    return slots > 0


def _prune(now: float, window: float) -> None:
    """Drop events whose armed interval ended before the window opened."""
    horizon = now - window
    if len(_arm_events) > MAX_RETAINED_ARMS:
        # Keep the newest events only; the oldest cannot matter after pruning
        # anyway, and this bounds memory even if prune is starved.
        del _arm_events[: len(_arm_events) - MAX_RETAINED_ARMS]
    if not _arm_events:
        return
    keep = [event for event in _arm_events if event.end > horizon]
    if len(keep) != len(_arm_events):
        _arm_events[:] = keep


def _window_events(now: float, window: float) -> list[_ArmEvent]:
    """Events whose armed interval intersects ``[now - window, now]``.

    ``start <= now`` (not ``<``) so an arm recorded at the very instant of the
    evaluation — the common case, since evaluation happens *on* the arm — is
    visible to its own check.
    """
    horizon = now - window
    return [
        event for event in _arm_events if event.end > horizon and event.start <= now
    ]


def _armed_seconds(events: list[_ArmEvent], now: float, window: float) -> float:
    """Union of the events' armed intervals, clipped to the window.

    Overlapping arms are counted **once**: two sessions refused by the same
    entry cooldown are one continuous armed period, not twice the starvation.
    """
    horizon = now - window
    intervals: list[tuple[float, float]] = []
    for event in events:
        start = max(event.start, horizon)
        end = min(event.end, now)
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        elif end > current_end:
            current_end = end
    total += current_end - current_start
    return min(total, window)


def _reason_counts(events: list[_ArmEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        counts[event.reason] = counts.get(event.reason, 0) + 1
    return counts


def _format_reason_counts(counts: Mapping[str, int]) -> str:
    """Render counts as ``token:count`` pairs, most frequent first.

    Ties break on the token so the line is deterministic and diffable.
    """
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ",".join(f"{token}:{count}" for token, count in ordered)


def _emit_warning(
    *,
    settings: Mapping,
    events: list[_ArmEvent],
    armed_seconds: float,
    health: Mapping,
) -> None:
    window = float(settings["window_seconds"])
    fraction = armed_seconds / window if window > 0 else 0.0
    line = (
        "%s armed_s=%.0f window_s=%.0f armed_fraction=%.2f arms=%d "
        "available_slots=%s llama_server_running=%s reasons=%s"
    )
    logger.warning(
        line,
        WARNING_TOKEN,
        armed_seconds,
        window,
        fraction,
        len(events),
        health.get("available_slots"),
        str(bool(health.get("llama_server_running"))).lower(),
        _format_reason_counts(_reason_counts(events)),
    )


def record_local_route_arm(
    reason: str,
    duration_seconds: float,
    *,
    now: float | None = None,
    config: object = None,
    health: Mapping | None = None,
) -> bool:
    """Record one local-route arm and evaluate the starvation warning.

    Called from the existing arm sites — never from routing decisions, and
    never on a timer. *reason* is a canonical routing token
    (``local_provider_cooldown`` / ``measured_cache_deficit`` /
    ``prefill_thrash_cooldown``); *duration_seconds* is the cooldown the arm
    applies, i.e. how long the route will refuse local work from it.

    Returns ``True`` when a warning was emitted by this call. Never raises: the
    caller is an error path, and observability must not add a failure mode.
    """
    try:
        if not isinstance(reason, str) or not reason:
            return False
        try:
            duration = float(duration_seconds)
        except (TypeError, ValueError):
            return False
        if duration <= 0:
            return False
        current = time.monotonic() if now is None else float(now)
        settings = starved_warning_settings(config)
        if not settings["enabled"]:
            return False
        window = float(settings["window_seconds"])
        _arm_events.append(_ArmEvent(current, current + duration, reason))
        _prune(current, window)
        return _maybe_warn(
            now=current, settings=settings, config=config, health=health
        )
    except Exception:  # pragma: no cover - defensive: never break an arm path
        return False


def _maybe_warn(
    *,
    now: float,
    settings: Mapping,
    config: object = None,
    health: Mapping | None = None,
) -> bool:
    global _last_warning_at
    window = float(settings["window_seconds"])
    events = _window_events(now, window)
    if len(events) < int(settings["min_arms"]):
        return False
    armed = _armed_seconds(events, now, window)
    if window <= 0 or (armed / window) < float(settings["min_armed_fraction"]):
        return False
    resolved_health = health if health is not None else default_engine_health(config)
    if not _engine_is_healthy(resolved_health):
        return False
    throttle = float(settings["throttle_seconds"])
    if _last_warning_at is not None and (now - _last_warning_at) < throttle:
        return False
    _emit_warning(
        settings=settings,
        events=events,
        armed_seconds=armed,
        health=resolved_health,
    )
    _last_warning_at = now
    return True


def starvation_snapshot(
    now: float | None = None,
    config: object = None,
    health: Mapping | None = None,
) -> dict:
    """Read-only snapshot of the tracker's state (observability and tests).

    Nothing in the routing path calls this — it exists so the state the warning
    is derived from can be inspected without parsing the log line. It adds no
    ``/llama/local/status`` field (the parent's AC10 fields remain the only
    status surface).
    """
    current = time.monotonic() if now is None else float(now)
    settings = starved_warning_settings(config)
    window = float(settings["window_seconds"])
    events = _window_events(current, window)
    armed = _armed_seconds(events, current, window)
    resolved_health = health if health is not None else default_engine_health(config)
    last_warning_age: float | None = None
    if _last_warning_at is not None:
        last_warning_age = max(0.0, current - _last_warning_at)
    return {
        "window_seconds": window,
        "armed_seconds": round(armed, 3),
        "armed_fraction": round(armed / window, 4) if window > 0 else 0.0,
        "arms": len(events),
        "reasons": _reason_counts(events),
        "throttle_seconds": float(settings["throttle_seconds"]),
        "last_warning_age_s": (
            round(last_warning_age, 3) if last_warning_age is not None else None
        ),
        "engine_healthy": _engine_is_healthy(resolved_health),
        "engine": {
            "llama_server_running": resolved_health.get("llama_server_running"),
            "available_slots": resolved_health.get("available_slots"),
            "cold_start": resolved_health.get("cold_start"),
        },
    }
