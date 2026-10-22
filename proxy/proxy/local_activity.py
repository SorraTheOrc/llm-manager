"""Last-observed local-model activity, for idle-aware routing.

LP-0MVGD9Z8Z0055Y5J. Routing is size/economics-driven and ignores whether the
GPU is doing anything, so a large (or cache-cold) prompt is bypassed to a paid
remote provider even while both slots have sat unused for minutes. This module
tracks when the local backend last produced work, so the routing gate can spend
obvious idle capacity instead.

The signal is intentionally cheap and truthful:

- a monotonic timestamp (immune to wall-clock jumps) updated when a local
  dispatch starts, when the first byte arrives, on every forwarded SSE chunk,
  and when the request finishes;
- a wall-clock timestamp of the same event for operator display;
- a monotonic baseline of *process start*, so a freshly restarted proxy with no
  traffic yet reports the true idle age rather than "unknown".

Consumers read :func:`local_idle_seconds` (the routing gate) and
:func:`activity_snapshot` (``/llama/local/status``). Nothing here calls
llama-server.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

#: Monotonic timestamp of the last observed local activity. Initialised to
#: import time so an idle process reports its own age as the idle age.
_last_activity_monotonic: float = time.monotonic()

#: Wall-clock timestamp paired with ``_last_activity_monotonic`` (display only).
_last_activity_wall: float = time.time()

#: What produced the last activity ("startup", "dispatch", "first_byte",
#: "chunk", "finish"). Diagnostic only.
_last_activity_source: str = "startup"


def record_local_activity(
    source: str = "activity",
    now: float | None = None,
    wall: float | None = None,
) -> None:
    """Record that the local backend just did work.

    ``now``/``wall`` are injectable for tests; production callers pass nothing.
    """
    global _last_activity_monotonic, _last_activity_wall, _last_activity_source
    _last_activity_monotonic = time.monotonic() if now is None else float(now)
    _last_activity_wall = time.time() if wall is None else float(wall)
    _last_activity_source = source or "activity"


def last_local_activity_ts() -> float:
    """Return the monotonic timestamp of the last observed local activity."""
    return _last_activity_monotonic


def local_idle_seconds(now: float | None = None) -> float:
    """Seconds since the local backend last did work (never negative)."""
    reference = time.monotonic() if now is None else float(now)
    return max(0.0, reference - _last_activity_monotonic)


def activity_snapshot(now: float | None = None) -> dict:
    """Return the idle picture for ``/llama/local/status``.

    ``idle_seconds`` is monotonic-derived (authoritative for routing);
    ``last_activity_at`` is the wall-clock rendering of the same event.
    """
    try:
        last_activity_at = datetime.fromtimestamp(
            _last_activity_wall, UTC
        ).isoformat()
    except (OverflowError, OSError, ValueError):
        last_activity_at = None
    return {
        "idle_seconds": round(local_idle_seconds(now), 3),
        "last_activity_at": last_activity_at,
        "last_activity_source": _last_activity_source,
    }


def reset_local_activity(now: float | None = None, wall: float | None = None) -> None:
    """Reset the baseline to *now* (test isolation / operator reset)."""
    record_local_activity("reset", now=now, wall=wall)
