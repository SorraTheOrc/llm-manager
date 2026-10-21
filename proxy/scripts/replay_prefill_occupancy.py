#!/usr/bin/env python3
"""Replay a proxy log window under the occupancy-aware admission guard.

LP-0MVICD2YW004DJPV AC5. For every request that actually entered the
contention queue in the window (the guard admitted, then the request logged
``contention_queue_wait``), recompute the verdict the occupancy-aware guard
would have produced::

    projected_wait = eta_prefill_s * (busy_slots + 1)

where ``busy_slots`` is 1 when the queue event logged ``saturation=prefill``
(an in-flight prefill occupies the serialised gate) and 0 otherwise. The
``local_admission_guard`` and ``contention_queue_wait`` lines are emitted
back-to-back for the same request, so the ETA/budget are attributed to the
wait that immediately follows them.

Each wait is paired with its outcome for the same session — the next
``contention_queue_dispatch`` or ``contention_queue_fallback_after_queue``
line, whose ``queued_duration`` is the wait actually paid. The report then
says how many of the measured queued fallbacks would now be refused at
admission (and how much queue wait that saves), how many would still
dispatch, and flags any request the new verdict would refuse that had in fact
dispatched — a "lost dispatch" that AC5 requires to be zero (or justified).

Usage::

    python3 scripts/replay_prefill_occupancy.py /var/log/llama-proxy/proxy.log.2026-10-21_14 \\
        --start "2026-10-21 14:56:00" --end "2026-10-21 16:49:00"
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import datetime

GUARD = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?local_admission_guard "
    r"decision=(?P<decision>\w+) reason=\S+ enforced=\S+ "
    r"eta_prefill_s=(?P<eta>[0-9.]+) budget_s=(?P<budget>[0-9.]+).*?"
    r"session=(?P<session>\S+)"
)
WAIT = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?contention_queue_wait .*?"
    r"saturation=(?P<saturation>\w+)"
)
OUTCOME = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"contention_queue_(?P<kind>dispatch|fallback_after_queue) "
    r"provider=\S+ session=(?P<session>\S+) "
    r"queued_duration=(?P<dur>[0-9.]+)s .*?reason=(?P<reason>\S+)"
)


def _ts(line: str) -> datetime:
    return datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")


def parse(path: str, start: datetime, end: datetime) -> tuple[list[dict], dict]:
    """Return ``(waits, outcomes)`` for the window.

    ``waits`` is a list of dicts with ``session``, ``eta``, ``budget`` and
    ``saturation``. ``outcomes`` maps ``session -> [(kind, duration, ts)]``.
    """
    waits: list[dict] = []
    outcomes: dict[str, list[tuple]] = defaultdict(list)
    last_admit: dict | None = None
    for line in open(path, errors="replace"):
        try:
            at = _ts(line)
        except ValueError:
            continue
        if at < start or at > end:
            last_admit = None
            continue
        guard = GUARD.search(line)
        if guard:
            if guard.group("decision") == "admit":
                last_admit = {
                    "session": guard.group("session"),
                    "eta": float(guard.group("eta")),
                    "budget": float(guard.group("budget")),
                }
            else:
                last_admit = None
            continue
        wait = WAIT.search(line)
        if wait:
            if last_admit is not None:
                waits.append(
                    {**last_admit, "saturation": wait.group("saturation")}
                )
                last_admit = None
            continue
        outcome = OUTCOME.search(line)
        if outcome:
            outcomes[outcome.group("session")].append(
                (
                    outcome.group("kind"),
                    float(outcome.group("dur")),
                    at,
                    outcome.group("reason"),
                )
            )
    return waits, outcomes


def replay(waits: list[dict], outcomes: dict) -> dict:
    """Apply the occupancy-aware verdict to every observed queue wait."""
    by_session: dict[str, list[dict]] = defaultdict(list)
    for wait in waits:
        by_session[wait["session"]].append(wait)

    rows: list[dict] = []
    for session, session_waits in by_session.items():
        events = sorted(outcomes.get(session, []), key=lambda e: e[2])
        for index, wait in enumerate(session_waits):
            outcome = events[index] if index < len(events) else None
            busy = 1 if wait["saturation"] == "prefill" else 0
            projected = wait["eta"] * (busy + 1)
            rows.append(
                {
                    **wait,
                    "busy_slots": busy,
                    "projected_wait_s": projected,
                    "refused": projected > wait["budget"],
                    "outcome": outcome[0] if outcome else "no_outcome",
                    "queued_duration_s": outcome[1] if outcome else 0.0,
                }
            )

    refused = [r for r in rows if r["refused"]]
    dispatched = [r for r in rows if r["outcome"] == "dispatch"]
    fallbacks = [r for r in rows if r["outcome"] == "fallback_after_queue"]
    lost = [r for r in refused if r["outcome"] == "dispatch"]
    refused_fallbacks = [r for r in refused if r["outcome"] == "fallback_after_queue"]
    return {
        "queued": len(rows),
        "dispatched": len(dispatched),
        "fallback_after_queue": len(fallbacks),
        "no_outcome": sum(1 for r in rows if r["outcome"] == "no_outcome"),
        "refused": len(refused),
        "refused_would_dispatch": len(lost),
        "refused_fallbacks": len(refused_fallbacks),
        "admits": len(rows) - len(refused),
        "admits_dispatched": len(dispatched) - len(lost),
        "fallbacks_remaining": len(fallbacks) - len(refused_fallbacks),
        "wait_saved_s": round(sum(r["queued_duration_s"] for r in refused), 1),
        "lost": lost,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="proxy log file to replay")
    parser.add_argument("--start", required=True, help='"YYYY-MM-DD HH:MM:SS"')
    parser.add_argument("--end", required=True, help='"YYYY-MM-DD HH:MM:SS"')
    args = parser.parse_args()
    fmt = "%Y-%m-%d %H:%M:%S"
    waits, outcomes = parse(
        args.log, datetime.strptime(args.start, fmt), datetime.strptime(args.end, fmt)
    )
    result = replay(waits, outcomes)
    print(f"window: {args.start} -> {args.end}")
    print(f"queue waits observed:            {result['queued']}")
    print(f"  dispatched:                    {result['dispatched']}")
    print(f"  fallback_after_queue:          {result['fallback_after_queue']}")
    print(f"  no outcome in window:          {result['no_outcome']}")
    print(f"occupancy-aware refuses:         {result['refused']}")
    print(f"  of which dispatched (LOST):    {result['refused_would_dispatch']}")
    print(f"occupancy-aware admits:          {result['admits']}")
    print(f"  of which still dispatch:       {result['admits_dispatched']}")
    print(f"fallbacks remaining:             {result['fallbacks_remaining']}")
    print(f"aggregate queue wait saved:      {result['wait_saved_s']:.0f} s "
          f"({result['wait_saved_s'] / 3600:.2f} h)")
    for row in result["lost"]:
        print("LOST-DISPATCH:", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
