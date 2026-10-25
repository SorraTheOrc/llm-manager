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

Two modes:

``occupancy`` (default, the original invocation)
    Recompute the occupancy-aware prefill-ETA verdict for every queued
    request in a **proxy** log window (the AC5 measurement above).

``residency`` (``--residency-replay``)
    LP-0MVN1IQTF0037WTG AC7 (integration half). Replay a **llama-server**
    slot log through the *real* residency-aware admission ladder: each task's
    measured ``n_past`` is projected into a
    :class:`proxy.residency.ResidencyView` (slot id + task owner +
    ``context_tokens``) and the attempt is decided by
    ``provider._should_skip_local`` with the production config defaults
    (``local_resident_prefix_min_ratio`` 0.5, cold budget 25 000 tokens). No
    copy of the ladder logic lives here — the replay exists to prove the
    production decision, not a paraphrase of it. The pinned 2026-10-24
    two-task fixture is
    ``proxy/tests/fixtures/slot_thrash_2026-10-24/tasks_92383_92422.log``.

Usage::

    # occupancy replay (unchanged CLI contract)
    python3 scripts/replay_prefill_occupancy.py /var/log/llama-proxy/proxy.log.2026-10-21_14 \\
        --start "2026-10-21 14:56:00" --end "2026-10-21 16:49:00"

    # residency replay over a llama-server slot log
    python3 scripts/replay_prefill_occupancy.py \\
        tests/fixtures/slot_thrash_2026-10-24/tasks_92383_92422.log \\
        --residency-replay --cold-budget 25000 --min-ratio 0.5
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Make the proxy package importable when run as a plain script.
_PROXY_ROOT = Path(__file__).resolve().parents[1]
if str(_PROXY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROXY_ROOT))

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


# ---------------------------------------------------------------------------
# Residency replay (LP-0MVN1IQTF0037WTG AC7 integration half)
# ---------------------------------------------------------------------------

#: llama-server announces the task on a fresh prompt with its size...
SLOT_TASK_PROMPT = re.compile(
    r"slot\s+update_slots: id\s+(?P<slot>\d+) \| task (?P<task>\d+) \| "
    r"new prompt, .*?task\.n_tokens = (?P<n_tokens>\d+)"
)
#: ...and reports the reusable prefix it actually found on the next line.
SLOT_TASK_N_PAST = re.compile(
    r"slot\s+update_slots: id\s+(?P<slot>\d+) \| task (?P<task>\d+) \| "
    r"n_past = (?P<n_past>\d+)"
)

#: One slot-log attempt: the measured reusable prefix for a task's prompt.
SlotAttempt = dict


def parse_slot_log(path: str) -> list[SlotAttempt]:
    """Parse a llama-server slot log into per-task residency attempts.

    Each attempt carries the task id, its slot, the prompt size
    (``n_tokens``) and the prefix llama-server actually found resident
    (``n_past``). A task whose ``n_past`` line is missing is reported with
    ``n_past=None`` (unmeasured) rather than assumed cold.
    """
    attempts: list[SlotAttempt] = []
    pending: dict[str, SlotAttempt] = {}
    for line in open(path, errors="replace"):
        match = SLOT_TASK_PROMPT.search(line)
        if match:
            pending[match.group("task")] = {
                "task": match.group("task"),
                "slot": int(match.group("slot")),
                "estimated_tokens": int(match.group("n_tokens")),
                "n_past": None,
            }
            continue
        match = SLOT_TASK_N_PAST.search(line)
        if match:
            attempt = pending.get(match.group("task"))
            if attempt is None:
                # n_past can precede the "new prompt" line on a resumed slot;
                # synthesise the attempt so the measurement is never lost.
                attempt = {
                    "task": match.group("task"),
                    "slot": int(match.group("slot")),
                    "estimated_tokens": 0,
                    "n_past": None,
                }
                pending[match.group("task")] = attempt
            attempt["n_past"] = int(match.group("n_past"))
    for attempt in pending.values():
        if attempt["estimated_tokens"] <= 0:
            # No size known: skip — an unpriced attempt proves nothing.
            continue
        attempts.append(attempt)
    return attempts


def _residency_view_for(attempt: SlotAttempt):
    """Project one attempt's measurement into a real ResidencyView."""
    from proxy.residency import ResidencyView, SlotResidency

    owner = f"task-{attempt['task']}"
    slots = (
        SlotResidency(
            endpoint="",
            slot_id=attempt["slot"],
            owner_session=owner,
            context_tokens=attempt["n_past"],
            is_processing=False,
        ),
    )
    return ResidencyView(slots=slots, snapshot_age=0.0, known=True)


def replay_residency(
    attempts: list[SlotAttempt],
    *,
    cold_budget: int = 25_000,
    min_ratio: float = 0.5,
) -> dict:
    """Decide every attempt with the production residency-aware ladder.

    *cold_budget* is the admission budget AC3 consumes (owned by
    ``LP-0MVN3KMGI009BZ0D``, 25 000 tokens). *min_ratio* is
    ``server.local_resident_prefix_min_ratio``.
    """
    from proxy import provider

    rows: list[dict] = []
    for attempt in attempts:
        view = _residency_view_for(attempt)
        session_id = f"task-{attempt['task']}"
        skip, reason = provider._should_skip_local(
            "Qwen3",
            session_id,
            {"messages": []},
            cold_budget,
            estimated_tokens=attempt["estimated_tokens"],
            resident_prefix_min_ratio=min_ratio,
            residency_view=view,
        )
        facts = provider.resolve_resident_prefix(
            session_id, attempt["estimated_tokens"], view=view
        )
        rows.append(
            {
                **attempt,
                "session": session_id,
                "reuse_ratio": round(facts.reuse_ratio, 4),
                "resident": facts.resident,
                "admitted": not skip,
                "reason": reason,
            }
        )

    over_budget = [r for r in rows if r["estimated_tokens"] > cold_budget]
    violations = [
        r
        for r in over_budget
        if r["admitted"] and not r["resident"]
    ]
    return {
        "attempts": len(rows),
        "over_budget": len(over_budget),
        "resident_admits": sum(1 for r in rows if r["admitted"] and r["resident"]),
        "cold_denials": sum(
            1 for r in rows if r["reason"] == "resident_prefix_insufficient"
        ),
        "violations": violations,
        "rows": rows,
    }


#: Route value the capture uses for an admission the idle-aware override
#: served (LP-0MVGD9Z8Z0055Y5J). Those are reported separately: an idle GPU
#: spends capacity it would otherwise waste, so they are not violations.
IDLE_EXEMPT_ROUTE = "idle_bypass"


def load_window_fixture(path: str) -> list[dict]:
    """Load the captured window JSONL (header record skipped)."""
    import json

    records: list[dict] = []
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "_meta" in record:
            continue
        records.append(record)
    return records


def replay_window(
    records: list[dict],
    *,
    cold_budget: int = 25_000,
    min_ratio: float = 0.5,
) -> dict:
    """Replay a captured thrash window through the residency-aware ladder.

    Modelling decision (documented, not hidden): the capture records **no
    measured resident prefix** — per-slot prefill progress was unobservable in
    that window (``prefill_tokens_computed: null`` for every record) and the
    thrash premise is that each session had lost its slot to another session.
    Each attempt is therefore replayed as what it measured: a session that owns
    no slot, i.e. a measured zero prefix. Attempts the capture attributes to
    the idle-aware override (``route == "idle_bypass"``) are reported
    separately — an idle GPU is allowed to spend capacity it would otherwise
    waste — and are never counted as violations.

    Returns counts plus the violating rows (must be empty).
    """
    from proxy.residency import ResidencyView

    from proxy import provider

    empty_view = ResidencyView(slots=(), snapshot_age=0.0, known=True)
    rows: list[dict] = []
    for record in records:
        estimated = int(record.get("estimated_tokens") or 0)
        if estimated <= 0:
            continue
        session = record.get("session") or "unknown"
        idle_exempt = record.get("route") == IDLE_EXEMPT_ROUTE
        if idle_exempt:
            rows.append(
                {
                    "session": session,
                    "estimated_tokens": estimated,
                    "admitted": True,
                    "reason": "idle_admit",
                    "idle_exempt": True,
                }
            )
            continue
        skip, reason = provider._should_skip_local(
            "Qwen3",
            session,
            {"messages": []},
            cold_budget,
            estimated_tokens=estimated,
            resident_prefix_min_ratio=min_ratio,
            residency_view=empty_view,
        )
        rows.append(
            {
                "session": session,
                "estimated_tokens": estimated,
                "admitted": not skip,
                "reason": reason,
                "idle_exempt": False,
            }
        )

    over_budget = [r for r in rows if r["estimated_tokens"] > cold_budget]
    violations = [
        r for r in over_budget if r["admitted"] and not r["idle_exempt"]
    ]
    idle_exempt = [r for r in rows if r["idle_exempt"]]
    return {
        "attempts": len(rows),
        "over_budget": len(over_budget),
        "idle_exempt": len(idle_exempt),
        "admitted": sum(1 for r in rows if r["admitted"]),
        "cold_denials": sum(
            1 for r in rows if r["reason"] == "resident_prefix_insufficient"
        ),
        "violations": violations,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="proxy log file to replay")
    parser.add_argument(
        "--start", help='"YYYY-MM-DD HH:MM:SS" (required for the occupancy replay)'
    )
    parser.add_argument(
        "--end", help='"YYYY-MM-DD HH:MM:SS" (required for the occupancy replay)'
    )
    parser.add_argument(
        "--residency-replay",
        action="store_true",
        help=(
            "replay a llama-server slot log through the residency-aware "
            "admission ladder instead of the occupancy guard"
        ),
    )
    parser.add_argument(
        "--window-fixture",
        help=(
            "replay a captured thrash-window JSONL (one attempt per line) "
            "through the residency-aware ladder"
        ),
    )
    parser.add_argument(
        "--cold-budget",
        type=int,
        default=25_000,
        help="cold-prefill admission budget in tokens (default 25000)",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.5,
        help="local_resident_prefix_min_ratio (default 0.5)",
    )
    args = parser.parse_args()
    if args.window_fixture:
        result = replay_window(
            load_window_fixture(args.window_fixture),
            cold_budget=args.cold_budget,
            min_ratio=args.min_ratio,
        )
        print(f"window fixture: {args.window_fixture}")
        print(f"attempts replayed:                {result['attempts']}")
        print(f"  over the cold budget:           {result['over_budget']}")
        print(f"  admitted:                       {result['admitted']}")
        print(f"  idle-exempt (reported apart):   {result['idle_exempt']}")
        print(f"  resident_prefix_insufficient:   {result['cold_denials']}")
        print(
            "over-budget cold admissions:      "
            f"{len(result['violations'])} (must be 0)"
        )
        return 0
    if args.residency_replay:
        result = replay_residency(
            parse_slot_log(args.log),
            cold_budget=args.cold_budget,
            min_ratio=args.min_ratio,
        )
        print(f"slot log: {args.log}")
        print(f"attempts replayed:                {result['attempts']}")
        print(f"  over the cold budget:           {result['over_budget']}")
        print(f"  resident admits:                {result['resident_admits']}")
        print(f"  resident_prefix_insufficient:   {result['cold_denials']}")
        print(
            "over-budget cold admissions:      "
            f"{len(result['violations'])} (must be 0)"
        )
        for row in result["rows"]:
            print(
                f"task {row['task']:>7} slot {row['slot']}: "
                f"estimated={row['estimated_tokens']} n_past={row['n_past']} "
                f"reuse={row['reuse_ratio']:.4f} resident={row['resident']} "
                f"decision={'admit' if row['admitted'] else row['reason']}"
            )
        return 0
    if not args.start or not args.end:
        parser.error("--start and --end are required for the occupancy replay")
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
