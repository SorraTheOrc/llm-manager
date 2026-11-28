#!/usr/bin/env python3
"""Attribute every ``prefill_deadline`` eviction to the path that admitted it.

LP-0MVZ5WV6W008UO5F AC1. The usage report counts
``local_eviction_after_acceptance reason=prefill_deadline`` events and the
alert rule treats any of them as a defect, but the count alone cannot answer
the only question triage asks: **which admission path let this dispatch
start?** The rung trade-off (``local_admission_guard_free_gate_deadline_budget``)
is only one of several answers, and the other paths need a different fix.

The tool scans ``<log-dir>/proxy.log*`` for a window (plain and gzipped files,
rotation-aware) and joins every ``prefill_deadline`` eviction to the admission
evidence for the same session at the measured dispatch start
(``eviction_ts - elapsed_s``):

``rung_accepted``
    A ``local_admission_guard decision=admit`` line priced a
    ``projected_wait_s`` that fitted the deadline-relative bound but not the
    queue budget (``bound_s > budget_s``) — i.e. the free-gate rung admitted
    it. These are the rung's documented admitted-and-doomed trade.

``priced_within_queue_budget``
    The guard admitted it inside the queue budget, yet the prefill never
    produced a first byte — the request was blocked behind someone else's
    prefill (occupancy priced at ~0). A pricing gap, not the rung.

``priced_above_bound``
    Admitted although the priced wait exceeded the bound it was compared
    against: should be impossible and is reported for that reason.

``idle_override`` / ``economic_override`` / ``capacity_override``
    A routing override admitted the dispatch without the guard adjudicating.

``high_priority_inert``
    ``high`` priority was *genuinely* inert: every time bound was disabled, so
    the guard returned ``no_budget`` and nothing was priced (legacy bare
    ``reason=no_budget`` lines). New inert lines log
    ``reason=inert_admit reason_inert=no_budget`` and class as
    ``inert_no_budget`` instead (LP-0MVZF4SU10032OZH). An **unlogged**
    high-priority eviction is ``inert_unlogged`` like any other priority
    (LP-0MVZQUT3A0039MLY).

``inert_no_budget`` / ``inert_insufficient_samples`` / ``inert_policy_off``
    (LP-0MVZF4SU10032OZH AC1/AC2) the inert admit was LOGGED with
    ``reason=inert_admit reason_inert=<reason>``, so the class is distinguished
    by the inert reason: ``no_budget`` (every time bound disabled, nothing
    priced), ``insufficient_samples`` (uncalibrated ``(model, bucket)``
    window), ``policy_off`` (guard disabled in config). This is what used to
    be invisible as ``inert_unlogged``.

``inert_unlogged``
    No admission evidence at all at the dispatch instant and no override: a
    log-rotation gap, a line the window predates, or an inert admit whose
    ``reason_inert=`` could not be read. Un-attributable from the log alone.

Usage::

    python3 attribute_prefill_deadline.py \
        --start '2026-11-01 11:09:07' --until '2026-11-02 11:09:07' [--json] [--rows]

The ``--json`` payload is the same summary the triage rule records on the
work item: the count per admission path, the priced new tokens each path
discarded (their sum is the report's ``wasted prefill tokens`` figure), and
whether every event was inside the cold-prefill budget.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
from collections import Counter
from datetime import datetime, timedelta

DEFAULT_LOG_DIR = "/var/log/llama-proxy"

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ ")
KV = re.compile(r"(\w+)=([^\s]+)")

EVICTION = "local_eviction_after_acceptance "
GUARD = "local_admission_guard decision="
OVERRIDES = (
    ("routing_idle_admit ", "idle_override"),
    ("routing_economic_bypass_local ", "economic_override"),
    ("routing_capacity_admit ", "capacity_override"),
)

#: A guard decision is logged immediately before the dispatch it adjudicates.
GUARD_BEFORE = timedelta(seconds=20)
GUARD_AFTER = timedelta(seconds=10)
#: An override is stamped at the routing decision, just before the dispatch.
OVERRIDE_BEFORE = timedelta(seconds=45)
OVERRIDE_AFTER = timedelta(seconds=10)

#: The inert ``no_budget`` reason: returned when every time bound is disabled.
#: Legacy lines carry ``reason=no_budget``; NEW inert lines (LP-0MVZF4SU10032OZH)
#: carry ``reason=inert_admit reason_inert=no_budget`` and are classed as
#: ``inert_no_budget``, never ``high_priority_inert``.
REASON_NO_BUDGET = "no_budget"

#: The inert admission reason logged for an inert admit (LP-0MVZF4SU10032OZH):
#: the guard admitted without adjudicating (an uncalibrated bucket, every time
#: bound disabled, or the policy off). The underlying reason rides in
#: ``reason_inert=`` and is what ``inert_unlogged`` is now split by.
INERT_ADMIT = "inert_admit"

#: ``reason_inert=`` values carried on a ``reason=inert_admit`` line.
INERT_REASON_INSUFFICIENT_SAMPLES = "insufficient_samples"
INERT_REASON_POLICY_OFF = "policy_off"

#: Stable class vocabulary, in report order. ``high_priority_inert`` requires a
#: logged legacy ``reason=no_budget`` decision; an unlogged ``high`` eviction is
#: ``inert_unlogged`` like any other unlogged priority (LP-0MVZQUT3A0039MLY).
#: The three ``inert_*`` classes (LP-0MVZF4SU10032OZH) split what used to be
#: the single ``inert_unlogged`` catch-all by the logged inert reason, so the
#: self-reinforcing uncalibrated-bucket trap is distinguishable from a
#: nothing-priced ``no_budget`` admit from the log alone.
ADMISSION_CLASSES = (
    "rung_accepted",
    "priced_within_queue_budget",
    "priced_above_bound",
    "idle_override",
    "economic_override",
    "capacity_override",
    "high_priority_inert",
    "inert_no_budget",
    "inert_insufficient_samples",
    "inert_policy_off",
    "inert_unlogged",
)

RUNG_ACCEPTED = "rung_accepted"
PRICED_WITHIN_QUEUE_BUDGET = "priced_within_queue_budget"
PRICED_ABOVE_BOUND = "priced_above_bound"
HIGH_PRIORITY_INERT = "high_priority_inert"
#: (LP-0MVZF4SU10032OZH) the inert classes the old ``inert_unlogged`` split
#: into, keyed by the logged ``reason_inert=`` value:
#: ``no_budget`` (nothing was priced — every time bound disabled),
#: ``insufficient_samples`` (uncalibrated ``(model, bucket)`` window), and
#: ``policy_off`` (guard disabled in config). ``inert_unlogged`` now means
#: "no admission evidence at all at the dispatch instant".
INERT_NO_BUDGET = "inert_no_budget"
INERT_INSUFFICIENT_SAMPLES = "inert_insufficient_samples"
INERT_POLICY_OFF = "inert_policy_off"
INERT_UNLOGGED = "inert_unlogged"

DEFAULT_BUDGET_TOKENS = 25_000
DEFAULT_DEADLINE_SECONDS = 120.0


def _ts(line: str) -> datetime | None:
    match = TS.match(line)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _kv(line: str, start: int) -> dict[str, str]:
    return dict(KV.findall(line[start:]))


def _num(fields: dict[str, str], key: str) -> float | None:
    raw = fields.get(key)
    if raw in (None, "None", "unknown"):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _int(fields: dict[str, str], key: str) -> int:
    raw = fields.get(key)
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return 0


def _lines(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as handle:
        yield from handle


def collect(
    log_dir: str,
    window_start: datetime,
    window_end: datetime,
) -> tuple[dict, dict, dict, list[dict]]:
    """Scan *log_dir* for the window and return the raw evidence by session.

    Returns ``(guard_decisions, overrides, priorities, evictions)``; every
    mapping is keyed by session id and every list keeps the parsed ``key=value``
    fields plus the line timestamp under ``"ts"``. Lines outside
    ``[window_start, window_end]`` are ignored, so a rotated file overlapping
    the window contributes only its in-window part.
    """
    guards: dict[str, list[dict]] = {}
    overrides: dict[str, list[dict]] = {}
    priorities: dict[str, list[tuple[datetime, str]]] = {}
    evictions: list[dict] = []

    for path in sorted(glob.glob(f"{log_dir}/proxy.log*")):
        for line in _lines(path):
            if "=" not in line:
                continue
            ts = _ts(line)
            if ts is None or ts < window_start or ts > window_end:
                continue
            if EVICTION in line:
                fields = _kv(line, line.index(EVICTION))
                fields["ts"] = ts
                evictions.append(fields)
                continue
            if GUARD in line:
                fields = _kv(line, line.index(GUARD))
                fields["ts"] = ts
                fields["verdict"] = line.split("decision=", 1)[1].split()[0]
                guards.setdefault(fields.get("session", ""), []).append(fields)
                continue
            matched = False
            for token, kind in OVERRIDES:
                if token in line:
                    fields = _kv(line, line.index(token))
                    fields["ts"] = ts
                    fields["kind"] = kind
                    overrides.setdefault(fields.get("session", ""), []).append(fields)
                    matched = True
                    break
            if matched:
                continue
            if " priority=" in line and "session=" in line:
                fields = _kv(line, line.index(" priority="))
                session, priority = fields.get("session"), fields.get("priority")
                if session and priority:
                    priorities.setdefault(session, []).append((ts, priority))

    return guards, overrides, priorities, evictions


def _nearest(events: list[dict], start: datetime, before: timedelta,
             after: timedelta, admits_only: bool = False) -> dict | None:
    hits = [
        event
        for event in events
        if start - before <= event["ts"] <= start + after
        and (not admits_only or event.get("verdict") == "admit")
    ]
    return max(hits, key=lambda event: event["ts"]) if hits else None


def classify(
    eviction: dict,
    guard_decisions: dict,
    overrides: dict,
    priorities: dict,
) -> dict:
    """Attribute one eviction event to the admission path that admitted it.

    The dispatch is assumed to have started ``elapsed_s`` before the eviction
    (the proxy measures the deadline from the dispatch start), so the admission
    evidence is looked up around that instant rather than around the eviction.
    """
    session = eviction.get("session", "")
    elapsed = _num(eviction, "elapsed_s") or 0.0
    dispatch_start = eviction["ts"] - timedelta(seconds=elapsed)

    row = {
        "ts": eviction["ts"].isoformat(sep=" "),
        "session": session,
        "estimated_tokens": _int(eviction, "estimated_tokens"),
        "new_tokens": _int(eviction, "new_tokens"),
        "priced_eta_s": _num(eviction, "eta_prefill_s"),
        "elapsed_s": elapsed,
        "dispatch_start": dispatch_start.isoformat(sep=" "),
    }
    seen = [item for item in priorities.get(session, []) if item[0] <= eviction["ts"]]
    row["priority"] = max(seen, key=lambda item: item[0])[1] if seen else None

    session_guards = guard_decisions.get(session, [])
    # A logged inert admit (``reason=inert_admit``, LP-0MVZF4SU10032OZH)
    # proves the guard was consulted and admitted without adjudicating; its
    # ``reason_inert=`` value is the attribution, so it must not enter the
    # priced-ETA lookup below. The legacy bare ``reason=no_budget`` line is
    # likewise not a priced admission — it only recognises the genuine
    # inert-high case (LP-0MVZQUT3A0039MLY). Everything else — including the
    # absence of any guard line — falls through to ``inert_unlogged``.
    inert_admit = _nearest(
        [
            decision
            for decision in session_guards
            if str(decision.get("reason", "")) == INERT_ADMIT
        ],
        dispatch_start, GUARD_BEFORE, GUARD_AFTER,
    )
    guard = _nearest(
        [
            decision
            for decision in session_guards
            if str(decision.get("reason", ""))
            not in (REASON_NO_BUDGET, INERT_ADMIT)
        ],
        dispatch_start, GUARD_BEFORE, GUARD_AFTER, admits_only=True,
    )
    inert_no_budget = _nearest(
        [
            decision
            for decision in session_guards
            if str(decision.get("reason", "")) == REASON_NO_BUDGET
        ],
        dispatch_start, GUARD_BEFORE, GUARD_AFTER,
    )
    override = _nearest(
        overrides.get(session, []), dispatch_start,
        OVERRIDE_BEFORE, OVERRIDE_AFTER,
    )

    if guard is not None:
        projected = _num(guard, "projected_wait_s")
        budget = _num(guard, "budget_s")
        bound = _num(guard, "bound_s")
        row.update(
            guard_projected_s=projected,
            guard_eta_s=_num(guard, "eta_prefill_s"),
            guard_budget_s=budget,
            guard_bound_s=bound,
            guard_deadline_budget_s=_num(guard, "deadline_budget_s"),
        )
        if projected is not None and budget is not None and projected <= budget:
            row["admission"] = PRICED_WITHIN_QUEUE_BUDGET
        elif projected is not None and bound is not None and projected <= bound:
            row["admission"] = RUNG_ACCEPTED
        else:
            row["admission"] = PRICED_ABOVE_BOUND
    elif override is not None:
        # A routing override is the path that ADMITTED the dispatch; an inert
        # guard merely abstained afterwards (the idle-override dispatch still
        # runs through the direct guard, which now logs its inert abstention).
        # The override must therefore win, so ``idle_override`` /
        # ``economic_override`` / ``capacity_override`` keep their exact
        # pre-change attribution.
        row["admission"] = override["kind"]
        row["override_reason"] = override.get("reason")
    elif inert_admit is not None:
        # An inert admit carries the underlying reason in ``reason_inert=``
        # (LP-0MVZF4SU10032OZH AC1) — the split AC2 asks for. When the value
        # is missing/unreadable the event stays ``inert_unlogged``: the split
        # is only as good as the evidence.
        _inert_reason = str(inert_admit.get("reason_inert", ""))
        row["inert_reason"] = _inert_reason
        row.update(guard_eta_s=_num(inert_admit, "eta_prefill_s"))
        if _inert_reason == REASON_NO_BUDGET:
            row["admission"] = INERT_NO_BUDGET
        elif _inert_reason == INERT_REASON_INSUFFICIENT_SAMPLES:
            row["admission"] = INERT_INSUFFICIENT_SAMPLES
        elif _inert_reason == INERT_REASON_POLICY_OFF:
            row["admission"] = INERT_POLICY_OFF
        else:
            row["admission"] = INERT_UNLOGGED
    elif row["priority"] == "high" and inert_no_budget is not None:
        row["admission"] = HIGH_PRIORITY_INERT
    else:
        row["admission"] = INERT_UNLOGGED
    return row


def summarise(
    rows: list[dict],
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
) -> dict:
    """Aggregate attributed rows into the recordable triage summary."""
    counts = Counter(row["admission"] for row in rows)
    tokens = Counter()
    for row in rows:
        tokens[row["admission"]] += row["new_tokens"]
    return {
        "evictions": len(rows),
        "by_admission": {name: counts[name] for name in ADMISSION_CLASSES if counts[name]},
        "new_tokens_by_admission": {
            name: tokens[name] for name in ADMISSION_CLASSES if tokens[name]
        },
        "all_within_cold_prefill_budget": all(
            row["new_tokens"] <= budget_tokens for row in rows
        ),
        "max_new_tokens": max((row["new_tokens"] for row in rows), default=0),
        "priced_eta_over_deadline": sum(
            1 for row in rows if (row["priced_eta_s"] or 0.0) > deadline_seconds
        ),
        "by_priority": dict(
            Counter(row["priority"] or "unknown" for row in rows)
        ),
    }


def attribute(
    log_dir: str,
    window_start: datetime,
    window_end: datetime,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
    include_rows: bool = False,
) -> dict:
    """Attribute every ``prefill_deadline`` eviction in the window."""
    guards, overrides, priorities, evictions = collect(
        log_dir, window_start, window_end
    )
    rows = [
        classify(event, guards, overrides, priorities)
        for event in evictions
        if event.get("reason") == "prefill_deadline"
    ]
    summary = summarise(rows, budget_tokens, deadline_seconds)
    if include_rows:
        summary["rows"] = rows
    return summary


def _parse_iso(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.fromisoformat(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="attribute_prefill_deadline.py",
        description=(
            "Attribute prefill_deadline post-acceptance evictions to the "
            "admission path that let each dispatch start."
        ),
    )
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--until", required=True, help="YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--rows", action="store_true", help="include the per-event attribution"
    )
    args = parser.parse_args(argv)

    window_start = _parse_iso(args.start)
    window_end = _parse_iso(args.until)
    if window_start >= window_end:
        print("error: --start must be before --until")
        return 2
    summary = attribute(
        args.log_dir, window_start, window_end, include_rows=args.rows
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    print(f"prefill_deadline evictions: {summary['evictions']}")
    for name, count in summary["by_admission"].items():
        print(
            f"  {count:4d}  {name:32s} "
            f"({summary['new_tokens_by_admission'][name]} priced new tokens)"
        )
    print(
        "  all within the cold-prefill budget: "
        f"{summary['all_within_cold_prefill_budget']} "
        f"(max new_tokens {summary['max_new_tokens']})"
    )
    print(
        "  priced ETA above the deadline: "
        f"{summary['priced_eta_over_deadline']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
