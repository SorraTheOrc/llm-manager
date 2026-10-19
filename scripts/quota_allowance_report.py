#!/usr/bin/env python3
"""Deterministic post-deploy allowance-utilisation measurement (LP-0MVF7KTMI005B667).

F10 of the quota-aware surge epic (LP-0MVF4FIP1001RD8P, parent AC13). It closes
the loop on the epic's actual outcome: for each ``opencode-*`` account, how much
of the monthly allowance was spent **at the moment of reset** for the first full
period after deployment, and whether the surge steering met its target
(>= 90% monthly consumption at reset, i.e. a residual <= 10% — the margin the
rolling 20% / weekly 50% caps need).

The tool is log-derived and fully deterministic — no LLM in the pipeline,
mirroring the ``.pi/skills/proxy-usage-analysis`` skill's rule-based approach
(its CLI is the behavioural reference; this tool reuses the shared,
rotation-aware log discovery in ``scripts/lib/proxy_logs.py`` so both read the
same corpus).

What the tool reads from ``proxy.log*``
---------------------------------------

- ``quota poll ok account=… rolling=…% weekly=…% monthly=…% seconds_to_reset=…s
  effective_headroom=…`` — per-account poll samples (``proxy/quota_state.py``).
  ``resetsAt`` is derived as ``poll timestamp + seconds_to_reset``, so every
  sample is a (time, monthly %) reading with an authoritative reset anchor.
- ``quota poll failed account=… error=… keeping_last_known=…`` — poll failures
  (the fail-open path), counted per period.
- ``quota_surge_bypass entered|left provider=… account=… … reason=quota_surge``
  — final-week surge transitions (``proxy/provider.py``); paired into intervals
  to measure surge-active time.
- ``Skipping provider=…: reason=quota_exhausted (account=…, monthly_percent=…%,
  …)`` — applied quota-driven skips (the hard stop).
- ``quota_skip dry_run would_skip=true … reason=quota_surge_dry_run`` — dry-run
  no-op decisions, counted separately.
- ``quota surge suspended|resumed account=…`` — stale-state suspension events
  (fail-safe mode).

Output
------

Per account, per monthly period: the monthly percentage from the poll closest
to (and immediately before) the reset, the residual (100 − percent), coverage
quality (how far that last poll was from the reset), surge-active hours (whole
period and final ``--surge-days`` window), hard-stop hours, quota-driven skips,
poll failures and stale suspensions. Each completed period is classified
against the target: ``target_met`` (>= 90%), ``below_target`` (tuning
proposal), ``overspend`` (>= the hard-stop threshold — critical follow-up) or
``insufficient_data``.

AC13d reproducibility: the invocation (argv, log dir, window, filters,
thresholds) is embedded in the JSON artifact, so the same inputs reproduce the
same figures.

Usage::

  ./scripts/quota_allowance_report.py --log-dir /var/log/llama-proxy \\
      --start 2026-10-19T18:00:00 --end 2026-11-24T00:00:00 \\
      --output-md docs/dev/quota-surge-post-deploy-2026-11-23.md \\
      --output-json docs/dev/quota-surge-post-deploy-2026-11-23.json

  ./scripts/quota_allowance_report.py --json > measurement.json   # machine readable

Exit codes:
  0 - success (the report is generated; outcomes may still be below target)
  1 - no quota poll data found in the window / unexpected error
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Shared rotation-aware proxy-log discovery (scripts/lib/proxy_logs.py) — the
# same corpus reader the prompt-cache and usage analysis use.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from lib.proxy_logs import iter_proxy_log_lines  # noqa: E402

UTC = dt.UTC

#: Epic target: at least 90% of the monthly allowance consumed at reset.
TARGET_PERCENT = 90.0
#: Default hard-stop threshold (matches ``quota_exhaustion_percent`` default).
DEFAULT_THRESHOLD = 99.5
#: Default final-week surge window (matches ``quota_surge_days`` default).
DEFAULT_SURGE_DAYS = 7
#: Two polls belong to the same monthly period when their derived resetsAt
#: differ by less than this (the upstream reset time drifts by seconds only).
PERIOD_TOLERANCE = dt.timedelta(hours=6)
#: Default max acceptable gap between the last pre-reset poll and the reset.
DEFAULT_MAX_POLL_GAP_HOURS = 6.0

_TS_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}")

POLL_OK_RE = re.compile(
    r"quota poll ok account=(?P<account>\S+) rolling=(?P<rolling>[0-9.]+)% "
    r"weekly=(?P<weekly>[0-9.]+)% monthly=(?P<monthly>[0-9.]+)% "
    r"seconds_to_reset=(?P<seconds>\d+)s effective_headroom=(?P<eff>[0-9.]+)"
)
POLL_FAIL_RE = re.compile(
    r"quota poll failed account=(?P<account>\S+) error=(?P<error>\S+) "
    r"keeping_last_known=(?P<keeping>\w+)"
)
SURGE_RE = re.compile(
    r"quota_surge_bypass (?P<state>entered|left) provider=(?P<provider>\S+) "
    r"account=(?P<account>\S+) resets_at=(?P<resets>\S+) "
    r"seconds_to_reset=(?P<seconds>\d+)s reason=(?P<reason>\S+)"
)
SKIP_RE = re.compile(
    r"Skipping provider=(?P<provider>[^: ]+): reason=quota_exhausted "
    r"\(account=(?P<account>\S+), monthly_percent=(?P<monthly>[0-9.]+)%, "
    r"effective_headroom=(?P<eff>[0-9.]+)%, threshold=(?P<threshold>[0-9.]+)%\)"
)
SKIP_DRY_RE = re.compile(
    r"quota_skip dry_run would_skip=true provider=(?P<provider>\S+) "
    r"account=(?P<account>\S+) effective_headroom=(?P<eff>[0-9.]+)% "
    r"resets_in=\d+s reason=quota_surge_dry_run"
)
SUSPEND_RE = re.compile(
    r"quota surge suspended account=(?P<account>\S+) "
    r".*?reason=quota_surge_suspended"
)


def parse_proxy_timestamp(line: str) -> dt.datetime | None:
    """Parse a proxy-log line timestamp (``YYYY-MM-DD HH:MM:SS,mmm``, UTC)."""
    match = _TS_RE.match(line)
    if not match:
        return None
    try:
        return dt.datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=UTC
        )
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Events parsed from the log
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PollSample:
    """One ``quota poll ok`` reading for an account."""

    ts: dt.datetime
    account: str
    rolling: float
    weekly: float
    monthly: float
    seconds_to_reset: int
    effective_headroom: float

    @property
    def resets_at(self) -> dt.datetime:
        """The monthly reset time observed at poll time (``ts`` + countdown)."""
        return self.ts + dt.timedelta(seconds=self.seconds_to_reset)


@dataclass
class Events:
    """Everything parsed from the logs for one account, time ordered."""

    account: str
    samples: list[PollSample] = field(default_factory=list)
    poll_failures: list[dt.datetime] = field(default_factory=list)
    skips: list[dt.datetime] = field(default_factory=list)
    dry_run_skips: list[dt.datetime] = field(default_factory=list)
    suspensions: list[dt.datetime] = field(default_factory=list)
    surge_entered: list[dt.datetime] = field(default_factory=list)
    surge_left: list[dt.datetime] = field(default_factory=list)


@dataclass
class Period:
    """One monthly allowance period of one account, anchored on its reset."""

    account: str
    resets_at: dt.datetime
    samples: list[PollSample] = field(default_factory=list)
    poll_failures: int = 0
    skips: int = 0
    dry_run_skips: int = 0
    suspensions: int = 0
    surge_intervals: list[tuple[dt.datetime, dt.datetime]] = field(
        default_factory=list
    )
    #: Start of the allowance period. For every period after the first it is
    #: the previous period's ``resetsAt`` (the true monthly boundary); for the
    #: first observed period it falls back to the first sample (measurement
    #: starts at deployment, so surge before it is outside the corpus).
    start_at: dt.datetime | None = None

    @property
    def first_sample_at(self) -> dt.datetime | None:
        return self.samples[0].ts if self.samples else None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_events(
    log_dir: Path, start: dt.datetime, end: dt.datetime
) -> dict[str, Events]:
    """Parse every quota line in ``[start, end)`` into per-account events.

    Only lines carrying a parseable proxy timestamp inside the window are
    considered, so rotated files whose names overstate their content cannot
    leak out-of-window readings (the shared reader performs no time filtering
    of its own).
    """
    events: dict[str, Events] = {}

    def bucket(account: str) -> Events:
        return events.setdefault(account, Events(account=account))

    for _path, line in iter_proxy_log_lines(log_dir):
        ts = parse_proxy_timestamp(line)
        if ts is None or ts < start or ts >= end:
            continue
        if "quota" not in line:
            continue

        match = POLL_OK_RE.search(line)
        if match:
            bucket(match.group("account")).samples.append(
                PollSample(
                    ts=ts,
                    account=match.group("account"),
                    rolling=float(match.group("rolling")),
                    weekly=float(match.group("weekly")),
                    monthly=float(match.group("monthly")),
                    seconds_to_reset=int(match.group("seconds")),
                    effective_headroom=float(match.group("eff")),
                )
            )
            continue

        match = POLL_FAIL_RE.search(line)
        if match:
            bucket(match.group("account")).poll_failures.append(ts)
            continue

        match = SURGE_RE.search(line)
        if match:
            entry = bucket(match.group("account"))
            if match.group("state") == "entered":
                entry.surge_entered.append(ts)
            else:
                entry.surge_left.append(ts)
            continue

        match = SKIP_RE.search(line)
        if match:
            bucket(match.group("account")).skips.append(ts)
            continue

        match = SKIP_DRY_RE.search(line)
        if match:
            bucket(match.group("account")).dry_run_skips.append(ts)
            continue

        match = SUSPEND_RE.search(line)
        if match:
            bucket(match.group("account")).suspensions.append(ts)

    return events


# ---------------------------------------------------------------------------
# Period derivation
# ---------------------------------------------------------------------------


def period_anchors(samples: list[PollSample]) -> list[dt.datetime]:
    """Distinct ``resetsAt`` anchors for an account, in time order.

    Consecutive polls inside one monthly period report the same reset time (up
    to seconds of drift), so anchors are clustered with :data:`PERIOD_TOLERANCE`.
    A rollover moves the anchor forward by roughly a month and starts a new
    period.
    """
    anchors: list[dt.datetime] = []
    for resets_at in sorted(sample.resets_at for sample in samples):
        if not anchors or (resets_at - anchors[-1]) > PERIOD_TOLERANCE:
            anchors.append(resets_at)
        else:
            # Keep the latest observed value within the cluster: the upstream
            # reset instant is authoritative and settles as the period ends.
            anchors[-1] = max(anchors[-1], resets_at)
    return anchors


def _period_index(anchors: list[dt.datetime], ts: dt.datetime) -> int | None:
    """Index of the period *ts* belongs to: the next anchor at or after ``ts``.

    An event that happens before a reset belongs to the allowance period that
    reset ends. Events after the last known anchor belong to no period (their
    period's reset has not been observed yet).
    """
    index = bisect.bisect_left(anchors, ts)
    return index if index < len(anchors) else None


def build_periods(events: Events, surge_window: dt.timedelta) -> list[Period]:
    """Group one account's events into monthly periods anchored on its resets."""
    if not events.samples:
        return []
    samples = sorted(events.samples, key=lambda sample: sample.ts)
    anchors = period_anchors(samples)
    periods = []
    for index, anchor in enumerate(anchors):
        period = Period(account=events.account, resets_at=anchor)
        period.start_at = anchors[index - 1] if index > 0 else None
        periods.append(period)

    for sample in samples:
        index = _period_index(anchors, sample.ts)
        if index is not None:
            periods[index].samples.append(sample)

    for ts in events.poll_failures:
        index = _period_index(anchors, ts)
        if index is not None:
            periods[index].poll_failures += 1
    for ts in events.skips:
        index = _period_index(anchors, ts)
        if index is not None:
            periods[index].skips += 1
    for ts in events.dry_run_skips:
        index = _period_index(anchors, ts)
        if index is not None:
            periods[index].dry_run_skips += 1
    for ts in events.suspensions:
        index = _period_index(anchors, ts)
        if index is not None:
            periods[index].suspensions += 1

    # Pair surge transitions into intervals, then clip each to a period.
    intervals = _pair_intervals(
        sorted(events.surge_entered), sorted(events.surge_left)
    )
    for start_at, end_at in intervals:
        index = _period_index(anchors, start_at)
        if index is None:
            continue
        periods[index].surge_intervals.append((start_at, end_at))

    for period in periods:
        period.surge_intervals = [
            (_clip(interval[0], period), _clip(interval[1], period))
            for interval in period.surge_intervals
        ]
        period.surge_intervals = [
            (start_at, end_at)
            for start_at, end_at in period.surge_intervals
            if end_at > start_at
        ]
    return periods


def _clip(ts: dt.datetime, period: Period) -> dt.datetime:
    """Clamp *ts* into the period's ``[start_at, resets_at]`` span.

    Surge events are attributed to the allowance period they fall inside;
    events after the reset belong to the next period and events before the
    period's start (previous reset) are dropped. The first observed period has
    no prior reset, so its lower bound is its first sample (measurement starts
    at deployment).
    """
    start_at = period.start_at or period.first_sample_at or (
        period.resets_at - dt.timedelta(days=30)
    )
    return min(max(ts, start_at), period.resets_at)


def _pair_intervals(
    entered: list[dt.datetime], left: list[dt.datetime]
) -> list[tuple[dt.datetime, dt.datetime]]:
    """Pair ``entered``/``left`` transitions into ``(start, end)`` intervals.

    Order-based per account, matching the once-per-transition logging in
    ``proxy/provider.py``. A re-entry while active moves the open edge; a
    ``left`` with no matching open state is ignored. An ``entered`` with no
    matching ``left`` stays open and is reported as a zero-length interval
    (presence, not duration).
    """
    intervals: list[tuple[dt.datetime, dt.datetime]] = []
    open_at: dt.datetime | None = None
    for ts in sorted(entered + left):
        if open_at is None:
            if ts in entered:
                open_at = ts
        elif ts in left:
            if ts > open_at:
                intervals.append((open_at, ts))
            open_at = None
        elif ts in entered:
            open_at = ts
    if open_at is not None:
        intervals.append((open_at, open_at))
    return intervals


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def classify(percent_at_reset: float | None, threshold: float) -> str:
    """Outcome label for one period (deterministic, threshold-aware)."""
    if percent_at_reset is None:
        return "insufficient_data"
    if percent_at_reset >= threshold:
        return "overspend"
    if percent_at_reset >= TARGET_PERCENT:
        return "target_met"
    return "below_target"


def build_markdown(measurement: dict, meta: dict) -> str:
    """Operator-facing Markdown report (deterministic; no LLM wording)."""
    threshold = meta["threshold_percent"]
    lines: list[str] = [
        f"# Quota surge post-deploy allowance utilisation — {meta['window_end'][:10]}",
        "",
        f"Generated {meta['generated_at']} by `scripts/quota_allowance_report.py` "
        f"(F10, LP-0MVF7KTMI005B667).",
        "",
        f"**Target:** ≥ {TARGET_PERCENT:.0f}% monthly consumption at reset per "
        f"account (residual ≤ {100 - TARGET_PERCENT:.0f}% — the margin the rolling "
        f"20% / weekly 50% caps need). **Hard stop:** {threshold:g}% "
        f"(`quota_exhaustion_percent`).",
        "",
        "## Invocation (reproducibility, AC13d)",
        "",
        "```bash",
        meta["argv"],
        "```",
        "",
        f"- log dir: `{meta['log_dir']}`",
        f"- window: `{meta['window_start']}` → `{meta['window_end']}`",
        f"- filters: threshold `{threshold:g}%`, surge window "
        f"`{meta['surge_days']}` days, max poll gap "
        f"`{meta['max_poll_gap_hours']}` h",
        f"- machine-readable data: `{meta['json_path'] or '(none)'}`",
        "",
        "## Per-account results",
        "",
    ]

    completed = [
        (account, m)
        for account in sorted(measurement["accounts"])
        for m in measurement["accounts"][account]
        if m["completed"]
    ]
    if completed:
        lines.extend(
            [
                "| Account | Reset | % at reset | Residual | Surge (final week, h) "
                "| Hard stop (h) | Skips | Poll fails | Outcome |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for account, m in completed:
            lines.append(
                f"| {account} | {m['resets_at'][:10]} "
                f"| {m['monthly_percent_at_reset']:g} "
                f"| {m['residual_percent']:g} "
                f"| {m['surge_final_window_hours']:g} "
                f"| {m['hard_stop_hours']:g} "
                f"| {m['quota_skip_count']} | {m['poll_failures']} "
                f"| {m['outcome']} |"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "No monthly period has completed inside the analysed window yet.",
                "",
            ]
        )

    lines.extend(["### Periods in the window", ""])
    for account in sorted(measurement["accounts"]):
        lines.append(f"**{account}**")
        lines.append("")
        for m in measurement["accounts"][account]:
            state = "completed" if m["completed"] else "in progress"
            if m["monthly_percent_at_reset"] is not None:
                percent = (
                    f"**{m['monthly_percent_at_reset']:g}% at reset** "
                    f"(residual {m['residual_percent']:g}%)"
                )
            elif m["monthly_percent_latest"] is not None:
                percent = (
                    f"latest reading {m['monthly_percent_latest']:g}% "
                    f"({m['coverage_gap_hours']} h before the scheduled reset; "
                    f"not yet a moment-of-reset measurement)"
                )
            else:
                percent = "no reading"
            lines.append(
                f"- reset `{m['resets_at']}` ({state}): {percent}, "
                f"surge {m['surge_active_hours']:g} h "
                f"(final window {m['surge_final_window_hours']:g} h), "
                f"hard stop {m['hard_stop_hours']:g} h, "
                f"skips {m['quota_skip_count']} "
                f"(dry-run {m['quota_skip_dry_run_count']}), "
                f"poll failures {m['poll_failures']}, "
                f"stale suspensions {m['stale_suspensions']} → **{m['outcome']}**"
            )
        lines.append("")

    lines.extend(["## Outcome notes", ""])
    notes = [_outcome_note(account, m, threshold) for account, m in completed]
    if notes:
        lines.extend(notes)
    else:
        lines.append(
            "- Nothing to classify yet: no account has crossed a monthly reset "
            "inside this window. Re-run after each account's next reset "
            "(schedule in `docs/dev/quota-surge.md` §9.1)."
        )
    lines.append("")

    summary = measurement["summary"]
    lines.extend(
        [
            "## Summary",
            "",
            f"- accounts measured: {summary['accounts_measured']}",
            f"- completed periods: {summary['completed_periods']}",
            f"- target met (≥ {TARGET_PERCENT:.0f}%): {summary['target_met']}",
            f"- below target: {summary['below_target']}",
            f"- overspend (≥ {threshold:g}%): {summary['overspend']}",
            f"- insufficient data: {summary['insufficient_data']}",
            "",
        ]
    )
    return "\n".join(lines)


def _outcome_note(account: str, m: dict, threshold: float) -> str:
    """One deterministic bullet per completed period with a classification."""
    percent = m["monthly_percent_at_reset"]
    if m["outcome"] == "overspend":
        return (
            f"- **{account}** finished at {percent:g}% — at/above the hard stop "
            f"({threshold:g}%). The overspend guard did not hold: a **critical** "
            f"follow-up is required (AC13c)."
        )
    if m["outcome"] == "below_target":
        return (
            f"- **{account}** finished at {percent:g}% (< {TARGET_PERCENT:g}%), "
            f"residual {m['residual_percent']:g}%. Follow-up: raise "
            f"`quota_pacing_target`, widen `quota_surge_days` and/or start the "
            f"ramp earlier — with this observed data attached (AC13c)."
        )
    if m["outcome"] == "target_met":
        return f"- **{account}** finished at {percent:g}% — target met."
    return (
        f"- **{account}** — no usable reading at reset "
        f"(coverage gap {m['coverage_gap_hours']} h)."
    )


def measure_period(
    period: Period,
    threshold: float,
    surge_window: dt.timedelta,
    now: dt.datetime,
    max_poll_gap: dt.timedelta,
) -> dict:
    """Compute the reported figures for one period.

    ``monthly_percent_at_reset`` is the monthly percent from the poll closest
    to (and immediately before) ``resets_at`` — the epic's moment-of-reset
    measurement, reported only once the reset has actually happened.
    ``monthly_percent_latest`` is the last reading before the reset anchor
    (available for in-progress periods too, for context). ``coverage_gap_hours``
    quantifies how close that reading was to the reset; a gap beyond
    ``max_poll_gap`` marks it ``approximate``.
    """
    samples = sorted(period.samples, key=lambda sample: sample.ts)
    before = [sample for sample in samples if sample.ts <= period.resets_at]
    last = before[-1] if before else None
    gap = (period.resets_at - last.ts) if last else None
    gap_hours = round(gap.total_seconds() / 3600.0, 2) if gap else None

    hard_stop_seconds = _sustained_seconds(samples, threshold)
    surge_window_start = period.resets_at - surge_window
    surge_final_seconds = sum(
        _overlap(start_at, end_at, surge_window_start, period.resets_at)
        for start_at, end_at in period.surge_intervals
    )
    surge_seconds = sum(
        (end_at - start_at).total_seconds() for start_at, end_at in period.surge_intervals
    )

    # The reset must have happened for there to be a moment-of-reset
    # measurement — a projected reading is never reported as one.
    completed = period.resets_at <= now
    percent_at_reset = last.monthly if (completed and last) else None
    outcome = classify(percent_at_reset, threshold) if completed else "in_progress"
    approximate = bool(gap and gap > max_poll_gap)
    return {
        "account": period.account,
        "resets_at": _iso(period.resets_at),
        "period_start": _iso(period.start_at or period.first_sample_at or period.resets_at),
        "completed": completed,
        "poll_count": len(samples),
        "monthly_percent_at_reset": percent_at_reset,
        "monthly_percent_latest": last.monthly if last else None,
        "effective_headroom_at_reset": (
            last.effective_headroom if (completed and last) else None
        ),
        "residual_percent": (
            round(100.0 - percent_at_reset, 1)
            if percent_at_reset is not None
            else None
        ),
        "coverage_gap_hours": gap_hours,
        "reading_approximate": approximate,
        "surge_active_hours": round(surge_seconds / 3600.0, 2),
        "surge_final_window_hours": round(surge_final_seconds / 3600.0, 2),
        "hard_stop_hours": round(hard_stop_seconds / 3600.0, 2),
        "quota_skip_count": period.skips,
        "quota_skip_dry_run_count": period.dry_run_skips,
        "poll_failures": period.poll_failures,
        "stale_suspensions": period.suspensions,
        "outcome": outcome,
    }


def _sustained_seconds(samples: list[PollSample], threshold: float) -> float:
    """Seconds spent at/above *threshold*, from consecutive-sample dwell time.

    Each consecutive pair both at/above the threshold contributes the gap
    between them, so the figure is the observed dwell time at the hard stop
    (bounded by the poll interval at both ends).
    """
    total = 0.0
    previous: PollSample | None = None
    for sample in samples:
        if sample.monthly >= threshold - 1e-9:
            if previous is not None and previous.monthly >= threshold - 1e-9:
                total += (sample.ts - previous.ts).total_seconds()
            previous = sample
        else:
            previous = None
    return total


def _overlap(
    start_a: dt.datetime, end_a: dt.datetime, start_b: dt.datetime, end_b: dt.datetime
) -> float:
    """Seconds of ``[start_a, end_a]`` that fall inside ``[start_b, end_b]``."""
    start = max(start_a, start_b)
    end = min(end_a, end_b)
    return max(0.0, (end - start).total_seconds())


def _iso(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def run_measurement(
    log_dir: Path,
    start: dt.datetime,
    end: dt.datetime,
    threshold: float,
    surge_days: int,
    max_poll_gap_hours: float,
    now: dt.datetime | None = None,
) -> dict:
    """Parse *log_dir* and return the per-account measurement (pure)."""
    surge_window = dt.timedelta(days=surge_days)
    max_poll_gap = dt.timedelta(hours=max_poll_gap_hours)
    reference = now or end

    accounts: dict[str, list[dict]] = {}
    events_by_account = parse_events(log_dir, start, end)
    for account, events in events_by_account.items():
        periods = build_periods(events, surge_window)
        if not periods:
            continue
        accounts[account] = [
            measure_period(period, threshold, surge_window, reference, max_poll_gap)
            for period in periods
        ]

    outcomes = [
        m["outcome"] for periods in accounts.values() for m in periods if m["completed"]
    ]
    summary = {
        "accounts_measured": len(accounts),
        "completed_periods": len(outcomes),
        "target_met": outcomes.count("target_met"),
        "below_target": outcomes.count("below_target"),
        "overspend": outcomes.count("overspend"),
        "insufficient_data": outcomes.count("insufficient_data"),
    }
    return {"accounts": accounts, "summary": summary}


def build_artifact(measurement: dict, meta: dict) -> dict:
    """The machine-readable JSON artifact (AC13b data + AC13d invocation)."""
    return {
        "generated_at": meta["generated_at"],
        "tool": "scripts/quota_allowance_report.py",
        "work_item": "LP-0MVF7KTMI005B667",
        "target_percent": TARGET_PERCENT,
        "invocation": {
            "argv": meta["argv"],
            "log_dir": meta["log_dir"],
            "window": {"start": meta["window_start"], "end": meta["window_end"]},
            "threshold_percent": meta["threshold_percent"],
            "surge_days": meta["surge_days"],
            "max_poll_gap_hours": meta["max_poll_gap_hours"],
        },
        "summary": measurement["summary"],
        "accounts": measurement["accounts"],
    }


def _parse_iso(value: str) -> dt.datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic post-deploy allowance-utilisation measurement from "
            "proxy logs (LP-0MVF7KTMI005B667, epic parent AC13)."
        )
    )
    parser.add_argument(
        "--log-dir", default="/var/log/llama-proxy",
        help="directory containing proxy.log* (default /var/log/llama-proxy)",
    )
    parser.add_argument(
        "--start", default=None,
        help="window start (ISO-8601; default: window end minus --days)",
    )
    parser.add_argument(
        "--end", default=None,
        help="window end (ISO-8601; default: --now or the current UTC time)",
    )
    parser.add_argument(
        "--now", default=None,
        help=(
            "freeze 'now' for period-completion decisions (ISO-8601); also "
            "serves as the default window end"
        ),
    )
    parser.add_argument(
        "--days", type=int, default=62,
        help="window length in days when --start is omitted (default 62)",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"hard-stop percent (default {DEFAULT_THRESHOLD:g})",
    )
    parser.add_argument(
        "--surge-days", type=int, default=DEFAULT_SURGE_DAYS,
        help=f"final-week surge window in days (default {DEFAULT_SURGE_DAYS})",
    )
    parser.add_argument(
        "--max-poll-gap-hours", type=float, default=DEFAULT_MAX_POLL_GAP_HOURS,
        help=(
            "gap beyond which the reading at reset is flagged approximate "
            f"(default {DEFAULT_MAX_POLL_GAP_HOURS:g} h)"
        ),
    )
    parser.add_argument("--output-md", default=None, help="Markdown report path")
    parser.add_argument("--output-json", default=None, help="JSON artifact path")
    parser.add_argument(
        "--json", action="store_true", help="print the JSON artifact to stdout"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress the stdout text summary"
    )
    args = parser.parse_args(raw_argv)

    now = _parse_iso(args.now) if args.now else None
    end = _parse_iso(args.end) if args.end else (
        now or dt.datetime.now(UTC)
    )
    start = (
        _parse_iso(args.start) if args.start else end - dt.timedelta(days=args.days)
    )

    meta = {
        "generated_at": _iso(dt.datetime.now(UTC)) or "",
        "log_dir": args.log_dir,
        "window_start": _iso(start) or "",
        "window_end": _iso(end) or "",
        "threshold_percent": args.threshold,
        "surge_days": args.surge_days,
        "max_poll_gap_hours": args.max_poll_gap_hours,
        "argv": " ".join(["scripts/quota_allowance_report.py", *raw_argv]),
        "json_path": args.output_json or "",
    }

    measurement = run_measurement(
        Path(args.log_dir),
        start,
        end,
        args.threshold,
        args.surge_days,
        args.max_poll_gap_hours,
        now=now or end,
    )
    if not measurement["accounts"]:
        print(
            f"no quota poll data in {args.log_dir} for "
            f"{meta['window_start']} → {meta['window_end']}",
            file=sys.stderr,
        )
        return 1

    artifact = build_artifact(measurement, meta)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build_markdown(measurement, meta))
    if args.json:
        print(json.dumps(artifact, indent=2, sort_keys=True))
    if not args.quiet:
        summary = measurement["summary"]
        print(
            f"accounts={summary['accounts_measured']} "
            f"completed_periods={summary['completed_periods']} "
            f"target_met={summary['target_met']} "
            f"below_target={summary['below_target']} "
            f"overspend={summary['overspend']} "
            f"insufficient_data={summary['insufficient_data']}"
        )
        if args.output_md:
            print(f"report: {args.output_md}")
        if args.output_json:
            print(f"json:   {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
