#!/usr/bin/env python3
"""Admission -> first-byte conversion per new-token band.

Baseline / verification harness for **LP-0MVS8T6F4006DGG3** (parent AC4: prove
the contended-pricing fix against numbers instead of an inferred 85 % figure).
This is a *pure log analyser*: it never changes proxy behaviour and never
requires the proxy to be running. It scans an arbitrary window of the proxy log
stream (``/var/log/llama-proxy``, live + rotated under either naming scheme,
plain or ``.gz``) via :mod:`scripts.lib.proxy_logs`.

Bands are on the **priced new (uncached) tokens** of a local dispatch:
``<8k``, ``8k-20k``, ``20k-32k``, ``32k-64k``, ``>=64k``. The 20k/32k
boundaries mirror ``local_admission_guard_eta_percentile_min_tokens`` and
``CONSERVATIVE_MIN_TOKENS``.

Per band the report carries:

- admission verdicts (``admit`` / ``reject`` and the reject ``reason`` counts),
- **conversion**: sessions with a ``local_admission_guard decision=admit``
  that later emit ``dispatch_first_byte_ms`` (a first byte *without* an admit
  is reported separately, never silently joined),
- **abandoned queue waits**: ``contention_queue_fallback_after_queue`` count
  and its share of the band's queued waits,
- queue-wait seconds.

Queue waits are banded **only from evidence belonging to that wait's own
request** (LP-0MVV29WR0009DY1A): the outcome line's own ``new_tokens``, else the
enqueue line's own ``new_tokens`` for the same wait chain, else an admission for
the same session that precedes the wait and has not already banded an earlier
wait (the legacy join, for windows logged before the enqueue line carried its
own pricing), else ``unknown``. A *later* admission is never borrowed: the
previous rule fell back to the session's latest admission even when it was
hours after the queue event, which banded ``<8k``-class waits into
``32k-64k``. The enqueue count (``queued_waits_enqueued``) is reported so the
abandoned-share denominator is auditable, and ``queue_banding`` records how
many waits were banded from each evidence class.

The parser tolerates pre-change log shapes (no ``rate_bucket`` /
``contended_samples``) and degrades a missing field to ``unknown`` rather than
dropping the event, so the tool runs unchanged against today's log format and
the post-change format.

``local_eviction_after_acceptance`` events are counted too (AC4's second
clause: no *new* eviction class after the change). Only lines whose logging
*message* starts with the marker count, so a proxied request body that merely
mentions the string can never inflate the count. Counts are reported in
total and by ``reason``, plus per new-token band (banded by the evicted
request's own ``new_tokens``, so it is not the same quantity as the admission
band), so the post-change record needs no hand-rolled log scan.

Rotation / decompression is delegated to ``scripts/lib/proxy_logs.py``. The
line parsing is deliberately local and narrowly scoped: the
``proxy-usage-analysis`` parser lives under ``.pi/`` agent infrastructure and
is not imported from repo scripts.

Usage::

    scripts/prefill_admission_conversion.py \
        --since "2026-10-19 00:00:00" --until "2026-10-20 00:00:00" --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:  # script executed directly (``python3 scripts/...``) vs imported as a module
    from lib.proxy_logs import (
        discover_proxy_log_files,
        iter_proxy_log_lines,
        parse_line_timestamp,
    )
except ImportError:  # pragma: no cover - import path for ``scripts`` on sys.path
    from scripts.lib.proxy_logs import (  # type: ignore[no-redef]
        discover_proxy_log_files,
        iter_proxy_log_lines,
        parse_line_timestamp,
    )

DEFAULT_LOG_DIR = Path("/var/log/llama-proxy")

#: ``(label, lower_inclusive, upper_exclusive)``; ``None`` upper = unbounded.
BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("<8k", 0, 8_000),
    ("8k-20k", 8_000, 20_000),
    ("20k-32k", 20_000, 32_000),
    ("32k-64k", 32_000, 64_000),
    (">=64k", 64_000, None),
)

UNKNOWN_BAND = "unknown"

#: Prefix of a logging *message body*: ``<asctime> - [<logger> - ]<LEVEL> - ``.
#: Used to anchor event markers to the start of the message so text that merely
#: *mentions* a marker (a proxied request body, an agent prompt) is never
#: mistaken for the event itself (LP-0MVV9JWL9003A6XR).
_LOG_MESSAGE_PREFIX = re.compile(
    r"(?:^|\s-\s)(?:[A-Za-z0-9_.]+\s-\s)?"
    r"(?:DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\s-\s+"
)

_KV_SESSION = re.compile(r"\bsession=([A-Za-z0-9_.:-]+)")
_KV_NEW_TOKENS = re.compile(r"\bnew_tokens=(\d+)")
_KV_DECISION = re.compile(r"\bdecision=([A-Za-z_]+)")
_KV_REASON = re.compile(r"\breason=([A-Za-z0-9_]+)")
_KV_RATE_BUCKET = re.compile(r"\brate_bucket=([A-Za-z_]+)")
_KV_CONTENDED_SAMPLES = re.compile(r"\bcontended_samples=(\d+)")
_KV_QUEUED = re.compile(r"\bqueued_duration=([\d.]+)s")
_KV_TRIGGER = re.compile(r"\btrigger=([A-Za-z_]+)")

_TS_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",
)


def band_for(new_tokens: int | None) -> str:
    """Return the new-token band label for *new_tokens* (``unknown`` when absent)."""
    if new_tokens is None:
        return UNKNOWN_BAND
    for label, lower, upper in BANDS:
        if new_tokens >= lower and (upper is None or new_tokens < upper):
            return label
    return UNKNOWN_BAND


def _message_starts_with(line: str, marker: str) -> bool:
    """Return ``True`` when *marker* begins the line's logging message body.

    Only the **leftmost** message prefix is considered, so a marker embedded
    inside a message (``- INFO - [remote] POST ... body={"..."}``, an echoed
    prompt that happens to contain ``- ERROR - <marker> reason=...``) can never
    satisfy the anchor: whatever follows the real prefix is the message text.
    A log-format change that stops emitting the ``<asctime> - [<logger> - ]
    <LEVEL> - `` prefix degrades the count to zero (an undercount) rather than
    letting unrelated text inflate it.
    """
    prefix = _LOG_MESSAGE_PREFIX.search(line)
    if prefix is None:
        return False
    return line[prefix.end():].startswith(marker)


def _first(pattern: re.Pattern[str], line: str) -> str | None:
    match = pattern.search(line)
    return match.group(1) if match else None


def _int(pattern: re.Pattern[str], line: str) -> int | None:
    raw = _first(pattern, line)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _float(pattern: re.Pattern[str], line: str) -> float | None:
    raw = _first(pattern, line)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_timestamp(value: str) -> datetime:
    """Parse a CLI ``--since``/``--until`` timestamp (naive local time)."""
    text = value.strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"invalid timestamp {value!r}; use 'YYYY-MM-DD HH:MM:SS' or ISO 8601"
    )


@dataclass
class Event:
    """One parsed log line relevant to the conversion report."""

    # admission | first_byte | queue_wait | queue_dispatch | queue_fallback |
    # eviction
    kind: str
    ts: datetime
    session: str | None
    new_tokens: int | None = None
    decision: str | None = None
    reason: str | None = None
    rate_bucket: str | None = None
    contended_samples: int | None = None
    queued_seconds: float | None = None
    #: Which local gate the request waited on (``generating`` | ``prefill`` |
    #: ``concurrency_limit``), as logged by the enqueue line.
    trigger: str | None = None


def parse_event(line: str, ts: datetime) -> Event | None:
    """Parse one log line into an :class:`Event`, or ``None`` when irrelevant.

    A missing field stays ``None`` rather than dropping the event, so log-format
    drift degrades a metric instead of breaking the report.
    """
    # Anchored to the message start: request-body echoes of this marker (agent
    # prompts quoting an eviction line) inflated the pre-change comparator 3x
    # before this anchor existed (LP-0MVV9JWL9003A6XR).
    if _message_starts_with(line, "local_eviction_after_acceptance"):
        return Event(
            kind="eviction",
            ts=ts,
            session=_first(_KV_SESSION, line),
            new_tokens=_int(_KV_NEW_TOKENS, line),
            reason=_first(_KV_REASON, line),
        )
    if "local_admission_guard decision=" in line:
        return Event(
            kind="admission",
            ts=ts,
            session=_first(_KV_SESSION, line),
            new_tokens=_int(_KV_NEW_TOKENS, line),
            decision=_first(_KV_DECISION, line),
            reason=_first(_KV_REASON, line),
            rate_bucket=_first(_KV_RATE_BUCKET, line),
            contended_samples=_int(_KV_CONTENDED_SAMPLES, line),
        )
    if "dispatch_first_byte_ms=" in line:
        return Event(
            kind="first_byte",
            ts=ts,
            session=_first(_KV_SESSION, line),
            new_tokens=_int(_KV_NEW_TOKENS, line),
        )
    # Queue lines are anchored too (LP-0MVV29WR0009DY1A): the enqueue count is
    # now a reported denominator, so a request body that merely quotes the
    # marker must never inflate it. Every genuine queue line starts its message
    # body, so the anchor changes no existing count.
    if _message_starts_with(line, "contention_queue_wait"):
        return Event(
            kind="queue_wait",
            ts=ts,
            session=_first(_KV_SESSION, line),
            new_tokens=_int(_KV_NEW_TOKENS, line),
            rate_bucket=_first(_KV_RATE_BUCKET, line),
            trigger=_first(_KV_TRIGGER, line),
        )
    if _message_starts_with(line, "contention_queue_fallback_after_queue"):
        return Event(
            kind="queue_fallback",
            ts=ts,
            session=_first(_KV_SESSION, line),
            new_tokens=_int(_KV_NEW_TOKENS, line),
            reason=_first(_KV_REASON, line),
            queued_seconds=_float(_KV_QUEUED, line),
        )
    if _message_starts_with(line, "contention_queue_dispatch"):
        return Event(
            kind="queue_dispatch",
            ts=ts,
            session=_first(_KV_SESSION, line),
            new_tokens=_int(_KV_NEW_TOKENS, line),
            queued_seconds=_float(_KV_QUEUED, line),
        )
    return None


@dataclass
class BandStats:
    """Aggregated counters for one new-token band."""

    admits: int = 0
    rejects: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    sessions_admitted: int = 0
    converted: int = 0
    unconverted: int = 0
    queue_dispatched: int = 0
    queue_abandoned: int = 0
    queue_enqueued: int = 0
    queue_wait_seconds: float = 0.0
    rate_buckets: dict[str, int] = field(default_factory=dict)
    evictions: int = 0

    @property
    def queued_waits(self) -> int:
        return self.queue_dispatched + self.queue_abandoned

    @property
    def conversion_share(self) -> float:
        total = self.converted + self.unconverted
        return (self.converted / total) if total else 0.0

    @property
    def abandoned_share(self) -> float:
        return (self.queue_abandoned / self.queued_waits) if self.queued_waits else 0.0

    def to_dict(self) -> dict:
        return {
            "admission_verdicts": {"admit": self.admits, "reject": self.rejects},
            "reject_reasons": dict(sorted(self.reject_reasons.items())),
            "rate_buckets": dict(sorted(self.rate_buckets.items())),
            "sessions_admitted": self.sessions_admitted,
            "converted": self.converted,
            "unconverted": self.unconverted,
            "conversion_share": round(self.conversion_share, 4),
            "queued_waits": self.queued_waits,
            "queue_dispatched": self.queue_dispatched,
            "queue_abandoned": self.queue_abandoned,
            "queue_enqueued": self.queue_enqueued,
            "abandoned_share": round(self.abandoned_share, 4),
            "queue_wait_seconds": round(self.queue_wait_seconds, 2),
            "evictions": self.evictions,
        }


def _ordered_bands() -> list[str]:
    return [label for label, _, _ in BANDS] + [UNKNOWN_BAND]


def build_report(events: list[Event], *, files: list[str]) -> dict:
    """Aggregate parsed *events* into the per-band conversion report."""
    stats: dict[str, BandStats] = {label: BandStats() for label in _ordered_bands()}

    def band_stats(new_tokens: int | None) -> BandStats:
        return stats[band_for(new_tokens)]

    # Event-level admission verdicts (a session may have several requests).
    for event in events:
        if event.kind != "admission":
            continue
        entry = band_stats(event.new_tokens)
        bucket = event.rate_bucket or "unknown"
        entry.rate_buckets[bucket] = entry.rate_buckets.get(bucket, 0) + 1
        if event.decision == "admit":
            entry.admits += 1
        else:
            entry.rejects += 1
            reason = event.reason or "unknown"
            entry.reject_reasons[reason] = entry.reject_reasons.get(reason, 0) + 1

    # Session-level conversion join: an admit followed (at or after) by a
    # first byte. Match on session; a first byte with no admit is reported
    # separately rather than silently joined.
    admits_by_session: dict[str, list[Event]] = {}
    first_byte_by_session: dict[str, list[Event]] = {}
    first_bytes_without_admit = 0
    for event in events:
        if event.session is None:
            continue
        if event.kind == "admission" and event.decision == "admit":
            admits_by_session.setdefault(event.session, []).append(event)
        elif event.kind == "first_byte":
            first_byte_by_session.setdefault(event.session, []).append(event)

    converted_sessions = 0
    for session, admits in admits_by_session.items():
        admits.sort(key=lambda e: e.ts)
        first_byte = first_byte_by_session.get(session)
        if not first_byte:
            # Admitted but never converted: band by the earliest admit.
            band_stats(admits[0].new_tokens).sessions_admitted += 1
            band_stats(admits[0].new_tokens).unconverted += 1
            continue
        first_byte_ts = min(event.ts for event in first_byte)
        # The band of the admit that preceded the first byte (else the earliest).
        preceding = [event for event in admits if event.ts <= first_byte_ts] or admits
        band_stats(preceding[-1].new_tokens).sessions_admitted += 1
        band_stats(preceding[-1].new_tokens).converted += 1
        converted_sessions += 1

    for session in first_byte_by_session:
        if session not in admits_by_session:
            first_bytes_without_admit += 1

    # Queue outcomes are banded from evidence belonging to *that wait's own
    # request* (LP-0MVV29WR0009DY1A AC1). The previous rule banded a wait by the
    # session's nearest preceding admission and, when none preceded it, by the
    # session's *latest* admission — even one hours later — which put <8k-class
    # waits into ``32k-64k`` (the committed record's phantom row: 45 queued
    # waits, 0 admits).
    #
    # The order of preference is now strictly own-request:
    #
    # 1. the outcome line's own ``new_tokens`` (post-fix log shape);
    # 2. the enqueue line's own ``new_tokens`` for the same wait chain;
    # 3. an admission for the same session that precedes the wait and has not
    #    already been consumed by an earlier wait chain — the legacy join, for
    #    windows logged before the enqueue line carried its own pricing;
    # 4. ``unknown``. A later admission is never borrowed.
    queue_band: dict[int, str] = {}
    banding = {
        "own_outcome": 0,
        "own_enqueue": 0,
        "session_chain": 0,
        "unknown": 0,
    }
    for session_events in _events_by_session(events).values():
        last_admission: Event | None = None
        consumed = False
        chain_new_tokens: int | None = None
        for event in session_events:
            if event.kind == "admission":
                last_admission = event
                consumed = False
                chain_new_tokens = None
                continue
            if event.kind == "queue_wait":
                # A new wait chain starts: its enqueue pricing, when logged,
                # bands the outcome that concludes it.
                chain_new_tokens = event.new_tokens
                continue
            if event.kind not in ("queue_dispatch", "queue_fallback"):
                continue
            if event.new_tokens is not None:
                evidence = event.new_tokens
                banding["own_outcome"] += 1
            elif chain_new_tokens is not None:
                evidence = chain_new_tokens
                banding["own_enqueue"] += 1
            elif last_admission is not None and not consumed:
                evidence = last_admission.new_tokens
                banding["session_chain"] += 1
            else:
                evidence = None
                banding["unknown"] += 1
            queue_band[id(event)] = band_for(evidence)
            # The chain is concluded: neither the admission nor the enqueue
            # evidence may band a later wait.
            chain_new_tokens = None
            consumed = True

    # The enqueue denominator (AC3): every ``contention_queue_wait`` line is one
    # queued wait, so the abandoned share's denominator is auditable against the
    # outcome lines instead of inferred from them. Enqueues are banded by their
    # own pricing where the log carries it; a pre-fix window reports the whole
    # denominator as unbanded rather than guessing.
    enqueued = 0
    enqueued_unbanded = 0
    for event in events:
        if event.kind != "queue_wait":
            continue
        enqueued += 1
        if event.new_tokens is None:
            enqueued_unbanded += 1
            continue
        band_stats(event.new_tokens).queue_enqueued += 1

    evictions_by_reason: dict[str, int] = {}
    for event in events:
        if event.kind != "eviction":
            continue
        reason = event.reason or "unknown"
        evictions_by_reason[reason] = evictions_by_reason.get(reason, 0) + 1
        band_stats(event.new_tokens).evictions += 1

    for event in events:
        if event.kind not in ("queue_dispatch", "queue_fallback"):
            continue
        entry = stats[queue_band.get(id(event), UNKNOWN_BAND)]
        if event.kind == "queue_dispatch":
            entry.queue_dispatched += 1
        else:
            entry.queue_abandoned += 1
        entry.queue_wait_seconds += event.queued_seconds or 0.0

    return {
        "bands": {
            label: stats[label].to_dict()
            for label in _ordered_bands()
            if _band_has_data(stats[label])
        },
        "totals": _totals(stats),
        "queued_waits_enqueued": enqueued,
        "queued_waits_enqueued_unbanded": enqueued_unbanded,
        "queue_banding": banding,
        "evictions": {
            "total": sum(evictions_by_reason.values()),
            "by_reason": dict(sorted(evictions_by_reason.items())),
        },
        "first_byte_without_admit": first_bytes_without_admit,
        "converted_sessions": converted_sessions,
        "log_files": files,
    }


def _events_by_session(events: list[Event]) -> dict[str, list[Event]]:
    """Group *events* by session, chronologically ordered (stable for ties)."""
    grouped: dict[str, list[Event]] = {}
    for event in events:
        if event.session is None:
            continue
        grouped.setdefault(event.session, []).append(event)
    for session_events in grouped.values():
        session_events.sort(key=lambda item: item.ts)
    return grouped


def _band_has_data(entry: BandStats) -> bool:
    return bool(
        entry.admits
        or entry.rejects
        or entry.queued_waits
        or entry.queue_enqueued
        or entry.sessions_admitted
        or entry.evictions
    )


def _totals(stats: dict[str, BandStats]) -> dict:
    total = BandStats()
    for entry in stats.values():
        total.admits += entry.admits
        total.rejects += entry.rejects
        total.sessions_admitted += entry.sessions_admitted
        total.converted += entry.converted
        total.unconverted += entry.unconverted
        total.queue_dispatched += entry.queue_dispatched
        total.queue_abandoned += entry.queue_abandoned
        total.queue_enqueued += entry.queue_enqueued
        total.queue_wait_seconds += entry.queue_wait_seconds
        total.evictions += entry.evictions
    return total.to_dict()


def collect(
    log_dir: Path,
    since: datetime,
    until: datetime,
) -> tuple[list[Event], list[str]]:
    """Scan *log_dir* for events inside ``[since, until]``.

    Returns ``(events, files)`` where *files* are the names of the log files
    that contributed at least one in-window parsed event.
    """
    events: list[Event] = []
    files: set[str] = set()
    for path, line in iter_proxy_log_lines(log_dir):
        ts = parse_line_timestamp(line)
        if ts is None or ts < since or ts > until:
            continue
        event = parse_event(line, ts)
        if event is None:
            continue
        events.append(event)
        files.add(path.name)
    return events, sorted(files)


def analyse(log_dir: Path, since: datetime, until: datetime) -> dict:
    """Return the full JSON report for the window."""
    events, files = collect(log_dir, since, until)
    report = build_report(events, files=files)
    report["since"] = since.isoformat(sep=" ")
    report["until"] = until.isoformat(sep=" ")
    report["log_dir"] = str(log_dir)
    report["log_files_scanned"] = [
        path.name for path in discover_proxy_log_files(log_dir)
    ]
    return report


def render_text(report: dict) -> str:
    """Render a compact human-readable table for the report."""
    lines = [
        f"Window: {report['since']} -> {report['until']}",
        f"Log dir: {report['log_dir']}",
        f"Log files read: {len(report['log_files'])}",
        "",
        (
            f"{'band':<10} {'admit':>6} {'reject':>7} {'conv':>5} {'unconv':>7} "
            f"{'conv%':>6} {'enq':>5} {'queue':>6} {'aband':>6} {'aban%':>6} "
            f"{'wait_s':>8}"
        ),
    ]
    for label, entry in report["bands"].items():
        lines.append(
            f"{label:<10} {entry['admission_verdicts']['admit']:>6} "
            f"{entry['admission_verdicts']['reject']:>7} {entry['converted']:>5} "
            f"{entry['unconverted']:>7} {entry['conversion_share'] * 100:>5.1f}% "
            f"{entry.get('queue_enqueued', 0):>5} "
            f"{entry['queued_waits']:>6} {entry['queue_abandoned']:>6} "
            f"{entry['abandoned_share'] * 100:>5.1f}% "
            f"{entry['queue_wait_seconds']:>8.1f}"
        )
    lines.append("")
    lines.append(
        f"Queued waits enqueued: {report.get('queued_waits_enqueued', 0)} "
        f"(enqueue line carried no pricing: "
        f"{report.get('queued_waits_enqueued_unbanded', 0)})"
    )
    banding = {
        "own_outcome": 0,
        "own_enqueue": 0,
        "session_chain": 0,
        "unknown": 0,
    }
    banding.update(report.get("queue_banding") or {})
    lines.append(
        f"Queue-wait banding: own-outcome={banding['own_outcome']} "
        f"own-enqueue={banding['own_enqueue']} "
        f"session-chain={banding['session_chain']} "
        f"unknown={banding['unknown']}"
    )
    lines.append("")
    lines.append(
        f"First byte without admit: {report['first_byte_without_admit']}"
    )
    evictions = report.get("evictions") or {}
    lines.append(
        f"local_eviction_after_acceptance: {evictions.get('total', 0)} "
        f"{evictions.get('by_reason') or {}}"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Per-new-token-band admission -> first-byte conversion and "
            "abandoned-queue-wait share (LP-0MVS8T6F4006DGG3 baseline harness)."
        )
    )
    parser.add_argument(
        "--since", required=True, type=parse_timestamp, help="window start (inclusive)"
    )
    parser.add_argument(
        "--until", required=True, type=parse_timestamp, help="window end (inclusive)"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"proxy log directory (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.until < args.since:
        print("error: --until must not precede --since", file=sys.stderr)
        return 2
    report = analyse(args.log_dir, args.since, args.until)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
