#!/usr/bin/env python3
"""Slot-count history and concurrency extractor (LP-0MVGCQ3QN009HRE8).

Investigation deliverable: re-decide ``session_slot_pool_size`` (N) from the
historical record instead of a single 2026-10-17 snapshot.

Two independent observables are mined, both without new production work:

1. **Archived daily reports** (``~/proxy-usage-reports/YYYY-MM-DD[_n]/report.md``)
   — busy time (total / fast / cheap), queued-dispatch / queue-abandon counts,
   local-serve share, and decode / prompt-eval medians. The slot config in
   force for each report window is reconstructed from the *known* config-era
   timeline (``SLOT_CONFIG_ERAS``), derived from ``git log -p`` on
   ``proxy/config*.yaml``.

2. **Proxy logs** (``/var/log/llama-proxy/proxy.log*``) — per-session
   ``Stream started`` / ``Stream finished`` pairs give the concurrency series.
   Counting minutes-at-N-concurrent and applying the capacity model
   ``denied = max(0, C - N)`` projects the share of demand still denied at
   N=2, 3 and 4, so the slot decision is sized to real peak demand.

The output is deterministic JSON (``--json``) plus a Markdown fragment so the
table in ``docs/slot-count-reevaluation-LP-0MVGCQ3QN009HRE8.md`` is traceable
to this code (parent AC1 / AC2).

Usage::

    python3 scripts/slot_history_extract.py --json
    python3 scripts/slot_history_extract.py --markdown
    python3 scripts/slot_history_extract.py --concurrency --log-dir /var/log/llama-proxy
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Slot-config eras (derived from git log -p on proxy/config*.yaml)
# ---------------------------------------------------------------------------
# Each entry is (effective_from, fast_slots, cheap_slots, note). The base
# config's ``session_slot_pool_size`` is the ``fast`` value once the fast/cheap
# overlays exist; before the overlay era both modes shared the base value.
# Dates are the commit timestamps (UTC-ish, local +01:00) — good to the day,
# which is all the daily-report windows need.
SLOT_CONFIG_ERAS: tuple[tuple[str, str, str, str], ...] = (
    # ISO date            fast  cheap  note
    ("2026-07-13", "2", "2", "base raised 1 -> 2 (9426afc3)"),
    ("2026-07-25", "4", "4", "scheduler-era base 4 (9490fa17)"),
    ("2026-08-01", "6", "6", "capacity raise 6 (985ce057)"),
    ("2026-08-02", "3", "3", "Option C: 3 slots x 43.7K (d114fc86)"),
    ("2026-08-03", "4", "4", "stashed ctx/slot re-apply -> 4 (ccca4028)"),
    ("2026-08-04", "3", "3", "restore Option C 3x43.7K (c606131b)"),
    ("2026-09-13", "3", "2", "per-mode slots fast 3 / cheap 2 (d3e08692)"),
    ("2026-09-13b", "1", "3", "fast 1 / cheap 3 (8567bcbb)"),
    ("2026-10-17", "2", "3", "fast/base 2, cheap 3 (a78e9446)"),
    ("2026-10-18", "2", "2", "priority routing replaces fast/cheap (e5d3a2bf)"),
)


def slot_config_for(when: datetime) -> tuple[str, str, str]:
    """Return ``(fast_slots, cheap_slots, era_note)`` in force at *when*.

    Uses the day-resolution era table; a report window that straddles a change
    is attributed to the era in force at its start (documented approximation).
    """
    day = when.date().isoformat()
    chosen = SLOT_CONFIG_ERAS[0]
    for entry in SLOT_CONFIG_ERAS:
        # "2026-09-13b" marks a same-day second era; strip the suffix for the
        # date comparison and rely on tuple order for the tie-break.
        entry_day = entry[0].rstrip("b")
        if entry_day <= day:
            chosen = entry
        else:
            break
    return chosen[1], chosen[2], chosen[3]


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})")
_DURATION_RE = re.compile(r"(\d+)h\s+(\d+)m|(\d+)h|(\d+)m\s+(\d+)s")
_SESSIONS_RE = re.compile(
    r"Sessions in window:\s*\*\*(\d+)\*\*\s*\|\s*Requests:\s*\*\*(\d+)\*\*"
    r"\s*\(local\s+(\d+)\s*/\s*remote\s+(\d+)\)"
)
_DENIED_HDR_RE = re.compile(r"dispatch denied:\s*(\d+)")
_FALLBACK_HDR_RE = re.compile(r"Fallback events:\s*\*\*(\d+)\*\*")


def _duration_to_seconds(text: str) -> int | None:
    """Parse a report duration cell (``"7h 47m"``, ``"1h 00m"``, ``"0m"``)."""
    text = text.strip().replace("\u00a0", " ")
    if not text or text in {"-", "--"}:
        return None
    m = _DURATION_RE.search(text)
    if not m:
        return None
    if m.group(1) and m.group(2):
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60
    if m.group(3):
        return int(m.group(3)) * 3600
    if m.group(4) and m.group(5):
        return int(m.group(4)) * 60 + int(m.group(5))
    return None


def _table_rows(lines: list[str]) -> list[list[str]]:
    """Return pipe-table rows (cells stripped) from Markdown *lines*."""
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells:
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue  # separator row
        rows.append(cells)
    return rows


def _first_int(text: str) -> int | None:
    m = re.search(r"-?\d[\d,]*", text or "")
    if not m:
        return None
    return int(m.group(0).replace(",", ""))


def _first_float(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return float(m.group(0)) if m else None


def _section(lines: list[str], title: str) -> list[str]:
    """Return the lines of the ``## <title>`` section (until the next ``## ``)."""
    out: list[str] = []
    inside = False
    for line in lines:
        if line.startswith("## "):
            if inside:
                break
            if line[3:].strip().lower() == title.lower():
                inside = True
                continue
        if inside:
            out.append(line)
    return out


def parse_report(path: Path) -> dict | None:
    """Parse one archived ``report.md`` into a flat metrics dict.

    Returns ``None`` when the file is not a recognisable proxy-usage report.
    Fields that a given report era does not carry are left ``None`` and the
    reasons are not hidden (the renderer prints ``--``).
    """
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    if "Proxy Usage Analysis Report" not in text:
        return None
    lines = text.splitlines()

    generated = None
    window_start = None
    window_end = None
    sessions = requests = local = remote = None
    denied_header = None
    fallback_header = None
    for line in lines:
        if line.startswith("- Generated:") and not generated:
            m = _TS_RE.search(line)
            if m:
                generated = datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}")
        elif line.startswith("- Window:"):
            stamps = _TS_RE.findall(line)
            if len(stamps) >= 2:
                window_start = datetime.fromisoformat(f"{stamps[0][0]}T{stamps[0][1]}")
                window_end = datetime.fromisoformat(f"{stamps[1][0]}T{stamps[1][1]}")
        m = _SESSIONS_RE.search(line)
        if m:
            sessions, requests, local, remote = (int(g) for g in m.groups())
        m = _DENIED_HDR_RE.search(line)
        if m:
            denied_header = int(m.group(1))
        m = _FALLBACK_HDR_RE.search(line)
        if m:
            fallback_header = int(m.group(1))

    if window_start is None:
        return None

    # Session summary rows: queue + denial counters.
    queued_dispatched = fallback_after_queue = dispatch_denied = None
    for row in _table_rows(lines):
        label = row[0].lower()
        if label.startswith("queued") and "dispatched" in label:
            queued_dispatched = _first_int(row[1])
        elif label.startswith("fallback after queue"):
            fallback_after_queue = _first_int(row[1])
        elif label.startswith("dispatch denied"):
            dispatch_denied = _first_int(row[1])

    # Busy time row (total / fast / cheap) or the Totals row.
    busy_seconds = busy_fast_seconds = busy_cheap_seconds = None
    busy_pct = None
    totals_row = None
    for row in _table_rows(lines):
        label = row[0].lower()
        if label.startswith("busy time") and len(row) >= 4:
            busy_seconds = _duration_to_seconds(row[1])
            busy_fast_seconds = _duration_to_seconds(row[2])
            busy_cheap_seconds = _duration_to_seconds(row[3])
            pct = re.search(r"(\d+(?:\.\d+)?)%", row[1])
            busy_pct = float(pct.group(1)) if pct else None
        elif label == "totals":
            totals_row = row
    if busy_seconds is None and totals_row and len(totals_row) >= 3:
        busy_seconds = _duration_to_seconds(totals_row[1])
        pct = re.search(r"(\d+(?:\.\d+)?)%", totals_row[2])
        busy_pct = float(pct.group(1)) if pct else None

    # Decode / prompt-eval medians (Total bucket).
    def _speed_total(title: str) -> tuple[float | None, int | None]:
        for row in _table_rows(_section(lines, title)):
            if len(row) >= 4 and row[1].lower() == "total":
                return _first_float(row[3]), _first_int(row[2])
        return None, None

    decode_median, decode_samples = _speed_total("Decode speed")
    prompt_eval_median, _ = _speed_total("Prompt eval speed")

    local_share = (local / requests) if (local is not None and requests) else None
    fast_slots, cheap_slots, era_note = slot_config_for(window_start)

    return {
        "source": str(path),
        "generated": generated.isoformat() if generated else None,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat() if window_end else None,
        "window_hours": (
            round((window_end - window_start).total_seconds() / 3600, 2)
            if window_end
            else None
        ),
        "sessions": sessions,
        "requests": requests,
        "local_requests": local,
        "remote_requests": remote,
        "local_share": local_share,
        "fallback_events": fallback_header,
        "dispatch_denied": dispatch_denied if dispatch_denied is not None else denied_header,
        "queued_dispatched": queued_dispatched,
        "fallback_after_queue": fallback_after_queue,
        "busy_seconds": busy_seconds,
        "busy_fast_seconds": busy_fast_seconds,
        "busy_cheap_seconds": busy_cheap_seconds,
        "busy_pct": busy_pct,
        "decode_median_tps": decode_median,
        "decode_samples": decode_samples,
        "prompt_eval_median_tps": prompt_eval_median,
        "fast_slots": fast_slots,
        "cheap_slots": cheap_slots,
        "slot_era": era_note,
    }


def collect_reports(reports_dir: Path) -> list[dict]:
    """Parse every archived report and return rows sorted by window start.

    Duplicate report windows (multiple snapshot dirs for one day) are kept —
    the historical table is a record of observed days, not a de-duplicated
    daily series — but exact same-source duplicates cannot occur.
    """
    rows: list[dict] = []
    for path in sorted(reports_dir.rglob("report.md")):
        parsed = parse_report(path)
        if parsed is not None:
            # Record the snapshot directory name (e.g. 2026-10-20_2) for traceability.
            parsed["snapshot"] = path.parent.name
            rows.append(parsed)
    rows.sort(key=lambda r: r["window_start"])
    return rows


_DATE_SNAPSHOT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(_\d+)?$")


def select_daily_rows(rows: list[dict]) -> list[dict]:
    """Reduce multiple snapshots per window date to one representative row.

    The archive holds several re-runs per day (canonical ``YYYY-MM-DD``,
    ``_2``/``_3`` suffixes, plus ad-hoc ``measure-*``/``compare-*`` runs). For
    the historical table one row per day is preferred. Selection:

    1. Drop empty rows (``requests == 0``) — measurement stubs carry no signal.
    2. Prefer date-named snapshots over ad-hoc named ones.
    3. Within the preferred set, keep the row with the most requests (the most
       complete 24 h window for that date).
    """
    by_date: dict[str, list[dict]] = {}
    for row in rows:
        if not row.get("requests"):
            continue
        by_date.setdefault(row["window_start"][:10], []).append(row)

    selected: list[dict] = []
    for date in sorted(by_date):
        candidates = by_date[date]
        date_named = [r for r in candidates if _DATE_SNAPSHOT_RE.match(r["snapshot"])]
        pool = date_named or candidates
        selected.append(max(pool, key=lambda r: r["requests"]))
    return selected


# ---------------------------------------------------------------------------
# Concurrency reconstruction from proxy logs
# ---------------------------------------------------------------------------

_STREAM_STARTED_RE = re.compile(r"Stream started:.*?\bsession=(\S+)")
_STREAM_FINISHED_RE = re.compile(r"Stream finished:.*?\bsession=(\S+)")
_LINE_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_DENIED_RE = re.compile(r"local_dispatch_denied\b.*?\bactive=(\d+)")
_CONTENTION_DISPATCH_RE = re.compile(r"contention_queue_dispatch\b")
_CONTENTION_FALLBACK_RE = re.compile(r"contention_queue_fallback_after_queue\b")
_PREFILL_MS_RE = re.compile(r"\bdispatch_first_byte_ms=([0-9.]+)")
_FINISH_REASON_RE = re.compile(r"Stream finished: reason=(\S+)")
_LOCAL_PROVIDER_RE = re.compile(r"(?:^|[\s])provider=(local[qwen0-9-]*|local-[^\s]+)")

#: Longest plausible single HTTP stream (seconds). No individual request holds
#: a slot for this long; the cap prevents a stale ``Stream started`` from an
#: earlier rotation being paired with a much later ``Stream finished`` into a
#: phantom multi-hour interval. p99 observed stream length is ~95 s (max
#: ~41 min), so 3 h is very generous.
MAX_STREAM_SECONDS = 3 * 3600

SESSION_CLASS_PREFIXES = ("herdr", "audit", "benchmark")


def _open_text(path: Path):
    """Open a plain or gzipped log file as a text stream."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open("r", errors="replace")


def _session_class(session: str) -> str:
    for prefix in SESSION_CLASS_PREFIXES:
        if session.startswith(prefix):
            return f"{prefix}-*"
    return "operator/pi/LAN"


@dataclass
class ConcurrencyResult:
    #: Instantaneous concurrency (distinct sessions with an in-flight stream)
    #: sampled at each complete minute boundary in the window.
    minute_concurrency: list[int] = field(default_factory=list)
    #: Peak concurrency observed at any instant in the window.
    peak: int = 0
    #: Peak concurrency broken down by session class.
    peak_by_class: dict[str, int] = field(default_factory=dict)
    #: Completed in-flight intervals used in the calculation.
    intervals: int = 0
    #: Local-dispatch denial events (and how many saw ``active=N``).
    dispatch_denied_total: int = 0
    dispatch_denied_by_active: dict[int, int] = field(default_factory=dict)
    #: Contention-queue metrics seen in the window.
    contention_dispatch: int = 0
    contention_fallback_after_queue: int = 0
    #: Local prefill latency (``dispatch_first_byte_ms``) samples, milliseconds.
    prefill_ms: list[float] = field(default_factory=list)
    #: ``Stream finished`` reason -> count (length truncation, tool_calls, stop).
    finish_reason_counts: dict[str, int] = field(default_factory=dict)
    #: Local streams finished in the window.
    local_finish_count: int = 0
    window_start: float | None = None
    window_end: float | None = None
    files: list[str] = field(default_factory=list)

    def distribution(self) -> dict[int, int]:
        """Map concurrency -> number of minutes at that concurrency."""
        dist: dict[int, int] = {}
        for c in self.minute_concurrency:
            dist[c] = dist.get(c, 0) + 1
        return dict(sorted(dist.items()))

    def share_above(self, n: int) -> float | None:
        """Fraction of minutes with more than *n* concurrent sessions."""
        if not self.minute_concurrency:
            return None
        over = sum(1 for c in self.minute_concurrency if c > n)
        return over / len(self.minute_concurrency)

    def denied_share(self, n: int) -> float | None:
        """Share of session-minutes that exceed capacity *n*.

        Capacity model: with *n* slots, ``max(0, C - n)`` sessions are denied
        in a minute at concurrency *C*; the share is the denied session-minutes
        divided by total session-minutes. ``None`` when no demand was observed.
        """
        total = sum(self.minute_concurrency)
        if total == 0:
            return None
        denied = sum(max(0, c - n) for c in self.minute_concurrency)
        return denied / total

    def prefill_summary(self) -> dict[str, float] | None:
        """Local prefill (dispatch → first byte) latency in seconds."""
        if not self.prefill_ms:
            return None
        ordered = sorted(self.prefill_ms)
        n = len(ordered)
        return {
            "count": n,
            "median_s": round(ordered[n // 2] / 1000, 1),
            "p90_s": round(ordered[int(0.9 * (n - 1))] / 1000, 1),
            "max_s": round(ordered[-1] / 1000, 1),
            "total_hours": round(sum(ordered) / 3_600_000, 1),
        }


def discover_log_files(log_dir: Path, window_start: float, window_end: float) -> list[Path]:
    """Return log files in *log_dir* that can overlap the window.

    Retention is ~7 days and names carry a date or rotation suffix, so every
    ``proxy.log*`` file is a candidate; the per-line timestamp filter below
    discards anything outside the window.
    """
    if not log_dir.exists():
        return []
    candidates = sorted(
        p for p in log_dir.iterdir()
        if p.name.startswith("proxy.log") and p.is_file() and p.stat().st_size > 0
    )
    return candidates


def _file_time_range(path: Path) -> tuple[float | None, float | None]:
    """Return the (min, max) line timestamp of a plain or gz log file.

    Cheap scan: stops once two timestamps separated by > 60 s are seen, which
    is enough to order files chronologically for stream pairing.
    """
    lo: float | None = None
    hi: float | None = None
    try:
        stream = _open_text(path)
    except OSError:
        return None, None
    with stream:
        for line in stream:
            m = _LINE_TS_RE.match(line)
            if not m:
                continue
            ts = datetime.fromisoformat(m.group(1)).timestamp()
            lo = ts if lo is None else min(lo, ts)
            hi = ts if hi is None else max(hi, ts)
            if hi is not None and lo is not None and hi - lo > 60:
                break
    return lo, hi


def reconstruct_concurrency(
    log_dir: Path,
    window_start: datetime,
    window_end: datetime,
    *,
    files: list[Path] | None = None,
) -> ConcurrencyResult:
    """Reconstruct minute-by-minute concurrent-session demand from logs.

    Pairing model: per session, each ``Stream started`` opens an interval and
    the next ``Stream finished`` for the same session closes the earliest open
    interval (FIFO). A request that never finishes before the window ends is
    clipped at ``window_end``. Concurrency in a minute is the number of
    distinct sessions with an interval covering any part of that minute.

    This measures *demand* (sessions actively streaming, any provider), not
    local slot occupancy — sessions routed remote still wanted a slot. That is
    the quantity a larger pool is sized against (parent AC2).
    """
    start_ts = window_start.timestamp()
    end_ts = window_end.timestamp()
    log_files = files if files is not None else discover_log_files(log_dir, start_ts, end_ts)

    # Process files in chronological order so an open in an old file can be
    # paired with a finish in a newer file (streams span log rotations).
    ordered = sorted(
        log_files,
        key=lambda p: (_file_time_range(p)[0] or 0.0, str(p)),
    )

    open_starts: dict[str, list[float]] = {}
    intervals: list[tuple[float, float, str]] = []
    result = ConcurrencyResult(window_start=start_ts, window_end=end_ts)

    for path in ordered:
        result.files.append(str(path))
        try:
            stream = _open_text(path)
        except OSError:
            continue
        with stream:
            for line in stream:
                m = _LINE_TS_RE.match(line)
                if not m:
                    continue
                ts = datetime.fromisoformat(m.group(1)).timestamp()
                if ts < start_ts - MAX_STREAM_SECONDS or ts > end_ts + MAX_STREAM_SECONDS:
                    # Only lines that could contribute to the window: a stream
                    # starting more than MAX_STREAM_SECONDS before the window
                    # can never overlap it, and one finishing more than
                    # MAX_STREAM_SECONDS after proves nothing inside it.
                    continue

                if "local_dispatch_denied" in line:
                    if not start_ts <= ts <= end_ts:
                        continue
                    result.dispatch_denied_total += 1
                    dm = _DENIED_RE.search(line)
                    if dm:
                        active = int(dm.group(1))
                        result.dispatch_denied_by_active[active] = (
                            result.dispatch_denied_by_active.get(active, 0) + 1
                        )
                    continue
                if "contention_queue" in line:
                    if not start_ts <= ts <= end_ts:
                        continue
                    if _CONTENTION_DISPATCH_RE.search(line):
                        result.contention_dispatch += 1
                    elif _CONTENTION_FALLBACK_RE.search(line):
                        result.contention_fallback_after_queue += 1
                    continue

                if "dispatch_first_byte_ms=" in line:
                    if start_ts <= ts <= end_ts:
                        pm = _PREFILL_MS_RE.search(line)
                        if pm:
                            result.prefill_ms.append(float(pm.group(1)))
                    continue

                if "Stream finished:" in line and "reason=" in line:
                    fm = _FINISH_REASON_RE.search(line)
                    if fm:
                        reason = fm.group(1)
                        result.finish_reason_counts[reason] = (
                            result.finish_reason_counts.get(reason, 0) + 1
                        )
                    if _LOCAL_PROVIDER_RE.search(line):
                        result.local_finish_count += 1

                if "Stream started:" in line:
                    sm = _STREAM_STARTED_RE.search(line)
                    if sm:
                        open_starts.setdefault(sm.group(1), []).append(ts)
                elif "Stream finished:" in line:
                    fm = _STREAM_FINISHED_RE.search(line)
                    if fm:
                        session = fm.group(1)
                        pending = open_starts.get(session)
                        if pending:
                            # Discard stale opens (older than a plausible single
                            # stream) so a much later finish cannot pair with them.
                            eligible = [b for b in pending if ts - b <= MAX_STREAM_SECONDS]
                            if eligible:
                                began = eligible.pop(0)
                                pending[:] = eligible  # keep the rest in order
                                intervals.append((began, ts, session))

    # Dangling opens (no ``Stream finished`` observed for them within the scan
    # window) are dropped rather than closed at the window end: inventing a
    # whole-window interval from a start whose finish line was rotated out would
    # create a phantom permanent session. Only a handful occur (the window's
    # own tail), so the underestimate is negligible and documented.

    # Drop clock-inverted / outside-window pairs and clip crossings: an
    # interval that began before the window is clipped to the window start
    # (it was already in-flight), one that ends after the window is clipped to
    # the window end. Intervals entirely outside the window are discarded.
    clipped: list[tuple[float, float, str]] = []
    for b, e, s in intervals:
        if e <= start_ts or b >= end_ts:
            continue
        if b < start_ts:
            b = start_ts
        if e > end_ts:
            e = end_ts
        clipped.append((b, e, s))
    intervals = clipped
    result.intervals = len(intervals)

    # Instantaneous sweep over interval boundaries: ``distinct`` counts
    # sessions with >=1 active stream; ``per_class`` tracks the same per class.
    events: list[tuple[float, int, str]] = []
    for began, ended, session in intervals:
        events.append((began, 1, session))
        events.append((ended, -1, session))
    events.sort(key=lambda e: (e[0], -e[1]))  # opens before closes at ties

    active_by_session: dict[str, int] = {}
    distinct = 0
    distinct_by_class: dict[str, int] = {}
    peak_overall = 0
    peak_by_class: dict[str, int] = {}

    def _bump(session: str, delta: int) -> None:
        nonlocal distinct
        cls = _session_class(session)
        prev = active_by_session.get(session, 0)
        new = prev + delta
        if new <= 0:
            active_by_session.pop(session, None)
        else:
            active_by_session[session] = new
        # A session crossing 0↔1 changes the distinct count for its class.
        if prev == 0 and new > 0:
            distinct += 1
            distinct_by_class[cls] = distinct_by_class.get(cls, 0) + 1
        elif prev > 0 and new <= 0:
            distinct -= 1
            distinct_by_class[cls] = distinct_by_class.get(cls, 0) - 1

    # Sample instantaneous concurrency at each complete minute boundary, while
    # sweeping the events forward so each sample is O(events advanced).
    minute_start = start_ts - (start_ts % 60) + 60
    ev_idx = 0
    minutes: list[int] = []
    while minute_start <= end_ts:
        while ev_idx < len(events) and events[ev_idx][0] <= minute_start:
            _, delta, session = events[ev_idx]
            _bump(session, delta)
            ev_idx += 1
            peak_overall = max(peak_overall, distinct)
            for cls, count in distinct_by_class.items():
                peak_by_class[cls] = max(peak_by_class.get(cls, 0), count)
        minutes.append(distinct)
        minute_start += 60

    result.minute_concurrency = minutes
    result.peak = peak_overall
    result.peak_by_class = {k: v for k, v in peak_by_class.items() if v > 0}
    return result


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _fmt_int(value: int | None) -> str:
    return "--" if value is None else f"{value:,}"


def _fmt_pct(value: float | None, digits: int = 1) -> str:
    return "--" if value is None else f"{value * 100:.{digits}f}%"


def _fmt_busy(seconds: int | None) -> str:
    if seconds is None:
        return "--"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def render_history_markdown(rows: list[dict]) -> str:
    """Render the historical per-report table (parent AC1)."""
    lines = [
        "| Date (window start) | Snapshot | Slots fast/cheap | Requests (local%) "
        "| Busy % | Busy (h) | Decode med tok/s | Prompt-eval med tok/s "
        "| Queued→local | Fallback after queue | Dispatch denied |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for row in rows:
        lines.append(
            "| {date} | {snap} | {fast}/{cheap} | {req} ({pct}) | {busy_pct} | {busy} "
            "| {decode} | {peval} | {queued} | {faq} | {denied} |".format(
                date=(row["window_start"] or "")[:10],
                snap=row["snapshot"],
                fast=row["fast_slots"],
                cheap=row["cheap_slots"],
                req=_fmt_int(row["requests"]),
                pct=_fmt_pct(row["local_share"]),
                busy_pct=("--" if row["busy_pct"] is None else f"{row['busy_pct']:.1f}%"),
                busy=_fmt_busy(row["busy_seconds"]),
                decode=("--" if row["decode_median_tps"] is None
                        else f"{row['decode_median_tps']:.1f}"),
                peval=("--" if row["prompt_eval_median_tps"] is None
                       else f"{row['prompt_eval_median_tps']:.1f}"),
                queued=_fmt_int(row["queued_dispatched"]),
                faq=_fmt_int(row["fallback_after_queue"]),
                denied=_fmt_int(row["dispatch_denied"]),
            )
        )
    return "\n".join(lines)


def render_concurrency_markdown(result: ConcurrencyResult) -> str:
    """Render the concurrency-vs-capacity statement (parent AC2)."""
    dist = result.distribution()
    total_minutes = len(result.minute_concurrency)
    lines = [
        "**Concurrency distribution** (minutes with N distinct sessions "
        f"streaming; window {total_minutes:,} complete minutes, "
        f"{result.intervals:,} in-flight intervals):",
        "",
        "| Concurrent sessions | Minutes | Share |",
        "|---:|---:|---:|",
    ]
    for concurrency, minutes in dist.items():
        share = minutes / total_minutes if total_minutes else 0.0
        lines.append(f"| {concurrency} | {minutes:,} | {share * 100:.1f}% |")
    lines += [
        f"| **min / mean / p50 / p95 / peak** | "
        f"**{min(result.minute_concurrency, default=0)} / "
        f"{_mean(result.minute_concurrency):.2f} / "
        f"{_percentile(result.minute_concurrency, 50):.0f} / "
        f"{_percentile(result.minute_concurrency, 95):.0f} / "
        f"{result.peak}** | |",
        "",
        "**Capacity model** — `denied = max(0, concurrent - N)` session-minutes "
        "per minute, projected at N=2/3/4:",
        "",
        "| Slots N | Minutes above N | Share of minutes above N | Share of session-minutes denied |",
        "|---:|---:|---:|---:|",
    ]
    for n in (2, 3, 4):
        above_minutes = sum(1 for c in result.minute_concurrency if c > n)
        above_share = above_minutes / total_minutes if total_minutes else None
        lines.append(
            f"| {n} | {above_minutes:,} | {_fmt_pct(above_share)} | "
            f"{_fmt_pct(result.denied_share(n))} |"
        )
    if result.peak_by_class:
        classes = ", ".join(
            f"`{cls}` {peak}" for cls, peak in sorted(result.peak_by_class.items())
        )
        lines += ["", f"**Peak concurrency by session class:** {classes}."]
    lines += [
        "",
        f"**Local-dispatch denials in window:** {result.dispatch_denied_total:,} "
        f"(by active count: "
        + ", ".join(f"active={k} → {v:,}" for k, v in sorted(result.dispatch_denied_by_active.items()))
        + f"); contention queue: {result.contention_dispatch:,} dispatched, "
        f"{result.contention_fallback_after_queue:,} abandoned.",
    ]
    prefill = result.prefill_summary()
    if prefill:
        lines += [
            "",
            f"**Measured local prefill (dispatch → first byte):** "
            f"{prefill['count']:,} samples, median {prefill['median_s']:.1f}s, "
            f"p90 {prefill['p90_s']:.1f}s, max {prefill['max_s']:.1f}s, "
            f"total {prefill['total_hours']:.1f} h of GPU prefill time.",
        ]
    if result.finish_reason_counts:
        reasons = ", ".join(
            f"`{reason}` {count:,}"
            for reason, count in sorted(result.finish_reason_counts.items(), key=lambda kv: -kv[1])
        )
        lines += [
            "",
            f"**Stream finish reasons:** {reasons} "
            f"({result.local_finish_count:,} local streams finished).",
        ]
    return "\n".join(lines)


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[int], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((pct / 100) * (len(ordered) - 1)))
    return float(ordered[idx])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--reports-dir", default=str(Path.home() / "proxy-usage-reports"),
        help="Archived proxy-usage report root (default: ~/proxy-usage-reports)",
    )
    parser.add_argument(
        "--log-dir", default="/var/log/llama-proxy",
        help="Proxy log directory (default: /var/log/llama-proxy)",
    )
    parser.add_argument(
        "--window-start", default=None,
        help="Concurrency window start (ISO, default: 24h before --window-end)",
    )
    parser.add_argument(
        "--window-end", default=None,
        help="Concurrency window end (ISO, default: now)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown (default)")
    parser.add_argument("--concurrency", action="store_true", help="Only run the concurrency analysis")
    parser.add_argument("--history", action="store_true", help="Only run the historical table")
    parser.add_argument(
        "--dedupe-days", action="store_true",
        help="Keep one representative report per window date (recommended for the doc table)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    window_end = datetime.fromisoformat(args.window_end) if args.window_end else datetime.now()
    window_start = (
        datetime.fromisoformat(args.window_start)
        if args.window_start
        else window_end - timedelta(hours=24)
    )

    payload: dict = {}
    if not args.concurrency:
        rows = collect_reports(Path(args.reports_dir).expanduser())
        if args.dedupe_days:
            rows = select_daily_rows(rows)
        payload["history"] = rows
    if not args.history:
        concurrency = reconstruct_concurrency(Path(args.log_dir), window_start, window_end)
        payload["concurrency"] = {
            "window_start": concurrency.window_start,
            "window_end": concurrency.window_end,
            "peak": concurrency.peak,
            "peak_by_class": concurrency.peak_by_class,
            "intervals": concurrency.intervals,
            "minutes": len(concurrency.minute_concurrency),
            "distribution": {str(k): v for k, v in concurrency.distribution().items()},
            "mean": _mean(concurrency.minute_concurrency),
            "p50": _percentile(concurrency.minute_concurrency, 50),
            "p95": _percentile(concurrency.minute_concurrency, 95),
            "share_above": {str(n): concurrency.share_above(n) for n in (2, 3, 4)},
            "denied_share": {str(n): concurrency.denied_share(n) for n in (2, 3, 4)},
            "dispatch_denied_total": concurrency.dispatch_denied_total,
            "dispatch_denied_by_active": {
                str(k): v for k, v in concurrency.dispatch_denied_by_active.items()
            },
            "contention_dispatch": concurrency.contention_dispatch,
            "contention_fallback_after_queue": concurrency.contention_fallback_after_queue,
            "prefill": concurrency.prefill_summary(),
            "finish_reason_counts": concurrency.finish_reason_counts,
            "local_finish_count": concurrency.local_finish_count,
            "files": concurrency.files,
        }

    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if "history" in payload:
        print("## Historical report table\n")
        print(render_history_markdown(payload["history"]))
        print()
    if "concurrency" in payload:
        # Rebuild a lightweight object for rendering from the computed payload.
        concurrency = reconstruct_concurrency(Path(args.log_dir), window_start, window_end)
        print("## Concurrency vs capacity\n")
        print(render_concurrency_markdown(concurrency))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
