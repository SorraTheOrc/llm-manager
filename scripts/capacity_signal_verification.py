#!/usr/bin/env python3
"""Reproducible capacity-signal measurement for an arbitrary log window.

This is the measurement tool behind the local-capacity acceptance criteria of
LP-0MVLNBCTN0084MT0 (parent fix: LP-0MVKSXR4P001V0CM). Before it existed there
was no single command that turned the proxy log stream into the figures the
criteria are written against (status-poll counts, ``slots_stale`` rate,
``capacity_hint`` distribution, saturated-without-live-work violations and
dispatch counts), so every reported number was an ad-hoc grep.

Why this tool exists / what it deliberately does *not* do
--------------------------------------------------------
Analysis only: it reads logs and never changes proxy behaviour, status
semantics or dispatch logic. Log-file discovery, gzip handling, per-line
timestamp filtering and the coverage verdict all come from
``scripts/lib/proxy_logs.py`` — this tool does not re-implement them
(LP-0MVLHMAWV008EKTI). ``.pi/skills/proxy-usage-analysis/`` is referenced
read-only as the nightly-report behavioural reference; nothing under ``.pi/``
is edited.

Usage
-----
::

  ./scripts/capacity_signal_verification.py --since '2026-10-23 00:00' \\
      --until '2026-10-23 06:30' --json
  ./scripts/capacity_signal_verification.py --since '2026-10-24 20:00' \\
      --until '2026-10-25 07:00' --window-closed --label post-fix-night
  ./scripts/capacity_signal_verification.py --since '2026-10-23 00:00' \\
      --until '2026-10-23 06:30' --dispatch-log /path/to/downtime-dispatches.log

``--since``/``--until`` are **local wall-clock** timestamps (the proxy logs are
written in local time; Europe/Dublin). ``--since`` is inclusive, ``--until``
exclusive, so adjacent windows never double-count a poll. Formats accepted:
``YYYY-MM-DD``, ``YYYY-MM-DD HH:MM``, ``YYYY-MM-DD HH:MM:SS``. With no bounds
the whole retained log directory is measured.

Exit codes
----------
``0``
    Measurement completed over a covered window (verdict ``PASS`` or
    ``TARGETS MISSED`` — target misses are reported, not fatal; see
    ``--fail-on-target-miss``). An empty-but-covered window is also ``0`` with
    ``polls = 0`` and verdict ``EMPTY``: an empty window is a result, not an
    error.
``2``
    ``--fail-on-target-miss`` was passed and a target was missed.
``3``
    ``COVERAGE GAP`` — the requested window is not fully represented by the
    retained logs (a hole between files, or a missing head/tail), so no figure
    may be read as complete. Never a silent partial result
    (LP-0MVLHMAWV008EKTI).
``1``
    Usage/environment error (missing log directory, no proxy logs found).

Derivation of every emitted figure
----------------------------------
Every number below is computed from the lines of
``/var/log/llama-proxy/proxy.log*`` whose leading ``YYYY-MM-DD HH:MM:SS``
timestamp falls inside ``[since, until)`` — the timestamp at the start of the
line, never the file name or mtime (a rotated file routinely holds lines past
its name-encoded hour, ``scripts/lib/proxy_logs.py``).

``status_polls``
    Count of log lines whose message is the structured status surface,
    i.e. a line matching ``\\bstatus_request\\b`` that carries a
    ``capacity_hint=`` field. This is one line per ``GET
    /llama/local/status`` request (``proxy/proxy/handlers.py`` →
    ``logger.info("status_request", ...)``). Re-derive by hand with::

        zcat -f /var/log/llama-proxy/proxy.log* \\
          | grep 'status_request' | grep -c 'capacity_hint='

``slots_stale.count`` / ``slots_stale.rate``
    Number of status polls whose line carries ``slots_stale=true``, and that
    count divided by ``slots_stale.denominator`` = the number of polls that
    carry the ``slots_stale`` field at all (``status_schema_version >= 2``;
    schema 1 polls have no such field and are excluded from the rate rather
    than counted as fresh). ``slots_stale`` keeps its documented meaning: the
    counts *or* the per-slot detail were served from cache
    (``proxy/docs/per-slot-status-verification.md``).

``capacity_hint.histogram``
    Value of ``capacity_hint=<token>`` per status poll (``idle``,
    ``saturated``, ...), plus each token's share of ``status_polls``.

``saturated_without_live_work``
    Status polls with ``capacity_hint=saturated`` that report **no** live local
    work on the same line. Live work is any of: ``local_in_flight_requests`` >
    0 (schema 3), ``local_active_query=true``, or ``active_query=true``.
    Schema 2 polls (pre LP-0MSKV3IEQ) have no ``local_in_flight_requests``
    field, so the ``local_active_query``/``active_query`` flags are the
    fallback for those lines. This is the "false saturated" signal the fix
    targets: a truthful status surface reports ``saturated`` only while real
    local streams run. Caveat: the derivation uses the status surface's own
    liveness fields, not an independent GPU/stream probe — the log carries no
    per-dispatch end marker, so overlapping dispatch windows cannot be
    reconstructed exactly. ``worst_timestamps`` lists the busiest
    whole-minute buckets (highest violating polls first) so an operator can go
    straight to the log lines, ``examples`` the first violating
    timestamps in order, and ``rows`` **every** violating poll as
    ``{timestamp, latency_ms, slots_detail_stale, available_slots}`` — the
    input to the attribution below (LP-0MVOG7IP80003MLV AC2/AC4).

``slots_stale_attribution``
    Every stale poll attributed to the cache that served it, plus the response
    latency that shows *why*: ``by_source`` (``detail_cache`` /
    ``counts_cache`` / ``unknown`` — the last for a poll that carries the
    legacy union flag but no per-source flag), ``by_latency_band``
    (``lt_1s``/``1s_to_2s``/``2s_to_5s``/``gte_5s``/``unknown``) and
    ``unattributed``. A stale poll served in under a second is normally a
    failure-backoff cache hit: ``_cached_slots_payload`` caches a non-200 for
    ``SLOTS_POLL_FAILURE_BACKOFF_SECONDS`` (default 5 s), so the polls right
    after a timeout are served instantly — and stale (LP-0MVOG7IP80003MLV
    AC4).

``abandonment_events``
    Local dispatch abandonments inside the window, by reason: a
    ``lease_released reason=<r>`` line where ``<r>`` is one of
    ``prefill_deadline``, ``session_evicted``, ``no_progress_watchdog`` or
    ``client_disconnect``. That records the proxy dropping an in-flight local
    dispatch while llama-server may still be finishing the job — so a *fresh*
    ``/slots`` snapshot can legitimately report the slot ``is_processing``
    with no proxy-side record left. Proxy-restart markers are reported the
    same way (``kind: proxy_restart``) so restart-bounded stale windows can be
    attributed. ``idle_timeout`` is excluded: it is the normal end of a
    *completed* stream, not the abandonment of an in-flight one.

``attribution``
    Each violating ``saturated`` poll attributed to the **nearest preceding**
    abandonment/restart event within ``--attribution-lookback-seconds``
    (default 600 s), as ``rows`` (``timestamp``, ``cause``,
    ``evidence_timestamp``, ``seconds_since_evidence``, ...), ``by_cause``,
    ``attributed_polls``, ``unattributed_polls`` and
    ``unattributed_timestamps``. A poll with no preceding event inside the
    lookback is reported unattributed — never guessed. The event scan reads up
    to one lookback *before* ``--since`` so a window's first polls can still be
    explained, while the reported ``abandonment_events`` tally stays scoped to
    the window. Attribution **explains** a residual; it never turns a missed
    target into a pass (the check still fails and the cause is stated beside
    it).

``--write-artifacts DIR``
    Persist the raw evidence (LP-0MVOG7IP80003MLV AC5): ``measurement.json``
    (this whole report), ``measurement.txt`` (the human rendering) and
    ``saturated_without_live_work_attribution.csv`` (one row per violating
    poll, so a reviewer can re-check every attribution against the log).

``saturated_without_available_slots``
    Status polls with ``capacity_hint=saturated`` **and**
    ``available_slots=0`` — the pool reported full. This subset is what the
    recorded pre-fix control figures were derived from (see the control-run
    section below); the tool always emits it next to the raw
    ``capacity_hint=saturated`` count so the historical number stays
    re-derivable.

``local_dispatches``
    Count of local dispatch completions: lines carrying
    ``dispatch_first_byte_ms=`` (emitted only in the local streaming path,
    ``proxy/proxy/router.py``) — one line per local dispatch that reached its
    first byte. Also split by ``cache_state=warm|cold`` and reported as a
    distinct ``session=`` count, with the ``session=herdr-*`` subset separated
    (herdr-orchestrated work).

``herdr_downtime_dispatches``
    Read from the herdr downtime dispatch log,
    ``<repo>/.worklog/downtime-dispatches.log`` (JSONL, written by
    ``ContextHub/packages/herdr/src/downtime-log.ts``), filtered to entries
    whose ``dispatchedAt`` falls inside the window and **de-duplicated on
    ``(itemId, dispatchedAt)``** — herdr appends a second, enriched row for
    the same dispatch once a pane is allocated, so the raw line count is about
    twice the dispatch count. ``dispatchedAt`` is an ISO-8601 **UTC**
    (``Z``) timestamp and is converted to the local wall clock before the
    window comparison, because ``--since``/``--until`` are local. The log is
    a bounded rolling file (~100 rows), so ``covered`` states whether its
    retained span contains the window: when ``covered`` is ``false`` a count
    of 0 means "not retained", not "no dispatches", and the verdict says so.

Coverage verdict
----------------
File → window mapping is ``scripts/lib/proxy_logs.py`` (dot ``proxy.log.*``
and dash ``proxy.log-*`` names, plain or ``.gz``, per-line timestamps). A
window is *covered* when at least one file contributes lines and:

* no adjacent-file boundary inside the window is further apart than
  ``--max-gap-seconds`` (default 3600 s) — the lost-rotation shape this guards
  against (2026-10-23 18:29:12 → 22:04:16);
* the head of the requested window is within ``--max-gap-seconds`` of the
  first retained line (a window that starts before its own data is measuring
  a shorter window than it claims);
* with ``--window-closed`` (the window has fully elapsed), the tail is
  likewise within ``--max-gap-seconds`` of the requested end. Without the flag
  the tail is informational, because a live window has no end yet;
* the retained logs reach the window at all: a window that lies entirely
  before or after every retained file (``kind: outside_retention``) is a
  coverage gap, not an empty result. An empty *covered* window (retention
  spans it, but no status polls were written) reports ``polls = 0`` with
  verdict ``EMPTY`` and exit 0 — empty is a result, not an error.

Any failure yields ``verdict: "COVERAGE GAP"`` and exit 3, with the offending
boundary named in ``coverage.gaps`` so the missing interval can be quoted.

Control run: reproducing the pre-fix baseline
---------------------------------------------
The recorded pre-fix evidence for LP-0MVLNBCTN0084MT0 ("2026-10-22 night") is
9,484 status polls, ``slots_stale=true`` on 7,543 (79.5 %) and
``capacity_hint=saturated`` on 7,807 (82.3 %). Those figures are reproduced
**exactly** by::

  ./scripts/capacity_signal_verification.py \\
    --since '2026-10-23 00:00' --until '2026-10-23 06:29:13' \\
    --label control-prefix

which selects the retained rotated file ``proxy.log.2026-10-23_00.gz``
(content 00:00:00 → 06:29:12) and nothing else:

* ``status_polls`` = 9,484
* ``slots_stale.count`` = 7,543 (79.53 %)
* ``capacity_hint.histogram`` = ``{idle: 1,069, saturated: 8,415}``
* ``saturated_without_available_slots.count`` = 7,807 (82.32 %)
* ``saturated_without_live_work.count`` = 1,773 (18.69 % of polls)
* ``local_dispatches.count`` = 59 (15 sessions, 4 herdr sessions)

The end bound is ``06:29:13`` because the next retained file starts at
06:29:16; any later ``--until`` pulls in more polls (e.g. ``06:30`` gives
9,495) and breaks the exact match.

Note on the window label: the recorded baseline was described as the
"20:00 → 07:00" night, but its poll total is the *content* of that single
rotated file, whose name encodes the rotation hour (``_00``), not its span.
Measuring the literal 20:00 → 07:00 window instead selects
``proxy.log-2026-10-23_00.gz`` (19:27 → 23:59), ``proxy.log.2026-10-23_00.gz``
and ``proxy.log.2026-10-23_06.gz`` and yields 17,299 polls with a
``slots_stale`` rate of 71.2 % and 70.9 % saturated-without-available-slots
(13,400 raw ``saturated``) — a different, wider population (the pre-midnight
hours were staler and busier). Both runs are reproducible; the narrower one is
the baseline's own population and is therefore the control. ``--label``
records which run an artifact is.

Related work: parent LP-0MVLNBCTN0084MT0; measurement children
LP-0MVOG7IP80003MLV, LP-0MVOG7IX6006MZ23; discovery/coverage library
``scripts/lib/proxy_logs.py`` (LP-0MUIEK483002TZ54).

Post-fix night measurement (LP-0MVOG7IP80003MLV)
-----------------------------------------------
The first application of this tool to a real post-fix window is recorded under
``~/proxy-usage-reports/2026-10-24_post-fix-night-measurement/`` (``REPORT.md``
plus the ``--write-artifacts`` output and the supplementary derivation tables).
Two findings from it shape how the numbers should be read:

* ``slots_stale`` is a **polling-path** health signal, not a
  capacity-truthfulness one. On 2026-10-24/25 it sat at 79.6 % because every
  proxy ``/slots`` query still went through the llama-server router (serialised
  behind the busy child, 5-7 s) and blew the 2.0 s/1.0 s budgets; routing the
  queries to the discovered child port only landed in f58e8964 (2026-10-26).
  A stale-rate figure is only meaningful with the build it was measured on.
* ``saturated``-without-live-work residuals split into two classes that must
  not be conflated: stale-detail false saturation (what the capacity fix
  eliminates — zero of 9,012 stale-detail polls in that window) and *fresh*
  detail from a slot llama-server is still finishing after the proxy abandoned
  the dispatch (``prefill_deadline``/``session_evicted`` — the attribution
  rows).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.proxy_logs import (  # noqa: E402
    discover_proxy_log_files,
    measure_log_coverage,
    open_proxy_log_text,
    parse_line_timestamp,
)

REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_LOG_DIR = Path("/var/log/llama-proxy")


def _shared_worklog_dir(repo_root: Path) -> Path:
    """The worklog directory shared with the main checkout.

    ``wl`` data and herdr's dispatch log live in the main checkout's
    ``.worklog/``. When this tool runs from a linked git worktree (the
    implementation/measurement workflow does exactly that) ``repo_root`` is the
    worktree, whose own ``.worklog/`` has no dispatch log — the shared one is
    next to the *main* checkout's ``.git``. A worktree's ``.git`` is a file
    containing ``gitdir: <main>/.git/worktrees/<name>``, so the main root is
    that path up to ``.git``. Falls back to *repo_root* for a normal checkout
    or an unparseable ``.git``.
    """
    dot_git = repo_root / ".git"
    try:
        if dot_git.is_file():
            text = dot_git.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                gitdir = Path(text.split(":", 1)[1].strip())
                parts = gitdir.parts
                if ".git" in parts:
                    main_root = Path(*parts[: parts.index(".git")])
                    return main_root / ".worklog"
    except OSError:
        pass
    return repo_root / ".worklog"


#: herdr's bounded rolling dispatch log (see
#: ``ContextHub/packages/herdr/src/downtime-log.ts``), resolved against the
#: shared (main-checkout) worklog so a worktree run still finds it.
DEFAULT_DISPATCH_LOG = (
    _shared_worklog_dir(REPO_ROOT) / "downtime-dispatches.log"
)

TOOL = "scripts/capacity_signal_verification.py"
WORK_ITEM = "LP-0MVOG6YRA000T9PK"

_TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")

#: The structured status surface: one line per ``GET /llama/local/status``.
_STATUS_POLL_RE = re.compile(r"\bstatus_request\b")
#: Fields read off a status poll line (``key=value``, value is non-space).
_STATUS_FIELD_RE = re.compile(
    r"\b(?P<key>capacity_hint|slots_stale|slots_detail_stale|available_slots|"
    r"local_active_query|active_query|local_in_flight_requests|"
    r"status_schema_version|latency_ms)=(?P<value>[^\s]+)"
)
#: A local dispatch lease release (``proxy/proxy/handlers.py`` → the
#: ``local_dispatch_records`` registry). ``reason=`` names why the proxy
#: stopped tracking the dispatch.
_LEASE_RELEASED_RE = re.compile(r"\blease_released\b.*\breason=(?P<reason>[a-z_]+)")
#: Lease-release reasons that record an *abandoned* local dispatch — the proxy
#: released the record while llama-server may still be finishing the job, so a
#: later ``/slots`` snapshot can still report the slot as processing.
#: ``idle_timeout`` is excluded (the normal end of a *completed* stream) and so
#: is ``non_explicit`` (nothing is known about why it was released).
_ABANDON_REASONS = frozenset(
    {"prefill_deadline", "session_evicted", "no_progress_watchdog",
     "client_disconnect"}
)
#: Proxy restart markers (the rotation line the proxy writes on startup, plus
#: uvicorn lifecycle lines) so restart-bounded stale windows can be attributed.
_PROXY_RESTART_RE = re.compile(
    r"Rotated previous proxy log aside|application startup|Started server process|"
    r"Uvicorn running"
)
#: ``latency_ms`` bands separating a fast fresh poll (< 1 s) from one that
#: exhausted the ``/slots`` query budget (2.0 s counts / 1.0 s detail) and
#: beyond.
_LATENCY_BANDS = ("lt_1s", "1s_to_2s", "2s_to_5s", "gte_5s", "unknown")
#: A completed local dispatch (emitted only on the local streaming path).
_LOCAL_DISPATCH_RE = re.compile(
    r"\bdispatch_first_byte_ms=(?P<first_byte>[0-9.]+)\b"
)
_CACHE_STATE_RE = re.compile(r"\bcache_state=(?P<state>warm|cold)\b")
_SESSION_RE = re.compile(r"\bsession=(?P<session>\S+)")

#: Verdict literals (also asserted by the unit tests).
VERDICT_PASS = "PASS"
VERDICT_TARGETS_MISSED = "TARGETS MISSED"
VERDICT_EMPTY = "EMPTY"
VERDICT_COVERAGE_GAP = "COVERAGE GAP"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_TARGET_MISSED = 2
EXIT_COVERAGE_GAP = 3


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_bound(text: str) -> dt.datetime:
    """Parse a local wall-clock window bound (``YYYY-MM-DD [HH:MM[:SS]]``)."""
    for fmt in _TS_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"invalid timestamp {text!r} (expected 'YYYY-MM-DD [HH:MM[:SS]]')"
    )


def _status_fields(line: str) -> dict[str, str]:
    """Return the ``key=value`` status fields present on *line*."""
    return {
        match.group("key"): match.group("value")
        for match in _STATUS_FIELD_RE.finditer(line)
    }


def _parse_latency_ms(value: str | None) -> int | None:
    """Parse the handler's ``latency_ms=`` field (``None`` when absent)."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_int(value: str | None) -> int | None:
    """Parse a non-negative integer log field (``None`` when absent/bogus)."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _latency_band(latency_ms: int | None) -> str:
    """Bucket a status-poll ``latency_ms`` (``unknown`` when absent)."""
    if latency_ms is None:
        return "unknown"
    if latency_ms < 1000:
        return "lt_1s"
    if latency_ms < 2000:
        return "1s_to_2s"
    if latency_ms < 5000:
        return "2s_to_5s"
    return "gte_5s"


def is_status_poll(line: str) -> bool:
    """True for a structured ``status_request`` line (the poll surface)."""
    return bool(_STATUS_POLL_RE.search(line)) and "capacity_hint=" in line


def has_live_local_work(fields: dict[str, str]) -> bool:
    """True when a status poll reports concurrent live local work.

    Schema 3 polls carry ``local_in_flight_requests`` (authoritative: derived
    from live dispatch records). Schema 2 polls do not, so the
    ``local_active_query`` / ``active_query`` flags are the fallback — which is
    exactly the pair the pre-fix false-``saturated`` incident was about.
    Any single positive signal counts as live work: a violating poll must show
    none of them.
    """
    in_flight = fields.get("local_in_flight_requests")
    if in_flight is not None:
        try:
            if int(in_flight) > 0:
                return True
        except ValueError:
            pass
    return (
        fields.get("local_active_query") == "true"
        or fields.get("active_query") == "true"
    )


def _in_window(ts: dt.datetime | None, since: dt.datetime | None,
               until: dt.datetime | None) -> bool:
    """``since`` inclusive, ``until`` exclusive (adjacent windows never overlap)."""
    if ts is None:
        return False
    if since is not None and ts < since:
        return False
    if until is not None and ts >= until:
        return False
    return True


# ---------------------------------------------------------------------------
# Status polls + local dispatches
# ---------------------------------------------------------------------------


def analyse_status_stream(
    log_dir: Path,
    *,
    since: dt.datetime | None = None,
    until: dt.datetime | None = None,
    limit: int = 10,
) -> dict:
    """Count status polls, capacity hints and local dispatches in a window.

    Streams every discovered log file (``scripts/lib/proxy_logs.py`` handles
    dot/dash names and ``.gz``) and filters **per line** on the leading
    timestamp, because a rotated file's name does not bound its content.
    """
    log_dir = Path(log_dir)
    polls = 0
    slots_stale = 0
    slots_stale_denominator = 0
    stale_by_source = {"detail_cache": 0, "counts_cache": 0, "unknown": 0}
    stale_by_latency_band: dict[str, int] = {band: 0 for band in _LATENCY_BANDS}
    hints: dict[str, int] = {}
    saturated = 0
    saturated_without_live_work = 0
    saturated_without_available_slots = 0
    violating_minutes: dict[str, int] = {}
    violating_examples: list[str] = []
    violating_rows: list[dict] = []
    local_dispatches = 0
    dispatches_by_cache_state: dict[str, int] = {}
    dispatch_sessions: set[str] = set()
    herdr_dispatch_sessions: set[str] = set()
    first_ts: dt.datetime | None = None
    last_ts: dt.datetime | None = None

    for path in discover_proxy_log_files(log_dir):
        with open_proxy_log_text(path) as handle:
            for line in handle:
                ts = parse_line_timestamp(line)
                if not _in_window(ts, since, until):
                    continue
                if first_ts is None or ts < first_ts:
                    first_ts = ts
                if last_ts is None or ts > last_ts:
                    last_ts = ts

                if is_status_poll(line):
                    polls += 1
                    fields = _status_fields(line)
                    hint = fields.get("capacity_hint")
                    if hint is not None:
                        hints[hint] = hints.get(hint, 0) + 1
                    stale = fields.get("slots_stale")
                    if stale is not None:
                        slots_stale_denominator += 1
                        if stale == "true":
                            slots_stale += 1
                            # Attribute the poll to the cache that served it
                            # (counts vs per-slot detail) and to the response
                            # latency that shows the upstream query failing.
                            detail_flag = fields.get("slots_detail_stale")
                            if detail_flag == "true":
                                stale_by_source["detail_cache"] += 1
                            elif detail_flag == "false":
                                stale_by_source["counts_cache"] += 1
                            else:
                                stale_by_source["unknown"] += 1
                            latency = _parse_latency_ms(fields.get("latency_ms"))
                            stale_by_latency_band[_latency_band(latency)] += 1
                    if hint == "saturated":
                        saturated += 1
                        if fields.get("available_slots") == "0":
                            saturated_without_available_slots += 1
                        if not has_live_local_work(fields):
                            saturated_without_live_work += 1
                            minute = ts.strftime("%Y-%m-%d %H:%M")
                            violating_minutes[minute] = (
                                violating_minutes.get(minute, 0) + 1
                            )
                            timestamp = ts.strftime("%Y-%m-%d %H:%M:%S")
                            if len(violating_examples) < limit:
                                violating_examples.append(timestamp)
                            violating_rows.append(
                                {
                                    "timestamp": timestamp,
                                    "latency_ms": _parse_latency_ms(
                                        fields.get("latency_ms")
                                    ),
                                    "slots_detail_stale": (
                                        fields.get("slots_detail_stale") == "true"
                                    ),
                                    "available_slots": _parse_int(
                                        fields.get("available_slots")
                                    ),
                                }
                            )
                    continue

                dispatch = _LOCAL_DISPATCH_RE.search(line)
                if dispatch is None:
                    continue
                local_dispatches += 1
                state = _CACHE_STATE_RE.search(line)
                key = state.group("state") if state else "unknown"
                dispatches_by_cache_state[key] = (
                    dispatches_by_cache_state.get(key, 0) + 1
                )
                session = _SESSION_RE.search(line)
                if session:
                    dispatch_sessions.add(session.group("session"))
                    if session.group("session").startswith("herdr-"):
                        herdr_dispatch_sessions.add(session.group("session"))

    def _rate(count: int, denominator: int) -> float:
        return round(count / denominator, 4) if denominator else 0.0

    worst = sorted(
        violating_minutes.items(), key=lambda item: (-item[1], item[0])
    )[:limit]
    return {
        "first_seen": first_ts.isoformat(sep=" ") if first_ts else None,
        "last_seen": last_ts.isoformat(sep=" ") if last_ts else None,
        "status_polls": polls,
        "slots_stale": {
            "count": slots_stale,
            "denominator": slots_stale_denominator,
            "rate": _rate(slots_stale, slots_stale_denominator),
        },
        "slots_stale_attribution": {
            "by_source": dict(stale_by_source),
            "by_latency_band": dict(stale_by_latency_band),
            "unattributed": stale_by_source["unknown"],
        },
        "capacity_hint": {
            "histogram": dict(sorted(hints.items())),
            "rates": {
                key: _rate(value, polls) for key, value in sorted(hints.items())
            },
        },
        "saturated_without_live_work": {
            "count": saturated_without_live_work,
            "rate_of_polls": _rate(saturated_without_live_work, polls),
            "rate_of_saturated": _rate(saturated_without_live_work, saturated),
            "worst_timestamps": [
                {"minute": minute, "polls": count} for minute, count in worst
            ],
            "examples": violating_examples,
            "rows": violating_rows,
        },
        "saturated_without_available_slots": {
            "count": saturated_without_available_slots,
            "rate_of_polls": _rate(saturated_without_available_slots, polls),
        },
        "local_dispatches": {
            "count": local_dispatches,
            "by_cache_state": dict(sorted(dispatches_by_cache_state.items())),
            "distinct_sessions": len(dispatch_sessions),
            "herdr_session_dispatches": len(herdr_dispatch_sessions),
        },
    }


# ---------------------------------------------------------------------------
# Abandonment events + violation attribution
# ---------------------------------------------------------------------------


def analyse_abandonment_events(
    log_dir: Path,
    *,
    since: dt.datetime | None = None,
    until: dt.datetime | None = None,
) -> list[dict]:
    """Local dispatch abandonments and proxy restarts inside the window.

    Attribution evidence for the ``saturated``-without-live-work residuals
    (and, historically, for restart-bounded stale windows). Two sources:

    * ``lease_released reason=<abandonment>`` — the proxy stopped tracking a
      local dispatch (prefill deadline exceeded, session evicted, watchdog,
      client disconnect) while llama-server may still be finishing the job, so
      a later fresh ``/slots`` snapshot can still report the slot processing;
    * proxy-restart markers (the ``Rotated previous proxy log aside`` line
      written on startup, plus uvicorn lifecycle lines).

    ``idle_timeout`` lease releases are deliberately excluded: that is the
    normal end of a completed stream, not the abandonment of an in-flight one.
    """
    log_dir = Path(log_dir)
    events: list[dict] = []
    # De-duplicate an identical line (same event, same session), never two
    # sessions abandoned in the same second: a bulk eviction releases several
    # leases with the same timestamp, and each is a real abandonment.
    seen: set[tuple[str, str, str, str]] = set()
    for path in discover_proxy_log_files(log_dir):
        with open_proxy_log_text(path) as handle:
            for line in handle:
                ts = parse_line_timestamp(line)
                if not _in_window(ts, since, until):
                    continue
                release = _LEASE_RELEASED_RE.search(line)
                if release and release.group("reason") in _ABANDON_REASONS:
                    session_match = _SESSION_RE.search(line)
                    session = (
                        session_match.group("session") if session_match else None
                    )
                    key = (
                        ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "lease_released",
                        release.group("reason"),
                        session or "",
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    events.append(
                        {
                            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                            "kind": "lease_released",
                            "reason": release.group("reason"),
                            "session": session,
                        }
                    )
                    continue
                if _PROXY_RESTART_RE.search(line):
                    key = (ts.strftime("%Y-%m-%d %H:%M:%S"), "proxy_restart", "", "")
                    if key in seen:
                        continue
                    seen.add(key)
                    events.append(
                        {
                            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                            "kind": "proxy_restart",
                            "reason": None,
                            "session": None,
                        }
                    )
    return sorted(events, key=lambda event: event["timestamp"])


def attribute_violations(
    violations: list[dict],
    events: list[dict],
    *,
    lookback_seconds: float = 600.0,
) -> dict:
    """Attribute every violating ``saturated`` poll to a preceding event.

    Each poll takes the **nearest preceding** abandonment/restart event inside
    ``lookback_seconds`` (``None`` — never guessed — when there is none, or when
    the only nearby event is *later* than the poll). Returns the full per-poll
    rows (``rows``), the cause histogram (``by_cause``) and the unattributed
    timestamps, so a reader can either accept the mechanism or go to the log.
    """
    events = sorted(events, key=lambda event: event["timestamp"])
    event_times = [event["timestamp"] for event in events]
    causes: dict[str, int] = {}
    rows: list[dict] = []
    unattributed: list[str] = []
    for poll in violations:
        row = {
            "timestamp": poll["timestamp"],
            "cause": None,
            "evidence_timestamp": None,
            "seconds_since_evidence": None,
            "latency_ms": poll.get("latency_ms"),
            "slots_detail_stale": poll.get("slots_detail_stale"),
            "available_slots": poll.get("available_slots"),
        }
        index = bisect.bisect_right(event_times, row["timestamp"]) - 1
        if index >= 0:
            event = events[index]
            seconds = (
                dt.datetime.strptime(
                    row["timestamp"], "%Y-%m-%d %H:%M:%S"
                )
                - dt.datetime.strptime(
                    event["timestamp"], "%Y-%m-%d %H:%M:%S"
                )
            ).total_seconds()
            if seconds <= lookback_seconds:
                cause = event["reason"] or event["kind"]
                row["cause"] = cause
                row["evidence_timestamp"] = event["timestamp"]
                row["seconds_since_evidence"] = int(seconds)
                causes[cause] = causes.get(cause, 0) + 1
        if row["cause"] is None:
            unattributed.append(row["timestamp"])
        rows.append(row)
    return {
        "lookback_seconds": lookback_seconds,
        "attributed_polls": len(violations) - len(unattributed),
        "unattributed_polls": len(unattributed),
        "unattributed_timestamps": unattributed,
        "by_cause": dict(sorted(causes.items())),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# herdr downtime dispatch log
# ---------------------------------------------------------------------------


def _to_local_naive(value: str) -> dt.datetime | None:
    """Parse an ISO-8601 ``dispatchedAt`` (UTC ``Z``) into local wall clock."""
    text = value.strip()
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # Already naive: treat it as local, matching the window bounds.
        return parsed
    return parsed.astimezone().replace(tzinfo=None)


def load_downtime_dispatches(
    path: Path | None,
    *,
    since: dt.datetime | None = None,
    until: dt.datetime | None = None,
    limit: int = 10,
) -> dict:
    """Count de-duplicated herdr downtime dispatches inside the window.

    ``downtime-dispatches.log`` is JSONL; herdr appends a base row and then an
    enriched row (with ``paneId``/``enrichment``) for the *same* dispatch, so
    rows are de-duplicated on ``(itemId, dispatchedAt)``. Timestamps are UTC
    and converted to the local wall clock before comparison.

    The log is bounded/rolling (~100 rows). ``covered`` reports whether its
    retained span contains the requested window: when ``False`` the count is
    *not* evidence of zero dispatches and the caller must say so.
    """
    result = {
        "path": str(path) if path is not None else None,
        "available": False,
        "covered": False,
        "log_first": None,
        "log_last": None,
        "count": 0,
        "rows_read": 0,
        "rows_skipped": 0,
        "entries": [],
        "note": "",
    }
    if path is None:
        result["note"] = "no dispatch log configured"
        return result
    path = Path(path)
    if not path.is_file():
        result["note"] = (
            "dispatch log not found — downtime dispatch count unavailable "
            "(0 is not evidence of no dispatches)"
        )
        return result
    result["available"] = True

    seen: set[tuple[str, str]] = set()
    first: dt.datetime | None = None
    last: dt.datetime | None = None
    matches: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            result["rows_read"] += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                result["rows_skipped"] += 1
                continue
            if not isinstance(row, dict):
                result["rows_skipped"] += 1
                continue
            dispatched_at = row.get("dispatchedAt")
            local_ts = _to_local_naive(dispatched_at or "")
            if local_ts is None:
                result["rows_skipped"] += 1
                continue
            if first is None or local_ts < first:
                first = local_ts
            if last is None or local_ts > last:
                last = local_ts
            if not _in_window(local_ts, since, until):
                continue
            item_id = str(row.get("itemId") or "")
            key = (item_id, str(dispatched_at))
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                {
                    "itemId": item_id,
                    "kind": row.get("kind"),
                    "title": row.get("title"),
                    "dispatchedAt": local_ts.isoformat(sep=" "),
                    "dispatchedAtUtc": str(dispatched_at),
                }
            )

    matches.sort(key=lambda entry: entry["dispatchedAt"])
    result["count"] = len(matches)
    result["entries"] = matches[:limit]
    result["log_first"] = first.isoformat(sep=" ") if first else None
    result["log_last"] = last.isoformat(sep=" ") if last else None
    if first is None:
        result["covered"] = False
        result["note"] = "dispatch log is empty — count unavailable"
        return result
    covered = first <= since if since is not None else True
    if until is not None and last is not None:
        covered = covered and last >= until
    result["covered"] = covered
    if not covered:
        result["note"] = (
            "dispatch log does not cover the requested window "
            f"(retained span {result['log_first']} .. {result['log_last']}); "
            "a count of 0 means 'not retained', not 'no dispatches'"
        )
    return result


# ---------------------------------------------------------------------------
# Coverage + verdict
# ---------------------------------------------------------------------------


def measure_coverage(
    log_dir: Path,
    *,
    since: dt.datetime | None = None,
    until: dt.datetime | None = None,
    max_gap_seconds: float = 3600.0,
    window_closed: bool = False,
) -> dict:
    """Coverage verdict for the window, built on ``scripts/lib/proxy_logs.py``.

    Adjacent-file gaps come from the shared library (which also handles the
    dot/dash/``.gz`` discovery). The head gap is always hard — a window that
    starts before its own data is measuring a shorter window than it claims.
    The tail gap is hard only with ``window_closed`` (a fully elapsed window
    with a long tail hole has lost data; a live window simply has no end yet).
    """
    log_dir = Path(log_dir)
    coverage = measure_log_coverage(
        log_dir, since=since, until=until, max_gap_seconds=max_gap_seconds
    )
    spans = [
        {
            "file": span.path.name,
            "first": span.first.isoformat(sep=" "),
            "last": span.last.isoformat(sep=" "),
            "lines": span.line_count,
        }
        for span in coverage.spans
    ]
    gaps = [
        {
            "kind": "between_files",
            "before": gap.before.name,
            "before_last": gap.before_last.isoformat(sep=" "),
            "after": gap.after.name,
            "after_first": gap.after_first.isoformat(sep=" "),
            "gap_seconds": gap.gap_seconds,
        }
        for gap in coverage.gaps
    ]
    head_gap = coverage.window_start_gap_seconds
    tail_gap = coverage.window_end_gap_seconds
    if not spans:
        # Nothing retained inside the window. Distinguish a genuinely quiet
        # window (retention spans it, there were simply no lines) from a window
        # the retained logs never reach at all — the latter is a coverage gap,
        # never an "empty" result.
        retained = measure_log_coverage(log_dir)
        if retained.spans:
            retained_first = min(span.first for span in retained.spans)
            retained_last = max(span.last for span in retained.spans)
            if until is not None and retained_first >= until:
                gaps.append(
                    {
                        "kind": "outside_retention",
                        "before": None,
                        "before_last": until.isoformat(sep=" "),
                        "after": retained.spans[0].path.name,
                        "after_first": retained_first.isoformat(sep=" "),
                        "gap_seconds": (retained_first - until).total_seconds(),
                    }
                )
            elif since is not None and retained_last < since:
                gaps.append(
                    {
                        "kind": "outside_retention",
                        "before": retained.spans[-1].path.name,
                        "before_last": retained_last.isoformat(sep=" "),
                        "after": None,
                        "after_first": since.isoformat(sep=" "),
                        "gap_seconds": (since - retained_last).total_seconds(),
                    }
                )
    if head_gap is not None and head_gap > max_gap_seconds:
        gaps.insert(
            0,
            {
                "kind": "window_head",
                "before": None,
                "before_last": since.isoformat(sep=" ") if since else None,
                "after": spans[0]["file"] if spans else None,
                "after_first": spans[0]["first"] if spans else None,
                "gap_seconds": head_gap,
            },
        )
    if window_closed and tail_gap is not None and tail_gap > max_gap_seconds:
        gaps.append(
            {
                "kind": "window_tail",
                "before": spans[-1]["file"] if spans else None,
                "before_last": spans[-1]["last"] if spans else None,
                "after": None,
                "after_first": until.isoformat(sep=" ") if until else None,
                "gap_seconds": tail_gap,
            }
        )
    return {
        "covered": bool(spans) and not gaps,
        "max_gap_seconds": max_gap_seconds,
        "window_closed": window_closed,
        "spans": spans,
        "gaps": gaps,
        "window_start_gap_seconds": head_gap,
        "window_end_gap_seconds": tail_gap,
    }


def evaluate(
    metrics: dict,
    coverage: dict,
    dispatches: dict,
    *,
    stale_rate_target: float = 0.05,
    min_downtime_dispatches: int = 10,
    attribution: dict | None = None,
) -> dict:
    """Turn the measured figures into a verdict (never a silent pass)."""
    checks: list[dict] = [
        {
            "target": "coverage: requested window fully represented by logs",
            "observed": len(coverage["gaps"]),
            "pass": coverage["covered"],
        }
    ]
    if not coverage["covered"]:
        return {
            "verdict": VERDICT_COVERAGE_GAP,
            "checks": checks,
            "notes": [
                "requested window is not fully covered — figures below are a "
                "partial result and must not be reported as a measurement"
            ],
        }

    notes: list[str] = []
    if metrics["status_polls"] == 0:
        notes.append("no status polls in the window (empty window, not an error)")
        return {"verdict": VERDICT_EMPTY, "checks": checks, "notes": notes}

    stale_rate = metrics["slots_stale"]["rate"]
    checks.append(
        {
            "target": f"slots_stale rate < {stale_rate_target:.1%} of status polls",
            "observed": stale_rate,
            "pass": stale_rate < stale_rate_target,
        }
    )
    violations = metrics["saturated_without_live_work"]["count"]
    checks.append(
        {
            "target": "capacity_hint=saturated only with live local work",
            "observed": violations,
            "pass": violations == 0,
        }
    )
    if dispatches["covered"]:
        checks.append(
            {
                "target": (
                    f"herdr downtime dispatches >= {min_downtime_dispatches}"
                ),
                "observed": dispatches["count"],
                "pass": dispatches["count"] >= min_downtime_dispatches,
            }
        )
    else:
        notes.append(
            "herdr downtime dispatch count not verifiable for this window: "
            f"{dispatches['note'] or 'dispatch log does not cover it'}"
        )
    if metrics["capacity_hint"]["histogram"].get("saturated", 0) and not metrics[
        "capacity_hint"
    ]["histogram"].get("idle", 0):
        notes.append(
            "every poll in the window reported capacity_hint=saturated — "
            "check the status surface before trusting the rate"
        )
    if attribution is not None and violations:
        # Attribution explains a residual; it never excuses it — the check
        # above stays failed and the mechanism is stated alongside.
        by_cause = ", ".join(
            f"{cause}={count}"
            for cause, count in attribution["by_cause"].items()
        ) or "none"
        notes.append(
            f"{attribution['attributed_polls']} of {violations} violating polls "
            f"attributed to a preceding abandoned local dispatch or proxy "
            f"restart within {attribution['lookback_seconds']:.0f}s "
            f"(by cause: {by_cause})"
        )
        if attribution["unattributed_polls"]:
            notes.append(
                f"{attribution['unattributed_polls']} violating polls are "
                "unattributable from the proxy log (no preceding abandonment "
                "or restart inside the lookback); see the attribution rows"
            )
    stale_attribution = metrics.get("slots_stale_attribution")
    if stale_attribution and stale_attribution["unattributed"]:
        notes.append(
            f"{stale_attribution['unattributed']} stale polls carry no "
            "per-source freshness flag, so the cache they were served from "
            "is unknown"
        )
    return {
        "verdict": (
            VERDICT_PASS
            if all(check["pass"] for check in checks)
            else VERDICT_TARGETS_MISSED
        ),
        "checks": checks,
        "notes": notes,
    }


def measure(
    log_dir: Path,
    *,
    since: dt.datetime | None = None,
    until: dt.datetime | None = None,
    dispatch_log: Path | None = DEFAULT_DISPATCH_LOG,
    max_gap_seconds: float = 3600.0,
    window_closed: bool = False,
    label: str | None = None,
    limit: int = 10,
    stale_rate_target: float = 0.05,
    min_downtime_dispatches: int = 10,
    attribution_lookback_seconds: float = 600.0,
) -> dict:
    """Full measurement: meta + metrics + coverage + attribution + verdict.

    The attribution block turns the violating ``saturated`` polls and the
    stale-poll source split into evidence-backed explanations (AC4-style
    attribution, LP-0MVOG7IP80003MLV), so a target miss ships with its cause
    rather than a bare number.
    """
    metrics = analyse_status_stream(
        log_dir, since=since, until=until, limit=limit
    )
    dispatches = load_downtime_dispatches(
        dispatch_log, since=since, until=until, limit=limit
    )
    coverage = measure_coverage(
        log_dir,
        since=since,
        until=until,
        max_gap_seconds=max_gap_seconds,
        window_closed=window_closed,
    )
    events = analyse_abandonment_events(
        log_dir,
        # Read the lookback *before* the window too: the first polls of a
        # window are routinely explained by an abandonment moments earlier,
        # so scanning only from ``since`` would mark them unattributable for
        # an artificial reason.
        since=(
            since - dt.timedelta(seconds=attribution_lookback_seconds)
            if since is not None
            else None
        ),
        until=until,
    )
    attribution = attribute_violations(
        metrics["saturated_without_live_work"]["rows"],
        events,
        lookback_seconds=attribution_lookback_seconds,
    )
    by_reason: dict[str, int] = {}
    in_window = 0
    for event in events:
        # The reported event tally stays scoped to the measured window; only
        # the attribution is allowed to look at the lead-in.
        if since is not None and event["timestamp"] < since.strftime(
            "%Y-%m-%d %H:%M:%S"
        ):
            continue
        in_window += 1
        reason = event["reason"] or event["kind"]
        by_reason[reason] = by_reason.get(reason, 0) + 1
    abandonment_events = {
        "count": in_window,
        "by_reason": dict(sorted(by_reason.items())),
    }
    verdict = evaluate(
        metrics,
        coverage,
        dispatches,
        stale_rate_target=stale_rate_target,
        min_downtime_dispatches=min_downtime_dispatches,
        attribution=attribution,
    )
    meta = {
        "tool": TOOL,
        "work_item": WORK_ITEM,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "log_dir": str(log_dir),
        "files": [path.name for path in discover_proxy_log_files(Path(log_dir))],
        "since": since.isoformat(sep=" ") if since else None,
        "until": until.isoformat(sep=" ") if until else None,
        "label": label,
        "window_closed": window_closed,
        "targets": {
            "stale_rate": stale_rate_target,
            "min_downtime_dispatches": min_downtime_dispatches,
            "max_gap_seconds": max_gap_seconds,
        },
    }
    return {
        "meta": meta,
        "coverage": coverage,
        "metrics": metrics,
        "abandonment_events": abandonment_events,
        "attribution": attribution,
        "herdr_downtime_dispatches": dispatches,
        "verdict": verdict,
    }


def write_artifacts(report: dict, out_dir: Path) -> list[Path]:
    """Persist the raw measurement and the per-poll attribution rows.

    ``measurement.json`` is the full report (tool-``--json`` output, coverage
    verdict, every derivation); ``measurement.txt`` the human rendering; the
    CSV the per-violating-poll attribution rows a reviewer can re-check line
    by line (AC5 artifact surface, LP-0MVOG7IP80003MLV).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "measurement.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "measurement.txt").write_text(render(report) + "\n", encoding="utf-8")
    csv_path = out_dir / "saturated_without_live_work_attribution.csv"
    rows = report["attribution"]["rows"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "cause",
                "evidence_timestamp",
                "seconds_since_evidence",
                "latency_ms",
                "slots_detail_stale",
                "available_slots",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return [out_dir / "measurement.json", out_dir / "measurement.txt", csv_path]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(report: dict) -> str:
    """Human-readable summary of a :func:`measure` report."""
    meta = report["meta"]
    metrics = report["metrics"]
    coverage = report["coverage"]
    dispatches = report["herdr_downtime_dispatches"]
    verdict = report["verdict"]
    out: list[str] = [
        f"Capacity-signal verification ({meta['work_item']})",
        "=" * 66,
        f"log_dir : {meta['log_dir']}",
        f"window  : {meta['since'] or '(start)'} .. {meta['until'] or '(end)'}",
        f"label   : {meta['label'] or '(none)'}",
        f"files   : {', '.join(meta['files']) or '(none)'}",
        "",
        f"status polls                    : {metrics['status_polls']}",
        "slots_stale                     : "
        f"{metrics['slots_stale']['count']} / "
        f"{metrics['slots_stale']['denominator']} "
        f"({metrics['slots_stale']['rate']:.2%})",
    ]
    histogram = metrics["capacity_hint"]["histogram"]
    if histogram:
        rendered = ", ".join(
            f"{key}={value} ({metrics['capacity_hint']['rates'][key]:.2%})"
            for key, value in histogram.items()
        )
        out.append(f"capacity_hint                   : {rendered}")
    violations = metrics["saturated_without_live_work"]
    out.append(
        "saturated without live work     : "
        f"{violations['count']} ({violations['rate_of_polls']:.2%} of polls)"
    )
    stale_attribution = metrics.get("slots_stale_attribution") or {}
    if stale_attribution:
        by_source = stale_attribution["by_source"]
        out.append(
            "  stale by cache source         : "
            + ", ".join(f"{key}={value}" for key, value in by_source.items())
        )
        bands = stale_attribution["by_latency_band"]
        out.append(
            "  stale by poll latency         : "
            + ", ".join(f"{key}={value}" for key, value in bands.items())
        )
    attribution = report.get("attribution")
    if attribution is not None and violations["count"]:
        by_cause = ", ".join(
            f"{cause}={count}"
            for cause, count in attribution["by_cause"].items()
        ) or "none"
        out.append(
            "  violating polls attributed    : "
            f"{attribution['attributed_polls']}/{violations['count']} "
            f"({by_cause}; lookback {attribution['lookback_seconds']:.0f}s)"
        )
        if attribution["unattributed_polls"]:
            out.append(
                "  unattributable violating polls: "
                f"{attribution['unattributed_polls']} "
                f"({', '.join(attribution['unattributed_timestamps'][:5])}"
                + ("…" if attribution["unattributed_polls"] > 5 else "")
                + ")"
            )
    abandonment = report.get("abandonment_events") or {}
    if abandonment:
        out.append(
            "abandoned local dispatches      : "
            f"{abandonment['count']} "
            f"({', '.join(f'{key}={value}' for key, value in abandonment['by_reason'].items()) or 'none'})"
        )
    no_slots = metrics["saturated_without_available_slots"]
    out.append(
        "saturated, available_slots=0    : "
        f"{no_slots['count']} ({no_slots['rate_of_polls']:.2%} of polls)"
    )
    if violations["worst_timestamps"]:
        worst = ", ".join(
            f"{entry['minute']}({entry['polls']})"
            for entry in violations["worst_timestamps"]
        )
        out.append(f"  worst minutes                 : {worst}")
    if violations["examples"]:
        out.append(f"  first violating polls         : {', '.join(violations['examples'])}")
    local = metrics["local_dispatches"]
    out.append(
        "local dispatches                : "
        f"{local['count']} (sessions={local['distinct_sessions']}, "
        f"herdr={local['herdr_session_dispatches']}, "
        f"cache={local['by_cache_state']})"
    )
    out.append(
        "herdr downtime dispatches       : "
        f"{dispatches['count']} "
        f"({dispatches['path'] or '(no log)'}, covered={dispatches['covered']})"
    )
    if dispatches["note"]:
        out.append(f"  note                          : {dispatches['note']}")
    if coverage["gaps"]:
        out.append(f"coverage gaps                   : {len(coverage['gaps'])}")
        for gap in coverage["gaps"]:
            out.append(
                f"  {gap['kind']}: {gap['before_last']} -> {gap['after_first']} "
                f"({gap['gap_seconds']:.0f}s)"
            )
    else:
        out.append("coverage gaps                   : none")
    out.append("")
    out.append(f"VERDICT: {verdict['verdict']}")
    for check in verdict["checks"]:
        out.append(
            f"  [{'ok ' if check['pass'] else 'FAIL'}] {check['target']} "
            f"(observed {check['observed']})"
        )
    for note in verdict["notes"]:
        out.append(f"  note: {note}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible capacity-signal metrics (status polls, slots_stale "
            "rate, capacity_hint distribution, saturated-without-live-work, "
            "dispatch counts) for a proxy-log window."
        )
    )
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument(
        "--since", default=None, help="inclusive local lower bound (YYYY-MM-DD [HH:MM[:SS]])"
    )
    parser.add_argument(
        "--until", default=None, help="exclusive local upper bound (same formats)"
    )
    parser.add_argument(
        "--dispatch-log",
        default=str(DEFAULT_DISPATCH_LOG),
        help=(
            "herdr downtime dispatch JSONL log "
            f"(default {DEFAULT_DISPATCH_LOG})"
        ),
    )
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=3600.0,
        help="largest tolerated hole inside the window (default 3600)",
    )
    parser.add_argument(
        "--window-closed",
        action="store_true",
        help="assert the window has fully elapsed (makes a tail gap fatal)",
    )
    parser.add_argument(
        "--stale-rate-target",
        type=float,
        default=0.05,
        help="slots_stale rate target as a fraction (default 0.05)",
    )
    parser.add_argument(
        "--min-downtime-dispatches",
        type=int,
        default=10,
        help="herdr downtime dispatch count target (default 10)",
    )
    parser.add_argument(
        "--fail-on-target-miss",
        action="store_true",
        help="exit 2 when a target is missed (default: report only, exit 0)",
    )
    parser.add_argument("--label", default=None, help="free-form label in meta")
    parser.add_argument(
        "--attribution-lookback-seconds",
        type=float,
        default=600.0,
        help=(
            "how far before a violating poll an abandonment/restart event may "
            "sit and still explain it (default 600)"
        ),
    )
    parser.add_argument(
        "--write-artifacts",
        default=None,
        metavar="DIR",
        help=(
            "write measurement.json, measurement.txt and the per-poll "
            "attribution CSV into DIR"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="worst timestamps / dispatch rows / examples to list (default 10)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable JSON")
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"error: log directory not found: {log_dir}", file=sys.stderr)
        return EXIT_ERROR
    if not discover_proxy_log_files(log_dir):
        print(f"error: no proxy logs found in {log_dir}", file=sys.stderr)
        return EXIT_ERROR

    try:
        since = parse_bound(args.since) if args.since else None
        until = parse_bound(args.until) if args.until else None
    except argparse.ArgumentTypeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    report = measure(
        log_dir,
        since=since,
        until=until,
        dispatch_log=Path(args.dispatch_log) if args.dispatch_log else None,
        max_gap_seconds=args.max_gap_seconds,
        window_closed=args.window_closed,
        label=args.label,
        limit=args.limit,
        stale_rate_target=args.stale_rate_target,
        min_downtime_dispatches=args.min_downtime_dispatches,
        attribution_lookback_seconds=args.attribution_lookback_seconds,
    )

    if args.write_artifacts:
        for path in write_artifacts(report, Path(args.write_artifacts)):
            print(f"artifact: {path}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render(report))

    verdict = report["verdict"]["verdict"]
    if verdict == VERDICT_COVERAGE_GAP:
        return EXIT_COVERAGE_GAP
    if verdict == VERDICT_TARGETS_MISSED and args.fail_on_target_miss:
        return EXIT_TARGET_MISSED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
