"""Streaming, line-based parser for llama-server eval-timing lines.

llama-server (router mode) writes one log per backend instance. Child
instances are identified by a ``[<port>]`` prefix; the router parent logs a
``spawning server instance with name=Qwen3 on port <port>`` line at the top
of each file, and the Qwen3 child port changes on every restart. When the
rotation splits that spawn line away from the child's eval lines (real
observation: ``llama-server.N.log`` copies often start mid-child-log), the
port is inferred from the single, otherwise-unannounced port carried by the
file's eval-timing lines; files in which the port stays ambiguous are
skipped with a named reason (never fatal). Eval timing lines carry **no
timestamp**:

    [32999] prompt eval time =   29504.01 ms / 11449 tokens (    2.58 ms per token,   388.05 tokens per second)
    [32999]        eval time =    3776.71 ms /   153 tokens (   24.68 ms per token,    40.51 tokens per second)

Because there are no per-line timestamps, each sample's timestamp is
approximated by the log file's last-write (mtime) time; fast/cheap bucketing
and window filtering use that approximation (documented in the report).

Parsing is tolerant: lines that do not match the eval-timing shape are
skipped (never fatal), and files where the Qwen3 port cannot be discovered
are counted as skipped (their eval lines cannot be attributed to a model).

Every scan records diagnostic context on :class:`SpeedStats` — the log
directory, the files discovered, per-file status/reason with the port
discovery outcome and eval-timing lines matched, files outside the window
(rotation) and unrecognised rotation names — so an empty speed section can
state *why* it is empty (:func:`empty_window_reasons`) instead of rendering
an unattributed blank (LP-0MVJNIZRO008S031).

Rotation naming schemes recognised: the live ``llama-server.log``, numbered
in-process rotations ``llama-server.N.log`` and the logrotate daily scheme
``llama-server.log-YYYY-MM-DD`` / ``llama-server.log-YYYY-MM-DD.gz`` (read
transparently through gzip).
"""

from __future__ import annotations

import gzip
import math
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Optional "[<port>] " prefix + eval timing body. llama.cpp aligns columns with
# variable whitespace, so all separators are whitespace-tolerant:
#   "eval time = <ms> ms / <n> tokens (<ms/token> ms per token, <x> tokens per second)"
EVAL_LINE_RE = re.compile(
    r"^(?:\[(\d+)\]\s*)?(?P<kind>prompt eval time|eval time)\s*=\s*"
    r"(?P<ms>[\d.]+)\s*ms\s*/\s*(?P<tokens>\d+)\s*tokens\s*\(\s*[\d.]+\s*ms per token,\s*"
    r"(?P<tok_s>[\d.]+)\s*tokens per second\)\s*$"
)

# Router parent line announcing a Qwen3 child instance.
QWEN3_SPAWN_RE = re.compile(r"name=Qwen3 on port (\d+)")

# Router parent line announcing any child instance (used to exclude the
# embedding child's port from Qwen3 port inference).
SPAWN_MODEL_PORT_RE = re.compile(r"name=([A-Za-z0-9._-]+) on port (\d+)")

# Log naming: live "llama-server.log", in-process rotated "llama-server.N.log"
# and logrotate daily "llama-server.log-YYYY-MM-DD[.gz]".
LIVE_NAME = "llama-server.log"
ROTATED_NAME_RE = re.compile(r"^llama-server\.(\d+)\.log$")
DATED_ROTATION_RE = re.compile(r"^llama-server\.log-\d{4}-\d{2}-\d{2}(\.gz)?$")

MODEL_QWEN3 = "Qwen3"

KIND_DECODE = "decode"
KIND_PROMPT_EVAL = "prompt_eval"

# Bucket keys used by SpeedStats.
TOTAL = "total"
FAST = "fast"
CHEAP = "cheap"
_BUCKET_KEYS = (TOTAL, FAST, CHEAP)

# Per-file diagnostic statuses (FileDiagnostic.status).
STATUS_PARSED = "parsed"
STATUS_MISSING_PORT = "skipped_missing_port"
STATUS_OUTSIDE_WINDOW = "skipped_outside_window"
STATUS_UNREADABLE = "skipped_unreadable"


@dataclass
class EvalTiming:
    """One parsed eval-timing line.

    ``kind`` is ``decode`` (``eval time``) or ``prompt_eval`` (``prompt eval
    time``). ``ts`` and ``model`` are filled in by :func:`iter_eval_timings`
    (the raw lines carry no timestamp or model name).
    """

    kind: str
    port: int | None
    ms: float
    tokens: int
    tok_s: float
    model: str | None = None
    ts: datetime | None = None


def parse_eval_line(line: str) -> EvalTiming | None:
    """Parse one llama-server log line into an :class:`EvalTiming`, or
    ``None`` if the line is not an eval-timing line (or cannot be parsed).

    ``total time`` lines (which carry no tok/s) and non-eval lines return
    ``None``.
    """
    if not line:
        return None
    m = EVAL_LINE_RE.match(line)
    if m is None:
        return None
    port = int(m.group(1)) if m.group(1) else None
    kind = KIND_DECODE if m.group("kind") == "eval time" else KIND_PROMPT_EVAL
    return EvalTiming(
        kind=kind,
        port=port,
        ms=float(m.group("ms")),
        tokens=int(m.group("tokens")),
        tok_s=float(m.group("tok_s")),
    )


def _open_text(path: Path):
    """Return a text-mode handle for a plain or gzip-compressed log.

    Returns ``None`` when the file cannot be opened (unreadable/corrupt gzip
    bodies surface later as OSError on read; callers treat any OSError as an
    empty scan).
    """
    try:
        if path.name.endswith(".gz"):
            return gzip.open(path, "rt", encoding="utf-8", errors="replace")
        return path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return None


def qwen3_port(path: Path) -> int | None:
    """Return the Qwen3 child port announced in ``path``, or ``None``.

    Only the router parent emits the spawn line, at the top of each file
    after a restart, so scanning stops at the first match. Files where a
    rotation split the spawn line away return ``None`` here; their port is
    inferred from the eval-line prefixes by :func:`_resolve_port` during the
    speed scan.
    """
    fh = _open_text(path)
    if fh is None:
        return None
    with fh:
        try:
            for line in fh:
                m = QWEN3_SPAWN_RE.search(line)
                if m is not None:
                    return int(m.group(1))
        except OSError:
            return None
    return None


@dataclass
class _ScanResult:
    """One pass over a llama-server log: counts plus the parsed eval lines."""

    lines: int = 0
    events: list[EvalTiming] = field(default_factory=list)
    eval_ports: Counter = field(default_factory=Counter)
    spawn_port: int | None = None
    excluded_ports: set[int] = field(default_factory=set)


def _scan_log(path: Path) -> _ScanResult:
    """Scan ``path`` once, collecting line/eval counts and port evidence.

    The Qwen3 child port comes from the announced spawn line when present;
    ports announced for *other* models (e.g. ``name=mxbai-embed on port N``)
    are recorded as exclusions so inference can never attribute their eval
    lines to Qwen3. Unreadable files yield an empty result.
    """
    result = _ScanResult()
    fh = _open_text(path)
    if fh is None:
        return result
    with fh:
        try:
            for line in fh:
                result.lines += 1
                m = SPAWN_MODEL_PORT_RE.search(line)
                if m is not None:
                    name = m.group(1)
                    port = int(m.group(2))
                    if name == MODEL_QWEN3 and result.spawn_port is None:
                        result.spawn_port = port
                    elif name != MODEL_QWEN3:
                        result.excluded_ports.add(port)
                ev = parse_eval_line(line)
                if ev is not None:
                    result.events.append(ev)
                    if ev.port is not None:
                        result.eval_ports[ev.port] += 1
        except OSError:
            # Corrupt gzip body / read error: keep what was scanned so far.
            return result
    return result


def _resolve_port(scan: _ScanResult) -> tuple[int | None, str]:
    """Resolve the Qwen3 child port for one scanned file.

    Returns ``(port, source)`` where ``source`` is ``"spawn_line"`` (an
    explicit Qwen3 spawn line), ``"inferred"`` (no spawn line, but every
    eval-timing line carries a single other-model-unannounced port — the
    rotation-split child-log case) or ``"none"`` (ambiguous/absent).
    """
    if scan.spawn_port is not None:
        return scan.spawn_port, "spawn_line"
    candidates = [p for p in scan.eval_ports if p not in scan.excluded_ports]
    if len(candidates) == 1:
        return candidates[0], "inferred"
    return None, "none"


def _missing_port_reason(scan: _ScanResult) -> str:
    """Human-readable reason a file's eval lines cannot be attributed."""
    if not scan.eval_ports:
        return "no Qwen3 spawn line and no eval-timing lines to infer the child port from"
    ports = ", ".join(str(p) for p in sorted(scan.eval_ports))
    if all(p in scan.excluded_ports for p in scan.eval_ports):
        return f"eval-timing lines are attributed to a non-Qwen3 child port ({ports})"
    return (
        "no Qwen3 spawn line and eval-timing lines carry multiple ports "
        f"({ports}); ambiguous attribution"
    )


def _file_sample_ts(
    name: str,
    mtime: datetime,
    live_span_start: datetime | None,
    window_start: datetime,
    window_end: datetime,
) -> tuple[datetime | None, bool]:
    """Return ``(sample_ts, in_window)`` for one log file.

    Mirrors :func:`iter_eval_timings`: the live file is appended to
    continuously so its mtime can exceed ``window_end`` (clamped), and its
    content starts at the previous rotation (``live_span_start``); rotated
    files are in the window iff their last write is inside it.
    """
    if name == LIVE_NAME:
        if live_span_start is not None and window_end < live_span_start:
            return None, False
        sample_ts = min(mtime, window_end)
        if sample_ts < window_start:
            return None, False
        return sample_ts, True
    if not (window_start <= mtime <= window_end):
        return None, False
    return mtime, True


def _llama_candidates(
    log_dir: Path, window_start: datetime
) -> tuple[list[Path], list[str]]:
    """Enumerate the recognised llama-server log files in ``log_dir``.

    Returns ``(candidates, outside_window_names)``. The live
    ``llama-server.log`` is always a candidate. Rotated files (numbered
    ``llama-server.N.log`` and daily ``llama-server.log-YYYY-MM-DD[.gz]``)
    carry no timestamp in their name, so overlap is decided by mtime (last
    write ≈ rotation time): a rotated file is a candidate iff
    ``mtime >= window_start``; files below that bound are returned separately
    as ``outside_window_names`` (the rotation diagnostic).
    """
    log_dir = Path(log_dir)
    candidates: list[Path] = []
    outside: list[str] = []
    if not log_dir.is_dir():
        return candidates, outside
    for p in log_dir.iterdir():
        if not p.is_file():
            continue
        if p.name == LIVE_NAME:
            candidates.append(p)
            continue
        if ROTATED_NAME_RE.match(p.name) is None and DATED_ROTATION_RE.match(p.name) is None:
            continue
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
        except OSError:
            continue
        if mtime >= window_start:
            candidates.append(p)
        else:
            outside.append(p.name)
    candidates.sort(key=lambda p: p.name)
    outside.sort()
    return candidates, outside


def discover_llama_logs(log_dir: Path, window_start: datetime) -> list[Path]:
    """Return the llama-server log files in ``log_dir`` that can overlap the
    analysis window, sorted by name.

    The live ``llama-server.log`` is always included. Rotated files
    (``llama-server.N.log`` and daily ``llama-server.log-YYYY-MM-DD[.gz]``)
    carry no timestamp in their name, so overlap is decided by mtime (last
    write before rotation): a rotated file is included iff
    mtime >= window_start.
    """
    candidates, _outside = _llama_candidates(log_dir, window_start)
    return candidates


def iter_eval_timings(
    path: Path,
    port: int | None,
    window_start: datetime,
    window_end: datetime,
    live_span_start: datetime | None = None,
) -> Iterator[EvalTiming]:
    """Stream-parse ``path`` line by line, yielding only eval timings for the
    given child ``port`` whose (mtime-approximated) timestamp falls inside
    ``[window_start, window_end]``.

    The file is never loaded into memory: lines are read one at a time and a
    cheap regex check skips non-eval lines. ``total time`` lines and
    malformed lines are skipped, never fatal. Gzip-compressed rotated files
    are read transparently.

    The live ``llama-server.log`` is written continuously, so its mtime can
    exceed ``window_end`` (the file is appended to while the analysis runs).
    For the live file the sample timestamp is clamped to ``window_end`` and
    the upper-bound check is dropped; ``live_span_start`` (the previous
    rotation time, i.e. the mtime of ``llama-server.1.log``) is used to skip
    the live file entirely when the window ended before its content began.
    """
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return
    sample_ts, in_window = _file_sample_ts(
        path.name, mtime, live_span_start, window_start, window_end
    )
    if not in_window or sample_ts is None:
        return
    fh = _open_text(path)
    if fh is None:
        return
    with fh:
        try:
            for line in fh:
                ev = parse_eval_line(line)
                if ev is None:
                    continue
                if port is not None and ev.port != port:
                    continue
                ev.ts = sample_ts
                ev.model = MODEL_QWEN3
                yield ev
        except OSError:
            return


@dataclass
class SpeedBucket:
    """Percentile summary of one kind of speed sample in one bucket."""

    count: int
    median: float | None
    p90: float | None
    p10: float | None


@dataclass
class FileDiagnostic:
    """Per-file outcome of scanning a llama-server log for eval timings.

    ``status`` is one of ``parsed`` / ``skipped_missing_port`` /
    ``skipped_outside_window`` / ``skipped_unreadable``. ``reason`` is the
    human-readable explanation rendered in the report. ``port_source`` is
    ``spawn_line`` / ``inferred`` / ``none`` — the Qwen3 port discovery
    outcome for this file. ``eval_lines`` counts every parseable eval-timing
    line (any port) and ``eval_lines_matched`` those attributed to Qwen3 and
    inside the window (equal to ``samples_in_window``).
    """

    name: str
    status: str
    reason: str
    port: int | None
    port_source: str
    mtime: str | None
    lines_scanned: int
    eval_lines: int
    eval_lines_matched: int
    samples_in_window: int
    ports_seen: list[int] = field(default_factory=list)


@dataclass
class SpeedStats:
    """Aggregated llama-server eval-timing speed stats.

    ``decode`` and ``prompt_eval`` map a bucket key (``total``/``fast``/
    ``cheap``) to a :class:`SpeedBucket`. ``files_skipped`` counts discovered
    files whose Qwen3 port could not be determined (their eval lines cannot
    be attributed).

    The remaining fields carry the diagnostic context that lets an empty
    window explain itself (LP-0MVJNIZRO008S031): ``log_dir`` is the path
    scanned, ``files_discovered`` every candidate file, ``file_diagnostics``
    the per-file outcome/reason, ``outside_window_files`` files present but
    outside the window (rotation), ``ignored_logs`` llama-server-prefixed
    names whose rotation scheme is unrecognised, and ``qwen3_ports`` the
    distinct ports successfully attributed.
    """

    decode: dict[str, SpeedBucket]
    prompt_eval: dict[str, SpeedBucket]
    files_parsed: int
    files_skipped: int
    log_dir: str | None = None
    files_discovered: list[str] = field(default_factory=list)
    file_diagnostics: list[FileDiagnostic] = field(default_factory=list)
    outside_window_files: list[str] = field(default_factory=list)
    ignored_logs: list[str] = field(default_factory=list)
    qwen3_ports: list[int] = field(default_factory=list)


def _percentile(sorted_values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile (numpy default) over sorted values."""
    if not sorted_values:
        return None
    n = len(sorted_values)
    if n == 1:
        return round(sorted_values[0], 1)
    pos = p / 100.0 * (n - 1)
    lo = int(math.floor(pos))
    frac = pos - lo
    hi = lo + 1
    val = sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac
    return round(val, 1)


def _bucket_stats(samples: dict[str, list[float]]) -> dict[str, SpeedBucket]:
    return {
        key: SpeedBucket(
            count=len(vals),
            median=_percentile(sorted(vals), 50),
            p90=_percentile(sorted(vals), 90),
            p10=_percentile(sorted(vals), 10),
        )
        for key, vals in samples.items()
    }


def _fmt_mtime(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def build_speed_stats(
    files: list[Path],
    window_start: datetime,
    window_end: datetime,
    schedule: object | None,
    mode_map: object | None = None,
    log_dir: Path | None = None,
) -> SpeedStats:
    """Aggregate eval-timing samples across the llama-server logs into
    :class:`SpeedStats`, recording per-file diagnostic context.

    Each sample is bucketed by its (mtime-approximated) timestamp using the
    fast/cheap periods — mode-aware (LP-0MSPZUD4G007IYGH) when the logs show
    mode transitions, so samples written during cheap hours count as cheap
    even when the analysis runs in fast mode; the total bucket always
    accumulates every sample.

    When ``log_dir`` is given the directory is also scanned so recognised
    ``llama-server*`` files that ``files`` does not contain (outside the
    window — the rotation cause) and unrecognised rotation names can be
    reported; otherwise only the passed ``files`` are analysed. Files whose
    Qwen3 port cannot be determined
    are counted in ``files_skipped`` and their eval lines are ignored (never
    fatal); a file whose single unannounced eval-line port is unambiguous has
    that port inferred so its samples are still attributed.
    """
    samples = {
        KIND_DECODE: {key: [] for key in _BUCKET_KEYS},
        KIND_PROMPT_EVAL: {key: [] for key in _BUCKET_KEYS},
    }
    files_parsed = 0
    files_skipped = 0
    diagnostics: list[FileDiagnostic] = []
    outside: list[str] = []
    ignored: list[str] = []
    qwen3_ports: list[int] = []

    def _bucket_key_for(ts: datetime) -> str:
        if mode_map is not None and getattr(mode_map, "transitions", None):
            return CHEAP if mode_map.period_for(ts).label == CHEAP else FAST
        if schedule is not None and getattr(schedule, "periods", None):
            return CHEAP if schedule.period_for(ts).label == CHEAP else FAST
        return FAST

    if log_dir is not None:
        log_dir_p: Path | None = Path(log_dir)
    elif files:
        log_dir_p = files[0].parent
    else:
        log_dir_p = None

    if log_dir_p is not None:
        candidates = list(files)
        if log_dir_p.is_dir():
            candidate_names = {p.name for p in candidates}
            for p in sorted(log_dir_p.iterdir()):
                if not p.is_file() or not p.name.startswith("llama-server"):
                    continue
                recognised = (
                    p.name == LIVE_NAME
                    or ROTATED_NAME_RE.match(p.name)
                    or DATED_ROTATION_RE.match(p.name)
                )
                if not recognised:
                    ignored.append(p.name)
                elif p.name not in candidate_names:
                    outside.append(p.name)
                    try:
                        mtime = datetime.fromtimestamp(p.stat().st_mtime)
                        mtime_str: str | None = _fmt_mtime(mtime)
                    except OSError:
                        mtime_str = None
                    diagnostics.append(
                        FileDiagnostic(
                            name=p.name,
                            status=STATUS_OUTSIDE_WINDOW,
                            reason=(
                                "last write is before the window start "
                                f"{_fmt_mtime(window_start)} (rotation)"
                            ),
                            port=None,
                            port_source="none",
                            mtime=mtime_str,
                            lines_scanned=0,
                            eval_lines=0,
                            eval_lines_matched=0,
                            samples_in_window=0,
                        )
                    )
    else:
        candidates = list(files)

    # The live llama-server.log's content starts when the previous file was
    # rotated (llama-server.1.log's last-write time); used to skip the live
    # file for windows that ended before its content began.
    live_span_start: datetime | None = None
    for path in candidates:
        if path.name == "llama-server.1.log":
            try:
                live_span_start = datetime.fromtimestamp(path.stat().st_mtime)
            except OSError:
                pass
            break

    for path in candidates:
        name = path.name
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            diagnostics.append(
                FileDiagnostic(
                    name=name,
                    status=STATUS_UNREADABLE,
                    reason="file cannot be stat'd (unreadable or vanished)",
                    port=None,
                    port_source="none",
                    mtime=None,
                    lines_scanned=0,
                    eval_lines=0,
                    eval_lines_matched=0,
                    samples_in_window=0,
                )
            )
            continue
        sample_ts, in_window = _file_sample_ts(
            name, mtime, live_span_start, window_start, window_end
        )
        if not in_window or sample_ts is None:
            outside.append(name)
            diagnostics.append(
                FileDiagnostic(
                    name=name,
                    status=STATUS_OUTSIDE_WINDOW,
                    reason=(
                        f"last write {_fmt_mtime(mtime)} is outside the window "
                        f"[{_fmt_mtime(window_start)}, {_fmt_mtime(window_end)}] "
                        "(rotation)"
                    ),
                    port=None,
                    port_source="none",
                    mtime=_fmt_mtime(mtime),
                    lines_scanned=0,
                    eval_lines=0,
                    eval_lines_matched=0,
                    samples_in_window=0,
                )
            )
            continue

        scan = _scan_log(path)
        port, port_source = _resolve_port(scan)
        if port is None:
            files_skipped += 1
            diagnostics.append(
                FileDiagnostic(
                    name=name,
                    status=STATUS_MISSING_PORT,
                    reason=_missing_port_reason(scan),
                    port=None,
                    port_source="none",
                    mtime=_fmt_mtime(mtime),
                    lines_scanned=scan.lines,
                    eval_lines=len(scan.events),
                    eval_lines_matched=0,
                    samples_in_window=0,
                    ports_seen=sorted(scan.eval_ports),
                )
            )
            continue

        files_parsed += 1
        if port not in qwen3_ports:
            qwen3_ports.append(port)
        matched = 0
        bucket_key = _bucket_key_for(sample_ts)
        for ev in scan.events:
            if ev.port != port:
                continue
            matched += 1
            ev.ts = sample_ts
            ev.model = MODEL_QWEN3
            group = samples[ev.kind]
            group[TOTAL].append(ev.tok_s)
            group[bucket_key].append(ev.tok_s)
        diagnostics.append(
            FileDiagnostic(
                name=name,
                status=STATUS_PARSED,
                reason=(
                    f"Qwen3 port {port} ({port_source}); {matched} eval-timing "
                    "line(s) matched"
                ),
                port=port,
                port_source=port_source,
                mtime=_fmt_mtime(mtime),
                lines_scanned=scan.lines,
                eval_lines=len(scan.events),
                eval_lines_matched=matched,
                samples_in_window=matched,
            )
        )

    return SpeedStats(
        decode=_bucket_stats(samples[KIND_DECODE]),
        prompt_eval=_bucket_stats(samples[KIND_PROMPT_EVAL]),
        files_parsed=files_parsed,
        files_skipped=files_skipped,
        log_dir=str(log_dir_p) if log_dir_p is not None else None,
        files_discovered=[p.name for p in candidates],
        file_diagnostics=diagnostics,
        outside_window_files=outside,
        ignored_logs=ignored,
        qwen3_ports=qwen3_ports,
    )


def _summarise_names(names: list[str], limit: int = 8) -> str:
    """Join file names for a report line, truncating long rotations."""
    shown = ", ".join(names[:limit])
    if len(names) > limit:
        return f"{shown}, … and {len(names) - limit} more"
    return shown


def empty_window_reasons(stats: SpeedStats) -> list[str]:
    """Return the named, self-diagnosing reasons a window has no eval-timing
    samples.

    Distinguishes the causes the report must state instead of rendering an
    unattributed blank (LP-0MVJNIZRO008S031):

    - nothing to scan — no llama-server log files found (missing dir/rotation);
    - rotation — recognised log files fall outside the window;
    - verbosity/build — parsed files contain no ``eval time`` lines at all,
      so llama-server is not emitting eval timing;
    - attribution — eval-timing lines exist but the Qwen3 child port cannot
      be determined (non-Qwen3 port, or ambiguous multiple ports);
    - rotation/naming — ``llama-server*`` files with an unrecognised name are
      not scanned at all.

    Always returns at least one reason (a generic fallback keeps the report
    self-diagnosing even for a cause not anticipated here).
    """
    reasons: list[str] = []
    parsed = [d for d in stats.file_diagnostics if d.status == STATUS_PARSED]
    missing = [d for d in stats.file_diagnostics if d.status == STATUS_MISSING_PORT]
    outside = list(stats.outside_window_files or [])

    if not stats.file_diagnostics and not outside and not stats.ignored_logs:
        reasons.append(
            f"no llama-server log files were found in {stats.log_dir or '<unknown>'}"
            " (directory missing, empty, or unreadable) — there is no eval-timing "
            "data to scan"
        )
    if outside:
        reasons.append(
            f"rotation: {len(outside)} llama-server log file(s) fall outside the "
            f"analysis window ({_summarise_names(outside)}) — the window's "
            "eval-timing data was rotated away (or was not written yet)"
        )
    if missing:
        names = ", ".join(d.name for d in missing)
        ports = sorted({p for d in missing for p in d.ports_seen})
        detail = f"; ports seen: {', '.join(str(p) for p in ports)}" if ports else ""
        reasons.append(
            f"attribution failure: the Qwen3 child port could not be determined in "
            f"{len(missing)} file(s) ({names}){detail} — their eval-timing lines "
            "cannot be matched to a model"
        )
    no_eval = [d.name for d in parsed if d.eval_lines == 0 and d.lines_scanned > 0]
    if no_eval:
        reasons.append(
            f"verbosity/build: {len(no_eval)} parsed file(s) ({', '.join(no_eval)}) "
            "contain no `eval time` / `prompt eval time` lines at all — llama-server "
            "is running without eval-timing output (log verbosity or build flags), "
            "so there are no samples to attribute"
        )
    if stats.ignored_logs:
        reasons.append(
            f"rotation/naming: {len(stats.ignored_logs)} llama-server log file(s) "
            f"({_summarise_names(stats.ignored_logs)}) have an unrecognised rotation "
            "name and were not scanned"
        )
    if not reasons:
        reasons.append(
            "unknown: no eval-timing samples were found and none of the diagnosable "
            "causes (rotation, verbosity/build, attribution) applies — inspect the "
            "llama-server log directly"
        )
    return reasons
