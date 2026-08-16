"""Streaming, line-based parser for llama-server eval-timing lines.

llama-server (router mode) writes one log per backend instance. Child
instances are identified by a ``[<port>]`` prefix; the router parent logs a
``spawning server instance with name=Qwen3 on port <port>`` line at the top
of each file, and the Qwen3 child port changes on every restart. Eval timing
lines carry **no timestamp**:

    [32999] prompt eval time =   29504.01 ms / 11449 tokens (    2.58 ms per token,   388.05 tokens per second)
    [32999]        eval time =    3776.71 ms /   153 tokens (   24.68 ms per token,    40.51 tokens per second)

Because there are no per-line timestamps, each sample's timestamp is
approximated by the log file's last-write (mtime) time; fast/cheap bucketing
and window filtering use that approximation (documented in the report).

Parsing is tolerant: lines that do not match the eval-timing shape are
skipped (never fatal), and files where the Qwen3 port cannot be discovered
are counted as skipped (their eval lines cannot be attributed to a model).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
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

# Log naming: live "llama-server.log" + rotated "llama-server.N.log".
LIVE_NAME = "llama-server.log"
ROTATED_NAME_RE = re.compile(r"^llama-server\.(\d+)\.log$")

MODEL_QWEN3 = "Qwen3"

KIND_DECODE = "decode"
KIND_PROMPT_EVAL = "prompt_eval"

# Bucket keys used by SpeedStats.
TOTAL = "total"
FAST = "fast"
CHEAP = "cheap"
_BUCKET_KEYS = (TOTAL, FAST, CHEAP)


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


def qwen3_port(path: Path) -> int | None:
    """Return the Qwen3 child port announced in ``path``, or ``None``.

    Only the router parent emits the spawn line, at the top of each file
    after a restart, so scanning stops at the first match.
    """
    try:
        fh = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return None
    with fh:
        for line in fh:
            m = QWEN3_SPAWN_RE.search(line)
            if m is not None:
                return int(m.group(1))
    return None


def discover_llama_logs(log_dir: Path, window_start: datetime) -> list[Path]:
    """Return the llama-server log files in ``log_dir`` that can overlap the
    analysis window, sorted by name.

    The live ``llama-server.log`` is always included. Rotated files
    (``llama-server.N.log``) carry no timestamp in their name, so overlap is
    decided by mtime (last write before rotation): a rotated file is included
    iff mtime >= window_start.
    """
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return []
    candidates: list[Path] = []
    for p in sorted(log_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name == LIVE_NAME:
            candidates.append(p)
            continue
        if ROTATED_NAME_RE.match(p.name) is None:
            continue
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
        except OSError:
            continue
        if mtime >= window_start:
            candidates.append(p)
    return sorted(candidates, key=lambda p: p.name)


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
    malformed lines are skipped, never fatal.

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
    is_live = path.name == LIVE_NAME
    if is_live:
        if live_span_start is not None and window_end < live_span_start:
            return
        sample_ts = min(mtime, window_end)
        if sample_ts < window_start:
            return
    else:
        if not (window_start <= mtime <= window_end):
            return
        sample_ts = mtime
    try:
        fh = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return
    with fh:
        for line in fh:
            ev = parse_eval_line(line)
            if ev is None:
                continue
            if port is not None and ev.port != port:
                continue
            ev.ts = sample_ts
            ev.model = MODEL_QWEN3
            yield ev


@dataclass
class SpeedBucket:
    """Percentile summary of one kind of speed sample in one bucket."""

    count: int
    median: float | None
    p90: float | None
    p10: float | None


@dataclass
class SpeedStats:
    """Aggregated llama-server eval-timing speed stats.

    ``decode`` and ``prompt_eval`` map a bucket key (``total``/``fast``/
    ``cheap``) to a :class:`SpeedBucket`. ``files_skipped`` counts discovered
    files whose Qwen3 port could not be determined (their eval lines cannot
    be attributed).
    """

    decode: dict[str, SpeedBucket]
    prompt_eval: dict[str, SpeedBucket]
    files_parsed: int
    files_skipped: int


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


def build_speed_stats(
    files: list[Path],
    window_start: datetime,
    window_end: datetime,
    schedule: object | None,
    mode_map: object | None = None,
) -> SpeedStats:
    """Aggregate eval-timing samples across ``files`` into :class:`SpeedStats`.

    Each sample is bucketed by its (mtime-approximated) timestamp using the
    fast/cheap periods — mode-aware (LP-0MSPZUD4G007IYGH) when the logs show
    mode transitions, so samples written during cheap hours count as cheap
    even when the analysis runs in fast mode; the total bucket always
    accumulates every sample. Files whose Qwen3 port cannot be discovered are
    counted in ``files_skipped`` and their eval lines are ignored (never
    fatal).
    """
    samples = {
        KIND_DECODE: {key: [] for key in _BUCKET_KEYS},
        KIND_PROMPT_EVAL: {key: [] for key in _BUCKET_KEYS},
    }
    files_parsed = 0
    files_skipped = 0

    def _bucket_key_for(ts: datetime) -> str:
        if mode_map is not None and getattr(mode_map, "transitions", None):
            return CHEAP if mode_map.period_for(ts).label == CHEAP else FAST
        if schedule is not None and getattr(schedule, "periods", None):
            return CHEAP if schedule.period_for(ts).label == CHEAP else FAST
        return FAST

    # The live llama-server.log's content starts when the previous file was
    # rotated (llama-server.1.log's last-write time); used to skip the live
    # file for windows that ended before its content began.
    live_span_start: datetime | None = None
    for path in files:
        if path.name == "llama-server.1.log":
            try:
                live_span_start = datetime.fromtimestamp(path.stat().st_mtime)
            except OSError:
                pass
            break

    for path in files:
        port = qwen3_port(path)
        if port is None:
            files_skipped += 1
            continue
        files_parsed += 1
        for ev in iter_eval_timings(
            path, port, window_start, window_end, live_span_start=live_span_start
        ):
            bucket_key = _bucket_key_for(ev.ts)
            group = samples[ev.kind]
            group[TOTAL].append(ev.tok_s)
            group[bucket_key].append(ev.tok_s)

    return SpeedStats(
        decode=_bucket_stats(samples[KIND_DECODE]),
        prompt_eval=_bucket_stats(samples[KIND_PROMPT_EVAL]),
        files_parsed=files_parsed,
        files_skipped=files_skipped,
    )
