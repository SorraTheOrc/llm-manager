#!/usr/bin/env python3
"""Re-measure prompt-eval (prefill) throughput from llama-server's own log.

Why this exists (LP-0MWCEHDRV000B7HD)
-------------------------------------

The shipped local route serialises prefills (``local_max_concurrent_prefills:
1``) with ``--ubatch-size 1024`` and ``--parallel 2``. The recorded baseline
says aggregate prefill throughput is ~254 tok/s and that overlapping prefills
halved per-request throughput while aggregate stayed flat (W3,
``proxy/docs/routing.md``). Re-measuring that needs GPU time, which competes
with live traffic.

llama-server already writes the ground truth for every request it serves::

    [43745] prompt eval time = 57125.15 ms / 18362 tokens ( 3.11 ms per token, 321.43 tokens per second)

This script parses those lines out of the retained server logs and reports,
per window and per prompt-size bucket:

- the **per-request** rate distribution (p50 / p90) — what a single caller
  sees, and
- the **aggregate** rate (``sum(tokens) / sum(prompt_ms)``) — what the engine
  actually sustained. The two differ whenever requests are serialised in a
  queue or overlap on the GPU; only the aggregate answers "does the gate bind".

Zero-token lines are skipped: llama.cpp prints ``0 tokens`` for a fully cached
prompt, which is a cache hit rather than a prefill measurement and would
otherwise flatten the distribution.

Usage::

    python3 scripts/prefill_throughput_report.py --log-dir /var/log/llama-proxy \\
        --since 2026-11-10 --json

    # bounded window: only the named files, still reported with their coverage
    python3 scripts/prefill_throughput_report.py \\
        --files /var/log/llama-proxy/llama-server.log --json
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ubatch_ab_benchmark import percentile

#: ``prompt eval time = 57125.15 ms / 18362 tokens ( 3.11 ms per token, 321.43 tokens per second)``
PROMPT_EVAL_RE = re.compile(
    r"prompt eval time\s*=\s*(?P<ms>\d+(?:\.\d+)?)\s*ms\s*/\s*"
    r"(?P<tokens>\d+)\s*tokens\s*\(.*?,\s*(?P<rate>\d+(?:\.\d+)?)\s*tokens per second\)"
)

#: Rotated server logs are named ``llama-server.log-YYYY-MM-DD.gz``.
ROTATED_RE = re.compile(r"llama-server\.log-(?P<day>\d{4}-\d{2}-\d{2})(?:\.gz)?$")

#: Prompt-size buckets the item quotes. The 40–80k band is the regime that
#: produces the minutes-long first-byte stalls, so it is reported on its own.
BUCKETS = ("lt8k", "8k-20k", "20k-40k", "40k-80k", "ge80k")

_BUCKET_EDGES = (
    (8_000, "lt8k"),
    (20_000, "8k-20k"),
    (40_000, "20k-40k"),
    (80_000, "40k-80k"),
)


def _open_text(path: Path):
    """Open a plain or gzip-compressed log as text."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, errors="replace")


def parse_prompt_eval_lines(source) -> list[dict]:
    """Extract measured prompt-eval timings from log line(s).

    *source* is an iterable of lines (a file, a list, a gzip handle) or a single
    ``Path``. Only well-formed lines reporting a **positive** token count are
    kept, so a cache hit or a truncated line can never enter the aggregate.
    """
    if isinstance(source, (str, Path)):
        with _open_text(Path(source)) as handle:
            return parse_prompt_eval_lines(handle)
    rows: list[dict] = []
    for line in source:
        if isinstance(line, Path):
            rows.extend(parse_prompt_eval_lines(line))
            continue
        match = PROMPT_EVAL_RE.search(line)
        if not match:
            continue
        tokens = int(match.group("tokens"))
        milliseconds = float(match.group("ms"))
        if tokens <= 0 or milliseconds <= 0:
            continue
        rows.append(
            {
                "prompt_tokens": tokens,
                "prompt_ms": milliseconds,
                "tokens_per_second": float(match.group("rate")),
            }
        )
    return rows


def bucket_for(tokens: int) -> str:
    for upper, name in _BUCKET_EDGES:
        if tokens < upper:
            return name
    return "ge80k"


def aggregate_tokens_per_second(rows: list[dict]) -> float:
    """Aggregate throughput: total tokens over total engine milliseconds."""
    total_ms = sum(float(row["prompt_ms"]) for row in rows)
    if total_ms <= 0:
        return 0.0
    total_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    return round(total_tokens / (total_ms / 1000.0), 1)


def _block(rows: list[dict]) -> dict:
    rates = [
        int(row["prompt_tokens"]) / (float(row["prompt_ms"]) / 1000.0) for row in rows if float(row["prompt_ms"]) > 0
    ]
    return {
        "requests": len(rows),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in rows),
        "prompt_ms": round(sum(float(row["prompt_ms"]) for row in rows), 1),
        "aggregate_tokens_per_second": aggregate_tokens_per_second(rows),
        "per_request": {
            "p50": round(percentile(rates, 50), 1),
            "p90": round(percentile(rates, 90), 1),
            "min": round(min(rates), 1) if rates else 0.0,
            "max": round(max(rates), 1) if rates else 0.0,
        },
    }


def summarise(rows: list[dict]) -> dict:
    """Summarise a window: overall plus one block per prompt-size bucket."""
    summary = _block(rows)
    summary["buckets"] = {
        name: _block([row for row in rows if bucket_for(int(row["prompt_tokens"])) == name]) for name in BUCKETS
    }
    return summary


def log_coverage(path: Path, live_start: datetime | None = None) -> tuple[datetime, datetime]:
    """Coverage of one log file as UTC datetimes.

    Rotation is *dateext*: the rotated file is named with the instant it was
    rotated, so ``llama-server.log-2026-11-11[.gz]`` holds the content written
    before that rotation, i.e. ``[2026-11-10, 2026-11-11)``. The live file
    covers ``live_start`` (the last rotation instant, when the caller knows it)
    ..its last write; its lines carry no timestamps, so without ``live_start``
    the file's birth time is the honest fallback.
    """
    match = ROTATED_RE.search(path.name)
    if match:
        end = datetime.strptime(match.group("day"), "%Y-%m-%d").replace(tzinfo=UTC)
        return end - timedelta(days=1), end
    stat = path.stat()
    end = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
    start = live_start or datetime.fromtimestamp(stat.st_ctime, tz=UTC)
    return start, max(start, end)


def resolve_files(log_dir: Path, files: list[str] | None, since: str | None) -> list[Path]:
    """Pick the server-log files in scope, newest last."""
    if files:
        return [Path(name) for name in files]
    candidates = sorted(log_dir.glob("llama-server.log*"))
    if not since:
        return candidates
    since_day = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC)
    selected = []
    for path in candidates:
        start, end = log_coverage(path)
        if end > since_day:
            selected.append(path)
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="/var/log/llama-proxy")
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Explicit log files (overrides --log-dir/--since).",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Include logs whose coverage ends after this UTC day (YYYY-MM-DD).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    parser.add_argument(
        "--start",
        default=None,
        help=(
            "UTC start of the live log's content, 'YYYY-MM-DD HH:MM:SS'. The "
            "live file's lines carry no timestamps, so its start must be "
            "stated (normally the last rotation instant) to make the quoted "
            "window honest."
        ),
    )
    args = parser.parse_args(argv)

    paths = resolve_files(Path(args.log_dir), args.files, args.since)
    # The live file starts where the most recent rotation ended; that is the
    # only trustworthy start for content whose lines carry no timestamps.
    live_end = datetime.now(UTC)
    rotated_ends = [log_coverage(path)[1] for path in paths if path.exists() and ROTATED_RE.search(path.name)]
    live_start = max((end for end in rotated_ends if end <= live_end), default=None)
    if args.start:
        live_start = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    rows: list[dict] = []
    coverages = []
    for path in paths:
        if not path.exists():
            continue
        coverages.append((str(path), *log_coverage(path, live_start=live_start)))
        rows.extend(parse_prompt_eval_lines(path))

    summary = summarise(rows)
    summary["window"] = {
        "files": [name for name, _start, _end in coverages],
        "start": min((start for _n, start, _e in coverages), default=None),
        "end": max((end for _n, _s, end in coverages), default=None),
    }
    summary["window"]["start"] = summary["window"]["start"].isoformat() if summary["window"]["start"] else None
    summary["window"]["end"] = summary["window"]["end"].isoformat() if summary["window"]["end"] else None
    summary["generated_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(json.dumps(summary, indent=2))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
