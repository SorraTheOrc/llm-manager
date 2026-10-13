#!/usr/bin/env python3
"""Reproducible verification of the liveness-based prefill lease extension.

Measures, from the proxy log stream, the events that define the long-local-
prefill kill/retry loop and whether it has stopped. This is the measurement
behind the live post-deploy verification of the restored extension
(LP-0MUXQ5QL40027VZ4) and its in-flight follow-up (LP-0MUXTW4MD003TVIC),
tracked by LP-0MUXQNPKE008K9MP.

Metrics (mirroring the verification acceptance criteria):

  * ``lease_extended_total`` / ``lease_extended_alive_true`` — AC1: the
    ``lease_extended_during_prefill ... progress=None alive=True`` liveness
    shape (and the ``in_flight=True`` follow-up shape) appear at all; before
    the fix the count was zero.
  * ``no_progress_watchdog_releases`` — AC2: ``lease_released ...
    reason=no_progress_watchdog`` kills. These are the events that fired for
    every local prefill longer than the no-progress timeout before the fix.
  * ``dispatch_first_byte_timeout`` and ``backend_retry_read_timeout`` —
    AC2: the client gave up / retried signals.
  * ``prefill_progress_unobservable`` — context: how often ``/slots`` gave
    no usable progress (the reason liveness/in-flight extension is needed).
  * ``local_stream_finishes`` — locally served completions.
  * ``long_prefill_local_finishes`` — AC3: local completions whose preceding
    prefill held the lease for more than ``--long-prefill-seconds``. These
    are the prefills that used to be re-queued repeatedly.

Usage:
  ./scripts/prefill_lease_verification.py                          # summary
  ./scripts/prefill_lease_verification.py --json                   # machine readable
  ./scripts/prefill_lease_verification.py --since '2026-10-12 06:07' \
      --until '2026-10-12 22:10' --json > after.json
  ./scripts/prefill_lease_verification.py --label post-fix --json

Exit codes:
  0 - success
  1 - no proxy logs found / unexpected error
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from lib.proxy_logs import discover_proxy_log_files, open_proxy_log_text  # noqa: E402

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")
_EXT_RE = re.compile(r"lease_extended_during_prefill session=(\S+)")
_WATCHDOG_RE = re.compile(r"lease_released session=(\S+) reason=no_progress_watchdog")
_FINISH_RE = re.compile(r"Stream finished:.*session=(\S+) provider=(\S+)")

_TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")


def _parse_ts(text: str) -> dt.datetime | None:
    m = _TS_RE.match(text)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")


def _parse_bound(text: str) -> dt.datetime:
    for fmt in _TS_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"invalid timestamp {text!r} (expected 'YYYY-MM-DD [HH:MM[:SS]]')"
    )


def analyse(
    log_dir: Path,
    *,
    since: dt.datetime | None = None,
    until: dt.datetime | None = None,
    long_prefill_seconds: int = 180,
    extension_gap_seconds: int = 90,
) -> dict:
    metrics = {
        "lease_extended_total": 0,
        "lease_extended_alive_true": 0,
        "lease_extended_alive_false": 0,
        "lease_extended_in_flight_true": 0,
        "no_progress_watchdog_releases": 0,
        "dispatch_first_byte_timeout": 0,
        "backend_retry_read_timeout": 0,
        "prefill_progress_unobservable": 0,
        "local_stream_finishes": 0,
        "long_prefill_local_finishes": 0,
    }

    # Per-session ordered (ts, kind); used for the long-prefill run analysis.
    per_session: dict[str, list[tuple[dt.datetime, str]]] = defaultdict(list)

    for path in discover_proxy_log_files(log_dir):
        with open_proxy_log_text(path) as fh:
            for line in fh:
                ts = _parse_ts(line)
                if ts is None:
                    continue
                if since is not None and ts < since:
                    continue
                if until is not None and ts >= until:
                    continue

                if "dispatch_first_byte_timeout" in line:
                    metrics["dispatch_first_byte_timeout"] += 1
                    continue
                if "backend_retry" in line and "error=ReadTimeout" in line:
                    metrics["backend_retry_read_timeout"] += 1
                    continue
                if "prefill_progress_unobservable" in line:
                    metrics["prefill_progress_unobservable"] += 1
                    continue

                if "lease_extended_during_prefill" in line:
                    m = _EXT_RE.search(line)
                    if not m:
                        continue
                    metrics["lease_extended_total"] += 1
                    if "alive=True" in line:
                        metrics["lease_extended_alive_true"] += 1
                    else:
                        metrics["lease_extended_alive_false"] += 1
                    if "in_flight=True" in line:
                        metrics["lease_extended_in_flight_true"] += 1
                    per_session[m.group(1)].append((ts, "ext"))
                    continue

                if "reason=no_progress_watchdog" in line:
                    m = _WATCHDOG_RE.search(line)
                    if m:
                        metrics["no_progress_watchdog_releases"] += 1
                    continue

                if "Stream finished:" in line:
                    m = _FINISH_RE.search(line)
                    if m and m.group(2) == "local":
                        metrics["local_stream_finishes"] += 1
                        per_session[m.group(1)].append((ts, "finish"))

    # AC3: count local finishes whose preceding prefill held the lease for
    # longer than ``long_prefill_seconds``. A "run" is a consecutive sequence
    # of extensions for one session with no gap larger than
    # ``extension_gap_seconds``; a local finish closes the run.
    gap = float(extension_gap_seconds)
    threshold = float(long_prefill_seconds)
    for events in per_session.values():
        events.sort(key=lambda e: e[0])
        run_start: dt.datetime | None = None
        last_ext: dt.datetime | None = None
        for ts, kind in events:
            if kind == "ext":
                if last_ext is None or (ts - last_ext).total_seconds() > gap:
                    run_start = ts
                last_ext = ts
            else:  # local finish
                if (
                    last_ext is not None
                    and (ts - last_ext).total_seconds() <= gap
                    and run_start is not None
                    and (last_ext - run_start).total_seconds() > threshold
                ):
                    metrics["long_prefill_local_finishes"] += 1
                run_start = None
                last_ext = None

    return metrics


def _build_meta(log_dir, since, until, long_prefill_seconds, label) -> dict:
    files = [p.name for p in discover_proxy_log_files(Path(log_dir))]
    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "log_dir": str(log_dir),
        "files": files,
        "since": since.isoformat(sep=" ") if since else None,
        "until": until.isoformat(sep=" ") if until else None,
        "long_prefill_seconds": long_prefill_seconds,
        "label": label,
    }


def _render(meta: dict, metrics: dict) -> str:
    out = []
    out.append("Prefill lease-extension verification (LP-0MUXQNPKE008K9MP)")
    out.append("=" * 66)
    out.append(f"log_dir : {meta['log_dir']}")
    out.append(f"files   : {', '.join(meta['files']) or '(none)'}")
    out.append(f"window  : {meta['since'] or '(start)'} .. {meta['until'] or '(end)'}")
    out.append("")
    out.append(f"lease_extended_during_prefill      : {metrics['lease_extended_total']}")
    out.append(f"  ... alive=True (liveness shape)  : {metrics['lease_extended_alive_true']}")
    out.append(f"  ... alive=False                  : {metrics['lease_extended_alive_false']}")
    out.append(f"  ... in_flight=True (AC1 shape)   : {metrics['lease_extended_in_flight_true']}")
    out.append(f"lease_released no_progress_watchdog: {metrics['no_progress_watchdog_releases']}")
    out.append(f"dispatch_first_byte_timeout        : {metrics['dispatch_first_byte_timeout']}")
    out.append(f"backend_retry error=ReadTimeout    : {metrics['backend_retry_read_timeout']}")
    out.append(f"prefill_progress_unobservable      : {metrics['prefill_progress_unobservable']}")
    out.append(f"local stream finishes              : {metrics['local_stream_finishes']}")
    out.append(
        f"long local prefill completions     : {metrics['long_prefill_local_finishes']} "
        f"(> {meta['long_prefill_seconds']}s)"
    )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="/var/log/llama-proxy")
    parser.add_argument("--since", default=None, help="inclusive lower bound (YYYY-MM-DD [HH:MM[:SS]])")
    parser.add_argument("--until", default=None, help="exclusive upper bound")
    parser.add_argument("--long-prefill-seconds", type=int, default=180)
    parser.add_argument("--extension-gap-seconds", type=int, default=90)
    parser.add_argument("--label", default=None, help="free-form label recorded in meta")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"error: log directory not found: {log_dir}", file=sys.stderr)
        return 1
    if not discover_proxy_log_files(log_dir):
        print(f"error: no proxy logs found in {log_dir}", file=sys.stderr)
        return 1

    since = _parse_bound(args.since) if args.since else None
    until = _parse_bound(args.until) if args.until else None

    metrics = analyse(
        log_dir,
        since=since,
        until=until,
        long_prefill_seconds=args.long_prefill_seconds,
        extension_gap_seconds=args.extension_gap_seconds,
    )
    meta = _build_meta(log_dir, since, until, args.long_prefill_seconds, args.label)

    if args.json:
        print(json.dumps({"meta": meta, "metrics": metrics}, indent=2))
    else:
        print(_render(meta, metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
