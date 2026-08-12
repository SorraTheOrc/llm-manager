"""Proxy usage analysis — turn the last 24h of llama-proxy logs into
per-session CSVs (fast/cheap) and an operator-facing report.

Usage:
    python3 analyze_proxy_usage.py --log-dir /var/log/llama-proxy \\
        --hours 24 [--output-dir ~/proxy-usage-reports] [--json]

See SKILL.md for the full usage and interpretation guide.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make sibling modules importable when run as a plain script.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import config_loader
import reporting

DEFAULT_LOG_DIR = "/var/log/llama-proxy"
DEFAULT_HOURS = 24
DEFAULT_OUTPUT_DIR = "~/proxy-usage-reports"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze_proxy_usage.py",
        description=(
            "Analyze the last N hours of llama-proxy logs: per-session CSVs split "
            "by fast/cheap (from the slot schedule) plus a recommendations report."
        ),
    )
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help=f"log directory (default: {DEFAULT_LOG_DIR})")
    parser.add_argument(
        "--llama-log-dir",
        default=DEFAULT_LOG_DIR,
        help=(
            "directory containing llama-server.log* for decode/prompt-eval speed "
            f"stats (default: {DEFAULT_LOG_DIR}; falls back to --log-dir)"
        ),
    )
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS, help=f"analysis window in hours (default: {DEFAULT_HOURS})")
    parser.add_argument("--start", help="window start, ISO 'YYYY-MM-DD HH:MM:SS' (overrides --hours)")
    parser.add_argument("--end", help="window end, ISO 'YYYY-MM-DD HH:MM:SS' (defaults to now)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--config", help="path to proxy/config.yaml (default: auto-discovered)")
    parser.add_argument("--json", action="store_true", help="print a JSON summary instead of the text summary")
    parser.add_argument("--quiet", action="store_true", help="suppress the stdout summary")
    return parser.parse_args(argv)


def _parse_iso(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.fromisoformat(value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.end:
        window_end = _parse_iso(args.end)
    else:
        window_end = datetime.now()
    if args.start:
        window_start = _parse_iso(args.start)
    else:
        window_start = window_end - timedelta(hours=args.hours)

    if window_start >= window_end:
        print("error: window start must be before window end", file=sys.stderr)
        return 2

    config = config_loader.load_proxy_config(config_loader.find_config_path(args.config))

    try:
        run = reporting.run_analysis(
            log_dir=Path(args.log_dir),
            window_start=window_start,
            window_end=window_end,
            output_dir=Path(args.output_dir).expanduser(),
            config=config,
            llama_log_dir=Path(args.llama_log_dir),
        )
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(reporting.summary_to_json(run.summary), indent=2, sort_keys=True))
        return 0
    if not args.quiet:
        data = reporting.summary_to_json(run.summary)
        decode = data.get("decode_speed") or {}
        decode_summary = (
            f", decode speed {decode.get('samples', 0)} samples "
            f"(median {decode.get('median_tok_s', '-')} tok/s)"
        )
        errors = data.get("errors") or 0
        busy = data.get("local_busy")
        busy_summary = ""
        if busy:
            busy_summary = (
                f", local model busy {busy['busy_pct']:.1f}% "
                f"({busy['busy_seconds']:.0f}s of {busy['window_seconds']:.0f}s "
                f"window, peak concurrency {busy['peak_concurrency']})"
            )
        print(
            f"Analyzed {len(run.files)} log file(s) from {args.log_dir}: "
            f"{data['sessions']} sessions, {data['total_requests']} requests "
            f"(local {data['local_requests']} / remote {data['remote_requests']}), "
            f"{data['fallback_events']} fallback events "
            f"({data['fallback_rate'] * 100:.1f}%), "
            f"{data.get('contention_dispatch', 0)} queued-dispatched-local, "
            f"{data.get('contention_fallback_after_queue', 0)} fallback-after-queue, "
            f"{errors} error event(s){decode_summary}{busy_summary}."
        )
        print(f"Outputs written to {args.output_dir}: "
              f"fast_sessions.csv, cheap_sessions.csv, errors.csv, report.md")
        if run.archived_to:
            print(f"Previous outputs archived to {run.archived_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
