"""Proxy usage analysis — turn the last 24h of llama-proxy logs into
per-session CSVs (fast/cheap) and an operator-facing report.

Usage:
    python3 analyze_proxy_usage.py --log-dir /var/log/llama-proxy \\
        --hours 24 [--output-dir ~/proxy-usage-reports] [--json]

The daily cron invocation needs no performance-metric flags: the committed
``metrics/performance_metrics.yaml`` defaults plus a live
``<output-dir>/performance_metrics_state.yaml`` are used automatically, and
``--json`` carries a ``performance_metrics`` block with each metric's
baseline/target/actual/verdict. ``--metrics-config`` / ``--metrics-state``
relocate those files, ``--no-metrics-update`` renders the section without
recording the window, and ``--seed-metrics-history [DIR]`` runs the one-off
backfill from archived reports instead of analysing logs.

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

import capacity_signals
import config_loader
import performance_metrics
import reporting

DEFAULT_LOG_DIR = "/var/log/llama-proxy"
DEFAULT_HOURS = 24
DEFAULT_OUTPUT_DIR = "~/proxy-usage-reports"
#: Default source for the one-off scorecard backfill: one subdirectory per
#: archived window, each holding that run's ``report.md``.
DEFAULT_METRICS_ARCHIVE_DIR = "~/proxy-usage-reports"


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
    parser.add_argument(
        "--stale-rate-target",
        type=float,
        default=capacity_signals.DEFAULT_STALE_RATE_TARGET,
        help=(
            "target share of status polls allowed to report slots_stale=true, "
            "evaluated in the report's 'Local capacity signals' section and in "
            "--json (default: %(default)s, i.e. 5%%); mirrors "
            "scripts/capacity_signal_verification.py --stale-rate-target"
        ),
    )
    parser.add_argument("--json", action="store_true", help="print a JSON summary instead of the text summary")
    parser.add_argument("--quiet", action="store_true", help="suppress the stdout summary")
    # Performance scorecard (LP-0MW1BUAQD001WL8X). Every default keeps the
    # unchanged 05:00 cron invocation working as-is: the committed defaults
    # file plus a live state file under --output-dir, updated on each run.
    parser.add_argument(
        "--metrics-config",
        default=str(performance_metrics.DEFAULTS_PATH),
        help=(
            "path to the performance-metric defaults (metric detail copy and any "
            "seeded history); read-only at runtime, never written "
            f"(default: {performance_metrics.DEFAULTS_PATH})"
        ),
    )
    parser.add_argument(
        "--metrics-state",
        help=(
            "path to the live performance-metric history file; this run records "
            "its window here "
            f"(default: <output-dir>/{performance_metrics.STATE_FILENAME})"
        ),
    )
    parser.add_argument(
        "--no-metrics-update",
        action="store_true",
        help=(
            "render the performance metrics section but do not record this "
            "window (no state write)"
        ),
    )
    parser.add_argument(
        "--seed-metrics-history",
        nargs="?",
        const=DEFAULT_METRICS_ARCHIVE_DIR,
        metavar="DIR",
        help=(
            "one-off backfill: seed the best-3 history from archived reports and "
            "exit without analysing logs; DIR holds one subdirectory per archived "
            f"window, each with a report.md (default: {DEFAULT_METRICS_ARCHIVE_DIR})"
        ),
    )
    return parser.parse_args(argv)


def _parse_iso(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.fromisoformat(value)


def _metrics_paths(args: argparse.Namespace, out_dir: Path) -> tuple[Path, Path]:
    """Resolve the committed defaults path and the live state file path.

    ``--metrics-state`` wins when given; otherwise the live history lives in
    the report output directory, exactly where the unchanged cron invocation
    expects it (LP-0MW1BUAQD001WL8X AC1).
    """
    metrics_config = Path(args.metrics_config).expanduser()
    metrics_state = (
        Path(args.metrics_state).expanduser()
        if args.metrics_state
        else out_dir / performance_metrics.STATE_FILENAME
    )
    return metrics_config, metrics_state


def _seed_metrics_history(
    archive_dir: Path, metrics_config: Path, metrics_state: Path
) -> dict:
    """Seed the best-3 history from archived report directories (idempotent).

    Each immediate subdirectory of *archive_dir* holding a ``report.md`` is one
    archived window, identified by the directory name (the daily run names its
    archive directory after the window it covers). Only the live state file is
    written; the committed defaults file is read, never modified, and
    re-running the seed never duplicates a window's measurement
    (LP-0MW1BUB1Z004AK21).
    """
    reports: list[tuple[str, str]] = []
    for report_path in sorted(archive_dir.glob("*/report.md")):
        try:
            reports.append(
                (
                    report_path.parent.name,
                    report_path.read_text(encoding="utf-8", errors="replace"),
                )
            )
        except OSError:
            continue
    state = performance_metrics.load_state(metrics_state, defaults_path=metrics_config)
    result = performance_metrics.seed_history(state, reports)
    performance_metrics.save_state(state, metrics_state)
    result["windows"] = len(reports)
    result["state_path"] = str(metrics_state)
    return result


def _print_seed_summary(result: dict) -> None:
    """UK-English summary of a backfill run: seeded / already present / skipped."""
    print(
        f"Seeded {result['seeded']} measurement(s) from {result['windows']} archived "
        f"report window(s); {result['already_present']} measurement(s) were already "
        f"present and left untouched."
    )
    skipped = result.get("skipped_metrics") or {}
    if skipped:
        print("Metrics with no parseable value in the archive (history accrues from the next run):")
        for metric_id, reason in skipped.items():
            print(f"  {metric_id}: {reason}")
    print(f"Live state file written: {result['state_path']}")


def _summary_output_paths(out_dir: Path) -> list[str]:
    """Return the output artifact paths in the preferred summary order (report first)."""
    return [
        str(out_dir / "report.md"),
        str(out_dir / "standard_sessions.csv"),
        str(out_dir / "low_priority_sessions.csv"),
        str(out_dir / "errors.csv"),
        str(out_dir / "errors.json"),
    ]


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

    out_dir = Path(args.output_dir).expanduser().resolve()
    metrics_config, metrics_state = _metrics_paths(args, out_dir)

    if args.seed_metrics_history:
        # One-off backfill is an early-exit mode: it reads archived reports and
        # never needs (or touches) a live log window.
        archive_dir = Path(args.seed_metrics_history).expanduser()
        if not archive_dir.is_dir():
            print(f"error: archive directory not found: {archive_dir}", file=sys.stderr)
            return 1
        result = _seed_metrics_history(archive_dir, metrics_config, metrics_state)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        elif not args.quiet:
            _print_seed_summary(result)
        return 0

    try:
        run = reporting.run_analysis(
            log_dir=Path(args.log_dir),
            window_start=window_start,
            window_end=window_end,
            output_dir=out_dir,
            config=config,
            llama_log_dir=Path(args.llama_log_dir),
            stale_rate_target=args.stale_rate_target,
            metrics_config=metrics_config,
            metrics_state=metrics_state,
            update_metrics=not args.no_metrics_update,
        )
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                reporting.summary_to_json(
                    run.summary,
                    run.mode_map,
                    config,
                    run.stale_rate_target,
                    metrics_config=metrics_config,
                    metrics_state=metrics_state,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.quiet:
        data = reporting.summary_to_json(
            run.summary,
            run.mode_map,
            config,
            run.stale_rate_target,
            metrics_config=metrics_config,
            metrics_state=metrics_state,
        )
        decode = data.get("decode_speed") or {}
        decode_summary = (
            f", decode speed {decode.get('samples', 0)} samples "
            f"(median {decode.get('median_tok_s', '-')} tok/s)"
        )
        errors = data.get("errors") or 0
        busy = data.get("local_busy")
        busy_summary = ""
        if busy:
            # Corrected busy figure (LP-0MVJNW6SN008AKH3): the dispatch →
            # finish union, with the retained decode-only lower bound named
            # next to it so the definition change stays visible in the text
            # summary (``--quiet`` suppresses this line entirely).
            busy_summary = (
                f", local model busy (dispatch → finish) {busy['busy_pct']:.1f}% "
                f"({busy['busy_seconds']:.0f}s of {busy['window_seconds']:.0f}s "
                f"window, peak concurrency {busy['peak_concurrency']}), "
                f"decode-only busy (lower bound) {busy['decode_busy_seconds']:.0f}s "
                f"(prefill busy {busy['prefill_busy_seconds']:.0f}s)"
            )
            gpu = data.get("gpu_busy")
            if gpu:
                # Hardware activity, not served local work (LP-0MVUIM7A50053EFN
                # AC2): the power figure is supplementary telemetry and must
                # never read as the local model utilisation figure, which is
                # the log-derived ``local_busy`` printed above.
                busy_summary += (
                    f", GPU busy — power-based (hardware, not local model "
                    f"utilisation) {gpu['busy_pct']:.1f}% "
                    f"({gpu['busy_seconds']:.0f}s)"
                )
        print(
            f"Analyzed {len(run.files)} log file(s) from {args.log_dir}: "
            f"{data['sessions']} sessions, {data['total_requests']} requests "
            f"(local {data['local_requests']} / remote {data['remote_requests']}), "
            f"{data['fallback_events']} fallback events "
            f"({data['fallback_rate'] * 100:.1f}%), "
            f"{data.get('contention_dispatch', 0)} queued-dispatched-local, "
            f"{data.get('contention_fallback_after_queue', 0)} fallback-after-queue, "
            f"{data.get('local_admission_reject', 0)} local-admission-refused "
            f"(prefill ETA), "
            f"{data.get('local_idle_admit', 0)} idle-admitted-local, "
            f"{data.get('residency_resident_admits', 0)} resident-prefix admits "
            f"/ {data.get('residency_evicted_denials', 0)} evicted denials "
            f"({data.get('residency_tenant_limit_denials', 0)} tenant-limit, "
            f"{data.get('local_evictions', 0)} evictions), "
            f"{errors} error event(s){decode_summary}{busy_summary}."
        )
        output_lines = _summary_output_paths(out_dir)
        print("Outputs written to:")
        for line in output_lines:
            print(f"  {line}")
        if run.archived_to:
            print(f"Previous outputs archived to {run.archived_to.resolve()}")
        if args.no_metrics_update:
            print("Performance metrics state: not updated (--no-metrics-update)")
        else:
            print(f"Performance metrics state written to: {metrics_state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
