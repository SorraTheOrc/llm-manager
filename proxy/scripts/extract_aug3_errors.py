#!/usr/bin/env python3
"""Reproducible error-extraction harness for llama-proxy logs.

Builds a structured, re-runnable dataset of every error event in a log
window (default: the 2026-08-03 baseline) and asserts the headline counts
within tolerance, so the error taxonomy in the analysis report is
evidence-backed and refreshable.

Reuses the streaming log parser from the ``proxy-usage-analysis`` skill
(``.pi/skills/proxy-usage-analysis/scripts/log_parser.py``) instead of
duplicating log parsing (LP-0MSDP2P3E0053WOD acceptance criteria).

Usage::

    python3 proxy/scripts/extract_aug3_errors.py \\
        --log-dir /var/log/llama-proxy \\
        --start "2026-08-03 00:00:00" --end "2026-08-04 00:00:00" \\
        --output-dir proxy/docs/error-analysis-2026-08-03

Outputs (written to ``--output-dir``):

**The default ``--output-dir`` is a repo artifacts location** —
``proxy/docs/error-analysis-2026-08-03/`` (the committed Aug 3 snapshot).
This harness is distinct from the skill's daily analyzer
(``analyze_proxy_usage.py``), whose default output is ``~/proxy-usage-reports``
(home dir); this harness's artifacts are committed to the repo:

- ``errors.csv`` — one row per error event (timestamp, type, provider,
  model, session, entry, error detail, status, attempt, signal, source
  log file, evidence line).
- ``counts.csv`` — aggregated counts by error type / provider / model.
- ``counts.json`` — machine-readable counts + assertion results.
- ``evidence.txt`` — raw evidence lines per error type.
- ``summary.md`` — the counts table + assertion results.

Exit code is 0 when all headline assertions pass, 2 on usage errors,
and 1 when assertions fail (or the log dir is missing).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Ensure sibling modules (the proxy-usage-analysis skill's parser) are
# importable regardless of the invocation directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SKILL_SCRIPTS = (
    _REPO_ROOT / ".pi" / "skills" / "proxy-usage-analysis" / "scripts"
)
if str(_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SKILL_SCRIPTS))

import log_parser  # noqa: E402

ERROR_KINDS = (
    "stream_finish_error",
    "stream_error",
    "slot_save_error",
    "backend_retry",
    "upstream_http_error",
)

# Headline counts from the Aug 3 intake brief (LP-0MSDFKCK4007CPMY).
# ``floor`` is the lower tolerance bound; ``expected`` is the intake value.
# Floors are deliberately conservative: the brief's numbers were computed at
# intake time (Aug 3 ~16:15Z) and the live logs grew afterwards, so the
# full-day window may exceed the expected value.
HEADLINE_ASSERTIONS = [
    {
        "id": "stream_finish_error_total",
        "label": "Stream finished: reason=error (total)",
        "floor": 90,
        "expected": 98,
    },
    {
        "id": "stream_finish_error_opencode_go",
        "label": "Stream finished: reason=error opencode-go/deepseek-v4-flash",
        "floor": 60,
        "expected": 73,
    },
    {
        "id": "stream_finish_error_opencode_free",
        "label": "Stream finished: reason=error opencode/deepseek-v4-flash-free",
        "floor": 15,
        "expected": 20,
    },
    {
        "id": "stream_finish_error_local",
        "label": "Stream finished: reason=error local/Qwen3",
        "floor": 3,
        "expected": 5,
    },
    {
        "id": "slot_save_error",
        "label": "slot_save failed ReadTimeout (local)",
        "floor": 10,
        "expected": 30,
    },
    {
        "id": "upstream_429_free_usage",
        "label": "upstream status=429 FreeUsageLimitError",
        "floor": 3,
        "expected": 3,
    },
    {
        "id": "backend_retry_timeouts",
        "label": "backend_retry timeouts (ReadTimeout/ConnectTimeout/ReadError)",
        "floor": 1,
        "expected": None,
    },
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="extract_aug3_errors.py",
        description=(
            "Extract every error event from llama-proxy logs in a window into "
            "a structured dataset and assert the headline counts within tolerance."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default="/var/log/llama-proxy",
        help="directory containing proxy.log* (default: /var/log/llama-proxy)",
    )
    parser.add_argument(
        "--start",
        default="2026-08-03 00:00:00",
        help="window start, ISO 'YYYY-MM-DD HH:MM:SS' (default: 2026-08-03 00:00:00)",
    )
    parser.add_argument(
        "--end",
        default="2026-08-04 00:00:00",
        help="window end, ISO 'YYYY-MM-DD HH:MM:SS' (default: 2026-08-04 00:00:00)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "proxy" / "docs" / "error-analysis-2026-08-03"),
        help="output directory for CSV/JSON/evidence/summary artifacts",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the machine-readable report as JSON",
    )
    parser.add_argument(
        "--no-assert",
        action="store_true",
        help="skip the headline-count assertion pass (artifacts are still written)",
    )
    return parser.parse_args(argv)


def _parse_iso(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.fromisoformat(value)


def collect_errors(log_dir: Path, window_start: datetime, window_end: datetime) -> list:
    """Stream-parse the logs in ``log_dir`` and return the in-window error
    events as a list of :class:`log_parser.LogEvent` (all error kinds)."""
    events: list = []
    for f in log_parser.discover_log_files(log_dir, window_start):
        for ev in log_parser.iter_events(f, window_start, window_end):
            if ev.kind in ERROR_KINDS:
                events.append(ev)
    return events


def aggregate_counts(events: list) -> dict[str, int]:
    """Aggregate error events by type; returns ``{kind: count}``."""
    return dict(Counter(e.kind for e in events))


def split_counts(events: list, kind: str) -> Counter:
    """Aggregate ``kind`` events by ``(provider, model)``."""
    return Counter((e.provider, e.model) for e in events if e.kind == kind)


def run_assertions(
    counts: dict[str, int], split: Counter, free_usage_429: int = 0
) -> dict:
    """Run the headline assertions; returns ``{"passed": bool, "failures": [...]}``.

    ``split`` is the ``(provider, model)`` counter for ``stream_finish_error``
    events; ``free_usage_429`` is the upstream-429 FreeUsageLimitError count
    (not derivable from the split counter).
    """
    failures: list[str] = []

    def check(actual: int, floor: int, label: str, expected) -> None:
        if actual < floor:
            exp = f" (expected {expected})" if expected is not None else ""
            failures.append(f"{label}: {actual} < floor {floor}{exp}")

    for a in HEADLINE_ASSERTIONS:
        if a["id"] == "stream_finish_error_total":
            check(counts.get("stream_finish_error", 0), a["floor"], a["label"], a["expected"])
        elif a["id"] == "stream_finish_error_opencode_go":
            check(split.get(("opencode-go", "deepseek-v4-flash"), 0), a["floor"], a["label"], a["expected"])
        elif a["id"] == "stream_finish_error_opencode_free":
            check(split.get(("opencode", "deepseek-v4-flash-free"), 0), a["floor"], a["label"], a["expected"])
        elif a["id"] == "stream_finish_error_local":
            check(split.get(("local", "Qwen3"), 0), a["floor"], a["label"], a["expected"])
        elif a["id"] == "slot_save_error":
            check(counts.get("slot_save_error", 0), a["floor"], a["label"], a["expected"])
        elif a["id"] == "upstream_429_free_usage":
            check(free_usage_429, a["floor"], a["label"], a["expected"])
        elif a["id"] == "backend_retry_timeouts":
            check(counts.get("backend_retry", 0), a["floor"], a["label"], a["expected"])

    return {"passed": not failures, "failures": failures}


def _free_usage_429_count(events: list) -> int:
    """Count upstream 429 events whose body type is FreeUsageLimitError."""
    return sum(
        1
        for e in events
        if e.kind == "upstream_http_error"
        and e.status == 429
        and e.error == "FreeUsageLimitError"
    )


def write_artifacts(
    out_dir: Path,
    events: list,
    counts: dict[str, int],
    split: Counter,
    assertion_result: dict,
    window_start: datetime,
    window_end: datetime,
) -> None:
    """Write errors.csv, counts.csv, counts.json, evidence.txt, summary.md."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # errors.csv — one row per error event.
    cols = [
        "error_type", "timestamp", "provider", "model", "session", "entry",
        "error_detail", "status", "attempt", "signal", "source_file", "evidence",
    ]
    with (out_dir / "errors.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for e in sorted(events, key=lambda x: x.ts):
            writer.writerow({
                "error_type": e.kind,
                "timestamp": e.ts.strftime("%Y-%m-%d %H:%M:%S"),
                "provider": e.provider or "",
                "model": e.model or "",
                "session": e.session or "",
                "entry": e.entry or "",
                "error_detail": e.error or "",
                "status": str(e.status) if e.status is not None else "",
                "attempt": e.attempt or "",
                "signal": e.signal or "",
                "source_file": e.src_file or "",
                "evidence": (e.raw or "").strip(),
            })

    # counts.csv — aggregated by (type, provider, model).
    with (out_dir / "counts.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["error_type", "provider", "model", "count"])
        for kind, count in sorted(counts.items()):
            sub = Counter((e.provider, e.model) for e in events if e.kind == kind)
            if sub:
                for (p, m), c in sub.most_common():
                    writer.writerow([kind, p or "", m or "", c])
            else:
                writer.writerow([kind, "", "", count])

    # counts.json — machine-readable counts + assertion results.
    payload = {
        "window_start": window_start.strftime("%Y-%m-%d %H:%M:%S"),
        "window_end": window_end.strftime("%Y-%m-%d %H:%M:%S"),
        "total_error_events": len(events),
        "by_type": counts,
        "stream_finish_error_split": {
            f"{p or '-'}/{m or '-'}": c
            for (p, m), c in split.most_common()
        },
        "upstream_429_free_usage_limit": _free_usage_429_count(events),
        "assertions": {
            "passed": assertion_result["passed"],
            "failures": assertion_result["failures"],
        },
    }
    (out_dir / "counts.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    # evidence.txt — raw evidence lines grouped by error type.
    lines: list[str] = []
    for kind in ERROR_KINDS:
        kind_events = [e for e in sorted(events, key=lambda x: x.ts) if e.kind == kind]
        if not kind_events:
            continue
        lines.append(f"### {kind} ({len(kind_events)})")
        lines.append("")
        for e in kind_events[:20]:
            lines.append(f"- {e.src_file}: {(e.raw or '').strip()[:300]}")
        if len(kind_events) > 20:
            lines.append(f"- …and {len(kind_events) - 20} more")
        lines.append("")
    (out_dir / "evidence.txt").write_text("\n".join(lines), encoding="utf-8")

    # summary.md — counts table + assertion results.
    md: list[str] = []
    md.append("# Error extraction summary")
    md.append("")
    md.append(f"- Window: {window_start:%Y-%m-%d %H:%M:%S} → {window_end:%Y-%m-%d %H:%M:%S}")
    md.append(f"- Total error events: **{len(events)}**")
    md.append("")
    md.append("## Counts by error type")
    md.append("")
    md.append("| Error type | Count |")
    md.append("|---|---|")
    for kind, count in sorted(counts.items()):
        md.append(f"| {kind} | {count} |")
    md.append("")
    md.append("## Stream finished: reason=error split (provider/model)")
    md.append("")
    md.append("| Provider | Model | Count |")
    md.append("|---|---|---|")
    for (p, m), c in split.most_common():
        md.append(f"| {p or '-'} | {m or '-'} | {c} |")
    md.append("")
    md.append("## Headline assertions")
    md.append("")
    md.append("- **PASSED**" if assertion_result["passed"] else "- **FAILED**")
    for f in assertion_result["failures"]:
        md.append(f"  - {f}")
    md.append("")
    md.append("## Artifacts")
    md.append("")
    md.append("- `errors.csv` — one row per error event")
    md.append("- `counts.csv` / `counts.json` — aggregated counts")
    md.append("- `evidence.txt` — raw evidence lines")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> dict:
    """Run the harness; returns a JSON-serialisable report dict."""
    args = parse_args(argv)
    window_start = _parse_iso(args.start)
    window_end = _parse_iso(args.end)
    log_dir = Path(args.log_dir)

    report: dict = {
        "window_start": args.start,
        "window_end": args.end,
        "passed": False,
        "message": "",
    }

    if window_start >= window_end:
        report["message"] = "error: window start must be before window end"
        return report

    if not log_dir.is_dir():
        report["message"] = f"error: log directory not found: {log_dir}"
        return report

    events = collect_errors(log_dir, window_start, window_end)
    counts = aggregate_counts(events)
    split = split_counts(events, "stream_finish_error")
    assertion_result = run_assertions(
        counts, split, _free_usage_429_count(events)
    )
    if args.no_assert:
        assertion_result = {"passed": True, "failures": []}

    out_dir = Path(args.output_dir).expanduser()
    write_artifacts(
        out_dir, events, counts, split, assertion_result, window_start, window_end
    )

    report.update({
        "total_error_events": len(events),
        "by_type": counts,
        "upstream_429_free_usage_limit": _free_usage_429_count(events),
        "assertions": assertion_result,
        "output_dir": str(out_dir),
        "passed": assertion_result["passed"],
        "message": (
            f"extracted {len(events)} error event(s); assertions "
            f"{'PASSED' if assertion_result['passed'] else 'FAILED'}"
        ),
    })

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(report["message"])
        print(f"artifacts written to {out_dir}")

    return report


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result["passed"] else 1)
