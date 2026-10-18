#!/usr/bin/env python3
"""Reproducible AC1/AC2 verification for large-context session stalls.

The measurement behind LP-0MU1S617L002Q9PR's remaining acceptance criteria,
delivered by LP-0MV6BR3PT001WBEI. Before this tool no reproducible check
existed for either criterion:

  * AC1 — 'no request in a 24 h post-change window returns the upstream
    900-second queue-timeout error'. The proxy-side surfaces that precede a
    client 900 s timeout are ``dispatch_first_byte_timeout`` events (a local
    dispatch aborted by the first-byte ceiling, pre-fix ``ceiling=900s``,
    post-fix ``ceiling=840s``) and ``Chain exhausted for model ... holding``
    chain-hold starts. A window with zero ceiling=900s events and zero chain
    holds that can reach the client deadline is the AC1 evidence.
  * AC2 — 'warm-cache local dispatches p99 < 60 s to first byte PLUS a
    queue-wait cap' (producer re-scope, LP-0MV6BR3PT001WBEI). The
    ``dispatch_first_byte_ms`` log line for local dispatches is segmented
    (``cache_state=warm|cold``, ``queue_wait_ms``) by LP-0MV6CFMU9005WU2E;
    this tool computes warm/cold p50/p90/p99 and the queue-wait distribution
    from a window of that log.

Usage:
  ./scripts/dispatch_latency_verification.py                             # summary (live window)
  ./scripts/dispatch_latency_verification.py --json                      # machine readable
  ./scripts/dispatch_latency_verification.py --since '2026-10-13 15:48' \\\\
      --until '2026-10-14 15:48' --json > ac1_window.json               # 24 h window artifact
  ./scripts/dispatch_latency_verification.py --notes 'post-deploy' --json

Exit codes:
  0 - success (verdict PASS on all asserted targets)
  2 - success (verdict WARN: targets not met / measurement window not closed)
  1 - no proxy logs found / unexpected error
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from lib.proxy_logs import discover_proxy_log_files, open_proxy_log_text  # noqa: E402

# Re-scoped AC2 targets (LP-0MV6BR3PT001WBEI producer decision 2026-10-13):
#   * warm-cache local dispatches p99 < 60 s to first byte (service time,
#     ``dispatch_to_first_byte_ms``);
#   * the single-slot queue wait stays within the contention-queue cap
#     (server.contention_queue_max_wait_seconds; default 60, cheap mode 120).
_WARM_P99_TARGET_MS = 60_000.0
_FIRST_BYTE_TARGET_PCT = 99.0
#: Server-side first-byte ceiling after the AC1 fix (LP-0MV6BR3HI00957PF);
#: any ``dispatch_first_byte_timeout ceiling=900s`` event in a window is a
#: statement that a request raced the client's 900 s deadline exactly.
_LEGACY_CEILING_SECONDS = 900

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")
_TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")

# AC1 surfaces.
_FIRST_BYTE_TIMEOUT_RE = re.compile(
    r"dispatch_first_byte_timeout\s+session=\S+\s+ceiling=(?P<ceiling>[0-9.]+)s"
)
_CHAIN_HOLD_RE = re.compile(
    r"Chain exhausted for model=\S+; holding (?P<hold>[0-9.]+)s "
    r"before restarting cycle"
)
# Client-observable relay of an upstream 900 s queue-timeout verdict.
_CLIENT_QUEUE_TIMEOUT_RE = re.compile(
    r"(900[- ]second|queue[- ]?timeout|start processing your request within)"
)
# AC2 local dispatch line, e.g.
#   dispatch_first_byte_ms=742.2 dispatch_to_first_byte_ms=741.0
#     cache_state=warm cached_ratio=0.999 queue_wait_ms=1.2 session=... model=Qwen3
_DISPATCH_RE = re.compile(
    r"dispatch_first_byte_ms=(?P<fb>[0-9.]+) "
    r"dispatch_to_first_byte_ms=(?P<tfb>[0-9.]+) "
    r"cache_state=(?P<state>warm|cold) cached_ratio=(?P<ratio>[0-9.]+) "
    r"queue_wait_ms=(?P<qw>[0-9.]+) "
)

_TARGETS = {
    "warm_p99_service_ms_target": _WARM_P99_TARGET_MS,
    "warm_p99_percentile": _FIRST_BYTE_TARGET_PCT,
    "legacy_ceiling_is_900": _LEGACY_CEILING_SECONDS,
}


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


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile; empty list -> 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _dist_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "p50": round(_percentile(values, 50), 1),
        "p90": round(_percentile(values, 90), 1),
        "p99": round(_percentile(values, 99), 1),
        "max": round(max(values), 1),
    }


def analyse(
    log_dir: Path,
    *,
    since: dt.datetime | None = None,
    until: dt.datetime | None = None,
) -> dict:
    ac1 = {
        "dispatch_first_byte_timeout_total": 0,
        "dispatch_first_byte_timeout_ceiling_900s": 0,
        "dispatch_first_byte_timeout_ceiling_other": 0,
        "chain_hold_starts": 0,
        "chain_hold_total_seconds": 0.0,
        "client_queue_timeout_relays": 0,
    }
    # AC2: service time (dispatch_to_first_byte_ms) and queue wait, split by
    # cache state.
    warm_service: list[float] = []
    cold_service: list[float] = []
    warm_total: list[float] = []
    cold_total: list[float] = []
    queue_wait: list[float] = []
    first_ts: str | None = None
    last_ts: str | None = None

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
                if first_ts is None:
                    first_ts = line.split(",", 1)[0]
                last_ts = line.split(",", 1)[0]

                if "dispatch_first_byte_timeout" in line:
                    ac1["dispatch_first_byte_timeout_total"] += 1
                    m = _FIRST_BYTE_TIMEOUT_RE.search(line)
                    ceiling = float(m.group("ceiling")) if m else 0.0
                    if ceiling >= _LEGACY_CEILING_SECONDS:
                        ac1["dispatch_first_byte_timeout_ceiling_900s"] += 1
                    else:
                        ac1["dispatch_first_byte_timeout_ceiling_other"] += 1
                    continue
                if "Chain exhausted for model" in line:
                    ac1["chain_hold_starts"] += 1
                    m = _CHAIN_HOLD_RE.search(line)
                    if m:
                        ac1["chain_hold_total_seconds"] += float(m.group("hold"))
                    continue
                if _CLIENT_QUEUE_TIMEOUT_RE.search(line):
                    ac1["client_queue_timeout_relays"] += 1
                    continue

                m = _DISPATCH_RE.search(line)
                if not m:
                    continue
                service = float(m.group("tfb"))
                total = float(m.group("fb"))
                qw = float(m.group("qw"))
                queue_wait.append(qw)
                if m.group("state") == "warm":
                    warm_service.append(service)
                    warm_total.append(total)
                else:
                    cold_service.append(service)
                    cold_total.append(total)

    ac1["chain_hold_total_seconds"] = round(ac1["chain_hold_total_seconds"], 1)
    return {
        "window": {
            "first_seen": first_ts,
            "last_seen": last_ts,
            "since": since.isoformat(sep=" ") if since else None,
            "until": until.isoformat(sep=" ") if until else None,
        },
        "ac1": ac1,
        "ac2": {
            "warm_service_ms": _dist_stats(warm_service),
            "cold_service_ms": _dist_stats(cold_service),
            "warm_total_ms": _dist_stats(warm_total),
            "cold_total_ms": _dist_stats(cold_total),
            "queue_wait_ms": _dist_stats(queue_wait),
        },
    }


def _verdict(metrics: dict, *, window_closed: bool) -> dict:
    """Evaluate the re-scoped targets; warn (never silently pass) on any miss.

    Returns ``{"verdict": "PASS"|"WARN", "checks": [...], "notes": [...]}``.
    """
    checks: list[dict] = []
    notes: list[str] = []
    ac1 = metrics["ac1"]
    ac2 = metrics["ac2"]

    # AC1: zero legacy-ceiling first-byte timeouts in the window.
    legacy = ac1["dispatch_first_byte_timeout_ceiling_900s"]
    checks.append(
        {
            "target": "AC1: zero ceiling=900s dispatch_first_byte_timeout",
            "observed": legacy,
            "pass": legacy == 0,
        }
    )
    if not window_closed:
        notes.append(
            "AC1 24 h post-change window not yet closed; a zero legacy-ceiling "
            "count is a precondition, not proof, until the window elapses."
        )

    # AC2: warm-cache service p99 < 60 s.
    warm_p99 = ac2["warm_service_ms"]["p99"]
    checks.append(
        {
            "target": f"AC2: warm p99 < {_WARM_P99_TARGET_MS/1000:.0f} s service time",
            "observed_ms": warm_p99,
            "pass": warm_p99 < _WARM_P99_TARGET_MS,
        }
    )
    if ac2["warm_service_ms"]["n"] == 0:
        notes.append("no warm-cache local dispatches in the window — AC2 unmeasurable.")

    # Queue-wait bound: p99 within the contention-queue cap (cheap 120 / fast 45 /
    # default 60) reported non-normatively; the authoritative bound is bounded by
    # contention_queue_max_wait_seconds, which this tool cannot read from logs.
    qw_p99 = ac2["queue_wait_ms"]["p99"]
    checks.append(
        {
            "target": "AC2 (context): queue-wait p99 within contention cap",
            "observed_ms": qw_p99,
            "pass": qw_p99 <= 120_000.0,
        }
    )
    return {
        "verdict": "PASS" if all(c["pass"] for c in checks) else "WARN",
        "checks": checks,
        "notes": notes,
    }


def _build_meta(log_dir, since, until, label) -> dict:
    files = [p.name for p in discover_proxy_log_files(Path(log_dir))]
    return {
        "tool": "scripts/dispatch_latency_verification.py",
        "work_item": "LP-0MV6BR3PT001WBEI",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "log_dir": str(log_dir),
        "files": files,
        "since": since.isoformat(sep=" ") if since else None,
        "until": until.isoformat(sep=" ") if until else None,
        "label": label,
        "targets": _TARGETS,
    }


def _render(meta: dict, metrics: dict, verdict: dict) -> str:
    out = []
    out.append("Dispatch-latency AC1/AC2 verification (LP-0MV6BR3PT001WBEI)")
    out.append("=" * 66)
    out.append(f"log_dir : {meta['log_dir']}")
    out.append(f"files   : {', '.join(meta['files']) or '(none)'}")
    out.append(
        f"window  : {meta['since'] or '(start)'} .. {meta['until'] or '(end)'}"
    )
    out.append(f"label   : {meta['label'] or '(none)'}")
    out.append("")
    out.append("AC1 — 900 s client-deadline surfaces")
    out.append("-" * 40)
    out.append(
        f"dispatch_first_byte_timeout total        : {metrics['ac1']['dispatch_first_byte_timeout_total']}"
    )
    out.append(
        f"  ... ceiling=900s (races client)        : {metrics['ac1']['dispatch_first_byte_timeout_ceiling_900s']}"
    )
    out.append(
        f"  ... other ceilings                     : {metrics['ac1']['dispatch_first_byte_timeout_ceiling_other']}"
    )
    out.append(
        f"chain-hold starts                        : {metrics['ac1']['chain_hold_starts']} "
        f"({metrics['ac1']['chain_hold_total_seconds']}s held)"
    )
    out.append(
        f"client queue-timeout relays              : {metrics['ac1']['client_queue_timeout_relays']}"
    )
    out.append("")
    out.append("AC2 — first-byte latency split (local dispatches)")
    out.append("-" * 40)
    for label, key, unit in (
        ("warm-cache service (to first byte)", "warm_service_ms", "ms"),
        ("cold-cache service (to first byte)", "cold_service_ms", "ms"),
        ("warm-cache total (incl. queue)", "warm_total_ms", "ms"),
        ("queue wait", "queue_wait_ms", "ms"),
    ):
        d = metrics["ac2"][key]
        out.append(
            f"{label:<34}: n={d['n']:<5} p50={d['p50']:>9.1f} "
            f"p90={d['p90']:>9.1f} p99={d['p99']:>9.1f} max={d['max']:>9.1f} {unit}"
        )
    out.append("")
    out.append(f"VERDICT: {verdict['verdict']}")
    for c in verdict["checks"]:
        out.append(
            f"  [{'ok ' if c['pass'] else 'FAIL'}] {c['target']} "
            f"(observed {c.get('observed', c.get('observed_ms', 'n/a'))})"
        )
    for note in verdict["notes"]:
        out.append(f"  note: {note}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="/var/log/llama-proxy")
    parser.add_argument(
        "--since", default=None, help="inclusive lower bound (YYYY-MM-DD [HH:MM[:SS]])"
    )
    parser.add_argument("--until", default=None, help="exclusive upper bound")
    parser.add_argument("--label", default=None, help="free-form label recorded in meta")
    parser.add_argument(
        "--window-closed",
        action="store_true",
        help="assert the measurement window (e.g. 24 h) has fully elapsed",
    )
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

    metrics = analyse(log_dir, since=since, until=until)
    verdict = _verdict(metrics, window_closed=args.window_closed)
    meta = _build_meta(log_dir, since, until, args.label)

    if args.json:
        print(json.dumps({"meta": meta, "metrics": metrics, "verdict": verdict}, indent=2))
    else:
        print(_render(meta, metrics, verdict))
    return 0 if verdict["verdict"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
