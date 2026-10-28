#!/usr/bin/env python3
"""Per-arm capture and pre-registered gate comparison for the slot-geometry A/B.

Protocol: docs/dev/slot-geometry-ab.md (parent LP-0MVMHNUXW0061FGN, child
LP-0MVNCMDW50043RSP). This is the measurement side: it captures one arm's
metrics for one window into a JSON archive, and compares two captured arms
through the numeric gate that is pre-registered in ``docs/dev/slot-geometry-ab.md``
BEFORE any arm is measured (so the decision rule cannot be moved afterwards).

Metrics come from two sources, both documented in the protocol:

  * the canonical proxy-usage-analysis pipeline (the standard report tooling,
    so both arms are measured identically) — local busy %, turns served
    locally, queue wait / ``fallback_after_queue`` counts, time to first
    token, prefill/decode phase slot-seconds, and error totals;
  * this script's own light log scan for the granularity the standard report
    does not expose per-status/per-reason: ``routing_skip_local`` divert
    counts per reason (with the legacy ``warm_cache_bypass`` alias folded
    into ``context_too_large`` exactly as the analysis pipeline does), local
    ``dispatch_to_first_byte_ms`` samples (the prefill latency the gate is
    written against), and HTTP 400/503 status-line counts.

Usage:
  slot_geometry_ab_report.py capture --label arm-a --arm A \
      --start '2026-10-28 22:00:00' --end '2026-10-29 06:00:00'
  slot_geometry_ab_report.py capture --label arm-a --arm A \
      --start ... --end ... --analyzer-json /path/summary.json   # offline
  slot_geometry_ab_report.py compare --baseline <arm-a>/arm.json \
      --candidate <arm-b>/arm.json

Exit codes:
  0 - capture succeeded / comparison reached a decision
  2 - comparison INCONCLUSIVE: the evidence gate (AC0, >=60 % busy night) is
      not met, so the criteria do not decide the geometry
  1 - no log files, analyzer failure, malformed input, comparison error
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from slot_geometry_arm import ARMS, probe_llama_geometry  # noqa: E402

# Repository root (this file lives in scripts/).
REPO_ROOT = SCRIPT_DIR.parent

# The standard report tooling — the same pipeline the daily report uses, so
# both arms are measured by exactly the same code (parent AC1).
ANALYZER_SCRIPT = (
    REPO_ROOT
    / ".pi"
    / "skills"
    / "proxy-usage-analysis"
    / "scripts"
    / "analyze_proxy_usage.py"
)

DEFAULT_LOG_DIR = Path("/var/log/llama-proxy")
DEFAULT_OUT_ROOT = REPO_ROOT / "benchmark-results" / "slot-geometry"
DEFAULT_CONFIG = REPO_ROOT / "proxy" / "config.yaml"

# Pre-registered decision gate (written into docs/dev/slot-geometry-ab.md and
# recorded on the parent BEFORE any arm window is measured).
GATE_MIN_BUSY_PCT = 60.0          # AC0: evidence gate, not a criterion
GATE_DIVERT_MIN_REDUCTION = 0.50  # (b) context_ceiling diverts must fall >= 50 %
GATE_MAX_TTFT_WORSENING = 0.30    # (c) p90 first-byte must not worsen by > 30 %
GATE_MAX_BUSY_REGRESSION_POINTS = 10.0  # (d) local busy % must not regress > 10 pts

# Reasons counted in the (b) criterion — the cap/bypass pair the parent's
# evidence singled out (2,064 ``context_too_large`` skips in 24 h).
CEILING_DIVERT_REASONS = ("context_too_large", "large_context_bypass")
#: Legacy reason alias (pre-LP-0MSF8XDG7000PERM) folded into the current name,
#: mirrored from the analysis pipeline's ``_normalize_reason``.
LEGACY_REASON_ALIASES = {"warm_cache_bypass": "context_too_large"}

_LINE_TS_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})")
_SKIP_REASON_RE = re.compile(r"\brouting_skip_local\b .*?\breason=([A-Za-z0-9_]+)")
_HTTP_STATUS_RE = re.compile(r"(?:\bstatus=|reason=HTTP\s+|\bHTTP\s+)([45]\d{2})\b")
_D2FB_RE = re.compile(r"\bdispatch_to_first_byte_ms=([0-9]+(?:\.[0-9]+)?)")


class AnalysisError(RuntimeError):
    """The underlying proxy-usage-analysis pipeline failed."""


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile; empty list -> 0.0 (mirrors the project
    convention in scripts/dispatch_latency_verification.py)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _parse_iso(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.fromisoformat(value)


def _line_ts(line: str) -> datetime | None:
    m = _LINE_TS_RE.match(line)
    if not m:
        return None
    try:
        return datetime(*(int(part) for part in m.groups()))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Geometry from the live config (derived, never configured directly:
# per-slot ctx = ctx_size // slots, clamped by the 4096-token output headroom
# mirroring proxy/provider.py:effective_per_slot_threshold).
# ---------------------------------------------------------------------------

_OUTPUT_HEADROOM = 4096


def _config_value(config_text: str, key: str) -> str | None:
    m = re.search(rf"^\s*{re.escape(key)}:\s*(\S+)", config_text, re.MULTILINE)
    return m.group(1) if m else None


def derive_geometry(config_path: Path | None) -> dict:
    """Resolve the routing geometry a capture ran under, from ``config.yaml``.

    ``per_slot_route_clamp`` mirrors ``provider.effective_per_slot_threshold``
    (``ctx_size // slots - 4096``); ``effective_warm_threshold`` mirrors the
    three-term model ``warm = min(physical clamp, economic ceiling)`` where
    the economic ceiling is ``round(ratio x ctx_size)`` capped by the
    absolute warm threshold (LP-0MU466L1X003RTHN). The reporter records the
    real values instead of assuming the arm definition held.
    """
    source = config_path or DEFAULT_CONFIG
    text = source.read_text(encoding="utf-8", errors="replace")
    ctx = int(_config_value(text, "local_model_ctx_size") or 0)
    slots = int(_config_value(text, "session_slot_pool_size") or 0)
    warm = int(_config_value(text, "local_large_context_warm_cache_threshold") or 0)
    cold = int(_config_value(text, "local_large_context_cold_cache_threshold") or 0)
    ratio = float(
        _config_value(text, "local_large_context_economic_ceiling_ratio") or 0.0
    )
    per_slot = ctx // slots if slots else 0
    clamp = per_slot - _OUTPUT_HEADROOM if per_slot > _OUTPUT_HEADROOM else 0
    economic = min(warm, round(ratio * ctx)) if ratio > 0 else warm
    effective_warm = min(clamp, economic) if clamp > 0 else economic
    return {
        "ctx_size": ctx,
        "slots": slots,
        "per_slot_ctx": per_slot,
        "per_slot_route_clamp": clamp,
        "warm_threshold": warm,
        "economic_ceiling_ratio": ratio,
        "cold_threshold": cold,
        "effective_warm_threshold": effective_warm,
        "band_width": max(effective_warm - cold, 0),
        "config_source": str(source),
    }


# ---------------------------------------------------------------------------
# Log scan (granularity the standard report does not expose)
# ---------------------------------------------------------------------------


def scan_log_window(log_dir: Path, start: datetime, end: datetime) -> dict:
    """One streaming pass over every proxy log file for the window.

    Returns divert counts per ``routing_skip_local`` reason, local
    ``dispatch_to_first_byte_ms`` samples, HTTP status-line counts, and scan
    accounting (files, lines scanned, unparseable lines). Malformed or
    out-of-window lines are never fatal; the counts are the contract.
    """
    sys.path.insert(0, str(SCRIPT_DIR / "lib"))
    from proxy_logs import discover_proxy_log_files, open_proxy_log_text

    diverts: dict[str, int] = {}
    first_byte: list[float] = []
    http: dict[str, int] = {"400": 0, "503": 0, "other": 0}
    files = []
    lines_scanned = 0
    unparsed = 0

    for path in discover_proxy_log_files(log_dir):
        files.append(path.name)
        with open_proxy_log_text(path) as fh:
            for line in fh:
                ts = _line_ts(line)
                if ts is None:
                    unparsed += 1
                    continue
                if ts < start or ts > end:
                    continue
                lines_scanned += 1
                skip = re.search(r"routing_skip_local", line)
                if skip:
                    reason_match = _SKIP_REASON_RE.search(line)
                    reason = reason_match.group(1) if reason_match else "(unknown)"
                    if reason == "warm_cache_bypass":
                        reason = LEGACY_REASON_ALIASES[reason]
                    diverts[reason] = diverts.get(reason, 0) + 1
                    continue
                for code_match in _HTTP_STATUS_RE.finditer(line):
                    code = code_match.group(1)
                    bucket = code if code in http else "other"
                    http[bucket] += 1
                for ms_match in _D2FB_RE.finditer(line):
                    try:
                        first_byte.append(float(ms_match.group(1)))
                    except ValueError:
                        continue

    ceiling = sum(diverts.get(r, 0) for r in CEILING_DIVERT_REASONS)
    return {
        "files": files,
        "scan": {
            "lines_scanned": lines_scanned,
            "unparsed_lines": unparsed,
        },
        "diverts_by_reason": diverts,
        "divert_total": sum(diverts.values()),
        "context_ceiling_diverts": ceiling,
        "local_dispatch_first_byte": {
            "samples": len(first_byte),
            "p50_ms": round(_percentile(first_byte, 50.0), 2),
            "p90_ms": round(_percentile(first_byte, 90.0), 2),
        },
        "http_errors": http,
    }


# ---------------------------------------------------------------------------
# The standard analysis (canonical tooling, both arms identical)
# ---------------------------------------------------------------------------


def run_analyzer(
    log_dir: Path, start: datetime, end: datetime, output_dir: Path
) -> dict:
    """Run ``analyze_proxy_usage.py --json`` for the window.

    The pipeline's own artifacts (``report.md``, per-session CSVs) land in
    ``output_dir`` so each arm's archive carries the standard report a
    reviewer reads; the JSON summary is returned for metric mapping.
    """
    if not ANALYZER_SCRIPT.is_file():
        raise AnalysisError(f"analyzer script not found: {ANALYZER_SCRIPT}")
    cmd = [
        "python3",
        str(ANALYZER_SCRIPT),
        "--log-dir",
        str(log_dir),
        "--start",
        start.strftime("%Y-%m-%d %H:%M:%S"),
        "--end",
        end.strftime("%Y-%m-%d %H:%M:%S"),
        "--output-dir",
        str(output_dir),
        "--json",
        "--quiet",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if proc.returncode != 0:
        raise AnalysisError(
            f"proxy-usage-analysis failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[-600:]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AnalysisError(
            f"proxy-usage-analysis emitted invalid JSON: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Metric mapping (parent AC3 names every metric this emits)
# ---------------------------------------------------------------------------


def build_capture(
    label: str,
    arm: str | None,
    start: datetime,
    end: datetime,
    analyzer: dict,
    scan: dict,
    geometry: dict,
    live: dict | None,
) -> dict:
    """Assemble the per-arm archive from the analysis summary + log scan."""
    busy = analyzer.get("local_busy") or {}
    prefill_vs_decode = busy.get("prefill_vs_decode") or {}
    t = analyzer.get("ttft") or {}
    total_ttft = t.get("total") or {}
    errors = analyzer.get("errors_by_type") or {}
    return {
        "schema_version": 1,
        "label": label,
        "arm": arm,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "window": {
            "start": start.strftime("%Y-%m-%d %H:%M:%S"),
            "end": end.strftime("%Y-%m-%d %H:%M:%S"),
            "hours": round((end - start).total_seconds() / 3600.0, 2),
        },
        "geometry": geometry,
        "live_geometry_probe": live,
        "metric_sources": {
            "local_busy_pct": "analyzer.local_busy.busy_pct",
            "turns_served_locally": "analyzer.local_requests",
            "queue_wait": "analyzer.contention_*",
            "ttft_all_streams": "analyzer.ttft.total",
            "slot_seconds": "analyzer.local_busy.prefill_vs_decode",
            "diverts_by_reason": "log scan routing_skip_local",
            "local_dispatch_first_byte": "log scan dispatch_to_first_byte_ms",
            "http_status_lines": "log scan status= / reason=HTTP / HTTP ",
        },
        "metrics": {
            "local_busy_pct": float(busy.get("busy_pct") or 0.0),
            "busy_seconds": float(busy.get("busy_seconds") or 0.0),
            "prefill_busy_seconds": float(busy.get("prefill_busy_seconds") or 0.0),
            "decode_busy_seconds": float(busy.get("decode_busy_seconds") or 0.0),
            "peak_concurrency": int(busy.get("peak_concurrency") or 0),
            "avg_concurrency": float(busy.get("avg_concurrency") or 0.0),
            "turns_served_locally": int(analyzer.get("local_requests") or 0),
            "remote_requests": int(analyzer.get("remote_requests") or 0),
            "sessions": int(analyzer.get("sessions") or 0),
            "diverts_by_reason": scan["diverts_by_reason"],
            "divert_total": scan["divert_total"],
            "context_ceiling_diverts": scan["context_ceiling_diverts"],
            "queue_waits": int(analyzer.get("contention_dispatch") or 0),
            "fallback_after_queue": int(
                analyzer.get("contention_fallback_after_queue") or 0
            ),
            "queue_wait_seconds": float(
                analyzer.get("contention_queued_duration_seconds") or 0.0
            ),
            "ttft_all_streams": {
                "samples": int(total_ttft.get("samples") or 0),
                "p50_ms": float(total_ttft.get("median") or 0.0),
                "p90_ms": float(total_ttft.get("p90") or 0.0),
            },
            "local_dispatch_first_byte": scan["local_dispatch_first_byte"],
            "prefill_slot_seconds": float(
                (prefill_vs_decode.get("prefill") or {}).get("slot_seconds") or 0.0
            ),
            "decode_slot_seconds": float(
                (prefill_vs_decode.get("decode") or {}).get("slot_seconds") or 0.0
            ),
            "errors": {
                "total": int(analyzer.get("errors") or 0),
                "slot_save_failed": int(errors.get("slot_save_error") or 0),
                "upstream_503": int(analyzer.get("upstream_status_503_errors") or 0),
                "http_400_lines": scan["http_errors"].get("400", 0),
                "http_503_lines": scan["http_errors"].get("503", 0),
                "by_type": errors,
            },
            "log_files": scan["files"],
            "scan": scan["scan"],
        },
    }


# ---------------------------------------------------------------------------
# Comparison through the pre-registered gate
# ---------------------------------------------------------------------------


def _criterion(
    name: str, baseline: float, candidate: float, passed: bool, detail: str
) -> dict:
    return {
        "name": name,
        "baseline": baseline,
        "candidate": candidate,
        "met": passed,
        "detail": detail,
    }


def compare_reports(baseline: dict, candidate: dict) -> dict:
    """Apply the pre-registered decision gate to two captured arms.

    Criteria (recorded in docs/dev/slot-geometry-ab.md before measurement):
      (a) turns served locally must not decrease;
      (b) ``context_too_large`` + ``large_context_bypass`` diverts must fall
          by at least 50 %;
      (c) p90 dispatch→first byte must not worsen by more than 30 % (falls
          back to the stream TTFT p90 when no local dispatch samples exist);
      (d) local busy % must not regress by more than 10 points.
    Decision precedence: any failed criterion means KEEP 2 slots (a measured
    regression is evidence of harm whether or not the window was contended);
    otherwise, the candidate wins only when the evidence gate is met — both
    arms measured on a night with busy >= 60 % (AC0). An uncontended window
    whose criteria all pass is INCONCLUSIVE, never an adoption.
    """
    if baseline.get("arm") and baseline.get("arm") == candidate.get("arm") and (
        baseline.get("label") == candidate.get("label")
    ):
        raise ValueError(
            f"baseline and candidate are the same arm ({baseline.get('arm')})"
        )
    b_m = baseline["metrics"]
    c_m = candidate["metrics"]
    b_g = baseline.get("geometry") or {}
    c_g = candidate.get("geometry") or {}

    criteria = []

    turns_b, turns_c = b_m["turns_served_locally"], c_m["turns_served_locally"]
    criteria.append(
        _criterion(
            "turns_served_locally",
            turns_b,
            turns_c,
            turns_c >= turns_b,
            f"{turns_c - turns_b:+d} local turns (must not decrease)",
        )
    )

    div_b = b_m["context_ceiling_diverts"]
    div_c = c_m["context_ceiling_diverts"]
    reduction = (div_b - div_c) / div_b if div_b else 1.0
    criteria.append(
        _criterion(
            "context_ceiling_diverts",
            div_b,
            div_c,
            div_c <= div_b * (1.0 - GATE_DIVERT_MIN_REDUCTION),
            f"{reduction * 100:.1f}% reduction (needs >= "
            f"{GATE_DIVERT_MIN_REDUCTION * 100:.0f}%)",
        )
        | {"reduction_pct": round(reduction * 100, 2)}
    )

    b_fb = b_m["local_dispatch_first_byte"]
    c_fb = c_m["local_dispatch_first_byte"]
    if b_fb["samples"] and c_fb["samples"]:
        source = "local_dispatch_first_byte"
        b_p90, c_p90 = b_fb["p90_ms"], c_fb["p90_ms"]
    else:
        source = "ttft_all_streams"
        b_p90, c_p90 = (
            b_m["ttft_all_streams"]["p90_ms"],
            c_m["ttft_all_streams"]["p90_ms"],
        )
    worsening = (c_p90 - b_p90) / b_p90 if b_p90 else 0.0
    criteria.append(
        _criterion(
            "first_byte_p90",
            b_p90,
            c_p90,
            c_p90 <= b_p90 * (1.0 + GATE_MAX_TTFT_WORSENING),
            f"{worsening * 100:+.1f}% p90 (must not worsen by > "
            f"{GATE_MAX_TTFT_WORSENING * 100:.0f}%) [{source}]",
        )
        | {"source": source}
    )

    b_busy, c_busy = b_m["local_busy_pct"], c_m["local_busy_pct"]
    criteria.append(
        _criterion(
            "local_busy_pct",
            b_busy,
            c_busy,
            c_busy >= b_busy - GATE_MAX_BUSY_REGRESSION_POINTS,
            f"{c_busy - b_busy:+.1f} pts busy (must not regress by > "
            f"{GATE_MAX_BUSY_REGRESSION_POINTS:.0f} pts)",
        )
    )

    gate_ok = c_busy >= GATE_MIN_BUSY_PCT and b_busy >= GATE_MIN_BUSY_PCT
    all_met = all(c["met"] for c in criteria)
    if not all_met:
        # A measured regression is evidence of harm regardless of contention.
        decision = "keep_2_slots"
    elif gate_ok:
        decision = "adopt_1x262k"
    else:
        # An uncontended window cannot demonstrate a win (AC0).
        decision = "inconclusive"

    return {
        "baseline": {"label": baseline.get("label"), "arm": baseline.get("arm")},
        "candidate": {"label": candidate.get("label"), "arm": candidate.get("arm")},
        "geometry": {
            "baseline": {"slots": b_g.get("slots"), "per_slot_ctx": b_g.get("per_slot_ctx")},
            "candidate": {"slots": c_g.get("slots"), "per_slot_ctx": c_g.get("per_slot_ctx")},
        },
        "evidence_gate": {
            "required_busy_pct": GATE_MIN_BUSY_PCT,
            "baseline_busy_pct": b_busy,
            "candidate_busy_pct": c_busy,
            "met": gate_ok,
        },
        "criteria": criteria,
        "decision": decision,
    }


def _render_capture(data: dict) -> str:
    m = data["metrics"]
    g = data["geometry"]
    lines = [
        f"Slot-geometry capture: {data['label']} (arm {data['arm'] or 'n/a'})",
        f"  window      : {data['window']['start']} -> {data['window']['end']}"
        f" ({data['window']['hours']} h)",
        f"  geometry    : {g['slots']} slot(s) x {g['per_slot_ctx']} ctx"
        f" (clamp {g['per_slot_route_clamp']}, warm {g['effective_warm_threshold']},"
        f" cold {g['cold_threshold']})",
        f"  local busy  : {m['local_busy_pct']:.1f}%"
        f" (peak {m['peak_concurrency']}, turns {m['turns_served_locally']})",
        f"  diverts     : {m['divert_total']} total"
        f" ({m['context_ceiling_diverts']} context-ceiling: {m['diverts_by_reason']})",
        f"  queue       : {m['queue_waits']} waits,"
        f" {m['fallback_after_queue']} fallback-after-queue,"
        f" {m['queue_wait_seconds']} s",
        f"  ttft        : local dispatch->first byte p50 {m['local_dispatch_first_byte']['p50_ms']} ms"
        f" / p90 {m['local_dispatch_first_byte']['p90_ms']} ms"
        f" ({m['local_dispatch_first_byte']['samples']} samples)",
        f"  slot-time   : prefill {m['prefill_slot_seconds']} s,"
        f" decode {m['decode_slot_seconds']} s",
        f"  errors      : {m['errors']['total']} total"
        f" (slot_save {m['errors']['slot_save_failed']},"
        f" HTTP 400 {m['errors']['http_400_lines']},"
        f" HTTP 503 {m['errors']['http_503_lines']})",
    ]
    return "\n".join(lines)


def _render_compare(data: dict) -> str:
    lines = [
        f"Slot-geometry comparison: baseline {data['baseline']['label']}"
        f" (arm {data['baseline']['arm']}) vs candidate {data['candidate']['label']}"
        f" (arm {data['candidate']['arm']})",
        f"  geometry: {data['geometry']['baseline']['slots']}x"
        f"{data['geometry']['baseline']['per_slot_ctx']} vs"
        f" {data['geometry']['candidate']['slots']}x"
        f"{data['geometry']['candidate']['per_slot_ctx']}",
        f"  evidence gate (>= {data['evidence_gate']['required_busy_pct']:.0f}% busy):"
        f" {'met' if data['evidence_gate']['met'] else 'NOT MET'}",
    ]
    for c in data["criteria"]:
        flag = "PASS" if c["met"] else "FAIL"
        lines.append(
            f"    [{flag}] {c['name']}: {c['baseline']} -> {c['candidate']}"
            f" ({c['detail']})"
        )
    lines.append(f"  decision: {data['decision']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    cap = sub.add_parser("capture", help="capture one arm's metrics for a window")
    cap.add_argument("--label", required=True)
    cap.add_argument("--arm", choices=sorted(ARMS))
    cap.add_argument("--start", required=True, help="window start YYYY-MM-DD HH:MM:SS")
    cap.add_argument("--end", required=True, help="window end YYYY-MM-DD HH:MM:SS")
    cap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    cap.add_argument("--out-dir", type=Path)
    cap.add_argument("--config", type=Path)
    cap.add_argument(
        "--analyzer-json",
        type=Path,
        help="pre-computed analyze_proxy_usage.py --json summary (offline)",
    )
    cap.add_argument("--llama-url", default="http://127.0.0.1:8080")
    cap.add_argument("--model", default="Qwen3")
    cap.add_argument("--no-live-probe", action="store_true")
    cap.add_argument("--json", action="store_true")

    cmp = sub.add_parser("compare", help="apply the pre-registered gate to two arms")
    cmp.add_argument("--baseline", type=Path, required=True)
    cmp.add_argument("--candidate", type=Path, required=True)
    cmp.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.mode == "capture":
        start, end = _parse_iso(args.start), _parse_iso(args.end)
        if start >= end:
            print("error: --start must be before --end", file=sys.stderr)
            return 1
        log_dir = args.log_dir
        out_dir = (args.out_dir or DEFAULT_OUT_ROOT / args.label).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            scan = scan_log_window(log_dir, start, end)
        except OSError as exc:
            print(f"error: cannot scan {log_dir}: {exc}", file=sys.stderr)
            return 1
        if not scan["files"]:
            print(
                f"error: no proxy log files found under {log_dir} "
                f"for the requested window",
                file=sys.stderr,
            )
            return 1
        analyzer_dir = out_dir / "analyzer"
        analyzer_dir.mkdir(parents=True, exist_ok=True)
        if args.analyzer_json:
            try:
                analyzer = json.loads(args.analyzer_json.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                print(f"error: cannot read --analyzer-json: {exc}", file=sys.stderr)
                return 1
        else:
            try:
                analyzer = run_analyzer(log_dir, start, end, analyzer_dir)
            except AnalysisError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
        try:
            geometry = derive_geometry(args.config)
        except (OSError, ValueError) as exc:
            print(f"error: cannot read geometry config: {exc}", file=sys.stderr)
            return 1
        live = None
        if not args.no_live_probe:
            try:
                live = probe_llama_geometry(args.llama_url, model=args.model)
                live = {**live, "ok": True}
            except Exception as exc:  # fail-soft: offline re-analysis is valid
                live = {"ok": False, "error": str(exc)[:300]}
        data = build_capture(
            args.label, args.arm, start, end, analyzer, scan, geometry, live
        )
        (out_dir / "analyzer" / "summary.json").write_text(
            json.dumps(analyzer, indent=2, sort_keys=True) + "\n"
        )
        (out_dir / "arm.json").write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n"
        )
        if args.json:
            print(json.dumps(data, indent=2, sort_keys=True))
        else:
            print(_render_capture(data))
            print(f"\narchived: {out_dir / 'arm.json'}")
        return 0

    if args.mode == "compare":
        for path in (args.baseline, args.candidate):
            if not path.is_file():
                print(f"error: comparison input not found: {path}", file=sys.stderr)
                return 1
        try:
            baseline = json.loads(args.baseline.read_text())
            candidate = json.loads(args.candidate.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: cannot read comparison input: {exc}", file=sys.stderr)
            return 1
        try:
            data = compare_reports(baseline, candidate)
        except (KeyError, ValueError, TypeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(data, indent=2, sort_keys=True))
        else:
            print(_render_compare(data))
        return 0 if data["decision"] != "inconclusive" else 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
