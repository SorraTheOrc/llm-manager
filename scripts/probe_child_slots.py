#!/usr/bin/env python3
"""Quantify the child-port ``GET /slots`` cost separately from proxy serialisation.

Why this exists
---------------
LP-0MVVXGI490089OX5 has to name the dominant contributor to ``/slots``
refreshes that exceed the handler's freshness budget. The proxy's
``status_request`` log line already carries per-phase timings
(``slots_counts_ms``, ``slots_detail_ms``), but both are **clamped by the
handler's own budgets** (a 2 s counts budget and a 1 s ``STATUS_QUERY_TIMEOUT``
detail budget), so a window-only analysis can prove "the phase exceeded its
budget" and never "how slow the backend actually was". Those are different
defects with different fixes: proxy-side fan-out/serialisation is fixed in the
handler, backend blocking is not.

This probe closes that gap with a **bounded, read-only** live measurement: it
times ``GET /slots?model=…`` **directly against the resolved llama-server child
port** while the GPU is busy, and pairs the samples with the proxy's own
``status_request`` facts taken immediately before and after, so backend cost and
proxy-side serialisation are quantified from the same moment.

Nothing here mutates state: plain HTTP GETs plus tail reads of two log files.

Child-port resolution
---------------------
The router ``llama-server`` (port 8080) spawns model children on dynamic ports
(``--port 0``). The rotation-proof signal is the per-request router log line
``proxying request to model Qwen3 on port 58113``; the startup line
``spawning server instance with name=Qwen3 on port 58113`` is only a fallback
because it disappears at the next logrotate. This mirrors the proxy's own
resolver (``proxy/proxy/router_helpers.py``, ``_PROXY_REQUEST_RE``) so the probe
measures the same endpoint the status handler targets.

Usage
-----
::

  # 12 timed child-port /slots calls while the GPU is busy (default)
  python3 scripts/probe_child_slots.py --model Qwen3 --json \\
      --write-artifact ~/proxy-usage-reports/post-f1-24h/ac2-child-port-probe

  # pin the port (skips log resolution) and shorten the probe
  python3 scripts/probe_child_slots.py --port 44409 --samples 10

Exit codes
----------
``0``
    The probe ran (whatever the latency distribution turned out to be).
``2``
    Usage error.
``3``
    No child port could be resolved for the requested model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.capacity_signals import (  # noqa: E402
    is_status_poll,
    parse_int,
    parse_phase_ms,
    percentile,
)

WORK_ITEM = "LP-0MVVXGI490089OX5"

DEFAULT_LLAMA_LOG = Path("/var/log/llama-proxy/llama-server.log")
DEFAULT_PROXY_LOG = Path("/var/log/llama-proxy/proxy.log")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_MODEL = "Qwen3"

#: llama.cpp's *router* writes one of these per proxied request, so the line
#: survives logrotate; the child port is the last one seen for the model.
_PROXY_REQUEST_RE = re.compile(r"proxying request to model (\S+) on port (\d+)")
#: Startup-only signal (lost at the next logrotate); fallback only.
_SPAWN_RE = re.compile(r"spawning server instance with name=(\S+) on port (\d+)")

#: Only the tail is scanned — the live signal is dense and the log is huge.
TAIL_SCAN_BYTES = 1024 * 1024

#: The handler's own detail budget (``STATUS_QUERY_TIMEOUT``), used as the
#: default "over budget" threshold so a sample can be compared with the log.
DEFAULT_BUDGET_MS = 1000


# ---------------------------------------------------------------------------
# Child-port resolution
# ---------------------------------------------------------------------------


def read_tail(path: Path, max_bytes: int = TAIL_SCAN_BYTES) -> str:
    """Return at most *max_bytes* from the end of *path* (``""`` when unreadable).

    A missing or unreadable log must never abort a probe: the caller can still
    pin ``--port`` or fail cleanly with exit code 3.
    """
    try:
        with Path(path).open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def resolve_child_port_from_text(text: str, model: str) -> int | None:
    """Return the child port for *model*: last ``proxying request`` line wins.

    Falls back to the startup ``spawning server instance`` line when the tail
    holds no proxied request for the model. Returns ``None`` when the model is
    not named at all — never a guessed port.
    """
    port: int | None = None
    for match in _PROXY_REQUEST_RE.finditer(text):
        if match.group(1) == model:
            port = int(match.group(2))
    if port is not None:
        return port
    for match in _SPAWN_RE.finditer(text):
        if match.group(1) == model:
            port = int(match.group(2))
    return port


def resolve_child_port(log_path: Path, model: str, max_bytes: int = TAIL_SCAN_BYTES) -> int | None:
    """Resolve *model*'s child port from the tail of ``llama-server.log``."""
    return resolve_child_port_from_text(read_tail(log_path, max_bytes), model)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One timed child-port ``/slots`` call."""

    index: int
    started_at: str
    latency_ms: float | None
    ok: bool
    timed_out: bool = False
    http_status: int | None = None
    processing_slots: int | None = None
    total_slots: int | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the JSON report."""
        return {
            "index": self.index,
            "started_at": self.started_at,
            "latency_ms": None if self.latency_ms is None else round(self.latency_ms, 1),
            "ok": self.ok,
            "timed_out": self.timed_out,
            "http_status": self.http_status,
            "processing_slots": self.processing_slots,
            "total_slots": self.total_slots,
            "error": self.error,
        }


def _slot_counts(payload: bytes) -> tuple[int | None, int | None]:
    """Return ``(processing, total)`` from a ``/slots`` payload, or ``(None, None)``.

    ``is_processing`` is the busy-ness evidence: a sample that returns with a
    processing slot was taken while the GPU was working on that slot.
    """
    try:
        slots = json.loads(payload)
    except (ValueError, TypeError):
        return None, None
    if not isinstance(slots, list):
        return None, None
    processing = 0
    for slot in slots:
        if isinstance(slot, dict) and slot.get("is_processing") is True:
            processing += 1
    return processing, len(slots)


def fetch_sample(
    index: int,
    url: str,
    timeout: float,
    *,
    opener: Callable[..., Any] | None = None,
) -> Sample:
    """Time one ``GET`` of *url*.

    A timeout is recorded as such (not as a slow success) because the blocking
    *is* the finding: llama.cpp serialises ``/slots`` behind the generation
    loop, so the endpoint can outlast any budget the proxy is willing to wait.
    """
    do_open = opener or urllib.request.urlopen
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    start = time.monotonic()
    try:
        with do_open(url, timeout=timeout) as response:
            payload = response.read()
            status = getattr(response, "status", None) or getattr(response, "code", None)
            latency_ms = (time.monotonic() - start) * 1000.0
    except urllib.error.HTTPError as exc:
        return Sample(
            index=index,
            started_at=started_at,
            latency_ms=(time.monotonic() - start) * 1000.0,
            ok=False,
            http_status=exc.code,
            error=f"HTTP {exc.code}",
        )
    except (TimeoutError, urllib.error.URLError) as exc:
        reason = getattr(exc, "reason", exc)
        timed_out = isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError) or "timed out" in str(exc)
        return Sample(
            index=index,
            started_at=started_at,
            latency_ms=(time.monotonic() - start) * 1000.0,
            ok=False,
            timed_out=timed_out,
            error=f"timeout after {timeout:g}s" if timed_out else f"{type(exc).__name__}: {exc}",
        )
    except OSError as exc:  # pragma: no cover - connection refused/reset
        return Sample(
            index=index,
            started_at=started_at,
            latency_ms=(time.monotonic() - start) * 1000.0,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    processing, total = _slot_counts(payload)
    return Sample(
        index=index,
        started_at=started_at,
        latency_ms=latency_ms,
        ok=True,
        http_status=status,
        processing_slots=processing,
        total_slots=total,
    )


def run_probe(
    url: str,
    *,
    samples: int = 12,
    interval: float = 0.15,
    timeout: float = 35.0,
    opener: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Sample]:
    """Take *samples* sequential timed calls, *interval* seconds apart."""
    out: list[Sample] = []
    for index in range(samples):
        if index and interval > 0:
            sleep(interval)
        out.append(fetch_sample(index, url, timeout, opener=opener))
    return out


def summarise(samples: list[Sample], *, budget_ms: float = DEFAULT_BUDGET_MS) -> dict[str, Any]:
    """p50/p95 (nearest-rank) over successful samples, plus the failure counts.

    Latency statistics deliberately use **only** successful samples: mixing a
    35 s timeout in as a "35 000 ms" latency would fabricate a measurement.
    Timeouts are counted separately, which is what makes them usable evidence.
    """
    latencies = sorted(s.latency_ms for s in samples if s.ok and s.latency_ms is not None)
    return {
        "n": len(samples),
        "ok": len(latencies),
        "timeouts": sum(1 for s in samples if s.timed_out),
        "errors": sum(1 for s in samples if not s.ok and not s.timed_out),
        "busy_samples": sum(1 for s in samples if (s.processing_slots or 0) > 0),
        "idle_samples": sum(1 for s in samples if s.ok and (s.processing_slots or 0) == 0),
        "budget_ms": budget_ms,
        "over_budget": sum(1 for value in latencies if value >= budget_ms),
        "p50_ms": _round(percentile(latencies, 50)),
        "p95_ms": _round(percentile(latencies, 95)),
        "min_ms": _round(latencies[0]) if latencies else None,
        "max_ms": _round(latencies[-1]) if latencies else None,
    }


def _round(value: Any) -> float | None:
    return None if value is None else round(float(value), 1)


# ---------------------------------------------------------------------------
# Proxy-side facts (the same-moment comparison)
# ---------------------------------------------------------------------------


def last_status_line_from_text(text: str) -> str | None:
    """Return the last ``status_request`` line in *text* (``None`` when absent)."""
    last: str | None = None
    for line in text.splitlines():
        if is_status_poll(line):
            last = line
    return last


def last_status_line(proxy_log: Path, max_bytes: int = TAIL_SCAN_BYTES) -> str | None:
    """Return the most recent ``status_request`` line in the log tail."""
    return last_status_line_from_text(read_tail(proxy_log, max_bytes))


#: The fields this probe compares its child-port samples against. They are read
#: with a local regex because ``lib.capacity_signals.status_fields`` deliberately
#: pins its byte-for-byte pre-F1 field set; the read-path facts live on the
#: additive per-phase surface.
_PROBE_FIELD_RE = re.compile(
    r"\b(?P<key>capacity_hint|local_in_flight_requests|measured_in_flight_requests|"
    r"slots_stale|slots_stale_reason|slots_snapshot_age_seconds|slots_snapshot_source|"
    r"status_query_ms|slots_counts_ms|slots_detail_ms|slots_upstream_calls)=(?P<value>[^\s]+)"
)


def status_facts(line: str | None) -> dict[str, Any]:
    """Extract the read-path facts the probe compares backend cost against.

    ``slots_counts_ms`` / ``slots_detail_ms`` are the handler's own phases, so
    ``child_port_p50_ms`` (this probe) against them separates backend blocking
    from proxy-side serialisation. Fields absent on a pre-F1 line stay
    ``None`` — the probe never fabricates a phase timing.
    """
    if not line:
        return {}
    fields = {match.group("key"): match.group("value") for match in _PROBE_FIELD_RE.finditer(line)}
    return {
        "line_at": line[:19],
        "capacity_hint": _none_if_literal(fields.get("capacity_hint")),
        "local_in_flight_requests": parse_int(fields.get("local_in_flight_requests")),
        "measured_in_flight_requests": parse_int(fields.get("measured_in_flight_requests")),
        "slots_stale": _none_if_literal(fields.get("slots_stale")),
        "slots_stale_reason": _none_if_literal(fields.get("slots_stale_reason")),
        "slots_snapshot_age_seconds": _float_or_none(fields.get("slots_snapshot_age_seconds")),
        "slots_snapshot_source": _none_if_literal(fields.get("slots_snapshot_source")),
        "status_query_ms": parse_phase_ms(fields.get("status_query_ms")),
        "slots_counts_ms": parse_phase_ms(fields.get("slots_counts_ms")),
        "slots_detail_ms": parse_phase_ms(fields.get("slots_detail_ms")),
        "slots_upstream_calls": parse_phase_ms(fields.get("slots_upstream_calls")),
    }


def _none_if_literal(value: str | None) -> str | None:
    return None if value in (None, "None") else value


def _float_or_none(value: str | None) -> float | None:
    try:
        return None if value in (None, "None") else round(float(value), 3)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(
    *,
    model: str,
    port: int,
    url: str,
    samples: list[Sample],
    summary: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    llama_log: Path,
    proxy_log: Path,
    timeout: float,
) -> dict[str, Any]:
    """Assemble the machine-readable probe report."""
    return {
        "meta": {
            "tool": "scripts/probe_child_slots.py",
            "work_item": WORK_ITEM,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "child_port": port,
            "url": url,
            "llama_log": str(llama_log),
            "proxy_log": str(proxy_log),
            "timeout_seconds": timeout,
            "samples_requested": len(samples),
        },
        "summary": summary,
        "proxy_before": before,
        "proxy_after": after,
        "samples": [sample.as_dict() for sample in samples],
    }


def render(report: dict[str, Any]) -> str:
    """Render the human-readable probe summary."""
    meta = report["meta"]
    summary = report["summary"]
    before = report.get("proxy_before") or {}
    after = report.get("proxy_after") or {}
    lines = [
        f"AC2 live child-port /slots probe ({meta['work_item']})",
        f"Run: {meta['generated_at']} | model={meta['model']} port={meta['child_port']} | "
        f"timeout={meta['timeout_seconds']:g}s",
        f"URL: {meta['url']} (port resolved from {meta['llama_log']}: 'proxying request ... on port')",
        "",
        "GPU state before: "
        + _fact_line(before),
        "GPU state after : "
        + _fact_line(after),
        "",
        f"samples: n={summary['n']} ok={summary['ok']} timeouts={summary['timeouts']} "
        f"errors={summary['errors']} busy={summary['busy_samples']} idle={summary['idle_samples']}",
        f"child /slots latency (ok samples): p50={summary['p50_ms']} ms p95={summary['p95_ms']} ms "
        f"min={summary['min_ms']} ms max={summary['max_ms']} ms",
        f"over the {summary['budget_ms']:g} ms budget: {summary['over_budget']}/{summary['ok']} ok samples",
    ]
    counts = before.get("slots_counts_ms")
    detail = before.get("slots_detail_ms")
    if counts is not None or detail is not None:
        lines.append(
            f"proxy read path at the same moment: slots_counts_ms={counts} slots_detail_ms={detail} "
            f"status_query_ms={before.get('status_query_ms')} "
            f"slots_upstream_calls={before.get('slots_upstream_calls')} "
            f"snapshot_source={before.get('slots_snapshot_source')}"
        )
    per_sample = ", ".join(
        "timeout" if sample["timed_out"] else f"{sample['latency_ms']} ms"
        for sample in report["samples"]
    )
    lines.append(f"per-sample: {per_sample}")
    return "\n".join(lines)


def _fact_line(facts: dict[str, Any]) -> str:
    if not facts:
        return "(no status_request line in the log tail)"
    return (
        f"capacity_hint={facts.get('capacity_hint')} "
        f"local_in_flight_requests={facts.get('local_in_flight_requests')} "
        f"measured_in_flight_requests={facts.get('measured_in_flight_requests')} "
        f"slots_stale={facts.get('slots_stale')} "
        f"snapshot_age={facts.get('slots_snapshot_age_seconds')}s "
        f"source={facts.get('slots_snapshot_source')}"
    )


def write_artifacts(report: dict[str, Any], out_dir: Path) -> list[Path]:
    """Write ``probe.json`` and ``probe.txt`` into *out_dir*; return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "probe.json"
    text_path = out_dir / "probe.txt"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(render(report) + "\n", encoding="utf-8")
    return [json_path, text_path]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Time child-port GET /slots directly (bypassing the proxy) while the GPU "
            "is busy, so backend cost and proxy-side serialisation are separated."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name to resolve (default Qwen3)")
    parser.add_argument("--port", type=int, default=None, help="pin the child port (skips log resolution)")
    parser.add_argument("--host", default=DEFAULT_HOST, help="child host (default 127.0.0.1)")
    parser.add_argument("--llama-log", default=str(DEFAULT_LLAMA_LOG), help="llama-server log for port resolution")
    parser.add_argument("--proxy-log", default=str(DEFAULT_PROXY_LOG), help="proxy log for same-moment facts")
    parser.add_argument("--samples", type=int, default=12, help="number of timed calls (default 12)")
    parser.add_argument("--interval", type=float, default=0.15, help="seconds between calls (default 0.15)")
    parser.add_argument("--timeout", type=float, default=35.0, help="per-call timeout in seconds (default 35)")
    parser.add_argument(
        "--budget-ms",
        type=float,
        default=DEFAULT_BUDGET_MS,
        help="over-budget threshold in ms (default 1000, the handler's STATUS_QUERY_TIMEOUT)",
    )
    parser.add_argument("--write-artifact", default=None, metavar="DIR", help="write probe.json/probe.txt into DIR")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    opener: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """CLI entry point. Returns the process exit code (0 ran, 2 usage, 3 no port)."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 2
    if args.samples <= 0:
        print("error: --samples must be a positive integer", file=sys.stderr)
        return 2

    port = args.port
    if port is None:
        port = resolve_child_port(Path(args.llama_log), args.model)
    if port is None:
        print(
            f"error: no child port resolved for model {args.model!r} in {args.llama_log}; "
            "pass --port to pin it",
            file=sys.stderr,
        )
        return 3

    url = f"http://{args.host}:{port}/slots?{urllib.parse.urlencode({'model': args.model})}"
    proxy_log = Path(args.proxy_log)
    before = status_facts(last_status_line(proxy_log))
    samples = run_probe(
        url,
        samples=args.samples,
        interval=args.interval,
        timeout=args.timeout,
        opener=opener,
        sleep=sleep,
    )
    after = status_facts(last_status_line(proxy_log))
    summary = summarise(samples, budget_ms=args.budget_ms)
    report = build_report(
        model=args.model,
        port=port,
        url=url,
        samples=samples,
        summary=summary,
        before=before,
        after=after,
        llama_log=Path(args.llama_log),
        proxy_log=proxy_log,
        timeout=args.timeout,
    )

    if args.write_artifact:
        for path in write_artifacts(report, Path(args.write_artifact)):
            print(f"artifact: {path}", file=sys.stderr)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
