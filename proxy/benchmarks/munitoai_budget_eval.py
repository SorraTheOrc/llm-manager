#!/usr/bin/env python3
"""Evaluate candidate ``rate_limit_rpm`` values for the ``munitoai-free`` entry.

Reproducible measurement backing work item LP-0MVDVJYN5001BMG4 ("Evaluate
whether munitoai-free rate_limit_rpm should be increased").

Reads the rolling proxy logs under ``/var/log/llama-proxy`` (compressed and
plain), filters strictly by on-line timestamps (rotation boundaries are not
trusted), and reports:

- per-window demand (arrivals to ``munitoai-free``) and ''what-if'' spill at
  candidate budgets via a sliding-window simulation using the limiter's own
  semantics (``<=``-pruned 60 s window, capacity N);
- upstream 429 evidence: the rolling-60 s dispatch rate at every observed
  ``upstream error status=429 url=https://api.munito.ai`` line, plus the
  per-demand-bin 429 rate in the pre-gate baseline;
- spill attribution: ``Fallback triggered ... from=munitoai-free`` lines by
  fallback reason, and remote-dispatch share per upstream host;
- latency: budget-wait distribution and MunitoAI first-token (``ttft_seconds``)
  distribution per window.

Everything is deterministic and rule-based; no LLM is involved. Output is a
Markdown report plus a JSON machine-readable copy.

Usage::

    python3 proxy/benchmarks/munitoai_budget_eval.py [--log-dir DIR] [--out DIR]
        [--gate-start "YYYY-MM-DD HH:MM:SS"] [--reserve-start "YYYY-MM-DD HH:MM:SS"]
        [--end "YYYY-MM-DD HH:MM:SS"] [--json]
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import sys
from collections import Counter, deque
from datetime import datetime, timedelta
from pathlib import Path

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}")
POST_RE = re.compile(r"\[remote\] POST \S+ -> (https?://[^/ ]+)")
FALLBACK_RE = re.compile(r"from=([a-z0-9-]+), to=([a-z0-9-]+), reason=([A-Za-z0-9_ ]+)$")
TTFT_RE = re.compile(r"ttft_seconds=([\d.]+) .*provider=(M[\w ]+)")

class LogReader:
    """Yield (timestamp, line) lines from every proxy.log* file, sorted, de-duplicated."""

    def __init__(self, log_dir: Path):
        self.files = sorted(log_dir.glob("proxy.log*"))

    def iter_lines(self):
        seen: set[tuple] = set()
        for f in self.files:
            opener = gzip.open if f.suffix == ".gz" else open
            try:
                with opener(f, "rt", errors="ignore") as fh:
                    for line in fh:
                        m = TS_RE.match(line)
                        if not m:
                            continue
                        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                        key = (ts, line[:160])
                        if key in seen:
                            continue
                        seen.add(key)
                        yield ts, line
            except OSError as e:
                print(f"      (skipping unreadable log {f.name}: {e})", file=sys.stderr)


def parse_event(ts: datetime, line: str) -> dict | None:
    """Map a log line to a typed event, or None if it is not one we track."""
    if "[remote] POST" in line:
        m = POST_RE.search(line)
        if m:
            host = m.group(1).lstrip("https://").lstrip("http://").split("/")[0]
            return {"ts": ts, "kind": "post", "host": host}
        return None
    if "url=https://api.munito.ai" in line and "upstream error status=429" in line:
        return {"ts": ts, "kind": "upstream_429"}
    if "Provider budget exhausted: skipping provider=munitoai-free" in line:
        return {"ts": ts, "kind": "skip"}
    if "Provider budget wait start: provider=munitoai-free" in line:
        return {"ts": ts, "kind": "wait_start"}
    if "Provider budget wait satisfied: provider=munitoai-free" in line:
        return {"ts": ts, "kind": "wait_satisfied",
                "waited": _waited(line)}
    if "Provider budget wait exhausted: provider=munitoai-free" in line:
        return {"ts": ts, "kind": "wait_exhausted", "waited": _waited(line)}
    if "Provider budget wait aborted: provider=munitoai-free" in line:
        return {"ts": ts, "kind": "wait_aborted", "waited": _waited(line)}
    if "Provider budget wait declined: provider=munitoai-free" in line:
        return {"ts": ts, "kind": "wait_declined"}
    if "Fallback triggered" in line and "from=munitoai-free" in line:
        m = FALLBACK_RE.search(line)
        if m:
            return {"ts": ts, "kind": "fallback", "to": m.group(2), "reason": m.group(3)}
        return None
    if "ttft_seconds=" in line and "provider=MunitoAI Free" in line:
        m = TTFT_RE.search(line)
        if m:
            return {"ts": ts, "kind": "ttft", "ttft": float(m.group(1))}
        return None
    return None


def _waited(line: str) -> float:
    m = re.search(r"waited=([\d.]+)s", line)
    return float(m.group(1)) if m else 0.0


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def describe(name, values):
    if not values:
        return {name: {"count": 0}}
    return {
        name: {
            "count": len(values),
            "p50": round(percentile(values, 50), 2),
            "p90": round(percentile(values, 90), 2),
            "p95": round(percentile(values, 95), 2),
            "p99": round(percentile(values, 99), 2),
            "max": round(max(values), 2),
        }
    }


def rolling_counts(timestamps, window_s=60):
    """For each minute boundary, the number of events in the preceding window_s."""
    if not timestamps:
        return []
    start = timestamps[0]
    end = timestamps[-1]
    counts = []
    minute = start.replace(second=0, microsecond=0)
    while minute <= end:
        lo = minute - timedelta(seconds=window_s)
        n = bisect.bisect_right(timestamps, minute) - bisect.bisect_left(timestamps, lo)
        counts.append(n)
        minute += timedelta(seconds=60)
    return counts


def simulate_capacity(arrivals, capacity):
    """Sliding-window simulation using the limiter's own semantics.

    A 60 s window pruned with ``<=`` (identical to rate_limiter.py); an arrival
    is served iff fewer than *capacity* arrivals are inside the window, else it
    spills.
    """
    window = deque()
    served = spilled = 0
    for ts in arrivals:
        cutoff = ts - timedelta(seconds=60)
        while window and window[0] <= cutoff:
            window.popleft()
        if len(window) < capacity:
            window.append(ts)
            served += 1
        else:
            spilled += 1
    return served, spilled


def analyze(log_dir: Path, gate_start: datetime, reserve_start: datetime, end: datetime):
    post_ts: list[datetime] = []  # munitoai-only dispatch attempts
    post_hosts_ts: list[tuple] = []
    up429_ts: list[datetime] = []
    skips: list[datetime] = []
    wait_satisfied: list[tuple[datetime, float]] = []
    wait_exhausted: list[tuple[datetime, float]] = []
    wait_aborted: list[tuple[datetime, float]] = []
    wait_declined: list[datetime] = []
    fallbacks: list[dict] = []
    ttft: list[tuple[datetime, float]] = []
    total_lines = 0

    for ts, line in LogReader(log_dir).iter_lines():
        if ts < gate_start - timedelta(days=2) or ts > end:
            continue
        total_lines += 1
        ev = parse_event(ts, line)
        if ev is None:
            continue
        kind = ev["kind"]
        if kind == "post":
            if ev["host"] == "api.munito.ai":
                post_ts.append(ts)
            post_hosts_ts.append((ts, ev["host"]))
        elif kind == "upstream_429":
            up429_ts.append(ts)
        elif kind == "skip":
            skips.append(ts)
        elif kind == "wait_satisfied":
            wait_satisfied.append((ts, ev["waited"]))
        elif kind == "wait_exhausted":
            wait_exhausted.append((ts, ev["waited"]))
        elif kind == "wait_aborted":
            wait_aborted.append((ts, ev["waited"]))
        elif kind == "wait_declined":
            wait_declined.append(ts)
        elif kind == "fallback":
            fallbacks.append(ev)
        elif kind == "ttft":
            ttft.append((ts, ev["ttft"]))

    def _slice(events, start, finish):
        if isinstance(events, list) and events and isinstance(events[0], tuple):
            return [e for e in events if start <= e[0] < finish]
        return [e for e in events if start <= e < finish]

    post_ts.sort()
    w_post = {
        "base": _slice(post_ts, gate_start - timedelta(hours=22), gate_start),
        "v1": _slice(post_ts, gate_start, reserve_start),
        "v2": _slice(post_ts, reserve_start, end),
    }
    up429 = _slice(up429_ts, gate_start - timedelta(hours=22), end)
    up429_base = _slice(up429_ts, gate_start - timedelta(hours=22), gate_start)
    up429_v1 = _slice(up429_ts, gate_start, reserve_start)
    up429_v2 = _slice(up429_ts, reserve_start, end)

    result = {
        "windows": {
            "base": {"start": str(gate_start - timedelta(hours=22)), "end": str(gate_start)},
            "v1": {"start": str(gate_start), "end": str(reserve_start)},
            "v2": {"start": str(reserve_start), "end": str(end)},
        },
        "events": {
            "posts_total": len(post_ts),
            "upstream_429_total": len(up429),
            "skips": len(skips),
            "wait_satisfied": len(wait_satisfied),
            "wait_exhausted": len(wait_exhausted),
            "wait_aborted": len(wait_aborted),
            "wait_declined": len(wait_declined),
        },
    }

    # --- per-window demand + spill attribution ---
    for name, (lbl, s, f) in {
        "base": ("base", gate_start - timedelta(hours=22), gate_start),
        "v1": ("v1", gate_start, reserve_start),
        "v2": ("v2", reserve_start, end),
    }.items():
        n_post = len(w_post[name])
        n_skip = len(_slice(skips, s, f))
        n_wait_exh = len(_slice(wait_exhausted, s, f))
        n_wait_abd = len(_slice(wait_aborted, s, f))
        n_wait_dec = len(_slice(wait_declined, s, f))
        arrivals = w_post[name] + _slice(skips, s, f) \
            + [t for t, _ in _slice(wait_exhausted, s, f)] \
            + [t for t, _ in _slice(wait_aborted, s, f)] \
            + _slice(wait_declined, s, f)
        arrivals.sort()
        n_429 = len(_slice(up429_ts, s, f))
        # spill attribution from fallbacks inside this window
        fb = [ev for ev in fallbacks if s <= ev["ts"] < f]
        fb_reason = Counter(ev["reason"] for ev in fb)
        fb_to = Counter(ev["to"] for ev in fb)
        # remote dispatch share by host in window
        host_counts = Counter(
            host for (ts, host) in post_hosts_ts if s <= ts < f
        )
        # rolling demand distribution (unconstrained arrivals)
        rc = rolling_counts(arrivals)
        rd = {"count": len(rc)}
        if rc:
            rd["p50"] = percentile(rc, 50)
            rd["p95"] = percentile(rc, 95)
            rd["p99"] = percentile(rc, 99)
            rd["max"] = max(rc)
            rd["minutes_gt_10"] = sum(1 for c in rc if c > 10)
            rd["minutes_gt_15"] = sum(1 for c in rc if c > 15)
            rd["minutes_gt_20"] = sum(1 for c in rc if c > 20)
            rd["share_minutes_gt_10"] = round(rd["minutes_gt_10"] / len(rc), 3)
        # what-if spill at candidate budgets
        whatif = {}
        for cap in (10, 12, 15, 20):
            served, spilled = simulate_capacity(arrivals, cap)
            whatif[str(cap)] = {
                "arrivals": len(arrivals),
                "served": served,
                "spilled": spilled,
                "spill_share": round(spilled / len(arrivals), 4) if arrivals else 0.0,
            }
        whatif["measured"] = {
            "arrivals": len(arrivals),
            "served": n_post,
            "spilled": n_skip + n_wait_exh + n_wait_abd + n_wait_dec,
            "spill_share": round((n_skip + n_wait_exh + n_wait_abd + n_wait_dec) / len(arrivals), 4)
            if arrivals else 0.0,
        }
        result.setdefault("windows_detail", {})[name] = {
            "hours": round((f - s).total_seconds() / 3600, 2),
            "dispatches_munitoai": n_post,
            "upstream_429s": n_429,
            "skips": n_skip,
            "wait_exhausted": n_wait_exh,
            "wait_aborted": n_wait_abd,
            "wait_declined": n_wait_dec,
            "arrivals": len(arrivals),
            "fallback_reasons_munitoai_free": dict(fb_reason),
            "fallback_targets": dict(fb_to),
            "remote_dispatch_share_by_host": dict(host_counts.most_common()),
            "rolling_demand_per_min": rd,
            "whatif": whatif,
        }

    # --- 429 onset: rolling-60 s dispatch rate at each upstream 429 ---
    onset_rates = []
    for t in up429_base + up429_v1 + up429_v2:
        lo = t - timedelta(seconds=60)
        n = bisect.bisect_right(post_ts, t) - bisect.bisect_left(post_ts, lo)
        onset_rates.append(n)
    result["upstream_429"] = {
        "count_base": len(up429_base),
        "count_v1": len(up429_v1),
        "count_v2": len(up429_v2),
        "rolling_60s_dispatch_rate_at_429": {
            "count": len(onset_rates),
            "min": min(onset_rates) if onset_rates else None,
            "median": percentile(onset_rates, 50) if onset_rates else None,
            "max": max(onset_rates) if onset_rates else None,
        },
    }
    # demand bins vs 429s in the 22 h pre-gate baseline
    base_start = gate_start - timedelta(hours=22)
    bin_edges = [0, 5, 10, 15, 20, 30, 40, 60, 1000]
    minute_samples = []
    minute = base_start.replace(second=0, microsecond=0)
    while minute < gate_start:
        lo = minute - timedelta(seconds=60)
        n = bisect.bisect_right(post_ts, minute) - bisect.bisect_left(post_ts, lo)
        n429 = sum(1 for t in up429_base if lo < t <= minute)
        minute_samples.append((n, n429))
        minute += timedelta(seconds=60)
    bins = {}
    for i in range(len(bin_edges) - 1):
        lo_e, hi_e = bin_edges[i], bin_edges[i + 1]
        sel = [s for s in minute_samples if lo_e <= s[0] < hi_e]
        if not sel:
            continue
        bins[f"{lo_e}-{hi_e-1 if hi_e < 1000 else '+'}"] = {
            "minutes": len(sel),
            "with_429": sum(1 for _, a in sel if a > 0),
            "429_count": sum(a for _, a in sel),
            "pct_minutes_with_429": round(100 * sum(1 for _, a in sel if a > 0) / len(sel), 1),
        }
    result["upstream_429"]["pre_gate_demand_bins"] = bins

    # --- latency ---
    for name, (lbl, s, f) in {
        "base": ("base", gate_start - timedelta(hours=22), gate_start),
        "v1": ("v1", gate_start, reserve_start),
        "v2": ("v2", reserve_start, end),
    }.items():
        tt = [v for (t, v) in _slice(ttft, s, f)]
        loc = {**describe(f"ttft_munitoai_{lbl}", tt)}
        if lbl == "v2":
            waits = [v for _, v in _slice(wait_satisfied, s, f)] \
                + [v for _, v in _slice(wait_exhausted, s, f)] \
                + [v for _, v in _slice(wait_aborted, s, f)]
            loc["wait_seconds_v2"] = {
                **describe("wait", waits)["wait"],
                "count_gt_15s": sum(1 for v in waits if v > 15),
                "count_eq_15s": sum(1 for v in waits if 14.99 <= v <= 15.01),
            }
        result.setdefault("latency", {})[name] = loc

    return result


def render(result) -> str:
    detail = result["windows_detail"]
    lines = []
    lines.append("# munitoai-free rate_limit_rpm evaluation (LP-0MVDVJYN5001BMG4)\n")
    lines.append("## Windows")
    for name, info in detail.items():
        lines.append(
            f"- **{name}** ({info['hours']:.1f} h): dispatches={info['dispatches_munitoai']} "
            f"arrivals={info['arrivals']} upstream_429={info['upstream_429s']}"
        )
    lines.append("")
    lines.append("## Demand (rolling-60 s arrivals to munitoai-free)")
    for name, info in detail.items():
        rd = info["rolling_demand_per_min"]
        lines.append(
            f"- {name}: minutes={rd['count']} p50={rd.get('p50')} p95={rd.get('p95')} "
            f"p99={rd.get('p99')} max={rd.get('max')} "
            f"share>10={rd.get('share_minutes_gt_10')}"
        )
    lines.append("")
    lines.append("## What-if spill at candidate budgets (sliding-60 s, limiter semantics)")
    for name, info in detail.items():
        if name == "base":
            continue
        lines.append(f"### {name}")
        for cap, wf in sorted(info["whatif"].items(), key=lambda kv: (kv[0] != "measured", kv[0])):
            lines.append(
                f"- capacity {cap:>8}: arrivals={wf['arrivals']} served={wf['served']} "
                f"spilled={wf['spilled']} spill_share={wf['spill_share']}"
            )
    lines.append("")
    lines.append("## Upstream 429 evidence")
    u = result["upstream_429"]
    lines.append(
        f"- counts: base={u['count_base']} v1={u['count_v1']} v2={u['count_v2']}\n"
        f"- rolling-60 s dispatch rate at 429: min={u['rolling_60s_dispatch_rate_at_429']['min']} "
        f"median={u['rolling_60s_dispatch_rate_at_429']['median']} "
        f"max={u['rolling_60s_dispatch_rate_at_429']['max']}"
    )
    lines.append("- pre-gate demand bins (rolling-60 s dispatch rate -> 429 rate):")
    for b, d in u["pre_gate_demand_bins"].items():
        lines.append(
            f"  - {b} disp/min: minutes={d['minutes']} with_429={d['with_429']} "
            f"429_count={d['429_count']} ({d['pct_minutes_with_429']}%)"
        )
    lines.append("")
    lines.append("## Spill attribution (fallback from munitoai-free)")
    for name, info in detail.items():
        if name == "base":
            continue
        fb = info["fallback_reasons_munitoai_free"]
        lines.append(f"- {name}: {fb}")
    lines.append("")
    lines.append("## Remote dispatch share by host")
    for name, info in detail.items():
        lines.append(f"- {name}: {info['remote_dispatch_share_by_host']}")
    lines.append("")
    lines.append("## Latency")
    for name, info in result["latency"].items():
        lines.append(f"- {name}: {info}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", default="/var/log/llama-proxy", help="proxy log directory")
    ap.add_argument("--out", default=".", help="output directory (report.md / report.json)")
    ap.add_argument("--gate-start", default="2026-10-18 13:05:09",
                    help="gate (rate_limit_rpm) go-live timestamp")
    ap.add_argument("--reserve-start", default="2026-10-18 14:23:12",
                    help="reserve + 15 s cap go-live timestamp")
    ap.add_argument("--end", default=None, help="analysis end timestamp (default: now)")
    ap.add_argument("--json", action="store_true", help="also print JSON summary")
    args = ap.parse_args()

    gate_start = datetime.strptime(args.gate_start, "%Y-%m-%d %H:%M:%S")
    reserve_start = datetime.strptime(args.reserve_start, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S") if args.end else datetime.now()

    result = analyze(Path(args.log_dir), gate_start, reserve_start, end)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(result, indent=2, default=str))
    (out / "report.md").write_text(render(result))
    print(render(result))
    if args.json:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
