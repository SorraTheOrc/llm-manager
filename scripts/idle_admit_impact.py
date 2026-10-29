#!/usr/bin/env python3
"""Before/after impact measurement for idle-aware local admission.

Reproducible measurement behind LP-0MVGHXD59008UOUD (*Measure idle-aware local
admission impact over >=24h*). The behaviour under test is shipped by
LP-0MVGD9Z8Z0055Y5J: when the local backend has been idle for
``server.local_idle_admit_seconds`` (default 60) and nothing is queued, in
flight or loading, an economic ``large_context_bypass`` is served locally
instead of falling back to a paid remote provider. It is off unless
``server.local_idle_admit_enabled: true``.

The tool is a pure log analyser — it never changes proxy behaviour and never
requires the proxy to be running. It scans an arbitrary pair of windows (a
baseline window and a post-deploy window) of the proxy log stream
(``/var/log/llama-proxy``, live + rotated under either naming scheme, plain or
``.gz``, via ``scripts/lib/proxy_logs.py``) and reports the acceptance-criteria
metrics for LP-0MVGHXD59008UOUD:

**AC1 — window aggregates next to the baselines.** ``local_idle_admit`` count,
local serve share, ``large_context_bypass`` count and remote fallback count for
both windows, plus ``local_dispatch_denied`` and local prefill slot-seconds.

**AC1 (LP-0MVK16GME007EYUW) — lease-gate shape attribution.** Every
``local_dispatch_denied`` line whose ``cold_start=False`` is a
``local_lease_active`` refusal (the dispatch gate refusing an otherwise
idle-admissible request because a foreign session holds the local lease —
``owner=``/``active=``). Refusals are split by the ``active=`` field: 0 →
**shape A** (inactive cooldown lease only: the owner finished and the record
sits reserved for the 30 s base cooldown), >= 1 → **shape B** (an ``active``
record with no GPU work behind it). The B split is corroborated with the
owner's ``lease_released reason=`` (``no_progress_watchdog`` /
``orphan_cleanup`` / ``idle_timeout`` → no work; ``prefill_deadline`` → a long
silent prefill; completion reasons → genuinely working) and ``status_request``
lines (``local_active_query=false`` for that owner → confirmed idle).
``lease_released`` lines are counted by raw reason and by group
(no-progress watchdog / adaptive expiry / explicit release / …), and the
bounded contention-queue outcomes (``contention_queue_dispatch`` /
``contention_queue_fallback_after_queue``) say how many refusals the 60 s
budget converted. When the window holds fewer than
``--min-lease-samples`` (default 5) attributable refusals the split
percentages are withheld and the run is a WARN (exit 2) — never a silent
vacuous pass.

**AC2 — idle-admitted vs non-idle local dispatches, same context bucket.**
``dispatch_first_byte_ms`` percentiles and prefill slot-seconds
(``dispatch_to_first_byte_ms``, the slot-held service time) for idle-admitted
requests, compared bucket-by-bucket with non-idle local dispatches. Idle-admitted
requests are expected to be **slow** — that is the trade — so every comparison
states its sample counts and the bucket edges.

**AC3 — denials and mid-stream timeouts.** ``local_dispatch_denied`` delta
between windows, and mid-stream timeouts (``ReadTimeout``, ``empty_response``,
``ConnectTimeout``, ``dispatch_first_byte_timeout``) correlated to an
idle-admitted request by session id inside a 60 s window. Timeout lines that
carry no session id are reported separately as *coincident* (never
*attributed*) — the limitation is stated in the output.

**AC4 — interactive traffic.** p50/p90 ``dispatch_first_byte_ms`` for the
interactive priority tier (``--interactive-priority``, default ``high``) before
vs after, with a noise band; a regression is reported together with the
idle-admitted prompts that overlap it.

**AC5 — enable-by-default decision or concrete tuning follow-up.** A
deterministic recommendation (``enable_by_default`` / ``tune`` / ``measure``)
with the reasons that produced it, so the producer either has its answer or has
the tuning decision (threshold, prompt-size cap, priority gating) recorded.

An extra ``counterfactual`` section estimates how often the override *would*
have fired in a window where it was not enabled, by reconstructing system-wide
local idle gaps from the log stream. It is an approximation (the proxy uses an
in-process monotonic clock and also checks the queue, leases and in-flight
prefills, none of which are fully reconstructable from logs) and is always
labelled as such.

Usage::

  # Last 24 h vs the 24 h before it (the default; runs on live logs):
  ./scripts/idle_admit_impact.py

  # Explicit baseline and post-deploy windows, machine readable:
  ./scripts/idle_admit_impact.py \\
      --before-start '2026-10-19 07:23' --before-end '2026-10-20 07:23' \\
      --after-start '2026-10-21 09:00' --after-end '2026-10-22 09:00' \\
      --window-closed --json > idle_admit_24h.json

  # Markdown section for a work-item comment or docs artifact:
  ./scripts/idle_admit_impact.py --markdown

Exit codes:
  0 - PASS (measurement complete and every AC check met)
  2 - WARN (measurement incomplete or a check missed; never silently passes)
  1 - invalid input / no proxy logs found

Parent: LP-0MVGHXD59008UOUD (feature: LP-0MVGD9Z8Z0055Y5J).
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

#: Reasons an economic bypass is eligible for the idle override. The physical
#: capacity cap (``context_too_large``) is never overridden.
_ECONOMIC_REASON = "large_context_bypass"

#: "Idle" threshold the override uses by default (``local_idle_admit_seconds``).
_DEFAULT_IDLE_THRESHOLD_S = 60.0

#: Session-correlation window for AC3 / AC2 (seconds).
_DEFAULT_CORRELATION_WINDOW_S = 60.0

#: Context buckets (prompt-token edges). The top bucket is open-ended.
_DEFAULT_BUCKET_EDGES = (16384, 32768, 65536, 131072)

#: Below this many idle-admitted samples the AC2/AC4 comparison is not
#: statistically meaningful and the verdict stays ``measure``.
_MIN_IDLE_SAMPLES = 5

#: A fallback-rate reduction at or above this percentage counts as material
#: for the enable-by-default recommendation.
_FALLBACK_REDUCTION_MIN_PCT = 2.0

#: Interactive p50/p90 movements within this percentage are "within noise".
_INTERACTIVE_NOISE_PCT = 10.0

#: Denial-rate increases within this percentage are tolerated (AC3 asks only
#: that denials do not increase).
_DENIED_TOLERANCE_PCT = 5.0

#: Mid-stream timeout reasons that AC3 attributes (fallback reasons + slot-save
#: read timeouts + first-byte ceiling aborts).
_TIMEOUT_REASONS = {"ReadTimeout", "empty_response", "ConnectTimeout"}

#: Below this many **attributable** lease-gate refusals (denials whose
#: ``active=`` field parsed) in the post-deploy window the shape A/B split is
#: withheld and the run is a WARN (LP-0MVK35HBI003SB3Z).
_MIN_LEASE_SAMPLES = 5

#: Corroboration window after a refusal in which its owner's ``lease_released``
#: and ``status_request`` lines are credited to that refusal. A wedged adaptive
#: lease can run to the 900 s ceiling, so the window matches the ceiling.
_LEASE_EVIDENCE_WINDOW_S = 900.0

#: ``lease_released`` reason → report group. The three groups AC1 of
#: LP-0MVK16GME007EYUW names (no-progress watchdog / adaptive expiry / explicit
#: release) are guaranteed group labels.
_LEASE_RELEASE_GROUPS = {
    "no_progress_watchdog": "no_progress_watchdog",
    "idle_timeout": "adaptive_expiry",
    "prefill_deadline": "adaptive_expiry",
    "orphan_cleanup": "orphan_cleanup",
    "explicit_release": "explicit_release",
    "non_explicit": "client_disconnect",
    "disconnect": "client_disconnect",
    "client_disconnect": "client_disconnect",
    "session_evicted": "session_evicted",
}

#: Release reasons that say the shape-B owner was doing real work (a normal
#: completion / the client left), i.e. **genuine contention**, not an idle lease.
_LEASE_WORKING_REASONS = {"explicit_release", "non_explicit", "disconnect", "client_disconnect", "session_evicted"}

#: Release reasons that confirm the shape-B active record had no work behind
#: it (wedged/abandoned stream, or a lease that idled out).
_LEASE_NO_WORK_REASONS = {"no_progress_watchdog", "orphan_cleanup", "idle_timeout"}

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) - ")
_TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")

_KV_SESSION = re.compile(r"\bsession=([A-Za-z0-9_.:-]+)")
_KV_PROVIDER = re.compile(r"\bprovider=([A-Za-z0-9_.:-]+)")
_KV_MODEL = re.compile(r"\bmodel=([A-Za-z0-9_.:/-]+)")
_KV_PRIORITY = re.compile(r"\bpriority=([A-Za-z_-]+)")
_KV_ESTIMATED = re.compile(r"\bestimated_tokens=(\d+)")
_KV_NEW_TOKENS = re.compile(r"\bnew_tokens=(\d+)")
_KV_CACHED_RATIO = re.compile(r"\bcached_ratio=([\d.]+)")
_KV_CACHE_STATE = re.compile(r"\bcache_state=([A-Za-z_-]+)")
_KV_CHAIN_HOLD = re.compile(r"\bchain_hold_cycle=(\d+)")
_KV_IDLE_SECONDS = re.compile(r"\bidle_seconds=([\d.]+)")
_KV_THRESHOLD = re.compile(r"\bthreshold_seconds=([\d.]+)")
_KV_ERROR = re.compile(r"\berror=([A-Za-z0-9_.]+)")

_DISPATCH_FB = re.compile(r"\bdispatch_first_byte_ms=([\d.]+)")
_DISPATCH_SERVICE = re.compile(r"\bdispatch_to_first_byte_ms=([\d.]+)")
_DISPATCH_QWAIT = re.compile(r"\bqueue_wait_ms=([\d.]+)")
# ``reason=`` is followed by the fallback evidence fields the proxy appends
# since LP-0MVMBMSWR0036RES (``session=``/``attempt=``/budget), so it is bounded
# by the next key=value pair rather than by end-of-line.
_FALLBACK_REASON = re.compile(r"\breason=([A-Za-z0-9_ ]+?)(?=\s+[a-z_]+=|\s*$)")
_SKIP_REASON = re.compile(r"\breason=([A-Za-z0-9_]+)")
_CEILING = re.compile(r"\bceiling=([\d.]+)s")

#: ``local_dispatch_denied`` payload fields (router.py): ``owner=`` may be
#: ``none``; ``active=`` is the occupancy count that decides shape A/B;
#: ``cold_start=True`` denials are a warmup deferral, not a lease refusal.
_KV_OWNER = re.compile(r"\bowner=([A-Za-z0-9_.:-]+)")
_KV_ACTIVE_COUNT = re.compile(r"\bactive=(\d+)")
_KV_COLD_START = re.compile(r"\bcold_start=(True|False|true|false)")
_KV_RELEASE_REASON = re.compile(r"lease_released\b.*?\breason=([A-Za-z0-9_]+)")

#: ``status_request`` (/llama/local/status) corroboration fields.
_KV_LOCAL_ACTIVE_QUERY = re.compile(r"\blocal_active_query=(true|false)")
_KV_OWNER_SESSION = re.compile(r"\blocal_owner_session_id=(\S+)")
_KV_OWNER_LEASE_REMAIN = re.compile(r"\blocal_owner_lease_remaining_seconds=(None|[\d.]+)")
_KV_PREFILL_ETA = re.compile(r"\bprefill_eta_seconds=(None|[\d.]+)")
_KV_IN_FLIGHT = re.compile(r"\blocal_in_flight_requests=(\d+)")
_KV_QUEUED = re.compile(r"\bqueued_duration=([\d.]+)s")


def _first(pattern: re.Pattern, text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _num(pattern: re.Pattern, text: str) -> float | None:
    raw = _first(pattern, text)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _int(pattern: re.Pattern, text: str) -> int | None:
    raw = _first(pattern, text)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_ts(line: str) -> dt.datetime | None:
    match = _TS_RE.match(line)
    if not match:
        return None
    stamp = f"{match.group(1)}.{match.group(2)}"
    try:
        return dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None


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
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _dist_stats(values: list[float]) -> dict:
    """Latency distribution in milliseconds plus the summed slot-seconds."""
    if not values:
        return {
            "n": 0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "max_ms": 0.0,
            "sum_slot_seconds": 0.0,
        }
    return {
        "n": len(values),
        "p50_ms": round(_percentile(values, 50), 1),
        "p90_ms": round(_percentile(values, 90), 1),
        "max_ms": round(max(values), 1),
        "sum_slot_seconds": round(sum(values) / 1000.0, 1),
    }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_event(line: str, ts: dt.datetime) -> dict | None:
    """Parse one log line into an event dict, or ``None`` when irrelevant.

    Every branch is tolerant: a missing field stays ``None`` rather than
    dropping the event, so log-format drift degrades a metric instead of
    breaking the measurement.
    """
    if "Fallback triggered for model=" in line:
        reason = _first(_FALLBACK_REASON, line)
        return {
            "kind": "fallback",
            "ts": ts,
            "session": _first(_KV_SESSION, line),
            "reason": reason,
            # AC3 classifies mid-stream timeouts, which surface as fallback
            # reasons (``empty_response``/``ReadTimeout``/``ConnectTimeout``).
            "timeout": reason in _TIMEOUT_REASONS,
        }
    if "dispatch_first_byte_ms=" in line:
        return {
            "kind": "local_dispatch",
            "ts": ts,
            "session": _first(_KV_SESSION, line),
            "model": _first(_KV_MODEL, line),
            "priority": _first(_KV_PRIORITY, line),
            "estimated_tokens": _int(_KV_ESTIMATED, line),
            "cache_state": _first(_KV_CACHE_STATE, line),
            "cached_ratio": _num(_KV_CACHED_RATIO, line),
            "first_byte_ms": _num(_DISPATCH_FB, line),
            "service_ms": _num(_DISPATCH_SERVICE, line),
            "queue_wait_ms": _num(_DISPATCH_QWAIT, line),
        }
    if "dispatch_first_byte_timeout" in line:
        return {
            "kind": "timeout",
            "ts": ts,
            "session": _first(_KV_SESSION, line),
            "reason": "dispatch_first_byte_timeout",
            "ceiling_s": _num(_CEILING, line),
        }
    if "slot_save failed" in line and "error=" in line:
        error = _first(_KV_ERROR, line)
        if error in _TIMEOUT_REASONS or (error or "").startswith("ReadTimeout"):
            return {
                "kind": "timeout",
                "ts": ts,
                "session": _first(_KV_SESSION, line),
                "reason": error,
                "ceiling_s": None,
            }
        return None
    if "local_dispatch_denied" in line:
        cold_raw = _first(_KV_COLD_START, line)
        return {
            "kind": "dispatch_denied",
            "ts": ts,
            "session": _first(_KV_SESSION, line),
            "owner": _first(_KV_OWNER, line),
            # ``active=`` is the occupancy count the gate refused on: 0 means
            # only inactive (cooldown) leases occupy the pool → shape A;
            # >= 1 means an ``active`` record exists → shape B (possibly with
            # no work behind it). ``None`` on pre-instrumentation logs.
            "active": _int(_KV_ACTIVE_COUNT, line),
            "cold_start": None if cold_raw is None else cold_raw.lower() == "true",
        }
    if "lease_released" in line:
        reason = _first(_KV_RELEASE_REASON, line) or _first(_SKIP_REASON, line)
        return {
            "kind": "lease_released",
            "ts": ts,
            "session": _first(_KV_SESSION, line),
            "reason": reason,
        }
    if "contention_queue_dispatch" in line or "contention_queue_fallback_after_queue" in line:
        return {
            "kind": "contention_queue",
            "ts": ts,
            # The queue is only reached on a local-capacity condition; the
            # ``reason=`` field says which (``local_lease_active`` is the lease
            # gate's bounded wait).
            "action": (
                "dispatch" if "contention_queue_dispatch" in line else "fallback_after_queue"
            ),
            "reason": _first(_SKIP_REASON, line),
            "session": _first(_KV_SESSION, line),
            "queued_seconds": _num(_KV_QUEUED, line),
        }
    if "status_request" in line:
        flag = _first(_KV_LOCAL_ACTIVE_QUERY, line)
        owner = _first(_KV_OWNER_SESSION, line)
        remaining = _num(_KV_OWNER_LEASE_REMAIN, line)
        prefill_eta = _num(_KV_PREFILL_ETA, line)
        return {
            "kind": "status_request",
            "ts": ts,
            "owner": None if owner in (None, "None") else owner,
            "local_active_query": None if flag is None else flag == "true",
            "owner_lease_remaining_seconds": remaining,
            "prefill_eta_seconds": prefill_eta,
            "in_flight": _int(_KV_IN_FLIGHT, line),
        }
    if "routing_idle_admit" in line:
        return {
            "kind": "idle_admit",
            "ts": ts,
            "session": _first(_KV_SESSION, line),
            "model": _first(_KV_MODEL, line),
            "reason": _first(_SKIP_REASON, line),
            "idle_seconds": _num(_KV_IDLE_SECONDS, line),
            "threshold_seconds": _num(_KV_THRESHOLD, line),
            "estimated_tokens": _int(_KV_ESTIMATED, line),
            "new_tokens": _int(_KV_NEW_TOKENS, line),
            "cached_ratio": _num(_KV_CACHED_RATIO, line),
            "cache_state": _first(_KV_CACHE_STATE, line),
        }
    if "routing_skip_local" in line:
        return {
            "kind": "routing_skip",
            "ts": ts,
            "session": _first(_KV_SESSION, line),
            "model": _first(_KV_MODEL, line),
            "reason": _first(_SKIP_REASON, line),
            "estimated_tokens": _int(_KV_ESTIMATED, line),
            "new_tokens": _int(_KV_NEW_TOKENS, line),
            "cached_ratio": _num(_KV_CACHED_RATIO, line),
            "cache_state": _first(_KV_CACHE_STATE, line),
            "chain_hold_cycle": _int(_KV_CHAIN_HOLD, line),
        }
    if "routing_check" in line:
        return {
            "kind": "routing_check",
            "ts": ts,
            "session": _first(_KV_SESSION, line),
            "model": _first(_KV_MODEL, line),
            "estimated_tokens": _int(_KV_ESTIMATED, line),
            "new_tokens": _int(_KV_NEW_TOKENS, line),
            "cached_ratio": _num(_KV_CACHED_RATIO, line),
            "cache_state": _first(_KV_CACHE_STATE, line),
        }
    if "Stream finished:" in line:
        provider = _first(_KV_PROVIDER, line)
        return {
            "kind": "stream_finished",
            "ts": ts,
            "session": _first(_KV_SESSION, line),
            "provider": provider,
            "is_local": provider == "local",
        }
    return None


def collect_events(log_dir: Path, *, since: dt.datetime | None, until: dt.datetime | None) -> list[dict]:
    """Return every parsed event whose timestamp falls in ``[since, until)``."""
    events: list[dict] = []
    for path in discover_proxy_log_files(Path(log_dir)):
        with open_proxy_log_text(path) as fh:
            for line in fh:
                ts = _parse_ts(line)
                if ts is None:
                    continue
                if since is not None and ts < since:
                    continue
                if until is not None and ts >= until:
                    continue
                event = parse_event(line, ts)
                if event is not None:
                    events.append(event)
    events.sort(key=lambda e: e["ts"])
    return events


# ---------------------------------------------------------------------------
# Window aggregation
# ---------------------------------------------------------------------------


def _bucket_label(tokens: int | None, edges: tuple[int, ...]) -> str:
    """Return the context-bucket label for a prompt-token estimate."""
    if tokens is None or tokens <= 0:
        return "unknown"
    low = 0
    for edge in edges:
        if tokens < edge:
            return f"{low}-{edge}"
        low = edge
    return f">={low}"


def _window_length_hours(start: dt.datetime, end: dt.datetime) -> float:
    return max((end - start).total_seconds() / 3600.0, 1e-9)


def _lease_release_group(reason: str | None) -> str:
    """Map a raw ``lease_released reason=`` onto its report group."""
    return _LEASE_RELEASE_GROUPS.get(reason or "", "other")


_LEASE_ATTRIBUTION_RULE = (
    "Every `local_dispatch_denied` line with `cold_start=False` is a "
    "`local_lease_active` refusal. The `active=` field on that line decides the "
    "shape: `active=0` → shape A (no active record at all, so the pool is held "
    "only by inactive cooldown leases — the owner finished and its record is "
    "reserved for the base cooldown); `active>=1` → shape B (an `active` record "
    "exists). Shape B is corroborated, not assumed: the owner's earliest "
    "`lease_released` inside 900 s of the refusal says whether the record had no "
    "work behind it (`no_progress_watchdog`, `orphan_cleanup`, `idle_timeout`), "
    "was a long silent prefill (`prefill_deadline`), or was genuinely working "
    "(a completion/disconnect reason); `status_request` lines that name the "
    "owner with `local_active_query=false` confirm an idle owner. Buckets are "
    "counted in that precedence order, so each refusal lands in exactly one."
)


def lease_gate_summary(
    denials: list[dict],
    releases: list[dict],
    statuses: list[dict],
    queues: list[dict],
    *,
    min_samples: int,
    evidence_window_s: float = _LEASE_EVIDENCE_WINDOW_S,
) -> dict:
    """Attribute every ``local_lease_active`` refusal to shape A or shape B.

    Read-only aggregation: the refusal counts are direct observations, the
    A/B split comes from the denial's own ``active=`` field, and the shape-B
    sub-split is corroborated by the owner's ``lease_released`` reason and by
    ``status_request`` snapshots. Percentages are withheld when fewer than
    *min_samples* refusals could be attributed, so a thin window can never read
    as a confident split.
    """
    refusals = [d for d in denials if d.get("cold_start") is not True]
    cold_start_refusals = len(denials) - len(refusals)

    releases_by_session: dict[str, list[dict]] = defaultdict(list)
    for event in releases:
        if event.get("session"):
            releases_by_session[event["session"]].append(event)
    statuses_by_owner: dict[str, list[dict]] = defaultdict(list)
    for event in statuses:
        if event.get("owner"):
            statuses_by_owner[event["owner"]].append(event)

    by_active = {"active_0": 0, "active_positive": 0, "unknown": 0}
    shape_a = 0
    shape_b = 0
    a_release_groups: dict[str, int] = defaultdict(int)
    b_evidence: dict[str, int] = defaultdict(int)
    owner_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"refusals": 0, "shape_a": 0, "shape_b": 0}
    )

    for denial in refusals:
        owner = denial.get("owner")
        if owner in (None, "none", "unknown"):
            owner = None
        active = denial.get("active")
        if owner:
            owner_stats[owner]["refusals"] += 1

        # Owner evidence: the earliest release after the refusal, and any
        # status snapshot naming the owner inside the evidence window.
        release = None
        if owner:
            for candidate in releases_by_session.get(owner, []):
                delay = (candidate["ts"] - denial["ts"]).total_seconds()
                if 0 <= delay <= evidence_window_s:
                    release = candidate
                    break
        status_idle = False
        if owner:
            for candidate in statuses_by_owner.get(owner, []):
                delay = (candidate["ts"] - denial["ts"]).total_seconds()
                if 0 <= delay <= evidence_window_s and candidate.get("local_active_query") is False:
                    status_idle = True
                    break
        release_reason = release.get("reason") if release else None

        if active is None:
            by_active["unknown"] += 1
            continue
        if active == 0:
            by_active["active_0"] += 1
            shape_a += 1
            if owner:
                owner_stats[owner]["shape_a"] += 1
            a_release_groups[_lease_release_group(release_reason) if release_reason else "none"] += 1
            continue

        by_active["active_positive"] += 1
        shape_b += 1
        if owner:
            owner_stats[owner]["shape_b"] += 1
        # Precedence: a hard release reason first (it is direct evidence of
        # what the record was doing), then the status snapshot, then a
        # completion reason, then no evidence at all.
        if release_reason in _LEASE_NO_WORK_REASONS:
            b_evidence[_lease_release_group(release_reason)] += 1
        elif release_reason == "prefill_deadline":
            b_evidence["prefill_deadline"] += 1
        elif status_idle:
            b_evidence["status_idle_owner"] += 1
        elif release_reason in _LEASE_WORKING_REASONS:
            b_evidence["owner_working"] += 1
        else:
            b_evidence["unconfirmed"] += 1

    samples = by_active["active_0"] + by_active["active_positive"]
    attributable = samples >= min_samples

    def _pct(count: int) -> float | None:
        if not attributable or samples == 0:
            return None
        return round(100.0 * count / samples, 1)

    no_work = (
        b_evidence.get("no_progress_watchdog", 0)
        + b_evidence.get("orphan_cleanup", 0)
        + b_evidence.get("adaptive_expiry", 0)
        + b_evidence.get("status_idle_owner", 0)
    )
    silent_prefill = b_evidence.get("prefill_deadline", 0)
    owner_working = b_evidence.get("owner_working", 0)
    unconfirmed = b_evidence.get("unconfirmed", 0)

    queue_counts = {"dispatch": 0, "fallback_after_queue": 0}
    for event in queues:
        if event.get("reason") != "local_lease_active":
            continue
        action = event.get("action")
        if action in queue_counts:
            queue_counts[action] += 1

    if samples == 0:
        headline = (
            "No attributable `local_lease_active` refusals in this window: the "
            "lease-gate cost cannot be attributed."
        )
    elif not attributable:
        headline = (
            f"Only {samples} attributable `local_lease_active` refusal(s) "
            f"(< {min_samples} required): shape A/B split withheld."
        )
    else:
        headline = (
            f"{samples} attributable `local_lease_active` refusal(s): shape A "
            f"{shape_a} ({_pct(shape_a)} %), shape B {shape_b} ({_pct(shape_b)} %). "
            f"Shape B evidence: no work behind it {no_work}, silent prefill "
            f"{silent_prefill}, owner working {owner_working}, unconfirmed "
            f"{unconfirmed}. Contention queue: converted {queue_counts['dispatch']} "
            f"to local, fell through {queue_counts['fallback_after_queue']}."
        )

    return {
        "refusals": len(refusals),
        "cold_start_refusals": cold_start_refusals,
        "samples": samples,
        "unattributed_refusals": by_active["unknown"],
        "min_samples": min_samples,
        "attribution_status": "ok" if attributable else "insufficient_samples",
        "attribution_rule": _LEASE_ATTRIBUTION_RULE,
        "refusals_by_active": by_active,
        "shape_a": {
            "label": (
                "inactive cooldown lease (owner finished; record reserved for its "
                "session for the base cooldown)"
            ),
            "count": shape_a,
            "pct": _pct(shape_a),
            "owner_release_groups": dict(sorted(a_release_groups.items(), key=lambda kv: -kv[1])),
        },
        "shape_b": {
            "label": (
                "`active` record with no GPU work behind it (wedged/abandoned "
                "stream or a long silent prefill)"
            ),
            "count": shape_b,
            "pct": _pct(shape_b),
            "no_work_behind_it": no_work,
            "silent_prefill": silent_prefill,
            "owner_working": owner_working,
            "unconfirmed": unconfirmed,
            "by_evidence": {
                key: b_evidence.get(key, 0)
                for key in (
                    "no_progress_watchdog",
                    "orphan_cleanup",
                    "adaptive_expiry",
                    "status_idle_owner",
                    "prefill_deadline",
                    "owner_working",
                    "unconfirmed",
                )
            },
        },
        "contention_queue": queue_counts,
        "top_owners": [
            {"owner": owner, **counts}
            for owner, counts in sorted(
                owner_stats.items(), key=lambda kv: (-kv[1]["refusals"], kv[0])
            )[:5]
        ],
        "headline": headline,
    }


def lease_release_summary(releases: list[dict]) -> dict:
    """Count ``lease_released`` lines by raw reason and by report group."""
    by_reason: dict[str, int] = defaultdict(int)
    by_group: dict[str, int] = defaultdict(int)
    for event in releases:
        reason = event.get("reason") or "unknown"
        by_reason[reason] += 1
        by_group[_lease_release_group(event.get("reason"))] += 1
    return {
        "total": len(releases),
        "by_reason": dict(sorted(by_reason.items(), key=lambda kv: (-kv[1], kv[0]))),
        "by_group": dict(sorted(by_group.items(), key=lambda kv: (-kv[1], kv[0]))),
        # The three groups the parent AC1 names explicitly; always present so
        # a reader never has to guess whether a zero means "none" or "missing".
        "groups_documented": ["no_progress_watchdog", "adaptive_expiry", "explicit_release"],
    }


def lease_status_summary(statuses: list[dict]) -> dict:
    """Aggregate the ``status_request`` corroboration fields."""
    remaining = [
        event["owner_lease_remaining_seconds"]
        for event in statuses
        if event.get("owner_lease_remaining_seconds") is not None
    ]
    return {
        "samples": len(statuses),
        "local_active_query_true": sum(
            1 for event in statuses if event.get("local_active_query") is True
        ),
        "local_active_query_false": sum(
            1 for event in statuses if event.get("local_active_query") is False
        ),
        "owner_lease_remaining_seconds_p50": (
            round(_percentile(remaining, 50), 1) if remaining else None
        ),
        "owner_lease_remaining_seconds_max": round(max(remaining), 1) if remaining else None,
        "prefill_in_flight_samples": sum(
            1 for event in statuses if event.get("prefill_eta_seconds") is not None
        ),
    }


def summarise_window(
    events: list[dict],
    *,
    bucket_edges: tuple[int, ...],
    correlation_window_s: float,
    min_lease_samples: int = _MIN_LEASE_SAMPLES,
) -> dict:
    """Aggregate one window's events into the AC1/AC2/AC3/AC4 metric set."""
    local_dispatches = [e for e in events if e["kind"] == "local_dispatch"]
    fallbacks = [e for e in events if e["kind"] == "fallback"]
    skips = [e for e in events if e["kind"] == "routing_skip"]
    idle_admits = [e for e in events if e["kind"] == "idle_admit"]
    denials = [e for e in events if e["kind"] == "dispatch_denied"]
    releases = [e for e in events if e["kind"] == "lease_released"]
    statuses = [e for e in events if e["kind"] == "status_request"]
    queues = [e for e in events if e["kind"] == "contention_queue"]
    timeouts = [e for e in events if e["kind"] == "timeout"] + [
        e for e in fallbacks if e.get("timeout")
    ]
    timeouts.sort(key=lambda e: e["ts"])
    streams = [e for e in events if e["kind"] == "stream_finished"]
    checks = [e for e in events if e["kind"] == "routing_check"]

    # ---- AC1 window aggregates -------------------------------------------
    local_streams = sum(1 for e in streams if e.get("is_local"))
    remote_streams = sum(1 for e in streams if not e.get("is_local"))
    served_total = local_streams + remote_streams
    bypass_skips = [
        e
        for e in skips
        if e.get("reason") == _ECONOMIC_REASON
        # chain_hold_cycle > 0 re-emits a held request; counting those would
        # double-count one client demand (mirrors the usage report).
        and not (e.get("chain_hold_cycle") or 0) > 0
    ]
    fallback_reasons: dict[str, int] = defaultdict(int)
    for event in fallbacks:
        fallback_reasons[event.get("reason") or "unknown"] += 1

    local_service_ms = [e["service_ms"] for e in local_dispatches if e.get("service_ms") is not None]

    ac1 = {
        "local_idle_admit": len(idle_admits),
        "local_streams": local_streams,
        "remote_streams": remote_streams,
        "local_serve_share_pct": round(100.0 * local_streams / served_total, 2) if served_total else 0.0,
        "large_context_bypass": len(bypass_skips),
        "remote_fallback_events": len(fallbacks),
        "fallback_by_reason": dict(sorted(fallback_reasons.items(), key=lambda kv: -kv[1])),
        "local_dispatch_denied": len(denials),
        "local_dispatch_events": len(local_dispatches),
        "local_prefill_slot_seconds": round(sum(local_service_ms) / 1000.0, 1),
        "timeout_events": len(timeouts),
    }

    # ---- AC2 idle vs non-idle local dispatches ---------------------------
    # Pair each idle_admit with the next local dispatch in the SAME session
    # inside the correlation window. A dispatch is claimed at most once, and
    # pairs are taken in timestamp order so the attribution is deterministic.
    dispatches_by_session: dict[str, list[dict]] = defaultdict(list)
    for index, event in enumerate(local_dispatches):
        if event.get("session"):
            dispatches_by_session[event["session"]].append({**event, "_index": index})

    claimed: set[int] = set()
    idle_records: list[dict] = []
    for admit in idle_admits:
        session = admit.get("session")
        match = None
        if session:
            for candidate in dispatches_by_session.get(session, []):
                if candidate["_index"] in claimed:
                    continue
                delta = (candidate["ts"] - admit["ts"]).total_seconds()
                if 0 <= delta <= correlation_window_s:
                    match = candidate
                    break
        if match is None:
            idle_records.append(
                {
                    "admit": admit,
                    "dispatch": None,
                    "tokens": admit.get("estimated_tokens"),
                    "bucket": _bucket_label(admit.get("estimated_tokens"), bucket_edges),
                }
            )
            continue
        claimed.add(match["_index"])
        # The idle_admit line carries the authoritative prompt size for this
        # request; the dispatch line's own estimate is a fallback only.
        tokens = admit.get("estimated_tokens") or match.get("estimated_tokens")
        idle_records.append(
            {
                "admit": admit,
                "dispatch": match,
                "tokens": tokens,
                "bucket": _bucket_label(tokens, bucket_edges),
            }
        )

    idle_service = [r["dispatch"]["service_ms"] for r in idle_records if r["dispatch"] and r["dispatch"].get("service_ms") is not None]
    unmatched_admits = sum(1 for r in idle_records if r["dispatch"] is None)

    # Non-idle dispatches: everything the override did not serve. Context
    # size comes from the dispatch line (instrumented) or, for logs predating
    # that field, the session's most recent ``routing_check`` before it.
    checks_by_session: dict[str, list[dict]] = defaultdict(list)
    for event in checks:
        if event.get("session"):
            checks_by_session[event["session"]].append(event)

    def _tokens_for(dispatch: dict) -> int | None:
        if dispatch.get("estimated_tokens"):
            return dispatch["estimated_tokens"]
        session = dispatch.get("session")
        if not session:
            return None
        best: dict | None = None
        for check in checks_by_session.get(session, []):
            if check["ts"] <= dispatch["ts"]:
                best = check
            else:
                break
        return best.get("estimated_tokens") if best else None

    non_idle_records = [
        {
            "dispatch": event,
            "tokens": _tokens_for(event),
            "bucket": _bucket_label(_tokens_for(event), bucket_edges),
        }
        for index, event in enumerate(local_dispatches)
        if index not in claimed
    ]

    buckets: dict[str, dict] = {}
    for label in sorted(
        {r["bucket"] for r in idle_records} | {r["bucket"] for r in non_idle_records},
        key=_bucket_sort_key,
    ):
        buckets[label] = {"idle": [], "non_idle": []}
    for record in idle_records:
        if record["dispatch"] and record["dispatch"].get("service_ms") is not None:
            buckets[record["bucket"]]["idle"].append(record["dispatch"])
    for record in non_idle_records:
        if record["dispatch"].get("service_ms") is not None:
            buckets[record["bucket"]]["non_idle"].append(record["dispatch"])

    def _latency(records: list[dict], field: str) -> list[float]:
        return [r[field] for r in records if r.get(field) is not None]

    ac2 = {
        "idle_admitted_sample_counts": {
            "idle_admit_events": len(idle_admits),
            "paired_local_dispatches": len(idle_records) - unmatched_admits,
            "unpaired_admits": unmatched_admits,
        },
        "idle_admitted_first_byte_ms": _dist_stats(_latency([r["dispatch"] for r in idle_records if r["dispatch"]], "first_byte_ms")),
        "idle_admitted_service_ms": _dist_stats(idle_service),
        "non_idle_first_byte_ms": _dist_stats(
            _latency([r["dispatch"] for r in non_idle_records], "first_byte_ms")
        ),
        "non_idle_service_ms": _dist_stats(
            _latency([r["dispatch"] for r in non_idle_records], "service_ms")
        ),
        "bucket_edges": list(bucket_edges),
        "by_bucket": {
            label: {
                "bucket_edges": label,
                "idle": {
                    "n": len(bucket["idle"]),
                    "first_byte_ms": _dist_stats(_latency(bucket["idle"], "first_byte_ms")),
                    "service_ms": _dist_stats(_latency(bucket["idle"], "service_ms")),
                },
                "non_idle": {
                    "n": len(bucket["non_idle"]),
                    "first_byte_ms": _dist_stats(_latency(bucket["non_idle"], "first_byte_ms")),
                    "service_ms": _dist_stats(_latency(bucket["non_idle"], "service_ms")),
                },
            }
            for label, bucket in buckets.items()
        },
        # Idle-admitted prompts are expected to be slow: state the trade
        # explicitly rather than leaving it implicit in the percentiles.
        "expected_trade": (
            "Idle-admitted prompts are served on an otherwise idle GPU and are "
            "expected to show higher first-byte latency and slot-seconds than "
            "non-idle dispatches in the same bucket; that cost is the trade for "
            "not paying a remote provider."
        ),
    }
    # ---- AC3 denials + timeout attribution -------------------------------
    idle_by_session: dict[str, list[dt.datetime]] = defaultdict(list)
    for admit in idle_admits:
        if admit.get("session"):
            idle_by_session[admit["session"]].append(admit["ts"])

    attributed: list[dict] = []
    coincident: list[dict] = []
    unrelated = 0
    for event in timeouts:
        session = event.get("session")
        matched = None
        if session:
            for admit_ts in idle_by_session.get(session, []):
                delta = (event["ts"] - admit_ts).total_seconds()
                if 0 <= delta <= correlation_window_s:
                    matched = admit_ts
                    break
        if matched is not None:
            attributed.append(
                {
                    "session": session,
                    "reason": event.get("reason"),
                    "ts": event["ts"].isoformat(sep=" "),
                    "seconds_after_idle_admit": round(
                        (event["ts"] - matched).total_seconds(), 1
                    ),
                }
            )
            continue
        if not session and any(
            abs((event["ts"] - admit["ts"]).total_seconds()) <= correlation_window_s
            for admit in idle_admits
        ):
            coincident.append(
                {
                    "reason": event.get("reason"),
                    "ts": event["ts"].isoformat(sep=" "),
                }
            )
            continue
        unrelated += 1

    ac3 = {
        "local_dispatch_denied": len(denials),
        "timeout_events": len(timeouts),
        "timeouts_attributed_to_idle_admit": attributed,
        "timeouts_coincident_with_idle_admit": coincident,
        "timeouts_unrelated": unrelated,
        "correlation_window_seconds": correlation_window_s,
        "attribution_limit": (
            "Timeout lines that carry no session id (e.g. 'slot_save failed "
            "... error=ReadTimeout', or a 'Fallback triggered' line from a "
            "rotated log predating LP-0MVMBMSWR0036RES's session= field) cannot "
            "be attributed per request; they are reported as coincident only "
            "when they fall inside the correlation window of an idle-admitted "
            "request. Session-bearing fallback lines are matched exactly."
        ),
    }

    # ---- AC4 interactive priority latency --------------------------------
    # Idle-admitted dispatches are EXCLUDED from the interactive comparison:
    # they are the treatment, not the control. Interactive harm shows up as
    # latency on the *other* requests at that tier (an idle-admitted prefill
    # occupies the GPU), so the comparison uses non-idle dispatches and the
    # admits themselves are reported separately as the causal evidence.
    non_idle_dispatches = [
        event for index, event in enumerate(local_dispatches) if index not in claimed
    ]
    ac4 = {
        "interactive_priority": None,  # filled by the caller's comparison
        "by_priority": {},
        "idle_admitted_by_priority": {},
        "semantics": (
            "by_priority covers non-idle dispatches only (the control); "
            "idle_admitted_by_priority counts the admits at each tier (the treatment)."
        ),
    }
    for priority in sorted({e.get("priority") for e in non_idle_dispatches if e.get("priority")}):
        records = [e for e in non_idle_dispatches if e.get("priority") == priority]
        ac4["by_priority"][priority] = {
            "n": len(records),
            "first_byte_ms": _dist_stats(_latency(records, "first_byte_ms")),
        }
    for record in idle_records:
        dispatch = record.get("dispatch")
        priority = (dispatch or {}).get("priority")
        if not priority:
            continue
        ac4["idle_admitted_by_priority"][priority] = (
            ac4["idle_admitted_by_priority"].get(priority, 0) + 1
        )

    return {
        "ac1": ac1,
        "ac2": ac2,
        "ac3": ac3,
        "ac4": ac4,
        # LP-0MVK35HBI003SB3Z: parent LP-0MVK16GME007EYUW AC1 evidence — how
        # much of the idle-capacity win the local dispatch lease gate costs,
        # split by blocking shape.
        "lease_gate": lease_gate_summary(
            denials,
            releases,
            statuses,
            queues,
            min_samples=min_lease_samples,
        ),
        "lease_released": lease_release_summary(releases),
        "lease_status": lease_status_summary(statuses),
        "event_counts": {
            "local_dispatch": len(local_dispatches),
            "stream_finished": len(streams),
            "fallback": len(fallbacks),
            "routing_skip": len(skips),
            "routing_check": len(checks),
            "idle_admit": len(idle_admits),
            "dispatch_denied": len(denials),
            "lease_released": len(releases),
            "status_request": len(statuses),
            "contention_queue": len(queues),
            "timeout": len(timeouts),
        },
        "_idle_records": idle_records,  # internal (rendered only in JSON detail)
        "_events": events,
    }


def _bucket_sort_key(label: str) -> tuple[int, str]:
    if label == "unknown":
        return (1, label)
    return (0, label)


def counterfactual(
    window: dict,
    *,
    idle_threshold_s: float,
) -> dict:
    """Estimate how often the override would have fired in *window*.

    Reconstruction: for every economic bypass skip, measure the gap since the
    last system-wide local activity (a local dispatch or a local stream
    finishing). A gap >= ``idle_threshold_s`` makes the request *eligible*.
    This is an approximation of ``proxy/local_activity.py``: it cannot see
    leases, queued requests, prefill-in-flight state or process restarts.
    """
    events: list[dict] = window["_events"]
    activity = [
        e["ts"]
        for e in events
        if e["kind"] == "local_dispatch" or (e["kind"] == "stream_finished" and e.get("is_local"))
    ]
    if not activity:
        return {
            "approximation": True,
            "idle_threshold_seconds": idle_threshold_s,
            "eligible_bypasses": 0,
            "measured_bypasses": 0,
            "note": "no local activity in the window — idle age cannot be reconstructed",
        }
    # Bisect-style scan: events are time-sorted, so walk both lists once.
    idx = 0
    last_activity: dt.datetime | None = None
    eligible = 0
    bypasses = 0
    gaps: list[float] = []
    for event in events:
        while idx < len(activity) and activity[idx] <= event["ts"]:
            last_activity = activity[idx]
            idx += 1
        if event["kind"] != "routing_skip":
            continue
        if event.get("reason") != _ECONOMIC_REASON:
            continue
        if (event.get("chain_hold_cycle") or 0) > 0:
            continue
        bypasses += 1
        if last_activity is None:
            continue
        gap = (event["ts"] - last_activity).total_seconds()
        gaps.append(gap)
        if gap >= idle_threshold_s:
            eligible += 1
    return {
        "approximation": True,
        "idle_threshold_seconds": idle_threshold_s,
        "measured_bypasses": bypasses,
        "eligible_bypasses": eligible,
        "eligible_pct_of_bypasses": round(100.0 * eligible / bypasses, 1) if bypasses else 0.0,
        "idle_gap_p50_seconds": round(_percentile(gaps, 50), 1) if gaps else None,
        "idle_gap_p90_seconds": round(_percentile(gaps, 90), 1) if gaps else None,
        "note": (
            "Approximation from log-derived local activity; the proxy's real "
            "guard also refuses while anything is leased, queued or prefilling, "
            "so the true fire count is <= eligible_bypasses."
        ),
    }


# ---------------------------------------------------------------------------
# Verdict / recommendation (AC5)
# ---------------------------------------------------------------------------


def _rate(count: float, hours: float) -> float:
    return count / hours if hours > 0 else 0.0


#: Sentinel change percentage used when a metric was zero in the baseline
#: window but non-zero after it (a real, unbounded increase).
_NEW_ACTIVITY_PCT = 999.0


def _pct_change(before: float, after: float) -> float:
    """Percentage change from *before* to *after* (0.0 -> 0.0 is no change)."""
    if before == 0:
        return 0.0 if after == 0 else _NEW_ACTIVITY_PCT
    return round(100.0 * (after - before) / before, 1)


def evaluate(
    before: dict,
    after: dict,
    *,
    hours_before: float,
    hours_after: float,
    interactive_priority: str,
    window_closed: bool,
    deployed_at: dt.datetime | None,
    after_start: dt.datetime | None,
    min_idle_samples: int = _MIN_IDLE_SAMPLES,
    min_lease_samples: int = _MIN_LEASE_SAMPLES,
) -> dict:
    """Evaluate the AC checks and produce the AC5 recommendation."""
    checks: list[dict] = []
    notes: list[str] = []

    before_ac1, after_ac1 = before["ac1"], after["ac1"]

    # Window precondition: a post-deploy window must be >= 24 h and must not
    # start before the deployment it measures.
    if window_closed and hours_after < 24.0:
        checks.append(
            {
                "target": "AC1: post-deploy window is >= 24 h",
                "observed_hours": round(hours_after, 2),
                "pass": False,
            }
        )
    else:
        checks.append(
            {
                "target": "AC1: post-deploy window is >= 24 h",
                "observed_hours": round(hours_after, 2),
                "pass": hours_after >= 24.0,
            }
        )
    if deployed_at is not None and after_start is not None and after_start < deployed_at:
        checks.append(
            {
                "target": "AC1: post-deploy window starts at/after deployment",
                "observed": f"{after_start.isoformat(sep=' ')} < {deployed_at.isoformat(sep=' ')}",
                "pass": False,
            }
        )

    measured = after_ac1["local_idle_admit"] > 0
    checks.append(
        {
            "target": "AC1: local_idle_admit events observed in the post-deploy window",
            "observed": after_ac1["local_idle_admit"],
            "pass": measured,
        }
    )
    if not measured:
        notes.append(
            "No `routing_idle_admit` events in the post-deploy window: the override "
            "is either disabled (`server.local_idle_admit_enabled`) or never fired. "
            "The impact cannot be measured until it is enabled and has run for >= 24 h."
        )

    # AC2: the comparison must carry samples, and idle-admitted prompts are
    # expected to be slow (no threshold — the trade, not a target).
    paired = after["ac2"]["idle_admitted_sample_counts"]["paired_local_dispatches"]
    checks.append(
        {
            "target": "AC2: idle-admitted dispatches paired with their dispatch event",
            "observed": paired,
            "pass": paired >= min_idle_samples,
        }
    )
    if after["ac2"]["idle_admitted_sample_counts"]["unpaired_admits"]:
        notes.append(
            f"{after['ac2']['idle_admitted_sample_counts']['unpaired_admits']} idle_admit "
            "event(s) had no local dispatch event inside the correlation window "
            "(possible: fallback after admission, or a restart mid-request)."
        )
    if paired:
        non_idle_total = after["ac2"]["non_idle_service_ms"]["n"]
        notes.append(
            "AC2 trade: idle-admitted prefill p50 "
            f"{after['ac2']['idle_admitted_service_ms']['p50_ms']} ms "
            f"(n={after['ac2']['idle_admitted_service_ms']['n']}) vs non-idle "
            f"{after['ac2']['non_idle_service_ms']['p50_ms']} ms (n={non_idle_total}); "
            f"slot-seconds {after['ac2']['idle_admitted_service_ms']['sum_slot_seconds']} vs "
            f"{after['ac2']['non_idle_service_ms']['sum_slot_seconds']}."
        )

    # AC3: denials must not increase; zero attributed timeouts.
    denied_before_rate = _rate(before_ac1["local_dispatch_denied"], hours_before)
    denied_after_rate = _rate(after_ac1["local_dispatch_denied"], hours_after)
    denied_change_pct = _pct_change(denied_before_rate, denied_after_rate)
    denied_ok = denied_change_pct <= _DENIED_TOLERANCE_PCT
    checks.append(
        {
            "target": f"AC3: local_dispatch_denied does not increase (<= {_DENIED_TOLERANCE_PCT:.0f}%)",
            "observed_pct_change": denied_change_pct,
            "before_per_hour": round(denied_before_rate, 1),
            "after_per_hour": round(denied_after_rate, 1),
            "pass": denied_ok,
        }
    )
    attributed = after["ac3"]["timeouts_attributed_to_idle_admit"]
    checks.append(
        {
            "target": "AC3: zero timeouts attributable to idle-admitted requests",
            "observed": len(attributed),
            "pass": not attributed,
        }
    )
    coincident = after["ac3"]["timeouts_coincident_with_idle_admit"]
    if coincident:
        notes.append(
            f"{len(coincident)} timeout event(s) carry no session id and fall inside the "
            "correlation window of an idle-admitted request: coincident, not attributable "
            "(see ac3.attribution_limit)."
        )

    # AC4: interactive latency within noise.
    interactive_before = before["ac4"]["interactive"]
    interactive_after = after["ac4"]["interactive"]
    p50_change = _pct_change(interactive_before["p50_ms"], interactive_after["p50_ms"])
    p90_change = _pct_change(interactive_before["p90_ms"], interactive_after["p90_ms"])
    interactive_ok = (
        interactive_after["n"] > 0
        and max(p50_change, p90_change) <= _INTERACTIVE_NOISE_PCT
    )
    if interactive_after["n"] == 0:
        notes.append(
            f"No local dispatch lines carry priority={interactive_priority} in the "
            "post-deploy window: the dispatch event only gained priority= (and "
            "estimated_tokens=) with LP-0MVGHXD59008UOUD, so AC4 needs a window "
            "logged after that instrumentation is deployed."
        )
    checks.append(
        {
            "target": f"AC4: {interactive_priority}-priority first byte within {_INTERACTIVE_NOISE_PCT:.0f}% p50/p90",
            "observed_p50_pct_change": p50_change,
            "observed_p90_pct_change": p90_change,
            "before": interactive_before,
            "after": interactive_after,
            "pass": interactive_ok,
        }
    )
    if not interactive_ok and interactive_after["n"] > 0:
        overlapping = [
            f"{record['admit']['session']}@{record['admit']['ts'].isoformat(sep=' ')}"
            for record in after["_idle_records"]
            if record["dispatch"] and record["dispatch"].get("priority") == interactive_priority
        ]
        if not overlapping:
            overlapping_text = (
                "none (no same-tier admits — the regression is indirect, "
                "e.g. GPU occupancy from a lower tier)"
            )
        else:
            overlapping_text = ", ".join(overlapping)
        notes.append(
            "Interactive latency regressed beyond the noise band; idle-admitted prompts at "
            f"the SAME priority tier: {overlapping_text}."
        )

    # ---- Parent AC1 (LP-0MVK16GME007EYUW) — lease-gate shape attribution ----
    # The idle-admission decision and the dispatch lease gate must agree about
    # what "idle" means; this check makes the lease-gate cost a measured input
    # to the conversion-rate decision instead of an inference. Too few samples
    # is an explicit WARN, never a silent vacuous pass.
    lease_after = after["lease_gate"]
    lease_samples = lease_after["samples"]
    checks.append(
        {
            "target": (
                f"AC1 (LP-0MVK16GME007EYUW): lease-gate refusals attributable "
                f"(>= {min_lease_samples} samples)"
            ),
            "observed": lease_samples,
            "shape_a": lease_after["shape_a"]["count"],
            "shape_b": lease_after["shape_b"]["count"],
            "pass": lease_samples >= min_lease_samples,
        }
    )
    if lease_samples < min_lease_samples:
        notes.append(
            f"Too few lease-gate samples in the post-deploy window "
            f"({lease_samples} < {min_lease_samples}): the shape A/B split percentages "
            "are withheld and the lease-gate effect on the measured conversion rate "
            "cannot be attributed. Extend the window, or confirm that "
            "`local_dispatch_denied ... active=` and `lease_released ... reason=` "
            "lines are present in the log stream."
        )
    elif lease_after["unattributed_refusals"]:
        notes.append(
            f"{lease_after['unattributed_refusals']} refusal(s) predate the `active=` "
            "field and could not be attributed; the A/B split covers the attributable "
            f"{lease_samples} only."
        )
    else:
        shape_b = lease_after["shape_b"]
        notes.append(
            "Lease-gate refusals: "
            f"{lease_after['headline']} "
            f"(shape A = inactive cooldown lease, shape B = `active` record with no "
            f"work behind it; of the {shape_b['count']} shape-B refusals, "
            f"{shape_b['no_work_behind_it']} had no work behind it, "
            f"{shape_b['silent_prefill']} were silent prefills, "
            f"{shape_b['owner_working']} were genuinely working owners and "
            f"{shape_b['unconfirmed']} lacked corroborating evidence)."
        )

    # ---- AC5 decision ----------------------------------------------------
    fallback_before_rate = _rate(before_ac1["remote_fallback_events"], hours_before)
    fallback_after_rate = _rate(after_ac1["remote_fallback_events"], hours_after)
    fallback_change_pct = _pct_change(fallback_before_rate, fallback_after_rate)
    reduction_pct = round(-fallback_change_pct, 1)
    serve_share_change = round(
        after_ac1["local_serve_share_pct"] - before_ac1["local_serve_share_pct"], 2
    )

    recommendation: dict = {
        "decision": "measure",
        "fallback_reduction_pct": reduction_pct,
        "local_serve_share_delta_pct_points": serve_share_change,
        "reasons": [],
        "follow_up": None,
    }
    if not measured:
        recommendation["reasons"].append(
            "no idle-admitted requests observed in the post-deploy window"
        )
        recommendation["follow_up"] = (
            "Enable `server.local_idle_admit_enabled: true` (threshold "
            "`server.local_idle_admit_seconds`, default 60), commit it, make the running "
            "proxy read it with `POST /admin/reload-config` (no restart, so in-flight "
            "streams survive and the process-local idle clock is not reset), then re-run "
            "this tool with a post-deploy window of >= 24 h."
        )
    elif paired < min_idle_samples:
        recommendation["reasons"].append(
            f"only {paired} idle-admitted dispatch sample(s); need >= {min_idle_samples}"
        )
        recommendation["follow_up"] = (
            "Extend the post-deploy window (or raise the prompt-size mix) until at least "
            f"{min_idle_samples} idle-admitted dispatches are paired."
        )
    elif (
        reduction_pct >= _FALLBACK_REDUCTION_MIN_PCT
        and denied_ok
        and interactive_ok
        and not attributed
    ):
        recommendation["decision"] = "enable_by_default"
        recommendation["reasons"].append(
            f"remote fallback rate fell {reduction_pct}% with no denial increase, no "
            "attributable timeout and interactive latency within noise"
        )
        recommendation["follow_up"] = (
            "Set `server.local_idle_admit_enabled: true` as the default and record the "
            "measured numbers in the config comment."
        )
    else:
        recommendation["decision"] = "tune"
        if reduction_pct < _FALLBACK_REDUCTION_MIN_PCT:
            recommendation["reasons"].append(
                f"remote fallback reduction {reduction_pct}% is below the "
                f"{_FALLBACK_REDUCTION_MIN_PCT}% materiality bar"
            )
        if not denied_ok:
            recommendation["reasons"].append(
                f"local_dispatch_denied rose {denied_change_pct}% per hour"
            )
        if attributed:
            recommendation["reasons"].append(
                f"{len(attributed)} timeout(s) attributable to idle-admitted requests"
            )
        if not interactive_ok:
            recommendation["reasons"].append(
                f"{interactive_priority}-priority first byte moved p50 {p50_change}% / "
                f"p90 {p90_change}%"
            )
        tuning: list[str] = []
        if not interactive_ok:
            tuning.append(
                "gate the override on priority (never admit for the interactive tier) "
                "and/or cap the admitted prompt size"
            )
        if denied_ok is False:
            tuning.append(
                "raise `server.local_idle_admit_seconds` so admission only happens in "
                "longer idle gaps"
            )
        if attributed:
            tuning.append(
                "cap the admitted prompt size so an idle-admitted prefill cannot outlive "
                "the client deadline"
            )
        if not tuning:
            tuning.append(
                "raise `server.local_idle_admit_seconds` progressively and re-measure "
                "until the bypass-to-local conversion is material"
            )
        recommendation["follow_up"] = (
            "Record the tuning in a follow-up work item: " + "; ".join(tuning) + "."
        )

    passed = all(check["pass"] for check in checks)
    if passed and recommendation["decision"] == "enable_by_default":
        pass
    elif passed:
        # Metrics are complete but the decision is not an enable-by-default
        # verdict — the AC still needs the recorded follow-up.
        notes.append(
            "All measurement checks passed; AC5 resolves to a tuning follow-up "
            "(see verdict.recommendation.follow_up)."
        )

    return {
        "verdict": "PASS" if passed else "WARN",
        "checks": checks,
        "notes": notes,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value}{suffix}"


def render(meta: dict, windows: dict, verdict: dict) -> str:
    out: list[str] = []
    out.append("Idle-aware local admission impact (LP-0MVGHXD59008UOUD)")
    out.append("=" * 68)
    out.append(f"log_dir     : {meta['log_dir']}")
    out.append(f"generated_at: {meta['generated_at']}")
    for label in ("before", "after"):
        window = windows[label]
        out.append(
            f"{label:<12}: {window['start'] or '(start)'} .. {window['end'] or '(end)'} "
            f"({window['hours']:.1f} h)"
        )
    out.append("")

    out.append("AC1 — window aggregates (baseline next to post-deploy)")
    out.append("-" * 68)
    out.append(f"{'metric':<34}{'before':>15}{'after':>15}")
    for key, title in (
        ("local_idle_admit", "local_idle_admit"),
        ("local_serve_share_pct", "local serve share %"),
        ("large_context_bypass", "large_context_bypass"),
        ("remote_fallback_events", "remote fallback events"),
        ("local_dispatch_denied", "local_dispatch_denied"),
        ("local_prefill_slot_seconds", "local prefill slot-seconds"),
        ("timeout_events", "mid-stream timeout events"),
    ):
        out.append(
            f"{title:<34}{windows['before']['ac1'][key]:>15}{windows['after']['ac1'][key]:>15}"
        )
    out.append("")

    out.append("AC1 (LP-0MVK16GME007EYUW) — lease-gate shape attribution")
    out.append("-" * 68)
    for label in ("before", "after"):
        lease = windows[label]["lease_gate"]
        released = windows[label]["lease_released"]
        status = windows[label]["lease_status"]
        out.append(
            f"[{label}] refusals={lease['refusals']} (cold-start {lease['cold_start_refusals']}) "
            f"attributable={lease['samples']} status={lease['attribution_status']}"
        )
        if lease["samples"]:
            def _pct_text(shape: dict) -> str:
                return "n/a" if shape["pct"] is None else f"{shape['pct']}%"

            out.append(
                f"  shape A={lease['shape_a']['count']} "
                f"({_pct_text(lease['shape_a'])}) "
                f"shape B={lease['shape_b']['count']} "
                f"({_pct_text(lease['shape_b'])})  "
                f"B: no-work={lease['shape_b']['no_work_behind_it']} "
                f"prefill={lease['shape_b']['silent_prefill']} "
                f"working={lease['shape_b']['owner_working']} "
                f"unconfirmed={lease['shape_b']['unconfirmed']}"
            )
            by_evidence = ", ".join(
                f"{key}={value}" for key, value in lease["shape_b"]["by_evidence"].items()
            )
            out.append(f"    B evidence: {by_evidence}")
        release_text = ", ".join(
            f"{key}={value}" for key, value in released["by_group"].items()
        ) or "none"
        out.append(f"  lease_released: {release_text}")
        out.append(
            f"  contention queue: dispatch={lease['contention_queue']['dispatch']} "
            f"fallback_after_queue={lease['contention_queue']['fallback_after_queue']}  "
            f"status samples={status['samples']} "
            f"(owner-active-query=false {status['local_active_query_false']}, "
            f"lease-remaining p50/max {status['owner_lease_remaining_seconds_p50']}/"
            f"{status['owner_lease_remaining_seconds_max']} s)"
        )
        if lease["samples"] and lease["attribution_status"] == "ok":
            out.append(f"  {lease['headline']}")
    out.append("")

    out.append("AC2 — idle-admitted vs non-idle local dispatches (same bucket)")
    out.append("-" * 68)
    for label in ("before", "after"):
        ac2 = windows[label]["ac2"]
        counts = ac2["idle_admitted_sample_counts"]
        out.append(
            f"[{label}] idle_admit={counts['idle_admit_events']} "
            f"paired={counts['paired_local_dispatches']} "
            f"unpaired={counts['unpaired_admits']}"
        )
        out.append(
            f"  idle    first-byte n={ac2['idle_admitted_first_byte_ms']['n']:<5} "
            f"p50={ac2['idle_admitted_first_byte_ms']['p50_ms']:>10.1f} "
            f"p90={ac2['idle_admitted_first_byte_ms']['p90_ms']:>10.1f} ms  "
            f"slot-s={ac2['idle_admitted_service_ms']['sum_slot_seconds']}"
        )
        out.append(
            f"  non-idle first-byte n={ac2['non_idle_first_byte_ms']['n']:<5} "
            f"p50={ac2['non_idle_first_byte_ms']['p50_ms']:>10.1f} "
            f"p90={ac2['non_idle_first_byte_ms']['p90_ms']:>10.1f} ms  "
            f"slot-s={ac2['non_idle_service_ms']['sum_slot_seconds']}"
        )
        for bucket, data in ac2["by_bucket"].items():
            out.append(
                f"  bucket {bucket:<12} idle n={data['idle']['n']:<4} "
                f"(p50 {data['idle']['service_ms']['p50_ms']:.1f} ms, "
                f"{data['idle']['service_ms']['sum_slot_seconds']} slot-s) | "
                f"non-idle n={data['non_idle']['n']:<4} "
                f"(p50 {data['non_idle']['service_ms']['p50_ms']:.1f} ms, "
                f"{data['non_idle']['service_ms']['sum_slot_seconds']} slot-s)"
            )
    out.append("")

    out.append("AC3 — denials and timeout attribution")
    out.append("-" * 68)
    for label in ("before", "after"):
        ac3 = windows[label]["ac3"]
        out.append(
            f"[{label}] denied={ac3['local_dispatch_denied']} "
            f"timeouts={ac3['timeout_events']} "
            f"attributed={len(ac3['timeouts_attributed_to_idle_admit'])} "
            f"coincident={len(ac3['timeouts_coincident_with_idle_admit'])}"
        )
    out.append("")

    out.append(
        f"AC4 — interactive ({windows['after']['ac4']['interactive_priority'] or 'high'}) "
        "first-byte latency"
    )
    out.append("-" * 68)
    for label in ("before", "after"):
        dist = windows[label]["ac4"].get("interactive", _dist_stats([]))
        out.append(
            f"[{label}] n={dist['n']:<5} p50={dist['p50_ms']:>10.1f} ms "
            f"p90={dist['p90_ms']:>10.1f} ms"
        )
    out.append("")

    cf = windows["after"].get("counterfactual", {})
    if cf:
        out.append("Counterfactual (APPROXIMATION — override eligibility from logs)")
        out.append("-" * 68)
        out.append(
            f"bypasses={cf.get('measured_bypasses')} "
            f"eligible(>= {cf.get('idle_threshold_seconds')}s idle)={cf.get('eligible_bypasses')} "
            f"({cf.get('eligible_pct_of_bypasses')}%) "
            f"idle-gap p50/p90={cf.get('idle_gap_p50_seconds')}/{cf.get('idle_gap_p90_seconds')} s"
        )
        out.append(f"note: {cf.get('note')}")
        out.append("")

    out.append(f"VERDICT: {verdict['verdict']}")
    for check in verdict["checks"]:
        observed = check.get("observed", check.get("observed_pct_change", check.get("observed_hours", "n/a")))
        out.append(f"  [{'ok  ' if check['pass'] else 'FAIL'}] {check['target']} (observed {observed})")
    for note in verdict["notes"]:
        out.append(f"  note: {note}")
    rec = verdict["recommendation"]
    out.append("")
    out.append(f"AC5 recommendation: {rec['decision']}")
    for reason in rec["reasons"]:
        out.append(f"  - {reason}")
    if rec.get("follow_up"):
        out.append(f"  follow-up: {rec['follow_up']}")
    return "\n".join(out)


def render_markdown(meta: dict, windows: dict, verdict: dict) -> str:
    """Compact markdown section for a work-item comment or docs artifact."""
    out: list[str] = []
    out.append("### Idle-aware local admission impact (LP-0MVGHXD59008UOUD)")
    out.append("")
    out.append(
        f"- baseline: {windows['before']['start']} → {windows['before']['end']} "
        f"({windows['before']['hours']:.1f} h)"
    )
    out.append(
        f"- post-deploy: {windows['after']['start']} → {windows['after']['end']} "
        f"({windows['after']['hours']:.1f} h)"
    )
    out.append("")
    out.append("| Metric | Baseline | Post-deploy |")
    out.append("|---|---:|---:|")
    for key, title in (
        ("local_idle_admit", "`local_idle_admit`"),
        ("local_serve_share_pct", "Local serve share %"),
        ("large_context_bypass", "`large_context_bypass`"),
        ("remote_fallback_events", "Remote fallback events"),
        ("local_dispatch_denied", "`local_dispatch_denied`"),
        ("local_prefill_slot_seconds", "Local prefill slot-seconds"),
        ("timeout_events", "Mid-stream timeouts"),
    ):
        out.append(
            f"| {title} | {windows['before']['ac1'][key]} | {windows['after']['ac1'][key]} |"
        )
    out.append("")
    out.append("| Bucket | idle n | idle p50 (ms) | idle slot-s | non-idle n | non-idle p50 (ms) | non-idle slot-s |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for bucket, data in windows["after"]["ac2"]["by_bucket"].items():
        out.append(
            f"| {bucket} | {data['idle']['n']} | {data['idle']['service_ms']['p50_ms']} | "
            f"{data['idle']['service_ms']['sum_slot_seconds']} | {data['non_idle']['n']} | "
            f"{data['non_idle']['service_ms']['p50_ms']} | "
            f"{data['non_idle']['service_ms']['sum_slot_seconds']} |"
        )
    out.append("")

    # Lease-gate shape attribution (LP-0MVK35HBI003SB3Z — parent LP-0MVK16GME007EYUW AC1).
    after_lease = windows["after"]["lease_gate"]
    out.append("| Lease-gate metric | Baseline | Post-deploy |")
    out.append("|---|---:|---:|")
    for key, title in (
        ("refusals", "`local_lease_active` refusals"),
        ("cold_start_refusals", "Cold-start denials (excluded)"),
        ("samples", "Attributable refusal samples"),
        ("attribution_status", "Attribution status"),
    ):
        out.append(
            f"| {title} | {windows['before']['lease_gate'][key]} | {after_lease[key]} |"
        )
    for shape_key, title in (
        ("shape_a", "Shape A (inactive cooldown lease)"),
        ("shape_b", "Shape B (`active` record, no work behind it)"),
    ):
        before_shape = windows["before"]["lease_gate"][shape_key]
        after_shape = after_lease[shape_key]
        before_cell = (
            f"{before_shape['count']} ({before_shape['pct']}%)"
            if before_shape["pct"] is not None
            else f"{before_shape['count']} (insufficient)"
        )
        after_cell = (
            f"{after_shape['count']} ({after_shape['pct']}%)"
            if after_shape["pct"] is not None
            else f"{after_shape['count']} (insufficient)"
        )
        out.append(f"| {title} | {before_cell} | {after_cell} |")
    for key, title in (
        ("no_work_behind_it", "— Shape B: no work behind it"),
        ("silent_prefill", "— Shape B: silent prefill"),
        ("owner_working", "— Shape B: genuinely working owner"),
        ("unconfirmed", "— Shape B: unconfirmed"),
    ):
        out.append(
            f"| {title} | {windows['before']['lease_gate']['shape_b'][key]} | "
            f"{after_lease['shape_b'][key]} |"
        )
    groups_before = windows["before"]["lease_released"]["by_group"]
    groups_after = windows["after"]["lease_released"]["by_group"]
    for group in ("no_progress_watchdog", "adaptive_expiry", "explicit_release"):
        out.append(
            f"| `lease_released` reason group: {group} | {groups_before.get(group, 0)} | "
            f"{groups_after.get(group, 0)} |"
        )
    queue_before = windows["before"]["lease_gate"]["contention_queue"]
    queue_after = after_lease["contention_queue"]
    out.append(
        f"| Contention queue: converted to local | {queue_before['dispatch']} | "
        f"{queue_after['dispatch']} |"
    )
    out.append(
        f"| Contention queue: fell through to remote | {queue_before['fallback_after_queue']} | "
        f"{queue_after['fallback_after_queue']} |"
    )
    if after_lease["attribution_status"] == "ok":
        out.append("")
        out.append(f"**Lease-gate answer:** {after_lease['headline']}")
    else:
        out.append("")
        out.append(
            f"**Lease-gate answer:** too few samples (`{after_lease['samples']} < "
            f"{after_lease['min_samples']}`) — split withheld, do not read the A/B "
            "counts as a confident attribution."
        )
    out.append("")
    out.append(
        f"Interactive ({windows['after']['ac4']['interactive_priority'] or 'high'}) first byte: "
        f"baseline p50 {windows['before']['ac4'].get('interactive', _dist_stats([]))['p50_ms']} ms / "
        f"post p50 {windows['after']['ac4'].get('interactive', _dist_stats([]))['p50_ms']} ms."
    )
    out.append("")
    out.append(f"**Verdict:** `{verdict['verdict']}` — AC5 recommendation "
               f"`{verdict['recommendation']['decision']}`.")
    for reason in verdict["recommendation"]["reasons"]:
        out.append(f"- {reason}")
    if verdict["recommendation"].get("follow_up"):
        out.append(f"- Follow-up: {verdict['recommendation']['follow_up']}")
    out.append("")
    out.append(f"_Generated by `scripts/idle_admit_impact.py` at {meta['generated_at']}._")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _derive_windows(
    *,
    before_start: dt.datetime | None,
    before_end: dt.datetime | None,
    after_start: dt.datetime | None,
    after_end: dt.datetime | None,
    hours: float,
    now: dt.datetime,
) -> tuple[tuple[dt.datetime, dt.datetime], tuple[dt.datetime, dt.datetime]]:
    """Resolve the two windows, defaulting to the last N h vs the N h before."""
    after_end = after_end or now
    after_start = after_start or (after_end - dt.timedelta(hours=hours))
    before_end = before_end or after_start
    before_start = before_start or (before_end - dt.timedelta(hours=hours))
    if before_end > after_start and before_end != before_start:
        # Overlapping windows would compare traffic with itself.
        raise ValueError(
            "baseline window must end at or before the post-deploy window starts "
            f"(baseline ends {before_end}, post-deploy starts {after_start})"
        )
    if before_start >= before_end:
        raise ValueError("baseline window start must be before its end")
    if after_start >= after_end:
        raise ValueError("post-deploy window start must be before its end")
    return (before_start, before_end), (after_start, after_end)


def _build_meta(args, log_dir, before_window, after_window, interactive_priority, deployed_at) -> dict:
    return {
        "tool": "scripts/idle_admit_impact.py",
        "work_item": "LP-0MVGHXD59008UOUD",
        "feature_work_item": "LP-0MVGD9Z8Z0055Y5J",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "log_dir": str(log_dir),
        "files": [p.name for p in discover_proxy_log_files(Path(log_dir))],
        "before": {"start": before_window[0].isoformat(sep=" "), "end": before_window[1].isoformat(sep=" ")},
        "after": {"start": after_window[0].isoformat(sep=" "), "end": after_window[1].isoformat(sep=" ")},
        "deployed_at": deployed_at.isoformat(sep=" ") if deployed_at else None,
        "label": args.label,
        "notes": args.notes,
        "interactive_priority": interactive_priority,
        "correlation_window_seconds": args.correlation_window,
        "idle_threshold_seconds": args.idle_threshold,
        "min_lease_samples": args.min_lease_samples,
        "bucket_edges": args.bucket_edges_list,
        "window_closed": args.window_closed,
        "lease_gate_work_item": "LP-0MVK35HBI003SB3Z",
        "lease_gate_parent_work_item": "LP-0MVK16GME007EYUW",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", default="/var/log/llama-proxy")
    parser.add_argument("--before-start", type=_parse_bound, default=None, help="baseline window start")
    parser.add_argument("--before-end", type=_parse_bound, default=None, help="baseline window end (exclusive)")
    parser.add_argument("--after-start", type=_parse_bound, default=None, help="post-deploy window start")
    parser.add_argument("--after-end", type=_parse_bound, default=None, help="post-deploy window end (exclusive)")
    parser.add_argument(
        "--hours",
        type=float,
        default=24.0,
        help="window length used when boundaries are omitted (default 24)",
    )
    parser.add_argument(
        "--deployed-at",
        type=_parse_bound,
        default=None,
        help="deployment time; the post-deploy window must start at/after it",
    )
    parser.add_argument(
        "--correlation-window",
        type=float,
        default=_DEFAULT_CORRELATION_WINDOW_S,
        help="session correlation window in seconds (default 60)",
    )
    parser.add_argument(
        "--idle-threshold",
        type=float,
        default=_DEFAULT_IDLE_THRESHOLD_S,
        help="idle threshold used by the counterfactual estimator (default 60)",
    )
    parser.add_argument(
        "--min-lease-samples",
        type=int,
        default=_MIN_LEASE_SAMPLES,
        dest="min_lease_samples",
        help=(
            "minimum attributable lease-gate refusals before the shape A/B split is "
            f"reported (default {_MIN_LEASE_SAMPLES}); below it the report WARNs"
        ),
    )
    parser.add_argument(
        "--interactive-priority",
        default="high",
        help="priority tier treated as interactive (default high)",
    )
    parser.add_argument(
        "--bucket-edges",
        default=",".join(str(edge) for edge in _DEFAULT_BUCKET_EDGES),
        help="comma-separated prompt-token bucket edges",
    )
    parser.add_argument("--label", default=None, help="free-form label recorded in meta")
    parser.add_argument("--notes", default=None, help="free-form notes recorded in meta")
    parser.add_argument(
        "--window-closed",
        action="store_true",
        help="assert the post-deploy window has fully elapsed (>= 24 h)",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--markdown", action="store_true", help="emit a markdown section")
    args = parser.parse_args(argv)

    try:
        args.bucket_edges_list = tuple(
            int(part) for part in str(args.bucket_edges).split(",") if part.strip()
        )
    except ValueError:
        print(f"error: invalid --bucket-edges {args.bucket_edges!r}", file=sys.stderr)
        return 1
    if not args.bucket_edges_list:
        print("error: --bucket-edges must contain at least one edge", file=sys.stderr)
        return 1

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"error: log directory not found: {log_dir}", file=sys.stderr)
        return 1
    if not discover_proxy_log_files(log_dir):
        print(f"error: no proxy logs found in {log_dir}", file=sys.stderr)
        return 1

    try:
        before_window, after_window = _derive_windows(
            before_start=args.before_start,
            before_end=args.before_end,
            after_start=args.after_start,
            after_end=args.after_end,
            hours=args.hours,
            now=dt.datetime.now(),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    windows: dict[str, dict] = {}
    for label, (start, end) in (("before", before_window), ("after", after_window)):
        events = collect_events(log_dir, since=start, until=end)
        summary = summarise_window(
            events,
            bucket_edges=args.bucket_edges_list,
            correlation_window_s=args.correlation_window,
            min_lease_samples=args.min_lease_samples,
        )
        summary["counterfactual"] = counterfactual(summary, idle_threshold_s=args.idle_threshold)
        summary["start"] = start.isoformat(sep=" ")
        summary["end"] = end.isoformat(sep=" ")
        summary["hours"] = round(_window_length_hours(start, end), 2)
        windows[label] = summary

    # AC4 comparison is computed once both windows are known.
    for label in ("before", "after"):
        ac4 = windows[label]["ac4"]
        ac4["interactive_priority"] = args.interactive_priority
        ac4["interactive"] = ac4["by_priority"].get(args.interactive_priority, {}).get(
            "first_byte_ms", _dist_stats([])
        )

    verdict = evaluate(
        windows["before"],
        windows["after"],
        hours_before=windows["before"]["hours"],
        hours_after=windows["after"]["hours"],
        interactive_priority=args.interactive_priority,
        window_closed=args.window_closed,
        deployed_at=args.deployed_at,
        after_start=after_window[0],
        min_lease_samples=args.min_lease_samples,
    )
    meta = _build_meta(
        args, log_dir, before_window, after_window, args.interactive_priority, args.deployed_at
    )

    if args.json:
        payload = {
            "meta": meta,
            "windows": {
                label: {key: value for key, value in window.items() if not key.startswith("_")}
                for label, window in windows.items()
            },
            "verdict": verdict,
        }
        print(json.dumps(payload, indent=2))
    elif args.markdown:
        print(render_markdown(meta, windows, verdict))
    else:
        print(render(meta, windows, verdict))
    return 0 if verdict["verdict"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
