"""Prometheus-based GPU busy-time measurement (coverage-honest).

GPU busy time is measured from power draw via the ``gpu:computing:bool``
recording rule (1 ≥ 90 W, 0 otherwise), evaluated every 15 s by Prometheus.

The driver's ``gpu_busy_percent`` cannot serve as a utilisation signal on this
Strix Halo APU: it is a **two-state detector, not a percentage**, whose step
sits at **~30 W** — it reads 1–10 % below 30 W and 92–99 % from 30 W upward.
Because the step is below the idle-with-display band (p50 45 W), idle-with-
context and computing both read ~99 % and the counter *inverts* across the band
that matters: re-measured 2026-11-06 on kernel 7.0.0-38 it read 99.2 % at
40–60 W (idle with a display) against 98.7 % at 90–100 W (computing), with
7.9 % false-busy and 1.7 % false-idle over the whole post-boot window
(LP-0MW5K7WKZ005VJXA; supersedes the retired 20 W latch claim of
LP-0MS9GA6QL007XTBP and LP-0MVSQQO910077LYL). AMD's own tooling offers no
alternative here: ``amd-smi``
reports ``N/A`` for every metric this module needs on this APU, and
``amdtop``/``rocm-smi`` merely display the same counter — so this
module reads Prometheus and must never shell out to an AMD tool (see
``monitoring/README.md``, "Why not amd-smi, rocm-smi or amdtop?", for the
measurement and the PromQL recipe that recomputes every band).

The recording rules live in monitoring/gpu_compute_rules.yaml and are
evaluated at ``evaluation_interval = 15 s``.  The 15 s cadence is
comfortably finer than the workload timescales (median decode 2.4 s, worst
prefill 108 s), so sub-15-second sampling adds no value.

Coverage, and why absence is never idle
---------------------------------------
The naive rolling-window average over whatever samples happen to be present
renormalises coverage away, so a sampling gap reads as "not applicable" and
*inflates* the busy fraction — a window where the sampler was down for 38 h
reported 51.3 % against a grid-honest 37.9 % for the same store and window
(51.3 % × 0.74 observed coverage; LP-0MW5K8YE9001QO39). This module
therefore classifies **every expected instant on a grid over the window**
instead: ``floor(window / step) + 1`` instants at ``step = 15 s`` starting at
the window start. Each instant is one of

- ``computing`` — the ``gpu:computing:bool`` rule has a sample ``≥ 0.5`` and
  the pipeline was demonstrably alive;
- ``idle`` — the rule has a sample ``0`` and the pipeline was demonstrably
  alive;
- ``unknown`` — everything else. Absence is never folded into ``idle`` and
  never into the numerator.

An instant is **observed** only when a sentinel proves the sampling pipeline
was alive at that instant — ``up{job="llama-proxy"} == 1``,
``up{job="node-exporter"} == 1``, or a
``llama_slots_sample_timestamp_seconds`` heartbeat no older than one scrape
interval — **and** the ``gpu:computing:bool`` rule actually has a sample
there. A stale heartbeat proves the sampler stopped reporting, so it does not
count.

By construction the three shares cover the window exactly:
``computing_seconds + idle_seconds + unknown_seconds == window_seconds`` and
``shares_pct`` sums to 100 %. Each grid instant has duration
``window_seconds / expected_samples``, so the seconds identity holds for the
grid the shares are computed on.

A window with **no observed instants at all** returns ``None`` — the pre-fix
"no row" behaviour is kept: the report omits the GPU row and ``--json``'s
``gpu_busy`` is ``null``, never a fabricated ``0 %``. A partially observed
window keeps the row and reports its unobserved share as ``unknown``.

What this measures — and what it is not
---------------------------------------
The decision figure for local model utilisation is the report's log-based
``local_busy`` (``BusyStats``, ``dispatch → finish``): it measures *slot
occupancy* from proxy events, i.e. the work the proxy actually served. This
module measures something different — when the *hardware* actually drew
compute-level power, from the 15 s hwmon series — independent of proxy
logging gaps and of how the proxy accounts for its own slots.

That makes this **supplementary hardware telemetry, not a local-utilisation
signal**: ``gpu:computing:bool`` counts any GPU consumer (other workloads,
non-local requests) and misses proxy-served work during log gaps, so it can
read far higher or lower than ``local_busy``. Use it for hardware questions —
was the GPU computing at all? what did it draw? — never as the local busy
figure.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

# Prometheus instance — loopback only; the monitoring README documents the
# exposure boundary (tailnet / loopback, no auth).
PROMETHEUS_URL = "http://127.0.0.1:9090"

# HTTP timeout for each Prometheus call (seconds).
_TIMEOUT_SECONDS = 5

#: Grid resolution and the recording rules' evaluation interval (15 s).
STEP_SECONDS = 15

#: Freshness bound for the slot heartbeat: a sample older than one scrape
#: interval proves the sampler stopped reporting, so the instant is unknown.
SCRAPE_INTERVAL_SECONDS = 15

#: Query expressions this module issues for one window.
BOOL_QUERY = "gpu:computing:bool"
UP_LLAMA_PROXY_QUERY = 'up{job="llama-proxy"}'
UP_NODE_EXPORTER_QUERY = 'up{job="node-exporter"}'
HEARTBEAT_QUERY = "llama_slots_sample_timestamp_seconds"
POWER_QUERY = "gpu:power_watts:avg5m"


def _share_pct(seconds: float, total_seconds: float) -> float:
    """Percentage *seconds* is of *total_seconds* (``0`` for a non-positive total)."""
    if total_seconds <= 0:
        return 0.0
    return seconds / total_seconds * 100.0


@dataclass
class GpuBusyResult:
    """GPU busy-time measured from power consumption, with explicit coverage.

    ``computing_fraction`` is the share of the window's **expected** grid
    instants on which the GPU drew ≥ 90 W (the recording-rule threshold that
    matches the proxy's wedge detector). It is computed over the full grid, so
    an unobserved instant lowers the fraction as ``unknown`` rather than
    renormalising it upward. ``busy_seconds`` (the ``computing`` share) plus
    ``idle_seconds`` plus ``unknown_seconds`` equals ``total_seconds`` by
    construction.

    This is supplementary hardware telemetry, **not** a local-utilisation
    signal: ``gpu:computing:bool`` counts any GPU consumer. Decisions use the
    log-derived ``local_busy`` (``dispatch → finish``) slot occupancy, of which
    this is a hardware-side complement.

    ``smoothed_power_values`` carries the ``gpu:power_watts:avg5m`` trace
    (1-min samples, derived from the 15-s raw series via a 5-min rolling
    average) over the same window; useful for identifying warm-up and
    cooldown transitions.
    """

    computing_fraction: float  # 0.0–1.0, over the expected grid
    total_seconds: float  # window length in seconds
    busy_seconds: float  # computing share of total_seconds
    idle_seconds: float  # idle share of total_seconds
    smoothed_power_values: list[tuple[float, float]]  # (unix_ts, avg5m_watts)
    unknown_seconds: float = 0.0  # unobserved share of total_seconds
    expected_samples: int = 0  # grid instants the window should contain
    observed_samples: int = 0  # computing + idle
    computing_samples: int = 0
    idle_samples: int = 0
    unknown_samples: int = 0

    @property
    def busy_pct(self) -> float:
        """The ``computing`` share of the window, in percent."""
        return self.computing_fraction * 100.0

    @property
    def idle_pct(self) -> float:
        """The ``idle`` share of the window, in percent."""
        return _share_pct(self.idle_seconds, self.total_seconds)

    @property
    def unknown_pct(self) -> float:
        """The ``unknown`` share of the window, in percent.

        Derived as ``100 − computing − idle`` so the three shares sum to 100 %
        by construction.
        """
        return 100.0 - self.busy_pct - self.idle_pct

    @property
    def shares_pct(self) -> dict[str, float]:
        """The three coverage shares (``computing`` / ``idle`` / ``unknown``)."""
        return {
            "computing": self.busy_pct,
            "idle": self.idle_pct,
            "unknown": self.unknown_pct,
        }


def _get_json(url: str) -> dict | None:
    """GET *url* and parse JSON; ``None`` on any network/parse error."""
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def query_prometheus(
    query: str,
    start: int,
    end: int,
    step: str = "60",
) -> dict | None:
    """Fire an instant query (``start == end``) or range query to Prometheus.

    Returns the raw JSON dict, or ``None`` on any network or parse error
    (best-effort: a report run must never fail because Prometheus is down).
    """
    if start == end:
        endpoint = "query"
        url = (
            f"{PROMETHEUS_URL}/api/v1/{endpoint}"
            f"?query={urllib.request.quote(query)}"
            f"&time={end}"
        )
    else:
        endpoint = "query_range"
        url = (
            f"{PROMETHEUS_URL}/api/v1/{endpoint}"
            f"?query={urllib.request.quote(query)}"
            f"&start={start}"
            f"&end={end}"
            f"&step={step}"
        )
    return _get_json(url)


def _parse_matrix(payload: dict | None) -> dict[int, float]:
    """Flatten a ``query_range`` matrix payload into ``{unix_ts: value}``.

    Multiple series are collapsed by their maximum: this host exposes one GPU
    and one llama-proxy, and "any target is up" is the honest reduction for a
    sentinel. Malformed samples are skipped rather than guessed.
    """
    points: dict[int, float] = {}
    if not isinstance(payload, dict):
        return points
    result = payload.get("data", {}).get("result")
    for series in result or []:
        for sample in series.get("values") or []:
            try:
                timestamp = int(float(sample[0]))
                value = float(sample[1])
            except (TypeError, ValueError, IndexError):
                continue
            previous = points.get(timestamp)
            points[timestamp] = value if previous is None else max(previous, value)
    return points


def _expected_sample_count(start: int, end: int, step: int) -> int:
    """Number of grid instants a coverage-honest window must account for."""
    return (end - start) // step + 1


def _align_to_grid(
    points: dict[int, float], *, start: int, step: int, expected: int
) -> dict[int, float]:
    """Snap samples onto the query grid, dropping anything out of the window.

    Prometheus returns samples on the requested grid, but a live store can hand
    back timestamps a fraction of a step off. A sample within half a step of a
    grid instant is attributed to it; anything further away, and anything out
    of the window, is dropped rather than silently shifting the coverage
    arithmetic.
    """
    tolerance = step / 2
    aligned: dict[int, float] = {}
    for raw_timestamp, value in points.items():
        index = round((raw_timestamp - start) / step)
        if index < 0 or index >= expected:
            continue
        grid_timestamp = start + index * step
        if abs(raw_timestamp - grid_timestamp) > tolerance:
            continue
        previous = aligned.get(grid_timestamp)
        aligned[grid_timestamp] = value if previous is None else max(previous, value)
    return aligned


def compute_gpu_busy(
    window_start: float,  # unix timestamp
    window_end: float,    # unix timestamp
) -> GpuBusyResult | None:
    """Compute coverage-honest GPU busy time from power consumption.

    Every expected grid instant is classified ``computing``, ``idle`` or
    ``unknown`` (see the module docstring). Returns ``None`` when Prometheus is
    unreachable, when the queries yield no usable result, or when the window
    has **no observed instants at all** (a fully-absent window keeps the
    pre-fix "no row" behaviour rather than reporting a fabricated ``0 %``).
    """
    total = window_end - window_start
    if total <= 0:
        return None

    start = int(round(window_start))
    end = int(round(window_end))
    if end <= start:
        return None
    step = STEP_SECONDS
    expected = _expected_sample_count(start, end, step)

    # 1. Every series in one query_range call, on the same window.
    bool_data = query_prometheus(BOOL_QUERY, start, end, step=str(step))
    if bool_data is None:
        return None
    up_llama = query_prometheus(UP_LLAMA_PROXY_QUERY, start, end, step=str(step))
    up_node = query_prometheus(UP_NODE_EXPORTER_QUERY, start, end, step=str(step))
    heartbeat = query_prometheus(HEARTBEAT_QUERY, start, end, step=str(step))

    computing_bool = _align_to_grid(
        _parse_matrix(bool_data), start=start, step=step, expected=expected
    )
    up_llama_points = _align_to_grid(
        _parse_matrix(up_llama), start=start, step=step, expected=expected
    )
    up_node_points = _align_to_grid(
        _parse_matrix(up_node), start=start, step=step, expected=expected
    )
    heartbeat_points = _align_to_grid(
        _parse_matrix(heartbeat), start=start, step=step, expected=expected
    )

    # 2. Classify every expected instant on the grid.
    computing = idle = 0
    for index in range(expected):
        timestamp = start + index * step
        flag = computing_bool.get(timestamp)
        alive = (
            up_llama_points.get(timestamp) == 1
            or up_node_points.get(timestamp) == 1
        )
        if not alive:
            beat = heartbeat_points.get(timestamp)
            alive = beat is not None and abs(timestamp - beat) <= SCRAPE_INTERVAL_SECONDS
        if not alive or flag is None:
            continue  # unknown: absence is never idle and never the numerator
        if flag >= 0.5:
            computing += 1
        else:
            idle += 1

    observed = computing + idle
    unknown = expected - observed
    if observed == 0:
        # Fully-absent window: keep the "no row" behaviour, never 0 %.
        return None

    # Each grid instant stands for window_seconds / expected_samples seconds,
    # so the three shares account for the whole window by construction.
    instant_seconds = total / expected
    computing_fraction = computing / expected

    # 3. Smoothed power trace (1-min samples) for the transition context.
    power_data = query_prometheus(POWER_QUERY, start, end, step="60")
    power_values = sorted(_parse_matrix(power_data).items())

    return GpuBusyResult(
        computing_fraction=computing_fraction,
        total_seconds=total,
        busy_seconds=computing_fraction * total,
        idle_seconds=idle * instant_seconds,
        smoothed_power_values=power_values,
        unknown_seconds=unknown * instant_seconds,
        expected_samples=expected,
        observed_samples=observed,
        computing_samples=computing,
        idle_samples=idle,
        unknown_samples=unknown,
    )
