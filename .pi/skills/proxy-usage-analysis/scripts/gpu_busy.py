"""Prometheus-based GPU busy-time measurement.

GPU busy time is measured from power draw via the ``gpu:computing:bool``
recording rule (1 ≥ 90 W, 0 otherwise), evaluated every 15 s by Prometheus.

The driver's ``gpu_busy_percent`` cannot serve as a utilisation signal on this
Strix Halo APU: it responds to load within seconds but **saturates at ~20 W**
(reads ~100 % for any draw above ~20 W), so idle-with-context stretches report
as busy. Measured 2026-10-28 over 7 days: 94.5 % mean busy against 28.1 % of
samples at compute power, 45.2 % false-busy (LP-0MS9GA6QL007XTBP, quantified in
LP-0MVSQQO910077LYL). AMD's own tooling offers no alternative here: ``amd-smi``
reports ``N/A`` for every metric this module needs on this APU, and
``amdtop``/``rocm-smi`` merely display the same saturating counter — so this
module reads Prometheus and must never shell out to an AMD tool (see
``monitoring/README.md``, "Why not amd-smi, rocm-smi or amdtop?").

The recording rules live in monitoring/gpu_compute_rules.yaml and are
evaluated at ``evaluation_interval = 15 s``.  The 15 s cadence is
comfortably finer than the workload timescales (median decode 2.4 s, worst
prefill 108 s), so sub-15-second sampling adds no value.

Why this is more accurate than the log-derived busy figure
----------------------------------------------------------
The report's log-based ``BusyStats`` measures *slot occupancy* from proxy
events (``dispatch → finish``), which includes time queued and excludes any
GPU work the proxy did not log.  The power-based fraction here measures when
the *hardware* actually drew compute-level power, from the 15 s hwmon series,
so it is independent of proxy logging gaps and of how the proxy accounts for
its own slots.
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


@dataclass
class GpuBusyResult:
    """GPU busy-time measured from power consumption.

    The ``computing_fraction`` is ``avg_over_time(gpu:computing:bool[Nh])`` —
    the fraction of the window during which the GPU drew ≥ 90 W (the
    recording-rule threshold that matches the proxy's wedge detector).
    It is the authoritative hardware-side busy measure, complementing the
    log-based ``BusyStats`` that measures slot occupancy from proxy logs.

    ``smoothed_power_values`` carries the ``gpu:power_watts:avg5m`` trace
    (1-min samples, derived from the 15-s raw series via a 5-min rolling
    average) over the same window; useful for identifying warm-up and
    cooldown transitions.
    """
    computing_fraction: float  # 0.0–1.0
    total_seconds: float  # window length in seconds
    busy_seconds: float  # computing_fraction * total_seconds
    idle_seconds: float
    smoothed_power_values: list[tuple[float, float]]  # (unix_ts, avg5m_watts)

    @property
    def busy_pct(self) -> float:
        return self.computing_fraction * 100.0

    @property
    def idle_pct(self) -> float:
        return (1.0 - self.computing_fraction) * 100.0


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


def compute_gpu_busy(
    window_start: float,  # unix timestamp
    window_end: float,    # unix timestamp
) -> GpuBusyResult | None:
    """Compute GPU busy-time from power consumption over a window.

    Returns ``None`` when Prometheus is unreachable or the query yields no
    result (the report then omits the GPU-busy row rather than guessing).
    """
    total = window_end - window_start
    if total <= 0:
        return None

    # 1. Computing fraction: avg_over_time(gpu:computing:bool[Nh]).
    #    An instant query at the window end; the rolling average is evaluated
    #    over the raw 15 s series, so the result already reflects the 15 s
    #    sampling. Round the lookback up to the nearest whole hour (Prometheus
    #    requires an integer-hour duration in the rule's query only for the
    #    documented examples; a fractional duration is valid too, but whole
    #    hours match how the report describes the window).
    hours = max(1, round(total / 3600))
    bool_query = f"avg_over_time(gpu:computing:bool[{hours}h])"
    bool_data = query_prometheus(bool_query, int(window_end), int(window_end))
    if bool_data is None:
        return None

    try:
        result = bool_data["data"]["result"]
        # Instant query → scalar value; range query → list of values. Accept
        # both so the helper is robust to either endpoint.
        if not result:
            return None
        if "value" in result[0]:
            fraction = float(result[0]["value"][1])
        else:
            fraction = float(result[0]["values"][-1][1])
    except (KeyError, IndexError, ValueError, TypeError):
        return None

    if not 0.0 <= fraction <= 1.0:
        return None

    # 2. Smoothed power trace (1-min samples) for the transition context.
    power_data = query_prometheus(
        "gpu:power_watts:avg5m",
        int(window_start),
        int(window_end),
        step="60",
    )
    power_values: list[tuple[float, float]] = []
    if power_data and power_data.get("data", {}).get("result"):
        for v in power_data["data"]["result"][0].get("values", []):
            power_values.append((float(v[0]), float(v[1])))

    return GpuBusyResult(
        computing_fraction=fraction,
        total_seconds=total,
        busy_seconds=fraction * total,
        idle_seconds=(1.0 - fraction) * total,
        smoothed_power_values=power_values,
    )
