"""Attributable GPU-activity share for the usage report (single producer).

The report reports **one** GPU-activity figure, and it is the attributable
share the doctrine designates: ``gpu:computing_workload:bool`` — the GPU drew
compute-level power (≥ 90 W) **and** at least one local llama-server slot was
processing. That is *our* local model's work, not any GPU consumer's, so it is
the only power-derived figure that answers "how much of the window did our
local model keep the GPU computing?".

Single-producer contract (LP-0MW5K90EQ001W0R1 AC2, LP-0MWG7KQVZ008DBZM)
-----------------------------------------------------------------------
``scripts/gpu_utilisation.py`` is the **only** surface allowed to turn
``gpu:computing_workload:bool`` into a busy figure. This module therefore
**imports that producer** and reuses its series loading, its grid-honest
classification and its per-hour bucketing; it performs no PromQL aggregation of
its own (a second derivation is a guard failure — see
``proxy/tests/test_monitoring_deployment.py::TestAttributableShareSingleProducer``).
That keeps the report's scorecard, its per-hour column and the CLI's own output
in exact agreement instead of deriving the same number three ways.

The power-only ``gpu:computing:bool`` share (any GPU consumer) is **not** a
reported GPU-usage figure any more. It survives only inside the producer as a
cross-check input and is never rendered by the report.

Why power rather than the driver's ``gpu_busy_percent``: on this Strix Halo APU
that counter is a two-state detector whose step sits at **~30 W**, so it
inverts across the idle-with-display band and cannot separate idle from
computing. See ``monitoring/README.md`` (and the GPU computing section of
``REFERENCE.md``) for that measurement and its PromQL recipe; the reason is
stated once there and not repeated here.

Coverage honesty
----------------
The share is computed over **every expected instant on a 15 s grid** across the
window (``floor(window / step) + 1`` instants). Each instant is one of

- ``computing`` — the workload rule has a sample ``≥ 0.5`` and the pipeline was
  demonstrably alive;
- ``idle`` — the pipeline was alive and the rule is ``0`` (no *our-work*
  computing at that instant);
- ``unknown`` — everything else. Absence is never folded into ``idle`` and
  never into the numerator.

The three shares therefore cover the window exactly
(``computing + idle + unknown == 100 %``). A window with **no observed
instants at all** returns ``None`` — the report omits the row and ``--json``'s
``gpu_busy`` is ``null``, never a fabricated ``0 %``. A partially observed
window keeps the row and reports its unobserved share as ``unknown``, and a
per-hour bucket with no observed instants reads ``unknown`` (never ``0 %``).

Resolution honesty
------------------
The share binarises on the producer's 15 s grid: a short task inside one sample
fills the whole instant. No surface may claim sub-15-second precision.

What the figures mean
---------------------
- ``gpu_busy`` (this module) — the attributable share: hardware power **and**
  a busy local slot, i.e. our local model's GPU work.
- ``local_busy`` (``BusyStats``, ``dispatch → finish``) — the log-derived slot
  occupancy / capacity-contention figure that gates and alerts quote. It is a
  proxy-accounting question, not a GPU measurement, and is retained where it is
  correctly scoped (the ``## Local model utilization`` busy rows, the
  ``local_busy`` ``--json`` key and the ``> 60 %`` local-busy capacity gate).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

# Prometheus instance — loopback only; the monitoring README documents the
# exposure boundary (tailnet / loopback, no auth).
PROMETHEUS_URL = "http://127.0.0.1:9090"

#: Grid resolution and the recording rules' evaluation interval (15 s). Shared
#: with the producer so the report and the CLI measure the same instants.
STEP_SECONDS = 15

#: Freshness bound for the slot heartbeat: a sample older than one scrape
#: interval proves the sampler stopped reporting, so the instant is unknown.
SCRAPE_INTERVAL_SECONDS = 15

#: Repository root, from ``.pi/skills/proxy-usage-analysis/scripts/gpu_busy.py``.
#: The producer lives at ``<root>/scripts/gpu_utilisation.py`` and is imported
#: from there rather than re-implemented here.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_PRODUCER_PATH = _REPO_ROOT / "scripts" / "gpu_utilisation.py"

_producer_module: ModuleType | None = None


def _producer() -> ModuleType:
    """Load ``scripts/gpu_utilisation.py`` once and return it.

    Loaded by file path (not ``sys.path``) so the report never depends on the
    caller's import path, and cached so the module body runs once per process.
    """
    global _producer_module
    if _producer_module is None:
        spec = importlib.util.spec_from_file_location(
            "_gpu_utilisation_producer", _PRODUCER_PATH
        )
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError(
                f"cannot load the GPU-utilisation producer at {_PRODUCER_PATH}"
            )
        module = importlib.util.module_from_spec(spec)
        # Register before executing: ``from __future__ import annotations`` makes
        # the producer's dataclass fields string annotations, and dataclasses
        # resolves them through ``sys.modules[cls.__module__]``.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _producer_module = module
    return _producer_module


@dataclass
class GpuBusyResult:
    """Attributable GPU-activity share for the window, with explicit coverage.

    ``computing_fraction`` is the share of the window's **expected** grid
    instants on which the GPU drew ≥ 90 W *and* a local slot was busy
    (``gpu:computing_workload:bool``, produced only by
    ``scripts/gpu_utilisation.py``). It is computed over the full grid, so an
    unobserved instant lowers the fraction as ``unknown`` rather than
    renormalising it upward. ``busy_seconds`` (the ``computing`` share) plus
    ``idle_seconds`` plus ``unknown_seconds`` equals ``total_seconds`` by
    construction.

    ``hourly_shares`` carries one
    :class:`gpu_utilisation.HourlyAttributable` per absolute hour intersecting
    the window, so the report's ``## Summary by hour`` column is the same
    measure bucketed by hour — not a second definition.

    ``smoothed_power_values`` is the 1-minute mean of the producer's 15 s power
    series (``gpu:power_watts:amdgpu``) over the window; useful for identifying
    warm-up and cooldown transitions.
    """

    computing_fraction: float  # 0.0–1.0, over the expected grid
    total_seconds: float  # window length in seconds
    busy_seconds: float  # computing share of total_seconds
    idle_seconds: float  # idle share of total_seconds
    smoothed_power_values: list[tuple[float, float]]  # (unix_ts, mean_watts)
    unknown_seconds: float = 0.0  # unobserved share of total_seconds
    expected_samples: int = 0  # grid instants the window should contain
    observed_samples: int = 0  # computing + idle
    computing_samples: int = 0
    idle_samples: int = 0
    unknown_samples: int = 0
    hourly_shares: list = field(default_factory=list)  # list[HourlyAttributable]

    @property
    def busy_pct(self) -> float:
        """The attributable (``computing``) share of the window, in percent."""
        return self.computing_fraction * 100.0

    @property
    def attributable_pct(self) -> float:
        """Alias for :attr:`busy_pct` naming the measure explicitly."""
        return self.busy_pct

    @property
    def idle_pct(self) -> float:
        """The ``idle`` share of the window, in percent."""
        if self.total_seconds <= 0:
            return 0.0
        return self.idle_seconds / self.total_seconds * 100.0

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


def _minute_means(points: dict[int, float]) -> list[tuple[float, float]]:
    """Collapse the 15 s power series into 1-minute means (``(ts, watts)``).

    Derived from the producer's own ``power_watts`` series — no extra PromQL is
    issued, so the report never becomes a second query surface.
    """
    buckets: dict[int, list[float]] = {}
    for timestamp, value in sorted(points.items()):
        buckets.setdefault(timestamp - timestamp % 60, []).append(value)
    return [
        (float(minute), sum(values) / len(values))
        for minute, values in sorted(buckets.items())
    ]


def compute_gpu_busy(
    window_start: float,  # unix timestamp
    window_end: float,    # unix timestamp
    *,
    getter=None,
    series: dict | None = None,
) -> GpuBusyResult | None:
    """Compute the attributable GPU-activity share via the single producer.

    ``getter`` overrides the producer's HTTP getter (``getter(url) -> payload``)
    for tests. ``series`` short-circuits the Prometheus load entirely — pass the
    producer's ``{query_key: {unix_ts: value}}`` mapping to compute from a
    recorded bundle. Returns ``None`` when Prometheus is unreachable, when the
    queries yield no usable result, or when the window has **no observed
    instants at all** (a fully-absent window keeps the "no row" behaviour
    rather than reporting a fabricated ``0 %``).
    """
    total = window_end - window_start
    if total <= 0:
        return None

    start = int(round(window_start))
    end = int(round(window_end))
    if end <= start:
        return None

    producer = _producer()
    window = producer.Window(start=start, end=end)

    if series is None:
        expected = producer.expected_sample_count(start, end, STEP_SECONDS)
        try:
            series = producer.load_series(
                getter=getter,
                prometheus_url=PROMETHEUS_URL,
                window=window,
                step_seconds=STEP_SECONDS,
                expected_samples=expected,
            )
        except producer.PrometheusError:
            return None

    utilisation = producer.compute_utilisation(
        series,
        window=window,
        step_seconds=STEP_SECONDS,
        scrape_interval_seconds=SCRAPE_INTERVAL_SECONDS,
    )
    if utilisation.observed_samples == 0:
        # Fully-absent window: keep the "no row" behaviour, never 0 %.
        return None

    hourly = producer.hourly_attributable_shares(
        series,
        window=window,
        step_seconds=STEP_SECONDS,
        scrape_interval_seconds=SCRAPE_INTERVAL_SECONDS,
    )
    computing_fraction = utilisation.computing_samples / utilisation.expected_samples

    return GpuBusyResult(
        computing_fraction=computing_fraction,
        total_seconds=total,
        busy_seconds=computing_fraction * total,
        idle_seconds=utilisation.idle_pct / 100.0 * total,
        smoothed_power_values=_minute_means(series.get("power_watts", {})),
        unknown_seconds=utilisation.unknown_pct / 100.0 * total,
        expected_samples=utilisation.expected_samples,
        observed_samples=utilisation.observed_samples,
        computing_samples=utilisation.computing_samples,
        idle_samples=utilisation.idle_samples,
        unknown_samples=utilisation.unknown_samples,
        hourly_shares=hourly,
    )
