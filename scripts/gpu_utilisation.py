#!/usr/bin/env python3
"""GPU utilisation from Prometheus, with an honest *unknown* share.

Usage
-----
::

    python3 scripts/gpu_utilisation.py --window 2h
    python3 scripts/gpu_utilisation.py --since 2026-10-31T02:13:00Z \\
        --until 2026-10-31T04:13:00Z --step 15s --json

The answer comes straight from the deployed Prometheus store
(``http://127.0.0.1:9090``, 90 d retention, 15 s scrape) — never from
``proxy.log`` — so "how busy was the GPU between 17:00 and 19:00?" is a direct
measurement that works even after the log has rotated away.

Why not ``gpu_busy_percent``
---------------------------
On this host (Strix Halo APU) the driver's busy counter saturates at ~20 W: it
reads ~100 % for any draw above ~20 W, so idle-with-context stretches report as
busy (45.2 % false-busy over 7 days, LP-0MS9GA6QL007XTBP). Utilisation is
therefore derived from power draw via the recording rules in
``monitoring/gpu_compute_rules.yaml``, and ``gpu_busy_percent`` must never
appear in a query here (a regression test enforces it).

The three shares
----------------
``computing``
    Instants where ``gpu:computing_workload:bool == 1`` — the GPU drew at least
    ``THRESHOLD_WATTS`` (90 W) **and** at least one llama-server slot was
    processing. That is *our* local model's work, not any GPU consumer's.
``idle``
    Instants where the pipeline was reporting and the workload rule is ``0``.
``unknown``
    Everything else: instants the store cannot speak for. A PromQL
    ``avg_over_time`` silently renormalises over the samples that happen to be
    present, so a window where the sampler was down for 30 minutes would report
    a *higher* computing fraction than reality and could not distinguish "idle"
    from "nobody was looking". Here the denominator is always the number of
    instants the window *should* contain, and the missing share is reported as
    unknown. Missing data is never idle.

An instant is **observed** only when both hold:

1. a sentinel proves the sampling pipeline was alive at that instant —
   ``up{job="llama-proxy"} == 1``, ``up{job="node-exporter"} == 1``, or a
   ``llama_slots_sample_timestamp_seconds`` heartbeat no older than one scrape
   interval (a *stale* heartbeat proves the sampler stopped reporting, so it
   does not count);
2. the computing-workload rule actually has a sample there, because without it
   the instant cannot be classified as computing *or* idle.

``computing`` + ``idle`` + ``unknown`` is 100 % by construction.

Resolution
----------
Sample expectations are ``floor(window / step) + 1`` instants on a grid starting
at the window start. The step defaults to the scrape interval (15 s) and may be
coarser, never finer — the store cannot support a finer claim than its scrape
interval, and both the step actually queried and the scrape floor are printed
so the output never implies second-level fidelity.

Peak busy slots is the maximum of the ``llama_slots_busy`` samples present in the
window (identical to ``max_over_time(llama_slots_busy[<window>])`` evaluated at
the window end), and the power mean/max are taken over the power samples the
store actually holds.

Deployment dependency
---------------------
Computing/idle classification needs the combined ``gpu:computing_workload:bool``
recording rule and the proxy-published ``llama_slots_busy`` series. On a host
where those are not deployed (or the sampler has stopped) every instant is
unclassifiable, so the honest answer is ``unknown 100%`` and exit code ``4`` —
that is the contract working, not a bug.

``--json`` schema (stable; ``schema_version`` is bumped on breaking changes)::

    {
      "schema_version": 1,
      "prometheus_url": "...",
      "window": {"start": ISO, "end": ISO, "start_unix": int, "end_unix": int,
                 "duration_seconds": int},
      "resolution": {"step": "15s", "step_seconds": 15,
                     "scrape_interval_seconds": 15, "expected_samples": 481},
      "coverage": {"expected_samples": 481, "observed_samples": 441,
                   "computing_samples": 68, "idle_samples": 373,
                   "unknown_samples": 40},
      "shares_pct": {"computing": 14.1372, "idle": 77.5468, "unknown": 8.316,
                     "power_only_computing": 14.1372},
      "power_watts": {"mean": 50.491, "max": 111.075, "samples": 441},
      "slots": {"peak_busy": 2, "samples": 441},
      "rule_cross_check": {"compared_samples": 441, "mismatches": 0},
      "warnings": []
    }

``rule_cross_check`` keeps the same two keys whether or not it fires; the
diagnosis (which rule diverged, and therefore which line of
``monitoring/gpu_compute_rules.yaml`` to check) is in ``warnings``.

Percentages are rounded to four decimal places; the sample counts are exact, so
nothing is lost by the rounding.

The rule cross-check
--------------------
``rule_cross_check`` recomputes ``gpu:computing:bool`` and
``gpu:computing_workload:bool`` locally from the power and slot series and
compares them with the recorded rules, so a drifted threshold or expression
shows up as a ``warnings`` entry instead of a silently different definition.

Both rules are evaluated by Prometheus on its own schedule, against the last
sample at or before their evaluation instant — which is not necessarily the
grid instant the CLI queries. At a power step or a slot-state change the
recording can therefore sit one grid sample behind the raw series for reasons
that have nothing to do with the definition. The comparison is consequently
made against a recomputation over a ±1-sample window around the instant
(earlier sample first, the direction a recording rule looks), so jitter is not
reported as drift; a real definition change diverges at *every* instant and is
still caught.

Instants whose inputs are all frozen (nothing scraped or evaluated since the
previous instant) are skipped: that is a dead sampler, and comparing one frozen
sample set with another reports a rule drift for an outage.

The warning names the half that actually diverged — ``gpu:computing:bool``
(then ``THRESHOLD_WATTS`` is the thing to check) or
``gpu:computing_workload:bool`` (then the power-and-slot expression is, and the
warning says the threshold is not the cause).

Exit codes
----------
``0`` success, ``1`` unexpected error, ``2`` invalid arguments, ``3`` Prometheus
unreachable or answering with an error, ``4`` no observed samples in the window
(all-unknown — the message says so and the JSON is still emitted).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime

DEFAULT_PROMETHEUS_URL = "http://127.0.0.1:9090"
DEFAULT_WINDOW = "2h"
#: The deployed ``scrape_interval`` (monitoring/prometheus.yml). Also the
#: freshness bound for the slot heartbeat: a sample older than one scrape
#: interval proves the sampler stopped reporting.
SCRAPE_INTERVAL_SECONDS = 15
#: Power threshold shared with ``monitoring/gpu_compute_rules.yaml``
#: (``gpu:computing:bool`` / ``gpu:computing_workload:bool``). The CLI
#: re-derives the definitions locally to cross-check the recorded rules, so a
#: drifted threshold shows up as a mismatch instead of a silently different
#: definition.
THRESHOLD_WATTS = 90.0
HTTP_TIMEOUT_SECONDS = 10.0
#: How many grid samples of staleness the rule cross-check tolerates on each
#: input. A recording rule is evaluated on its own schedule against the last
#: sample at or before that instant, so at a power step or a slot-state change
#: it can legitimately see one grid sample earlier (or later) than the raw
#: series query for the same instant. Comparing the recorded rule against a
#: recomputation over that ±1-sample window separates timestamp jitter from a
#: genuine definition drift (LP-0MW5K8YWG0069ZD8).
CROSS_CHECK_STALENESS_SAMPLES = 1

#: Every PromQL expression this CLI issues, in the order it issues them. Keeping
#: them in one mapping means the shared-definition contract can be inspected
#: (and guarded against ``gpu_busy_percent``) in a single place.
QUERIES = {
    "power_watts": "gpu:power_watts:amdgpu",
    "computing_bool": "gpu:computing:bool",
    "computing_workload_bool": "gpu:computing_workload:bool",
    "slots_busy": "llama_slots_busy",
    "slot_heartbeat": "llama_slots_sample_timestamp_seconds",
    "up_llama_proxy": 'up{job="llama-proxy"}',
    "up_node_exporter": 'up{job="node-exporter"}',
}

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_PROMETHEUS_UNAVAILABLE = 3
EXIT_NO_DATA = 4

_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


class PrometheusError(RuntimeError):
    """Prometheus could not be queried, or answered with an error."""


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


def parse_duration(text: str) -> int:
    """Parse ``15s`` / ``90m`` / ``2h`` / ``1h30m`` / bare seconds into a count.

    Raises ``ValueError`` with the offending text for anything else, including
    a negative or empty value.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty duration")
    if raw.isdigit():
        return int(raw)

    total = 0
    index = 0
    while index < len(raw):
        digits_start = index
        while index < len(raw) and raw[index].isdigit():
            index += 1
        if digits_start == index:
            raise ValueError(_duration_error(text))
        amount = int(raw[digits_start:index])
        if index >= len(raw) or raw[index] not in _DURATION_UNITS:
            raise ValueError(_duration_error(text))
        total += amount * _DURATION_UNITS[raw[index]]
        index += 1
    return total


def _duration_error(text: str) -> str:
    return (
        f"cannot parse duration {text!r}: expected a number of seconds or a "
        "value like 15s, 90m, 2h or 1h30m"
    )


def parse_timestamp(text: str) -> int:
    """Parse unix seconds or an ISO-8601 timestamp into unix seconds."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty timestamp")
    try:
        return int(float(raw))
    except ValueError:
        pass

    normalised = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    try:
        parsed = datetime.fromisoformat(normalised)
    except ValueError as exc:
        raise ValueError(
            f"cannot parse timestamp {text!r}: expected unix seconds or "
            "ISO-8601 such as 2026-10-31T02:13:00Z"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())


@dataclass(frozen=True)
class Window:
    """An inclusive-unix-second query window."""

    start: int
    end: int

    @property
    def duration_seconds(self) -> int:
        return self.end - self.start


def resolve_window(*, now, window=None, since=None, until=None) -> Window:
    """Resolve the query window from ``--window`` or ``--since``/``--until``.

    ``--window`` is a duration ending now and is mutually exclusive with the
    absolute bounds (mixing them is ambiguous, so it is rejected rather than
    guessed). ``--since`` without ``--until`` means "until now"; ``--until``
    without ``--since`` is rejected.
    """
    if window is not None and (since is not None or until is not None):
        raise ValueError(
            "--window cannot be combined with --since/--until: use --window for "
            "a duration ending now, or --since/--until for absolute bounds"
        )
    if until is not None and since is None:
        raise ValueError("--until requires --since")

    if since is not None:
        start = parse_timestamp(since)
        end = parse_timestamp(until) if until is not None else int(now)
    else:
        seconds = parse_duration(window if window is not None else DEFAULT_WINDOW)
        if seconds <= 0:
            raise ValueError("--window must be positive")
        end = int(now)
        start = end - seconds

    if end <= start:
        raise ValueError(
            f"window end ({_iso(end)}) must be after its start ({_iso(start)})"
        )
    return Window(start=start, end=end)


def resolve_scrape_interval(text: str) -> int:
    """Resolve the scrape interval (the freshness bound and resolution floor)."""
    seconds = parse_duration(text)
    if seconds <= 0:
        raise ValueError("--scrape-interval must be positive")
    return seconds


def resolve_step(text, scrape_interval_seconds: int) -> int:
    """Resolve the query step; it may be coarser than, never finer than, a scrape."""
    if text is None:
        return scrape_interval_seconds
    seconds = parse_duration(text)
    if seconds <= 0:
        raise ValueError("--step must be positive")
    if seconds < scrape_interval_seconds:
        raise ValueError(
            f"--step {seconds}s is finer than the {scrape_interval_seconds}s "
            "scrape interval: the store cannot support that resolution"
        )
    return seconds


def expected_sample_count(start: int, end: int, step_seconds: int) -> int:
    """Number of grid instants an honest window report must account for."""
    return (end - start) // step_seconds + 1


# ---------------------------------------------------------------------------
# Prometheus access
# ---------------------------------------------------------------------------


def query_range_url(
    prometheus_url: str, query: str, start: int, end: int, step_seconds: int
) -> str:
    base = prometheus_url.rstrip("/")
    params = urllib.parse.urlencode(
        {
            "query": query,
            "start": int(start),
            "end": int(end),
            "step": int(step_seconds),
        }
    )
    return f"{base}/api/v1/query_range?{params}"


def http_get_json(url: str) -> dict:
    """GET *url* and parse JSON, raising ``PrometheusError`` with the URL."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise PrometheusError(f"Prometheus answered HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise PrometheusError(f"cannot reach Prometheus at {url}: {exc}") from exc

    try:
        payload = json.loads(body)
    except ValueError as exc:
        raise PrometheusError(f"Prometheus returned non-JSON for {url}") from exc
    return _checked_payload(payload, url)


def _checked_payload(payload, url: str) -> dict:
    if not isinstance(payload, dict) or payload.get("status") != "success":
        detail = payload.get("error") if isinstance(payload, dict) else payload
        raise PrometheusError(f"Prometheus query failed for {url}: {detail}")
    return payload


def query_range(
    prometheus_url: str,
    query: str,
    start: int,
    end: int,
    step_seconds: int,
    *,
    getter=None,
) -> dict:
    """Issue one ``query_range`` call through *getter* (injectable for tests)."""
    url = query_range_url(prometheus_url, query, start, end, step_seconds)
    payload = (getter or http_get_json)(url)
    return _checked_payload(payload, url)


def parse_matrix(payload: dict) -> dict[int, float]:
    """Flatten a ``query_range`` matrix payload into ``{unix_ts: value}``.

    Multiple series are collapsed by their maximum: that is the honest
    reduction for every series queried here (peak GPU power across chips, peak
    busy slots, and "any target is up"), and this host exposes one GPU and one
    llama-proxy.
    """
    points: dict[int, float] = {}
    for series in (payload.get("data") or {}).get("result") or []:
        for sample in series.get("values") or []:
            try:
                timestamp = int(float(sample[0]))
                value = float(sample[1])
            except (TypeError, ValueError, IndexError):
                continue
            previous = points.get(timestamp)
            points[timestamp] = value if previous is None else max(previous, value)
    return points


def align_to_grid(
    points: dict[int, float], *, start: int, step: int, expected: int
) -> dict[int, float]:
    """Snap samples onto the query grid, dropping anything out of the window.

    Prometheus returns samples on the requested grid, but a live store can hand
    back timestamps a fraction of a step off (rounding, step changes). A sample
    within half a step of a grid instant is attributed to it; anything further
    away, and anything outside the window, is dropped rather than silently
    shifting the coverage arithmetic.
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


def load_series(
    *,
    getter,
    prometheus_url: str,
    window: Window,
    step_seconds: int,
    expected_samples: int,
) -> dict[str, dict[int, float]]:
    """Query every series in ``QUERIES`` and align each onto the window grid."""
    series: dict[str, dict[int, float]] = {}
    for key, query in QUERIES.items():
        payload = query_range(
            prometheus_url,
            query,
            window.start,
            window.end,
            step_seconds,
            getter=getter,
        )
        series[key] = align_to_grid(
            parse_matrix(payload),
            start=window.start,
            step=step_seconds,
            expected=expected_samples,
        )
    return series


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def power_at_compute(power_watts: float, threshold: float = THRESHOLD_WATTS) -> bool:
    """``gpu:computing:bool``: the GPU drew compute-level power (inclusive)."""
    return power_watts >= threshold


def classify_instant(
    power_watts: float,
    busy_slots: float,
    threshold: float = THRESHOLD_WATTS,
) -> bool:
    """``gpu:computing_workload:bool``: compute power **and** a busy slot."""
    return power_at_compute(power_watts, threshold) and busy_slots >= 1


def cross_check_inputs_frozen(
    *,
    power: dict[int, float],
    slots_busy: dict[int, float],
    workload_bool: dict[int, float],
    heartbeat: dict[int, float],
    timestamp: int,
    step_seconds: int,
) -> bool:
    """True when every cross-check input is the same sample as the previous instant.

    Nothing has been scraped or evaluated since the previous instant, so the
    latest samples are Prometheus' lookback handing back frozen values: the
    recorded rule froze on the slot state it last saw, the raw series froze on
    the next scrape, and the two can disagree by a sample forever after. That
    is a dead sampler, not a definition drift, so the instant is skipped
    instead of reporting a rule drift for it (LP-0MW5K8YWG0069ZD8).
    """
    previous = timestamp - step_seconds
    for series in (power, slots_busy, workload_bool, heartbeat):
        value = series.get(timestamp)
        if value is None or series.get(previous) != value:
            return False
    return True


def staleness_candidates(
    points: dict[int, float],
    timestamp: int,
    step_seconds: int,
    samples: int = CROSS_CHECK_STALENESS_SAMPLES,
) -> list[float]:
    """Values *timestamp* could legitimately have been evaluated against.

    A recording rule evaluates against the last sample at or before its own
    instant, which is not necessarily the grid instant the CLI queries, so a
    cross-check that only reads ``points[timestamp]`` counts every slot-state
    or power step as a definition drift. The candidate list is the sample at
    the instant first, then its neighbours by increasing distance and earlier
    before later (the direction a recording rule looks), with duplicates
    dropped.
    """
    offsets = [0]
    for distance in range(1, max(samples, 0) + 1):
        offsets.extend((-distance, distance))

    candidates: list[float] = []
    for offset in offsets:
        value = points.get(timestamp + offset * step_seconds)
        if value is not None and value not in candidates:
            candidates.append(value)
    return candidates


def cross_check_warning(
    *,
    mismatches: int,
    compared_samples: int,
    power_mismatches: int,
    workload_mismatches: int,
    threshold_watts: float,
) -> str:
    """Explain a cross-check failure by naming the rule that actually drifted.

    ``THRESHOLD_WATTS`` is named only for a power-classification divergence: a
    slot-join drift cannot be produced by the threshold, so pointing the
    operator at it would send them to the wrong line
    (LP-0MW5K8YWG0069ZD8).
    """
    causes = []
    if power_mismatches:
        causes.append(
            f"{power_mismatches} instants where gpu:computing:bool is not the "
            f"local {threshold_watts:g} W classification (check the threshold "
            f"in monitoring/gpu_compute_rules.yaml against "
            f"THRESHOLD_WATTS={threshold_watts:g})"
        )
    if workload_mismatches:
        remedy = (
            "check the power-and-slot expression in "
            "monitoring/gpu_compute_rules.yaml"
        )
        if not power_mismatches:
            remedy += (
                " — the power classification itself agrees, so the power "
                "threshold is not the cause"
            )
        causes.append(
            f"{workload_mismatches} instants where gpu:computing_workload:bool "
            f"is not the local recomputation ({remedy})"
        )
    return (
        f"the recorded rules disagree with the local recomputation at "
        f"{mismatches} of {compared_samples} instants (outside a "
        f"±{CROSS_CHECK_STALENESS_SAMPLES}-sample staleness window): "
        + "; ".join(causes)
    )


@dataclass(frozen=True)
class GpuUtilisation:
    """Measured GPU utilisation for one window, with explicit coverage."""

    window: Window
    step_seconds: int
    scrape_interval_seconds: int
    expected_samples: int
    observed_samples: int
    computing_samples: int
    idle_samples: int
    unknown_samples: int
    power_only_computing_samples: int
    power_mean_watts: float | None
    power_max_watts: float | None
    power_samples: int
    peak_busy_slots: int | None
    slot_samples: int
    cross_check_samples: int
    cross_check_mismatches: int
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        total = self.computing_samples + self.idle_samples + self.unknown_samples
        if self.expected_samples <= 0 or total != self.expected_samples:
            raise ValueError(
                "coverage must account for every expected sample: "
                f"{total} != {self.expected_samples}"
            )
        if self.observed_samples != self.computing_samples + self.idle_samples:
            raise ValueError("observed samples must be exactly computing + idle")

    @property
    def computing_pct(self) -> float:
        return 100.0 * self.computing_samples / self.expected_samples

    @property
    def idle_pct(self) -> float:
        return 100.0 * self.idle_samples / self.expected_samples

    @property
    def unknown_pct(self) -> float:
        # By construction: computing + idle + unknown == 100 %.
        return 100.0 - self.computing_pct - self.idle_pct

    @property
    def power_only_computing_pct(self) -> float:
        return 100.0 * self.power_only_computing_samples / self.expected_samples

    def to_dict(self, *, prometheus_url: str) -> dict:
        """The documented, stable ``--json`` schema."""
        return {
            "schema_version": 1,
            "prometheus_url": prometheus_url,
            "window": {
                "start": _iso(self.window.start),
                "end": _iso(self.window.end),
                "start_unix": self.window.start,
                "end_unix": self.window.end,
                "duration_seconds": self.window.duration_seconds,
            },
            "resolution": {
                "step": format_duration(self.step_seconds),
                "step_seconds": self.step_seconds,
                "scrape_interval_seconds": self.scrape_interval_seconds,
                "expected_samples": self.expected_samples,
            },
            "coverage": {
                "expected_samples": self.expected_samples,
                "observed_samples": self.observed_samples,
                "computing_samples": self.computing_samples,
                "idle_samples": self.idle_samples,
                "unknown_samples": self.unknown_samples,
            },
            "shares_pct": {
                "computing": round(self.computing_pct, 4),
                "idle": round(self.idle_pct, 4),
                "unknown": round(self.unknown_pct, 4),
                "power_only_computing": round(self.power_only_computing_pct, 4),
            },
            "power_watts": {
                "mean": None
                if self.power_mean_watts is None
                else round(self.power_mean_watts, 3),
                "max": None
                if self.power_max_watts is None
                else round(self.power_max_watts, 3),
                "samples": self.power_samples,
            },
            "slots": {"peak_busy": self.peak_busy_slots, "samples": self.slot_samples},
            "rule_cross_check": {
                "compared_samples": self.cross_check_samples,
                "mismatches": self.cross_check_mismatches,
            },
            "warnings": list(self.warnings),
        }

    def render_human(self) -> str:
        peak = "n/a" if self.peak_busy_slots is None else str(self.peak_busy_slots)
        lines = [
            f"GPU utilisation {_iso(self.window.start)} .. {_iso(self.window.end)}"
            f" ({format_duration(self.window.duration_seconds)})",
            f"  computing {self.computing_pct:.1f}% / idle {self.idle_pct:.1f}%"
            f" / unknown {self.unknown_pct:.1f}%",
            f"  power-only computing {self.power_only_computing_pct:.1f}%"
            " (gpu:computing:bool: any GPU consumer, not just ours)",
            f"  power mean {_watts(self.power_mean_watts)} / max"
            f" {_watts(self.power_max_watts)} ({self.power_samples} samples);"
            f" peak {peak} busy slots",
            f"  resolution: step {format_duration(self.step_seconds)} (scrape floor"
            f" {format_duration(self.scrape_interval_seconds)});"
            f" {self.expected_samples} expected samples, {self.observed_samples}"
            f" observed, {self.unknown_samples} unknown",
        ]
        return "\n".join(lines)


def compute_utilisation(
    series: dict[str, dict[int, float]],
    *,
    window: Window,
    step_seconds: int,
    scrape_interval_seconds: int,
    threshold_watts: float = THRESHOLD_WATTS,
) -> GpuUtilisation:
    """Classify every expected instant and account for the unobserved ones.

    *series* maps each ``QUERIES`` key to grid-aligned ``{unix_ts: value}``
    samples. The window's grid defines the denominator; every instant is
    computing, idle or unknown, and the three must cover it exactly.
    """
    expected = expected_sample_count(window.start, window.end, step_seconds)
    power = series.get("power_watts", {})
    computing_bool = series.get("computing_bool", {})
    workload_bool = series.get("computing_workload_bool", {})
    slots_busy = series.get("slots_busy", {})
    heartbeat = series.get("slot_heartbeat", {})
    up_llama_proxy = series.get("up_llama_proxy", {})
    up_node_exporter = series.get("up_node_exporter", {})

    def pipeline_alive(timestamp: int) -> bool:
        if up_llama_proxy.get(timestamp) == 1 or up_node_exporter.get(timestamp) == 1:
            return True
        beat = heartbeat.get(timestamp)
        return beat is not None and abs(timestamp - beat) <= scrape_interval_seconds

    computing = idle = power_only = 0
    cross_checked = mismatches = 0
    power_mismatches = workload_mismatches = 0
    for index in range(expected):
        timestamp = window.start + index * step_seconds
        workload = workload_bool.get(timestamp)
        computing_flag = computing_bool.get(timestamp)
        alive = pipeline_alive(timestamp)

        if alive and workload is not None:
            if workload >= 0.5:
                computing += 1
            else:
                idle += 1
        if alive and computing_flag is not None and computing_flag >= 0.5:
            power_only += 1

        if (
            timestamp in power
            and timestamp in slots_busy
            and workload is not None
            and not cross_check_inputs_frozen(
                power=power,
                slots_busy=slots_busy,
                workload_bool=workload_bool,
                heartbeat=heartbeat,
                timestamp=timestamp,
                step_seconds=step_seconds,
            )
        ):
            cross_checked += 1
            power_here = staleness_candidates(power, timestamp, step_seconds)
            slots_here = staleness_candidates(slots_busy, timestamp, step_seconds)
            workload_agrees = any(
                classify_instant(recorded_power, busy, threshold_watts)
                == (workload >= 0.5)
                for recorded_power in power_here
                for busy in slots_here
            )
            power_agrees = True
            if computing_flag is not None:
                power_agrees = any(
                    power_at_compute(recorded_power, threshold_watts)
                    == (computing_flag >= 0.5)
                    for recorded_power in power_here
                )
            power_mismatches += int(not power_agrees)
            workload_mismatches += int(not workload_agrees)
            mismatches += int(not (power_agrees and workload_agrees))

    observed = computing + idle
    unknown = expected - observed
    power_values = [power[timestamp] for timestamp in sorted(power)]
    slot_values = [slots_busy[timestamp] for timestamp in sorted(slots_busy)]

    warnings: list[str] = []
    if mismatches:
        warnings.append(
            cross_check_warning(
                mismatches=mismatches,
                compared_samples=cross_checked,
                power_mismatches=power_mismatches,
                workload_mismatches=workload_mismatches,
                threshold_watts=threshold_watts,
            )
        )

    return GpuUtilisation(
        window=window,
        step_seconds=step_seconds,
        scrape_interval_seconds=scrape_interval_seconds,
        expected_samples=expected,
        observed_samples=observed,
        computing_samples=computing,
        idle_samples=idle,
        unknown_samples=unknown,
        power_only_computing_samples=power_only,
        power_mean_watts=(sum(power_values) / len(power_values)) if power_values else None,
        power_max_watts=max(power_values) if power_values else None,
        power_samples=len(power_values),
        peak_busy_slots=int(round(max(slot_values))) if slot_values else None,
        slot_samples=len(slot_values),
        cross_check_samples=cross_checked,
        cross_check_mismatches=mismatches,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_duration(seconds: int) -> str:
    """Compact duration: ``15s``, ``1m30s``, ``2h``, ``1h30m``."""
    remaining = int(seconds)
    if remaining <= 0:
        return "0s"
    parts = []
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        if remaining >= size:
            parts.append(f"{remaining // size}{unit}")
            remaining %= size
    return "".join(parts)


def _iso(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _watts(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} W"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu_utilisation.py",
        description=(
            "Report GPU computing/idle/unknown time, mean and max power, and peak "
            "busy llama-server slots for a window, measured from Prometheus. "
            "Missing samples are reported as unknown, never as idle."
        ),
        epilog=(
            "Exit codes: 0 success, 1 unexpected error, 2 invalid arguments, "
            "3 Prometheus unreachable or erroring, 4 no observed samples in the "
            "window (all unknown). See the module docstring for the --json schema."
        ),
    )
    parser.add_argument(
        "--window",
        metavar="DURATION",
        default=None,
        help=f"window length ending now (default: {DEFAULT_WINDOW}); "
        "mutually exclusive with --since/--until",
    )
    parser.add_argument(
        "--since",
        metavar="TIME",
        default=None,
        help="window start: unix seconds or ISO-8601 (e.g. 2026-10-31T02:13:00Z)",
    )
    parser.add_argument(
        "--until",
        metavar="TIME",
        default=None,
        help="window end (requires --since); defaults to now",
    )
    parser.add_argument(
        "--step",
        metavar="DURATION",
        default=None,
        help=f"query step (default: the {SCRAPE_INTERVAL_SECONDS}s scrape interval); "
        "coarser is allowed, finer is rejected",
    )
    parser.add_argument(
        "--scrape-interval",
        metavar="DURATION",
        default=f"{SCRAPE_INTERVAL_SECONDS}s",
        help="store resolution floor and slot-heartbeat freshness bound "
        f"(default: {SCRAPE_INTERVAL_SECONDS}s)",
    )
    parser.add_argument(
        "--prometheus-url",
        default=DEFAULT_PROMETHEUS_URL,
        help=f"Prometheus base URL (default: {DEFAULT_PROMETHEUS_URL})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the documented machine-readable schema instead of prose",
    )
    return parser


def main(argv=None, *, getter=None, now=None) -> int:
    """Run the CLI; returns the process exit code (see the module docstring)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        scrape_interval = resolve_scrape_interval(args.scrape_interval)
        step_seconds = resolve_step(args.step, scrape_interval)
        window = resolve_window(
            now=time.time() if now is None else now,
            window=args.window,
            since=args.since,
            until=args.until,
        )
    except ValueError as exc:
        print(f"gpu_utilisation: {exc}", file=sys.stderr)
        return EXIT_USAGE

    expected = expected_sample_count(window.start, window.end, step_seconds)
    try:
        series = load_series(
            getter=getter,
            prometheus_url=args.prometheus_url,
            window=window,
            step_seconds=step_seconds,
            expected_samples=expected,
        )
    except PrometheusError as exc:
        print(f"gpu_utilisation: {exc}", file=sys.stderr)
        return EXIT_PROMETHEUS_UNAVAILABLE

    result = compute_utilisation(
        series,
        window=window,
        step_seconds=step_seconds,
        scrape_interval_seconds=scrape_interval,
    )

    if args.json:
        print(json.dumps(result.to_dict(prometheus_url=args.prometheus_url), indent=2))
    else:
        print(result.render_human())
    for warning in result.warnings:
        print(f"note: {warning}", file=sys.stderr)

    if result.observed_samples == 0:
        print(
            f"gpu_utilisation: no observed samples in {_iso(window.start)} .. "
            f"{_iso(window.end)} (unknown {result.unknown_pct:.1f}%): the sampling "
            "pipeline was not reporting, so the GPU was neither busy nor idle in "
            "any measurable sense. Check prometheus.service, node_exporter.service "
            "and the proxy's slot sampler.",
            file=sys.stderr,
        )
        return EXIT_NO_DATA
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
