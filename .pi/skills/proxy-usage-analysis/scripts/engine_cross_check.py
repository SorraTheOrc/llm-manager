"""Engine cross-check: proxy-inferred phase slot-time vs llama-server eval timings.

LP-0MVNWULLI006ZYXX. The report's local slot-time figures
(``## Prefill vs decode``, ``local_busy.prefill_vs_decode``) are **inferred
proxy-side** from proxy log events. llama-server independently reports the true
per-task duration of every prompt-eval and decode call in its eval-timing lines,
which :mod:`llama_log_parser` already parses (for tok/s) and attributes to the
Qwen3 child port. This module sums those durations per phase and compares them
with the proxy's own figure, so a divergence between what the proxy thinks the
local engine did and what the engine reports doing becomes visible in the daily
report instead of staying invisible.

Metric types — the caveat the report must state:

- the proxy figure is **slot-seconds**: the sum of the per-stream phase
  durations (the integral of concurrency);
- the engine figure is a **sum of per-task durations**, which is the same
  *kind* of quantity (slot-time, not wall-clock) because llama-server log lines
  carry no timestamps — they can only be summed, never turned into a time
  series or a union;
- neither figure is comparable with ``BusyStats.busy_seconds`` / ``busy_pct``,
  which are a wall-clock **union** of merged intervals. No arithmetic here
  mixes the two.

A non-zero delta is expected, not a defect: the proxy counts per-stream phases
plus its own queueing/retries/cancellations, while the engine counts per-task
eval calls. The *direction* is the diagnostic:

- **proxy above engine** — the proxy accounted for phase time the engine never
  (fully) performed: queueing, retries, cancelled or abandoned work;
- **engine above proxy** — the engine performed local work the proxy did not
  attribute: unlogged or partially-logged streams, or traffic whose eval lines
  landed in the window while its proxy events did not.

Availability is explicit (AC 4): a phase whose engine samples are missing, or
whose proxy samples are missing, is reported as **unavailable with a reason**
and carries ``None`` deltas — never a misleading ``0`` or ``100%``.
"""

from __future__ import annotations

import math

import aggregation
import llama_log_parser

# Default divergence threshold in percent (absolute percentage delta per
# phase). A structural skew is expected (see the module docstring), so only a
# large divergence signals an accounting problem; 25 % is the shipped default
# and is configurable via ``local_engine_cross_check_threshold_pct`` in
# ``proxy/config.yaml``. ``0`` disables the warning recommendation.
DEFAULT_THRESHOLD_PCT = 25.0

# A phase diverging by at least twice the threshold is escalated to a HIGH
# severity recommendation: at that size the skew is no longer the expected
# structural difference between per-stream and per-task accounting.
HIGH_SEVERITY_MULTIPLIER = 2.0

PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"
PHASES = (PHASE_PREFILL, PHASE_DECODE)

# Phase -> (engine eval-timing kind, llama-server line label).
_ENGINE_KIND = {
    PHASE_PREFILL: llama_log_parser.KIND_PROMPT_EVAL,
    PHASE_DECODE: llama_log_parser.KIND_DECODE,
}
_LINE_LABEL = {
    PHASE_PREFILL: "`prompt eval time`",
    PHASE_DECODE: "`eval time`",
}
_PROXY_REASON = {
    PHASE_PREFILL: (
        "the proxy logs recorded no measurable prefill phase for the window "
        "(no per-stream `dispatch_first_byte_ms` first byte)"
    ),
    PHASE_DECODE: (
        "the proxy logs recorded no measurable decode phase for the window "
        "(no per-stream `Stream finished`)"
    ),
}

METRIC_CAVEAT = (
    "Engine figures are **sums of llama-server's per-task eval durations** "
    "(slot-seconds): the log lines carry no timestamps, so they can only be "
    "summed, never turned into a time series or a union. They are therefore "
    "**not wall-clock** and are **not comparable with `busy_seconds` / "
    "`busy_pct`** (a merged wall-clock union). No figure here mixes the two."
)

INTERPRETATION = (
    "A non-zero delta is expected — the proxy counts per-stream phases plus its "
    "own queueing, retries and cancellations, the engine counts per-task eval "
    "calls. Direction is the signal: **proxy above engine** suggests queueing, "
    "retries or cancelled work — phase time the engine never (fully) performed; "
    "**engine above proxy** suggests unattributed local traffic the proxy did "
    "not account for (unlogged or partially-logged streams, or eval lines whose "
    "proxy events fell outside the window)."
)

# Window boundedness of the engine side (llama-server lines carry no
# timestamps): the parser attributes a whole log file to the window when the
# file's last write falls inside it, so a window much shorter than the live log
# file compares an in-window proxy figure against the file's whole span.
WINDOW_APPROXIMATION = (
    "Window boundedness: llama-server lines carry no timestamps, so the engine "
    "totals cover every log file whose last write falls in the window (in "
    "practice the daily rotation spans the analysis window). For a window much "
    "shorter than the live `llama-server.log`, the engine span is longer than "
    "the proxy's, which reads as engine ≫ proxy and is a windowing artefact, "
    "not unattributed traffic."
)


def threshold_pct(config: dict | None) -> float:
    """Configured divergence threshold in percent (``0`` disables the warning).

    A missing, malformed, negative or non-finite value falls back to
    :data:`DEFAULT_THRESHOLD_PCT`; the loader already coerces the scalar to a
    float, so a fractional threshold (``12.5``) survives.
    """
    raw = (config or {}).get("local_engine_cross_check_threshold_pct")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD_PCT
    if not math.isfinite(value) or value < 0.0:
        return DEFAULT_THRESHOLD_PCT
    return value


def _bucket_count(buckets: object) -> int:
    """Sample count of the ``total`` bucket of a speed-stats bucket map."""
    if not isinstance(buckets, dict):
        return 0
    bucket = buckets.get(llama_log_parser.TOTAL)
    return int(getattr(bucket, "count", 0) or 0)


def _engine_totals(speed: object | None) -> tuple[dict[str, float], dict[str, int]]:
    """Per-phase engine seconds and sample counts from the parser's totals.

    The totals are produced by ``llama_log_parser.build_speed_stats`` inside the
    same Qwen3-port attribution loop that feeds the speed percentiles, so the
    window filtering, rotation handling, port inference and diagnostic
    bookkeeping are inherited unchanged (LP-0MVNWULLI006ZYXX constraint).
    """
    seconds = {phase: 0.0 for phase in PHASES}
    samples = {phase: 0 for phase in PHASES}
    if speed is None:
        return seconds, samples
    totals = getattr(speed, "engine_seconds", None) or {}
    for phase, kind in _ENGINE_KIND.items():
        try:
            seconds[phase] = round(float(totals.get(kind) or 0.0), 3)
        except (TypeError, ValueError):
            seconds[phase] = 0.0
    samples[PHASE_PREFILL] = _bucket_count(getattr(speed, "prompt_eval", None))
    samples[PHASE_DECODE] = _bucket_count(getattr(speed, "decode", None))
    return seconds, samples


def _proxy_phase(busy: object | None, phase: str) -> tuple[float, int, int]:
    """``(slot_seconds, samples, streams)`` for one phase of the proxy side."""
    stats = None
    if busy is not None:
        stats = (
            getattr(busy, "prefill_stats", None)
            if phase == PHASE_PREFILL
            else getattr(busy, "decode_stats", None)
        )
    if stats is None:
        return 0.0, 0, 0
    return (
        round(float(getattr(stats, "total_seconds", 0.0) or 0.0), 3),
        int(getattr(stats, "count", 0) or 0),
        int(getattr(stats, "stream_count", 0) or 0),
    )


def _engine_reasons(speed: object | None) -> list[str]:
    """The parser's named empty-window reasons (engine side).

    Reuses ``llama_log_parser.empty_window_reasons`` so the cross-check speaks
    the same diagnostic vocabulary as the ``## Decode speed`` /
    ``## Prompt eval speed`` sections (missing logs, rotation, verbosity/build,
    attribution failure) instead of inventing a new one.
    """
    if speed is None:
        return []
    try:
        return list(llama_log_parser.empty_window_reasons(speed))
    except Exception:  # defensive: never fail the report on a diagnostic
        return []


def compute(summary: object, config: dict | None = None) -> dict:
    """Build the machine-readable engine cross-check for ``summary``.

    Returns a dict with the proxy/engine seconds, absolute and percentage
    deltas, per-phase availability and reason, the configured threshold and the
    divergent phases. Every figure is either a real measurement or ``None``
    (with a reason) — never a fabricated ``0``/``100%``.
    """
    threshold = threshold_pct(config)
    busy = getattr(summary, "busy", None)
    speed = getattr(summary, "speed", None)
    engine_seconds, engine_samples = _engine_totals(speed)
    total_engine_samples = sum(engine_samples.values())
    engine_unavailable = _engine_reasons(speed) if total_engine_samples == 0 else []

    phases: dict[str, dict] = {}
    available: list[str] = []
    divergent: list[str] = []
    for phase in PHASES:
        proxy_seconds, proxy_samples, proxy_streams = _proxy_phase(busy, phase)
        entry: dict = {
            "phase": phase,
            "proxy_slot_seconds": proxy_seconds,
            "proxy_samples": proxy_samples,
            "proxy_streams": proxy_streams,
            "engine_seconds": engine_seconds[phase],
            "engine_samples": engine_samples[phase],
            "delta_seconds": None,
            "delta_pct": None,
            "available": False,
            "reason": None,
            "divergent": False,
        }
        if speed is None:
            entry["reason"] = (
                "no llama-server eval-timing data was supplied to the analysis"
            )
        elif engine_samples[phase] == 0:
            if total_engine_samples == 0:
                detail = "; ".join(engine_unavailable) if engine_unavailable else ""
                entry["reason"] = (
                    "no Qwen3-attributed engine eval-timing lines in the window"
                    + (f" — {detail}" if detail else "")
                )
            else:
                entry["reason"] = (
                    f"the Qwen3-attributed eval timings in the window carry no "
                    f"{_LINE_LABEL[phase]} lines "
                    f"({total_engine_samples - engine_samples[phase]} other line(s) "
                    "attributed)"
                )
        elif proxy_samples == 0:
            entry["reason"] = _PROXY_REASON[phase]
        else:
            engine_phase_seconds = engine_seconds[phase]
            delta = round(proxy_seconds - engine_phase_seconds, 3)
            delta_pct = (
                round(delta / engine_phase_seconds * 100.0, 1)
                if engine_phase_seconds > 0.0
                else None
            )
            entry["delta_seconds"] = delta
            entry["delta_pct"] = delta_pct
            if delta_pct is not None:
                entry["available"] = True
                available.append(phase)
                entry["divergent"] = threshold > 0.0 and abs(delta_pct) > threshold
                if entry["divergent"]:
                    divergent.append(phase)
            else:
                entry["reason"] = (
                    "the engine reported zero seconds for this phase, so a "
                    "percentage delta is undefined"
                )
        phases[phase] = entry

    if not available:
        status = "unavailable"
    elif len(available) == len(PHASES):
        status = "ok"
    else:
        status = "partial"

    return {
        "unit": aggregation.SLOT_SECONDS_UNIT,
        "engine_metric": "sum of per-task eval durations (slot-seconds, not wall-clock)",
        "metric_caveat": METRIC_CAVEAT,
        "interpretation": INTERPRETATION,
        "window_approximation": WINDOW_APPROXIMATION,
        "threshold_pct": threshold,
        "threshold_enabled": threshold > 0.0,
        "status": status,
        "prefill": phases[PHASE_PREFILL],
        "decode": phases[PHASE_DECODE],
        "available_phases": available,
        "divergent_phases": divergent,
        "engine_unavailable_reasons": engine_unavailable,
    }
