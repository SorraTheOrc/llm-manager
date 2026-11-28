"""Quantify GPU compute burned on work no client received (LP-0MW2SQNHJ0044PV6).

The usage report already counts ``local_eviction_after_acceptance`` events and
the priced tokens they discarded, but a count cannot answer the operator's
question — *how much GPU compute did this window burn on work nobody got, and
why?* This module produces that answer for one window and hands it to
:mod:`reporting` for three renderings (a summary table near the start of
``report.md``, a detail section in the body and an appendix).

**One source of truth.** The attribution rules (occupancy / capacity /
predicted-warmth), the abandoned-prefill window model, the release→cancel
accounting and the backend accounting are *not* re-implemented here: the
committed analyser ``proxy/scripts/analyse_prefill_deadline_residual.py``
computes them and this module normalises its payload. Duplicating those rules
would let the two tools disagree about the same window.

**Honest degradation.** Every input can be missing or unreliable, and the
result says so instead of publishing a number that looks exact:

- no analyser (not a source checkout) → ``status="unavailable"``;
- no ``llama-server`` log, or a log whose coverage is not ``exact`` → the
  backend-side figures are marked indicative and the caveat names the reason;
- a window that contains a proxy or backend restart → ``restart_free=False``
  and the verdict is ``inconclusive``: such a window cannot certify anything
  because the restart both discards prefills and re-prices capacity.

Thresholds are the producer-agreed ones recorded on LP-0MVN3KMGI009BZ0D:
user-class ``prefill_deadline`` + ``measured_cache_deficit`` ≤ 0.5 events/h,
occupancy-attributed ≤ 25 % of deadline events, capacity-attributed ≥ 50 %
⇒ not certified.

Usage::

    from wasted_compute import analyse_window
    wc = analyse_window(Path("/var/log/llama-proxy"), start, end)
    wc.verdict            # "pass" | "fail" | "inconclusive" | "unknown"
    wc.to_json()          # the ``--json`` ``wasted_compute`` block
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import ModuleType

LOGGER = logging.getLogger(__name__)

#: Producer-agreed user-class rate target (LP-0MVN3KMGI009BZ0D AC4 re-scope).
DEFAULT_RATE_TARGET_PER_HOUR = 0.5
#: Producer-agreed guards: occupancy share ceiling and capacity share floor.
OCCUPANCY_GUARD = 0.25
CAPACITY_FLOOR = 0.5
#: The two reasons counted together so a reason conversion cannot hide a defect.
COMPOSITE_REASONS = ("prefill_deadline", "measured_cache_deficit")
#: Attribution classes, in reporting order.
ATTRIBUTION_CLASSES = ("predicted-warmth", "capacity", "occupancy")

#: The stall-cost percentiles published per reason (LP-0MW7V30AZ007JB5Z AC2).
STALL_PERCENTILES = (50, 90, 95, 99)
#: Boundary below which an abort is a fast detour, not a stall (AC2 ``< 1 s``).
SUB_SECOND_STALL_S = 1.0
#: Boundary of the "short stall" class (AC2 ``< 5 s``).
SHORT_STALL_S = 5.0
#: Slack under the deadline that still counts as a full-deadline wait (AC2).
NEAR_DEADLINE_SLACK_S = 1.0
#: Fallback deadline when the analyser payload predates the ``deadline_seconds``
#: field: the shipped ``local_prefill_deadline_seconds`` at the time of writing.
DEFAULT_PREFILL_DEADLINE_S = 180.0
#: Env override for the analyser path (tests, non-source installs).
ANALYSER_ENV = "PROXY_USAGE_WASTED_COMPUTE_ANALYSER"
ANALYSER_RELATIVE = Path("proxy") / "scripts" / "analyse_prefill_deadline_residual.py"
DEFAULT_LOG_GLOB = "proxy.log*"
DEFAULT_LLAMA_GLOB = "llama-server*.log"


@dataclass(frozen=True)
class ReasonRow:
    """One post-acceptance eviction reason, split by session class."""

    reason: str
    total: int
    user: int
    audit: int
    per_hour: float | None
    user_per_hour: float | None

    def to_json(self) -> dict:
        return {
            "reason": self.reason,
            "total": self.total,
            "user": self.user,
            "audit": self.audit,
            "per_hour": self.per_hour,
            "user_per_hour": self.user_per_hour,
        }


@dataclass(frozen=True)
class StallRow:
    """Stall cost for one eviction reason, split by session class (AC1/AC2).

    ``stall_seconds_per_hour`` is ``Σ elapsed_s / window_hours`` — the figure
    that can reward making aborts faster, which an event rate cannot.
    """

    reason: str
    events: int
    user_events: int
    audit_events: int
    total_stall_seconds: float
    total_stall_seconds_per_hour: float | None
    user_stall_seconds: float
    user_stall_seconds_per_hour: float | None
    audit_stall_seconds: float
    audit_stall_seconds_per_hour: float | None
    user_percentiles: dict
    total_percentiles: dict
    user_buckets: dict

    @property
    def user_median_stall_seconds(self) -> float | None:
        return self.user_percentiles.get("p50")

    def to_json(self) -> dict:
        return {
            "reason": self.reason,
            "events": self.events,
            "user_events": self.user_events,
            "audit_events": self.audit_events,
            "total_stall_seconds": self.total_stall_seconds,
            "total_stall_seconds_per_hour": self.total_stall_seconds_per_hour,
            "user_stall_seconds": self.user_stall_seconds,
            "user_stall_seconds_per_hour": self.user_stall_seconds_per_hour,
            "audit_stall_seconds": self.audit_stall_seconds,
            "audit_stall_seconds_per_hour": self.audit_stall_seconds_per_hour,
            "user_percentiles": dict(self.user_percentiles),
            "total_percentiles": dict(self.total_percentiles),
            "user_buckets": dict(self.user_buckets),
        }


@dataclass(frozen=True)
class StallCost:
    """The window's stall-weighted cost, with the (b2) composite (AC1/AC2).

    ``composite_user_*`` merges the two reasons the producer counts together
    (:data:`COMPOSITE_REASONS`) for non-``audit`` sessions, so the report can
    show the same population the per-event (b2) rate scores — priced in
    seconds of user waiting instead of events.
    """

    hours: float | None
    deadline_seconds: float
    total_events: int
    total_stall_seconds: float
    stall_seconds_per_hour: float | None
    composite_user_events: int
    composite_user_stall_seconds: float
    composite_user_stall_seconds_per_hour: float | None
    composite_user_percentiles: dict
    composite_user_buckets: dict
    deadline_user_percentiles: dict
    deadline_user_buckets: dict
    rows: tuple[StallRow, ...] = ()

    @property
    def deadline_row(self) -> StallRow | None:
        for row in self.rows:
            if row.reason == "prefill_deadline":
                return row
        return None

    @property
    def deficit_row(self) -> StallRow | None:
        for row in self.rows:
            if row.reason == "measured_cache_deficit":
                return row
        return None

    def to_json(self) -> dict:
        return {
            "hours": self.hours,
            "deadline_seconds": self.deadline_seconds,
            "total_events": self.total_events,
            "total_stall_seconds": self.total_stall_seconds,
            "stall_seconds_per_hour": self.stall_seconds_per_hour,
            "composite_user_events": self.composite_user_events,
            "composite_user_stall_seconds": self.composite_user_stall_seconds,
            "composite_user_stall_seconds_per_hour": (
                self.composite_user_stall_seconds_per_hour
            ),
            "composite_user_percentiles": dict(self.composite_user_percentiles),
            "composite_user_buckets": dict(self.composite_user_buckets),
            "deadline_user_percentiles": dict(self.deadline_user_percentiles),
            "deadline_user_buckets": dict(self.deadline_user_buckets),
            "reasons": [row.to_json() for row in self.rows],
        }


@dataclass(frozen=True)
class WastedCompute:
    """The wasted-GPU-compute view of one window."""

    status: str = "unavailable"
    reason: str = ""
    hours: float | None = None
    reasons: tuple[ReasonRow, ...] = ()
    total_events: int = 0
    events_per_hour: float | None = None
    composite_user_events: int = 0
    composite_user_per_hour: float | None = None
    rate_target_per_hour: float = DEFAULT_RATE_TARGET_PER_HOUR
    rate_ok: bool | None = None
    attribution: dict[str, int] = field(default_factory=dict)
    attribution_total: int = 0
    occupancy_share: float = 0.0
    capacity_share: float = 0.0
    occupancy_ok: bool | None = None
    capacity_ok: bool | None = None
    overlap_events: int = 0
    overlap_total: int = 0
    overlap_basis: str = "unavailable"
    overlap_measured: bool = False
    releases_total: int = 0
    releases_observed: int = 0
    cancel_p50_s: float | None = None
    cancel_p90_s: float | None = None
    priced_tokens_discarded: int = 0
    tokens_after_cancel: int = 0
    wasted_tokens: int = 0
    backend_completed: int | None = None
    client_consumed: int | None = None
    coverage_verdict: str | None = None
    coverage_exact: bool = False
    capture_available: bool = False
    proxy_restarts: int = 0
    backend_restart_evictions: int = 0
    restart_free: bool = True
    verdict: str = "unknown"
    caveats: tuple[str, ...] = ()
    #: Stall-weighted cost of the same window (LP-0MW7V30AZ007JB5Z): what the
    #: event rate above cannot price. ``None`` when the analyser payload has no
    #: ``stall_cost`` block (a payload from before this surface existed).
    stall_cost: StallCost | None = None

    # -- derived presentation helpers -------------------------------------

    @property
    def available(self) -> bool:
        return self.status == "ok"

    def reason_row(self, reason: str) -> ReasonRow | None:
        for row in self.reasons:
            if row.reason == reason:
                return row
        return None

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "hours": self.hours,
            "reasons": [row.to_json() for row in self.reasons],
            "total_events": self.total_events,
            "events_per_hour": self.events_per_hour,
            "composite_user_events": self.composite_user_events,
            "composite_user_per_hour": self.composite_user_per_hour,
            "rate_target_per_hour": self.rate_target_per_hour,
            "rate_ok": self.rate_ok,
            "attribution": dict(self.attribution),
            "attribution_total": self.attribution_total,
            "occupancy_share": self.occupancy_share,
            "capacity_share": self.capacity_share,
            "occupancy_ok": self.occupancy_ok,
            "capacity_ok": self.capacity_ok,
            "abandoned_prefill_overlap": {
                "events": self.overlap_events,
                "total": self.overlap_total,
                "basis": self.overlap_basis,
                "measured_available": self.overlap_measured,
            },
            "release_cancel": {
                "releases_total": self.releases_total,
                "releases_with_observed_cancel": self.releases_observed,
                "cancel_to_stop_p50_s": self.cancel_p50_s,
                "cancel_to_stop_p90_s": self.cancel_p90_s,
            },
            "waste": {
                "priced_tokens_discarded": self.priced_tokens_discarded,
                "tokens_evaluated_after_cancel": self.tokens_after_cancel,
                "wasted_tokens": self.wasted_tokens,
                "backend_completed_prefills": self.backend_completed,
                "client_consumed_local_streams": self.client_consumed,
            },
            "coverage": {
                "verdict": self.coverage_verdict,
                "exact": self.coverage_exact,
            },
            "capture_available": self.capture_available,
            "proxy_restarts_in_window": self.proxy_restarts,
            "backend_restarts_in_window": self.backend_restart_evictions,
            "restart_free": self.restart_free,
            "verdict": self.verdict,
            "caveats": list(self.caveats),
            "stall_cost": (
                self.stall_cost.to_json() if self.stall_cost is not None else None
            ),
        }


# ---------------------------------------------------------------------------
# Analyser acquisition
# ---------------------------------------------------------------------------


def find_analyser(start: Path | None = None) -> Path | None:
    """Locate the committed prefill-deadline analyser, or ``None``.

    The override env var wins; otherwise walk up from this file (and from the
    working directory) looking for ``proxy/scripts/analyse_prefill_deadline_residual.py``.
    """
    override = os.environ.get(ANALYSER_ENV)
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None
    seeds = [start] if start is not None else []
    seeds.append(Path(__file__).resolve().parent)
    seeds.append(Path.cwd())
    for seed in seeds:
        for parent in [seed, *seed.parents][:8]:
            candidate = parent / ANALYSER_RELATIVE
            if candidate.is_file():
                return candidate
    return None


_ANALYSER_CACHE: dict[str, ModuleType] = {}


def load_analyser(path: Path | None = None) -> ModuleType | None:
    """Import the analyser by path (never via ``sys.path``), cached."""
    target = path or find_analyser()
    if target is None:
        return None
    key = str(target)
    cached = _ANALYSER_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        spec = importlib.util.spec_from_file_location(
            "_wasted_compute_analyser", target
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        LOGGER.warning("wasted_compute: analyser import failed: %s", exc)
        return None
    _ANALYSER_CACHE[key] = module
    return module


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyse_window(
    log_dir: Path | str,
    start: datetime,
    end: datetime,
    *,
    llama_log_dir: Path | str | None = None,
    rate_target_per_hour: float = DEFAULT_RATE_TARGET_PER_HOUR,
    analyser: ModuleType | None = None,
) -> WastedCompute:
    """Analyse one window and return its :class:`WastedCompute`.

    Never raises: any failure yields ``status="unavailable"`` with the reason
    recorded, so the report renders a caveat instead of losing the whole run.
    """
    log_dir = Path(log_dir)
    module = analyser if analyser is not None else load_analyser()
    if module is None:
        return WastedCompute(
            status="unavailable",
            reason=(
                "analyser not found (proxy/scripts/analyse_prefill_deadline_residual.py)"
            ),
        )
    paths = sorted(log_dir.glob(DEFAULT_LOG_GLOB))
    if not paths:
        return WastedCompute(
            status="unavailable",
            reason=f"no {DEFAULT_LOG_GLOB} files in {log_dir}",
        )
    llama_spec = _llama_spec(llama_log_dir or log_dir)
    try:
        failures: list[str] = []
        lines = module.read_log_lines(paths, failures)
        log = module.parse_window(lines, start, end)
        log.files_read = len(paths) - len(failures)
        dataset = module.build_report(
            log,
            start,
            end,
            log_dir=str(log_dir),
            llama_log=llama_spec,
        )
    except Exception as exc:
        LOGGER.warning("wasted_compute: analysis failed: %s", exc)
        return WastedCompute(status="unavailable", reason=f"analysis failed: {exc}")
    return normalise(dataset, rate_target_per_hour=rate_target_per_hour)


def _llama_spec(llama_log_dir: Path | str) -> str | None:
    directory = Path(llama_log_dir)
    if not any(directory.glob(DEFAULT_LLAMA_GLOB)):
        return None
    return str(directory / DEFAULT_LLAMA_GLOB)


def normalise(
    dataset: dict, *, rate_target_per_hour: float = DEFAULT_RATE_TARGET_PER_HOUR
) -> WastedCompute:
    """Turn an analyser payload into a :class:`WastedCompute` (pure)."""
    if not isinstance(dataset, dict):
        return WastedCompute(
            status="unavailable", reason="analyser returned no dataset"
        )
    hours = _opt_float((dataset.get("window") or {}).get("hours"))
    if not hours or hours <= 0:
        return WastedCompute(
            status="unavailable", reason="analysis window has no duration"
        )

    split = dataset.get("reason_split") or {}
    reasons: list[ReasonRow] = []
    for reason, row in split.items():
        total = int(row.get("total") or 0)
        reasons.append(
            ReasonRow(
                reason=str(reason),
                total=total,
                user=int(row.get("user") or 0),
                audit=int(row.get("audit") or 0),
                per_hour=_opt_float(row.get("total_per_hour")),
                user_per_hour=_opt_float(row.get("user_per_hour")),
            )
        )
    reasons.sort(key=lambda row: (-row.total, row.reason))

    attribution_block = dataset.get("attribution") or {}
    totals = attribution_block.get("totals") or {}
    attribution = {
        name: int(totals.get(name) or 0) for name in ATTRIBUTION_CLASSES
    }
    attribution_total = sum(attribution.values())
    occupancy_share = _share(attribution["occupancy"], attribution_total)
    capacity_share = _share(attribution["capacity"], attribution_total)

    overlap = attribution_block.get("abandoned_prefill_overlap") or {}
    release = dataset.get("release_cancel_accounting") or {}
    cancel_seconds = release.get("cancel_to_stop_seconds") or {}
    backend = dataset.get("backend_prefill_accounting") or {}
    coverage = (dataset.get("abandoned_prefill_windows") or {}).get("coverage") or {}

    events = dataset.get("events") or []
    priced_tokens = sum(
        int(event.get("new_tokens") or 0)
        for event in events
        if event.get("new_tokens") is not None
    )
    after_cancel = int((release.get("tokens_evaluated_after_cancel") or {}).get("total") or 0)

    composite_reasons = [
        row for row in reasons if row.reason in COMPOSITE_REASONS
    ]
    composite_events = sum(row.user for row in composite_reasons)
    composite_rate = composite_events / hours
    rate_ok = composite_rate <= rate_target_per_hour

    evictions = dataset.get("evictions") or {}
    total_events = int(evictions.get("total") or 0)
    backend_rows = [
        row for row in reasons if row.reason == "backend_restart"
    ]
    backend_restarts = sum(row.total for row in backend_rows)
    proxy_restarts = len((dataset.get("source") or {}).get("restarts_in_window") or [])
    restart_free = proxy_restarts == 0 and backend_restarts == 0

    coverage_verdict = coverage.get("verdict")
    coverage_exact = coverage_verdict == "exact"
    capture_available = bool(attribution_block.get("capture_available"))

    verdict = _verdict(
        rate_ok=rate_ok,
        occupancy_share=occupancy_share,
        capacity_share=capacity_share,
        restart_free=restart_free,
        attribution_total=attribution_total,
    )

    caveats: list[str] = []
    if not capture_available:
        caveats.append(
            "no measured prefill-progress capture in this window: the "
            "attribution rests on the admission price alone"
        )
    if coverage_verdict is None:
        caveats.append(
            "no llama-server log supplied: abandoned-prefill overlap, "
            "release->cancel latency and backend waste are unavailable"
        )
    elif not coverage_exact:
        caveats.append(
            "llama-server log coverage is "
            f"'{coverage_verdict}' ({coverage.get('entries')} backend segments "
            f"joined against {coverage.get('releases')} releases): the "
            "per-release and backend figures are indicative, not exact"
        )
    if proxy_restarts:
        caveats.append(
            f"window contains {proxy_restarts} proxy restart(s): a restart "
            "re-prices capacity and discards prefills, so the window cannot "
            "certify anything"
        )
    if backend_restarts:
        caveats.append(
            f"window contains {backend_restarts} backend restart(s) that "
            "discarded in-flight dispatches: not certifiable (a restart with no "
            "in-flight dispatch leaves no eviction event, so this is a lower "
            "bound)"
        )
    if not restart_free:
        caveats.append(
            "restart-free window required for a certifiable rate; use a "
            "window that starts after the last proxy/backend restart"
        )

    return WastedCompute(
        status="ok",
        reason="",
        hours=hours,
        reasons=tuple(reasons),
        total_events=total_events,
        events_per_hour=_opt_float(evictions.get("per_hour")),
        composite_user_events=composite_events,
        composite_user_per_hour=composite_rate,
        rate_target_per_hour=rate_target_per_hour,
        rate_ok=rate_ok,
        attribution=attribution,
        attribution_total=attribution_total,
        occupancy_share=occupancy_share,
        capacity_share=capacity_share,
        occupancy_ok=occupancy_share <= OCCUPANCY_GUARD,
        capacity_ok=capacity_share < CAPACITY_FLOOR,
        overlap_events=int(overlap.get("events") or 0),
        overlap_total=int(overlap.get("total") or 0),
        overlap_basis=str(overlap.get("basis") or "unavailable"),
        overlap_measured=bool(overlap.get("measured_available")),
        releases_total=int(release.get("releases_total") or 0),
        releases_observed=int(release.get("releases_with_observed_cancel") or 0),
        cancel_p50_s=_opt_float(cancel_seconds.get("p50")),
        cancel_p90_s=_opt_float(cancel_seconds.get("p90")),
        priced_tokens_discarded=priced_tokens,
        tokens_after_cancel=after_cancel,
        wasted_tokens=priced_tokens + after_cancel,
        backend_completed=_opt_int(backend.get("backend_completed_prefills")),
        client_consumed=_opt_int(backend.get("proxy_consumed_local_streams")),
        coverage_verdict=coverage_verdict,
        coverage_exact=coverage_exact,
        capture_available=capture_available,
        proxy_restarts=proxy_restarts,
        backend_restart_evictions=backend_restarts,
        restart_free=restart_free,
        verdict=verdict,
        caveats=tuple(caveats),
        stall_cost=normalise_stall_cost(
            dataset.get("stall_cost"), hours=hours
        ),
    )


# ---------------------------------------------------------------------------
# Stall-weighted cost (LP-0MW7V30AZ007JB5Z)
# ---------------------------------------------------------------------------


def _percentiles(values, points=STALL_PERCENTILES) -> dict:
    """Percentile block over *values* (same linear interpolation as elsewhere).

    Deliberately identical to ``analyse_prefill_deadline_residual._percentiles``
    and ``reporting._percentile`` so the composite computed here cannot
    disagree with the per-reason figures the analyser published (a test pins
    that agreement).
    """
    numbers = sorted(float(v) for v in values if v is not None)
    if not numbers:
        return {"count": 0, **{f"p{p}": None for p in points}, "min": None,
                "max": None, "mean": None}
    result: dict = {"count": len(numbers)}
    for point in points:
        rank = (len(numbers) - 1) * (point / 100.0)
        low = int(rank)
        high = min(low + 1, len(numbers) - 1)
        value = numbers[low] + (numbers[high] - numbers[low]) * (rank - low)
        result[f"p{point}"] = round(value, 3)
    result["min"] = round(numbers[0], 3)
    result["max"] = round(numbers[-1], 3)
    result["mean"] = round(sum(numbers) / len(numbers), 3)
    return result


def _stall_buckets(values, deadline_seconds: float) -> dict:
    """The ``< 1 s`` / ``< 5 s`` / ``>= deadline - 1 s`` split (AC2)."""
    threshold = round(float(deadline_seconds) - NEAR_DEADLINE_SLACK_S, 3)
    numbers = [float(v) for v in values]
    total = len(numbers)

    def _bucket(count: int) -> dict:
        return {"count": count, "share": round(count / total, 4) if total else 0.0}

    near = _bucket(sum(1 for v in numbers if v >= threshold))
    near["threshold_s"] = threshold
    return {
        "under_1s": _bucket(sum(1 for v in numbers if v < SUB_SECOND_STALL_S)),
        "under_5s": _bucket(sum(1 for v in numbers if v < SHORT_STALL_S)),
        "at_or_near_deadline": near,
    }


def _values_of(row: dict | None) -> list[float]:
    """The raw per-class elapsed seconds the analyser published for a reason."""
    if not row:
        return []
    values = row.get("values")
    if not isinstance(values, list):
        return []
    return [float(value) for value in values if _opt_float(value) is not None]


def normalise_stall_cost(block, *, hours: float | None) -> StallCost | None:
    """Turn the analyser's ``stall_cost`` payload into a :class:`StallCost`.

    Pure, and tolerant of an older payload: a dataset without ``stall_cost``
    yields ``None`` (the report then omits the stall rows instead of printing
    zeros it did not measure).
    """
    if not isinstance(block, dict) or not block:
        return None
    by_reason = block.get("by_reason") or {}
    if not isinstance(by_reason, dict):
        return None
    deadline_seconds = _opt_float(block.get("deadline_seconds"))
    if deadline_seconds is None:
        deadline_seconds = DEFAULT_PREFILL_DEADLINE_S
    window_hours = _opt_float(block.get("hours")) or hours

    rows: list[StallRow] = []
    composite_values: list[float] = []
    for reason, payload in by_reason.items():
        if not isinstance(payload, dict):
            continue
        classes = payload.get("by_session_class") or {}
        user = classes.get("user") or {}
        audit = classes.get("audit") or {}
        rows.append(
            StallRow(
                reason=str(reason),
                events=int(payload.get("events") or 0),
                user_events=int(user.get("events") or 0),
                audit_events=int(audit.get("events") or 0),
                total_stall_seconds=_float_or_zero(payload.get("stall_seconds")),
                total_stall_seconds_per_hour=_opt_float(
                    payload.get("stall_seconds_per_hour")
                ),
                user_stall_seconds=_float_or_zero(user.get("stall_seconds")),
                user_stall_seconds_per_hour=_opt_float(
                    user.get("stall_seconds_per_hour")
                ),
                audit_stall_seconds=_float_or_zero(audit.get("stall_seconds")),
                audit_stall_seconds_per_hour=_opt_float(
                    audit.get("stall_seconds_per_hour")
                ),
                user_percentiles=dict(user.get("percentiles") or {}),
                total_percentiles=dict(payload.get("percentiles") or {}),
                user_buckets=dict(user.get("buckets") or {}),
            )
        )
        if reason in COMPOSITE_REASONS:
            composite_values.extend(_values_of(user))
    rows.sort(key=lambda row: (-row.total_stall_seconds, row.reason))

    total = block.get("total") or {}
    deadline_row = next(
        (row for row in rows if row.reason == "prefill_deadline"), None
    )
    return StallCost(
        hours=window_hours,
        deadline_seconds=deadline_seconds,
        total_events=int(total.get("events") or 0),
        total_stall_seconds=_float_or_zero(total.get("stall_seconds")),
        stall_seconds_per_hour=_opt_float(total.get("stall_seconds_per_hour")),
        composite_user_events=len(composite_values),
        composite_user_stall_seconds=round(sum(composite_values), 3),
        composite_user_stall_seconds_per_hour=(
            round(sum(composite_values) / window_hours, 3) if window_hours else None
        ),
        composite_user_percentiles=_percentiles(composite_values),
        composite_user_buckets=_stall_buckets(composite_values, deadline_seconds),
        deadline_user_percentiles=(
            dict(deadline_row.user_percentiles) if deadline_row else {}
        ),
        deadline_user_buckets=(
            dict(deadline_row.user_buckets) if deadline_row else {}
        ),
        rows=tuple(rows),
    )


def _float_or_zero(value) -> float:
    number = _opt_float(value)
    return round(number, 3) if number is not None else 0.0


def _verdict(
    *,
    rate_ok: bool,
    occupancy_share: float,
    capacity_share: float,
    restart_free: bool,
    attribution_total: int,
) -> str:
    """Overall verdict: ``pass`` / ``fail`` / ``inconclusive`` / ``unknown``."""
    if attribution_total == 0 and rate_ok:
        # No deadline events at all: nothing was attributed, nothing to fail.
        return "pass" if restart_free else "inconclusive"
    if not rate_ok or occupancy_share > OCCUPANCY_GUARD or capacity_share >= CAPACITY_FLOOR:
        return "fail"
    return "pass" if restart_free else "inconclusive"


def _share(part: int, total: int) -> float:
    return (part / total) if total else 0.0


def _opt_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
