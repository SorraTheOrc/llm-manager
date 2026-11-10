"""Cold-prefill thrash-window replay harness (LP-0MVN4LAV9007PM2X).

The parent item (LP-0MVN3KMGI009BZ0D) argues that the proxy admits cold
prefills it cannot possibly finish before ``local_prefill_deadline_seconds``
and then throws the computed tokens away. Its measured evidence was an
anecdote in prose. This module turns that anecdote into an executable
baseline:

* :func:`load_fixture` reads the checked-in capture of the 2026-10-24
  18:38–23:47 thrash window (one JSONL header + one record per local
  admission attempt, captured by
  ``proxy/scripts/capture_prefill_thrash_window.py``).
* :func:`replay` feeds every attempt through the **real** admission path
  (:func:`proxy.prefill_eta.evaluate_admission`) and classifies the result, so
  later children (budget cap, deadline-relative pricing, eviction counter) can
  assert their post-change behaviour against the same oracle without
  re-parsing production logs.
* :func:`replay_result_to_json` serialises a replay so a failure or a report
  can carry the numbers instead of an eyeball verdict (AC3).
* :func:`measure_admission_overhead` replays the same fixture through the
  **real local admission path** and aggregates the production
  ``admission_overhead_ms=`` figure into p50/p95, with the budget verdict
  (LP-0MW959R5S009WDM8, parent AC4). It reuses this fixture and this module's
  replay discipline rather than standing up a second harness.

Two documented modelling decisions:

1. **The pinned calibration profile** (:data:`PINNED_BAND_RATE_TPS`) prices the
   measured 20–30k band at its p75 prompt-eval rate (270 tokens/s, from the
   parent item's measured table). The estimator fits *one* coefficient per
   context bucket and the 20–30k, 30–40k and 48k bands all fall in the same
   16k–64k bucket, so a bucket-linear fit cannot express the measured knee.
   The pinned profile is therefore the band-anchored lower envelope the
   parent's AC3 maths uses (25k ≈ 93 s at 270 t/s) and it is applied to every
   prompt size in that bucket; :func:`band_rate` keeps the measured band table
   for the *waste* model, which is what AGENTS.md's "no re-implemented
   production logic" rule still allows here (it is a measurement, not the
   production decision).
2. **Deadline risk** is priced from that same pinned band table rather than
   from ``evaluate_admission``'s verdict, because the fixture measures what
   production *did* (125 deadline cancellations) and not what an uncalibrated
   guard *would* have decided. The guard's own verdict is replayed separately
   and recorded per attempt.

This module is an analysis tool: no production module may import it (AC5).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from proxy import prefill_eta

#: Sentinel attribute used by the AC5 test to prove no production module
#: imported the harness (an import of this module would copy the name).
REPLAY_HARNESS_SENTINEL = "cold-prefill-thrash-window-replay"

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "prefill_thrash_2026_10_24.jsonl"
)
#: The fixture must stay small enough to live in the repo (AC4).
MAX_FIXTURE_BYTES = 2 * 1024 * 1024

#: Measured prompt-eval rates (tokens/second) per prompt-size band, this host,
#: 2026-10-24/25 (268 llama-server samples). Used by the waste model.
BAND_RATES: tuple[tuple[int, float], ...] = (
    (10_000, 281.0),
    (20_000, 281.0),
    (30_000, 249.0),
    (40_000, 181.0),
    (50_000, 152.0),
    (10**9, 114.0),
)

#: p75 of the 20–30k band — the rate the pinned calibration profile prices at
#: (the parent item's AC3 maths: 25k ≈ 93 s, inside 0.95 × 120 s = 114 s).
PINNED_BAND_RATE_TPS = 270.0
#: Prompt sizes the pinned profile seeds its calibration window with (spread
#: inside the 20–30k band so the fit identifies the coefficient).
PINNED_PROFILE_TOKENS = (20_000, 22_000, 24_000, 26_000, 28_000, 30_000)
#: Samples recorded per seeded size; above the guard's ``min_samples`` (5).
PINNED_PROFILE_REPEATS = 3

#: Deadline the fixture window ran under (production value on 2026-10-24).
FIXTURE_DEADLINE_SECONDS = 90.0

MODEL = "Qwen3"

#: Budget for the latency candidate-first admission adds to the request hot
#: path (parent AC4): the p95 of the per-admission overhead, measured on a
#: replay, must stay at or below this. The added work is one in-process
#: registry lookup plus the residency resolution — no upstream query, no
#: poller, no background task — so 50 ms is a generous ceiling.
ADMISSION_OVERHEAD_BUDGET_MS: float = 50.0

#: The production decision-line field this harness reads back. Production owns
#: the name (``proxy.provider.ADMISSION_OVERHEAD_FIELD``); the F5 regression
#: pin asserts the two agree, so a rename cannot leave the replay measuring a
#: field nothing emits.
ADMISSION_OVERHEAD_TOKEN = "admission_overhead_ms="

#: Log line the per-admission overhead is attached to (the admission verdict).
ADMISSION_DECISION_TOKEN = "routing_resident_prefix"


def band_rate(prompt_tokens: int) -> float:
    """Measured prompt-eval rate (tokens/s) for a prompt of *prompt_tokens*."""
    for upper, rate in BAND_RATES:
        if prompt_tokens <= upper:
            return rate
    return BAND_RATES[-1][1]


def dev_guard_config(**server_overrides: Any) -> dict:
    """Config reproducing the shipped ``dev`` admission settings.

    ``local_prefill_deadline_seconds`` stays at the fixture's 90 s and the
    cold-prefill budget is absent (it does not exist on ``dev`` yet), so a
    replay with this config measures *today's* behaviour.
    """
    server: dict[str, Any] = {
        "session_slot_pool_size": 2,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": 60,
        "contention_queue_max_depth": 8,
        "local_prefill_deadline_seconds": int(FIXTURE_DEADLINE_SECONDS),
        "local_admission_guard_policy": "reject",
        "local_admission_guard_min_samples": 5,
        "local_admission_guard_window_samples": 200,
        "local_admission_guard_reference_tokens": 32_000,
        "local_admission_guard_eta_percentile": 75,
    }
    server.update(server_overrides)
    return {"server": server}


def seed_pinned_profile(
    tokens: Iterable[int] = PINNED_PROFILE_TOKENS,
    rate_tps: float = PINNED_BAND_RATE_TPS,
    repeats: int = PINNED_PROFILE_REPEATS,
    model: str = MODEL,
) -> None:
    """Seed the estimator's rolling window with the pinned band profile.

    Each seeded dispatch is cache-cold and takes ``tokens / rate_tps``
    seconds, so the fitted per-token cost is ``1000 / rate_tps`` ms.
    """
    prefill_eta.reset()
    for prompt_tokens in tokens:
        service_ms = (prompt_tokens / rate_tps) * 1000.0
        for _ in range(repeats):
            prefill_eta.record_dispatch_sample(
                model, prompt_tokens, 0.0, service_ms
            )


def seed_production_like_profile() -> None:
    """Leave the calibration window empty — the state production was in.

    The fixture window's guard decisions were inert for most traffic
    (``local_admission_guard`` appears 19 times across 1 978 attempts), which
    is what "uncalibrated bucket ⇒ fail-open" means in practice.
    """
    prefill_eta.reset()


@dataclass
class ReplayAttempt:
    """One replayed admission attempt."""

    ts: str
    session: str
    route: str
    estimated_tokens: int
    cached_ratio: float
    new_tokens: int
    outcome: str
    guard_reason: str
    guard_admit: bool
    #: Modelled prefill seconds at the pinned band rate for ``new_tokens``.
    modelled_prefill_seconds: float
    #: True when the modelled prefill cannot finish inside the deadline.
    deadline_risk: bool
    #: Tokens the deadline would compute before cancelling this prefill.
    wasted_tokens: float


@dataclass
class ReplayResult:
    """Structured replay outcome for later assertions (AC3)."""

    attempts: list[ReplayAttempt]
    deadline_seconds: float
    config_deadline_seconds: float
    budget_tokens: int
    totals: dict[str, Any] = field(default_factory=dict)

    @property
    def deadline_cancelled(self) -> list[ReplayAttempt]:
        """Attempts the fixture recorded as cancelled at the deadline."""
        return [a for a in self.attempts if a.outcome == "prefill_deadline_exceeded"]

    @property
    def admitted_and_doomed(self) -> list[ReplayAttempt]:
        """Attempts the admission path admits although they cannot finish."""
        return [
            a
            for a in self.attempts
            if a.guard_admit and a.deadline_risk and a.route != "large_context_bypass"
        ]

    @property
    def wasted_tokens(self) -> float:
        """Modelled tokens computed then discarded by deadline cancellations.

        Costed per distinct cancelled prefill (``session``, prompt size): a
        session that re-attempts the same prompt pays the prefill again, but
        the fixture's waste model — like the parent item's baseline — costs
        one discarded prefill per distinct task.
        """
        return _waste_per_task(self.deadline_cancelled)

    @property
    def wasted_tokens_admitted(self) -> float:
        """Modelled waste among attempts today's path admits (per task)."""
        return _waste_per_task(self.admitted_and_doomed)


def _waste_per_task(attempts: list[ReplayAttempt]) -> float:
    """Sum modelled waste over distinct (session, prompt-size) tasks."""
    tasks: dict[tuple[str, int], float] = {}
    for attempt in attempts:
        tasks.setdefault(
            (attempt.session, attempt.estimated_tokens), attempt.wasted_tokens
        )
    return sum(tasks.values())


def load_fixture(path: Path | None = None) -> tuple[dict[str, Any], list[dict]]:
    """Load the checked-in fixture: ``(header_meta, records)``."""
    fixture = Path(path) if path is not None else FIXTURE_PATH
    with fixture.open(encoding="utf-8") as handle:
        first = handle.readline()
        header = json.loads(first)
        records = [json.loads(line) for line in handle if line.strip()]
    if "_meta" not in header:  # pragma: no cover - fixture corruption alarm
        raise AssertionError("fixture is missing its `_meta` header line")
    return header["_meta"], records


def replay(
    records: list[dict],
    *,
    config: dict | None = None,
    deadline_seconds: float = FIXTURE_DEADLINE_SECONDS,
    model: str = MODEL,
) -> ReplayResult:
    """Replay *records* through the real admission path.

    The caller seeds the calibration window first (:func:`seed_pinned_profile`
    or :func:`seed_production_like_profile`) so the replay is reproducible.
    """
    cfg = config or dev_guard_config()
    server = cfg.get("server", {})
    config_deadline = float(
        server.get(prefill_eta.PREFILL_DEADLINE_KEY, deadline_seconds)
    )
    budget_tokens = int(server.get("local_cold_prefill_budget_tokens", 0) or 0)

    attempts: list[ReplayAttempt] = []
    for record in records:
        estimated = int(record["estimated_tokens"])
        ratio = float(record["cached_ratio"])
        new_tokens = int(record["new_tokens"])
        decision = prefill_eta.evaluate_admission(
            cfg, model, estimated, ratio, priority=record.get("priority", "medium")
        )
        rate = band_rate(new_tokens)
        modelled = new_tokens / rate
        attempts.append(
            ReplayAttempt(
                ts=str(record.get("ts", "")),
                session=str(record.get("session", "")),
                route=str(record.get("route", "queue")),
                estimated_tokens=estimated,
                cached_ratio=ratio,
                new_tokens=new_tokens,
                outcome=str(record.get("outcome", "other")),
                guard_reason=str(decision.reason),
                guard_admit=bool(decision.admit),
                modelled_prefill_seconds=modelled,
                deadline_risk=modelled > deadline_seconds,
                wasted_tokens=min(new_tokens, deadline_seconds * rate),
            )
        )

    result = ReplayResult(
        attempts=attempts,
        deadline_seconds=deadline_seconds,
        config_deadline_seconds=config_deadline,
        budget_tokens=budget_tokens,
    )
    result.totals = {
        "attempts": len(attempts),
        "guard_admitted": sum(1 for a in attempts if a.guard_admit),
        "guard_refused": sum(1 for a in attempts if not a.guard_admit),
        "deadline_cancelled": len(result.deadline_cancelled),
        "deadline_cancelled_tasks": len(
            {(a.session, a.estimated_tokens) for a in result.deadline_cancelled}
        ),
        "admitted_and_doomed": len(result.admitted_and_doomed),
        "wasted_tokens": round(result.wasted_tokens),
        "wasted_tokens_admitted": round(result.wasted_tokens_admitted),
        "idle_bypass_attempts": sum(1 for a in attempts if a.route == "idle_bypass"),
        "idle_bypass_over_25k": sum(
            1
            for a in attempts
            if a.route == "idle_bypass" and a.new_tokens > 25_000
        ),
    }
    return result


def replay_result_to_json(result: ReplayResult) -> dict[str, Any]:
    """Serialise a replay for machine-readable output (CLI/API convention)."""
    return {
        "totals": dict(result.totals),
        "deadline_seconds": result.deadline_seconds,
        "config_deadline_seconds": result.config_deadline_seconds,
        "budget_tokens": result.budget_tokens,
        "attempts": [
            {
                "ts": a.ts,
                "session": a.session,
                "route": a.route,
                "estimated_tokens": a.estimated_tokens,
                "cached_ratio": a.cached_ratio,
                "new_tokens": a.new_tokens,
                "outcome": a.outcome,
                "guard_reason": a.guard_reason,
                "guard_admit": a.guard_admit,
                "modelled_prefill_seconds": round(a.modelled_prefill_seconds, 1),
                "deadline_risk": a.deadline_risk,
            }
            for a in result.attempts
        ],
    }


# ---------------------------------------------------------------------------
# Admission-overhead replay (LP-0MW959R5S009WDM8, parent AC4)
#
# Candidate-first admission adds two steps to the request hot path: the slot
# candidate acquisition and the residency resolution that prices the verdict
# against it. The production decision line reports that cost per admission as
# ``admission_overhead_ms=``; this section replays the checked-in fixture
# through the real admission path and turns those per-admission figures into
# the p50/p95 the budget is enforced against.
#
# The harness owns the fixture, the sampling and the percentile/budget maths;
# the *caller* owns the ladder-specific fake pool state (it passes ``decide``),
# so no second residency ladder is stood up here (F1 AC5c).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmissionOverheadReport:
    """Per-admission overhead samples plus the AC4 p50/p95 budget verdict.

    ``p50_ms``/``p95_ms`` are ``None`` only when nothing was measured (no
    sample carried the field) — never coerced to a fabricated zero, so a
    silently missing measurement reads as *unmeasured* rather than *fast*.
    """

    samples_ms: tuple[float, ...]
    attempts: int
    measured: int
    budget_ms: float = ADMISSION_OVERHEAD_BUDGET_MS
    p50_ms: float | None = None
    p95_ms: float | None = None
    provenance: str = ""

    @property
    def over_budget(self) -> bool:
        """True when the measured p95 exceeds the budget (the AC4 failure)."""
        return self.p95_ms is not None and self.p95_ms > self.budget_ms

    @property
    def within_budget(self) -> bool:
        """True only when the budget was **measured and met**.

        Deliberately not the negation of :attr:`over_budget`: a report with no
        measurement is neither — it is *unmeasured*, so a silently absent
        figure can never read as a passed budget (LP-0MW959R5S009WDM8 AC4d).
        """
        return self.p95_ms is not None and self.p95_ms <= self.budget_ms

    def to_json(self) -> dict[str, Any]:
        """Machine-readable form (CLI/API convention)."""
        return {
            "attempts": self.attempts,
            "measured": self.measured,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "max_ms": max(self.samples_ms) if self.samples_ms else None,
            "budget_ms": self.budget_ms,
            "over_budget": self.over_budget,
            "within_budget": self.within_budget,
            "provenance": self.provenance,
        }


def percentile(samples: Iterable[float], p: float) -> float | None:
    """Nearest-rank *p*-th percentile of *samples*, or ``None`` when empty.

    Nearest rank (``ceil(p/100 * n)``-th smallest) rather than interpolation: a
    latency percentile is a *measured* sample, so the reported p95 is always a
    duration the admission path actually paid rather than an average of two.
    """
    values = sorted(float(sample) for sample in samples)
    if not values:
        return None
    if p <= 0:
        return values[0]
    rank = math.ceil((p / 100.0) * len(values)) - 1
    return values[min(max(rank, 0), len(values) - 1)]


def admission_overhead_report(
    samples: Iterable[float],
    *,
    attempts: int | None = None,
    budget_ms: float = ADMISSION_OVERHEAD_BUDGET_MS,
    provenance: str = "",
) -> AdmissionOverheadReport:
    """Aggregate per-admission overhead samples into the AC4 budget verdict."""
    values = tuple(float(sample) for sample in samples)
    return AdmissionOverheadReport(
        samples_ms=values,
        attempts=len(values) if attempts is None else int(attempts),
        measured=len(values),
        budget_ms=float(budget_ms),
        p50_ms=percentile(values, 50),
        p95_ms=percentile(values, 95),
        provenance=provenance,
    )


class _AdmissionOverheadCollector(logging.Handler):
    """Collect the ``admission_overhead_ms=`` figures production emits."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.samples: list[float] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - malformed record
            return
        if ADMISSION_DECISION_TOKEN not in message:
            return
        index = message.find(ADMISSION_OVERHEAD_TOKEN)
        if index < 0:
            return
        token = message[index + len(ADMISSION_OVERHEAD_TOKEN) :].split()
        if not token or token[0] == "none":
            return
        try:
            self.samples.append(float(token[0]))
        except ValueError:  # pragma: no cover - production never emits this
            return


def measure_admission_overhead(
    attempts: Iterable[Any],
    decide: Callable[[Any], Any],
    *,
    budget_ms: float = ADMISSION_OVERHEAD_BUDGET_MS,
    logger_name: str = "llama-proxy",
    provenance: str = "",
) -> AdmissionOverheadReport:
    """Replay *attempts* through the real admission path; aggregate its cost.

    *decide* runs **one** admission for a fixture record (the caller installs
    the ladder-specific fake pool state, because that is the residency ladder's
    surface, not this harness's). The per-admission figure is read back from the
    production decision line itself, so the replay measures the same number the
    operator sees — not a re-implementation of it
    (LP-0MW959R5S009WDM8 AC4a/AC4c/AC4e).

    Returns the p50/p95 report; ``over_budget`` is the AC4 failure and
    ``p50_ms``/``p95_ms`` are ``None`` when no attempt reported a figure.
    """
    logger = logging.getLogger(logger_name)
    collector = _AdmissionOverheadCollector()
    previous_level = logger.level
    logger.addHandler(collector)
    logger.setLevel(logging.INFO)
    count = 0
    try:
        for attempt in attempts:
            decide(attempt)
            count += 1
    finally:
        logger.removeHandler(collector)
        logger.setLevel(previous_level)
    return admission_overhead_report(
        collector.samples,
        attempts=count,
        budget_ms=budget_ms,
        provenance=provenance,
    )
