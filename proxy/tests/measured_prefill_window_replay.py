"""Measured-prefill reference-window replay harness (F5).

LP-0MVR7IXAI002QXXD (parent LP-0MVQIT9QY007QU27, Option D).

The parent's reference window (2026-10-26 09:06 → 2026-10-27 09:06) is the
evidence the measured-cache early abort is judged against: 120
``local_eviction_after_acceptance`` events, 29 of them "fake-warm" (predicted
reuse ≥ 0.99, priced ``new_tokens`` 17–300, cancelled at the 120 s deadline).

What this harness does
----------------------

:func:`load_fixture` reads the checked-in capture
(``fixtures/measured_prefill_window_2026_10_26.jsonl``, produced by
``proxy/scripts/capture_measured_prefill_window.py``): the eviction events
plus the measured ``n_past`` / ``slot.prompt.tokens.size()`` evidence
population from the same llama-server build.

:func:`replay` feeds every event through the **real** decision
(:func:`proxy.measured_cache_abort.decide_measured_cache_abort`) with the
window's own measured cache classes:

``reprefill``
    The measured re-prefill floor: the smallest ``n_past`` llama-server
    reported for a prompt in the event's size band (measured 356–6 919 against
    18k–78k prompts). This is the class the fix targets — the KV-cache
    checkpoint was invalidated and the prompt is being computed from scratch.

``best_case``
    The measured best case: the largest reuse llama-server reported in the
    size band (up to ``n_past = total - 1``, observed repeatedly). This bound
    the working warm population, which the abort must never touch; where the
    best observed reuse is still poor, the dispatch genuinely needs the work.

llama-server log lines carry no timestamps, so a proxy event cannot be joined
to a specific llama-server task; the harness therefore replays **both**
measured classes and reports the outcome for each instead of inventing a
per-event cache. That keeps the claim auditable: the classification is a
function of measured ground truth from the same window, not of a guess.

Waste model
-----------

``waste_before`` is the prefill compute discarded by a deadline cancellation:
``min(measured_new_tokens, elapsed_s × prefill rate p10)`` — the work
llama-server can have computed in the time it was allowed (p10 prompt-eval
rate 43.1 tok/s measured on this host, parent finding 4). ``waste_after`` is
0 for an aborted dispatch because the F1 contract pins the first
``prompt_progress`` event to before any prefill compute (``processed = 0``),
so the abort discards nothing but the detour latency.

Incident replay (the benefit gate)
----------------------------------

:func:`load_incident_fixture` / :func:`replay_incident` replay the **five
recorded aborts** of the 2026-10-31 pin-local incident
(LP-0MVWDZ1RH006HTFS, parent LP-0MVWBD91Z004U135 AC8) through the same real
decision, with the *benefit gate* configured from the shipped
``proxy/config.yaml``. Pre-gate every one of those five rows aborts (measured
deficit 14 380 tokens, admission-priced 48) and arms a 600 s thrash cooldown;
with the gate the measured cold ETA (56.1 s at the shipped default fit) fits
the 114 s prefill-deadline budget, so the row continues and no cooldown is
armed. The replay is the numeric oracle the wiring tests assert against.

This module is analysis/test tooling: no production module may import it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml
from proxy.measured_cache_abort import (
    ENABLED_KEY,
    MEASURED_CACHE_DEFICIT_REASON,
    AbortDecision,
    AbortSettings,
    decide_measured_cache_abort,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "measured_prefill_window_2026_10_26.jsonl"
)

#: The five recorded aborts of the 2026-10-31 pin-local incident
#: (LP-0MVWDZ1RH006HTFS / parent AC8).
INCIDENT_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "measured_cache_abort_incident_2026_10_31.jsonl"
)

#: The shipped configuration (``proxy/config.yaml``) — the gate's comparator is
#: resolved from it, exactly as production resolves it.
SHIPPED_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

#: The model the incident pane was dispatched to (``routing_check
#: provider=local-qwen3``), i.e. the calibration key the gate must price with.
INCIDENT_MODEL = "local-qwen3"

#: Measured host prompt-eval rates (2026-10-31, 224 paired samples) used only to
#: state the incident row's measured range in an assertion message.
HOST_MEDIAN_TOKENS_PER_S = 371.0
HOST_P25_TOKENS_PER_S = 170.0

#: Measured prompt-eval p10 on this host (parent finding 4). Used only by the
#: waste model, never by a decision.
PRE_EVAL_RATE_P10_TPS = 43.1

#: The parent's fake-warm definition: "priced as fully cached".
FAKE_WARM_PREDICTED_REUSE = 0.99

#: Size-band tolerance when matching measured evidence to a prompt size.
SIZE_BAND_TOLERANCE = 0.02

MODE_REPREFILL = "reprefill"
MODE_BEST_CASE = "best_case"
MODES = (MODE_REPREFILL, MODE_BEST_CASE)


@dataclass(frozen=True)
class MeasuredCache:
    """One measured cache class for a prompt size."""

    name: str
    cache_tokens: int
    source: str


@dataclass(frozen=True)
class ReplayedEvent:
    """One replayed eviction event."""

    ts: str
    session: str
    total: int
    priced_new_tokens: int
    elapsed_s: float
    fake_warm: bool
    measured: MeasuredCache | None
    decision: AbortDecision
    waste_before: float
    waste_after: float

    @property
    def abortable(self) -> bool:
        return self.decision.abort

    @property
    def reason(self) -> str | None:
        return self.decision.reason


@dataclass
class ReplayResult:
    """Structured replay outcome for assertions and JSON reports."""

    mode: str
    events: list[ReplayedEvent]
    enabled: bool
    totals: dict[str, Any] = field(default_factory=dict)


def load_fixture(path: Path | None = None) -> tuple[dict[str, Any], list[dict]]:
    """Load the checked-in fixture: ``(meta, records)``."""
    fixture = Path(path) if path is not None else FIXTURE_PATH
    with fixture.open(encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        records = [json.loads(line) for line in handle if line.strip()]
    if "_meta" not in header:  # pragma: no cover - fixture corruption alarm
        raise AssertionError("fixture is missing its `_meta` header line")
    return header["_meta"], records


def evictions(records: Iterable[dict]) -> list[dict]:
    """The ``local_eviction_after_acceptance`` records."""
    return [record for record in records if record.get("kind") == "eviction"]


def evidence(records: Iterable[dict]) -> list[dict]:
    """The measured ``n_past`` / prompt-size observations."""
    return [record for record in records if record.get("kind") == "evidence"]


def fake_warm(records: Iterable[dict]) -> list[dict]:
    """The fake-warm subset of the eviction records."""
    return [record for record in evictions(records) if record.get("fake_warm")]


def warm_population(
    records: Iterable[dict],
    *,
    min_prompt_tokens: int = 20_000,
    min_reuse: float = 0.99,
) -> list[dict]:
    """The working warm population: large prompts with near-full reuse."""
    return [
        observation
        for observation in evidence(records)
        if observation["prompt_tokens"] >= min_prompt_tokens
        and observation["cache_tokens"] / observation["prompt_tokens"] >= min_reuse
    ]


def size_band(
    total: int,
    observations: Iterable[dict],
    *,
    tolerance: float = SIZE_BAND_TOLERANCE,
) -> list[dict]:
    """Measured observations within *tolerance* of *total* prompt tokens."""
    window = max(60, int(total * tolerance))
    return [
        observation
        for observation in observations
        if abs(observation["prompt_tokens"] - total) <= window
    ]


def measured_cache(
    total: int,
    observations: Iterable[dict],
    mode: str,
) -> MeasuredCache | None:
    """The measured cache class for *total* under *mode*.

    ``reprefill`` returns the smallest measured ``n_past`` in the size band
    (the re-prefill floor llama-server reported for a prompt of this size);
    ``best_case`` returns the largest measured reuse in the band. The band
    falls back to the whole evidence population when it is empty, and the
    result is ``None`` when there is no measured evidence at all.
    """
    if mode not in MODES:
        raise ValueError(f"unknown replay mode: {mode!r}")
    pool = list(observations)
    if not pool:
        return None
    band = size_band(total, pool) or pool
    if mode == MODE_REPREFILL:
        chosen = min(band, key=lambda obs: obs["cache_tokens"])
        return MeasuredCache(
            name=MODE_REPREFILL,
            cache_tokens=int(chosen["cache_tokens"]),
            source=f"smallest measured n_past in band (task {chosen.get('task')})",
        )
    chosen = max(band, key=lambda obs: obs["cache_tokens"] / obs["prompt_tokens"])
    observed_total = int(chosen["prompt_tokens"])
    observed_cache = int(chosen["cache_tokens"])
    # Scale the measured ceiling onto this event's prompt size (the band
    # members differ by a few hundred tokens); keep it below ``total`` so the
    # observation is always a valid measurement.
    cache = observed_cache if observed_total == total else min(
        total - 1, int(round(observed_cache * total / observed_total))
    )
    return MeasuredCache(
        name=MODE_BEST_CASE,
        cache_tokens=max(0, cache),
        source=f"largest measured reuse in band (task {chosen.get('task')})",
    )


def settings_for(enabled: bool = True) -> AbortSettings:
    """Resolved settings for the **shipped** configuration, or its rollback.

    The settings are resolved from ``proxy/config.yaml`` itself (only the
    enable switch is overridden), so a replay measures what a deployment of
    that config does — including the measured-cache *benefit gate*, whose
    comparator (``budget_safety_fraction x local_prefill_deadline_seconds`` =
    114 s) and calibrated model come from the same block.

    Resolving the shipped config is behaviour-neutral while the gate does not
    exist yet (the deadline keys are simply not part of ``AbortSettings``), and
    it is what keeps these rows honest afterwards instead of replaying a
    configuration no deployment runs. ``enabled=False`` is the documented
    rollback switch.
    """
    config = shipped_config()
    server = dict(config.get("server") or {})
    server[ENABLED_KEY] = 1 if enabled else 0
    # No model: an unpriced calibration key resolves to the shipped default fit
    # (3.9 ms/token), which keeps the replayed ETA independent of whatever
    # samples an earlier test module left in the process-wide window.
    return AbortSettings.from_config({"server": server})


def replay(
    records: list[dict],
    *,
    mode: str = MODE_REPREFILL,
    enabled: bool = True,
    settings: AbortSettings | None = None,
) -> ReplayResult:
    """Replay every eviction event through the real abort decision.

    *mode* selects which measured cache class is replayed (see the module
    docstring). *enabled* drives the documented rollback switch, so
    ``enabled=False`` reproduces pre-change behaviour (every event continues).
    """
    observations = evidence(records)
    effective = settings if settings is not None else settings_for(enabled)

    replayed: list[ReplayedEvent] = []
    for event in evictions(records):
        total = int(event["estimated_tokens"])
        priced = int(event["new_tokens"])
        elapsed = float(event["elapsed_s"])
        measured = measured_cache(total, observations, mode)
        if measured is None:
            decision = decide_measured_cache_abort(None, priced, effective)
            measured_new: int | None = None
        else:
            decision = decide_measured_cache_abort(
                {"total": total, "cache": measured.cache_tokens},
                priced,
                effective,
            )
            measured_new = total - measured.cache_tokens
        computed_before_cancel = elapsed * PRE_EVAL_RATE_P10_TPS
        waste_before = (
            min(measured_new, computed_before_cancel)
            if measured_new is not None
            else computed_before_cancel
        )
        replayed.append(
            ReplayedEvent(
                ts=str(event["ts"]),
                session=str(event["session"]),
                total=total,
                priced_new_tokens=priced,
                elapsed_s=elapsed,
                fake_warm=bool(event.get("fake_warm")),
                measured=measured,
                decision=decision,
                waste_before=waste_before,
                waste_after=0.0 if decision.abort else waste_before,
            )
        )

    aborted = [event for event in replayed if event.abortable]
    replayed_fake_warm = [event for event in replayed if event.fake_warm]
    result = ReplayResult(
        mode=mode,
        events=replayed,
        enabled=effective.enabled,
    )
    result.totals = {
        "mode": mode,
        "enabled": effective.enabled,
        "events": len(replayed),
        "fake_warm": len(replayed_fake_warm),
        "abortable": len(aborted),
        "abortable_fake_warm": sum(1 for event in aborted if event.fake_warm),
        "continued_fake_warm": sum(
            1 for event in replayed_fake_warm if not event.abortable
        ),
        "abort_reasons": sorted(
            {event.reason for event in aborted if event.reason is not None}
        ),
        "continued_details": sorted(
            {event.decision.detail for event in replayed if not event.abortable}
        ),
        "priced_new_tokens": sum(event.priced_new_tokens for event in replayed),
        "measured_new_tokens_abortable": sum(
            (event.decision.measured_new_tokens or 0) for event in aborted
        ),
        "waste_before_tokens": round(
            sum(event.waste_before for event in replayed)
        ),
        "waste_after_tokens": round(sum(event.waste_after for event in replayed)),
        "waste_reclaimed_tokens": round(
            sum(event.waste_before - event.waste_after for event in replayed)
        ),
    }
    return result


def replay_result_to_json(result: ReplayResult) -> dict[str, Any]:
    """Serialise a replay for machine-readable output (CLI/API convention)."""
    return {
        "totals": dict(result.totals),
        "events": [
            {
                "ts": event.ts,
                "session": event.session,
                "total": event.total,
                "priced_new_tokens": event.priced_new_tokens,
                "elapsed_s": event.elapsed_s,
                "fake_warm": event.fake_warm,
                "measured_cache": (
                    None if event.measured is None else event.measured.cache_tokens
                ),
                "abort": event.abortable,
                "reason": event.reason,
                "detail": event.decision.detail,
                "measured_new_tokens": event.decision.measured_new_tokens,
                "deficit_ratio": (
                    None
                    if event.decision.deficit_ratio is None
                    else round(event.decision.deficit_ratio, 4)
                ),
                "waste_before_tokens": round(event.waste_before),
                "waste_after_tokens": round(event.waste_after),
            }
            for event in result.events
        ],
    }


# ---------------------------------------------------------------------------
# Benefit-gate incident replay (LP-0MVWDZ1RH006HTFS, parent AC8)
# ---------------------------------------------------------------------------


def load_incident_fixture(
    path: Path | None = None,
) -> tuple[dict[str, Any], list[dict]]:
    """Load the incident fixture: ``(meta, records)``."""
    fixture = Path(path) if path is not None else INCIDENT_FIXTURE_PATH
    with fixture.open(encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        records = [json.loads(line) for line in handle if line.strip()]
    if "_meta" not in header:  # pragma: no cover - fixture corruption alarm
        raise AssertionError("fixture is missing its `_meta` header line")
    return header["_meta"], records


def incident_aborts(records: Iterable[dict]) -> list[dict]:
    """The recorded ``measured_cache_abort`` events of the incident."""
    return [record for record in records if record.get("kind") == "abort"]


def shipped_config() -> dict:
    """The shipped ``proxy/config.yaml`` (the gate's real input)."""
    return yaml.safe_load(SHIPPED_CONFIG_PATH.read_text(encoding="utf-8")) or {}


def incident_settings(
    config: dict | None = None,
    **overrides: Any,
) -> AbortSettings:
    """Gate settings resolved from the shipped config, priced for the incident.

    ``AbortSettings.from_config`` supplies the comparator (``0.95 x 120 s`` =
    114 s with the shipped keys); ``model`` and ``config`` are what let the
    decision price the **measured** deficit with the admission guard's own
    calibrated call (parent decision D3(a)).
    """
    resolved = shipped_config() if config is None else config
    settings = AbortSettings.from_config(resolved)
    return replace(settings, model=INCIDENT_MODEL, config=resolved, **overrides)


@dataclass(frozen=True)
class IncidentReplayRecord:
    """One recorded abort re-decided under the benefit gate."""

    ts: str
    total: int
    cache: int
    priced_new_tokens: float
    wait_budget_s: float
    deadline_s: float
    decision: AbortDecision
    eta_at_host_median_s: float
    eta_at_p25_s: float

    @property
    def abort(self) -> bool:
        return self.decision.abort

    @property
    def measured_eta_s(self) -> float | None:
        return self.decision.measured_eta_s

    @property
    def budget_s(self) -> float | None:
        return self.decision.budget_s


@dataclass
class IncidentReplay:
    """Structured replay outcome for the incident fixture."""

    records: list[IncidentReplayRecord]
    settings: AbortSettings
    totals: dict[str, Any] = field(default_factory=dict)


def replay_incident(
    records: list[dict] | None = None,
    *,
    settings: AbortSettings | None = None,
    config: dict | None = None,
) -> IncidentReplay:
    """Re-decide every recorded incident abort under the benefit gate.

    The caller owns the calibration window (``prefill_eta.reset()`` for the
    shipped default-fit row). The armed-cooldown count is derived from the
    abort verdict because ``router.py`` arms
    ``prefill_thrash_cooldown_started`` inside the abort branch — a declined
    abort therefore arms nothing, which is the parent's AC3.
    """
    source = load_incident_fixture()[1] if records is None else records
    effective = settings if settings is not None else incident_settings(config)

    replayed: list[IncidentReplayRecord] = []
    for record in incident_aborts(source):
        total = int(record["total"])
        cache = int(record["cache"])
        decision = decide_measured_cache_abort(
            {"total": total, "cache": cache},
            record["priced_new_tokens"],
            effective,
        )
        measured_new = total - cache
        replayed.append(
            IncidentReplayRecord(
                ts=str(record.get("ts", "")),
                total=total,
                cache=cache,
                priced_new_tokens=float(record["priced_new_tokens"]),
                wait_budget_s=float(record["wait_budget_s"]),
                deadline_s=float(record["deadline_s"]),
                decision=decision,
                eta_at_host_median_s=measured_new / HOST_MEDIAN_TOKENS_PER_S,
                eta_at_p25_s=measured_new / HOST_P25_TOKENS_PER_S,
            )
        )

    aborted = [row for row in replayed if row.abort]
    result = IncidentReplay(records=replayed, settings=effective)
    result.totals = {
        "aborts": len(aborted),
        "continued": len(replayed) - len(aborted),
        "cooldowns_armed": len(aborted),
        "details": sorted({row.decision.detail for row in replayed}),
        "measured_eta_s": sorted(
            {
                round(row.measured_eta_s, 3)
                for row in replayed
                if row.measured_eta_s is not None
            }
        ),
        "budget_s": sorted(
            {round(row.budget_s, 3) for row in replayed if row.budget_s is not None}
        ),
        "wait_budget_s": sorted({row.wait_budget_s for row in replayed}),
        "deadline_s": sorted({row.deadline_s for row in replayed}),
    }
    return result


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI helper
    """Print the replay report for both measured classes (operator utility)."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    _meta, records = load_fixture(args.fixture)
    reports = {
        "reprefill": replay(records, mode=MODE_REPREFILL),
        "best_case": replay(records, mode=MODE_BEST_CASE),
        "rollback": replay(records, mode=MODE_REPREFILL, enabled=False),
    }
    if args.json:
        print(
            json.dumps(
                {
                    name: replay_result_to_json(result)
                    for name, result in reports.items()
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    for name, result in reports.items():
        print(json.dumps(result.totals, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
