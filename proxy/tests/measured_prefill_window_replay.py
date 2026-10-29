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

This module is analysis/test tooling: no production module may import it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from proxy.measured_cache_abort import (
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
    """Resolved settings for the shipped configuration, or its rollback."""
    return AbortSettings.from_config(
        {"server": {"local_measured_cache_abort_enabled": 1 if enabled else 0}}
    )


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
