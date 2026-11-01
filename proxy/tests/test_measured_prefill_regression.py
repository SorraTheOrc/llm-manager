"""Warm-path unchanged + reference-window replay for the measured-prefill abort (F5).

LP-0MVR7IXAI002QXXD (parent LP-0MVQIT9QY007QU27, Option D).

The parent shipped the *measured* cache-deficit early abort because every
admission-time warmth predicate is a prediction. This module locks in the
no-regression guarantees the parent's AC3/AC6/AC7 depend on:

* the working warm population (large prompt, measured reuse ≈ 0.999) never
  trips the abort and still completes locally, byte-for-byte unchanged;
* the admission ladder (``session_slot`` / ``measured`` / ``non_resident`` /
  ``unknown``) resolves and decides identically with the feature enabled and
  with the rollback key set;
* replaying the parent's 2026-10-26 reference window classifies the 29
  fake-warm events as ``measured_cache_deficit`` abortable under the measured
  re-prefill cache, and releases them in seconds rather than at the 120 s
  deadline (measured end to end through the real ``proxy_to_local``);
* with the rollback key set, the same replay reproduces the pre-change
  behaviour exactly (no abort, the deadline stands).

The replay harness is :mod:`measured_prefill_window_replay`; the fixture is
the checked-in capture produced by
``proxy/scripts/capture_measured_prefill_window.py``.
"""

from __future__ import annotations

import time

import measured_prefill_window_replay as window
import proxy.provider as provider
import pytest
import test_measured_cache_abort_wiring as wiring
from fastapi.responses import JSONResponse, StreamingResponse
from test_local_resident_prefix_admission import _install_slot_state, _slot
from test_measured_cache_abort_wiring import (
    _CONTENT_CHUNK,
    _DONE,
    _FINISH_CHUNK,
    _drive_local,
    _progress_event,
)

from proxy import prefill_eta

MODEL = "Qwen3"
#: The documented defaults the shipped configuration resolves to.
DEFAULT_BOUND_TOKENS = 2_000
DEFAULT_MIN_DEFICIT_RATIO = 0.25
#: The parent's measured deadline; the early release must beat it by an order
#: of magnitude (the replay asserts < 10 s, i.e. ≪ 120 s).
PREFILL_DEADLINE_SECONDS = 120

#: The shipped comparator the benefit gate enforces (parent decision D1(a)):
#: ``local_admission_guard_budget_safety_fraction x
#: local_prefill_deadline_seconds`` = 0.95 x 120 s.
SHIPPED_BUDGET_S = 0.95 * PREFILL_DEADLINE_SECONDS
#: Shipped default prefill fit (``DEFAULT_PREFILL_MS_PER_TOKEN``): the fixture's
#: prompt-size buckets are uncalibrated, so this is the honest rate basis for
#: the partition below (``_drive_signature`` pins the same window).
PREFILL_MS_PER_TOKEN = 3.9

#: The warm-path signature the parent measured: 56k prompt, reuse ≈ 0.999.
WARM_TOTAL_TOKENS = 56_000
WARM_CACHE_TOKENS = 55_944

SERVER_CONFIG_ON = {
    "server": {
        "local_measured_cache_abort_enabled": 1,
        "local_large_context_cold_cache_threshold": 25_000,
        "local_large_context_warm_cache_threshold": 0,
        "local_resident_prefix_min_ratio": 0.5,
        "session_slot_pool_size": 2,
        "local_session_tenants": 2,
        "local_tenancy_idle_seconds": 30,
    }
}


def _fixture_records() -> list[dict]:
    _meta, records = window.load_fixture()
    return records


#: Every fake-warm cluster of the reference window, keyed by prompt size.
FAKE_WARM_CLUSTERS: tuple[tuple[int, int], ...] = tuple(
    sorted(
        {
            (event["estimated_tokens"], event["new_tokens"])
            for event in window.fake_warm(_fixture_records())
        }
    )
)


def _measured_eta_s(total: int) -> float:
    """Measured cold-prefill ETA for *total* at the shipped default fit.

    Uses the same ``reprefill`` measured cache class the rows replay with, so
    the partition below is the decision input, not a second model.
    """
    measured = window.measured_cache(
        total, window.evidence(_fixture_records()), window.MODE_REPREFILL
    )
    assert measured is not None
    return (total - measured.cache_tokens) * PREFILL_MS_PER_TOKEN / 1000.0


#: Clusters whose measured prefill exceeds the benefit budget: still detoured.
BUDGET_ALLOWED_CLUSTERS = tuple(
    cluster
    for cluster in FAKE_WARM_CLUSTERS
    if _measured_eta_s(cluster[0]) > SHIPPED_BUDGET_S
)
#: Clusters whose measured prefill fits it: served locally once the gate lands.
BUDGET_NEUTRAL_CLUSTERS = tuple(
    cluster
    for cluster in FAKE_WARM_CLUSTERS
    if _measured_eta_s(cluster[0]) <= SHIPPED_BUDGET_S
)


def test_the_replay_clusters_partition_at_the_benefit_budget():
    """The 114 s comparator splits the window: 2 clusters fit, 10 do not.

    Pinned so a fixture or fit change cannot silently empty one side of the
    split (which would turn the two rows below into no-ops).
    """
    assert BUDGET_NEUTRAL_CLUSTERS == ((18_878, 17), (22_973, 175))
    assert len(BUDGET_ALLOWED_CLUSTERS) == len(FAKE_WARM_CLUSTERS) - 2 == 10
    for total, _priced in BUDGET_ALLOWED_CLUSTERS:
        assert _measured_eta_s(total) > SHIPPED_BUDGET_S
    for total, _priced in BUDGET_NEUTRAL_CLUSTERS:
        assert _measured_eta_s(total) < SHIPPED_BUDGET_S


def _ladder_inputs(config: dict) -> dict:
    """Resolve the admission ladder's inputs from *config* as the router does."""
    cold_threshold, warm_threshold = provider._effective_large_context_thresholds(
        config
    )
    return {
        "cold_cache_threshold": cold_threshold,
        "warm_cache_threshold": warm_threshold,
        "resident_prefix_min_ratio": provider._effective_resident_prefix_min_ratio(
            config
        ),
        "session_tenants": provider._resolve_tenants(config),
        "tenancy_idle_seconds": provider._resolve_tenancy_idle(config),
        "economic_bypass_serves_local": provider._economic_bypass_serves_local(
            config, "medium"
        ),
    }


def _ladder_decision(config: dict, *, session_id: str, estimated_tokens: int):
    inputs = _ladder_inputs(config)
    from proxy.residency import get_residency_view

    return provider._should_skip_local(
        MODEL,
        session_id,
        {"messages": []},
        inputs["cold_cache_threshold"],
        estimated_tokens=estimated_tokens,
        warm_cache_threshold=inputs["warm_cache_threshold"],
        economic_bypass_serves_local=inputs["economic_bypass_serves_local"],
        resident_prefix_min_ratio=inputs["resident_prefix_min_ratio"],
        residency_view=get_residency_view(),
        endpoint="",
        session_tenants=inputs["session_tenants"],
        tenancy_idle_seconds=inputs["tenancy_idle_seconds"],
    )


async def _drive_signature(
    monkeypatch,
    *,
    total: int,
    priced: int,
    cache: int,
    enabled: bool = True,
):
    """Drive one measured signature through the real ``proxy_to_local``.

    The calibration window is **reset** first: every row here asserts the
    shipped default prefill fit (3.9 ms/token), which is also what makes the
    benefit gate's verdict for the row deterministic instead of depending on
    whatever samples an earlier test module left behind.
    """
    prefill_eta.reset()
    monkeypatch.setattr(wiring, "PROMPT_TOKENS", total)
    monkeypatch.setattr(
        wiring._Request,
        "_local_admission_evidence",
        {
            "estimated_tokens": total,
            "cached_ratio": (total - priced) / total,
            "new_tokens": priced,
        },
    )
    extra = {} if enabled else {"local_measured_cache_abort_enabled": 0}
    return await _drive_local(
        [_progress_event(total, cache), _CONTENT_CHUNK, _FINISH_CHUNK, _DONE],
        monkeypatch=monkeypatch,
        server_config_extra=extra,
    )


# ---------------------------------------------------------------------------
# AC1 — the warm population and the admission ladder are untouched
# ---------------------------------------------------------------------------


def test_reference_window_ships_the_measured_warm_population():
    """The fixture carries the working warm population, not just evictions.

    The parent measured 252 working large local prefills (≥ 20k tokens, 177
    first byte < 10 s) in the same window; the checked-in evidence population
    contains the large tasks with near-full measured reuse (≥ 0.99), so the
    regression sample is measured ground truth rather than a synthetic guess.
    """
    records = _fixture_records()
    warm = window.warm_population(records)

    assert len(warm) >= 150, f"expected the large warm population, got {len(warm)}"
    assert all(observation["prompt_tokens"] >= 20_000 for observation in warm)
    assert all(
        observation["cache_tokens"] / observation["prompt_tokens"] >= 0.99
        for observation in warm
    )
    # The 252-population signature is represented: large prompt, reuse ≈ 0.999.
    assert any(
        observation["prompt_tokens"] >= 40_000
        and observation["cache_tokens"] / observation["prompt_tokens"] >= 0.999
        for observation in warm
    )


def test_measured_warm_population_never_aborts():
    """AC1: every measured warm observation continues, at any token bound.

    The ratio floor (default 0.25) is what makes a genuinely warm dispatch
    unreachable **for any ``bound_tokens``**: a measured reuse ≥ 0.99 leaves a
    deficit ratio ≤ 0.01, far below the floor, so even ``0`` slack and a
    ``0``-priced dispatch cannot release it.
    """
    observations = window.warm_population(_fixture_records())
    assert observations, "no warm observations in the fixture"

    for observation in observations:
        total = observation["prompt_tokens"]
        cache = observation["cache_tokens"]
        for settings in (
            window.settings_for(enabled=True),
            window.AbortSettings(
                enabled=True,
                bound_tokens=0,
                min_deficit_ratio=window.AbortSettings().min_deficit_ratio,
            ),
        ):
            decision = window.decide_measured_cache_abort(
                {"total": total, "cache": cache}, 0, settings
            )
            assert decision.abort is False, (total, cache, settings)
            assert decision.detail == "below_deficit_ratio_floor", (
                total,
                cache,
                decision,
            )
            assert decision.deficit_ratio is not None
            assert decision.deficit_ratio <= 0.01


@pytest.mark.asyncio
async def test_warm_stream_completes_locally_unchanged(monkeypatch):
    """AC1: a genuinely warm 56k dispatch streams the full response locally."""
    result, received, sent = await _drive_signature(
        monkeypatch,
        total=WARM_TOTAL_TOKENS,
        priced=WARM_TOTAL_TOKENS - WARM_CACHE_TOKENS,
        cache=WARM_CACHE_TOKENS,
    )

    assert sent.get("return_progress") is True
    assert isinstance(result, StreamingResponse)
    assert result.status_code == 200
    assert received == _CONTENT_CHUNK + _FINISH_CHUNK + _DONE
    assert b"prompt_progress" not in received

    infos = wiring._all_log_lines("info")
    errors = wiring._all_log_lines("error")
    assert not any("measured_cache_deficit" in line for line in errors)
    assert not any("reason=prefill_deadline" in line for line in errors)
    # The warm measurement is still recorded for attribution (parent AC6).
    assert any(
        line.startswith("prefill_progress_measured")
        and f"cache={WARM_CACHE_TOKENS}" in line
        for line in infos
    )
    assert wiring.server.local_dispatch_records, "the served dispatch keeps its lease"


@pytest.mark.parametrize(
    "cell",
    [
        # measured residency: the slot holds the session's prompt
        {"name": "measured", "slots": [{"slot_id": 0, "context_tokens": 738}], "owner": "sess-a"},
        # session_slot residency: the session owns the slot, n_past unexposed
        {"name": "session_slot", "slots": [{"slot_id": 0, "context_tokens": None}], "owner": "sess-a"},
        # non_resident: the slot belongs to a different session
        {"name": "non_resident", "slots": [{"slot_id": 0, "context_tokens": 49103}], "owner": "sess-other"},
        # unknown: no snapshot and no ownership — falls open to the prediction
        {"name": "unknown", "slots": [], "owner": None},
    ],
    ids=["measured", "session_slot", "non_resident", "unknown"],
)
def test_admission_ladder_identical_with_the_feature_enabled_and_disabled(
    monkeypatch, cell
):
    """AC1: the residency ladder decides byte-for-byte the same either way.

    Every ladder rung (``measured``, ``session_slot``, ``non_resident``,
    ``unknown``) is resolved from the same cached view and config-derived
    thresholds; adding the measured-cache feature keys must not move a single
    decision or change the provenance the decision is logged with.
    """
    owners = {} if cell["owner"] is None else {("", 0): cell["owner"]}
    _install_slot_state(
        monkeypatch,
        slots=[_slot(s["slot_id"], s["context_tokens"]) for s in cell["slots"]],
        owners=owners,
    )
    config_off = {
        "server": dict(
            SERVER_CONFIG_ON["server"], local_measured_cache_abort_enabled=0
        )
    }

    from proxy.residency import get_residency_view

    facts_on = provider.resolve_resident_prefix(
        "sess-a", 49_103, view=get_residency_view(), endpoint=""
    )
    decision_on = _ladder_decision(
        SERVER_CONFIG_ON, session_id="sess-a", estimated_tokens=49_103
    )
    facts_off = provider.resolve_resident_prefix(
        "sess-a", 49_103, view=get_residency_view(), endpoint=""
    )
    decision_off = _ladder_decision(
        config_off, session_id="sess-a", estimated_tokens=49_103
    )

    assert facts_on.source == cell["name"]
    assert (facts_on.source, facts_on.resident, facts_on.reuse_ratio) == (
        facts_off.source,
        facts_off.resident,
        facts_off.reuse_ratio,
    )
    assert decision_on == decision_off


def test_ladder_threshold_resolution_ignores_the_feature_keys():
    """AC1: the feature's config keys feed no admission-ladder input."""
    config_off = {
        "server": dict(
            SERVER_CONFIG_ON["server"], local_measured_cache_abort_enabled=0
        )
    }
    assert _ladder_inputs(SERVER_CONFIG_ON) == _ladder_inputs(config_off)


# ---------------------------------------------------------------------------
# AC2 — the reference window replays as measured-cache-deficit abortable
# ---------------------------------------------------------------------------


def test_reference_window_matches_the_parent_measurement():
    """AC2: the fixture reproduces the parent's 120 events / 29 fake-warm."""
    meta, records = window.load_fixture()

    assert meta["counts"]["evictions"] == 120
    assert meta["counts"]["fake_warm"] == 29
    assert meta["counts"]["fake_warm_clusters"] == 12
    assert len(window.evictions(records)) == 120

    fake_warm = window.fake_warm(records)
    assert len(fake_warm) == 29
    assert all(event["predicted_reuse_ratio"] >= 0.99 for event in fake_warm)
    assert all(event["new_tokens"] <= 300 for event in fake_warm)
    clusters = {(event["estimated_tokens"], event["new_tokens"]) for event in fake_warm}
    assert (56_468, 162) in clusters
    # The remaining 91 events are cold-priced deadline cancellations.
    assert len(window.evictions(records)) - len(fake_warm) == 91


def test_replay_classifies_fake_warm_events_as_measured_cache_deficit():
    """AC2: every fake-warm event whose measured prefill overruns still aborts.

    The 26 events of the ten clusters whose measured cold prefill exceeds the
    shipped 114 s comparator (``_measured_eta_s``: 115-284 s at the default
    fit) abort at their admission-priced tokens. The remaining three events
    live in the two mid-size clusters (18 878 / 22 973 tokens, 72-88 s) that
    the benefit gate now **serves** — that half is asserted end to end by
    ``test_replay_events_within_the_budget_are_served_locally``.

    The measured re-prefill floor for each prompt size comes from llama-server's
    own ``n_past`` observations in the same window (356-6 919 reused tokens on
    18k-78k prompts). At that measurement every event here exceeds its
    admission price by far more than the configured bound and sits far above
    the deficit-ratio floor — the exact combination the decision targets.
    """
    result = window.replay(_fixture_records(), mode=window.MODE_REPREFILL)

    assert result.totals["fake_warm"] == 29
    # 29 pre-gate; 26 once the benefit gate declines the two mid-size clusters.
    assert result.totals["abortable_fake_warm"] >= 26
    assert "measured_cache_deficit" in result.totals["abort_reasons"]

    beyond_budget = [
        event
        for event in result.events
        if event.fake_warm and _measured_eta_s(event.total) > SHIPPED_BUDGET_S
    ]
    assert len(beyond_budget) == 26
    assert sum(event.waste_before for event in beyond_budget) > 130_000

    for event in beyond_budget:
        assert event.abortable, event
        assert event.reason == window.MEASURED_CACHE_DEFICIT_REASON
        assert event.decision.measured_new_tokens is not None
        assert (
            event.decision.measured_new_tokens
            > event.priced_new_tokens + DEFAULT_BOUND_TOKENS
        )
        assert event.decision.deficit_ratio is not None
        assert event.decision.deficit_ratio >= DEFAULT_MIN_DEFICIT_RATIO
        # Pre-change the whole priced prefill was discarded; the abort
        # discards nothing because the first progress event precedes compute.
        assert event.waste_after == 0.0
        assert event.waste_before > 0.0

    report = window.replay_result_to_json(result)
    assert report["totals"]["abortable_fake_warm"] >= 26
    reported_aborts = {
        entry["ts"]
        for entry in report["events"]
        if entry["fake_warm"] and entry["reason"] == "measured_cache_deficit"
    }
    assert {event.ts for event in beyond_budget} <= reported_aborts


def test_full_reuse_measurement_never_aborts_a_fake_warm_priced_event():
    """AC2/AC1: if the measurement shows full reuse, the dispatch is protected.

    ``n_past = total - 1`` is repeatedly observed in the window (the retry of a
    warm session). A fake-warm *price* with that measurement must continue —
    the abort fires on measurement, never on the prediction.
    """
    settings = window.settings_for(enabled=True)
    for event in window.fake_warm(_fixture_records()):
        total = event["estimated_tokens"]
        decision = window.decide_measured_cache_abort(
            {"total": total, "cache": total - 1}, event["new_tokens"], settings
        )
        assert decision.abort is False, (total, event["new_tokens"])
        assert decision.detail == "below_deficit_ratio_floor"


def test_best_case_replay_protects_the_measured_reuse_ceiling():
    """AC2: replaying the best measured reuse leaves most events continuing."""
    result = window.replay(_fixture_records(), mode=window.MODE_BEST_CASE)

    assert result.totals["fake_warm"] == 29
    assert result.totals["continued_fake_warm"] >= 20
    assert result.totals["abortable_fake_warm"] < 10
    assert set(result.totals["continued_details"]) <= {
        "below_deficit_ratio_floor",
        "within_bound",
        "invalid_measurement",
        # The benefit gate's continue token: a best-case-reuse event that still
        # reached the bound check but fits the 114 s comparator now continues
        # here instead of aborting.
        "within_eta_budget",
    }


@pytest.mark.parametrize(
    "cluster",
    BUDGET_ALLOWED_CLUSTERS,
    ids=lambda cluster: f"{cluster[0]}tok-{cluster[1]}priced",
)
@pytest.mark.asyncio
async def test_replay_events_beyond_the_budget_release_in_seconds(
    monkeypatch, cluster
):
    """AC2/AC4: a measured prefill that overruns the budget still detours.

    Once the benefit gate is wired (parent LP-0MVWBD91Z004U135 AC1) only the
    clusters whose **measured** cold prefill exceeds the shipped comparator
    detour; they keep today's seconds-scale release, never a 120 s deadline
    wait.
    """
    total, priced = cluster
    assert _measured_eta_s(total) > SHIPPED_BUDGET_S, cluster
    measured = window.measured_cache(
        total, window.evidence(_fixture_records()), window.MODE_REPREFILL
    )
    assert measured is not None

    started = time.monotonic()
    result, received, _sent = await _drive_signature(
        monkeypatch,
        total=total,
        priced=priced,
        cache=measured.cache_tokens,
    )
    elapsed = time.monotonic() - started

    assert isinstance(result, JSONResponse), type(result).__name__
    assert result.status_code == 503
    assert received == b""
    assert elapsed < 10, f"expected seconds-scale release, took {elapsed:.1f}s"
    assert elapsed < PREFILL_DEADLINE_SECONDS / 10

    errors = wiring._all_log_lines("error")
    assert any(
        "local_eviction_after_acceptance" in line
        and "reason=measured_cache_deficit" in line
        and f"priced_new_tokens={priced}" in line
        for line in errors
    ), errors
    assert not any("reason=prefill_deadline" in line for line in errors)


@pytest.mark.parametrize(
    "cluster",
    BUDGET_NEUTRAL_CLUSTERS,
    ids=lambda cluster: f"{cluster[0]}tok-{cluster[1]}priced",
)
@pytest.mark.asyncio
async def test_replay_events_within_the_budget_are_served_locally(
    monkeypatch, cluster
):
    """AC4 (parent AC1/AC4): a mid-size fake-warm cluster is **served**.

    These are the reference-window clusters whose measured cold prefill fits the
    shipped comparator (72-88 s against 114 s), i.e. the same shape as the
    pin-local incident. They must complete locally: no
    ``local_eviction_after_acceptance``, no thrash cooldown, and the client gets
    the whole stream. Green since the benefit gate landed
    (LP-0MVWDZ1GG004UGIK); a regression must fail the gate loudly.
    """
    total, priced = cluster
    eta_s = _measured_eta_s(total)
    assert eta_s <= SHIPPED_BUDGET_S, cluster
    measured = window.measured_cache(
        total, window.evidence(_fixture_records()), window.MODE_REPREFILL
    )
    assert measured is not None

    result, received, _sent = await _drive_signature(
        monkeypatch,
        total=total,
        priced=priced,
        cache=measured.cache_tokens,
    )

    assert isinstance(result, StreamingResponse), (
        f"{total}-token cluster: the measured cold prefill needs {eta_s:.1f} s "
        f"at the shipped fit, inside the {SHIPPED_BUDGET_S:.0f} s prefill-deadline "
        "budget (the incident's measured range is 38.8-84.6 s), so the "
        f"dispatch must be served locally (got {type(result).__name__})"
    )
    assert result.status_code == 200
    assert received == _CONTENT_CHUNK + _FINISH_CHUNK + _DONE

    errors = wiring._all_log_lines("error")
    infos = wiring._all_log_lines("info")
    assert not any("local_eviction_after_acceptance" in line for line in errors), errors
    assert not any("measured_cache_deficit" in line for line in errors), errors
    assert not any("prefill_thrash_cooldown" in line for line in infos + errors)


# ---------------------------------------------------------------------------
# AC3 — the rollback key reproduces the pre-change behaviour exactly
# ---------------------------------------------------------------------------


def test_rollback_replay_reproduces_the_pre_change_window_exactly():
    """AC3: disabled, the whole window keeps today's deadline behaviour.

    Not a half-off switch: no event is classified abortable in either measured
    class, every decision reports the ``disabled`` detail, and no wasted token
    is reclaimed (the 120 s deadline remains the only release path).
    """
    records = _fixture_records()
    for mode in window.MODES:
        result = window.replay(records, mode=mode, enabled=False)
        assert result.enabled is False
        assert result.totals["abortable"] == 0
        assert result.totals["abortable_fake_warm"] == 0
        assert result.totals["continued_fake_warm"] == 29
        assert result.totals["abort_reasons"] == []
        assert result.totals["continued_details"] == ["disabled"]
        assert result.totals["waste_reclaimed_tokens"] == 0
        assert result.totals["waste_after_tokens"] == result.totals["waste_before_tokens"]


@pytest.mark.asyncio
async def test_rollback_dispatch_keeps_todays_stream_and_deadline(monkeypatch):
    """AC3: with the knob off, no injection, no capture, no early release."""
    result, received, sent = await _drive_signature(
        monkeypatch,
        total=56_468,
        priced=162,
        cache=356,
        enabled=False,
    )

    assert "return_progress" not in sent
    assert isinstance(result, StreamingResponse)
    assert result.status_code == 200
    # Pre-change: the progress chunk is ordinary upstream data.
    assert received == b"".join(
        [_progress_event(56_468, 356), _CONTENT_CHUNK, _FINISH_CHUNK, _DONE]
    )
    infos = wiring._all_log_lines("info")
    errors = wiring._all_log_lines("error")
    assert not any(line.startswith("prefill_progress_measured") for line in infos)
    assert not any("measured_cache_deficit" in line for line in errors)
