"""Cold-regime replay proof (LP-0MW957TAE000QD9N).

Parent LP-0MW7V2D13004ZO65 AC3/AC10, test half. Every assertion here reads a
**committed** fixture through the shipped calibration + admission path; no test
touches a live log or a live recording, so the cold rate, the capacity refusals
and the 2026-11-03 mispricing are reproducible from the repository forever.

AC mapping
----------

- **AC1 — cold rate within tolerance.** Replaying the certified-window
  fixture's abandoned prefills through the real recorder calibrates the
  large-prompt buckets, and the fitted rate sits within ±20 % of the measured
  3.81-4.26 ms/token — and inside the measured band itself — in both ms/token
  and the implied tok/s. The rate is proven to be the fixture's own
  least-squares fit (never the shipped default), and the window's measured cold
  completions are shown to fall below the shipped ``processed >= 32 000`` floor
  (which is why the abandoned source exists).
- **AC2 — replay proof.** Replaying the window's predicted-warm events refuses
  the 55k/67k/90k population at admission with the deadline-capacity reason, and
  no admitted row has a cold projection above its capacity bound.
- **AC3 — incident replay (parent AC10).** The reconstructed 2026-11-03 turn is
  priced cold by the shipped large-append rule (``cached_ratio`` forced to 0),
  admitted at the shipped deadline with its cold price inside it, and refused at
  admission — with the row naming both the price and the refusal — below that
  price; the capacity bound is shown to be the binding refusal when the row is
  priced from its measured new tokens.
- **AC4 — deterministic and self-documenting.** Every fixture row carries its
  provenance; the harness resets calibration per test; the replay reads only
  committed files.
"""

from __future__ import annotations

import json
from pathlib import Path

import prefill_cold_regime_replay as replay
import pytest

from proxy import prefill_eta

MODEL = replay.MODEL


@pytest.fixture(autouse=True)
def _fresh_replay_state():
    """AC4: every test starts from an empty calibration window and clean pricing."""
    replay.reset()
    yield
    replay.reset()


def _band() -> dict[str, float]:
    return replay.rate_band()


# ===========================================================================
# AC1 — the calibrated cold rate
# ===========================================================================


def test_certified_window_calibrates_the_cold_rate_inside_the_measured_band():
    """AC1: the fixture fills the large-prompt buckets and fits the measured band.

    The population the fixture calibrates from is stated, not assumed: the
    recorder accepts exactly the abandoned rows that clear its cold floors, and
    those are what the fitted rate derives from.
    """
    accepted = replay.load_cold_window()
    abandoned = replay.abandoned_rows()
    assert len(accepted) == sum(
        1
        for row in abandoned
        if row["processed"] >= prefill_eta.COLD_MIN_PROCESSED_TOKENS
        and row["processed"] > row["cache"]
        and row["cache"] / row["total"] < prefill_eta.WARM_CACHE_RATIO
    )
    assert len(accepted) >= prefill_eta.MIN_CALIBRATION_SAMPLES * 2
    assert {prefill_eta.bucket_for(row["total"]) for row in accepted} == {1, 2}

    band = _band()
    for prompt_tokens, bucket in ((55_000, 1), (90_000, 2)):
        assert prefill_eta.bucket_for(prompt_tokens) == bucket
        capacity = replay.cold_rate(prompt_tokens)
        assert capacity.calibrated is True
        rate = capacity.cold_ms_per_token
        assert rate is not None
        # AC1 tolerance: within ±20 % of the measured 3.81-4.26 ms/token ...
        assert band["min_ms_per_token"] * 0.8 <= rate <= band["max_ms_per_token"] * 1.2
        # ... and inside the measured band itself (the stronger claim).
        assert band["min_ms_per_token"] <= rate <= band["max_ms_per_token"]
        tokens_per_second = 1000.0 / rate
        assert (
            band["min_tokens_per_second"] * 0.8
            <= tokens_per_second
            <= band["max_tokens_per_second"] * 1.2
        )
        assert (
            band["min_tokens_per_second"]
            <= tokens_per_second
            <= band["max_tokens_per_second"]
        )


def test_fitted_rate_is_the_fixtures_own_rate_not_a_default():
    """AC1 'no hand-seeded rate': the rate is the accepted rows' own fit.

    ``DEFAULT_PREFILL_MS_PER_TOKEN`` (3.9) is also inside the measured band, so
    "within tolerance" alone would not prove calibration. The fit must equal the
    rows' least-squares-through-origin rate times the shipped conservatism
    factor, and must differ from the default.
    """
    accepted = replay.load_cold_window()
    config = replay.shipped_config()
    for prompt_tokens, bucket in ((55_000, 1), (90_000, 2)):
        rows = [
            row
            for row in accepted
            if prefill_eta.bucket_for(row["total"]) == bucket
        ]
        denominator = sum(row["processed"] ** 2 for row in rows)
        assert denominator > 0
        implied_rate = (
            sum(row["processed"] * row["time_ms"] for row in rows) / denominator
        )
        coeffs = prefill_eta.coefficients(
            MODEL, prompt_tokens, config, cached_ratio=0.0
        )
        assert coeffs.rate_bucket == "cold"
        assert coeffs.calibrated is True
        # Identical-service windows uplift by 1.0 (p50 == p75); the fitted rate
        # is therefore exactly the rows' fit times the conservatism factor.
        assert coeffs.uplift == pytest.approx(1.0, rel=1e-9)
        assert coeffs.prefill_ms_per_token == pytest.approx(
            implied_rate * prefill_eta.CONSERVATISM_FACTOR, rel=1e-9
        )
        assert coeffs.prefill_ms_per_token != pytest.approx(
            prefill_eta.DEFAULT_PREFILL_MS_PER_TOKEN
            * prefill_eta.CONSERVATISM_FACTOR,
            rel=1e-6,
        )


def test_measured_cold_completions_cannot_calibrate_the_shipped_window():
    """The AC1 oracle population is below the shipped completed-sample floor.

    Each measured row is a genuine cold completion at the cited rate, yet every
    one finished below ``COLD_MIN_PROCESSED_TOKENS`` (32,000) — so
    ``record_cold_measured_sample`` drops them all and the capacity stays
    uncalibrated (fail-open). This pins the dependency's premise: the replay can
    only calibrate because the abandoned-prefill source ships
    (LP-0MWGAEBWJ00334W2), not because the completed path could supply it.
    """
    rows = replay.measured_cold_rows()
    band = _band()
    assert len(rows) >= 5
    for row in rows:
        assert band["min_ms_per_token"] <= row["ms_per_token"] <= band["max_ms_per_token"]
        assert row["tokens_per_second"] == pytest.approx(
            1000.0 / row["ms_per_token"], abs=0.1
        )
        assert row["measured_cached_ratio"] < prefill_eta.COLD_MAX_CACHED_RATIO
        assert row["processed"] < prefill_eta.COLD_MIN_PROCESSED_TOKENS
        assert row["below_processed_floor"] is True
        prefill_eta.record_cold_measured_sample(
            MODEL,
            row["total"],
            row["measured_cached_ratio"],
            row["processed"],
            row["time_ms"],
        )

    config = replay.shipped_config()
    coeffs = prefill_eta.coefficients(MODEL, 55_000, config, cached_ratio=0.0)
    assert coeffs.cold_samples == 0
    assert coeffs.calibrated is False
    # The shipped flat cold-prefill cap is an independent bound (it refuses a
    # 55k cold prompt on token count alone), so it is disabled here — the
    # documented ``local_cold_prefill_budget_tokens: 0`` rollback — to isolate
    # the deadline token capacity this assertion is about.
    capacity_only = replay.shipped_config(local_cold_prefill_budget_tokens=0)
    decision = prefill_eta.evaluate_admission(
        capacity_only, MODEL, 55_000, 0.0, "low", residency_measured=False
    )
    # Uncalibrated capacity never manufactures a refusal (fail-open).
    assert decision.admit is True
    assert decision.deadline_capacity_calibrated is False


# ===========================================================================
# AC2 — replay the predicted-warm events
# ===========================================================================


def test_replay_refuses_the_predicted_warm_large_events():
    """AC2: the 55k/67k/90k predicted-warm events are refused at admission.

    Each was admitted on a predicted prefix reuse (>= 0.98) that llama-server
    then disproved by still prefilling at the deadline. Replayed on the honest
    basis — a prediction is not evidence, so the whole prompt is priced cold —
    every one is refused with the distinct deadline-capacity reason.
    """
    replay.load_cold_window()
    outcomes = replay.replay_predicted_warm()
    assert len(outcomes) == len(replay.predicted_warm_rows())

    large = [outcome for outcome in outcomes if outcome.row["total"] >= 55_000]
    assert large, "the fixture must carry the 55k+ predicted-warm population"
    totals = {outcome.row["total"] for outcome in large}
    named = replay.load_window()[0]["named_replay_events"]
    representatives = {
        named["55k_band"],
        named["67k_band"],
        named["90k_band"],
    }
    assert representatives <= totals
    assert any(55_000 <= total < 65_536 for total in totals)  # 55k band
    assert any(65_536 <= total < 70_000 for total in totals)  # 67k band
    assert any(total >= 85_000 for total in totals)  # 90k band

    for outcome in large:
        decision = outcome.decision
        assert decision.admit is False, outcome.row
        assert decision.reason == prefill_eta.REASON_DEADLINE_CAPACITY, outcome.row
        assert decision.deadline_capacity_calibrated is True
        assert decision.capacity_basis == "cold"
        assert decision.capacity_priced_tokens == outcome.row["total"]


def test_no_admitted_event_projects_above_the_capacity_bound():
    """AC2: over every admitted row, the cold projection fits the bound.

    The bound is a request property, so it is asserted in tokens and in seconds
    (the capacity restated at the decision's own rate) against the shipped
    deadline.
    """
    replay.load_cold_window()
    config = replay.shipped_config()
    deadline_seconds = float(
        config["server"]["local_prefill_deadline_seconds"]
    )
    outcomes = replay.replay_predicted_warm(config)
    admitted = [outcome for outcome in outcomes if outcome.admitted]
    assert admitted, "the window's small predicted-warm events are admissible"
    for outcome in admitted:
        assert outcome.decision.capacity_basis == "cold"
        assert outcome.capacity_tokens is not None
        assert outcome.priced_tokens <= outcome.capacity_tokens, outcome.row
        assert outcome.cold_projection_seconds <= deadline_seconds, outcome.row


def test_uncalibrated_window_admits_the_same_events_fail_open():
    """Without the calibration source the bound is inert (pre-fix behaviour).

    This is both the honest baseline the proof improves on and the documented
    fail-open contract: an uncalibrated cold window never manufactures a
    refusal.
    """
    outcomes = replay.replay_predicted_warm()
    assert outcomes
    for outcome in outcomes:
        assert outcome.admitted is True, outcome.row
        assert outcome.decision.deadline_capacity_calibrated is False


# ===========================================================================
# AC3 — the 2026-11-03 incident
# ===========================================================================


def test_incident_turn_is_priced_cold_by_the_shipped_large_append_rule():
    """AC3: the 51,945-byte tool append forces a full-prefill price.

    The pre-dispatch price trusts the session's measured 0.99 reuse, which
    describes the *previous* prompt and knows nothing about the appended
    SKILL.md. The shipped rule zeroes the credit above
    ``local_large_append_bytes_threshold``, so the row is priced cold.
    """
    replay.load_cold_window()
    pricing = replay.price_incident_turn()
    assert pricing.row["append"]["bytes"] == replay.INCIDENT_APPEND_BYTES
    assert pricing.priced_cached_ratio == 0.0
    assert pricing.forced_full_prefill is True
    assert pricing.capacity_tokens is not None
    # The cold price is the whole measured prompt at the calibrated rate.
    capacity = replay.cold_rate(replay.INCIDENT_PROMPT_TOKENS)
    assert capacity.calibrated is True
    assert pricing.cold_price_seconds == pytest.approx(
        replay.INCIDENT_PROMPT_TOKENS * capacity.cold_ms_per_token / 1000.0
    )


def test_incident_row_is_admitted_at_the_shipped_deadline_with_its_cold_price():
    """AC3(a): at 180 s the row fits — priced cold, inside deadline and capacity.

    This is the correction LP-0MW0KZUUS002LICZ delivered: the turn's work is
    priced honestly and still completes inside the shipped deadline, so it is
    not a doomed dispatch.
    """
    replay.load_cold_window()
    outcome = replay.replay_incident()
    decision = outcome.decision
    assert decision.admit is True
    assert decision.reason == "within_budget"
    assert decision.rate_bucket == "cold"
    assert decision.capacity_basis == "cold"
    assert decision.deadline_capacity_calibrated is True
    assert outcome.cold_price_seconds < decision.prefill_deadline_seconds
    assert outcome.cold_price_seconds < decision.deadline_budget_seconds
    assert outcome.priced_tokens <= outcome.capacity_tokens


def test_incident_row_is_refused_below_its_cold_price_instead_of_dispatched():
    """AC3(b): below its own cold price the row is refused — it is not dispatched.

    The reference deadline is derived from the calibrated rate, never picked by
    hand: ``total * cold_ms_per_token / 1000`` is the price the row names, and a
    deadline 10 % under it must refuse at admission. The shipped precedence
    reports the temporal bound (which fires on the same cold price) rather than
    the redundant capacity; both bounds are asserted to be exceeded.
    """
    replay.load_cold_window()
    cold = replay.price_incident_turn()
    deadline_seconds = cold.cold_price_seconds * 0.9
    outcome = replay.replay_incident(deadline_seconds=deadline_seconds)
    decision = outcome.decision
    assert decision.admit is False
    assert decision.reason == "prefill_eta_exceeds_safety_budget"
    assert outcome.cold_price_seconds > deadline_seconds
    assert outcome.over_capacity is True
    assert outcome.priced_tokens > outcome.capacity_tokens
    # Refused at admission: the prefill the deadline would cancel never starts.
    assert decision.projected_wait_seconds is not None
    assert decision.projected_wait_seconds >= deadline_seconds


def test_incident_capacity_bound_is_the_binding_refusal_for_measured_new_tokens():
    """AC3(b) isolating the capacity: the distinct reason on the measured basis.

    Pricing the row from the measured new tokens (6,622 resident, 13,006 new)
    while residency is unmeasured keeps the capacity on the whole prompt, so at
    the tighter deadline the capacity — not the temporal bound — is what
    refuses. This is the distinct, documented ``prefill_eta_exceeds_capacity``
    refusal the parent AC10 asks for.
    """
    replay.load_cold_window()
    cold = replay.price_incident_turn()
    deadline_seconds = cold.cold_price_seconds * 0.9
    outcome = replay.replay_incident(
        deadline_seconds=deadline_seconds,
        admission_ratio=replay.incident_measured_ratio(cold),
    )
    decision = outcome.decision
    assert decision.admit is False
    assert decision.reason == prefill_eta.REASON_DEADLINE_CAPACITY
    assert outcome.over_capacity is True
    assert decision.capacity_priced_tokens == replay.INCIDENT_PROMPT_TOKENS
    # The temporal bounds are satisfied — only the capacity refuses this row.
    assert decision.eta_prefill_seconds is not None
    assert decision.eta_prefill_seconds < decision.deadline_budget_seconds
    assert decision.eta_prefill_seconds < decision.prefill_deadline_seconds
    assert outcome.cold_price_seconds > decision.prefill_deadline_seconds


def test_incident_fixture_records_both_turns_at_their_measured_numbers():
    """AC3 evidence anchor: the fixture reproduces the recorded incidents numbers.

    Turn 1 is the 51,945-byte append (19,628-token prompt, 6,622 cached, 13,006
    new, 173 priced new, 53.2 s first byte); turn 2 repeats the underestimate at
    72.0 s. The numbers are the recorded evidence, not inferred ones.
    """
    turns = replay.incident_turns()
    assert [turn["turn"] for turn in turns] == [1, 2]
    first, repeat = turns
    assert first["append"]["bytes"] == 51_945
    assert first["prompt_tokens"] == 19_628
    assert first["cached_tokens"] == 6_622
    assert first["new_tokens"] == 13_006
    assert first["priced_new_tokens"] == 173
    assert first["measured_new_tokens"] == 12_936
    assert first["first_byte_ms"] == 53_200
    assert first["measured_cache_abort_deficit_ratio"] == pytest.approx(0.659)
    assert repeat["priced_new_tokens"] == 4_695
    assert repeat["measured_new_tokens"] == 9_264
    assert repeat["first_byte_ms"] == 72_000
    assert repeat["measured_cache_abort_deficit_ratio"] == pytest.approx(0.285)
    # The prompt total of the repeat is not in the recorded evidence.
    assert repeat["prompt_tokens"] is None


# ===========================================================================
# AC4 — deterministic and self-documenting
# ===========================================================================


def test_every_fixture_row_carries_provenance_and_names_its_source():
    """AC4: the fixtures document themselves; the pruned recording is named."""
    window_meta, window_rows = replay.load_window()
    incident_meta, incident_rows = replay.load_incident()

    for row in window_rows + incident_rows:
        assert isinstance(row["provenance"], str) and row["provenance"].strip()

    assert set(window_meta["record_kinds"]) == set(replay.WINDOW_KINDS)
    assert window_meta["source"].startswith("/var/log/llama-proxy")
    assert window_meta["cold_rate_band_ms_per_token"]["min"] == 3.81
    assert window_meta["cold_rate_band_ms_per_token"]["max"] == 4.26

    recording = incident_meta["original_recording"]
    assert recording["status"] == "pruned"
    assert recording["path"].endswith("herdr-1793695298-3654733-20033")
    assert incident_meta["kind"] == "reconstructed"

    repo_root = Path(replay.__file__).resolve().parents[2]
    for fixture in (replay.WINDOW_FIXTURE, replay.INCIDENT_FIXTURE):
        assert fixture.resolve().is_relative_to(repo_root)


def test_loader_rejects_a_row_without_provenance(tmp_path):
    """A row without provenance is fixture corruption, not a replayed event."""
    fixture = tmp_path / "unprovenanced.jsonl"
    fixture.write_text(
        json.dumps({"_meta": {"work_item": "test"}})
        + "\n"
        + json.dumps({"kind": "abandoned", "total": 90_000})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="without provenance"):
        replay.load_fixture(fixture)


def test_replay_resets_calibration_and_is_deterministic():
    """AC4: the same fixture replays identically from an empty window."""

    def signature(outcomes):
        return [
            (
                outcome.row["total"],
                outcome.admitted,
                outcome.decision.reason,
                outcome.priced_tokens,
                outcome.capacity_tokens,
            )
            for outcome in outcomes
        ]

    replay.load_cold_window()
    first = signature(replay.replay_predicted_warm())
    assert prefill_eta.coefficients(
        MODEL, 90_000, replay.shipped_config(), cached_ratio=0.0
    ).cold_samples > 0

    replay.reset()
    assert prefill_eta.coefficients(
        MODEL, 90_000, replay.shipped_config(), cached_ratio=0.0
    ).cold_samples == 0
    replay.load_cold_window()
    second = signature(replay.replay_predicted_warm())
    assert first == second


def test_replay_reads_only_committed_files(monkeypatch):
    """AC4: no row reads a live log or a live recording.

    Every file the replay opens must live inside the repository (the two
    fixtures and ``proxy/config.yaml``). A production log under ``/var/log`` or
    a session recording under ``~/.llm-proxy`` would make the proof
    unreproducible and is rejected here.
    """
    repo_root = Path(replay.__file__).resolve().parents[2]
    opened: list[Path] = []
    real_open = Path.open

    def _guarded_open(self, *args, **kwargs):
        opened.append(Path(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _guarded_open)
    replay.load_cold_window()
    replay.replay_predicted_warm()
    replay.replay_incident()

    assert opened, "the replay must have read the committed fixtures"
    outside = [path for path in opened if not path.resolve().is_relative_to(repo_root)]
    assert outside == []
