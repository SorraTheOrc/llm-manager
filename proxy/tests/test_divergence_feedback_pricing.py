"""Divergence feedback into the session pricing model (LP-0MW81DS4W008DKDV).

Parent: ``LP-0MW7V292T0020G02`` (AC10, merged from ``LP-0MW0KZUUS002LICZ``).

The routing ``cached_ratio`` is a *prediction*: the measured ratio of the
previous local response, credited forward from the session's recorded prompt
prefix (``provider.record_prompt_prefix`` -> ``_last_prompt_prefix`` ->
``provider.expected_cached_ratio``).  When the prediction is wrong the wrong
number is carried into the next turn, because nothing writes the measured
truth back into the model.

Measured 2026-11-03 08:41:54 (certified window used by the parent):

- ``routing_check estimated_tokens=18516 new_tokens=173 cached_ratio=0.99``
- llama-server then prefilled 13,002 tokens;
- ``measured_cache_abort measured_new_tokens=12936 priced_new_tokens=173
  deficit_ratio=0.659``, 53.2 s to first byte — repeated at 08:43-08:44
  (``deficit_ratio=0.285``, 72.0 s).

On 2026-11-07 20:48:51 the same shape appeared at 55,492 tokens
(``cached_ratio=1.00 new_tokens=172``) and was evicted at 180.1 s.

The rule under test — **post-measurement divergence feeds back into the
session pricing model**:

- when the measured ``(total - cache) / total`` deficit is at or above
  ``server.local_divergence_feedback_min_deficit_ratio`` (default ``0.25``),
  the session's next pricing basis is corrected to what was actually measured:
  the measured reuse ratio becomes ``cache / total`` and the recorded prefix
  credit is capped at the ``cache`` tokens proven resident;
- the correction is **bounded**: it applies to at most
  ``local_divergence_feedback_max_turns`` recorded dispatches (default 1) and
  expires after ``local_divergence_feedback_max_age_seconds`` (default 900),
  so a session can never be poisoned permanently — the next dispatch rewrites
  the basis and a genuine warm turn restores the credit;
- it is **cleared** on session invalidation, slot reassignment and model
  switch, exactly like the measurement it derives from;
- ``local_divergence_feedback_enabled: false`` restores the previous pricing
  byte-for-byte.

AC coverage:

- AC1 (feedback): a divergent measurement corrects the next turn's price;
- AC2 (routed cold): the second turn is priced from the corrected model and,
  with the unmeasured-cold rule, is routed cold (and also routes cold on the
  predicted path when that rule is rolled back);
- AC3 (bounded + cleared): one-turn/TTL bounds, and every invalidation path.

Only the documented in-process stores are touched: no HTTP, no sleeps, no
``/slots`` query.
"""

from __future__ import annotations

import logging

import pytest
from proxy.provider import (
    DEFAULT_DIVERGENCE_FEEDBACK_MAX_TURNS,
    DIVERGENCE_FEEDBACK_ENABLED_KEY,
    DIVERGENCE_FEEDBACK_MAX_AGE_SECONDS_KEY,
    DIVERGENCE_FEEDBACK_MAX_TURNS_KEY,
    DIVERGENCE_FEEDBACK_MIN_DEFICIT_RATIO_KEY,
    _cached_ratio_source,
    _divergence_feedback,
    _last_cached_ratio,
    _last_prompt_prefix,
    _should_skip_local,
    clear_cached_ratio_for_session,
    clear_divergence_feedback,
    expected_cached_ratio,
    record_measured_divergence,
    record_prompt_prefix,
    set_cached_ratio,
)

from proxy import observability, provider
from proxy import session as session_registry

MODEL = "Qwen3"
OTHER_MODEL = "Qwen3-Other"
SESSION = "sess-divergence"
OTHER_SESSION = "sess-other"

#: Shipped thresholds (``proxy/config.yaml`` server block) — unchanged here.
COLD_THRESHOLD = 38_000
WARM_THRESHOLD = 120_000
RESIDENT_PREFIX_MIN_RATIO = 0.5

#: The 2026-11-07 20:48:51 shape: a 55,492-token prompt priced
#: ``cached_ratio=1.00 new_tokens=172`` and then evicted at 180.1 s.
INCIDENT_TOTAL = 55_492
INCIDENT_CACHE = 200
INCIDENT_PRICED = 172
INCIDENT_DEFICIT = (INCIDENT_TOTAL - INCIDENT_CACHE) / INCIDENT_TOTAL

#: The next turn's prompt estimate (the previous prompt plus a small append).
NEXT_TURN_ESTIMATE = 55_600

#: The optimistic ratio the session's model believed before the feedback.
OPTIMISTIC_RATIO = 0.99


@pytest.fixture(autouse=True)
def _clean_state():
    """Isolate every in-process store the pricing model reads."""
    _last_cached_ratio.clear()
    _cached_ratio_source.clear()
    _last_prompt_prefix.clear()
    _divergence_feedback.clear()
    observability._last_slot_details_cache[:] = []
    observability._last_slot_details_at = 0.0
    session_registry._slot_owners.clear()
    yield
    _last_cached_ratio.clear()
    _cached_ratio_source.clear()
    _last_prompt_prefix.clear()
    _divergence_feedback.clear()
    observability._last_slot_details_cache[:] = []
    observability._last_slot_details_at = 0.0
    session_registry._slot_owners.clear()


def _base_messages() -> list[dict]:
    """The already-dispatched prefix the session's model was recorded for."""
    return [
        {"role": "system", "content": "you are a coding agent"},
        {"role": "user", "content": "read the skill file"},
    ]


def _next_turn_messages() -> list[dict]:
    """The incoming turn: the recorded prefix plus one appended message."""
    return _base_messages() + [{"role": "user", "content": "next turn"}]


def _prime_optimistic_session(*, prev_prompt_tokens: int = INCIDENT_TOTAL) -> None:
    """Record the under-estimating model the feedback must correct.

    The measured ratio is optimistic (``0.99``) and the recorded prefix is the
    previous prompt's full size, so the next turn is credited as warm — the
    exact state that priced the incident ``cached_ratio=0.99 new_tokens=173``.
    """
    set_cached_ratio(MODEL, SESSION, OPTIMISTIC_RATIO)
    record_prompt_prefix(MODEL, SESSION, prev_prompt_tokens, _base_messages())


def _price(
    messages=None,
    *,
    estimated_tokens: int = NEXT_TURN_ESTIMATE,
    server_config: dict | None = None,
    model: str = MODEL,
    session: str = SESSION,
) -> float:
    """Price the turn exactly as the router's predicted path does."""
    return expected_cached_ratio(
        model,
        session,
        estimated_tokens,
        _next_turn_messages() if messages is None else messages,
        server_config=server_config,
    )


def _route(
    server_config: dict | None = None,
    *,
    estimated_tokens: int = NEXT_TURN_ESTIMATE,
    unmeasured_cold_enabled: bool = True,
):
    """Run the local admission ladder exactly as the router calls it."""
    return _should_skip_local(
        MODEL,
        SESSION,
        {"messages": _next_turn_messages()},
        COLD_THRESHOLD,
        estimated_tokens=estimated_tokens,
        warm_cache_threshold=WARM_THRESHOLD,
        resident_prefix_min_ratio=RESIDENT_PREFIX_MIN_RATIO,
        unmeasured_cold_enabled=unmeasured_cold_enabled,
        server_config=server_config,
    )


def _record(**overrides):
    """Record the incident's measured divergence (overridable per test)."""
    kwargs = {
        "total": INCIDENT_TOTAL,
        "cache": INCIDENT_CACHE,
        "server_config": None,
        "priced_new_tokens": INCIDENT_PRICED,
    }
    kwargs.update(overrides)
    return record_measured_divergence(
        MODEL,
        SESSION,
        kwargs["total"],
        kwargs["cache"],
        kwargs["server_config"],
        priced_new_tokens=kwargs["priced_new_tokens"],
    )


# ---------------------------------------------------------------------------
# AC1 — the divergence corrects the next turn's price
# ---------------------------------------------------------------------------


class TestDivergenceCorrectsTheNextPrice:
    def test_the_uncorrected_model_is_the_defect(self):
        """Baseline: without feedback the next turn is credited as warm."""
        _prime_optimistic_session()
        ratio = _price()
        assert ratio == pytest.approx(OPTIMISTIC_RATIO)
        # 55,600 × (1 − 0.99) ≈ 556 new tokens: far under the cold threshold,
        # which is why the incident was admitted and then re-prefilled.
        assert int(NEXT_TURN_ESTIMATE * (1 - ratio)) < COLD_THRESHOLD

    def test_divergent_measurement_corrects_the_next_price(self):
        """AC1: the measured deficit feeds back into the pricing model."""
        _prime_optimistic_session()
        correction = _record()

        assert correction is not None
        assert correction.reusable_tokens == INCIDENT_CACHE
        assert correction.deficit_ratio == pytest.approx(INCIDENT_DEFICIT)
        assert (MODEL, SESSION) in _divergence_feedback

        # The measured reuse ratio becomes the model's ratio: 200 / 55,492.
        assert _last_cached_ratio[(MODEL, SESSION)] == pytest.approx(
            INCIDENT_CACHE / INCIDENT_TOTAL
        )
        ratio = _price()
        assert ratio == pytest.approx(INCIDENT_CACHE / NEXT_TURN_ESTIMATE, abs=1e-3)
        # The price is now the measured work, not the priced 173–556 tokens.
        assert int(NEXT_TURN_ESTIMATE * (1 - ratio)) > COLD_THRESHOLD

    def test_the_correction_replaces_the_prefix_credit_on_the_next_dispatch(self):
        """The corrected turn's own prefix record keeps the measured ceiling."""
        _prime_optimistic_session()
        _record()

        # The corrected turn completes and records its own (larger) prompt.
        record_prompt_prefix(MODEL, SESSION, NEXT_TURN_ESTIMATE, _next_turn_messages())
        stored_tokens, _fingerprints = _last_prompt_prefix[(MODEL, SESSION)]
        assert stored_tokens == INCIDENT_CACHE
        # One turn was applied (the default bound) so the correction is spent.
        assert (MODEL, SESSION) not in _divergence_feedback

    def test_a_below_threshold_measurement_leaves_the_model_untouched(self):
        """A warm dispatch is not a divergence: no correction, same price."""
        _prime_optimistic_session()
        # 1 % deficit — the measured warm path (ratio floor ceiling is 0.01).
        assert _record(total=56_000, cache=55_440) is None
        assert _divergence_feedback == {}
        assert _last_cached_ratio[(MODEL, SESSION)] == pytest.approx(OPTIMISTIC_RATIO)
        assert _price() == pytest.approx(OPTIMISTIC_RATIO)

    def test_the_threshold_is_inclusive_and_configurable(self):
        """AC1: the configurable threshold decides (>= boundary)."""
        _prime_optimistic_session()
        # A 30 % deficit with the threshold at exactly 0.30 fires (inclusive).
        at_threshold = {DIVERGENCE_FEEDBACK_MIN_DEFICIT_RATIO_KEY: 0.30}
        assert _record(total=1_000, cache=700, server_config=at_threshold) is not None

        _divergence_feedback.clear()
        _prime_optimistic_session()
        stricter = {DIVERGENCE_FEEDBACK_MIN_DEFICIT_RATIO_KEY: 0.31}
        assert _record(total=1_000, cache=700, server_config=stricter) is None
        assert _price() == pytest.approx(OPTIMISTIC_RATIO)

    def test_flat_and_nested_config_shapes_are_accepted(self):
        """The resolver reads the ``server`` section or a flat dict."""
        _prime_optimistic_session()
        nested = {
            "server": {
                DIVERGENCE_FEEDBACK_MIN_DEFICIT_RATIO_KEY: INCIDENT_DEFICIT,
                DIVERGENCE_FEEDBACK_MAX_TURNS_KEY: 2,
            }
        }
        correction = _record(server_config=nested)
        assert correction is not None
        assert correction.turns_remaining == 2


# ---------------------------------------------------------------------------
# AC2 — the corrected price routes the second turn cold
# ---------------------------------------------------------------------------


class TestSecondTurnRoutesCold:
    def test_second_turn_is_routed_cold_by_the_unmeasured_rule(self):
        """AC2: corrected price + unmeasured residency ⇒ cold (remote)."""
        _prime_optimistic_session()
        _record()

        skip, reason = _route()
        assert (skip, reason) == (True, provider.SKIP_REASON_UNMEASURED_COLD)

    def test_corrected_price_alone_routes_cold_on_the_predicted_path(self):
        """AC2: with the unmeasured-cold rule rolled back the price decides.

        The corrected price (≈ full prefill) exceeds the 38,000 cold budget on
        its own, so the predicted rung ``large_context_bypass`` still sends the
        turn remote — the feedback does not depend on the other rule.
        """
        _prime_optimistic_session()
        _record()

        skip, reason = _route(unmeasured_cold_enabled=False)
        assert (skip, reason) == (True, "large_context_bypass")

    def test_warm_session_still_prices_warm_without_a_divergence(self):
        """Regression: no divergence ⇒ the optimistic price is unchanged."""
        _prime_optimistic_session()
        skip, reason = _route(unmeasured_cold_enabled=False)
        assert (skip, reason) == (False, None)


# ---------------------------------------------------------------------------
# AC3 — bounded feedback (no permanent poisoning)
# ---------------------------------------------------------------------------


class TestFeedbackIsBounded:
    def test_default_bound_is_one_turn(self):
        """The correction applies to exactly the next recorded dispatch."""
        assert DEFAULT_DIVERGENCE_FEEDBACK_MAX_TURNS == 1
        _prime_optimistic_session()
        _record()
        record_prompt_prefix(MODEL, SESSION, NEXT_TURN_ESTIMATE, _next_turn_messages())
        assert _last_prompt_prefix[(MODEL, SESSION)][0] == INCIDENT_CACHE
        assert (MODEL, SESSION) not in _divergence_feedback

        # The following dispatch records its own full prompt again: a genuine
        # warm turn restores the credit (the model is never poisoned for good).
        third_turn = _next_turn_messages() + [{"role": "user", "content": "third"}]
        record_prompt_prefix(MODEL, SESSION, NEXT_TURN_ESTIMATE, third_turn)
        assert _last_prompt_prefix[(MODEL, SESSION)][0] == NEXT_TURN_ESTIMATE

    def test_max_turns_bound_is_configurable(self):
        """A bound of two applies the ceiling to two recorded dispatches."""
        _prime_optimistic_session()
        config = {DIVERGENCE_FEEDBACK_MAX_TURNS_KEY: 2}
        assert _record(server_config=config) is not None

        record_prompt_prefix(MODEL, SESSION, NEXT_TURN_ESTIMATE, _next_turn_messages())
        assert _last_prompt_prefix[(MODEL, SESSION)][0] == INCIDENT_CACHE
        assert (MODEL, SESSION) in _divergence_feedback

        record_prompt_prefix(MODEL, SESSION, NEXT_TURN_ESTIMATE, _next_turn_messages())
        assert _last_prompt_prefix[(MODEL, SESSION)][0] == INCIDENT_CACHE
        assert (MODEL, SESSION) not in _divergence_feedback

    def test_feedback_expires_after_its_max_age(self, monkeypatch):
        """A quiet session cannot carry a stale correction forever."""
        clock = {"now": 1_000.0}
        monkeypatch.setattr(provider.time, "monotonic", lambda: clock["now"])

        _prime_optimistic_session()
        config = {DIVERGENCE_FEEDBACK_MAX_AGE_SECONDS_KEY: 30}
        assert _record(server_config=config) is not None

        clock["now"] += 31
        record_prompt_prefix(MODEL, SESSION, NEXT_TURN_ESTIMATE, _next_turn_messages())
        assert _last_prompt_prefix[(MODEL, SESSION)][0] == NEXT_TURN_ESTIMATE
        assert (MODEL, SESSION) not in _divergence_feedback

    def test_a_malformed_max_turns_falls_back_to_the_default(self):
        """A typo never silently disables the bound (or unbounds it)."""
        _prime_optimistic_session()
        correction = _record(server_config={DIVERGENCE_FEEDBACK_MAX_TURNS_KEY: "lots"})
        assert correction is not None
        assert correction.turns_remaining == DEFAULT_DIVERGENCE_FEEDBACK_MAX_TURNS

    @pytest.mark.parametrize("disabled", [False, 0, "off", "false"])
    def test_disabled_feedback_restores_the_previous_price(self, disabled):
        """AC3 rollback: one config value restores the pre-change pricing."""
        _prime_optimistic_session()
        before = _price()
        assert _record(server_config={DIVERGENCE_FEEDBACK_ENABLED_KEY: disabled}) is None
        assert _divergence_feedback == {}
        assert _price() == pytest.approx(before)

    def test_malformed_config_fails_open_to_pre_change_pricing(self):
        """An unreadable config never corrects a price on a guess."""
        _prime_optimistic_session()
        assert _record(server_config={DIVERGENCE_FEEDBACK_ENABLED_KEY: "maybe"}) is None
        assert _record(server_config=["not", "a", "mapping"]) is None
        assert _divergence_feedback == {}
        assert _price() == pytest.approx(OPTIMISTIC_RATIO)


# ---------------------------------------------------------------------------
# AC3 — cleared on session invalidation, slot reassignment, model switch
# ---------------------------------------------------------------------------


class TestFeedbackIsCleared:
    def test_session_invalidation_clears_the_feedback(self):
        _prime_optimistic_session()
        _record()
        clear_cached_ratio_for_session(SESSION)
        assert (MODEL, SESSION) not in _divergence_feedback
        assert (MODEL, SESSION) not in _last_prompt_prefix

    def test_slot_reassignment_clears_the_feedback(self):
        """A slot changing hands invalidates the prefix basis it measured."""
        _prime_optimistic_session()
        _record()
        session_registry._free_slot_assignment(SESSION)
        assert (MODEL, SESSION) not in _divergence_feedback

    def test_model_switch_clears_a_departing_models_feedback(self):
        """A switch keeps only the names provably belonging to the target."""
        from proxy.lifecycle import _forget_switched_model_measurements

        _prime_optimistic_session()
        _record()
        departing = "Qwen3-Departing"
        set_cached_ratio(departing, SESSION, OPTIMISTIC_RATIO)
        assert (
            record_measured_divergence(
                departing, SESSION, INCIDENT_TOTAL, INCIDENT_CACHE
            )
            is not None
        )

        _forget_switched_model_measurements(MODEL, OTHER_MODEL)
        assert (departing, SESSION) not in _divergence_feedback
        # The requested/target names are kept (MODEL is one of them here).
        assert (MODEL, SESSION) in _divergence_feedback

    def test_model_switch_keeps_the_target_models_feedback(self):
        from proxy.lifecycle import _forget_switched_model_measurements

        _prime_optimistic_session()
        _record()
        _forget_switched_model_measurements(MODEL, MODEL)
        assert (MODEL, SESSION) in _divergence_feedback

    def test_a_backend_restart_clears_every_correction(self):
        from proxy.lifecycle import _forget_switched_model_measurements

        _prime_optimistic_session()
        _record()
        _forget_switched_model_measurements(MODEL, MODEL, backend_restarted=True)
        assert _divergence_feedback == {}

    def test_clear_is_dimensioned_by_the_identity_known(self):
        """Session-only, model-only and pair clears, like the measurement store."""
        _prime_optimistic_session()
        _record()
        set_cached_ratio(OTHER_MODEL, SESSION, OPTIMISTIC_RATIO)
        record_measured_divergence(
            OTHER_MODEL, SESSION, INCIDENT_TOTAL, INCIDENT_CACHE, None
        )
        set_cached_ratio(MODEL, OTHER_SESSION, OPTIMISTIC_RATIO)
        record_measured_divergence(
            MODEL, OTHER_SESSION, INCIDENT_TOTAL, INCIDENT_CACHE, None
        )

        clear_divergence_feedback(None, SESSION)
        assert (MODEL, SESSION) not in _divergence_feedback
        assert (OTHER_MODEL, SESSION) not in _divergence_feedback
        assert (MODEL, OTHER_SESSION) in _divergence_feedback

        clear_divergence_feedback(MODEL, None)
        assert (MODEL, OTHER_SESSION) not in _divergence_feedback


# ---------------------------------------------------------------------------
# Scoping and fail-open behaviour
# ---------------------------------------------------------------------------


class TestScopingAndFailOpen:
    def test_other_sessions_and_models_are_not_corrected(self):
        _prime_optimistic_session()
        _record()

        # Another session of the same model keeps its own (warm) basis.
        set_cached_ratio(MODEL, OTHER_SESSION, OPTIMISTIC_RATIO)
        record_prompt_prefix(MODEL, OTHER_SESSION, INCIDENT_TOTAL, _base_messages())
        record_prompt_prefix(MODEL, OTHER_SESSION, NEXT_TURN_ESTIMATE, _next_turn_messages())
        assert _last_prompt_prefix[(MODEL, OTHER_SESSION)][0] == NEXT_TURN_ESTIMATE
        assert (MODEL, OTHER_SESSION) not in _divergence_feedback

        # Another model of the same session is untouched too.
        set_cached_ratio(OTHER_MODEL, SESSION, OPTIMISTIC_RATIO)
        record_prompt_prefix(OTHER_MODEL, SESSION, INCIDENT_TOTAL, _base_messages())
        record_prompt_prefix(
            OTHER_MODEL, SESSION, NEXT_TURN_ESTIMATE, _next_turn_messages()
        )
        assert _last_prompt_prefix[(OTHER_MODEL, SESSION)][0] == NEXT_TURN_ESTIMATE

    @pytest.mark.parametrize(
        ("total", "cache"),
        [
            (0, 0),
            (-1, 0),
            (100, 200),
            (100, -1),
            ("not-a-number", 0),
            (None, None),
            (True, False),
        ],
    )
    def test_invalid_measurements_are_ignored(self, total, cache):
        """A malformed measurement is not a divergence — never raises."""
        _prime_optimistic_session()
        assert _record(total=total, cache=cache) is None
        assert _divergence_feedback == {}
        assert _price() == pytest.approx(OPTIMISTIC_RATIO)

    def test_blank_identity_is_a_no_op(self):
        _prime_optimistic_session()
        assert record_measured_divergence("", SESSION, INCIDENT_TOTAL, INCIDENT_CACHE) is None
        assert record_measured_divergence(MODEL, None, INCIDENT_TOTAL, INCIDENT_CACHE) is None
        assert _divergence_feedback == {}

    def test_the_correction_is_logged(self, caplog):
        """AC1 observability: the correction is attributable from the log."""
        _prime_optimistic_session()
        with caplog.at_level(logging.INFO, logger=provider.logger.name):
            _record()

        line = next(
            record.getMessage()
            for record in caplog.records
            if "routing_divergence_feedback" in record.getMessage()
        )
        assert "model=Qwen3" in line
        assert f"session={SESSION}" in line
        assert "deficit_ratio=" in line
        assert f"measured_reusable_tokens={INCIDENT_CACHE}" in line
        assert "measured_new_tokens=" in line
        assert f"priced_new_tokens={INCIDENT_PRICED}" in line

    def test_the_applied_correction_is_logged(self, caplog):
        """The turn that is priced from the correction is attributable."""
        _prime_optimistic_session()
        _record()
        with caplog.at_level(logging.INFO, logger=provider.logger.name):
            record_prompt_prefix(
                MODEL, SESSION, NEXT_TURN_ESTIMATE, _next_turn_messages()
            )

        line = next(
            record.getMessage()
            for record in caplog.records
            if "routing_divergence_feedback_applied" in record.getMessage()
        )
        assert f"recorded_tokens={NEXT_TURN_ESTIMATE}" in line
        assert f"corrected_tokens={INCIDENT_CACHE}" in line
