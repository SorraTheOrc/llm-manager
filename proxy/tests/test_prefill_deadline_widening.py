"""Widened local prefill deadline: 120 s -> 180 s (LP-0MW1V0LPM000QLUC).

Problem (measured 2026-11-04 01:00-04:00 UTC, 140 requests, 138 ended
``All providers exhausted``): 61 local refusals logged
``reason=prefill_eta_exceeds_safety_budget`` at ``bound_s=114.0`` with priced
``eta_prefill_s`` between 117.0 s and 169.2 s. Every one of them clears the
widened bound, and the GPU was idle with both slots free when they were
refused.

The representative refusal (2026-11-04 02:01:49, ``provider=local-qwen3``)::

    local_admission_guard decision=reject
      reason=prefill_eta_exceeds_safety_budget eta_prefill_s=128.1
      bound_s=114.0 deadline_s=120.0 new_tokens=23069 ...

Acceptance criteria covered here:

* AC1 — ``proxy/config.yaml`` sets ``local_prefill_deadline_seconds: 180`` and
  the documented startup ordering still holds.
* AC2 — ``prefill_deadline_budget_seconds`` resolves to **171.0 s** from the
  shipped config (``0.95 x 180 s``).
* AC3 — the 2026-11-04 02:01:49 case is **admitted**, and the same priced ETA
  is refused against the pre-change bound, so the widening is what admits it.
* AC4 — the shutdown/safety behaviour at the new bound is unchanged: nothing
  priced at or above the deadline is admitted.
* The refactor AC — the in-code default is ONE constant shared by
  ``router_helpers._get_prefill_deadline_seconds`` and the documented default.

Every figure the *code* enforces is derived from the shipped config or from
the production helper; the two deliberately frozen literals are the behavioural
claims of this item (180 / 171.0) and are part of its recorded residual edit
set — see the work item.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from proxy import prefill_eta, router_helpers

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "proxy" / "config.yaml"

DEADLINE_KEY = "local_prefill_deadline_seconds"
SAFETY_FRACTION_KEY = "local_admission_guard_budget_safety_fraction"

#: The shipped decision this item records: ``120 -> 180`` s, so the derived
#: safety budget moves ``114 -> 171`` s. Deliberate literal pins (AC1/AC2).
SHIPPED_DEADLINE_S = 180.0
SHIPPED_BUDGET_S = 171.0

#: The model the 2026-11-04 refusals were logged against.
MODELLED_MODEL = "Qwen3"

#: Deterministic calibration: a single-rate window for the bucket the refused
#: prompts fell in. Percentile pricing is switched off and the flat cold-prefill
#: cap disabled in ``_guard_config`` so this module prices the *deadline* bound
#: alone (both knobs are inputs of other gates, not of the bound under test).
_SEED_TOKENS = 27_369
_SEED_MS_PER_TOKEN = 3.9
_SEED_SAMPLES = 20

#: The priced ETA of the recorded 02:01:49 refusal.
RECORDED_ETA_S = 128.1


@pytest.fixture(autouse=True)
def _clean_calibration():
    prefill_eta.reset()
    yield
    prefill_eta.reset()


def _shipped_server() -> dict:
    return (yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {})["server"]


def _guard_config(**overrides) -> dict:
    """The shipped config with the *other* gates' knobs neutralised."""
    server = dict(_shipped_server())
    server["local_admission_guard_eta_percentile_min_tokens"] = 10**9
    server["local_cold_prefill_budget_tokens"] = 0
    server.update(overrides)
    return {"server": server}


def _seed_window() -> None:
    for _ in range(_SEED_SAMPLES):
        prefill_eta.record_dispatch_sample(
            MODELLED_MODEL,
            _SEED_TOKENS,
            0.0,
            _SEED_TOKENS * _SEED_MS_PER_TOKEN,
        )


def _admission(tokens: int, config: dict | None = None):
    _seed_window()
    return prefill_eta.evaluate_admission(
        _guard_config() if config is None else config,
        MODELLED_MODEL,
        tokens,
        0.0,
    )


# ---------------------------------------------------------------------------
# AC1 — the shipped value and the documented ordering
# ---------------------------------------------------------------------------

def test_shipped_config_sets_the_widened_deadline():
    """AC1: ``local_prefill_deadline_seconds`` ships at 180."""
    assert float(_shipped_server()[DEADLINE_KEY]) == pytest.approx(SHIPPED_DEADLINE_S)


def test_shipped_config_passes_the_documented_ordering():
    """AC1: 30 < 180 < 840 < 900 < the client 900 s queue deadline."""
    config = {"server": _shipped_server()}
    server = config["server"]

    assert prefill_eta.validate_prefill_timeout_ordering(config) == []

    deadline = float(server[DEADLINE_KEY])
    # Both upper bounds are documented defaults (commented out in config.yaml
    # unless an operator overrides them), so read them with the same default.
    first_byte = float(server.get("local_dispatch_first_byte_timeout_seconds", 840))
    max_prefill = float(server.get("local_dispatch_max_prefill_seconds", 900))

    assert float(server["contention_queue_max_wait_seconds"]) < deadline
    assert deadline < first_byte == 840.0
    assert 840.0 < max_prefill == 900.0
    assert deadline < prefill_eta.CLIENT_QUEUE_DEADLINE_SECONDS == 900.0


# ---------------------------------------------------------------------------
# AC2 — the derived safety budget
# ---------------------------------------------------------------------------

def test_shipped_safety_budget_is_the_derived_value():
    """AC2: the comparator is ``fraction x deadline`` = 171 s shipped."""
    server = _shipped_server()
    budget = prefill_eta.prefill_deadline_budget_seconds({"server": server})

    assert budget == pytest.approx(
        float(server[SAFETY_FRACTION_KEY]) * float(server[DEADLINE_KEY])
    )
    assert budget == pytest.approx(SHIPPED_BUDGET_S)
    # The guard enforces the same derived number, so advertise/enforce agree.
    assert _admission(_SEED_TOKENS).deadline_budget_seconds == pytest.approx(
        SHIPPED_BUDGET_S
    )


def test_the_budget_tracks_the_deadline_it_is_derived_from():
    """A retuned deadline moves the budget with it — no second edit."""
    fraction = float(_shipped_server()[SAFETY_FRACTION_KEY])

    for deadline in (120.0, 200.0):
        budget = prefill_eta.prefill_deadline_budget_seconds(
            _guard_config(**{DEADLINE_KEY: deadline})
        )
        assert budget == pytest.approx(fraction * deadline)


# ---------------------------------------------------------------------------
# AC3 — the recorded 2026-11-04 02:01:49 refusal is now admitted
# ---------------------------------------------------------------------------

def test_the_2026_11_04_refused_prompt_is_now_admitted():
    """AC3: ``eta_prefill_s=128.1`` clears the widened bound and is admitted."""
    decision = _admission(_SEED_TOKENS)

    assert decision.calibrated is True
    assert decision.eta_prefill_seconds == pytest.approx(RECORDED_ETA_S, abs=0.2)
    assert decision.deadline_budget_seconds == pytest.approx(SHIPPED_BUDGET_S)
    # The recorded refusals priced 117.0-169.2 s: strictly inside the band the
    # widened bound and the old bound straddle.
    assert 114.0 < decision.eta_prefill_seconds < SHIPPED_BUDGET_S
    assert decision.admit is True
    assert decision.reason == "within_budget"


def test_the_same_priced_eta_is_refused_against_the_pre_change_bound():
    """AC3 control: the 120 s deadline (114 s bound) is what used to refuse it.

    Without this row the admission above could be explained by anything; with
    it, the value change is what admits the recorded prompt.
    """
    decision = _admission(_SEED_TOKENS, _guard_config(**{DEADLINE_KEY: 120.0}))

    assert decision.eta_prefill_seconds == pytest.approx(RECORDED_ETA_S, abs=0.2)
    assert decision.deadline_budget_seconds == pytest.approx(114.0)
    assert decision.admit is False
    assert decision.reason == "prefill_eta_exceeds_safety_budget"


# ---------------------------------------------------------------------------
# AC4 — the safety behaviour at the new bound is unchanged
# ---------------------------------------------------------------------------

def test_a_prompt_above_the_widened_bound_is_still_refused():
    """AC4: nothing priced between the bound and the deadline is admitted."""
    decision = _admission(37_600)

    assert decision.eta_prefill_seconds > SHIPPED_BUDGET_S
    assert decision.eta_prefill_seconds < SHIPPED_DEADLINE_S
    assert decision.admit is False
    assert decision.reason == "prefill_eta_exceeds_safety_budget"


def test_a_prompt_priced_at_the_new_deadline_is_still_refused():
    """AC4: the deadline backstop itself still refuses (no shutdown change)."""
    decision = _admission(39_500)

    assert decision.prefill_deadline_seconds == pytest.approx(SHIPPED_DEADLINE_S)
    assert decision.eta_prefill_seconds > SHIPPED_DEADLINE_S
    assert decision.admit is False
    assert decision.reason.startswith("prefill_eta_exceeds_")


# ---------------------------------------------------------------------------
# The refactor AC — ONE in-code default constant
# ---------------------------------------------------------------------------

class _Srv:
    """Minimal ``server``-shaped stub for the router helper."""

    def __init__(self, config: dict | None = None, *, raise_on_config: bool = False):
        self._config = config or {"server": {}}
        self._raise = raise_on_config

    @property
    def config(self) -> dict:
        if self._raise:
            raise RuntimeError("boom")
        return self._config


def test_the_router_fallback_uses_the_shared_default_constant(monkeypatch):
    """One constant behind both the fallback and the documented default.

    Patching the module constant must move the fallback — that is what proves
    a single source of truth rather than two literals that happen to agree.
    """
    monkeypatch.setattr(prefill_eta, "DEFAULT_PREFILL_DEADLINE_SECONDS", 137.0)

    assert router_helpers._get_prefill_deadline_seconds(_Srv()) == 137.0
    assert (
        router_helpers._get_prefill_deadline_seconds(_Srv(raise_on_config=True))
        == 137.0
    )
    # An explicit value still wins over the fallback.
    configured = {"server": {DEADLINE_KEY: 45}}
    assert router_helpers._get_prefill_deadline_seconds(_Srv(configured)) == 45.0


def test_the_documented_default_names_the_constant_not_a_literal():
    """The docstring points at the constant, so prose and code cannot drift."""
    doc = router_helpers._get_prefill_deadline_seconds.__doc__ or ""

    assert "DEFAULT_PREFILL_DEADLINE_SECONDS" in doc
    assert prefill_eta.DEFAULT_PREFILL_DEADLINE_SECONDS == 90
