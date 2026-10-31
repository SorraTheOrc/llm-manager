"""Benefit-gate decision rows for the measured-cache early abort.

LP-0MVWDZ157001KYLO (parent LP-0MVWBD91Z004U135). These rows pin the decision
surface of the *benefit* gate before the implementation lands
(LP-0MVWDZ1GG004UGIK turns them green):

* the abort fires only when the **measured** cold prefill genuinely cannot
  finish inside the request's own prefill-deadline budget
  (``budget_safety_fraction x local_prefill_deadline_seconds`` = 114 s with the
  shipped ``proxy/config.yaml``) — the admission wait budget
  (``contention_queue_max_wait_seconds``, 60 s) is **not** the comparator and
  the recorded admission evidence (``budget_s=60.0`` / ``deadline_s=120.0`` /
  ``new_tokens=49``) is never an input to the decision;
* the ETA is priced from the measured deficit ``total - cache`` with the
  admission guard's own calibrated model, never from the admission-priced
  ``new_tokens``;
* an unresolvable budget fails open (continue), so a missing budget can never
  manufacture an abort.

Rate basis (recorded on purpose — a future recalibration that re-breaks the
incident must fail loudly, not silently)
--------------------------------------------------------------------------
Measured on the host on 2026-10-31 from 224 paired
``dispatch_to_first_byte_ms`` / ``measured_new_tokens`` samples:

* host prefill rate: median **371 tok/s**, p25 **170 tok/s**;
* shipped default (uncalibrated) fit ``DEFAULT_PREFILL_MS_PER_TOKEN = 3.9``
  (256 tok/s) — ``14 380 x 3.9 ms = 56.08 s``, ``56 000 x 3.9 ms = 218.4 s``;
* 14 380 cold tokens therefore cost **38.8 s** at the median rate and
  **84.6 s** at the p25 rate;
* 56 000 cold tokens cost **150.9 s** at the median rate;
* comparator budget: ``0.95 x 120 s = 114 s``.

Both incident and huge-prompt rows assert those numbers, so the incident prompt
keeps continuing even at the p25 rate (84.6 s < 114 s) and the huge prompt keeps
aborting at every measured rate (150.9 s and 218.4 s > 114 s).

API contract pinned here
------------------------
``AbortSettings`` gains ``model``, ``config``, ``budget_seconds`` (resolved by
:meth:`AbortSettings.from_config` via
:func:`proxy.prefill_eta.prefill_deadline_budget_seconds`) and ``eta_estimator``
— a one-argument callable that receives the **measured** deficit in tokens and
returns seconds; ``None`` uses
``prefill_eta.estimate_prefill_seconds(model, deficit, 0.0, config)``.
``AbortDecision`` gains ``measured_eta_s`` and ``budget_s``.

``budget_seconds is None`` means "the benefit gate is not configured" and
preserves the pre-gate verdict byte-for-byte — that is what keeps every
pre-existing decision row (built from ``AbortSettings()`` without a config)
unchanged. ``budget_seconds <= 0`` means "configured, but no usable budget" and
fails open. ``DETAIL_WITHIN_ETA_BUDGET`` (``"within_eta_budget"``) is the single
new continue token.

Budget resolution distinguishes **absent** from **present but zero** (parent
AC6, D1(a)): an absent / unreadable / non-positive
``local_prefill_deadline_seconds`` or ``local_admission_guard_budget_safety_fraction``
yields no usable budget (fail open — a default must never manufacture an abort),
an explicitly present safety fraction of ``0`` yields the bare
``local_prefill_deadline_seconds``, and a fraction in ``(0, 1]`` yields
``fraction x deadline``. ``proxy/config.yaml`` declares both keys, so the
shipped 114 s comparator is unaffected by that distinction.

Why every row is ``xfail(strict=True)`` today
---------------------------------------------
This item is test-first: the rows must *fail* on today's code (they reference
``budget_s`` / ``measured_eta_s`` / ``DETAIL_WITHIN_ETA_BUDGET`` /
``prefill_deadline_budget_seconds``, none of which exist yet) while the
changed-scope test gate has to stay green for this item to reach review.
``xfail(strict=True)`` states exactly that: xfail now, and a hard failure the
moment a row starts *passing* — so the implementation item must delete
``pytestmark`` below rather than leave the rows silently unfixed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml
from proxy.measured_cache_abort import (
    DETAIL_ABORT,
    DETAIL_DISABLED,
    MEASURED_CACHE_DEFICIT_REASON,
    AbortSettings,
    decide_measured_cache_abort,
)
from proxy.prompt_progress import PromptProgress

from proxy import measured_cache_abort, prefill_eta

#: Remove this line in LP-0MVWDZ1GG004UGIK (the gate implementation) — every row
#: in this module must then pass outright.
pytestmark = pytest.mark.xfail(
    strict=True,
    reason="benefit gate lands in LP-0MVWDZ1GG004UGIK",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "proxy" / "config.yaml"

#: The model the incident pane was dispatched to (``routing_check
#: provider=local-qwen3``), i.e. the calibration key the gate must price with.
MODELLED_MODEL = "local-qwen3"

#: The incident prompt: 14 380 tokens of real cold prefill, admitted on a priced
#: 48 new tokens (the optimism that caused the incident).
INCIDENT_TOTAL = 14_380
INCIDENT_PRICED = 48

#: The population the feature exists for: a huge cold prompt priced as warm.
HUGE_TOTAL = 56_000
HUGE_PRICED = 100

#: Shipped comparator inputs (``proxy/config.yaml``) and their product.
SHIPPED_DEADLINE_S = 120.0
SHIPPED_SAFETY_FRACTION = 0.95
SHIPPED_BUDGET_S = 114.0

#: The incident's recorded admission evidence (``local_admission_guard`` line):
#: ``eta_prefill_s=8.2 budget_s=60.0 deadline_s=120.0 new_tokens=49``. Recorded
#: here so the wait-budget row can name exactly the evidence a naive wiring
#: would have handed to the gate; the decision takes none of it as an input.
INCIDENT_ADMISSION_EVIDENCE = {
    "eta_prefill_s": 8.2,
    "budget_s": 60.0,
    "deadline_s": 120.0,
    "new_tokens": 49,
}

#: Measured host rates (2026-10-31, 224 samples) and the shipped default fit.
HOST_MEDIAN_TOKENS_PER_S = 371.0
HOST_P25_TOKENS_PER_S = 170.0
DEFAULT_PREFILL_MS_PER_TOKEN = 3.9

#: A spread calibration window for the 56 000-token bucket: the fitted p75 is
#: 135 000 ms, so the conservative uplift prices the prompt above the point
#: estimate (``1.2 x mean(108 333 ms) = 130 000 ms``) and well above 114 s.
SEED_WINDOW_MS = (60_000.0, 70_000.0, 80_000.0, 90_000.0, 150_000.0, 200_000.0)

_DROP = object()
_SHIPPED_SERVER = (yaml.safe_load(CONFIG_PATH.read_text()) or {})["server"]


@pytest.fixture(autouse=True)
def _clean_calibration():
    """Every row starts from the uncalibrated shipped defaults."""
    prefill_eta.reset()
    yield
    prefill_eta.reset()


def _shipped_config(**overrides) -> dict:
    """The shipped ``server`` config with *overrides* applied.

    ``_DROP`` removes a key entirely, so "absent" can be distinguished from
    "present but zero".
    """
    server = dict(_SHIPPED_SERVER)
    for key, value in overrides.items():
        if value is _DROP:
            server.pop(key, None)
        else:
            server[key] = value
    return {"server": server}


def _settings(config: dict | None = None, **overrides) -> AbortSettings:
    """Settings resolved from *config* (shipped by default) for the incident model."""
    resolved = measured_cache_abort.AbortSettings.from_config(
        _shipped_config() if config is None else config
    )
    return dataclasses.replace(resolved, model=MODELLED_MODEL, **overrides)


def _progress(total: int, cache: int) -> PromptProgress:
    """A b8782 ``prompt_progress`` measurement."""
    return PromptProgress(total=total, cache=cache, processed=0, time_ms=0.0)


def _within_eta_budget() -> str:
    """The new continue token, read from the module under test."""
    return measured_cache_abort.DETAIL_WITHIN_ETA_BUDGET


def _seed_huge_prompt_window() -> None:
    """Seed a spread 56 000-token window so conservative pricing engages."""
    for service_ms in SEED_WINDOW_MS:
        prefill_eta.record_dispatch_sample(
            MODELLED_MODEL, HUGE_TOTAL, 0.0, service_ms
        )


def _eta_at_host_rate(tokens: int, tokens_per_s: float) -> float:
    """The measured cold-prefill ETA for *tokens* at a measured host rate."""
    return tokens / tokens_per_s


# ---------------------------------------------------------------------------
# AC1 — the incident row continues (parent AC1(a), D2 -> abort=False)
# ---------------------------------------------------------------------------


def test_incident_prompt_continues_within_the_eta_budget() -> None:
    """14 380 cold tokens priced at 48 must continue, not be cancelled.

    Rate basis (see module docstring): the shipped default fit prices the
    measured deficit at 14 380 x 3.9 ms = 56.08 s — 38.8 s at the host median
    371 tok/s and 84.6 s at the p25 170 tok/s — against the 114 s comparator
    (0.95 x 120 s). Every measured rate therefore fits, with >= 29 s of margin.
    """
    decision = decide_measured_cache_abort(
        _progress(INCIDENT_TOTAL, 0), INCIDENT_PRICED, _settings()
    )

    assert decision.abort is False
    assert decision.continue_verdict is True
    assert decision.reason is None
    assert decision.detail == _within_eta_budget()
    assert decision.measured_new_tokens == INCIDENT_TOTAL
    assert decision.deficit_ratio == pytest.approx(1.0)
    assert decision.measured_eta_s == pytest.approx(56.1, abs=0.5)
    assert decision.budget_s == pytest.approx(SHIPPED_BUDGET_S)

    # The same row still continues at the slowest measured host rate, which is
    # what makes `abort=False` robust to a future recalibration (parent AC1a).
    median_eta = _eta_at_host_rate(INCIDENT_TOTAL, HOST_MEDIAN_TOKENS_PER_S)
    p25_eta = _eta_at_host_rate(INCIDENT_TOTAL, HOST_P25_TOKENS_PER_S)
    assert median_eta == pytest.approx(38.8, abs=0.1)
    assert p25_eta == pytest.approx(84.6, abs=0.1)
    assert p25_eta < decision.budget_s, (
        "incident row must continue at the p25 measured rate "
        f"(p25 ETA {p25_eta:.1f} s vs budget {decision.budget_s:.1f} s)"
    )


# ---------------------------------------------------------------------------
# AC2 — the huge-prompt row still aborts (parent AC1(b))
# ---------------------------------------------------------------------------


def test_huge_prompt_aborts_on_the_shipped_default_fit() -> None:
    """56 000 cold tokens priced at 100 still abort (218.4 s > 114 s).

    Rate basis: uncalibrated default fit, 56 000 x 3.9 ms = 218.4 s; the
    measured median rate gives 150.9 s. Both exceed the 114 s comparator, so
    the abort keeps today's behaviour for the population it exists for.
    """
    decision = decide_measured_cache_abort(
        _progress(HUGE_TOTAL, 0), HUGE_PRICED, _settings()
    )

    assert decision.abort is True
    assert decision.reason == MEASURED_CACHE_DEFICIT_REASON
    assert decision.detail == DETAIL_ABORT
    assert decision.measured_new_tokens == HUGE_TOTAL
    assert decision.measured_eta_s == pytest.approx(218.4, abs=0.5)
    assert decision.budget_s == pytest.approx(SHIPPED_BUDGET_S)
    assert decision.measured_eta_s > decision.budget_s

    default_fit_eta = HUGE_TOTAL * DEFAULT_PREFILL_MS_PER_TOKEN / 1000.0
    median_eta = _eta_at_host_rate(HUGE_TOTAL, HOST_MEDIAN_TOKENS_PER_S)
    assert default_fit_eta == pytest.approx(218.4)
    assert median_eta == pytest.approx(150.9, abs=0.1)
    assert median_eta > decision.budget_s


def test_huge_prompt_aborts_with_the_conservative_p75_uplift() -> None:
    """With a calibrated window the p75 price (135 s) still exceeds 114 s.

    Determinism: a synthetic spread window for the 56 000-token bucket
    (``_seed_huge_prompt_window``); the uplift is what makes the row exercise
    the conservative path rather than the default fit.
    """
    _seed_huge_prompt_window()
    measured_eta_s, coefficients = prefill_eta.estimate_prefill_seconds(
        MODELLED_MODEL, HUGE_TOTAL, 0.0, _shipped_config()
    )
    assert coefficients.calibrated is True
    assert coefficients.uplift > 1.0, "row must exercise the p75 uplift"
    assert measured_eta_s == pytest.approx(135.0, abs=0.5)

    decision = decide_measured_cache_abort(
        _progress(HUGE_TOTAL, 0), HUGE_PRICED, _settings()
    )

    assert decision.abort is True
    assert decision.detail == DETAIL_ABORT
    assert decision.measured_eta_s == pytest.approx(measured_eta_s)
    assert decision.measured_eta_s > decision.budget_s == pytest.approx(
        SHIPPED_BUDGET_S
    )


# ---------------------------------------------------------------------------
# AC3 — the comparator is the prefill-deadline budget (D1(a))
# ---------------------------------------------------------------------------


def test_a_zero_safety_fraction_falls_back_to_the_deadline() -> None:
    """``budget_safety_fraction: 0`` => the comparator is the 120 s deadline."""
    config = _shipped_config(local_admission_guard_budget_safety_fraction=0)
    settings = _settings(config)

    assert prefill_eta.prefill_deadline_budget_seconds(config) == pytest.approx(
        SHIPPED_DEADLINE_S
    )
    decision = decide_measured_cache_abort(
        _progress(INCIDENT_TOTAL, 0), INCIDENT_PRICED, settings
    )

    assert decision.budget_s == pytest.approx(SHIPPED_DEADLINE_S)
    assert decision.abort is False
    assert decision.detail == _within_eta_budget()


def test_a_low_deadline_makes_the_incident_prompt_abort() -> None:
    """The comparator is deadline-derived: 40 s => 38 s, so 56.1 s aborts."""
    config = _shipped_config(local_prefill_deadline_seconds=40)
    decision = decide_measured_cache_abort(
        _progress(INCIDENT_TOTAL, 0), INCIDENT_PRICED, _settings(config)
    )

    assert decision.budget_s == pytest.approx(38.0)
    assert decision.measured_eta_s == pytest.approx(56.1, abs=0.5)
    assert decision.abort is True
    assert decision.detail == DETAIL_ABORT


@pytest.mark.parametrize("wait_budget", [1, 30, 55])
def test_the_admission_wait_budget_never_decides_the_verdict(wait_budget: int) -> None:
    """``contention_queue_max_wait_seconds`` is not the comparator (parent AC1).

    The incident ETA (56.1 s) exceeds each of these wait budgets, and the
    incident's recorded admission evidence carries ``budget_s=60.0`` /
    ``deadline_s=120.0`` / ``new_tokens=49`` — yet the row continues, because
    the decision's only inputs are the measurement, the priced value and the
    deadline-derived budget.
    """
    assert INCIDENT_ADMISSION_EVIDENCE["new_tokens"] != INCIDENT_PRICED
    config = _shipped_config(contention_queue_max_wait_seconds=wait_budget)
    decision = decide_measured_cache_abort(
        _progress(INCIDENT_TOTAL, 0), INCIDENT_PRICED, _settings(config)
    )

    assert decision.budget_s == pytest.approx(SHIPPED_BUDGET_S)
    assert decision.budget_s != pytest.approx(float(wait_budget))
    assert decision.measured_eta_s == pytest.approx(56.1, abs=0.5)
    assert decision.measured_eta_s > wait_budget, (
        "row must be one the wait budget would abort if it were the comparator"
    )
    assert decision.abort is False, "the wait budget must not decide"
    assert decision.detail == _within_eta_budget()


def test_the_comparator_helper_is_the_enforced_threshold() -> None:
    """The advertised threshold and the enforced one are single-sourced."""
    config = _shipped_config()
    expected = prefill_eta.prefill_deadline_budget_seconds(config)
    decision = decide_measured_cache_abort(
        _progress(INCIDENT_TOTAL, 0), INCIDENT_PRICED, _settings(config)
    )

    assert expected == pytest.approx(SHIPPED_BUDGET_S)
    assert decision.budget_s == pytest.approx(expected)


def test_an_eta_equal_to_the_budget_continues() -> None:
    """The gate aborts only when the ETA *exceeds* the budget."""
    at_budget = decide_measured_cache_abort(
        _progress(INCIDENT_TOTAL, 0),
        INCIDENT_PRICED,
        _settings(eta_estimator=lambda _tokens: SHIPPED_BUDGET_S),
    )
    over_budget = decide_measured_cache_abort(
        _progress(INCIDENT_TOTAL, 0),
        INCIDENT_PRICED,
        _settings(eta_estimator=lambda _tokens: SHIPPED_BUDGET_S + 0.001),
    )

    assert at_budget.budget_s == pytest.approx(SHIPPED_BUDGET_S)
    assert at_budget.abort is False
    assert at_budget.detail == _within_eta_budget()
    assert over_budget.abort is True
    assert over_budget.detail == DETAIL_ABORT


# ---------------------------------------------------------------------------
# AC4 — gate input provenance (parent AC10)
# ---------------------------------------------------------------------------


def test_the_gate_prices_the_measured_deficit_not_the_price() -> None:
    """``measured_eta_s`` is priced from 14 380 tokens, never from the priced 48.

    An ETA of ~8 s (the admission-priced optimism) or ~0.19 s (48 tokens at the
    default fit) would mean the gate is pricing the price — the exact inversion
    that caused the incident.
    """
    config = _shipped_config()
    decision = decide_measured_cache_abort(
        _progress(INCIDENT_TOTAL, 0), INCIDENT_PRICED, _settings(config)
    )

    priced_eta_s, _coefficients = prefill_eta.estimate_prefill_seconds(
        MODELLED_MODEL, INCIDENT_PRICED, 0.0, config
    )
    assert priced_eta_s == pytest.approx(0.1872, abs=0.001)
    assert decision.measured_eta_s is not None
    assert decision.measured_eta_s >= 30.0, (
        "the measured-rate floor: a 14 380-token cold prefill cannot be ~8 s"
    )
    assert decision.measured_eta_s > 100 * priced_eta_s


def test_the_gate_prices_the_deficit_uncached() -> None:
    """The deficit is by definition uncached: ``cached_ratio=0.0`` (impl AC4)."""
    config = _shipped_config()
    decision = decide_measured_cache_abort(
        _progress(INCIDENT_TOTAL, 0), INCIDENT_PRICED, _settings(config)
    )

    cold_eta_s, _ = prefill_eta.estimate_prefill_seconds(
        MODELLED_MODEL, INCIDENT_TOTAL, 0.0, config
    )
    warm_eta_s, _ = prefill_eta.estimate_prefill_seconds(
        MODELLED_MODEL, INCIDENT_TOTAL, 0.9, config
    )
    assert cold_eta_s == pytest.approx(56.082, abs=0.01)
    assert warm_eta_s == pytest.approx(12.0792, abs=0.01)
    assert decision.measured_eta_s == pytest.approx(cold_eta_s)


def test_the_gate_prices_with_the_calibrated_model_and_config() -> None:
    """The gate uses the admission guard's own calibrated call (D3(a)).

    Seeded 56 000-token window: the shipped config prices at the p75 (135 s),
    ``local_admission_guard_eta_percentile: 0`` rolls back to the point fit
    (130 s), and a missing model would ignore the window entirely (218.4 s).
    All three exceed the 114 s comparator, so the abort stands either way — the
    ETA is what must move.
    """
    _seed_huge_prompt_window()
    shipped = decide_measured_cache_abort(
        _progress(HUGE_TOTAL, 0), HUGE_PRICED, _settings()
    )
    rolled_back = decide_measured_cache_abort(
        _progress(HUGE_TOTAL, 0),
        HUGE_PRICED,
        _settings(_shipped_config(local_admission_guard_eta_percentile=0)),
    )

    assert shipped.measured_eta_s == pytest.approx(135.0, abs=0.5)
    assert rolled_back.measured_eta_s == pytest.approx(130.0, abs=0.5)
    assert shipped.abort is rolled_back.abort is True


def test_the_estimator_receives_the_measured_deficit_tokens() -> None:
    """A recording estimator sees ``total - cache`` once, never the price."""
    seen_tokens: list[int] = []

    def estimator(tokens: int) -> float:
        seen_tokens.append(tokens)
        return 500.0

    decision = decide_measured_cache_abort(
        _progress(INCIDENT_TOTAL, 0),
        INCIDENT_PRICED,
        _settings(eta_estimator=estimator),
    )

    assert seen_tokens == [INCIDENT_TOTAL]
    assert decision.measured_eta_s == pytest.approx(500.0)
    assert decision.budget_s == pytest.approx(SHIPPED_BUDGET_S)
    assert decision.abort is True, "the injected estimator decided the verdict"


def test_feeding_the_price_as_the_measurement_changes_the_verdict() -> None:
    """Companion to the provenance row: the gate's *input* decides.

    With a 19 s comparator (``local_prefill_deadline_seconds: 20``) and the
    token bound disabled, the two calls differ only in the measurement: the
    real 14 380-token deficit aborts (56.1 s > 19 s) while the priced 48 tokens
    fed as if measured continues (0.19 s < 19 s). Feeding the price through the
    gate therefore changes both the ETA and the verdict — the optimism inversion
    cannot be reintroduced through the gate's input.
    """
    config = _shipped_config(local_prefill_deadline_seconds=20)
    settings = _settings(config, bound_tokens=0)

    measured = decide_measured_cache_abort(
        _progress(INCIDENT_TOTAL, 0), 0, settings
    )
    priced_as_measured = decide_measured_cache_abort(
        _progress(INCIDENT_PRICED, 0), 0, settings
    )

    assert measured.budget_s == pytest.approx(19.0)
    assert priced_as_measured.budget_s == pytest.approx(19.0)

    assert measured.measured_eta_s == pytest.approx(56.1, abs=0.5)
    assert priced_as_measured.measured_eta_s == pytest.approx(0.1872, abs=0.001)
    assert measured.abort is True
    assert priced_as_measured.abort is False
    assert measured.detail == DETAIL_ABORT
    assert priced_as_measured.detail == _within_eta_budget()


def test_the_continue_token_is_its_own_vocabulary() -> None:
    """One new ``detail`` token, distinct from the existing ones (impl AC1b)."""
    assert measured_cache_abort.DETAIL_WITHIN_ETA_BUDGET == "within_eta_budget"
    assert measured_cache_abort.DETAIL_WITHIN_ETA_BUDGET not in {
        DETAIL_ABORT,
        DETAIL_DISABLED,
        measured_cache_abort.DETAIL_WITHIN_BOUND,
        measured_cache_abort.DETAIL_BELOW_RATIO_FLOOR,
    }


# ---------------------------------------------------------------------------
# AC5 — fail open on a missing budget (parent AC6)
# ---------------------------------------------------------------------------

#: Budget configurations that resolve to no usable comparator. The huge-prompt
#: row is used as the probe: a comparator of 0 treated as binding (``eta > 0``)
#: would abort it, so "continue" is the fail-open assertion, not a coincidence.
UNUSABLE_BUDGETS = [
    pytest.param(
        {
            "local_prefill_deadline_seconds": _DROP,
            "local_admission_guard_budget_safety_fraction": _DROP,
        },
        id="budget-keys-absent",
    ),
    pytest.param(
        {"local_prefill_deadline_seconds": _DROP}, id="deadline-absent"
    ),
    pytest.param(
        {"local_admission_guard_budget_safety_fraction": _DROP},
        id="safety-fraction-absent",
    ),
    pytest.param(
        {"local_prefill_deadline_seconds": 0}, id="deadline-explicit-zero"
    ),
    pytest.param(
        {"local_prefill_deadline_seconds": -60}, id="deadline-negative"
    ),
    pytest.param(
        {"local_prefill_deadline_seconds": "banana"}, id="deadline-non-numeric"
    ),
    pytest.param(
        {"local_prefill_deadline_seconds": float("nan")}, id="deadline-nan"
    ),
    pytest.param(
        {"local_admission_guard_budget_safety_fraction": -0.5},
        id="safety-fraction-negative",
    ),
    pytest.param(
        {"local_admission_guard_budget_safety_fraction": "banana"},
        id="safety-fraction-non-numeric",
    ),
    pytest.param(
        {"local_admission_guard_budget_safety_fraction": float("nan")},
        id="safety-fraction-nan",
    ),
]


@pytest.mark.parametrize("overrides", UNUSABLE_BUDGETS)
def test_a_missing_budget_never_manufactures_an_abort(overrides: dict) -> None:
    """An unresolvable budget resolves to continue, never to ``deficit_over_bound``."""
    decision = decide_measured_cache_abort(
        _progress(HUGE_TOTAL, 0), HUGE_PRICED, _settings(_shipped_config(**overrides))
    )

    assert decision.abort is False
    assert decision.reason is None
    assert decision.detail != DETAIL_ABORT
    assert decision.budget_s in (None, 0.0)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param("server: {}", id="config-is-a-string"),
        pytest.param({"server": ["not", "a", "mapping"]}, id="server-not-a-mapping"),
    ],
)
def test_an_unreadable_config_fails_open(config) -> None:
    """A config we cannot read disables the feature (never over-aborts)."""
    decision = decide_measured_cache_abort(
        _progress(HUGE_TOTAL, 0), HUGE_PRICED, _settings(config)
    )

    assert decision.abort is False
    assert decision.detail == DETAIL_DISABLED
    assert decision.budget_s in (None, 0.0)


def _raising_estimator(_tokens: int) -> float:
    raise RuntimeError("calibration unavailable")


@pytest.mark.parametrize(
    "estimator",
    [
        pytest.param(_raising_estimator, id="estimator-raises"),
        pytest.param(lambda _tokens: None, id="estimator-returns-none"),
        pytest.param(lambda _tokens: "x", id="estimator-returns-string"),
        pytest.param(lambda _tokens: float("nan"), id="estimator-returns-nan"),
        pytest.param(lambda _tokens: float("inf"), id="estimator-returns-inf"),
        pytest.param(object(), id="estimator-not-callable"),
    ],
)
def test_an_unusable_eta_estimator_fails_open(estimator) -> None:
    """An ETA we cannot price never releases a dispatch (never raises)."""
    decision = decide_measured_cache_abort(
        _progress(HUGE_TOTAL, 0),
        HUGE_PRICED,
        _settings(eta_estimator=estimator),
    )

    assert decision.abort is False
    assert decision.detail != DETAIL_ABORT
    assert decision.budget_s == pytest.approx(SHIPPED_BUDGET_S)
    assert decision.measured_eta_s is None


# ---------------------------------------------------------------------------
# AC6 — pre-gate rows unchanged: the gate is inert without a budget
# ---------------------------------------------------------------------------


def test_the_gate_is_inert_when_no_budget_is_configured() -> None:
    """``AbortSettings()`` (no config) keeps today's token-bound verdict.

    This is what keeps every pre-existing decision row byte-for-byte: they build
    ``AbortSettings`` by hand (no ``config``), so no budget is resolved and the
    benefit gate must not touch their verdict — notably
    ``test_bound_boundary_is_inclusive_for_continue``, whose 5 001-token abort
    would otherwise be declined by the 114 s comparator.
    """
    decision = decide_measured_cache_abort(
        _progress(HUGE_TOTAL, 0), HUGE_PRICED, AbortSettings()
    )

    assert decision.abort is True
    assert decision.detail == DETAIL_ABORT
    assert decision.budget_s is None
    assert decision.measured_eta_s is None
