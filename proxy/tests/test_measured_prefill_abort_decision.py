"""Option D early-abort decision pinned as a pure function (LP-0MVR7HJD0007RFEM).

Parent: LP-0MVQIT9QY007QU27 (``local_eviction_after_acceptance
reason=prefill_deadline``). The producer approved Option D — release a local
dispatch early when the *measured* prefill progress shows it is re-prefilling
far more than admission priced — because no admission-time predicate can tell
a genuinely warm prompt from a clobbered one (all ``routing_resident_prefix``
decisions carry ``n_past_actual=none``).

The signal is the b8782 ``prompt_progress`` event (contract pinned by
:mod:`proxy.prompt_progress`, F1): the first event carries
``cache = n_prompt_tokens_cache`` (the exact reused prefix) against
``total`` (the full prompt), so ``total - cache`` is the work actually being
computed. A warm dispatch shows ``cache ≈ 0.999 × total``; a fake-warm one
shows a tiny ``cache`` against a large ``total``.

These tests pin the decision (F3) before it is wired into the dispatch loop
(F4, LP-0MVR7IN0O000DPXN), so the warm path the parent measured — 252 local
prefills ≥ 20k tokens, 177 of them with first byte < 10 s — cannot regress.

AC mapping:
  AC1 fake-warm aborts (56k prompt, tiny cache, priced 150);
  AC2 warm never aborts, including the <10 s first-byte large-prompt population;
  AC3 small/cheap prompts unchanged;
  AC4 rollback: disabled config restores the pre-change verdict;
  AC5 fail open on missing/invalid measurement, bad pricing, or an exception;
  AC6 documented bound semantics + monotonicity.
"""

from __future__ import annotations

import json
import random

import pytest
from proxy.measured_cache_abort import (
    BOUND_TOKENS_KEY,
    DETAIL_ABORT,
    DETAIL_BELOW_RATIO_FLOOR,
    DETAIL_DISABLED,
    DETAIL_ERROR,
    DETAIL_INVALID_MEASUREMENT,
    DETAIL_INVALID_PRICED,
    DETAIL_NO_MEASUREMENT,
    DETAIL_WITHIN_BOUND,
    ENABLED_KEY,
    MEASURED_CACHE_DEFICIT_REASON,
    MIN_DEFICIT_RATIO_KEY,
    AbortSettings,
    decide_measured_cache_abort,
)
from proxy.prompt_progress import PromptProgress, parse_prompt_progress_event

#: The b8782 event shape verified live (F1): ``prompt_progress`` is a
#: top-level key next to ``choices``, never nested inside a delta.
B8782_FAKE_WARM_EVENT = (
    b'data: {"choices":[{"delta":{"role":"assistant"}}],'
    b'"prompt_progress":{"total":56000,"cache":200,"processed":0,"time_ms":0}}\n\n'
)

#: The fake-warm signature from the measured window: a ~56k prompt admitted on
#: a priced 17–182 new tokens, cancelled at the 120 s deadline.
FAKE_WARM_TOTAL = 56_000
FAKE_WARM_CACHE = 200
FAKE_WARM_PRICED = 150

#: Working large-prompt dispatches from the measured window (first byte < 10 s):
#: ``cache ≈ 0.999 × total``, so the priced compute equals the real work. The
#: population is what makes a naive "big prompt ⇒ abort" heuristic unsafe.
WARM_LARGE_PROMPT_POPULATION = (
    (20_480, 20_460, 20, 1.1),
    (30_000, 29_970, 30, 2.6),
    (40_000, 39_960, 40, 3.4),
    (56_000, 55_944, 56, 6.8),
    (60_000, 59_940, 60, 9.4),
    (86_000, 85_914, 86, 8.5),
    (127_872, 127_744, 128, 9.9),
)


def _progress(
    total: int | float | None,
    cache: int | float | None,
    processed: int | None = 0,
    time_ms: float | None = 0.0,
) -> PromptProgress:
    """Build a measurement with the b8782 field names."""
    return PromptProgress(total=total, cache=cache, processed=processed, time_ms=time_ms)


def _settings(**overrides) -> AbortSettings:
    """Default settings (enabled, bound 2 000, ratio floor 0.25) with overrides."""
    values: dict[str, object] = {
        "enabled": True,
        "bound_tokens": 2_000,
        "min_deficit_ratio": 0.25,
    }
    values.update(overrides)
    return AbortSettings(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AC1 — fake-warm aborts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bound", [0, 100, 1_000, 2_000, 5_000, 20_000, 50_000])
def test_fake_warm_aborts_at_any_sensible_bound(bound: float) -> None:
    """~55 800 measured new tokens against a priced 150 aborts at any sane bound."""
    decision = decide_measured_cache_abort(
        _progress(FAKE_WARM_TOTAL, FAKE_WARM_CACHE),
        FAKE_WARM_PRICED,
        _settings(bound_tokens=bound),
    )

    assert decision.abort is True
    assert decision.reason == MEASURED_CACHE_DEFICIT_REASON
    assert decision.detail == DETAIL_ABORT
    assert decision.measured_new_tokens == 55_800
    assert decision.deficit_ratio == pytest.approx(55_800 / 56_000)
    assert decision.priced_new_tokens == 150
    assert decision.bound_tokens == bound
    assert decision.continue_verdict is False


def test_fake_warm_aborts_from_a_real_b8782_event() -> None:
    """The measured event shape alone carries enough ground truth to abort."""
    progress = parse_prompt_progress_event(B8782_FAKE_WARM_EVENT)

    assert progress is not None
    assert progress.measured_new_tokens == 55_800
    decision = decide_measured_cache_abort(progress, FAKE_WARM_PRICED, _settings())

    assert decision.abort is True
    assert decision.reason == MEASURED_CACHE_DEFICIT_REASON


def test_decision_accepts_the_measurement_as_mapping_or_object() -> None:
    """F4's stream loop may hand over a dict; both shapes decide identically."""
    from_object = decide_measured_cache_abort(
        _progress(FAKE_WARM_TOTAL, FAKE_WARM_CACHE), FAKE_WARM_PRICED, _settings()
    )
    from_mapping = decide_measured_cache_abort(
        {"total": FAKE_WARM_TOTAL, "cache": FAKE_WARM_CACHE},
        FAKE_WARM_PRICED,
        _settings(),
    )

    assert from_mapping == from_object
    assert from_mapping.abort is True


# ---------------------------------------------------------------------------
# AC2 — warm never aborts, at any prompt size or bound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "total,cache,priced,first_byte_s", WARM_LARGE_PROMPT_POPULATION
)
@pytest.mark.parametrize("bound", [0, 2_000, 100_000])
@pytest.mark.parametrize("ratio_floor", [0.0, 0.25])
def test_warm_large_prompt_never_aborts(
    total: int,
    cache: int,
    priced: float,
    first_byte_s: float,
    bound: float,
    ratio_floor: float,
) -> None:
    """``cache ≈ 0.999 × total`` continues even when the ratio floor is 0.

    With ``bound=0`` the comparison degenerates to ``measured_new <= priced``,
    which is the exact boundary a genuinely warm dispatch sits on (the priced
    compute *is* the real work) — so the bound is inclusive and the warm path
    is unreachable for any prompt size.
    """
    progress = _progress(total, cache)
    assert progress.measured_cached_ratio is not None
    assert progress.measured_cached_ratio >= 0.99, "population must be warm"
    assert first_byte_s < 10.0, "population is the sub-10 s first-byte cohort"

    decision = decide_measured_cache_abort(
        progress, priced, _settings(bound_tokens=bound, min_deficit_ratio=ratio_floor)
    )

    assert decision.abort is False
    assert decision.continue_verdict is True
    assert decision.reason is None
    assert decision.measured_new_tokens == total - cache


def test_warm_ratio_floor_alone_blocks_a_warm_prompt_at_zero_slack() -> None:
    """The ratio floor makes the warm path safe even if the price is wrong.

    A warm dispatch whose priced compute is wildly under-estimated (a *negative*
    slack of 2 000 tokens) still continues, because a 0.1 % deficit is nowhere
    near the fake-warm signature.
    """
    decision = decide_measured_cache_abort(
        _progress(200_000, 199_800), 0, _settings(bound_tokens=0)
    )

    assert decision.abort is False
    assert decision.detail == DETAIL_BELOW_RATIO_FLOOR
    assert decision.measured_new_tokens == 200


# ---------------------------------------------------------------------------
# AC3 — small/cheap prompts unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "total,cache,priced,bound",
    [
        (1_200, 0, 1_200, 2_000),  # small cold prompt: deficit within bound
        (800, 0, 820, 2_000),
        (5_000, 3_500, 900, 2_000),  # small partial reuse, over the ratio floor
        (30_000, 10_000, 20_000, 2_000),  # deficit exactly at the bound
    ],
)
def test_small_or_cheap_prompts_continue(
    total: int, cache: int, priced: float, bound: float
) -> None:
    """``total - cache <= priced + bound`` is always a continue."""
    decision = decide_measured_cache_abort(
        _progress(total, cache), priced, _settings(bound_tokens=bound)
    )

    assert decision.abort is False
    assert decision.continue_verdict is True
    assert decision.detail == DETAIL_WITHIN_BOUND


def test_bound_boundary_is_inclusive_for_continue() -> None:
    """Exactly ``priced + bound`` continues; one token more aborts."""
    settings = _settings(bound_tokens=1_000)

    at_bound = decide_measured_cache_abort(_progress(10_000, 5_000), 4_000, settings)
    assert at_bound.abort is False
    assert at_bound.measured_new_tokens == 5_000 == 4_000 + 1_000
    assert at_bound.detail == DETAIL_WITHIN_BOUND

    over_bound = decide_measured_cache_abort(_progress(10_000, 4_999), 4_000, settings)
    assert over_bound.abort is True
    assert over_bound.measured_new_tokens == 5_001


# ---------------------------------------------------------------------------
# AC4 — rollback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "disabled_value", [0, False, "0", "false", "off", "OFF", "disabled", ""]
)
def test_disabled_config_restores_the_pre_change_verdict(disabled_value) -> None:
    """Every documented "off" spelling resolves to enabled=False."""
    settings = AbortSettings.from_config({"server": {ENABLED_KEY: disabled_value}})
    assert settings.enabled is False

    decision = decide_measured_cache_abort(
        _progress(FAKE_WARM_TOTAL, FAKE_WARM_CACHE), FAKE_WARM_PRICED, settings
    )

    assert decision.abort is False
    assert decision.reason is None
    assert decision.detail == DETAIL_DISABLED


def test_rollback_continues_for_every_measurement_shape() -> None:
    """Disabled means no new verdict exists at all — the old path, verbatim."""
    settings = AbortSettings.from_config({"server": {ENABLED_KEY: 0}})

    for total, cache in [
        (FAKE_WARM_TOTAL, FAKE_WARM_CACHE),  # the defect itself
        (40_000, 39_960),  # warm
        (1_200, 0),  # small cold
        (0, 0),  # degenerate
        (56_000, 10**9),  # nonsensical
    ]:
        decision = decide_measured_cache_abort(_progress(total, cache), 150, settings)
        assert decision.abort is False
        assert decision.reason is None
        assert decision.detail == DETAIL_DISABLED
        assert decision.measured_new_tokens is None


# ---------------------------------------------------------------------------
# AC5 — fail open
# ---------------------------------------------------------------------------


class _ExplodingProgress:
    """A measurement whose field access raises — capture must stay fail-open."""

    cache = 0

    @property
    def total(self):  # pragma: no cover - the raise is the point
        raise RuntimeError("measurement unavailable")


@pytest.mark.parametrize(
    "progress,expected_detail",
    [
        (None, DETAIL_NO_MEASUREMENT),
        (_progress(None, 200), DETAIL_NO_MEASUREMENT),
        (_progress(56_000, None), DETAIL_NO_MEASUREMENT),
        ({"total": 56_000}, DETAIL_NO_MEASUREMENT),  # mapping missing ``cache``
        (_progress("56000", "200"), DETAIL_NO_MEASUREMENT),  # strings
        (_progress(56_000.5, 200), DETAIL_NO_MEASUREMENT),  # fractional token
        (_progress(float("nan"), 200), DETAIL_NO_MEASUREMENT),
        (_progress(float("inf"), 200), DETAIL_NO_MEASUREMENT),
        (_progress(True, 0), DETAIL_NO_MEASUREMENT),
        (object(), DETAIL_NO_MEASUREMENT),  # no measurement fields at all
        (_progress(0, 0), DETAIL_INVALID_MEASUREMENT),
        (_progress(56_000, 60_000), DETAIL_INVALID_MEASUREMENT),  # cache > total
        (_progress(56_000, -1), DETAIL_INVALID_MEASUREMENT),
        (_progress(-56_000, 0), DETAIL_INVALID_MEASUREMENT),
        (_ExplodingProgress(), DETAIL_ERROR),
    ],
)
def test_missing_or_invalid_measurement_continues(progress, expected_detail) -> None:
    """No usable measurement can never release a dispatch."""
    decision = decide_measured_cache_abort(progress, FAKE_WARM_PRICED, _settings())

    assert decision.abort is False
    assert decision.reason is None
    assert decision.detail == expected_detail
    assert decision.continue_verdict is True


@pytest.mark.parametrize(
    "priced", [None, "150", -1, float("nan"), float("inf"), True, [150], {}]
)
def test_invalid_priced_new_tokens_continues(priced) -> None:
    """An unusable admission price fails open rather than guessing."""
    decision = decide_measured_cache_abort(
        _progress(FAKE_WARM_TOTAL, FAKE_WARM_CACHE), priced, _settings()
    )

    assert decision.abort is False
    assert decision.detail == DETAIL_INVALID_PRICED


@pytest.mark.parametrize("priced", [150, 150.4, 150.6])
def test_fractional_priced_new_tokens_is_accepted(priced: float) -> None:
    """Admission prices are floats (est_tokens × (1 − ratio)); accept them."""
    decision = decide_measured_cache_abort(
        _progress(FAKE_WARM_TOTAL, FAKE_WARM_CACHE), priced, _settings()
    )

    assert decision.abort is True
    assert decision.priced_new_tokens == pytest.approx(priced)


def test_unusable_settings_object_continues() -> None:
    """A wiring mistake in the settings object must not abort a dispatch."""
    decision = decide_measured_cache_abort(
        _progress(FAKE_WARM_TOTAL, FAKE_WARM_CACHE), FAKE_WARM_PRICED, object()
    )

    assert decision.abort is False
    assert decision.detail == DETAIL_ERROR


# ---------------------------------------------------------------------------
# AC6 — bound semantics and monotonicity
# ---------------------------------------------------------------------------


def test_defaults_are_documented_and_safe_for_the_warm_path() -> None:
    """Defaults: enabled, 2 000 token slack, 0.25 deficit-ratio floor."""
    defaults = AbortSettings()
    assert defaults.enabled is True
    assert defaults.bound_tokens == 2_000
    assert defaults.min_deficit_ratio == 0.25

    # The default bound/floor catch the defect ...
    assert decide_measured_cache_abort(
        _progress(FAKE_WARM_TOTAL, FAKE_WARM_CACHE), FAKE_WARM_PRICED
    ).abort is True
    # ... and cannot reach the warm population.
    for total, cache, priced, _first_byte_s in WARM_LARGE_PROMPT_POPULATION:
        assert decide_measured_cache_abort(_progress(total, cache), priced).abort is False


def _random_measurements(seed: int = 20_261_027, count: int = 300):
    """Deterministic mixed population spanning warm, cold, and fake-warm."""
    rng = random.Random(seed)
    for _ in range(count):
        total = rng.randint(1, 120_000)
        cache = rng.randint(0, total)
        priced = rng.choice(
            [0, float(rng.randint(0, 40_000)), rng.random() * 5_000]
        )
        yield total, cache, priced


def test_increasing_the_token_bound_only_ever_blocks_aborts() -> None:
    """Monotone: a larger slack can never turn a continue into an abort."""
    bounds = [0, 1, 250, 2_000, 10_000, 50_000, 200_000]

    for total, cache, priced in _random_measurements():
        verdicts = [
            decide_measured_cache_abort(
                _progress(total, cache), priced, _settings(bound_tokens=bound)
            ).abort
            for bound in bounds
        ]
        assert verdicts == sorted(verdicts, reverse=True), (
            total,
            cache,
            priced,
            verdicts,
        )


def test_raising_the_deficit_ratio_floor_only_ever_blocks_aborts() -> None:
    """Monotone: a higher ratio floor can never turn a continue into an abort."""
    floors = [0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 1.0]

    for total, cache, priced in _random_measurements(seed=7_311, count=200):
        verdicts = [
            decide_measured_cache_abort(
                _progress(total, cache),
                priced,
                _settings(min_deficit_ratio=floor),
            ).abort
            for floor in floors
        ]
        assert verdicts == sorted(verdicts, reverse=True), (
            total,
            cache,
            priced,
            verdicts,
        )


def test_ratio_floor_can_block_an_abort_the_bound_would_allow() -> None:
    """Both conditions must hold: a big deficit with a high cache ratio waits.

    ``deficit_ratio`` gates on *how much* of the prompt had to be recomputed;
    ``bound_tokens`` gates on how far that exceeds the price. A 30 %-priced
    prompt that re-prefilled 40 % is over any small slack, but a 0.6 floor
    says "this is not the fake-warm signature".
    """
    progress = _progress(100_000, 60_000)  # 40 000 new, ratio 0.40
    priced = 5_000

    permissive = decide_measured_cache_abort(progress, priced, _settings())
    assert permissive.abort is True

    strict = decide_measured_cache_abort(
        progress, priced, _settings(min_deficit_ratio=0.6)
    )
    assert strict.abort is False
    assert strict.detail == DETAIL_BELOW_RATIO_FLOOR


# ---------------------------------------------------------------------------
# Configuration resolution (rollback knob parsing)
# ---------------------------------------------------------------------------


def test_absent_config_uses_documented_defaults() -> None:
    """No config at all (or no keys) means the documented defaults."""
    assert AbortSettings.from_config(None) == AbortSettings()
    assert AbortSettings.from_config({"server": {}}) == AbortSettings()


def test_config_accepts_flat_and_nested_shapes() -> None:
    """``server``-nested (shipped shape) and flat dicts resolve identically."""
    values = {BOUND_TOKENS_KEY: 500, MIN_DEFICIT_RATIO_KEY: 0.1}
    nested = AbortSettings.from_config({"server": values})
    flat = AbortSettings.from_config(values)

    assert nested == flat == AbortSettings(
        enabled=True, bound_tokens=500, min_deficit_ratio=0.1
    )


@pytest.mark.parametrize(
    "enabled_value", [1, True, "1", "true", "yes", "on", "enabled"]
)
def test_enabled_spellings(enabled_value) -> None:
    settings = AbortSettings.from_config({"server": {ENABLED_KEY: enabled_value}})
    assert settings.enabled is True


@pytest.mark.parametrize(
    "server_cfg",
    [
        {BOUND_TOKENS_KEY: -1},
        {BOUND_TOKENS_KEY: "2000"},
        {BOUND_TOKENS_KEY: float("nan")},
        {MIN_DEFICIT_RATIO_KEY: 1.5},
        {MIN_DEFICIT_RATIO_KEY: -0.1},
        {MIN_DEFICIT_RATIO_KEY: "0.25"},
        {ENABLED_KEY: "maybe"},
    ],
)
def test_malformed_config_fails_open(server_cfg) -> None:
    """A config we cannot understand disables the feature (never over-aborts)."""
    settings = AbortSettings.from_config({"server": server_cfg})

    assert settings.enabled is False
    assert (
        decide_measured_cache_abort(
            _progress(FAKE_WARM_TOTAL, FAKE_WARM_CACHE), FAKE_WARM_PRICED, settings
        ).abort
        is False
    )


def test_non_mapping_config_fails_open() -> None:
    assert AbortSettings.from_config("server: {}").enabled is False
    assert AbortSettings.from_config({"server": ["not", "a", "mapping"]}).enabled is False


def test_decision_never_raises_on_json_roundtrip_measurements() -> None:
    """Fuzz the public entry point: whatever arrives, a verdict comes back."""
    rng = random.Random(4_242)
    junk = [
        None,
        True,
        0,
        -1,
        "",
        "56000",
        [],
        {},
        {"total": 1},
        {"total": "x", "cache": "y"},
        float("nan"),
        json.loads('{"total": 56000, "cache": 200}'),
    ]

    for _ in range(100):
        progress = rng.choice(junk + [_progress(56_000, 200)])
        priced = rng.choice(junk)
        decision = decide_measured_cache_abort(progress, priced)
        assert isinstance(decision.abort, bool)
        if decision.abort:
            assert decision.reason == MEASURED_CACHE_DEFICIT_REASON
