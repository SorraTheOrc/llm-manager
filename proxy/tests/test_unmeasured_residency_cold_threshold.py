"""Unmeasured residency must apply the 38 000 cold threshold (no silent warm).

TDD pin for ``LP-0MW7V292T0020G02`` AC1/AC2 (child 1,
``LP-0MW81DDGF001MFPK``), written **before** the enforcement lands (child 2,
``LP-0MW81DFML006O5KL``). Every cell below is a pure call: no HTTP, no sleeps,
no ``/slots`` query on the admission path.

Measured evidence that motivates the rule (certified 2026-11-06 14:15:50 →
2026-11-08 00:00, 33.736 h):

- **55 of 60** ``prefill_deadline`` events had ``estimated_tokens > 38 000``
  and **none** exceeded the 120 000 warm threshold — the failures live exactly
  in the band that only the predicted-warm path admits.
- The resident-prefix probe ran ~4 500 times: 3 363 of its verdicts were
  ``measurement=unavailable:no_slot``, only 466 were genuine measurements.
- 2026-11-07 20:48:51: a **55 492-token** prompt was priced
  ``cached_ratio=1.00 new_tokens=172``, admitted at ``eta_prefill_s=130.8``
  and evicted at 180.1 s, immediately after
  ``routing_resident_prefix … residency_source=non_resident
  measurement=unavailable:no_slot``. Nothing had been measured; the
  predicted-warm fail-open admitted it anyway.

The rule pinned here (parent AC1/AC2):

- ``estimated_tokens > 38 000`` on an **unmeasured** residency basis
  (``residency_source=unknown`` — ``measurement=unavailable:stale_view``, a
  missing ``/slots`` snapshot, or a probe that raises) ⇒ **not dispatched
  locally**; it falls through to the remote chain with an explicit reason and
  the decision never depends on the predicted cached ratio;
- ``estimated_tokens <= 38 000`` ⇒ local, measured or not: exactly 38 000 is
  the documented boundary and stays local;
- a **measured-resident** prompt (``residency_source`` in ``{measured,
  measured_last_dispatch, session_slot}``) keeps the 120 000 warm threshold and
  is priced from its live prefix, so a measured-resident 55k prompt is still
  served locally;
- a **measured zero** (fresh view, this session owns no slot ⇒
  ``residency_source=non_resident``) keeps denying via
  ``resident_prefix_insufficient`` — the new unmeasured rule must not replace
  that measured denial.

Only the documented residency sources are stubbed: the cached ``/slots``
snapshot (``observability._last_slot_details_cache`` /
``_last_slot_details_at``), the in-process ownership registry
(``session._slot_owners``) and — for the last-dispatch rung — the
``proxy.measured_prefill`` store. The autouse fixtures reset all three, so
every cell runs in isolation and in any order.

Red-first encoding (repository convention): the cells that pin behaviour the
implementation item (child 2) has not landed yet carry
``pytest.mark.xfail(strict=True)``. They are genuinely red today — run them
without the marker to see the unmeasured 55k admission — and ``strict=True``
means child 2 **must** remove the marker once the enforcement lands: an xfail
that starts passing is a suite failure, never a silent pass. The remaining
cells already hold and are unmarked, so they guard against over-denial (an
unmeasured 30k prompt, the 38 000 boundary, every measured-resident source and
the 120 000 warm cap) while the pinned cells await child 2.

Note for child 2: ``test_unknown_state_falls_back_to_prediction`` in
``test_local_resident_prefix_admission.py`` pins the **retired** fail-open
behaviour this matrix replaces; it is the one existing test expected to change
when the enforcement lands (the warm-path and large-context tests are not
affected).
"""

from __future__ import annotations

import time

import pytest
from proxy.measured_prefill import MeasuredPrefill

from proxy import measured_prefill, observability, provider
from proxy import residency as residency_module
from proxy import session as session_registry

#: Reason carried by every red-first cell; child 2 removes the markers.
PENDING_ENFORCEMENT = (
    "pending LP-0MW81DFML006O5KL (unmeasured residency applies the 38k cold "
    "threshold); remove this xfail marker when the enforcement lands"
)

#: Shipped thresholds (``proxy/config.yaml`` server block). The enforcement
#: must not move them; it only decides which one governs a given basis.
COLD_THRESHOLD = 38_000
WARM_THRESHOLD = 120_000

#: The residency gate the project ships (``local_resident_prefix_min_ratio``).
RESIDENT_PREFIX_MIN_RATIO = 0.5

MODEL = "Qwen3"
SESSION = "sess-a"
OTHER_SESSION = "sess-other"

#: The 2026-11-07 failure size, and the resident prefix used to make the
#: 38k..120k band measured-resident (30 000 / 55 492 ≈ 0.54 ≥ the 0.5 gate,
#: leaving 25 492 live new tokens — under the 38 000 cold cap).
LARGE_PROMPT = 55_492
MEASURED_PREFIX = 30_000

#: Predicted cached ratio that makes the *predicted* path admit anything:
#: ``new_tokens = 55 492 × 0.01 ≈ 555`` tokens. The unmeasured decision must
#: ignore it entirely and reach the cold threshold instead.
WARM_LOOKING_PREDICTION = 0.99


@pytest.fixture(autouse=True)
def _reset_residency_state():
    """Isolate the cached snapshot and the ownership registry."""
    observability._last_slot_details_cache[:] = []
    observability._last_slot_details_at = 0.0
    session_registry._slot_owners.clear()
    yield
    observability._last_slot_details_cache[:] = []
    observability._last_slot_details_at = 0.0
    session_registry._slot_owners.clear()


@pytest.fixture(autouse=True)
def _reset_measured_prefill_state():
    """Isolate the in-process measurement store (last-dispatch rung)."""
    measured_prefill.reset_measured_prefill()
    yield
    measured_prefill.reset_measured_prefill()


# ---------------------------------------------------------------------------
# Stubs for the documented residency sources
# ---------------------------------------------------------------------------

def _slot(slot_id: int, context_tokens: int | None, *, processing: bool = True) -> dict:
    """One cached ``/slots`` entry in the shape the snapshot stores."""
    return {
        "slot_id": slot_id,
        "is_processing": processing,
        "n_decoded": None,
        "context_tokens": context_tokens,
    }


def _install(slots: list[dict], owners: dict[tuple, str], *, age: float = 0.5) -> None:
    """Install a slot snapshot + ownership registry (fresh by default)."""
    observability._last_slot_details_cache[:] = [dict(s) for s in slots]
    observability._last_slot_details_at = time.monotonic() - age
    session_registry._slot_owners.clear()
    session_registry._slot_owners.update(owners)


def _prime_measurement(total: int = MEASURED_PREFIX, *, session: str = SESSION) -> None:
    """Record a prior measured dispatch for ``(MODEL, session)``."""
    measured_prefill.set_measured_prefill(
        MODEL,
        session,
        MeasuredPrefill(total=total, cache=0, processed=0, time_ms=1234.0),
    )


def _arrange(kind: str) -> None:
    """Arrange one residency basis from the parent's closed vocabulary.

    ``measured`` / ``measured_last_dispatch`` / ``session_slot`` are the three
    measured-resident sources AC1 names; ``non_resident`` is a *measured zero*
    (fresh view, no slot of this session's); ``unmeasured`` is no fresh
    snapshot and no ownership — the genuinely unknown basis.
    """
    if kind == "unmeasured":
        _install([], {})
        return
    if kind == "measured":
        _install([_slot(0, MEASURED_PREFIX)], {("", 0): SESSION})
        return
    if kind == "measured_last_dispatch":
        # The deployed b8782 build reports no per-slot ``context_tokens``.
        _install([_slot(0, None)], {("", 0): SESSION})
        _prime_measurement()
        return
    if kind == "session_slot":
        # Owns the slot, size unknown, nothing measured for the pair.
        _install([_slot(0, None)], {("", 0): SESSION})
        return
    if kind == "non_resident":
        _install([_slot(0, LARGE_PROMPT)], {("", 0): OTHER_SESSION})
        return
    raise AssertionError(f"unknown residency basis {kind!r}")


def _route(
    monkeypatch: pytest.MonkeyPatch,
    *,
    estimated_tokens: int,
    predicted_ratio: float,
    residency_view=None,
):
    """Run the local admission ladder exactly as the router calls it."""
    monkeypatch.setattr(
        provider, "expected_cached_ratio", lambda *a, **k: predicted_ratio
    )
    return provider._should_skip_local(
        MODEL,
        SESSION,
        {"messages": []},
        COLD_THRESHOLD,
        estimated_tokens=estimated_tokens,
        warm_cache_threshold=WARM_THRESHOLD,
        resident_prefix_min_ratio=RESIDENT_PREFIX_MIN_RATIO,
        residency_view=residency_view,
    )


# ---------------------------------------------------------------------------
# AC2 — the unmeasured / measured admission matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    (
        "basis", "estimated_tokens", "predicted_ratio", "expected_skip",
        "expected_reason", "expected_source",
    ),
    [
        pytest.param(
            "unmeasured", LARGE_PROMPT, WARM_LOOKING_PREDICTION, True, None,
            provider.RESIDENCY_SOURCE_UNKNOWN,
            id="unmeasured-55k-not-local",
            marks=pytest.mark.xfail(strict=True, reason=PENDING_ENFORCEMENT),
        ),
        pytest.param(
            "unmeasured", 30_000, WARM_LOOKING_PREDICTION, False, None,
            provider.RESIDENCY_SOURCE_UNKNOWN,
            id="unmeasured-30k-local",
        ),
        pytest.param(
            "measured", LARGE_PROMPT, 0.0, False, None,
            provider.RESIDENCY_SOURCE_MEASURED,
            id="measured-55k-local",
        ),
        pytest.param(
            "measured_last_dispatch", LARGE_PROMPT, 0.0, False, None,
            provider.RESIDENCY_SOURCE_MEASURED_LAST_DISPATCH,
            id="measured-last-dispatch-55k-local",
        ),
        pytest.param(
            "session_slot", LARGE_PROMPT, 0.0, False, None,
            provider.RESIDENCY_SOURCE_SESSION_SLOT,
            id="session-slot-55k-local",
        ),
        pytest.param(
            "non_resident", LARGE_PROMPT, 0.0, True,
            provider.SKIP_REASON_RESIDENT_PREFIX,
            provider.RESIDENCY_SOURCE_NON_RESIDENT,
            id="measured-zero-55k-denied",
        ),
    ],
)
def test_admission_matrix(
    monkeypatch,
    basis,
    estimated_tokens,
    predicted_ratio,
    expected_skip,
    expected_reason,
    expected_source,
):
    """The residency basis — never the prediction — selects the threshold."""
    _arrange(basis)
    view = residency_module.get_residency_view()

    # The basis itself is part of the cell: a fixture that silently degrades
    # (e.g. ``measured_last_dispatch`` falling through to ``session_slot``)
    # would otherwise pass the admission assertion while covering nothing.
    facts = provider.resolve_resident_prefix(
        SESSION, estimated_tokens, view=view, model_name=MODEL
    )
    assert facts.source == expected_source
    assert facts.resident is (
        expected_source
        in {
            provider.RESIDENCY_SOURCE_MEASURED,
            provider.RESIDENCY_SOURCE_MEASURED_LAST_DISPATCH,
            provider.RESIDENCY_SOURCE_SESSION_SLOT,
        }
    )

    skip, reason = _route(
        monkeypatch,
        estimated_tokens=estimated_tokens,
        predicted_ratio=predicted_ratio,
        residency_view=view,
    )

    assert skip is expected_skip
    if expected_reason is not None:
        assert reason == expected_reason
    elif expected_skip:
        # The unmeasured cold denial must name itself: an operator reading the
        # log has to be able to explain why a 55k prompt went remote.
        assert isinstance(reason, str) and reason, (
            "an unmeasured cold denial must carry an explicit skip reason"
        )
    else:
        assert reason is None


# ---------------------------------------------------------------------------
# AC2 — the documented boundary
# ---------------------------------------------------------------------------

def test_exactly_the_cold_threshold_still_routes_local(monkeypatch):
    """38 000 is the documented boundary: ``<=`` admits (unmeasured included)."""
    _arrange("unmeasured")

    skip, reason = _route(
        monkeypatch,
        estimated_tokens=COLD_THRESHOLD,
        predicted_ratio=WARM_LOOKING_PREDICTION,
        residency_view=residency_module.get_residency_view(),
    )

    assert (skip, reason) == (False, None)


@pytest.mark.parametrize(
    "predicted_ratio",
    [
        # Predicted cold: the ladder already refuses this today, and must keep
        # refusing it after the enforcement lands (the unmarked control).
        pytest.param(0.0, id="predicted-cold"),
        pytest.param(
            WARM_LOOKING_PREDICTION,
            id="predicted-warm",
            marks=pytest.mark.xfail(strict=True, reason=PENDING_ENFORCEMENT),
        ),
        # The "fully cached" extreme the 2026-11-07 incident logged
        # (``cached_ratio=1.00 new_tokens=172``).
        pytest.param(
            1.0,
            id="predicted-fully-cached",
            marks=pytest.mark.xfail(strict=True, reason=PENDING_ENFORCEMENT),
        ),
    ],
)
def test_one_token_above_the_cold_threshold_is_denied_when_unmeasured(
    monkeypatch, predicted_ratio
):
    """38 001 tokens is cold work: the prediction may not readmit it.

    ``predicted_ratio=1.0`` is the "fully cached" extreme the 2026-11-07
    incident logged (``cached_ratio=1.00 new_tokens=172``); reaching the cold
    threshold must not depend on it.
    """
    _arrange("unmeasured")

    skip, reason = _route(
        monkeypatch,
        estimated_tokens=COLD_THRESHOLD + 1,
        predicted_ratio=predicted_ratio,
        residency_view=residency_module.get_residency_view(),
    )

    assert skip is True
    assert isinstance(reason, str) and reason


# ---------------------------------------------------------------------------
# AC2 — a probe that cannot answer fails to the cold threshold
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=PENDING_ENFORCEMENT)
def test_probe_error_fails_to_the_cold_threshold(monkeypatch):
    """A raising residency probe is *unmeasured*, never silently warm.

    ``resolve_resident_prefix`` swallows a view-build failure and reports
    ``residency_source=unknown``. That unknown basis must land on the cold
    threshold, so a broken probe can only make the proxy more conservative.
    """

    def _boom(**_kwargs):
        raise RuntimeError("residency probe unavailable")

    monkeypatch.setattr(residency_module, "get_residency_view", _boom)

    skip, reason = _route(
        monkeypatch,
        estimated_tokens=LARGE_PROMPT,
        predicted_ratio=WARM_LOOKING_PREDICTION,
        residency_view=None,
    )

    assert skip is True
    assert isinstance(reason, str) and reason


def test_unmeasured_basis_is_classified_unknown_never_warm(monkeypatch):
    """No snapshot and no ownership ⇒ ``unknown`` / ``unavailable:stale_view``.

    The classification is the input the enforcement decision reads, so it is
    pinned directly: a basis that cannot be measured must never be reported as
    a measurement.
    """
    _arrange("unmeasured")

    facts = provider.resolve_resident_prefix(
        SESSION,
        LARGE_PROMPT,
        view=residency_module.get_residency_view(),
        model_name=MODEL,
    )

    assert facts.source == provider.RESIDENCY_SOURCE_UNKNOWN
    assert facts.measurement == provider.MEASUREMENT_UNAVAILABLE_STALE_VIEW
    assert facts.resident is False
    assert facts.n_past_actual is None


# ---------------------------------------------------------------------------
# AC1/AC4 — a measured-resident prompt keeps the 120 000 warm threshold
# ---------------------------------------------------------------------------

def test_measured_resident_prompt_above_the_cold_threshold_serves_locally(monkeypatch):
    """A measured-resident 90k prompt is in the warm band and stays local."""
    # 60 000 of a 90 000-token prompt is resident (0.67 ≥ the 0.5 gate), so the
    # live new-token term is 30 000 — inside the 38 000 cold cap.
    _install([_slot(0, 60_000)], {("", 0): SESSION})

    skip, reason = _route(
        monkeypatch,
        estimated_tokens=90_000,
        predicted_ratio=WARM_LOOKING_PREDICTION,
        residency_view=residency_module.get_residency_view(),
    )

    assert (skip, reason) == (False, None)


def test_measured_resident_prompt_is_still_denied_above_the_warm_threshold(
    monkeypatch,
):
    """The 120 000 hard cap is unchanged and evaluated before the cold check."""
    _install([_slot(0, 130_000)], {("", 0): SESSION})

    skip, reason = _route(
        monkeypatch,
        estimated_tokens=WARM_THRESHOLD + 1,
        predicted_ratio=0.0,
        residency_view=residency_module.get_residency_view(),
    )

    assert (skip, reason) == (True, "context_too_large")
