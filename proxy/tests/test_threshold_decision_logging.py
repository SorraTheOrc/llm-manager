"""Every admission decision names the threshold that governed it.

Auditable threshold decision logging (``LP-0MW81DHPH00650NY``, parent
``LP-0MW7V292T0020G02`` AC3).

The residency ladder (``LP-0MVN1IQTF0037WTG``) already reports *what* was
measured — ``residency_source=`` plus the additive ``measurement=`` provenance
— but not *which threshold* the verdict was held to. An operator reading
``decision=admit estimated_tokens=55492`` therefore cannot tell a cold-priced
admission (new tokens under the 38 000 cold cap) from a measured-resident
admission (the 120 000 warm threshold governing, ``LP-0MW81DFML006O5KL``)
without reconstructing the ladder by hand. The rule this module pins:

- ``threshold_applied=cold`` / ``threshold_value=<cold cap>`` when the decision
  has **no measured residency** — the unmeasured-cold denial
  (``residency_source=unknown``) and the measured-zero denial
  (``residency_source=non_resident``) both fall to the cold threshold, because
  the warm threshold has no basis without a measurement;
- ``threshold_applied=warm`` / ``threshold_value=<warm cap>`` when the decision
  *is* measured-resident (``measured`` / ``measured_last_dispatch`` /
  ``session_slot``) — the parent names this the "measured-warm" decision and
  keeps the 120 000 cap for it;
- ``threshold_applied=none`` / ``threshold_value=none`` when the decision has no
  residency basis at all (the ladder is disabled: ``facts`` is ``None``), so no
  token threshold is attributed to it.

Values are read from the thresholds the caller passed to the ladder, never from
module constants, so a retune of ``local_large_context_cold_cache_threshold``
or ``local_large_context_warm_cache_threshold`` is visible on the line.

**Additive only.** ``threshold_applied=`` and ``threshold_value=`` are appended
after ``session=``: every pre-change key keeps its relative order, and the
shipped ``proxy-usage-analysis`` parser (``log_parser.parse_log_line``) parses
the new line with its existing fields intact — pinned below by parsing the
emitted line through that parser rather than by matching a format string.
"""

from __future__ import annotations

import logging
import re
import time

import pytest

from proxy import measured_prefill, observability, provider
from proxy import residency as residency_module
from proxy import session as session_registry

MODEL = "Qwen3"
SESSION = "sess-a"
OTHER_SESSION = "sess-other"

#: Shipped thresholds (``proxy/config.yaml`` server block).
COLD_THRESHOLD = 38_000
WARM_THRESHOLD = 120_000
RESIDENT_PREFIX_MIN_RATIO = 0.5

#: The 2026-11-07 admission size, and a resident prefix that clears the 0.5
#: ratio gate while leaving 25 492 live new tokens — a *measured-warm*
#: admission at 55 492 total tokens.
LARGE_PROMPT = 55_492
MEASURED_PREFIX = 30_000
WARM_LOOKING_PREDICTION = 0.99


@pytest.fixture(autouse=True)
def _reset_residency_state():
    """Isolate the cached ``/slots`` snapshot and the ownership registry."""
    observability._last_slot_details_cache[:] = []
    observability._last_slot_details_at = 0.0
    session_registry._slot_owners.clear()
    measured_prefill.reset_measured_prefill()
    yield
    observability._last_slot_details_cache[:] = []
    observability._last_slot_details_at = 0.0
    session_registry._slot_owners.clear()
    measured_prefill.reset_measured_prefill()


# ---------------------------------------------------------------------------
# Helpers (same documented stubs as the residency matrix)
# ---------------------------------------------------------------------------

def _slot(slot_id: int, context_tokens: int | None) -> dict:
    return {
        "slot_id": slot_id,
        "is_processing": True,
        "n_decoded": None,
        "context_tokens": context_tokens,
    }


def _install(slots: list[dict], owners: dict[tuple, str], *, age: float = 0.5) -> None:
    observability._last_slot_details_cache[:] = [dict(s) for s in slots]
    observability._last_slot_details_at = time.monotonic() - age
    session_registry._slot_owners.clear()
    session_registry._slot_owners.update(owners)


def _arrange(basis: str) -> None:
    """Arrange one documented residency basis."""
    if basis == "unmeasured":
        _install([], {})
        return
    if basis == "measured":
        _install([_slot(0, MEASURED_PREFIX)], {("", 0): SESSION})
        return
    if basis == "measured_last_dispatch":
        _install([_slot(0, None)], {("", 0): SESSION})
        measured_prefill.set_measured_prefill(
            MODEL,
            SESSION,
            measured_prefill.MeasuredPrefill(
                total=MEASURED_PREFIX, cache=0, processed=0, time_ms=1234.0
            ),
        )
        return
    if basis == "session_slot":
        _install([_slot(0, None)], {("", 0): SESSION})
        return
    if basis == "non_resident":
        _install([_slot(0, LARGE_PROMPT)], {("", 0): OTHER_SESSION})
        return
    raise AssertionError(f"unknown residency basis {basis!r}")


def _route(
    monkeypatch: pytest.MonkeyPatch,
    *,
    estimated_tokens: int = LARGE_PROMPT,
    predicted_ratio: float = WARM_LOOKING_PREDICTION,
    cold_threshold: int = COLD_THRESHOLD,
    warm_threshold: int = WARM_THRESHOLD,
    resident_prefix_min_ratio: float = RESIDENT_PREFIX_MIN_RATIO,
):
    """Run the ladder exactly as the router calls it."""
    monkeypatch.setattr(
        provider, "expected_cached_ratio", lambda *a, **k: predicted_ratio
    )
    return provider._should_skip_local(
        MODEL,
        SESSION,
        {"messages": []},
        cold_threshold,
        estimated_tokens=estimated_tokens,
        warm_cache_threshold=warm_threshold,
        resident_prefix_min_ratio=resident_prefix_min_ratio,
        residency_view=residency_module.get_residency_view(),
    )


def _decision_line(caplog, monkeypatch, **route_kwargs) -> str:
    """Run one admission and return its ``routing_resident_prefix`` message."""
    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        _route(monkeypatch, **route_kwargs)
    lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("routing_resident_prefix")
    ]
    assert lines, "every admission decision must emit a decision line"
    return lines[-1]


def _field(line: str, name: str) -> str | None:
    match = re.search(rf"\b{name}=(\S+)", line)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# AC2 — the threshold basis is on the line, for both named decisions
# ---------------------------------------------------------------------------

def test_unmeasured_cold_decision_reports_the_cold_threshold(caplog, monkeypatch):
    """AC2 (unmeasured-cold): 55k with nothing measured is held to 38 000.

    The line must carry the residency basis it decided on (``unknown`` /
    ``unavailable:stale_view`` *and* the cold threshold), so the 2026-11-07
    55 492-token prompt's detour to the remote chain is attributable from the
    line alone.
    """
    _arrange("unmeasured")

    line = _decision_line(caplog, monkeypatch)

    assert _field(line, "decision") == provider.SKIP_REASON_UNMEASURED_COLD
    assert _field(line, "residency_source") == provider.RESIDENCY_SOURCE_UNKNOWN
    assert (
        _field(line, "measurement")
        == provider.MEASUREMENT_UNAVAILABLE_STALE_VIEW
    )
    assert _field(line, "threshold_applied") == provider.THRESHOLD_APPLIED_COLD
    assert _field(line, "threshold_value") == str(COLD_THRESHOLD)


def test_measured_warm_admission_reports_the_warm_threshold(caplog, monkeypatch):
    """AC2 (measured-warm): a measured-resident 55k prompt keeps 120 000.

    The same prompt size is *admitted* here because the session's resident
    prefix was measured at 30 000 tokens — the warm threshold governs — and the
    line must say so rather than looking like a cold-priced admission.
    """
    _arrange("measured")

    line = _decision_line(caplog, monkeypatch, predicted_ratio=0.0)

    assert _field(line, "decision") == "admit"
    assert _field(line, "residency_source") == provider.RESIDENCY_SOURCE_MEASURED
    assert _field(line, "measurement") == provider.MEASUREMENT_MEASURED
    assert _field(line, "threshold_applied") == provider.THRESHOLD_APPLIED_WARM
    assert _field(line, "threshold_value") == str(WARM_THRESHOLD)


# ---------------------------------------------------------------------------
# The whole basis → threshold mapping is pinned, not just the two AC2 cells
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("basis", "expected_decision", "expected_threshold"),
    [
        # No measurement ⇒ the warm cap has no basis ⇒ cold governs.
        pytest.param(
            "unmeasured",
            provider.SKIP_REASON_UNMEASURED_COLD,
            provider.THRESHOLD_APPLIED_COLD,
            id="unmeasured-cold",
        ),
        # A measured zero is still no reuse: priced cold, denied by the ratio.
        pytest.param(
            "non_resident",
            provider.SKIP_REASON_RESIDENT_PREFIX,
            provider.THRESHOLD_APPLIED_COLD,
            id="measured-zero-cold",
        ),
        # Measured residency in its three documented forms keeps the warm cap.
        pytest.param("measured", "admit", provider.THRESHOLD_APPLIED_WARM,
                     id="measured-warm"),
        pytest.param(
            "measured_last_dispatch",
            "admit",
            provider.THRESHOLD_APPLIED_WARM,
            id="measured-last-dispatch-warm",
        ),
        pytest.param(
            "session_slot", "admit", provider.THRESHOLD_APPLIED_WARM,
            id="session-slot-warm",
        ),
    ],
)
def test_threshold_basis_by_residency_source(
    caplog, monkeypatch, basis, expected_decision, expected_threshold
):
    """The basis, never the predicted ratio, selects the threshold."""
    _arrange(basis)

    line = _decision_line(caplog, monkeypatch, predicted_ratio=0.0)

    assert _field(line, "decision") == expected_decision
    assert _field(line, "threshold_applied") == expected_threshold


def test_threshold_values_follow_the_configured_thresholds(caplog, monkeypatch):
    """A retune moves the reported value too — the numbers are not hardcoded.

    Both cells use thresholds that differ from the shipped 38 000 / 120 000, so
    a future retune stays readable from the same line: a 41 000-token unmeasured
    prompt reports ``cold``/40 000, and a measured-resident one reports
    ``warm``/100 000.
    """
    _arrange("unmeasured")
    line = _decision_line(
        caplog,
        monkeypatch,
        estimated_tokens=41_000,
        cold_threshold=40_000,
        warm_threshold=100_000,
    )
    assert _field(line, "threshold_applied") == provider.THRESHOLD_APPLIED_COLD
    assert _field(line, "threshold_value") == "40000"

    caplog.clear()
    _arrange("measured")
    line = _decision_line(
        caplog,
        monkeypatch,
        predicted_ratio=0.0,
        cold_threshold=40_000,
        warm_threshold=100_000,
    )
    assert _field(line, "threshold_applied") == provider.THRESHOLD_APPLIED_WARM
    assert _field(line, "threshold_value") == "100000"


# ---------------------------------------------------------------------------
# No basis (ladder disabled) is stated, never left ambiguous
# ---------------------------------------------------------------------------

def test_no_residency_basis_reports_none_and_never_raises(caplog):
    """A line logged without facts attributes no threshold — and cannot raise.

    The tenancy rung logs through the same line when the residency ladder is
    disabled (``local_resident_prefix_min_ratio: 0``), so there is no token
    threshold to attribute; the field says ``none`` instead of guessing.
    """
    with caplog.at_level(logging.INFO, logger="llama-proxy"):
        provider._log_residency_decision(
            session_id=SESSION,
            estimated_tokens=LARGE_PROMPT,
            decision=provider.SKIP_REASON_TENANT_LIMIT,
        )

    line = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("routing_resident_prefix")
    )
    assert _field(line, "threshold_applied") == provider.THRESHOLD_APPLIED_NONE
    assert _field(line, "threshold_value") == "none"


def test_threshold_applied_is_a_closed_vocabulary():
    """Operators and parsers see one of a fixed set, never a free-form string."""
    assert provider.THRESHOLD_APPLIED_VOCABULARY == (
        provider.THRESHOLD_APPLIED_COLD,
        provider.THRESHOLD_APPLIED_WARM,
        provider.THRESHOLD_APPLIED_NONE,
    )


# ---------------------------------------------------------------------------
# AC3 — additive only: existing keys keep their order, the parser keeps working
# ---------------------------------------------------------------------------

#: The pre-change key order of the decision line (``proxy/docs/routing.md``).
#: The new fields must be appended after it, never interleaved.
LEGACY_KEY_ORDER = (
    "provider=",
    "model=",
    "decision=",
    "n_past_actual=",
    "estimated_tokens=",
    "reuse_ratio=",
    "slot_owner_session=",
    "residency_source=",
    "measurement=",
    "min_ratio=",
    "tenants=",
    "occupants=",
    "session=",
)


def test_existing_keys_keep_their_order_and_new_keys_are_appended(
    caplog, monkeypatch
):
    """AC3: the additive suffix cannot disturb a positional consumer.

    Three additive groups are appended, in order: the threshold pair
    (``LP-0MW81DHPH00650NY``), the per-admission latency figure
    (``LP-0MW959R5S009WDM8``) and the slot candidate the verdict was measured
    against (``LP-0MW959RRO008AAH9``). Nothing may be interleaved into the
    pre-change keys, which is what this pin is for.
    """
    _arrange("measured")

    line = _decision_line(caplog, monkeypatch, predicted_ratio=0.0)

    keys = [token.split("=", 1)[0] for token in line.split()[1:]]
    assert keys[: len(LEGACY_KEY_ORDER)] == [
        key.rstrip("=") for key in LEGACY_KEY_ORDER
    ], line
    assert keys[len(LEGACY_KEY_ORDER) :] == [
        "threshold_applied",
        "threshold_value",
        "admission_overhead_ms",
        "slot_candidate",
    ], line


def test_shipped_parser_still_parses_the_new_line(caplog, monkeypatch):
    """AC3: ``proxy-usage-analysis`` keeps its fields on a new-format line.

    Parsed through the shipped parser (not a format-string match), so the
    assertion is about the operator-visible contract the report consumes: the
    new keys are additive and the existing fields are unchanged.
    """
    import sys
    from pathlib import Path

    scripts = (
        Path(__file__).resolve().parents[2]
        / ".pi/skills/proxy-usage-analysis/scripts"
    )
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import log_parser

    _arrange("measured")
    line = _decision_line(caplog, monkeypatch, predicted_ratio=0.0)

    event = log_parser.parse_log_line(
        f"2026-11-08 12:00:00,000 - INFO - {line}"
    )

    assert event is not None, f"the shipped parser must still parse: {line!r}"
    assert event.kind == "residency_decision"
    assert event.decision == "admit"
    assert event.session == SESSION
    assert event.estimated_tokens == LARGE_PROMPT
    assert event.n_past_actual == MEASURED_PREFIX
    assert event.reuse_ratio == pytest.approx(MEASURED_PREFIX / LARGE_PROMPT)
    assert event.slot_owner_session == SESSION
    assert event.residency_source == provider.RESIDENCY_SOURCE_MEASURED
    assert event.measurement == provider.MEASUREMENT_MEASURED
