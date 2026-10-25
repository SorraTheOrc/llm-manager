"""Cold-prefill budget / deadline configuration surface (LP-0MVN4LBEQ005QRVL).

Lands the configuration half of LP-0MVN3KMGI009BZ0D: the new cold-prefill
budget keys, the widened prefill deadline, and a fail-fast assertion that keeps
the local timeout ordering coherent. The admission *decision* logic is
deliberately untouched here (LP-0MVN4LBZ500281Z7 wires the budget in), so the
only new refusal path is the startup ordering validator.

AC mapping:

- AC1 shipped keys (values + documentation) in ``proxy/config.yaml``;
- AC2 legacy-safe defaults, malformed values warned and defaulted, and the
  "percentile floor above the budget" warning;
- AC3 the deadline/queue/first-byte/max-prefill/client ordering is asserted and
  each inversion is FATAL;
- AC4 the rollback set is documented in ``config.yaml`` and ``docs/routing.md``;
- AC5 no decision change: an absent key reproduces today's behaviour.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from proxy import prefill_eta

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "proxy" / "config.yaml"
ROUTING_DOC_PATH = REPO_ROOT / "proxy" / "docs" / "routing.md"

#: Shipped values the parent item specifies (LP-0MVN3KMGI009BZ0D proposal).
SHIPPED = {
    "local_prefill_deadline_seconds": 120,
    "local_cold_prefill_budget_tokens": 25000,
    "local_admission_guard_eta_percentile_min_tokens": 20000,
    "local_admission_guard_budget_safety_fraction": 0.95,
    "local_admission_guard_reference_tokens": 25000,
}


@pytest.fixture(autouse=True)
def _clean_calibration():
    prefill_eta.reset()
    yield
    prefill_eta.reset()


def _shipped_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _cfg(**server_overrides) -> dict:
    server = {
        "session_slot_pool_size": 2,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": 60,
        "contention_queue_max_depth": 8,
    }
    server.update(server_overrides)
    return {"server": server}


# ---------------------------------------------------------------------------
# AC1: the keys ship with the documented values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("key", "expected"), sorted(SHIPPED.items()))
def test_shipped_config_declares_the_budget_and_deadline_knobs(key, expected):
    server = _shipped_config()["server"]

    assert server[key] == expected


def test_shipped_keys_are_documented_in_place():
    """Each new/retuned key carries a comment stating units and rollback."""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    for key in SHIPPED:
        idx = text.index(f"  {key}:")
        preamble = text[:idx].rsplit("\n\n", 1)[-1]
        assert key in preamble or "#" in preamble, f"{key} has no comment block"
    assert "ROLLBACK" in text
    assert "25000" in text and "20000" in text and "0.95" in text


# ---------------------------------------------------------------------------
# AC2: legacy-safe defaults and malformed-value handling
# ---------------------------------------------------------------------------

def test_absent_keys_reproduce_todays_behaviour():
    guard = prefill_eta.resolve_config(_cfg())

    assert guard.cold_prefill_budget_tokens == 0, "absent budget must be disabled"
    assert (
        guard.eta_percentile_min_tokens == prefill_eta.CONSERVATIVE_MIN_TOKENS
    ), "absent percentile floor must keep the old hard-coded 32768"
    assert guard.budget_safety_fraction == 0.0, "absent safety fraction must be off"
    assert guard.prefill_deadline_seconds == 90.0
    assert guard.reference_tokens == 32_000


def test_present_keys_are_resolved():
    guard = prefill_eta.resolve_config(_cfg(**SHIPPED))

    assert guard.cold_prefill_budget_tokens == 25000
    assert guard.eta_percentile_min_tokens == 20000
    assert guard.budget_safety_fraction == pytest.approx(0.95)
    assert guard.prefill_deadline_seconds == pytest.approx(120.0)
    assert guard.reference_tokens == 25000


@pytest.mark.parametrize(
    "key", ["local_cold_prefill_budget_tokens", "local_admission_guard_eta_percentile_min_tokens"]
)
def test_malformed_integer_knobs_fall_back_with_a_warning(key, caplog):
    with caplog.at_level(logging.WARNING, logger="llama-proxy"):
        guard = prefill_eta.resolve_config(_cfg(**{key: "not-a-number"}))

    assert any(key in record.getMessage() for record in caplog.records), (
        "a typo in a startup-visible knob must be diagnosable, not silent"
    )
    assert getattr(guard, "cold_prefill_budget_tokens" if "budget" in key
                   else "eta_percentile_min_tokens") == (
        0 if "budget" in key else prefill_eta.CONSERVATIVE_MIN_TOKENS
    )


def test_malformed_safety_fraction_falls_back_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="llama-proxy"):
        guard = prefill_eta.resolve_config(
            _cfg(local_admission_guard_budget_safety_fraction="half")
        )

    assert guard.budget_safety_fraction == 0.0
    assert any(
        "local_admission_guard_budget_safety_fraction" in r.getMessage()
        for r in caplog.records
    )


def test_safety_fraction_above_one_is_rejected_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="llama-proxy"):
        guard = prefill_eta.resolve_config(
            _cfg(local_admission_guard_budget_safety_fraction=1.5)
        )

    assert guard.budget_safety_fraction == 0.0
    assert any(
        "local_admission_guard_budget_safety_fraction" in r.getMessage()
        for r in caplog.records
    )


def test_percentile_floor_above_budget_warns_that_the_band_guard_is_dead(caplog):
    with caplog.at_level(logging.WARNING, logger="llama-proxy"):
        prefill_eta.resolve_config(
            _cfg(
                local_cold_prefill_budget_tokens=20000,
                local_admission_guard_eta_percentile_min_tokens=32768,
            )
        )

    assert any(
        "silently dead" in r.getMessage() for r in caplog.records
    ), "a percentile floor above the budget makes the band guard a no-op"


def test_percentile_floor_above_budget_is_not_warned_when_budget_disabled(caplog):
    """With the cap off (rollback) the interaction is moot — no false alarm."""
    with caplog.at_level(logging.WARNING, logger="llama-proxy"):
        prefill_eta.resolve_config(
            _cfg(
                local_cold_prefill_budget_tokens=0,
                local_admission_guard_eta_percentile_min_tokens=32768,
            )
        )

    assert not any("silently dead" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# AC3: ordering assertion (startup FATAL)
# ---------------------------------------------------------------------------

def test_shipped_config_passes_the_ordering_validator():
    assert prefill_eta.validate_prefill_timeout_ordering(_shipped_config()) == []


def test_ordering_holds_for_the_documented_values():
    """The documented chain: 60 < 120 < 840 < 900 and deadline < client 900."""
    guard = prefill_eta.resolve_config(_shipped_config())
    server = _shipped_config()["server"]
    # The first-byte ceiling and the lifetime wedge are documented defaults
    # (commented out in config.yaml unless an operator overrides them).
    first_byte = float(
        server.get("local_dispatch_first_byte_timeout_seconds", 840)
    )
    max_prefill = float(server.get("local_dispatch_max_prefill_seconds", 900))

    assert (
        server["contention_queue_max_wait_seconds"]
        < guard.prefill_deadline_seconds
        < first_byte
        < max_prefill
    )
    assert guard.prefill_deadline_seconds < prefill_eta.CLIENT_QUEUE_DEADLINE_SECONDS


@pytest.mark.parametrize(
    ("overrides", "expected_fragment"),
    [
        (
            {"contention_queue_max_wait_seconds": 120, "local_prefill_deadline_seconds": 120},
            "contention_queue_max_wait_seconds",
        ),
        (
            {
                "local_prefill_deadline_seconds": 840,
                "local_dispatch_first_byte_timeout_seconds": 840,
            },
            "local_dispatch_first_byte_timeout_seconds",
        ),
        (
            {
                "local_dispatch_first_byte_timeout_seconds": 950,
                "local_dispatch_max_prefill_seconds": 900,
            },
            "local_dispatch_max_prefill_seconds",
        ),
        (
            {"local_prefill_deadline_seconds": 900},
            "client queue deadline",
        ),
    ],
    ids=["queue>=deadline", "deadline>=first-byte", "first-byte>max-prefill",
         "deadline>=client-deadline"],
)
def test_each_ordering_inversion_is_fatal(overrides, expected_fragment):
    problems = prefill_eta.validate_prefill_timeout_ordering(_cfg(**overrides))

    assert problems, "a violating config must be refused, not silently accepted"
    assert all(p.startswith("FATAL: ") for p in problems)
    assert any(expected_fragment in p for p in problems)


def test_disabled_bounds_are_not_ordered():
    """0 means "no bound" and must not be treated as a violation."""
    problems = prefill_eta.validate_prefill_timeout_ordering(
        _cfg(
            local_prefill_deadline_seconds=0,
            local_dispatch_first_byte_timeout_seconds=0,
            local_dispatch_max_prefill_seconds=0,
        )
    )

    assert problems == []


def test_startup_validator_raises_on_a_violating_config():
    """The server startup hook turns the ordering problems into a ValueError."""
    server = pytest.importorskip("proxy.server")

    with pytest.raises(ValueError, match="Prefill timeout config validation failed"):
        server._startup_validate_prefill_timeout_config(
            _cfg(
                contention_queue_max_wait_seconds=120,
                local_prefill_deadline_seconds=120,
            )
        )


def test_startup_validator_accepts_the_shipped_config():
    server = pytest.importorskip("proxy.server")

    server._startup_validate_prefill_timeout_config(_shipped_config())


# ---------------------------------------------------------------------------
# AC4: rollback documented
# ---------------------------------------------------------------------------

def test_rollback_set_is_documented_in_config_and_docs():
    config_text = CONFIG_PATH.read_text(encoding="utf-8")
    docs_text = ROUTING_DOC_PATH.read_text(encoding="utf-8")

    for text in (config_text, docs_text):
        assert "ROLLBACK" in text or "Rollback" in text
        for token in (
            "local_cold_prefill_budget_tokens",
            "local_admission_guard_eta_percentile_min_tokens",
            "local_admission_guard_budget_safety_fraction",
            "local_prefill_deadline_seconds",
        ):
            assert token in text


# ---------------------------------------------------------------------------
# AC5: no admission-decision change yet
# ---------------------------------------------------------------------------

def test_budget_key_does_not_change_the_admission_decision_yet():
    """The decision logic is a sibling child's job — this child only ships knobs.

    A 40k cold prompt priced below the deadline is still admitted even though
    it exceeds the 25k budget: enforcing the cap is LP-0MVN4LBZ500281Z7.
    """
    for _ in range(6):
        prefill_eta.record_dispatch_sample("Qwen3", 40_000, 0.0, 20_000.0)
    config = _shipped_config()

    decision = prefill_eta.evaluate_admission(
        config, "Qwen3", 40_000, 0.0, priority="medium"
    )

    assert decision.admit is True
    assert decision.reason == "within_budget"
