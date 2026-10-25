"""2026-10-24 slot-thrash replay (LP-0MVN1IQTF0037WTG AC7, integration half).

Two executable proofs over *real* captured data:

1. **The divergence case is decided by measurement.** The same 49,103-token
   prompt measured ``n_past=738`` (1.5 % reusable) on its first attempt (task
   92383, slot 0) and ``n_past=49103`` (100 % reusable) on the immediate retry
   (task 92422, slot 1). Replaying that extract through the **production**
   ladder denies the cold attempt with ``resident_prefix_insufficient`` and
   admits the retry — under the old predicted-ratio pricing (sidecar 0.99) both
   would have been admitted and the first would have burned its whole prefill
   deadline before being cancelled.

2. **No over-budget cold admission in the pinned window.** Replaying the
   captured 2026-10-24 18:38–23:47 window (1978 local admission attempts)
   through the same ladder yields zero admissions whose prompt exceeds the
   25,000-token cold budget without a resident prefix, with the idle-exempt
   admissions (``route=idle_bypass``) reported separately rather than counted
   as violations.

The replay driver is ``proxy/scripts/replay_prefill_occupancy.py``; it feeds
:func:`proxy.provider._should_skip_local` and
:func:`proxy.residency.ResidencyView` — the same functions production runs — so
these assertions cannot drift from the shipped decision. No network, no
llama-server, no ``/slots`` call: both fixtures are committed.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "proxy" / "scripts" / "replay_prefill_occupancy.py"
SLOT_FIXTURE = (
    REPO_ROOT
    / "proxy"
    / "tests"
    / "fixtures"
    / "slot_thrash_2026-10-24"
    / "tasks_92383_92422.log"
)
WINDOW_FIXTURE = REPO_ROOT / "proxy" / "tests" / "fixtures" / "prefill_thrash_2026_10_24.jsonl"

COLD_BUDGET = 25_000
MIN_RATIO = 0.5

#: The measured facts this window pinned (parent item's problem statement).
PROMPT_TOKENS = 49_103
COLD_N_PAST = 738


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "replay_prefill_occupancy", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _load_driver()


# ---------------------------------------------------------------------------
# AC7 — the resident retry is classified by measurement, not prediction
# ---------------------------------------------------------------------------

def test_slot_fixture_parses_the_two_measured_attempts():
    """The committed extract carries both measurements, in order."""
    attempts = driver.parse_slot_log(str(SLOT_FIXTURE))

    assert [a["task"] for a in attempts] == ["92383", "92422"]
    first, retry = attempts
    assert (first["slot"], first["estimated_tokens"], first["n_past"]) == (
        0,
        PROMPT_TOKENS,
        COLD_N_PAST,
    )
    assert (retry["slot"], retry["estimated_tokens"], retry["n_past"]) == (
        1,
        PROMPT_TOKENS,
        PROMPT_TOKENS,
    )


def test_cold_attempt_denied_and_resident_retry_admitted():
    """Tasks 92383 / 92422 replay as raw measurement says they should."""
    result = driver.replay_residency(
        driver.parse_slot_log(str(SLOT_FIXTURE)),
        cold_budget=COLD_BUDGET,
        min_ratio=MIN_RATIO,
    )

    first, retry = result["rows"]
    assert first["task"] == "92383"
    assert first["reuse_ratio"] == pytest.approx(COLD_N_PAST / PROMPT_TOKENS, abs=1e-4)
    assert first["admitted"] is False
    assert first["reason"] == "resident_prefix_insufficient"

    assert retry["task"] == "92422"
    assert retry["reuse_ratio"] == pytest.approx(1.0)
    assert retry["admitted"] is True
    assert retry["reason"] is None

    assert result["resident_admits"] == 1
    assert result["cold_denials"] == 1
    assert result["violations"] == []


def test_replay_is_reproducible():
    """Re-running the replay yields identical numbers (hermetic fixture)."""
    first = driver.replay_residency(driver.parse_slot_log(str(SLOT_FIXTURE)))
    second = driver.replay_residency(driver.parse_slot_log(str(SLOT_FIXTURE)))

    assert first["rows"] == second["rows"]
    assert first["violations"] == second["violations"] == []


# ---------------------------------------------------------------------------
# AC7 — zero over-budget cold admissions in the pinned window
# ---------------------------------------------------------------------------

def test_pinned_window_has_no_over_budget_cold_admission():
    """1978 real attempts: none is admitted over budget without residency."""
    records = driver.load_window_fixture(str(WINDOW_FIXTURE))
    result = driver.replay_window(
        records, cold_budget=COLD_BUDGET, min_ratio=MIN_RATIO
    )

    assert result["attempts"] == len(records) > 1000
    assert result["violations"] == []
    # The window's own captured totals: every deadline cancellation happened on
    # an over-budget cold attempt, i.e. exactly the class the ladder refuses.
    assert result["over_budget"] > 1000
    assert result["cold_denials"] > 1000


def test_idle_exempt_admissions_are_reported_separately():
    """Idle-served cold work is not a violation (an idle GPU wastes nothing)."""
    records = driver.load_window_fixture(str(WINDOW_FIXTURE))
    result = driver.replay_window(records)

    meta = json.loads(WINDOW_FIXTURE.read_text().splitlines()[0])["_meta"]
    expected_idle = meta["totals"]["idle_bypass_attempts"]
    assert result["idle_exempt"] == expected_idle > 0
    # Every idle-exempt attempt is over budget yet admitted — the reason they
    # must be bucketed apart from violations rather than folded into them.
    idle_rows = [r for r in result["rows"] if r["idle_exempt"]]
    assert all(r["estimated_tokens"] > COLD_BUDGET for r in idle_rows)
    assert all(r["reason"] == "idle_admit" for r in idle_rows)


# ---------------------------------------------------------------------------
# The driver stays an operator tool: the original CLI contract still works
# ---------------------------------------------------------------------------

def _occupancy_log(tmp_path: Path) -> Path:
    log = tmp_path / "proxy.log"
    log.write_text(
        "\n".join(
            [
                "2026-10-21 14:56:00,100 - INFO - local_admission_guard "
                "decision=admit reason=within_budget enforced=true "
                "eta_prefill_s=30.0 budget_s=60.0 busy_slots=0 "
                "in_flight_prefills=0 free_slot=true priority=low policy=admit "
                "model=Qwen3 samples=10 provider=local-qwen3 session=sess-1",
                "2026-10-21 14:56:00,200 - INFO - contention_queue_wait "
                "saturation=prefill session=sess-1",
                "2026-10-21 14:56:12,200 - INFO - contention_queue_dispatch "
                "provider=local session=sess-1 queued_duration=12.0s "
                "depth=1 policy=queue reason=slot_freed",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return log


def test_legacy_occupancy_cli_still_works(tmp_path, capsys):
    """The pre-existing invocation (log + --start/--end) is unchanged."""
    log = _occupancy_log(tmp_path)
    exit_code = driver.main.__wrapped__ if hasattr(driver.main, "__wrapped__") else None
    assert exit_code is None  # sanity: main is a plain function

    import sys

    argv = sys.argv
    sys.argv = [
        "replay_prefill_occupancy.py",
        str(log),
        "--start",
        "2026-10-21 14:00:00",
        "--end",
        "2026-10-21 15:00:00",
    ]
    try:
        assert driver.main() == 0
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "queue waits observed:            1" in out
    assert "occupancy-aware refuses:" in out


def test_occupancy_replay_verdict_unchanged(tmp_path):
    """The occupancy maths (projected = eta * (busy+1)) is untouched."""
    log = _occupancy_log(tmp_path)
    waits, outcomes = driver.parse(
        str(log),
        datetime(2026, 10, 21, 14, 0, 0),
        datetime(2026, 10, 21, 15, 0, 0),
    )
    result = driver.replay(waits, outcomes)

    assert result["queued"] == 1
    assert result["dispatched"] == 1
    # ETA 30 s with one in-flight prefill ⇒ 60 s projected, budget 60 s ⇒ admit.
    assert result["refused"] == 0
    assert result["admits_dispatched"] == 1
