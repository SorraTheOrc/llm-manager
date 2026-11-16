"""Occupancy attribution from measured slot state (LP-0MW8VP0XR007UGQF).

Parent LP-0MW7V2DK400365ER AC3. In the certified 2026-11-06 window only 9 of
60 ``prefill_deadline`` events were attributed occupancy while
``available_slots`` read ``0`` in all 60 — because ``/slots`` times out under a
long prefill and the dispatch-record count stays at 1 even when a second
backend slot is genuinely busy.

The fix derives backend occupancy from the *same* measured progress signal
that drives the lease (fresh log-derived ``_slot_progress_cache`` entries,
union ``/slots`` ``is_processing``), publishes it as the additive
``measured_in_flight_requests`` status field, and has the residual analyser
take ``max(local_in_flight_requests, measured_in_flight_requests)``.

These tests pin the helper derivation, the status field, and the analyser
attribution so the two cannot drift.
"""

import importlib
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from proxy import observability as obs

# ===========================================================================
# measured_occupied_slot_ids / count_measured_occupancy
# ===========================================================================


class TestMeasuredOccupiedSlots:
    def setup_method(self):
        obs._slot_progress_cache.clear()

    def teardown_method(self):
        obs._slot_progress_cache.clear()

    def test_counts_slots_processing_in_slot_detail(self):
        slots = [
            {"slot_id": 0, "is_processing": True},
            {"slot_id": 1, "is_processing": False},
        ]
        assert obs.measured_occupied_slot_ids(slots) == {0}
        assert obs.count_measured_occupancy(slots) == 1

    def test_counts_slots_with_fresh_log_progress_even_when_slot_detail_stale(self):
        """A /slots timeout (empty detail) still yields occupancy from the log."""
        import time

        now = time.time()
        obs._slot_progress_cache[0] = {"n_tokens": 1000, "progress": 0.1, "timestamp": now}
        obs._slot_progress_cache[1] = {"n_tokens": 500, "progress": 0.05, "timestamp": now}
        # /slots detail timed out → empty list.
        assert obs.count_measured_occupancy([]) == 2

    def test_union_of_detail_and_log_sources(self):
        """Union, not max: each source can miss a slot the other sees."""
        import time

        now = time.time()
        obs._slot_progress_cache[1] = {"n_tokens": 500, "progress": 0.05, "timestamp": now}
        slots = [
            {"slot_id": 0, "is_processing": True},
            {"slot_id": 1, "is_processing": False},
        ]
        assert obs.measured_occupied_slot_ids(slots) == {0, 1}
        assert obs.count_measured_occupancy(slots) == 2

    def test_stale_log_entry_not_counted(self):
        import time

        obs._slot_progress_cache[0] = {
            "n_tokens": 1000,
            "progress": 0.1,
            "timestamp": time.time() - (obs.SLOT_PROGRESS_MAX_AGE_SECONDS + 60),
        }
        assert obs.count_measured_occupancy([]) == 0

    def test_none_detail_slot_id_ignored(self):
        slots = [{"slot_id": None, "is_processing": True}]
        assert obs.count_measured_occupancy(slots) == 0

    def test_raw_id_key_also_supported(self):
        """The raw /slots shape uses ``id``; both key shapes are honoured."""
        slots = [{"id": 2, "is_processing": True}]
        assert obs.measured_occupied_slot_ids(slots) == {2}


# ===========================================================================
# Analyser occupancy attribution
# ===========================================================================


@pytest.fixture(name="script")
def _script():
    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module("analyse_prefill_deadline_residual")


def _window_log(script, lines, start, end):
    return script.parse_window(lines, start, end)


AUDIT_SESSION = "audit-occupancy"
START = datetime(2026, 1, 1, 0, 0, 0)
END = datetime(2026, 1, 1, 1, 0, 0)


def _status_line(
    ts: datetime,
    *,
    local_in_flight: int,
    measured_in_flight: int | None,
    available_slots: int = 0,
) -> str:
    parts = [
        ts.strftime("%Y-%m-%d %H:%M:%S,000 - INFO - status_request"),
        f"available_slots={available_slots}",
        "capacity_hint=saturated",
        f"local_in_flight_requests={local_in_flight}",
    ]
    if measured_in_flight is not None:
        parts.append(f"measured_in_flight_requests={measured_in_flight}")
    return " ".join(parts) + "\n"


def _eviction_line(ts: datetime, session: str = AUDIT_SESSION) -> str:
    return (
        ts.strftime("%Y-%m-%d %H:%M:%S,000 - INFO -")
        + " local_eviction_after_acceptance reason=prefill_deadline "
        f"session={session} estimated_tokens=40000 new_tokens=30000 "
        "eta_prefill_s=100 elapsed_s=120 endpoint=local\n"
    )


def _event_for(script, line_ts: datetime):
    event = script.EvictionEvent(
        ts=line_ts,
        session=AUDIT_SESSION,
        reason="prefill_deadline",
        estimated_tokens=40000,
        new_tokens=30000,
        eta_prefill_s=100.0,
        elapsed_s=120.0,
        endpoint="local",
    )
    return event


class TestAnalyserOccupancy:
    def test_measured_in_flight_parsed_from_status_line(self, script):
        lines = [
            _status_line(
                START + timedelta(seconds=60),
                local_in_flight=1,
                measured_in_flight=2,
            )
        ]
        log = script.parse_window(lines, START, END)
        assert log.status_polls[0].measured_in_flight_requests == 2
        assert log.status_polls[0].in_flight_requests == 1

    def test_occupancy_evidence_uses_measured_max(self, script):
        poll_ts = START + timedelta(seconds=60)
        log = script.parse_window(
            [_status_line(poll_ts, local_in_flight=1, measured_in_flight=2)],
            START,
            END,
        )
        event = _event_for(script, START + timedelta(seconds=120))
        evidence = script.occupancy_evidence(log, event)
        assert evidence["in_flight_requests_max"] == 2
        assert evidence["tenant_present"] is True

    def test_pre_field_window_falls_back_to_dispatch_count(self, script):
        """A window captured before the field falls back and stays comparable."""
        poll_ts = START + timedelta(seconds=60)
        log = script.parse_window(
            [_status_line(poll_ts, local_in_flight=1, measured_in_flight=None)],
            START,
            END,
        )
        event = _event_for(script, START + timedelta(seconds=120))
        evidence = script.occupancy_evidence(log, event)
        assert evidence["in_flight_requests_max"] == 1
        assert evidence["tenant_present"] is False
