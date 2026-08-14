"""Live verification harness for per-slot details in /llama/local/status.

LP-0MSORPUMX002LLIA AC6: the status endpoint must let herdr's downtime
worker (ContextHub WL-0MSG7P9N8009PCKG) require the SAME N slots free for
the full idle threshold. This harness verifies the proxy-side enabler live
against a running proxy + llama-server:

  - the ``slots`` array reflects llama-server /slots state (shape AC1/AC4),
  - ``available_slots`` agrees with the number of non-processing slots
    (counts stay consistent with per-slot data, AC2),
  - slot_ids are stable identities across polls, so the SAME slots can be
    required free for a full window (AC6).

Skipped by default; run on demand against a live proxy:

    RUN_LIVE_SLOTS_VERIFY=1 pytest tests/test_slots_live_verify.py -v

The 2026-08-14 live run (new code on :8000, llama-server :8080 router-mode,
Qwen3 loaded, 2 active slots) passed: free slot_ids across 4 polls (15 s
window) = [[1],[1],[0,1],[0,1]]; intersection = [1] — slot 1 stayed free
for the whole window while slot 0 processed. Full evidence in
docs/per-slot-status-verification.md.
"""

import os

import pytest
import requests

pytestmark = [pytest.mark.integration, pytest.mark.live]

if os.getenv("RUN_LIVE_SLOTS_VERIFY", "0") not in ("1", "true", "yes"):
    pytest.skip(
        "live per-slot status verification is disabled; "
        "set RUN_LIVE_SLOTS_VERIFY=1 to run on demand",
        allow_module_level=True,
    )

LIVE_PROXY_URL = os.environ.get("LIVE_PROXY_BASE_URL", "http://localhost:8000")
STATUS_PATH = "/llama/local/status"
POLLS = int(os.environ.get("LIVE_SLOTS_POLLS", "4"))
POLL_INTERVAL_S = float(os.environ.get("LIVE_SLOTS_POLL_INTERVAL_S", "3.0"))


def _fetch_status() -> dict:
    resp = requests.get(f"{LIVE_PROXY_URL}{STATUS_PATH}", timeout=5)
    assert resp.status_code == 200
    return resp.json()


def test_slots_shape_and_counts_agree_with_per_slot_data():
    """AC1/AC2/AC4: slots array shape + counts consistent with per-slot data."""
    payload = _fetch_status()
    assert isinstance(payload.get("slots"), list)
    if not payload["llama_server_running"]:
        assert payload["slots"] == []  # AC2: empty when down
        return
    for slot in payload["slots"]:
        assert set(slot.keys()) == {"slot_id", "is_processing", "n_decoded"}
        assert isinstance(slot["slot_id"], int)
        assert isinstance(slot["is_processing"], bool)
        assert slot["n_decoded"] is None or isinstance(slot["n_decoded"], int)
    # When per-slot data is present, counts must agree with it (AC2: counts
    # behavior unchanged but consistent with the exposed slots).
    if payload["slots"]:
        free = sum(1 for s in payload["slots"] if not s["is_processing"])
        assert payload["available_slots"] == free
        assert payload["total_slots"] == len(payload["slots"])


def test_same_slots_stay_free_across_polls():
    """AC6: the SAME N slots can be required free for the full window.

    Each poll yields a set of free slot_ids; when at least one slot is free
    in every poll, the intersection must be non-empty — i.e. there exists a
    slot that stayed free for the entire window, which the counts-only
    ``available_slots`` cannot guarantee (the herdr same-slot requirement).
    """
    free_sets: list[list[int]] = []
    for i in range(POLLS):
        payload = _fetch_status()
        assert isinstance(payload.get("slots"), list)
        free = [s["slot_id"] for s in payload["slots"] if not s["is_processing"]]
        free_sets.append(free)
        if i < POLLS - 1:
            import time

            time.sleep(POLL_INTERVAL_S)

    assert len(free_sets) == POLLS
    every_poll_has_free = all(len(f) > 0 for f in free_sets)
    if every_poll_has_free:
        intersection = set(free_sets[0])
        for f in free_sets[1:]:
            intersection &= set(f)
        assert len(intersection) > 0, (
            f"every poll had >=1 free slot but no single slot stayed free: {free_sets}"
        )
