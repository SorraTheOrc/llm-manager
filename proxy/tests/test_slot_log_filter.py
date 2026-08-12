"""Unit tests for the pure per-slot log filtering helpers.

Covers the slot-log relevance rule (LP-0MSHET5SI000LYSK):

- llama-server.log lines are attributed by ``id <N> |`` / ``id=<N>`` markers.
- proxy.log lines are attributed by ``session=<uuid>`` (primary) and
  ``slot=<n>`` (fallback) markers.
- Lines that cannot be attributed (``slot=none``, no markers, non-slot
  llama-server lines) are excluded from slot sections.
"""

import pytest
from proxy.slot_log_filter import (
    extract_llama_slot_id,
    extract_proxy_session_id,
    extract_proxy_slot_id,
    filter_log_lines_for_slot,
    line_matches_slot,
)

# Realistic llama-server.log line shapes observed on the ai machine.
LLAMA_SLOT2_UPDATE = "[57463] slot update_slots: id  2 | task 209403 | n_tokens = 16750, prompt_tokens = 120"
LLAMA_SLOT3_UPDATE = "[57463] slot update_slots: id  3 | task 209410 | n_tokens = 9000, prompt_tokens = 99"
LLAMA_SLOT2_TIMING = "slot print_timing: id  2 | task 209403 | prompt_per_second = 42.34"
LLAMA_SLOT2_LAUNCH = "slot launch_slot_: id  2 | task 209403 | n_past = 0"
LLAMA_SLOT2_RELEASE = "slot      release: id  2 | task 209403"
LLAMA_SLOT5_PROGRESS = "slot update_slots: id=5 n_tokens=4096 progress=0.17"
LLAMA_NON_SLOT = "srv  log_server_r: server is listening on port 8080"

# Realistic proxy.log line shapes.
SESSION_A = "11111111-2222-3333-4444-555555555555"
SESSION_B = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
PROXY_SLOT_SAVE_A = f"slot_save success session={SESSION_A} slot=2"
PROXY_LEASE_A = f"lease_renewed session={SESSION_A}"
PROXY_SLOT_SAVE_B = f"slot_save success session={SESSION_B} slot=3"
PROXY_UNMAPPED = "dispatch line session=00000000-0000-0000-0000-000000000000 slot=none"
PROXY_NO_MARKERS = "GET /v1/chat/completions 200 OK"


# ---------------------------------------------------------------------------
# extract_* helpers
# ---------------------------------------------------------------------------


class TestExtractHelpers:
    def test_extract_llama_slot_id_from_update_line(self):
        assert extract_llama_slot_id(LLAMA_SLOT2_UPDATE) == 2

    def test_extract_llama_slot_id_from_equals_form(self):
        assert extract_llama_slot_id(LLAMA_SLOT5_PROGRESS) == 5

    def test_extract_llama_slot_id_from_timing_line(self):
        assert extract_llama_slot_id(LLAMA_SLOT2_TIMING) == 2

    def test_extract_llama_slot_id_none_for_non_slot_line(self):
        assert extract_llama_slot_id(LLAMA_NON_SLOT) is None

    def test_extract_llama_slot_id_none_for_non_string(self):
        assert extract_llama_slot_id(None) is None
        assert extract_llama_slot_id(42) is None

    def test_extract_proxy_session_id(self):
        assert extract_proxy_session_id(PROXY_SLOT_SAVE_A) == SESSION_A
        assert extract_proxy_session_id(PROXY_LEASE_A) == SESSION_A

    def test_extract_proxy_session_id_none_when_absent(self):
        assert extract_proxy_session_id(PROXY_NO_MARKERS) is None
        assert extract_proxy_session_id("slot=none") is None

    def test_extract_proxy_slot_id(self):
        assert extract_proxy_slot_id(PROXY_SLOT_SAVE_A) == 2
        assert extract_proxy_slot_id(PROXY_SLOT_SAVE_B) == 3

    def test_extract_proxy_slot_id_none_for_unmapped(self):
        assert extract_proxy_slot_id(PROXY_UNMAPPED) is None
        assert extract_proxy_slot_id(PROXY_NO_MARKERS) is None


# ---------------------------------------------------------------------------
# line_matches_slot — llama-server.log rule (`id <N> |` / `id=<N>`)
# ---------------------------------------------------------------------------


class TestLlamaLineMatching:
    def test_positive_llama_update_line(self):
        assert line_matches_slot(LLAMA_SLOT2_UPDATE, 2, source="llama") is True

    def test_negative_llama_update_line_other_slot(self):
        assert line_matches_slot(LLAMA_SLOT2_UPDATE, 3, source="llama") is False

    def test_positive_llama_timing_line(self):
        assert line_matches_slot(LLAMA_SLOT2_TIMING, 2, source="llama") is True

    def test_positive_llama_launch_and_release_lines(self):
        assert line_matches_slot(LLAMA_SLOT2_LAUNCH, 2, source="llama") is True
        assert line_matches_slot(LLAMA_SLOT2_RELEASE, 2, source="llama") is True

    def test_positive_llama_equals_form(self):
        assert line_matches_slot(LLAMA_SLOT5_PROGRESS, 5, source="llama") is True

    def test_negative_non_slot_llama_line(self):
        assert line_matches_slot(LLAMA_NON_SLOT, 2, source="llama") is False

    def test_negative_empty_or_non_string(self):
        assert line_matches_slot("", 2, source="llama") is False
        assert line_matches_slot("   ", 2, source="llama") is False
        assert line_matches_slot(None, 2, source="llama") is False


# ---------------------------------------------------------------------------
# line_matches_slot — proxy.log rule (`session=<uuid>` primary, `slot=<n>` fallback)
# ---------------------------------------------------------------------------


class TestProxyLineMatching:
    def test_positive_proxy_session_match(self):
        assert line_matches_slot(PROXY_SLOT_SAVE_A, 2, session_id=SESSION_A, source="proxy") is True
        assert line_matches_slot(PROXY_LEASE_A, 2, session_id=SESSION_A, source="proxy") is True

    def test_positive_proxy_session_match_ignores_slot_id(self):
        # session is the primary key; even a line tagged slot=3 belongs to
        # the session that owns it.
        assert line_matches_slot(PROXY_SLOT_SAVE_B, 3, session_id=SESSION_B, source="proxy") is True

    def test_negative_proxy_different_session(self):
        assert line_matches_slot(PROXY_LEASE_A, 2, session_id=SESSION_B, source="proxy") is False

    def test_positive_proxy_slot_fallback_without_session(self):
        # No session_id known (e.g. stale mapping) — slot=<n> marker is used.
        assert line_matches_slot(PROXY_SLOT_SAVE_A, 2, session_id=None, source="proxy") is True

    def test_negative_proxy_slot_fallback_other_slot(self):
        assert line_matches_slot(PROXY_SLOT_SAVE_A, 3, session_id=None, source="proxy") is False

    def test_ambiguous_unmapped_slot_none_line(self):
        # slot=none / unknown session — must NOT be attributed to any slot.
        assert line_matches_slot(PROXY_UNMAPPED, 2, session_id=SESSION_A, source="proxy") is False

    def test_ambiguous_no_markers_line(self):
        assert line_matches_slot(PROXY_NO_MARKERS, 2, session_id=SESSION_A, source="proxy") is False


# ---------------------------------------------------------------------------
# line_matches_slot — auto source detection
# ---------------------------------------------------------------------------


class TestAutoSourceDetection:
    def test_auto_detects_llama_line(self):
        assert line_matches_slot(LLAMA_SLOT2_UPDATE, 2) is True

    def test_auto_detects_proxy_session_line(self):
        assert line_matches_slot(PROXY_LEASE_A, 2, session_id=SESSION_A) is True

    def test_auto_detects_proxy_slot_line(self):
        assert line_matches_slot(PROXY_SLOT_SAVE_A, 2) is True

    def test_auto_detection_of_unattributable_line(self):
        assert line_matches_slot(PROXY_NO_MARKERS, 2) is False
        assert line_matches_slot(LLAMA_NON_SLOT, 2) is False

    def test_llama_line_never_matches_proxy_rule(self):
        # A llama line must not be attributed via the proxy rule even when
        # an unrelated session id appears in the text.
        mixed = f"[57463] slot update_slots: id  2 | task 1 | session={SESSION_A}"
        assert line_matches_slot(mixed, 2) is True
        assert line_matches_slot(mixed, 2, session_id=SESSION_B) is True  # llama rule wins


# ---------------------------------------------------------------------------
# filter_log_lines_for_slot
# ---------------------------------------------------------------------------


class TestFilterLogLinesForSlot:
    def test_filters_mixed_llama_lines(self):
        lines = [LLAMA_SLOT2_UPDATE, LLAMA_SLOT3_UPDATE, LLAMA_NON_SLOT, LLAMA_SLOT2_RELEASE]
        kept = filter_log_lines_for_slot(lines, 2, source="llama")
        assert kept == [LLAMA_SLOT2_UPDATE, LLAMA_SLOT2_RELEASE]

    def test_filters_mixed_proxy_lines_by_session(self):
        lines = [PROXY_SLOT_SAVE_A, PROXY_LEASE_A, PROXY_SLOT_SAVE_B, PROXY_UNMAPPED, PROXY_NO_MARKERS]
        kept = filter_log_lines_for_slot(lines, 2, session_id=SESSION_A, source="proxy")
        assert kept == [PROXY_SLOT_SAVE_A, PROXY_LEASE_A]

    def test_returns_empty_for_no_match(self):
        assert filter_log_lines_for_slot([LLAMA_NON_SLOT], 2, source="llama") == []
        assert filter_log_lines_for_slot([], 2, source="llama") == []

    def test_handles_non_string_entries(self):
        lines = [LLAMA_SLOT2_UPDATE, None, 42, ""]
        assert filter_log_lines_for_slot(lines, 2, source="llama") == [LLAMA_SLOT2_UPDATE]

    def test_string_input_is_treated_as_single_line_list(self):
        # The helper accepts an iterable; a bare string is one line.
        assert filter_log_lines_for_slot(LLAMA_SLOT2_UPDATE, 2, source="llama") == [LLAMA_SLOT2_UPDATE]
