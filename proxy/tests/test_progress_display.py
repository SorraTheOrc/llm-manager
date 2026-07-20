"""Tests for progress formatting and threshold-based logging."""

import io
import logging
import sys

from proxy import lifecycle, server

# ---------------------------------------------------------------------------
# format_progress tests – clean log-friendly output (no ANSI, no \r)
# ---------------------------------------------------------------------------


def test_format_progress_basic():
    s = server.format_progress(26988, 40494, 0.658083, model_name="Qwen3", slot_id=2, tokens_per_sec=45.2)
    assert "Processing 26988/40494 tokens (65%)" in s
    assert "[slot:2 Qwen3]" in s, "model/slot prefix must be present"
    assert "@ 45.2 tok/s" in s, "tokens-per-second must be present"
    assert not s.startswith("\r"), "progress string must NOT start with carriage return"
    assert "\x1b" not in s, "ANSI escape codes must NOT be present"


def test_format_progress_zero_tokens():
    s = server.format_progress(0, 1000, 0.0, model_name="gemma4", slot_id=0, tokens_per_sec=None)
    assert "Processing 0/1000 tokens (0%)" in s
    assert "[slot:0 gemma4]" in s
    assert "@ --.- tok/s" in s, "None TPS should show placeholder"


def test_format_progress_100_percent():
    s = server.format_progress(50000, 50000, 1.0, model_name="Qwen3", slot_id=3, tokens_per_sec=123.4)
    assert "(100%)" in s
    assert "[slot:3 Qwen3]" in s
    assert "@ 123.4 tok/s" in s


def test_format_progress_percentage_truncation():
    # 0.659 -> 65%
    s = server.format_progress(10, 1000, 0.659, model_name="default", slot_id=1)
    assert "(65%)" in s


def test_format_progress_no_ansi():
    """Ensure no ANSI escape codes or carriage returns are present."""
    s = server.format_progress(1, 2, 0.5, model_name="test", slot_id=1, tokens_per_sec=99.9)
    assert "\x1b" not in s, "No ANSI escape codes"
    assert "\r" not in s, "No carriage return"


def test_format_progress_default_model_name():
    """When model_name is not provided, defaults to 'unknown'."""
    s = server.format_progress(10, 100, 0.5)
    assert "[slot:0 unknown]" in s


def test_format_progress_tps_one_decimal():
    """TPS should be formatted to one decimal place."""
    s = server.format_progress(100, 200, 0.5, model_name="m", slot_id=1, tokens_per_sec=45.678)
    assert "@ 45.7 tok/s" in s


def test_format_progress_tps_whole_number():
    """Whole number TPS should still show one decimal."""
    s = server.format_progress(100, 200, 0.5, model_name="m", slot_id=1, tokens_per_sec=50.0)
    assert "@ 50.0 tok/s" in s


def test_format_progress_tps_very_small():
    """Very small TPS should show properly."""
    s = server.format_progress(1, 200, 0.005, model_name="slow", slot_id=0, tokens_per_sec=0.3)
    assert "@ 0.3 tok/s" in s


def test_format_progress_tps_none_when_elapsed_zero():
    """When tokens_per_sec is None (first progress update), show placeholder."""
    s = server.format_progress(50, 500, 0.1, model_name="Qwen3", slot_id=1, tokens_per_sec=None)
    assert "@ --.- tok/s" in s


def test_format_progress_clean_output():
    """Output should be plain text without terminal control characters."""
    s = server.format_progress(100, 200, 0.5, model_name="Qwen3", slot_id=1, tokens_per_sec=45.2)
    # Should be a simple, clean line
    assert s == "[slot:1 Qwen3] Processing 100/200 tokens (50%) @ 45.2 tok/s"


# ---------------------------------------------------------------------------
# extract_progress_data tests – should remain unchanged
# ---------------------------------------------------------------------------


def test_extract_progress_data_basic():
    from proxy.handlers import extract_progress_data
    line = "slot 3 : prompt processing, n_tokens=100, progress=0.50"
    parsed = extract_progress_data(line)
    assert parsed is not None
    slot_id, n_tokens, progress = parsed
    assert slot_id == 3
    assert n_tokens == 100
    assert progress == 0.50


def test_extract_progress_data_invalid():
    from proxy.handlers import extract_progress_data
    assert extract_progress_data(None) is None
    assert extract_progress_data("") is None
    assert extract_progress_data("not a progress line") is None


# ---------------------------------------------------------------------------
# Progress threshold-tracking logic tests
# ---------------------------------------------------------------------------


def _make_mock_progress_line(slot_id: int, n_tokens: int, progress: float) -> str:
    """Build a simulated llama-server stdout line for a given progress value."""
    return (
        f"slot {slot_id} : prompt processing, n_tokens={n_tokens}, progress={progress}\n"
    )


class TestProgressThresholdTracking:
    """Tests for the per-slot progress threshold tracking in _stream_output."""

    def test_logs_at_start_and_each_threshold(self):
        """Verifies progress entries are logged at start and at each 10% milestone."""
        logger = logging.getLogger("test_progress_threshold")
        logger.setLevel(logging.INFO)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.handlers.clear()
        logger.addHandler(handler)

        dst = io.StringIO()

        # Simulate a slot progressing through thresholds
        # Build a buffer with progress values: start (0.0), 10%, 25%, 50%, 75%, 100%
        lines = [
            "slot 1 : prompt processing, n_tokens=0, progress=0.00\n",    # start (0%) -> log
            "slot 1 : prompt processing, n_tokens=50, progress=0.05\n",   # 5% -> skip (no threshold crossed)
            "slot 1 : prompt processing, n_tokens=100, progress=0.10\n",  # 10% -> log
            "slot 1 : prompt processing, n_tokens=250, progress=0.25\n",  # 25% -> log (20% threshold)
            "slot 1 : prompt processing, n_tokens=350, progress=0.35\n",  # 35% -> log (30%)
            "slot 1 : prompt processing, n_tokens=500, progress=0.50\n",  # 50% -> log
            "slot 1 : prompt processing, n_tokens=750, progress=0.75\n",  # 75% -> log (70%)
            "slot 1 : prompt processing, n_tokens=1000, progress=1.00\n", # 100% -> log
        ]
        src = io.StringIO("".join(lines))

        # Mock time.monotonic to return increasing values so tokens_per_sec is computed
        fake_time = [10.0, 10.5, 12.0, 14.0, 17.0, 21.0, 26.0, 32.0]
        time_index = [0]
        original_monotonic = lifecycle.time.monotonic

        def fake_monotonic():
            t = fake_time[time_index[0] % len(fake_time)]
            time_index[0] += 1
            return t

        lifecycle.time.monotonic = fake_monotonic
        try:
            lifecycle._stream_output(src, dst, model_name="Qwen3", logger=logger)
        finally:
            lifecycle.time.monotonic = original_monotonic

        log_output = buf.getvalue()
        lines_logged = [line for line in log_output.split("\n") if line.strip()]

        # Should have logged at 0%, 10%, 20%, 30%, 50%, 70%, 100% (thresholds crossed)
        # 0%: progress=0.00
        # 10%: progress=0.10
        # 25%: crosses 20% threshold
        # 35%: crosses 30% threshold
        # 50%: crosses 50% threshold
        # 75%: crosses 70% threshold
        # 100%: progress=1.00
        assert len(lines_logged) >= 7, f"Expected >=7 log entries, got {len(lines_logged)}: {lines_logged}"

        # Verify key milestones appear
        all_text = " ".join(lines_logged)
        assert "0%" in all_text or "(0%)" in all_text
        assert "100%" in all_text or "(100%)" in all_text

    def test_logs_only_at_threshold_boundaries(self):
        """Multiple lines within the same 10% bucket should only log once."""
        logger = logging.getLogger("test_threshold_dedup")
        logger.setLevel(logging.INFO)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.handlers.clear()
        logger.addHandler(handler)

        dst = io.StringIO()

        # Many lines at 33-35% (all in the 30% threshold), only first should log
        lines = [
            "slot 1 : prompt processing, n_tokens=33, progress=0.33\n",  # crosses 30% -> log
            "slot 1 : prompt processing, n_tokens=34, progress=0.34\n",  # same threshold -> skip
            "slot 1 : prompt processing, n_tokens=35, progress=0.35\n",  # same threshold -> skip
        ]
        src = io.StringIO("".join(lines))

        fake_time = [10.0, 10.5, 11.0]
        time_index = [0]
        original_monotonic = lifecycle.time.monotonic

        def fake_monotonic():
            t = fake_time[time_index[0] % len(fake_time)]
            time_index[0] += 1
            return t

        lifecycle.time.monotonic = fake_monotonic
        try:
            lifecycle._stream_output(src, dst, model_name="Qwen3", logger=logger)
        finally:
            lifecycle.time.monotonic = original_monotonic

        log_output = buf.getvalue()
        lines_logged = [line for line in log_output.split("\n") if line.strip()]

        # Only 1 progress log expected (start at 0% threshold, then 30%)
        # Actually need to think about this: first line at 33% crosses 0 (start) and 30%
        # Wait, with the threshold tracking, when slot_id not seen before, we log
        # the current pct threshold as start. Then when pct moves to a new threshold, we log again.
        #
        # Slot 1 first seen at 33%: log (threshold 0 recorded, no that's wrong)
        # Let me reconsider the logic.
        #
        # The threshold tracking should be:
        # - On first data for slot: compute current threshold, log, record it
        # - On subsequent data: compute current threshold, if different from recorded, log and update
        #
        # first line at 33%: threshold = int(33/10)*10 = 30 -> log "30%", record threshold=30
        # second line at 34%: threshold = 30, same as recorded -> skip
        # third line at 35%: threshold = 30, same -> skip
        #
        # So 1 progress log entry expected (30%)
        assert len(lines_logged) == 1, f"Expected 1 log entry, got {len(lines_logged)}: {lines_logged}"
        assert "33" in lines_logged[0] or "Processing 33" in lines_logged[0]

    def test_separate_slots_tracked_independently(self):
        """Each slot has its own progress threshold tracking."""
        logger = logging.getLogger("test_separate_slots")
        logger.setLevel(logging.INFO)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.handlers.clear()
        logger.addHandler(handler)

        dst = io.StringIO()

        lines = [
            "slot 1 : prompt processing, n_tokens=10, progress=0.10\n",  # slot 1 at 10% -> log
            "slot 2 : prompt processing, n_tokens=20, progress=0.20\n",  # slot 2 at 20% -> log
            "slot 1 : prompt processing, n_tokens=20, progress=0.20\n",  # slot 1 at 20% -> log (crossed 10->20)
            "slot 2 : prompt processing, n_tokens=30, progress=0.30\n",  # slot 2 at 30% -> log (crossed 20->30)
        ]
        src = io.StringIO("".join(lines))

        fake_time = [10.0, 10.5, 11.0, 11.5]
        time_index = [0]
        original_monotonic = lifecycle.time.monotonic

        def fake_monotonic():
            t = fake_time[time_index[0] % len(fake_time)]
            time_index[0] += 1
            return t

        lifecycle.time.monotonic = fake_monotonic
        try:
            lifecycle._stream_output(src, dst, model_name="Qwen3", logger=logger)
        finally:
            lifecycle.time.monotonic = original_monotonic

        log_output = buf.getvalue()
        lines_logged = [line for line in log_output.split("\n") if line.strip()]

        # Each line should produce a log entry (each crosses a new threshold for its slot)
        # slot 1: starts at 10% (first seen, log as 10%), then 20% (crosses 20)
        # slot 2: starts at 20% (first seen, log as 20%), then 30% (crosses 30)
        assert len(lines_logged) == 4, f"Expected 4 log entries, got {len(lines_logged)}: {lines_logged}"

    def test_multiple_slots_independent_tracking(self):
        """Multiple slots can be tracked simultaneously."""
        logger = logging.getLogger("test_multiple_slots")
        logger.setLevel(logging.INFO)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.handlers.clear()
        logger.addHandler(handler)

        dst = io.StringIO()

        lines = [
            "slot 1 : prompt processing, n_tokens=0, progress=0.00\n",
            "slot 5 : prompt processing, n_tokens=0, progress=0.00\n",
            "slot 1 : prompt processing, n_tokens=50, progress=0.50\n",
        ]
        src = io.StringIO("".join(lines))

        fake_time = [10.0, 10.5, 11.0]
        time_index = [0]
        original_monotonic = lifecycle.time.monotonic

        def fake_monotonic():
            t = fake_time[time_index[0] % len(fake_time)]
            time_index[0] += 1
            return t

        lifecycle.time.monotonic = fake_monotonic
        try:
            lifecycle._stream_output(src, dst, model_name="Qwen3", logger=logger)
        finally:
            lifecycle.time.monotonic = original_monotonic

        log_output = buf.getvalue()
        lines_logged = [line for line in log_output.split("\n") if line.strip()]

        # slot 1: 0% (start), 50% -> 2 entries
        # slot 5: 0% (start) -> 1 entry
        assert len(lines_logged) == 3, f"Expected 3 log entries, got {len(lines_logged)}: {lines_logged}"

    def test_final_entry_at_100_percent(self):
        """When progress reaches 1.0, a final log entry is recorded."""
        logger = logging.getLogger("test_final_100")
        logger.setLevel(logging.INFO)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.handlers.clear()
        logger.addHandler(handler)

        dst = io.StringIO()

        lines = [
            "slot 1 : prompt processing, n_tokens=50, progress=0.50\n",
            "slot 1 : prompt processing, n_tokens=100, progress=1.00\n",
        ]
        src = io.StringIO("".join(lines))

        fake_time = [10.0, 11.0]
        time_index = [0]
        original_monotonic = lifecycle.time.monotonic

        def fake_monotonic():
            t = fake_time[time_index[0] % len(fake_time)]
            time_index[0] += 1
            return t

        lifecycle.time.monotonic = fake_monotonic
        try:
            lifecycle._stream_output(src, dst, model_name="Qwen3", logger=logger)
        finally:
            lifecycle.time.monotonic = original_monotonic

        log_output = buf.getvalue()
        lines_logged = [line for line in log_output.split("\n") if line.strip()]

        # 50% (start), 100% (final)
        assert len(lines_logged) == 2, f"Expected 2 log entries, got {len(lines_logged)}: {lines_logged}"
        assert "100%" in lines_logged[1], f"Final entry should contain 100%: {lines_logged[1]}"

    def test_interleaved_multi_slot_progress(self):
        """Interleaved slot 0/slot 1 progress is logged independently for each slot.

        Both slots progress through 0%, 10%, 50%, 100% with interleaved lines.
        Each slot should independently log at its own threshold boundaries.
        """
        logger = logging.getLogger("test_interleaved_multi")
        logger.setLevel(logging.INFO)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.handlers.clear()
        logger.addHandler(handler)

        dst = io.StringIO()

        # Interleave slot 0 and slot 1 progress at 0%, 10%, 50%, 100%
        lines = [
            "slot 0 : prompt processing, n_tokens=0, progress=0.00\n",   # slot 0 start -> log (0%)
            "slot 1 : prompt processing, n_tokens=0, progress=0.00\n",   # slot 1 start -> log (0%)
            "slot 0 : prompt processing, n_tokens=10, progress=0.10\n",  # slot 0 10% -> log
            "slot 1 : prompt processing, n_tokens=10, progress=0.10\n",  # slot 1 10% -> log
            "slot 0 : prompt processing, n_tokens=50, progress=0.50\n",  # slot 0 50% -> log
            "slot 1 : prompt processing, n_tokens=50, progress=0.50\n",  # slot 1 50% -> log
            "slot 0 : prompt processing, n_tokens=100, progress=1.00\n", # slot 0 100% -> log
            "slot 1 : prompt processing, n_tokens=100, progress=1.00\n", # slot 1 100% -> log
        ]
        src = io.StringIO("".join(lines))

        fake_time = [10.0, 10.1, 10.5, 10.6, 12.0, 12.1, 15.0, 15.1]
        time_index = [0]
        original_monotonic = lifecycle.time.monotonic

        def fake_monotonic():
            t = fake_time[time_index[0] % len(fake_time)]
            time_index[0] += 1
            return t

        lifecycle.time.monotonic = fake_monotonic
        try:
            lifecycle._stream_output(src, dst, model_name="Qwen3", logger=logger)
        finally:
            lifecycle.time.monotonic = original_monotonic

        log_output = buf.getvalue()
        lines_logged = [line for line in log_output.split("\n") if line.strip()]

        # slot 0: 0%, 10%, 50%, 100% -> 4 entries
        # slot 1: 0%, 10%, 50%, 100% -> 4 entries
        # Total: 8 entries
        assert len(lines_logged) == 8, (
            f"Expected 8 log entries (4 per slot), got {len(lines_logged)}: {lines_logged}"
        )

        # Verify slot 0 and slot 1 are each represented
        slot0_entries = [line for line in lines_logged if "[slot:0" in line]
        slot1_entries = [line for line in lines_logged if "[slot:1" in line]
        assert len(slot0_entries) == 4, f"Expected 4 slot 0 entries, got {len(slot0_entries)}: {slot0_entries}"
        assert len(slot1_entries) == 4, f"Expected 4 slot 1 entries, got {len(slot1_entries)}: {slot1_entries}"

        # Verify milestones are present for both slots
        assert any("0%" in entry for entry in slot0_entries), f"Slot 0 should have 0% entry: {slot0_entries}"
        assert any("100%" in entry for entry in slot0_entries), f"Slot 0 should have 100% entry: {slot0_entries}"
        assert any("0%" in entry for entry in slot1_entries), f"Slot 1 should have 0% entry: {slot1_entries}"
        assert any("100%" in entry for entry in slot1_entries), f"Slot 1 should have 100% entry: {slot1_entries}"

    def test_different_completion_speeds(self):
        """Slots completing at different speeds are independently tracked.

        Slot 1 finishes (100%) while slot 0 is still at 50%.
        Slot 0 later catches up to 100%.
        """
        logger = logging.getLogger("test_diff_speeds")
        logger.setLevel(logging.INFO)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.handlers.clear()
        logger.addHandler(handler)

        dst = io.StringIO()

        lines = [
            "slot 0 : prompt processing, n_tokens=0, progress=0.00\n",   # slot 0 start -> log
            "slot 1 : prompt processing, n_tokens=0, progress=0.00\n",   # slot 1 start -> log
            "slot 0 : prompt processing, n_tokens=25, progress=0.25\n",  # slot 0 25% -> log (crosses 20%)
            "slot 1 : prompt processing, n_tokens=50, progress=0.50\n",  # slot 1 50% -> log
            "slot 1 : prompt processing, n_tokens=100, progress=1.00\n", # slot 1 100% -> log (finishes first)
            "slot 0 : prompt processing, n_tokens=50, progress=0.50\n",  # slot 0 50% -> log
            "slot 0 : prompt processing, n_tokens=100, progress=1.00\n", # slot 0 100% -> log (finishes later)
        ]
        src = io.StringIO("".join(lines))

        fake_time = [10.0, 10.1, 10.5, 11.0, 12.0, 13.0, 15.0]
        time_index = [0]
        original_monotonic = lifecycle.time.monotonic

        def fake_monotonic():
            t = fake_time[time_index[0] % len(fake_time)]
            time_index[0] += 1
            return t

        lifecycle.time.monotonic = fake_monotonic
        try:
            lifecycle._stream_output(src, dst, model_name="Qwen3", logger=logger)
        finally:
            lifecycle.time.monotonic = original_monotonic

        log_output = buf.getvalue()
        lines_logged = [line for line in log_output.split("\n") if line.strip()]

        # slot 0: 0% (start), 25% (crosses 20%), 50%, 100% -> 4 entries
        # slot 1: 0% (start), 50%, 100% -> 3 entries
        # Total: 7 entries
        assert len(lines_logged) == 7, (
            f"Expected 7 log entries, got {len(lines_logged)}: {lines_logged}"
        )

        slot0_entries = [line for line in lines_logged if "[slot:0" in line]
        slot1_entries = [line for line in lines_logged if "[slot:1" in line]
        assert len(slot0_entries) == 4, f"Expected 4 slot 0 entries, got {len(slot0_entries)}: {slot0_entries}"
        assert len(slot1_entries) == 3, f"Expected 3 slot 1 entries, got {len(slot1_entries)}: {slot1_entries}"

        # Slot 1 should finish before slot 0
        _slot1_indices = [i for i, line in enumerate(lines_logged) if "[slot:1" in line]
        slot0_100_idx = next(i for i, line in enumerate(lines_logged) if "[slot:0" in line and "100%" in line)
        slot1_100_idx = next(i for i, line in enumerate(lines_logged) if "[slot:1" in line and "100%" in line)
        assert slot1_100_idx < slot0_100_idx, (
            f"Slot 1 should reach 100% before slot 0: idx {slot1_100_idx} vs {slot0_100_idx}"
        )

    def test_non_consecutive_slot_numbers(self):
        """Non-consecutive slot numbers (e.g., 3 and 7) are tracked independently."""
        logger = logging.getLogger("test_non_consecutive")
        logger.setLevel(logging.INFO)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.handlers.clear()
        logger.addHandler(handler)

        dst = io.StringIO()

        lines = [
            "slot 3 : prompt processing, n_tokens=0, progress=0.00\n",    # slot 3 start -> log
            "slot 7 : prompt processing, n_tokens=0, progress=0.00\n",    # slot 7 start -> log
            "slot 3 : prompt processing, n_tokens=30, progress=0.30\n",   # slot 3 30% -> log (crosses 30%)
            "slot 7 : prompt processing, n_tokens=70, progress=0.70\n",   # slot 7 70% -> log (crosses 70%)
            "slot 3 : prompt processing, n_tokens=100, progress=1.00\n",  # slot 3 100% -> log
            "slot 7 : prompt processing, n_tokens=100, progress=1.00\n",  # slot 7 100% -> log
        ]
        src = io.StringIO("".join(lines))

        fake_time = [10.0, 10.1, 10.5, 10.6, 12.0, 12.1]
        time_index = [0]
        original_monotonic = lifecycle.time.monotonic

        def fake_monotonic():
            t = fake_time[time_index[0] % len(fake_time)]
            time_index[0] += 1
            return t

        lifecycle.time.monotonic = fake_monotonic
        try:
            lifecycle._stream_output(src, dst, model_name="Qwen3", logger=logger)
        finally:
            lifecycle.time.monotonic = original_monotonic

        log_output = buf.getvalue()
        lines_logged = [line for line in log_output.split("\n") if line.strip()]

        # slot 3: 0%, 30%, 100% -> 3 entries
        # slot 7: 0%, 70%, 100% -> 3 entries
        # Total: 6 entries
        assert len(lines_logged) == 6, (
            f"Expected 6 log entries, got {len(lines_logged)}: {lines_logged}"
        )

        slot3_entries = [line for line in lines_logged if "[slot:3" in line]
        slot7_entries = [line for line in lines_logged if "[slot:7" in line]
        assert len(slot3_entries) == 3, f"Expected 3 slot 3 entries, got {len(slot3_entries)}: {slot3_entries}"
        assert len(slot7_entries) == 3, f"Expected 3 slot 7 entries, got {len(slot7_entries)}: {slot7_entries}"

        # Verify milestones
        assert any("0%" in entry for entry in slot3_entries), f"Slot 3 should have 0% entry: {slot3_entries}"
        assert any("100%" in entry for entry in slot3_entries), f"Slot 3 should have 100% entry: {slot3_entries}"
        assert any("0%" in entry for entry in slot7_entries), f"Slot 7 should have 0% entry: {slot7_entries}"
        assert any("100%" in entry for entry in slot7_entries), f"Slot 7 should have 100% entry: {slot7_entries}"


# ---------------------------------------------------------------------------
# _stream_output integration tests (logger vs stderr behavior)
# ---------------------------------------------------------------------------


class TestStreamOutputLogging:
    """Tests that _stream_output uses logger.info() instead of sys.stderr."""

    def test_stream_output_logs_instead_of_stderr(self, monkeypatch):
        """_stream_output should log progress via logger, not write to stderr."""
        stderr_writes = []

        def fake_stderr_write(msg):
            stderr_writes.append(msg)

        monkeypatch.setattr(sys.stderr, "write", fake_stderr_write)

        logger = logging.getLogger("test_stderr_replaced")
        logger.setLevel(logging.INFO)
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.handlers.clear()
        logger.addHandler(handler)

        dst = io.StringIO()

        lines = [
            "slot 1 : prompt processing, n_tokens=100, progress=0.50\n",
            "not a progress line\n",
            "slot 2 : prompt processing, n_tokens=200, progress=1.00\n",
        ]
        src = io.StringIO("".join(lines))

        fake_time = [10.0, 10.5, 11.0]
        time_index = [0]
        original_monotonic = lifecycle.time.monotonic

        def fake_monotonic():
            t = fake_time[time_index[0] % len(fake_time)]
            time_index[0] += 1
            return t

        lifecycle.time.monotonic = fake_monotonic
        try:
            lifecycle._stream_output(src, dst, model_name="Qwen3", logger=logger)
        finally:
            lifecycle.time.monotonic = original_monotonic

        log_output = buf.getvalue()

        # Should NOT have written to stderr (no progress-related writes)
        progress_stderr = [w for w in stderr_writes if "slot" in w or "Processing" in w or "tok/s" in w]
        assert len(progress_stderr) == 0, f"Progress data should not go to stderr: {progress_stderr}"

        # Should have logged to the logger
        assert "Processing" in log_output, f"Expected 'Processing' in log output: {log_output}"
        assert "[slot:1 Qwen3]" in log_output
        assert "[slot:2 Qwen3]" in log_output
        assert "tok/s" in log_output
