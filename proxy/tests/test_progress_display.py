"""Tests for progress formatting and threshold-based logging."""

import logging
import re
import io
import threading
import sys
from proxy import server
from proxy import lifecycle


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
        lines_logged = [l for l in log_output.split("\n") if l.strip()]

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
        lines_logged = [l for l in log_output.split("\n") if l.strip()]

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
        lines_logged = [l for l in log_output.split("\n") if l.strip()]

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
        lines_logged = [l for l in log_output.split("\n") if l.strip()]

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
        lines_logged = [l for l in log_output.split("\n") if l.strip()]

        # 50% (start), 100% (final)
        assert len(lines_logged) == 2, f"Expected 2 log entries, got {len(lines_logged)}: {lines_logged}"
        assert "100%" in lines_logged[1], f"Final entry should contain 100%: {lines_logged[1]}"


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
