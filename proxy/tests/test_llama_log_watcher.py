"""Unit tests for the llama-server log watcher (unload_lru event monitoring)."""

import time
from datetime import datetime, timedelta
from pathlib import Path
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib
lifecycle = importlib.import_module("proxy.lifecycle")
from proxy.lifecycle import (
    _parse_unload_lru,
    _UnloadLruTracker,
    _check_unload_lru_threshold,
)


class TestParseUnloadLru:
    """Test parsing unload_lru events from llama-server log lines."""

    def test_parse_standard_line(self):
        """Standard unload_lru log line from llama-server."""
        line = "unload_lru: unloading model 'qwen3-8b' (slot 3)\n"
        result = _parse_unload_lru(line)
        assert result is True

    def test_parse_variant_format(self):
        """Variant unload_lru log formats."""
        line = "unload_lru: slot 2 evicted\n"
        result = _parse_unload_lru(line)
        assert result is True

    def test_parse_no_match(self):
        """Non-matching log line returns None."""
        line = "prompt processing, n_tokens=100, progress=0.50\n"
        result = _parse_unload_lru(line)
        assert result is None

    def test_parse_empty_line(self):
        """Empty log line returns None."""
        assert _parse_unload_lru("") is None

    def test_parse_malformed_binary(self):
        """Binary/malformed line doesn't crash."""
        assert _parse_unload_lru(b"\x00\x01\x02") is None


class TestUnloadLruTracker:
    """Test rolling-window counting of unload_lru events."""

    def test_tracker_empty_initial(self):
        """New tracker has zero events."""
        tracker = _UnloadLruTracker(window_minutes=5, threshold=3)
        assert tracker.count() == 0
        assert not _check_unload_lru_threshold(tracker)

    def test_tracker_single_event(self):
        """Single event is below threshold."""
        tracker = _UnloadLruTracker(window_minutes=5, threshold=3)
        tracker.record()
        assert tracker.count() == 1
        assert not _check_unload_lru_threshold(tracker)

    def test_tracker_exact_threshold(self):
        """Exactly threshold events in window triggers alert."""
        tracker = _UnloadLruTracker(window_minutes=5, threshold=3)
        tracker.record()
        tracker.record()
        tracker.record()
        assert tracker.count() == 3
        assert _check_unload_lru_threshold(tracker)

    def test_tracker_below_threshold(self):
        """Below threshold events does not trigger alert."""
        tracker = _UnloadLruTracker(window_minutes=5, threshold=3)
        tracker.record()
        tracker.record()
        assert tracker.count() == 2
        assert not _check_unload_lru_threshold(tracker)

    def test_tracker_above_threshold(self):
        """Above threshold events triggers alert."""
        tracker = _UnloadLruTracker(window_minutes=5, threshold=3)
        for _ in range(5):
            tracker.record()
        assert tracker.count() == 5
        assert _check_unload_lru_threshold(tracker)

    def test_events_expire_after_window(self):
        """Events older than the window are pruned."""
        from datetime import timedelta
        tracker = _UnloadLruTracker(window_minutes=5, threshold=3)
        now = datetime.now()

        # Directly inject old events into the tracker
        old_time = now - timedelta(minutes=6)
        tracker._events = [old_time, old_time, old_time]

        # Prune old events and check count
        tracker.prune()
        assert tracker.count() == 0
        assert not _check_unload_lru_threshold(tracker)

    def test_partial_expiry(self, monkeypatch):
        """Only events outside the window are pruned, recent ones remain."""
        tracker = _UnloadLruTracker(window_minutes=5, threshold=3)
        now = datetime.now()

        # Record 2 old events
        tracker._events = [now - timedelta(minutes=6), now - timedelta(minutes=6)]
        # Record 2 recent events
        tracker.record()
        tracker.record()

        tracker.prune()
        assert tracker.count() == 2

    def test_tracker_resets_after_alert(self):
        """Tracker resets after alert is triggered."""
        tracker = _UnloadLruTracker(window_minutes=5, threshold=3)
        for _ in range(3):
            tracker.record()
        assert _check_unload_lru_threshold(tracker)
        # After checking, alert state is recorded
        assert tracker.alerted is True

    def test_threshold_configurable(self):
        """Threshold and window are configurable."""
        tracker = _UnloadLruTracker(window_minutes=10, threshold=5)
        assert tracker.window_minutes == 10
        assert tracker.threshold == 5
        for _ in range(4):
            tracker.record()
        assert not _check_unload_lru_threshold(tracker)
        tracker.record()
        assert _check_unload_lru_threshold(tracker)

    def test_malformed_log_lines_ignored(self):
        """Malformed lines are ignored by the tracker."""
        tracker = _UnloadLruTracker(window_minutes=5, threshold=3)
        # Record valid events
        tracker.record()
        tracker.record()
        # Check count (should just be 2)
        assert tracker.count() == 2
        assert not _check_unload_lru_threshold(tracker)
        tracker.record()
        assert _check_unload_lru_threshold(tracker)
