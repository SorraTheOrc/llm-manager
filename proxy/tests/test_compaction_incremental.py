"""
Tests for incremental summarization in proxy compaction (R5,
LP-0MTTPXIIX005Y0Z9).

Verifies:
- The proxy detects a previous compaction summary (the marker message the
  planner injects on compaction) in the session's message list.
- When a previous summary exists the summarizer is invoked with it and uses
  Pi's UPDATE_SUMMARIZATION_PROMPT (``<previous-summary>`` + merge rules).
- Without a previous summary the summarizer uses the standard full
  SUMMARIZATION_PROMPT and receives ``previous_summary=None``.
- A second compaction pass folds the new middle turns into the previous
  summary and replaces the marker (summary lifecycle: create -> update).
"""

from unittest.mock import MagicMock, patch

import pytest
from proxy.compaction import (
    _summary_message,
    extract_previous_summary,
    plan_session_compaction,
)
from proxy.provider import (
    _SUMMARIZATION_PROMPT,
    _UPDATE_SUMMARIZATION_PROMPT,
)


def _session_large_enough():
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "FIRST"}]
    for i in range(60):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


def _cfg():
    return {
        "server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
            "compaction_trigger_ratio": 0.70,
        }
    }


def _big_estimator(m):
    return 1000 * len(m)


def _capturing_summarizer(captured):
    """Summarizer recording (middle_messages, previous_summary) per call."""

    def summarizer(middle_messages, previous_summary=None):
        captured.append((middle_messages, previous_summary))
        return "SPY SUMMARY"

    return summarizer


class TestPreviousSummaryDetection:
    def test_no_marker_returns_none(self):
        assert extract_previous_summary([{"role": "user", "content": "hi"}]) is None

    def test_marker_message_returns_inner_text(self):
        msgs = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "FIRST"},
            _summary_message("SUMMARY ONE"),
        ]
        assert extract_previous_summary(msgs) == "SUMMARY ONE"

    def test_last_marker_wins(self):
        msgs = [
            {"role": "user", "content": "FIRST"},
            _summary_message("SUMMARY ONE"),
            {"role": "user", "content": "q"},
            _summary_message("SUMMARY TWO"),
        ]
        assert extract_previous_summary(msgs) == "SUMMARY TWO"

    def test_empty_marker_ignored(self):
        msgs = [
            {"role": "user", "content": "FIRST"},
            _summary_message(""),
        ]
        assert extract_previous_summary(msgs) is None


class TestPromptSelection:
    def test_full_prompt_when_no_previous_summary(self):
        """No marker in session -> previous_summary=None and full template."""
        captured = []
        summarizer = _capturing_summarizer(captured)
        result = plan_session_compaction(
            _session_large_enough(),
            _cfg(),
            "fast",
            summarizer=summarizer,
            estimate_tokens=_big_estimator,
        )
        assert result["action"] == "compact"
        middle, previous_summary = captured[0]
        assert previous_summary is None
        assert result["previous_summary"] is None

    def test_update_prompt_when_previous_summary_exists(self):
        """Marker in session -> previous text passed to the summarizer."""
        captured = []
        summarizer = _capturing_summarizer(captured)
        session = _session_large_enough()
        session.insert(2, _summary_message("PREVIOUS SUMMARY"))
        result = plan_session_compaction(
            session,
            _cfg(),
            "fast",
            summarizer=summarizer,
            estimate_tokens=_big_estimator,
        )
        assert result["action"] == "compact"
        middle, previous_summary = captured[0]
        assert previous_summary == "PREVIOUS SUMMARY"
        # The marker message carrying the old summary is NOT re-summarised
        assert all("PREVIOUS SUMMARY" not in str(m.get("content", "")) for m in middle)


class TestSummarizerRequestShape:
    """build_local_summarizer shapes the user message per mode."""

    def _capture_body(self, middle_messages, previous_summary=None):
        with patch("proxy.compaction_summarizer.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"choices": [{"message": {"content": "S"}}]}
            mock_client.post.return_value = resp
            from proxy.compaction_summarizer import build_local_summarizer

            summarizer = build_local_summarizer({}, llama_port=8080)
            assert summarizer(middle_messages, previous_summary) == "S"
            args, kwargs = mock_client.post.call_args
            return kwargs.get("json") or args[1]

    def test_full_mode_no_previous_summary_tag(self):
        body = self._capture_body([{"role": "user", "content": "hi"}])
        user = next(m for m in body["messages"] if m["role"] == "user")["content"]
        assert "<previous-summary>" not in user
        assert user.endswith(_SUMMARIZATION_PROMPT)

    def test_incremental_mode_embeds_previous_summary_and_update_prompt(self):
        body = self._capture_body([{"role": "user", "content": "new work"}], "OLD CHECKPOINT")
        user = next(m for m in body["messages"] if m["role"] == "user")["content"]
        assert "<previous-summary>\nOLD CHECKPOINT\n</previous-summary>" in user
        assert user.endswith(_UPDATE_SUMMARIZATION_PROMPT)
        # Pi's merge rules present in the incremental template
        assert "PRESERVE all existing information from the previous summary" in _UPDATE_SUMMARIZATION_PROMPT

    def test_update_template_keeps_structured_sections(self):
        """AC3: the incremental prompt preserves the structured sections."""
        for section in (
            "## Goal",
            "## Constraints & Preferences",
            "## Progress",
            "## Key Decisions",
            "## Next Steps",
            "## Critical Context",
        ):
            assert section in _UPDATE_SUMMARIZATION_PROMPT


class TestTwoPhaseLifecycle:
    """Compaction #1 creates a marker; compaction #2 updates it (AC8)."""

    def test_second_compaction_uses_previous_summary(self):
        captured = []
        summarizer = _capturing_summarizer(captured)

        # Phase 1: full compaction on a fresh (marker-free) session
        first = plan_session_compaction(
            _session_large_enough(),
            _cfg(),
            "fast",
            summarizer=summarizer,
            estimate_tokens=_big_estimator,
        )
        assert first["action"] == "compact"
        assert len(captured) == 1
        assert captured[0][1] is None

        # Phase 2: more work accumulated past the trigger; the compacted
        # history (now marker-bearing) is evaluated again
        compacted = first["messages"]
        grown = list(compacted)
        for i in range(60):
            grown.append({"role": "user", "content": f"new-q{i}"})
            grown.append({"role": "assistant", "content": f"new-a{i}"})
        second = plan_session_compaction(
            grown,
            _cfg(),
            "fast",
            summarizer=summarizer,
            estimate_tokens=_big_estimator,
        )
        assert second["action"] == "compact"
        assert len(captured) == 2
        _, previous_summary = captured[1]
        assert previous_summary == "SPY SUMMARY"
        assert second["summary_text"] == "SPY SUMMARY"
        # The compacted history still ends with a single fresh marker
        markers = [
            m
            for m in second["messages"]
            if isinstance(m.get("content"), str)
            and m["content"].startswith("The conversation history before this point was compacted")
        ]
        assert len(markers) == 1
        assert "SPY SUMMARY" in markers[0]["content"]
