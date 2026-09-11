"""
Summarizer fail-open must never produce an empty, content-destroying
"successful" compaction (LP-0MTXGU8T00066WVH).

Regression coverage for the bug where ``build_local_summarizer`` timed out
and returned ``""`` while ``plan_session_compaction`` still reported
``action="compact"`` / ``applied=True`` — dropping the middle turns with an
empty marker that ``extract_previous_summary`` could not detect, so
compaction never converged.

Verifies:
- AC1: empty / whitespace summaries are never applied; the session history
  is returned untouched and the action is ``remote_with_guidance``.
- AC2: the preventing failure is logged at WARNING with the session id, the
  failure kind and the attempt count.
- AC3: after a *successful* compaction, a subsequent compaction detects the
  injected summary and takes the incremental UPDATE path.
- AC4: a summarizer that returns ``""`` and one that raises/times out are
  both covered.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from proxy.compaction import (
    EmptySummary,
    decide_session_compaction,
    extract_previous_summary,
    log_compaction_event,
    plan_session_compaction,
)

_EST_PER_MESSAGE = 1000


def counting_estimator(messages) -> int:
    return _EST_PER_MESSAGE * len(messages) + sum(
        len(str(m.get("content", ""))) for m in messages
    )


def fast_config() -> dict:
    return {
        "server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
            "compaction_trigger_ratio": 0.70,
        }
    }


def make_session(num_turns: int) -> list[dict]:
    msgs = [{"role": "system", "content": "SYSTEM_PROMPT"}]
    msgs.append({"role": "user", "content": "FIRST_USER_PROMPT"})
    msgs.append({"role": "assistant", "content": "FIRST_ASSISTANT_REPLY"})
    for i in range(1, num_turns + 1):
        msgs.append({"role": "user", "content": f"user_q_{i}"})
        msgs.append({"role": "assistant", "content": f"assistant_a_{i}"})
    return msgs


def _plan(messages, summarizer):
    return plan_session_compaction(
        messages,
        fast_config(),
        "fast",
        summarizer=summarizer,
        estimate_tokens=counting_estimator,
    )


# ===================================================================
# AC1 / AC4 — blank or raising summarizer never drops turns
# ===================================================================


class TestEmptySummaryNeverApplied:
    def test_empty_string_summary_is_not_compacted(self):
        messages = make_session(60)  # ~124K > 58.3K trigger
        result = _plan(messages, lambda m, p=None: "")
        assert result["action"] == "remote_with_guidance"
        assert result["reason"] == "summarizer_failed"
        assert result["messages"] is messages  # untouched
        assert result["summary_text"] is None
        assert result["estimated_after"] == result["estimated_before"]

    def test_whitespace_summary_is_not_compacted(self):
        messages = make_session(60)
        result = _plan(messages, lambda m, p=None: "   \n\t  ")
        assert result["action"] == "remote_with_guidance"
        assert result["reason"] == "summarizer_failed"
        assert result["messages"] is messages

    def test_raising_summarizer_fails_open_without_propagating(self):
        messages = make_session(60)

        def boom(middle_messages, previous_summary=None):
            raise TimeoutError("summarizer timed out")

        result = _plan(messages, boom)
        assert result["action"] == "remote_with_guidance"
        assert result["reason"] == "summarizer_failed"
        assert result["messages"] is messages
        assert result["summarizer_failure_kind"] == "TimeoutError"
        assert result["summarizer_attempts"] == 1

    def test_sentinel_kind_and_attempts_propagate(self):
        messages = make_session(60)
        result = _plan(messages, lambda m, p=None: EmptySummary("timeout", 3))
        assert result["action"] == "remote_with_guidance"
        assert result["summarizer_failure_kind"] == "timeout"
        assert result["summarizer_attempts"] == 3

    def test_plain_empty_string_reports_empty_summary_kind(self):
        messages = make_session(60)
        result = _plan(messages, lambda m, p=None: "")
        assert result["summarizer_failure_kind"] == "empty_summary"

    def test_decide_reports_not_applied_and_keeps_history(self):
        messages = make_session(60)
        decision = decide_session_compaction(
            messages,
            fast_config(),
            "fast",
            summarizer=lambda m, p=None: "",
            estimate_tokens=counting_estimator,
            session_id="sess-failopen",
        )
        assert decision["action"] == "remote_with_guidance"
        assert decision["applied"] is False
        assert decision["messages"] is messages

    def test_no_empty_marker_injected(self):
        # The bug injected ``_summary_message("")``; assert no marker at all
        # appears in the (unchanged) history.
        messages = make_session(60)
        result = _plan(messages, lambda m, p=None: "")
        assert not any(
            isinstance(m.get("content"), str)
            and "The conversation history before this point was compacted" in m["content"]
            for m in result["messages"]
        )


# ===================================================================
# AC2 — failure is visible at WARNING with session / kind / attempts
# ===================================================================


class TestFailureVisibility:
    def test_failure_logged_at_warning_with_context(self, caplog):
        messages = make_session(60)
        result = _plan(messages, lambda m, p=None: EmptySummary("timeout", 3))
        with caplog.at_level("WARNING", logger="llama-proxy.compaction"):
            fields = log_compaction_event(
                result,
                session_id="01a091ac-deadbeef",
                estimate_tokens=counting_estimator,
            )
        assert fields is not None
        assert fields["action"] == "remote_with_guidance"
        assert fields["failure_kind"] == "timeout"
        assert fields["attempts"] == 3
        assert fields["session"] == "01a091ac"
        assert fields["session_id"] == "01a091ac-deadbeef"
        rec = next(r for r in caplog.records if r.name == "llama-proxy.compaction")
        assert rec.levelno == 30  # WARNING
        msg = rec.getMessage()
        assert "failure_kind=timeout" in msg
        assert "attempts=3" in msg
        assert "01a091ac" in msg

    def test_raising_summarizer_failure_is_logged(self, caplog):
        def boom(middle_messages, previous_summary=None):
            raise RuntimeError("transport exploded")

        result = _plan(make_session(60), boom)
        with caplog.at_level("WARNING", logger="llama-proxy.compaction"):
            fields = log_compaction_event(
                result,
                session_id="sess-boom",
                estimate_tokens=counting_estimator,
            )
        assert fields["failure_kind"] == "RuntimeError"
        assert fields["attempts"] == 1

    def test_healthy_compaction_has_no_failure_fields(self, caplog):
        # AC1 regression on the success path: the field set stays exactly
        # the documented event schema (no failure fields mixed in).
        result = _plan(
            make_session(60),
            lambda m, p=None: "REAL SUMMARY with content",
        )
        assert result["action"] == "compact"
        fields = log_compaction_event(
            result,
            session_id="sess-ok",
            estimate_tokens=counting_estimator,
        )
        assert "failure_kind" not in fields
        assert "attempts" not in fields


# ===================================================================
# AC3 — previous-summary detection holds after a successful compaction
# ===================================================================


class TestPreviousSummaryLifecycle:
    def test_successful_compaction_then_incremental_update(self):
        calls: list[str | None] = []

        def summarizer(middle_messages, previous_summary=None):
            calls.append(previous_summary)
            return f"SUMMARY-{len(calls)}"

        # Phase 1 (CREATION): fresh session, no marker.
        first = _plan(make_session(60), summarizer)
        assert first["action"] == "compact"
        assert calls == [None]
        assert extract_previous_summary(first["messages"]) == "SUMMARY-1"

        # Phase 2 (UPDATE): grow the compacted history past the trigger.
        grown = list(first["messages"])
        for i in range(60):
            grown.append({"role": "user", "content": f"new-q{i}"})
            grown.append({"role": "assistant", "content": f"new-a{i}"})
        second = _plan(grown, summarizer)
        assert second["action"] == "compact"
        # The planner detected the injected summary and switched to UPDATE.
        assert calls[1] == "SUMMARY-1"
        assert second["previous_summary"] == "SUMMARY-1"
        assert extract_previous_summary(second["messages"]) == "SUMMARY-2"

    def test_empty_summary_would_have_never_converged(self):
        # Demonstrates the original bug's non-convergence: with an empty
        # summary the marker is undetectable, so extract_previous_summary
        # returns None and the next pass re-runs CREATION. Our fix never
        # injects an empty marker, so it does not apply.
        from proxy.compaction import _summary_message

        empty_marker_history = [
            {"role": "user", "content": "FIRST"},
            _summary_message(""),
        ]
        assert extract_previous_summary(empty_marker_history) is None

        messages = make_session(60)
        result = _plan(messages, lambda m, p=None: "")
        assert extract_previous_summary(result["messages"]) is None
        assert result["messages"] is messages  # unchanged, no marker added


# ===================================================================
# Summarizer transport layer — sentinel carries kind / attempts
# ===================================================================


def _mock_client_with_error(error) -> MagicMock:
    client = MagicMock()
    client.post.side_effect = error
    return client


class TestSummarizerSentinel:
    def _run(self, client, config=None, messages=None):
        with (
            patch("proxy.compaction_summarizer.httpx.Client") as mock_cls,
            patch("proxy.compaction_summarizer.time.sleep"),
        ):
            mock_cls.return_value.__enter__.return_value = client
            from proxy.compaction_summarizer import build_local_summarizer

            summarizer = build_local_summarizer(config or {}, llama_port=8080)
            return summarizer(messages or [{"role": "user", "content": "hi"}])

    def test_connect_error_sentinel_kind_and_attempts(self):
        result = self._run(_mock_client_with_error(httpx.ConnectError("refused")))
        assert result == ""
        assert isinstance(result, EmptySummary)
        assert result.kind == "connect"
        assert result.attempts == 3  # default 2 retries + initial

    def test_timeout_sentinel_kind(self):
        result = self._run(_mock_client_with_error(httpx.ReadTimeout("slow")))
        assert result == ""
        assert result.kind == "timeout"
        assert result.attempts == 3

    def test_http_500_sentinel_kind(self):
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "boom"
        client.post.return_value = resp
        result = self._run(client)
        assert result == ""
        assert result.kind == "http_500"
        assert result.attempts == 3

    def test_permanent_4xx_sentinel_single_attempt(self):
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 401
        resp.text = "unauthorized"
        client.post.return_value = resp
        result = self._run(client)
        assert result == ""
        assert result.kind == "http_401"
        assert result.attempts == 1

    def test_empty_completion_sentinel_kind(self):
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": ""}}]}
        client.post.return_value = resp
        result = self._run(client)
        assert result == ""
        assert result.kind == "empty_completion"

    def test_real_summarizer_empty_output_drives_remote_path(self):
        # End-to-end: the production summarizer returning a sentinel makes
        # the planner fail open rather than drop the middle turns.
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "   "}}]}
        client.post.return_value = resp
        with patch("proxy.compaction_summarizer.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value = client
            from proxy.compaction_summarizer import build_local_summarizer

            summarizer = build_local_summarizer({}, llama_port=8080)
            messages = make_session(60)
            result = plan_session_compaction(
                messages,
                fast_config(),
                "fast",
                summarizer=summarizer,
                estimate_tokens=counting_estimator,
            )
        assert result["action"] == "remote_with_guidance"
        assert result["reason"] == "summarizer_failed"
        assert result["summarizer_failure_kind"] == "empty_completion"
        assert result["messages"] is messages
