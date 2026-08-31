"""Tests verifying that the proxy preserves Qwen thinking content (reasoning_content).

Covers AC2 (proxy does not strip thinking blocks or reasoning content) and AC4
(thinking content survives into the recorded session transcript).

These tests add coverage that the existing suite does not: the session
recorder's write/read round-trip of a Qwen-style response carrying
``reasoning_content``. SSE-stream extraction and thinking-only semantics are
covered by ``test_session_manager.py`` and ``test_incremental_ingestion.py``.

Relevant work-item: LP-0MT5YLL36000ZYRT
"""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

pytestmark = pytest.mark.refactor_parity


def _run(coro):
    """Run an async coroutine to completion (sync test helper)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# AC2/AC4: Session recorder round-trip preserves reasoning_content
# ---------------------------------------------------------------------------

class TestSessionRecorderRoundTrip:
    """Verify that the session recorder preserves reasoning_content in recordings."""

    def test_recorder_preserves_reasoning_content_in_response(self):
        """A recorded provider_to_client response with reasoning_content round-trips intact."""
        from proxy.session_recorder import SessionRecorder

        with TemporaryDirectory() as tmpdir:
            recorder = SessionRecorder(recording_path=tmpdir)
            _run(
                recorder.record_response(
                    session_id="test-session-think",
                    direction="provider_to_client",
                    payload={
                        "id": "chatcmpl-123",
                        "model": "qwen3.7-max",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "reasoning_content": "Let me think about the capital of France.\nThe answer is Paris.",
                                    "content": "Paris",
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 50,
                            "completion_tokens": 120,
                            "total_tokens": 170,
                        },
                    },
                    model="qwen3.7-max",
                    provider="opencode-go",
                )
            )
            # Verify the recording file exists and round-trips
            recording_path = Path(tmpdir) / "test-session-think"
            assert recording_path.exists(), "Recording directory not created"
            recording_files = list(recording_path.glob("*.json"))
            assert len(recording_files) == 1, f"Expected 1 recording file, found {len(recording_files)}"
            recorded = json.loads(recording_files[0].read_text())
            # Round-trip: reasoning_content must survive unchanged
            msg = recorded["payload"]["choices"][0]["message"]
            assert msg["reasoning_content"] == "Let me think about the capital of France.\nThe answer is Paris."
            assert msg["content"] == "Paris"
            # Envelope metadata retained
            assert recorded["model"] == "qwen3.7-max"
            assert recorded["provider"] == "opencode-go"

    def test_recorder_preserves_tool_calls_with_reasoning_content(self):
        """Tool calls generated during thinking preserve reasoning_content in recording."""
        from proxy.session_recorder import SessionRecorder

        with TemporaryDirectory() as tmpdir:
            recorder = SessionRecorder(recording_path=tmpdir)
            _run(
                recorder.record_response(
                    session_id="test-session-tool-think",
                    direction="provider_to_client",
                    payload={
                        "id": "chatcmpl-456",
                        "model": "qwen3.6-plus",
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "message": {
                                    "role": "assistant",
                                    "reasoning_content": "I need to check the filesystem.",
                                    "tool_calls": [
                                        {
                                            "id": "call-1",
                                            "type": "function",
                                            "function": {
                                                "name": "bash",
                                                "arguments": '{"command": "ls -la"}',
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                    },
                    model="qwen3.6-plus",
                    provider="opencode-go",
                )
            )
            recording_path = Path(tmpdir) / "test-session-tool-think"
            recording_files = list(recording_path.glob("*.json"))
            assert len(recording_files) == 1
            recorded = json.loads(recording_files[0].read_text())
            msg = recorded["payload"]["choices"][0]["message"]
            assert "reasoning_content" in msg
            assert "I need to check the filesystem." in msg["reasoning_content"]
            assert len(msg["tool_calls"]) == 1
            assert msg["tool_calls"][0]["function"]["name"] == "bash"


# ---------------------------------------------------------------------------
# AC4: Full Qwen-style thinking stream → session transcript
# ---------------------------------------------------------------------------

class TestQwenStyleThinkingPreservation:
    """End-to-end thinking preservation for a Qwen-style streaming response."""

    def test_full_qwen_stream_round_trip(self):
        """A Qwen3 thinking stream: reasoning_content captured, then recorded."""
        from proxy.session import extract_streamed_assistant_message_from_sse
        from proxy.session_recorder import SessionRecorder

        # Simulated Qwen3 stream: thinking during the reasoning phase, then content
        sse_stream = (
            'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
            'data: {"choices":[{"delta":{"reasoning_content":"Step 1: Analyze the input."}}]}\n'
            'data: {"choices":[{"delta":{"reasoning_content":"Step 2: Determine the answer."}}]}\n'
            'data: {"choices":[{"delta":{"reasoning_content":"Step 3: Formulate the response."}}]}\n'
            'data: {"choices":[{"delta":{"content":"The analysis shows that the result is 42."}}]}\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":25,"completion_tokens":95}}\n'
            'data: [DONE]\n'
        )

        # Step 1: Extract assistant message from SSE (session transcript assembly)
        assistant_msg = extract_streamed_assistant_message_from_sse(sse_stream)

        assert assistant_msg is not None
        assert assistant_msg["role"] == "assistant"
        assert "reasoning_content" in assistant_msg

        # Verify all thinking content is preserved
        reasoning = assistant_msg["reasoning_content"]
        assert "Step 1" in reasoning
        assert "Step 2" in reasoning
        assert "Step 3" in reasoning
        assert "Analyze the input" in reasoning
        assert "Determine the answer" in reasoning
        assert "Formulate the response" in reasoning

        # Verify content is also captured
        assert assistant_msg["content"] == "The analysis shows that the result is 42."

        # Step 2: Record and round-trip to the session transcript store
        with TemporaryDirectory() as tmpdir:
            recorder = SessionRecorder(recording_path=tmpdir)
            _run(
                recorder.record_response(
                    session_id="qwen-think-full-test",
                    direction="provider_to_client",
                    payload={
                        "id": "chatcmpl-qwen",
                        "model": "qwen3.7-max",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": assistant_msg,
                            }
                        ],
                    },
                    model="qwen3.7-max",
                    provider="opencode-go",
                )
            )
            recording_path = Path(tmpdir) / "qwen-think-full-test"
            recording_files = list(recording_path.glob("*.json"))
            assert len(recording_files) == 1
            recorded = json.loads(recording_files[0].read_text())
            msg = recorded["payload"]["choices"][0]["message"]
            assert msg["reasoning_content"] == assistant_msg["reasoning_content"]
            assert msg["content"] == assistant_msg["content"]
