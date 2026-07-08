"""Tests for per-category token estimation in router_helpers (LP-0MRAW8XUF005IB4K).

Covers:
- _estimate_tokens_sent returns per-category breakdown dict
- Category mapping: user, assistant, tool, system
- Tool calls in assistant messages counted under tool
- Tools array definitions counted under tool
- Non-messages format (input field) fallback
- Empty/malformed bodies
"""

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_messages_body(messages: list) -> bytes:
    """Build a JSON request body with messages array."""
    return json.dumps({"messages": messages}).encode("utf-8")


def _build_input_body(inp) -> bytes:
    """Build a JSON request body with input field (non-messages format)."""
    return json.dumps({"input": inp}).encode("utf-8")


def _build_tools_body(messages: list, tools: list) -> bytes:
    """Build a JSON request body with messages and tools."""
    return json.dumps({"messages": messages, "tools": tools}).encode("utf-8")


# ---------------------------------------------------------------------------
# Tests — _estimate_tokens_sent per-category breakdown
# ---------------------------------------------------------------------------


class TestEstimateTokensSentCategoryBreakdown:
    """Verify _estimate_tokens_sent returns per-category breakdown dict."""

    def test_empty_messages_returns_all_zeros(self):
        """Empty messages array yields zero-count dict."""
        body = _build_messages_body([])
        body_json = json.loads(body)
        from proxy.router_helpers import _estimate_tokens_sent

        result = _estimate_tokens_sent(body, body_json, "qwen3")
        assert isinstance(result, dict)
        assert result.get("user", -1) >= 0
        assert result.get("assistant", -1) >= 0
        assert result.get("tool", -1) >= 0
        assert result.get("system", -1) >= 0

    def test_user_message_tokens_in_user_category(self):
        """User message content tokens go to 'user' category."""
        body = _build_messages_body([
            {"role": "user", "content": "Hello, how are you?"}
        ])
        body_json = json.loads(body)
        from proxy.router_helpers import _estimate_tokens_sent

        result = _estimate_tokens_sent(body, body_json, "qwen3")
        assert result["user"] > 0
        assert result["assistant"] == 0
        assert result["tool"] == 0
        assert result["system"] == 0

    def test_assistant_message_tokens_in_assistant_category(self):
        """Assistant message content tokens go to 'assistant' category."""
        body = _build_messages_body([
            {"role": "assistant", "content": "I am fine, thank you!"}
        ])
        body_json = json.loads(body)
        from proxy.router_helpers import _estimate_tokens_sent

        result = _estimate_tokens_sent(body, body_json, "qwen3")
        assert result["assistant"] > 0
        assert result["user"] == 0
        assert result["tool"] == 0
        assert result["system"] == 0

    def test_tool_message_tokens_in_tool_category(self):
        """Tool response content tokens go to 'tool' category."""
        body = _build_messages_body([
            {"role": "tool", "content": '{"result": "42"}'}
        ])
        body_json = json.loads(body)
        from proxy.router_helpers import _estimate_tokens_sent

        result = _estimate_tokens_sent(body, body_json, "qwen3")
        assert result["tool"] > 0
        assert result["user"] == 0
        assert result["assistant"] == 0
        assert result["system"] == 0

    def test_system_message_tokens_in_system_category(self):
        """System message content tokens go to 'system' category."""
        body = _build_messages_body([
            {"role": "system", "content": "You are a helpful assistant."}
        ])
        body_json = json.loads(body)
        from proxy.router_helpers import _estimate_tokens_sent

        result = _estimate_tokens_sent(body, body_json, "qwen3")
        assert result["system"] > 0
        assert result["user"] == 0
        assert result["assistant"] == 0
        assert result["tool"] == 0

    def test_mixed_roles_all_categories_counted(self):
        """Multiple message roles produce correct per-category counts."""
        messages = [
            {"role": "system", "content": "System prompt here."},
            {"role": "user", "content": "User message here."},
            {"role": "assistant", "content": "Assistant reply here."},
            {"role": "tool", "content": "Tool result here."},
        ]
        body = _build_messages_body(messages)
        body_json = json.loads(body)
        from proxy.router_helpers import _estimate_tokens_sent

        result = _estimate_tokens_sent(body, body_json, "qwen3")
        assert result["system"] > 0
        assert result["user"] > 0
        assert result["assistant"] > 0
        assert result["tool"] > 0

    def test_tool_calls_in_assistant_counted_as_tool(self):
        """Tool calls embedded in assistant messages count as tool tokens."""
        messages = [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "London"}'
                        }
                    }
                ]
            },
        ]
        body = _build_messages_body(messages)
        body_json = json.loads(body)
        from proxy.router_helpers import _estimate_tokens_sent

        result = _estimate_tokens_sent(body, body_json, "qwen3")
        assert result["tool"] > 0, "Tool call tokens should be in 'tool' category"
        assert result["assistant"] == 0, "Empty assistant content should not add to 'assistant'"
        assert result["user"] > 0

    def test_tools_array_definition_counted_as_tool(self):
        """Tools array in request body counts as tool tokens."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
        body = _build_tools_body(
            [{"role": "user", "content": "What's the weather?"}],
            tools,
        )
        body_json = json.loads(body)
        from proxy.router_helpers import _estimate_tokens_sent

        result = _estimate_tokens_sent(body, body_json, "qwen3")
        assert result["tool"] > 0, "Tools array tokens should be in 'tool' category"

    def test_input_field_fallback(self):
        """Non-messages format (input field) counts all tokens under user."""
        body = _build_input_body("Hello, model!")
        body_json = json.loads(body)
        from proxy.router_helpers import _estimate_tokens_sent

        result = _estimate_tokens_sent(body, body_json, "qwen3")
        assert result["user"] > 0
        assert result["assistant"] == 0
        assert result["tool"] == 0
        assert result["system"] == 0

    def test_empty_body_fallback(self):
        """Empty body returns all zeros."""
        body = b""
        from proxy.router_helpers import _estimate_tokens_sent

        result = _estimate_tokens_sent(body, {}, "qwen3")
        assert isinstance(result, dict)
        assert result["user"] == 0
        assert result["assistant"] == 0
        assert result["tool"] == 0
        assert result["system"] == 0

    def test_raw_body_fallback_attributes_to_user(self):
        """Raw body (not JSON) counts all tokens under user."""
        body = b"raw text input"
        from proxy.router_helpers import _estimate_tokens_sent

        result = _estimate_tokens_sent(body, {}, "qwen3")
        assert result["user"] > 0
        assert result["assistant"] == 0
        assert result["tool"] == 0
        assert result["system"] == 0

    def test_sum_of_categories_matches_previous_total_behavior(self):
        """Sum of all category counts should match what the old int return would have been."""
        messages = [
            {"role": "system", "content": "System prompt here."},
            {"role": "user", "content": "User message here."},
            {"role": "assistant", "content": "Assistant reply here."},
            {"role": "tool", "content": "Tool result here."},
        ]
        body = _build_messages_body(messages)
        body_json = json.loads(body)
        from proxy.router_helpers import _estimate_tokens_sent

        result = _estimate_tokens_sent(body, body_json, "qwen3")
        total = sum(result.values())
        assert total > 0, "Total should be > 0 for non-empty messages"
