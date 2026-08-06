"""
Test-first: tool-call history sanitizer for remote sends.

Red tests for LP-0MSC4V113006XZG1 (parent LP-0MSC1BNP90017L9K).

RCA (F1, LP-0MSC4UJXU008HVV5) confirmed the remote 400 rejection shapes:
- missing ``tool_call_id`` on tool messages (400, both opencode zen/go & deepseek)
- dangling/mismatched ``tool_call_id`` (400, both)
- missing ``id``/``type`` on assistant tool_calls entries (400, both)
- empty ``tool_calls`` array (400, both)

Accepted (do NOT alter): content:null + tool_calls, content:null +
reasoning_content + tool_calls, truncated arguments JSON.

Sanitizer policy (operator decisions): always-on, hybrid — repair where
unambiguous, prune where not. Assertions target the public helper
``_sanitize_remote_messages(messages) -> list[dict]`` in
``proxy.proxy_remote`` (implemented in F5).
"""

import copy

import pytest


def _sanitize(messages):
    """Call the sanitizer under test (implemented in F5)."""
    from proxy.proxy_remote import _sanitize_remote_messages

    return _sanitize_remote_messages(messages)


def _valid_history():
    """A well-formed tool-call turn (should pass through unchanged)."""
    return [
        {"role": "user", "content": "list files"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_01",
                    "type": "function",
                    "function": {"name": "ls", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_01", "content": "file1 file2"},
        {"role": "user", "content": "thanks"},
    ]


# ---------------------------------------------------------------------------
# Red tests: malformed shapes
# ---------------------------------------------------------------------------


def test_valid_tool_call_sequence_unchanged():
    """Valid tool-call sequences keep structure and values (regression guard).

    The only mutation applied to a valid sequence is the reasoning_content
    round-trip repair (LP-0MSGU3JNU0092AFQ): assistant messages missing the
    field gain ``reasoning_content: ""``. Content, tool_calls, tool messages
    and all other fields must be byte-identical.
    """
    history = _valid_history()
    result = _sanitize(copy.deepcopy(history))
    assert result[1]["content"] == history[1]["content"]
    assert result[1]["tool_calls"] == history[1]["tool_calls"]
    assert result[2] == history[2], "tool message must be unchanged"
    assert result[0] == history[0] and result[3] == history[3], "user messages must be unchanged"
    assert result[1]["reasoning_content"] == "", (
        "assistant reasoning_content must be normalized to '' (round-trip repair)"
    )


def test_content_null_repaired_when_tool_calls_present():
    """assistant content:null + tool_calls → content repaired to '' (never null)."""
    history = _valid_history()
    history[1]["content"] = None
    result = _sanitize(copy.deepcopy(history))
    assert result[1]["content"] == "", "content must be repaired to empty string"
    assert len(result[1].get("tool_calls", [])) == 1, "valid tool_call preserved"


def test_reasoning_content_preserved_on_remote_sends():
    """reasoning_content must be PRESERVED on remote sends (LP-0MSCGTYWA006NAZC).

    Console Go (opencode.ai/zen/go) runs thinking mode and requires the
    assistant reasoning_content it previously generated to be echoed back in
    the next request; stripping it causes HTTP 400. Both zen/go and
    api.deepseek.com accept reasoning_content (RCA probe, LP-0MSC4UJXU008HVV5).
    """
    history = _valid_history()
    history[1]["reasoning_content"] = "thinking about ls"
    result = _sanitize(copy.deepcopy(history))
    assert result[1]["reasoning_content"] == "thinking about ls", (
        "reasoning_content must be preserved for thinking-mode round-trips"
    )


def test_reasoning_content_preserved_alone_without_tool_calls():
    """A pure reasoning turn (no tool_calls) keeps its reasoning_content."""
    messages = [
        {"role": "user", "content": "think about it"},
        {"role": "assistant", "content": "", "reasoning_content": "deep thought"},
    ]
    result = _sanitize(copy.deepcopy(messages))
    assert result[1]["reasoning_content"] == "deep thought"


def test_reasoning_content_preserved_while_tool_call_repairs_apply():
    """reasoning_content + tool_calls: preserve reasoning, still repair content.

    Regression guard for the AC: a multi-turn request carrying assistant
    reasoning_content AND tool_calls must pass through with reasoning preserved
    while the original RCA repairs (content:null -> '', missing type ->
    'function') still apply.
    """
    history = _valid_history()
    history[1]["content"] = None
    del history[1]["tool_calls"][0]["type"]
    history[1]["reasoning_content"] = "thinking about ls"
    result = _sanitize(copy.deepcopy(history))
    assert result[1]["reasoning_content"] == "thinking about ls", (
        "reasoning_content must survive alongside tool-call repairs"
    )
    assert result[1]["content"] == "", "content:null must still be repaired"
    assert result[1]["tool_calls"][0]["type"] == "function", (
        "missing type must still be repaired to 'function'"
    )


# ---------------------------------------------------------------------------
# reasoning_content round-trip repair (LP-0MSGU3JNU0092AFQ)
# ---------------------------------------------------------------------------
#
# Remote thinking-mode providers (Console opencode.ai/zen, Console Go
# opencode.ai/zen/go, api.deepseek.com) require the `reasoning_content` field
# to be present on EVERY assistant message in a multi-turn request. The client
# (opencode) drops the empty `reasoning_content: ""` that the upstream emitted
# on tool-call-only turns, so the field is entirely absent on those messages
# when the history is re-sent. The sanitizer must inject `reasoning_content:
# ""` (matching upstream emission) on assistant messages where the field is
# missing or null — additive-only; existing values are never touched.


def test_reasoning_content_injected_when_missing_on_tool_call_turn():
    """Assistant tool-call message missing reasoning_content -> injected as ''.

    This is the exact rejection shape from the recorded 400 traffic (session
    019fd49c-...): 3 tool-call-only assistant messages lacked the field while
    30 others carried it, and the upstream rejected the whole request.
    """
    history = _valid_history()
    assert "reasoning_content" not in history[1], "precondition: field absent"
    result = _sanitize(copy.deepcopy(history))
    assert result[1]["reasoning_content"] == "", (
        "missing reasoning_content must be injected as '' on assistant tool-call turns"
    )
    assert result[1]["tool_calls"][0]["id"] == "call_01", (
        "tool_calls must be unchanged by the reasoning injection"
    )


def test_reasoning_content_injected_when_null():
    """reasoning_content: null -> normalized to '' (additive repair)."""
    messages = [
        {"role": "user", "content": "think about it"},
        {"role": "assistant", "content": "", "reasoning_content": None},
    ]
    result = _sanitize(copy.deepcopy(messages))
    assert result[1]["reasoning_content"] == "", (
        "null reasoning_content must be normalized to ''"
    )


def test_reasoning_content_non_empty_preserved_not_overwritten():
    """Existing non-empty reasoning_content must never be modified."""
    history = _valid_history()
    history[1]["reasoning_content"] = "deep thinking here"
    result = _sanitize(copy.deepcopy(history))
    assert result[1]["reasoning_content"] == "deep thinking here", (
        "existing reasoning_content must be preserved verbatim"
    )


def test_reasoning_content_not_injected_on_non_assistant_roles():
    """Injection applies only to assistant messages; other roles untouched."""
    history = _valid_history()
    result = _sanitize(copy.deepcopy(history))
    assert "reasoning_content" not in result[0], "user message must stay unchanged"
    assert "reasoning_content" not in result[2], "tool message must stay unchanged"
    assert "reasoning_content" not in result[3], "user message must stay unchanged"


def test_mixed_history_all_assistant_messages_get_reasoning_content():
    """Recorded failure shape: mixed history -> every assistant msg has the field.

    Mirrors the recorded 400 request (session 019fd49c-..., 2026-08-06): 33
    assistant messages, 30 with non-empty reasoning_content and 3 tool-call-only
    messages missing the field entirely. After sanitize, all 33 must carry the
    field and existing values must be untouched.
    """
    messages = [
        {"role": "user", "content": "do work"},
        {"role": "assistant", "content": "ok", "reasoning_content": "thought 1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "ls", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "file1"},
        {"role": "assistant", "content": "ok", "reasoning_content": "thought 2"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_b", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_b", "content": "ok"},
        {"role": "assistant", "content": "done", "reasoning_content": "thought 3"},
    ]
    result = _sanitize(copy.deepcopy(messages))
    assistant_indices = [i for i, m in enumerate(result) if m.get("role") == "assistant"]
    assert len(assistant_indices) == 5, "all assistant messages must survive"
    for i in assistant_indices:
        assert "reasoning_content" in result[i], (
            f"assistant message {i} must carry reasoning_content after sanitize"
        )
    # Existing values preserved verbatim; injected values are empty strings.
    assert result[1]["reasoning_content"] == "thought 1"
    assert result[4]["reasoning_content"] == "thought 2"
    assert result[7]["reasoning_content"] == "thought 3"
    assert result[2]["reasoning_content"] == ""
    assert result[5]["reasoning_content"] == ""


def test_missing_tool_call_id_pruned():
    """Tool message without tool_call_id → 400; sanitizer must drop it."""
    history = _valid_history()
    del history[2]["tool_call_id"]
    result = _sanitize(copy.deepcopy(history))
    assert all(m.get("role") != "tool" for m in result), (
        "tool message with missing tool_call_id must be pruned"
    )


def test_dangling_tool_call_id_pruned():
    """Tool message referencing an unknown tool_call_id → 400; must be pruned."""
    history = _valid_history()
    history[2]["tool_call_id"] = "call_NOPE"
    result = _sanitize(copy.deepcopy(history))
    assert all(m.get("role") != "tool" for m in result), (
        "tool message with dangling tool_call_id must be pruned"
    )


def test_missing_tool_call_id_type_pruned():
    """Assistant tool_calls entry missing id/type → 400; entry must be pruned."""
    history = _valid_history()
    del history[1]["tool_calls"][0]["id"]
    del history[1]["tool_calls"][0]["type"]
    result = _sanitize(copy.deepcopy(history))
    assert result[1].get("tool_calls") != [{"function": {}}], (
        "invalid tool_calls entry must not survive"
    )
    # The dependent tool message references the pruned id → also pruned.
    assert all(m.get("role") != "tool" for m in result), (
        "tool message referencing a pruned tool_call must be pruned"
    )


def test_missing_type_repaired():
    """tool_calls entry missing type → repaired to 'function' (unambiguous)."""
    history = _valid_history()
    del history[1]["tool_calls"][0]["type"]
    result = _sanitize(copy.deepcopy(history))
    assert result[1]["tool_calls"][0]["type"] == "function", (
        "missing type must be repaired to 'function'"
    )


def test_empty_tool_calls_pruned():
    """Empty tool_calls array → 400; key must be removed."""
    history = _valid_history()
    history[1]["tool_calls"] = []
    result = _sanitize(copy.deepcopy(history))
    assert "tool_calls" not in result[1], "empty tool_calls array must be removed"


def test_truncated_arguments_preserved():
    """Truncated arguments JSON is ACCEPTED by remotes (RCA) — leave as-is."""
    history = _valid_history()
    history[1]["tool_calls"][0]["function"]["arguments"] = '{"x": "unterminated'
    result = _sanitize(copy.deepcopy(history))
    assert result[1]["tool_calls"][0]["function"]["arguments"] == '{"x": "unterminated', (
        "truncated arguments are accepted by remotes; must not be altered"
    )


def test_arguments_missing_repaired_to_empty():
    """function.arguments missing → repaired to '' (unambiguous)."""
    history = _valid_history()
    del history[1]["tool_calls"][0]["function"]["arguments"]
    result = _sanitize(copy.deepcopy(history))
    assert result[1]["tool_calls"][0]["function"]["arguments"] == "", (
        "missing arguments must be repaired to empty string"
    )
