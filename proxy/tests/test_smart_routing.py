"""Tests for smart routing logic that bypasses local for large-context
requests based on cached_tokens ratio from llama.cpp instead of inferred
cache-cold state machine.

(LP-0MRP44W7I0085I6N: Replace proxy cache-cold state machine with
cached_tokens-based routing)
"""

from proxy.provider import _estimate_prompt_tokens_for_routing, _should_skip_local

# ---------------------------------------------------------------------------
# Cached-tokens storage tests
# ---------------------------------------------------------------------------


class TestLastCachedRatio:
    """Tests for the _last_cached_ratio in-memory dict."""

    def setup_method(self):
        from proxy.provider import _last_cached_ratio
        _last_cached_ratio.clear()

    def test_new_session_defaults_to_cold(self):
        """A new (model, session) pair should have no entry (default = 0.0 cold)."""
        from proxy.provider import _last_cached_ratio
        assert ("Qwen3", "sess_new") not in _last_cached_ratio

    def test_update_cached_ratio(self):
        """Updating ratio should store it."""
        from proxy.provider import update_cached_ratio, _last_cached_ratio
        update_cached_ratio("Qwen3", "sess_a", cached_tokens=80, prompt_tokens=100)
        assert _last_cached_ratio[("Qwen3", "sess_a")] == 0.8

    def test_fully_cached_ratio(self):
        """Prompt with all tokens cached should give ratio of 1.0."""
        from proxy.provider import update_cached_ratio, _last_cached_ratio
        update_cached_ratio("Qwen3", "sess_a", cached_tokens=100, prompt_tokens=100)
        assert _last_cached_ratio[("Qwen3", "sess_a")] == 1.0

    def test_zero_cached_tokens(self):
        """No cached tokens should give ratio of 0.0."""
        from proxy.provider import update_cached_ratio, _last_cached_ratio
        update_cached_ratio("Qwen3", "sess_a", cached_tokens=0, prompt_tokens=100)
        assert _last_cached_ratio[("Qwen3", "sess_a")] == 0.0

    def test_zero_prompt_tokens_defaults_to_zero(self):
        """Zero prompt tokens should not cause division by zero; defaults to 0.0."""
        from proxy.provider import update_cached_ratio, _last_cached_ratio
        update_cached_ratio("Qwen3", "sess_a", cached_tokens=0, prompt_tokens=0)
        assert _last_cached_ratio[("Qwen3", "sess_a")] == 0.0

    def test_multiple_models_independent(self):
        """Different models should have independent ratios."""
        from proxy.provider import update_cached_ratio, _last_cached_ratio
        update_cached_ratio("Qwen3", "sess_a", cached_tokens=80, prompt_tokens=100)
        update_cached_ratio("gemma4", "sess_a", cached_tokens=10, prompt_tokens=100)
        assert _last_cached_ratio[("Qwen3", "sess_a")] == 0.8
        assert _last_cached_ratio[("gemma4", "sess_a")] == 0.1

    def test_multiple_sessions_independent(self):
        """Different sessions should have independent ratios."""
        from proxy.provider import update_cached_ratio, _last_cached_ratio
        update_cached_ratio("Qwen3", "sess_a", cached_tokens=80, prompt_tokens=100)
        update_cached_ratio("Qwen3", "sess_b", cached_tokens=10, prompt_tokens=100)
        assert _last_cached_ratio[("Qwen3", "sess_a")] == 0.8
        assert _last_cached_ratio[("Qwen3", "sess_b")] == 0.1

    def test_cache_cold_ratio_defaults_conservative(self):
        """The _get_cached_ratio helper should return 0.0 for unknown pairs."""
        from proxy.provider import _get_cached_ratio
        assert _get_cached_ratio("Qwen3", "unknown_session") == 0.0

    def test_get_cached_ratio_after_update(self):
        """After updating, _get_cached_ratio should return the stored value."""
        from proxy.provider import _get_cached_ratio, update_cached_ratio
        update_cached_ratio("Qwen3", "sess_a", cached_tokens=80, prompt_tokens=100)
        assert _get_cached_ratio("Qwen3", "sess_a") == 0.8


# ---------------------------------------------------------------------------
# Cached-tokens SSE parsing tests
# ---------------------------------------------------------------------------


class TestExtractCachedTokens:
    """Tests for the cached_tokens extraction from SSE usage events."""

    def test_extract_from_usage_event(self):
        """Extract cached_tokens from a usage dict."""
        from proxy.provider import _extract_cached_tokens_from_usage
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 80}}
        assert _extract_cached_tokens_from_usage(usage) == 80

    def test_no_cached_tokens_field(self):
        """Missing cached_tokens field should return 0."""
        from proxy.provider import _extract_cached_tokens_from_usage
        usage = {"prompt_tokens": 100, "completion_tokens": 50}
        assert _extract_cached_tokens_from_usage(usage) == 0

    def test_none_usage(self):
        """None usage should return 0."""
        from proxy.provider import _extract_cached_tokens_from_usage
        assert _extract_cached_tokens_from_usage(None) == 0

    def test_empty_usage(self):
        """Empty usage dict should return 0."""
        from proxy.provider import _extract_cached_tokens_from_usage
        assert _extract_cached_tokens_from_usage({}) == 0

    def test_cached_tokens_zero(self):
        """cached_tokens=0 should return 0."""
        from proxy.provider import _extract_cached_tokens_from_usage
        usage = {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0}}
        assert _extract_cached_tokens_from_usage(usage) == 0

    def test_extract_from_sse_text(self):
        """Extract cached_tokens from full SSE response text."""
        from proxy.provider import _extract_cached_tokens_from_sse_text
        sse = (
            'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],'
            '"usage":{"prompt_tokens":100,"completion_tokens":50,'
            '"prompt_tokens_details":{"cached_tokens":80}}}\n\n'
        )
        assert _extract_cached_tokens_from_sse_text(sse) == 80

    def test_no_usage_in_sse(self):
        """SSE text without usage event should return 0."""
        from proxy.provider import _extract_cached_tokens_from_sse_text
        sse = (
            'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
            'data: [DONE]\n\n'
        )
        assert _extract_cached_tokens_from_sse_text(sse) == 0

    def test_empty_sse_text(self):
        """Empty SSE text should return 0."""
        from proxy.provider import _extract_cached_tokens_from_sse_text
        assert _extract_cached_tokens_from_sse_text("") == 0

    def test_extract_from_non_streaming_json(self):
        """Extract cached_tokens from a non-streaming JSON response body."""
        from proxy.provider import _extract_cached_tokens_from_usage
        body = {"choices": [{"message": {"content": "Hello"}}], "usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 70}}}
        assert _extract_cached_tokens_from_usage(body.get("usage")) == 70


# ---------------------------------------------------------------------------
# Smart routing decision tests (cached_tokens-based)
# ---------------------------------------------------------------------------


class TestCachedTokensRouting:
    """Tests for the routing decision logic using cached_tokens ratio."""

    def setup_method(self):
        from proxy.provider import _last_cached_ratio
        _last_cached_ratio.clear()

    def test_warm_cache_small_request_routes_local(self):
        """Warm cache (ratio >= 1) + small tokens → routes local (normal)."""
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        threshold = 40000
        from proxy.provider import update_cached_ratio
        update_cached_ratio("Qwen3", "sess_a", cached_tokens=100, prompt_tokens=100)
        should_skip = _should_skip_local("Qwen3", "sess_a", body, threshold)
        assert should_skip is False

    def test_warm_cache_large_request_routes_local(self):
        """Warm cache (ratio >= 1) + large tokens → routes local (cache fully warm)."""
        phrase = "test message content for token estimation "
        body = {"messages": [{"role": "user", "content": phrase * 7000}]}  # ~42K tokens
        threshold = 40000
        from proxy.provider import update_cached_ratio
        update_cached_ratio("Qwen3", "sess_a", cached_tokens=100, prompt_tokens=100)
        should_skip = _should_skip_local("Qwen3", "sess_a", body, threshold)
        assert should_skip is False

    def test_cold_cache_small_request_routes_local(self):
        """Cold cache (no entry, defaults to 0.0) + small tokens → routes local."""
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        threshold = 40000
        should_skip = _should_skip_local("Qwen3", "sess_a", body, threshold)
        assert should_skip is False

    def test_cold_cache_large_request_skips_local(self):
        """Cold cache (ratio < 1) + large tokens > threshold → skips local."""
        phrase = "test message content for token estimation "
        body = {"messages": [{"role": "user", "content": phrase * 7000}]}  # ~42K tokens
        threshold = 40000
        should_skip = _should_skip_local("Qwen3", "sess_a", body, threshold)
        assert should_skip is True

    def test_partially_warm_cache_large_request_skips_local(self):
        """Partially warm cache (ratio 0.5 < 1) + large tokens → skips local."""
        phrase = "test message content for token estimation "
        body = {"messages": [{"role": "user", "content": phrase * 7000}]}  # ~42K tokens
        threshold = 40000
        from proxy.provider import update_cached_ratio
        update_cached_ratio("Qwen3", "sess_a", cached_tokens=50, prompt_tokens=100)
        should_skip = _should_skip_local("Qwen3", "sess_a", body, threshold)
        assert should_skip is True

    def test_threshold_zero_disables_bypass(self):
        """Threshold=0 → always routes local regardless of cache state."""
        phrase = "test message content for token estimation "
        body = {"messages": [{"role": "user", "content": phrase * 7000}]}
        threshold = 0
        should_skip = _should_skip_local("Qwen3", "sess_a", body, threshold)
        assert should_skip is False

    def test_exact_threshold_does_not_skip(self):
        """Tokens below threshold should NOT skip local even when cache is cold."""
        phrase = "test message content for token estimation "
        body = {"messages": [{"role": "user", "content": phrase * 3500}]}  # ~36K tokens
        threshold = 40000
        should_skip = _should_skip_local("Qwen3", "sess_a", body, threshold)
        assert should_skip is False

    def test_precomputed_estimated_tokens_used(self):
        """Pre-computed estimated_tokens should be used when provided."""
        from proxy.provider import update_cached_ratio
        update_cached_ratio("Qwen3", "sess_a", cached_tokens=10, prompt_tokens=100)
        should_skip = _should_skip_local(
            "Qwen3", "sess_a", {"messages": []}, 40000, estimated_tokens=50000
        )
        assert should_skip is True

    def test_default_cold_for_new_session(self):
        """A session with no cached ratio entry defaults to cold."""
        phrase = "test message content for token estimation "
        body = {"messages": [{"role": "user", "content": phrase * 7000}]}
        threshold = 40000
        should_skip = _should_skip_local("Qwen3", "sess_never_seen", body, threshold)
        assert should_skip is True

    def test_warm_session_not_affected_by_other_cold_sessions(self):
        """A warm session should not be affected by cold sessions on same model."""
        phrase = "test message content for token estimation "
        body_big = {"messages": [{"role": "user", "content": phrase * 7000}]}  # ~42K tokens
        threshold = 40000
        from proxy.provider import update_cached_ratio
        # Session A is warm
        update_cached_ratio("Qwen3", "sess_a", cached_tokens=100, prompt_tokens=100)
        # Session B is cold (no entry)
        assert _should_skip_local("Qwen3", "sess_a", body_big, threshold) is False  # warm → local
        assert _should_skip_local("Qwen3", "sess_b", body_big, threshold) is True   # cold → skip


# ---------------------------------------------------------------------------
# Token estimation tests (preserved from previous implementation)
# ---------------------------------------------------------------------------


class TestEstimatePromptTokensForRouting:
    """Tests for _estimate_prompt_tokens_for_routing function."""

    def test_empty_body(self):
        """Empty body should return 0."""
        assert _estimate_prompt_tokens_for_routing({}) == 0

    def test_no_messages(self):
        """Body without messages list should return 0."""
        assert _estimate_prompt_tokens_for_routing({"model": "test"}) == 0

    def test_empty_messages(self):
        """Empty messages list should return 0."""
        assert _estimate_prompt_tokens_for_routing({"messages": []}) == 0

    def test_small_message(self):
        """A small message should yield a small token estimate."""
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        tokens = _estimate_prompt_tokens_for_routing(body)
        assert 1 <= tokens <= 5

    def test_large_message(self):
        """A large message should yield a large estimate using tiktoken."""
        phrase = "test message content for token estimation "
        large_content = phrase * 12000  # ~126K tokens with tiktoken
        body = {"messages": [{"role": "user", "content": large_content}]}
        tokens = _estimate_prompt_tokens_for_routing(body)
        assert tokens >= 65_000
        assert tokens <= 140_000

    def test_40k_threshold_crossing(self):
        """A message with >40K actual tokens should be detected as above threshold."""
        phrase = "test message content for token estimation "
        content = phrase * 7000  # ~42K actual tokens with tiktoken
        body = {"messages": [{"role": "user", "content": content}]}
        tokens = _estimate_prompt_tokens_for_routing(body)
        assert tokens > 40_000

    def test_below_40k(self):
        """A message at ~20K tokens should be below 40K threshold."""
        content = "x" * 40_000  # ~20K estimated
        body = {"messages": [{"role": "user", "content": content}]}
        tokens = _estimate_prompt_tokens_for_routing(body)
        assert tokens < 40_000

    def test_system_message_counted(self):
        """System messages SHOULD be counted in routing estimate.

        System prompts were previously excluded but are now included
        (LP-0MRGT35H1003D1PM) to avoid underestimating large contexts.
        """
        body = {
            "messages": [
                {"role": "system", "content": "x" * 100_000},  # ~25K tokens system
                {"role": "user", "content": "Hello"},
            ]
        }
        tokens = _estimate_prompt_tokens_for_routing(body)
        # Should be large (system content is now counted)
        assert tokens >= 10000

    def test_combined_user_assistant_messages_counted(self):
        """Both user and assistant messages should be counted."""
        phrase = "test message "
        body = {
            "messages": [
                {"role": "user", "content": phrase * 20000},
                {"role": "assistant", "content": phrase * 20000},
            ]
        }
        tokens = _estimate_prompt_tokens_for_routing(body)
        # Should count both: ~130K total with tiktoken
        assert tokens >= 75_000
        assert tokens <= 140_000

    def test_array_content_counts_text_only(self):
        """Array content (multimodal) should count text fields only."""
        phrase = "test message content for token estimation "
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": phrase * 10000},  # ~60K tokens
                        {"type": "image", "url": "http://example.com/img.jpg"},
                    ],
                }
            ]
        }
        tokens = _estimate_prompt_tokens_for_routing(body)
        # Should count only the text part: ~105K tokens with tiktoken
        assert tokens >= 55_000
        assert tokens <= 115_000

    def test_reasoning_content_counted(self):
        """Assistant reasoning_content should be counted (LP-0MRDE669Y003V1SO)."""
        phrase = "test message content for token estimation "
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": phrase * 10000,  # ~105K tokens with tiktoken
                },
            ]
        }
        tokens = _estimate_prompt_tokens_for_routing(body)
        # Should count reasoning_content: ~105K tokens
        assert tokens >= 55_000
        assert tokens <= 115_000

    def test_tool_calls_counted(self):
        """Tool call function names and arguments should be counted (LP-0MRDE669Y003V1SO)."""
        phrase = "test message content for token estimation "
        args_content = '{"command": "' + phrase * 5000 + '"}'
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "bash",
                                "arguments": args_content,
                            },
                        },
                        {
                            "id": "call_2",
                            "function": {
                                "name": "read",
                                "arguments": args_content,
                            },
                        },
                    ],
                },
            ]
        }
        tokens = _estimate_prompt_tokens_for_routing(body)
        # Should count both tool calls: ~105K total with tiktoken
        assert tokens >= 55_000
        assert tokens <= 115_000

    def test_mixed_content_reasoning_tool_calls_counted(self):
        """Mixed content, reasoning, and tool calls should all be counted (LP-0MRDE669Y003V1SO)."""
        phrase = "test message "
        phrase2 = "test message content for token estimation "
        body = {
            "messages": [
                {"role": "user", "content": phrase * 6000},
                {
                    "role": "assistant",
                    "content": phrase * 6000,
                    "reasoning_content": phrase2 * 5000,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "test_func",
                                "arguments": phrase2 * 5000,
                            },
                        },
                    ],
                },
            ]
        }
        tokens = _estimate_prompt_tokens_for_routing(body)
        # Should count content + reasoning + tool_call args: ~144K total with tiktoken
        assert tokens >= 75_000
        assert tokens <= 155_000


# ---------------------------------------------------------------------------
# Threshold config tests (simplified - only cold threshold remains)
# ---------------------------------------------------------------------------


class TestLargeContextThresholdConfig:
    """Tests for the large-context threshold configuration parsing."""

    def test_threshold_disabled_when_zero(self):
        """Threshold of 0 should disable smart routing."""
        from proxy.provider import _get_large_context_threshold
        config = {"server": {"local_large_context_cold_cache_threshold": 0}}
        assert _get_large_context_threshold(config) == 0

    def test_threshold_disabled_when_absent(self):
        """Missing threshold config should return 0 (disabled)."""
        from proxy.provider import _get_large_context_threshold
        config = {"server": {}}
        assert _get_large_context_threshold(config) == 0

    def test_threshold_at_40000(self):
        """Threshold of 40000 should be parsed correctly."""
        from proxy.provider import _get_large_context_threshold
        config = {"server": {"local_large_context_cold_cache_threshold": 40000}}
        assert _get_large_context_threshold(config) == 40000

    def test_threshold_flat_config(self):
        """Flat (non-nested) config should also work for test compatibility."""
        from proxy.provider import _get_large_context_threshold
        config = {"local_large_context_cold_cache_threshold": 40000}
        assert _get_large_context_threshold(config) == 40000

    def test_threshold_legacy_key(self):
        """Legacy key (local_large_context_fallback_threshold) should still work."""
        from proxy.provider import _get_large_context_threshold
        config = {"server": {"local_large_context_fallback_threshold": 40000}}
        assert _get_large_context_threshold(config) == 40000


# ---------------------------------------------------------------------------
# Production wiring tests
# ---------------------------------------------------------------------------


class TestProductionWiring:
    """Tests for production wiring of cached_tokens ratio updates.

    Verifies that _update_session_and_slot calls update_cached_ratio after
    a successful slot save, and that _invalidate_session_and_slot clears
    cached ratio entries.
    (LP-0MRMMBZ7T007ER59)
    """

    def setup_method(self):
        from proxy.provider import _last_cached_ratio
        _last_cached_ratio.clear()

    def test_invalidation_clears_cached_ratio(self):
        """_invalidate_session_and_slot should clear _last_cached_ratio entries
        for the invalidated session (LP-0MRMMBZ7T007ER59)."""
        from proxy.provider import _last_cached_ratio, update_cached_ratio
        from proxy.session import _invalidate_session_and_slot

        # Set up cached ratio entries
        update_cached_ratio("Qwen3", "sess_a", cached_tokens=80, prompt_tokens=100)
        update_cached_ratio("gemma4", "sess_a", cached_tokens=10, prompt_tokens=100)
        update_cached_ratio("Qwen3", "sess_b", cached_tokens=50, prompt_tokens=100)
        assert len(_last_cached_ratio) == 3

        # Run invalidation (no slot_filename to avoid file I/O)
        import asyncio
        asyncio.run(_invalidate_session_and_slot(
            session_id="sess_a",
            reason="test_invalidation",
            slot_filename=None,
        ))

        # sess_a entries should be cleared; sess_b should remain
        assert ("Qwen3", "sess_a") not in _last_cached_ratio
        assert ("gemma4", "sess_a") not in _last_cached_ratio
        assert ("Qwen3", "sess_b") in _last_cached_ratio
        assert _last_cached_ratio[("Qwen3", "sess_b")] == 0.5

    def test_invalidation_no_session_id(self):
        """_invalidate_session_and_slot with None session_id should not crash."""
        from proxy.session import _invalidate_session_and_slot

        import asyncio
        # Should not raise
        asyncio.run(_invalidate_session_and_slot(
            session_id=None,
            reason="test_no_session",
            slot_filename=None,
        ))
