"""Tests for smart routing logic that bypasses local for large-context
requests when the slot cache is invalidated (LP-0MRCSSBTM002NK3B)."""

import pytest
from unittest.mock import MagicMock, patch

from proxy.provider import (
    _model_cache_cold,
    mark_model_cache_cold,
    clear_model_cache_cold,
    is_model_cache_cold,
    _estimate_prompt_tokens_for_routing,
)


class TestModelCacheColdTracking:
    """Tests for the in-memory cache-invalidation tracking."""

    def setup_method(self):
        _model_cache_cold.clear()

    def test_initial_state_is_warm(self):
        """All models start with cache considered valid (warm)."""
        assert is_model_cache_cold("Qwen3") is False
        assert is_model_cache_cold("gemma4") is False
        assert is_model_cache_cold("nonexistent") is False

    def test_mark_cache_cold(self):
        """Marking a model as cold should be reflected in is_model_cache_cold."""
        mark_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is True

    def test_mark_cache_cold_twice_is_idempotent(self):
        """Marking a model cold multiple times should not raise."""
        mark_model_cache_cold("Qwen3")
        mark_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is True

    def test_clear_cache_cold(self):
        """Clearing cache-cold for a model should return it to warm."""
        mark_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is True
        clear_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is False

    def test_clear_cache_cold_idempotent(self):
        """Clearing an already-warm model should not raise."""
        clear_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is False

    def test_multiple_models_independent(self):
        """Cache states for different models should be independent."""
        mark_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is True
        assert is_model_cache_cold("gemma4") is False
        clear_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is False
        assert is_model_cache_cold("gemma4") is False


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
        """A large message (e.g., 90K tokens) should yield a large estimate.

        ~360K chars / 4 = ~90K tokens.
        """
        large_content = "x" * 360_000
        body = {"messages": [{"role": "user", "content": large_content}]}
        tokens = _estimate_prompt_tokens_for_routing(body)
        assert tokens >= 80_000
        assert tokens <= 100_000

    def test_40k_threshold_crossing(self):
        """A message at ~41K tokens should be detected as above 40K threshold."""
        content = "x" * 164_000  # ~41K tokens
        body = {"messages": [{"role": "user", "content": content}]}
        tokens = _estimate_prompt_tokens_for_routing(body)
        assert tokens > 40_000

    def test_below_40k(self):
        """A message at ~10K tokens should be below 40K threshold."""
        content = "x" * 40_000  # ~10K tokens
        body = {"messages": [{"role": "user", "content": content}]}
        tokens = _estimate_prompt_tokens_for_routing(body)
        assert tokens < 40_000

    def test_system_message_ignored(self):
        """System messages should NOT be counted in routing estimate.

        The routing estimate is for the user/assistant context messages only,
        as system prompts are typically small and not the driver of large-context
        timeouts.
        """
        body = {
            "messages": [
                {"role": "system", "content": "x" * 100_000},  # 25K tokens system
                {"role": "user", "content": "Hello"},
            ]
        }
        tokens = _estimate_prompt_tokens_for_routing(body)
        # Should be small (just "Hello"), not large (system is excluded)
        assert 1 <= tokens <= 10

    def test_combined_user_assistant_messages_counted(self):
        """Both user and assistant messages should be counted."""
        body = {
            "messages": [
                {"role": "user", "content": "x" * 80_000},  # ~20K
                {"role": "assistant", "content": "y" * 80_000},  # ~20K
            ]
        }
        tokens = _estimate_prompt_tokens_for_routing(body)
        # Should count both: ~40K total
        assert tokens >= 35_000
        assert tokens <= 45_000

    def test_array_content_counts_text_only(self):
        """Array content (multimodal) should count text fields only."""
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "x" * 160_000},  # ~40K
                        {"type": "image", "url": "http://example.com/img.jpg"},
                    ],
                }
            ]
        }
        tokens = _estimate_prompt_tokens_for_routing(body)
        # Should count only the text part
        assert tokens >= 35_000
        assert tokens <= 45_000


class TestSmartRoutingThresholdConfig:
    """Tests for the threshold configuration parsing."""

    def test_threshold_disabled_when_zero(self):
        """Threshold of 0 should disable smart routing."""
        from proxy.provider import _get_large_context_fallback_threshold
        config = {"server": {"local_large_context_fallback_threshold": 0}}
        assert _get_large_context_fallback_threshold(config) == 0

    def test_threshold_disabled_when_absent(self):
        """Missing threshold config should disable smart routing (returns 0)."""
        from proxy.provider import _get_large_context_fallback_threshold
        config = {"server": {}}
        assert _get_large_context_fallback_threshold(config) == 0

    def test_threshold_at_40000(self):
        """Threshold of 40000 should be parsed correctly."""
        from proxy.provider import _get_large_context_fallback_threshold
        config = {"server": {"local_large_context_fallback_threshold": 40000}}
        assert _get_large_context_fallback_threshold(config) == 40000

    def test_threshold_flat_config(self):
        """Flat (non-nested) config should also work for test compatibility."""
        from proxy.provider import _get_large_context_fallback_threshold
        config = {"local_large_context_fallback_threshold": 40000}
        assert _get_large_context_fallback_threshold(config) == 40000


class TestSmartRoutingDecision:
    """Tests for the smart routing decision logic.

    Verifies that when cache is cold and tokens > threshold, local is skipped;
    otherwise local routing proceeds normally.
    """

    def setup_method(self):
        _model_cache_cold.clear()

    def test_warm_cache_small_request_routes_local(self):
        """AC 4: Warm cache + <=40K tokens → routes local (normal)."""
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        threshold = 40000
        cache_cold = False
        should_skip = _should_skip_local(cache_cold, body, threshold)
        assert should_skip is False

    def test_warm_cache_large_request_routes_local(self):
        """AC 1 & 4: Warm cache + >40K tokens → routes local (normal)."""
        body = {"messages": [{"role": "user", "content": "x" * 200_000}]}
        threshold = 40000
        cache_cold = False
        should_skip = _should_skip_local(cache_cold, body, threshold)
        assert should_skip is False

    def test_cold_cache_small_request_routes_local(self):
        """AC 4: Cold cache + <=40K tokens → routes local (normal)."""
        mark_model_cache_cold("Qwen3")
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        threshold = 40000
        should_skip = _should_skip_local(is_model_cache_cold("Qwen3"), body, threshold)
        assert should_skip is False

    def test_cold_cache_large_request_skips_local(self):
        """AC 3: Cold cache + >40K tokens → skips local."""
        mark_model_cache_cold("Qwen3")
        body = {"messages": [{"role": "user", "content": "x" * 200_000}]}
        threshold = 40000
        should_skip = _should_skip_local(is_model_cache_cold("Qwen3"), body, threshold)
        assert should_skip is True

    def test_threshold_disabled_cold_cache_large_request_routes_local(self):
        """Threshold=0 (disabled) → always routes local regardless of cache state."""
        mark_model_cache_cold("Qwen3")
        body = {"messages": [{"role": "user", "content": "x" * 200_000}]}
        threshold = 0
        should_skip = _should_skip_local(is_model_cache_cold("Qwen3"), body, threshold)
        assert should_skip is False

    def test_cold_cache_exact_threshold_does_not_skip(self):
        """Exactly at threshold (<=) should NOT skip local."""
        mark_model_cache_cold("Qwen3")
        # 160_000 chars / 4 = 40_000 tokens
        body = {"messages": [{"role": "user", "content": "x" * 160_000}]}
        threshold = 40000
        should_skip = _should_skip_local(is_model_cache_cold("Qwen3"), body, threshold)
        assert should_skip is False

    def test_cold_cache_just_above_threshold_skips(self):
        """Just above threshold (>40K) should skip local when cache is cold."""
        mark_model_cache_cold("Qwen3")
        # 164_000 chars / 4 = 41_000 tokens
        body = {"messages": [{"role": "user", "content": "x" * 164_000}]}
        threshold = 40000
        should_skip = _should_skip_local(is_model_cache_cold("Qwen3"), body, threshold)
        assert should_skip is True


# The function tested above lives in provider.py; we import it here.
# It's defined for test purposes - the real implementation is in provider.py
from proxy.provider import _should_skip_local
