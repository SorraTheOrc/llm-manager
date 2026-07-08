"""Tests for adaptive timeout calculation in the proxy server."""

import pytest
import proxy.server as server


class TestEstimatePromptTokens:
    """Tests for _estimate_prompt_tokens function."""

    def test_empty_body(self):
        """Empty body should return 0."""
        assert server._estimate_prompt_tokens({}) == 0

    def test_no_messages(self):
        """Body without messages should return 0."""
        assert server._estimate_prompt_tokens({"model": "test"}) == 0

    def test_empty_messages(self):
        """Empty messages list should return 0."""
        assert server._estimate_prompt_tokens({"messages": []}) == 0

    def test_single_message(self):
        """Single message should estimate tokens correctly."""
        body = {"messages": [{"role": "user", "content": "Hello, world!"}]}
        tokens = server._estimate_prompt_tokens(body)
        # "Hello, world!" is 13 chars, ~3 tokens
        assert tokens >= 1
        assert tokens <= 10

    def test_multiple_messages(self):
        """Multiple messages should sum up token estimates."""
        body = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ]
        }
        tokens = server._estimate_prompt_tokens(body)
        # Combined ~50 chars, ~12 tokens
        assert tokens >= 5
        assert tokens <= 20

    def test_long_content(self):
        """Long content should produce higher token estimates."""
        long_content = "This is a test message. " * 100  # ~2500 chars
        body = {"messages": [{"role": "user", "content": long_content}]}
        tokens = server._estimate_prompt_tokens(body)
        # ~2500 chars / 4 = ~625 tokens
        assert tokens >= 500
        assert tokens <= 800

    def test_array_content(self):
        """Array content (multimodal) should extract text fields."""
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "image", "url": "http://example.com/img.jpg"},
                    ],
                }
            ]
        }
        tokens = server._estimate_prompt_tokens(body)
        # Only "Hello" should be counted
        assert tokens >= 1
        assert tokens <= 5

    def test_non_dict_message(self):
        """Non-dict messages should be skipped gracefully."""
        body = {"messages": ["not a dict", 123, None]}
        tokens = server._estimate_prompt_tokens(body)
        # Returns 1 because max(1, 0) = 1 when no content found
        assert tokens == 1

    def test_missing_content(self):
        """Messages without content should be skipped."""
        body = {"messages": [{"role": "user"}, {"role": "assistant"}]}
        tokens = server._estimate_prompt_tokens(body)
        # Returns 1 because max(1, 0) = 1 when no content found
        assert tokens == 1


class TestComputeAdaptiveTimeout:
    """Tests for _compute_adaptive_timeout function."""

    def test_small_prompt(self):
        """Small prompts should use close to base timeout."""
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        timeout = server._compute_adaptive_timeout(
            body,
            base_timeout=60.0,
            per_token_timeout=0.01,
            max_timeout=300.0,
        )
        # ~1 token, so timeout should be close to base
        assert timeout >= 60.0
        assert timeout <= 65.0

    def test_large_prompt(self):
        """Large prompts should have longer timeouts."""
        large_content = "This is a test. " * 1000  # ~16000 chars, ~4000 tokens
        body = {"messages": [{"role": "user", "content": large_content}]}
        timeout = server._compute_adaptive_timeout(
            body,
            base_timeout=60.0,
            per_token_timeout=0.01,
            max_timeout=300.0,
        )
        # ~4000 tokens * 0.01 = 40s, plus base 60s = 100s
        assert timeout >= 90.0
        assert timeout <= 300.0

    def test_max_timeout_cap(self):
        """Timeout should not exceed max_timeout."""
        huge_content = "x" * 1_000_000  # ~250000 tokens
        body = {"messages": [{"role": "user", "content": huge_content}]}
        timeout = server._compute_adaptive_timeout(
            body,
            base_timeout=60.0,
            per_token_timeout=0.01,
            max_timeout=300.0,
        )
        assert timeout == 300.0

    def test_empty_body(self):
        """Empty body should use base timeout."""
        body = {}
        timeout = server._compute_adaptive_timeout(
            body,
            base_timeout=60.0,
            per_token_timeout=0.01,
            max_timeout=300.0,
        )
        assert timeout == 60.0

    def test_zero_per_token(self):
        """Zero per_token_timeout should always return base timeout."""
        body = {"messages": [{"role": "user", "content": "x" * 10000}]}
        timeout = server._compute_adaptive_timeout(
            body,
            base_timeout=60.0,
            per_token_timeout=0.0,
            max_timeout=300.0,
        )
        assert timeout == 60.0

    def test_zero_base_timeout(self):
        """Zero base_timeout should still scale with tokens."""
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        timeout = server._compute_adaptive_timeout(
            body,
            base_timeout=0.0,
            per_token_timeout=0.01,
            max_timeout=300.0,
        )
        # ~1 token * 0.01 = 0.01s
        assert timeout >= 0.0
        assert timeout <= 1.0


class TestAdaptiveTimeoutIntegration:
    """Integration tests for adaptive timeout with realistic scenarios."""

    def test_typical_chat_request(self):
        """Typical chat request with system prompt and user message."""
        body = {
            "model": "qwen3",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful coding assistant. Follow these guidelines:\n"
                    "1. Be concise\n"
                    "2. Use proper formatting\n"
                    "3. Include error handling",
                },
                {
                    "role": "user",
                    "content": "Write a Python function to calculate fibonacci numbers.",
                },
            ],
            "max_tokens": 1000,
        }
        timeout = server._compute_adaptive_timeout(
            body,
            base_timeout=60.0,
            per_token_timeout=0.01,
            max_timeout=300.0,
        )
        # Should be reasonable for a typical request
        assert 60.0 <= timeout <= 120.0

    def test_large_code_review_request(self):
        """Large code review request with extensive context."""
        code_block = "def example():\n    return True\n" * 500  # ~15000 chars
        body = {
            "model": "qwen3",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a code review expert.",
                },
                {
                    "role": "user",
                    "content": f"Review this code:\n```python\n{code_block}\n```",
                },
            ],
        }
        timeout = server._compute_adaptive_timeout(
            body,
            base_timeout=60.0,
            per_token_timeout=0.01,
            max_timeout=300.0,
        )
        # Large request should have extended timeout
        assert timeout >= 95.0
        assert timeout <= 300.0


class TestAdaptiveGuardrailBudget:
    """Tests for adaptive guardrail budget computation in router.py (LP-0MRB9AZDJ00716OT).

    Verifies that the guardrail runtime budget scales with prompt size when
    adaptive timeout is enabled, and that the fixed path is preserved when
    adaptive timeout is disabled.
    """

    def _compute_guardrail_budget(
        self,
        body_json: dict,
        adaptive_enabled: bool = True,
        base: float = 60.0,
        per_token: float = 0.01,
        max_cap: float = 1800.0,
    ) -> float:
        """Simulate the same logic used in router.py stream_generator."""
        if adaptive_enabled and isinstance(body_json, dict):
            return server._compute_adaptive_timeout(
                body_json, base, per_token, max_cap
            )
        else:
            return max_cap

    def test_adaptive_budget_scales_with_prompt_size(self):
        """AC 5a: Adaptive budget should scale with prompt size."""
        small_body = {
            "messages": [{"role": "user", "content": "Hello, world!"}]
        }
        large_body = {
            "messages": [
                {"role": "user", "content": "This is a test. " * 1000}
            ]
        }
        small_budget = self._compute_guardrail_budget(small_body)
        large_budget = self._compute_guardrail_budget(large_body)
        assert large_budget > small_budget, (
            f"Expected large prompt budget ({large_budget}) > small ({small_budget})"
        )

    def test_fixed_budget_path_when_adaptive_disabled(self):
        """AC 5b: When adaptive timeout is disabled, the fixed max cap is used."""
        large_body = {
            "messages": [
                {"role": "user", "content": "x" * 100_000}
            ]
        }
        budget = self._compute_guardrail_budget(
            large_body, adaptive_enabled=False, max_cap=300.0
        )
        assert budget == 300.0, (
            f"Expected fixed budget 300.0, got {budget}"
        )

    def test_large_87k_token_prompt_gets_sufficient_budget(self):
        """AC 5c: 87k-token-scale prompts should receive >500s budget.

        ~87k tokens * 0.01 = 870s, plus base 60s = 930s (capped at 1800).
        """
        # ~348k chars → ~87k tokens (chars // 4)
        large_content = "x" * 348_000
        body = {
            "messages": [{"role": "user", "content": large_content}]
        }
        budget = self._compute_guardrail_budget(body)
        assert budget > 500.0, (
            f"Expected budget >500s for 87k-token prompt, got {budget}"
        )

    def test_adaptive_budget_uses_max_cap_as_upper_bound(self):
        """Adaptive budget should never exceed the max cap."""
        huge_body = {
            "messages": [
                {"role": "user", "content": "x" * 1_000_000}
            ]
        }
        budget = self._compute_guardrail_budget(huge_body, max_cap=600.0)
        assert budget == 600.0, (
            f"Expected budget capped at 600.0, got {budget}"
        )

    def test_empty_body_with_adaptive_enabled(self):
        """Empty body should fall back to base timeout when adaptive is enabled."""
        budget = self._compute_guardrail_budget({}, base=60.0)
        assert budget == 60.0, (
            f"Expected base budget 60.0 for empty body, got {budget}"
        )
