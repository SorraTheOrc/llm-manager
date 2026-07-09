"""Tests for adaptive timeout calculation in the proxy server."""

import pytest
import proxy.server as server


class TestEstimatePromptTokens:
    """Tests for _estimate_prompt_tokens function."""

    def test_90k_token_with_new_per_token_rate(self):
        """AC 2: 90K-token request with per_token=0.015 must get >= 1050s timeout.

        ~360K chars / 4 = ~90K tokens.
        60 + 0.015 * 90000 = 1410s
        """
        large_content = "x" * 360_000
        body = {"messages": [{"role": "user", "content": large_content}]}
        timeout = server._compute_adaptive_timeout(
            body,
            base_timeout=60.0,
            per_token_timeout=0.015,
            max_timeout=1500.0,
        )
        assert timeout >= 1050.0, (
            f"Expected timeout >= 1050s for 90K tokens with per_token=0.015, got {timeout}"
        )

    def test_90k_token_does_not_exceed_max(self):
        """90K-token request with per_token=0.015 should NOT exceed max (1500).

        ~360K chars / 4 = ~90K tokens.
        60 + 0.015 * 90000 = 1410s < 1500, so not capped.
        """
        large_content = "x" * 360_000
        body = {"messages": [{"role": "user", "content": large_content}]}
        timeout = server._compute_adaptive_timeout(
            body,
            base_timeout=60.0,
            per_token_timeout=0.015,
            max_timeout=1500.0,
        )
        assert timeout <= 1500.0

    def test_old_per_token_insufficient_for_90k(self):
        """Regression: old per_token=0.01 gives ~960s for 90K, insufficient."""
        large_content = "x" * 360_000
        body = {"messages": [{"role": "user", "content": large_content}]}
        timeout = server._compute_adaptive_timeout(
            body,
            base_timeout=60.0,
            per_token_timeout=0.01,
            max_timeout=1500.0,
        )
        assert timeout < 1050.0, (
            f"Expected old per_token=0.01 to give < 1050s for 90K, got {timeout}"
        )

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


# ═══════════════════════════════════════════════════════════════════════════════
# LP-0MRDI0NQV000838X: Remote-specific timeout configuration keys
# ═══════════════════════════════════════════════════════════════════════════════
# Tests for _compute_request_timeout() with remote-specific override keys.
# ═══════════════════════════════════════════════════════════════════════════════


class TestRemoteSpecificTimeoutConfig:
    """Tests for remote-specific timeout config keys (LP-0MRDI0NQV000838X).

    Verifies that:
    1. Remote-specific keys (llama_remote_request_timeout_base_seconds and
       llama_remote_request_timeout_per_token_seconds) are used when configured.
    2. Fallback to local values when remote keys are not configured.
    3. Local path continues to use local keys when no remote keys are set.
    """

    def _make_body(self, content: str = "Hello") -> dict:
        return {"messages": [{"role": "user", "content": content}]}

    def _make_body_large(self, content_size: int = 100_000) -> dict:
        return {
            "messages": [{"role": "user", "content": "x" * content_size}]
        }

    # AC1: Remote-specific keys are used when configured

    def test_remote_uses_remote_specific_base_when_configured(self):
        """AC1: remote=True uses llama_remote_request_timeout_base_seconds when set."""
        from proxy.router_helpers import _compute_request_timeout

        config = {
            "llama_adaptive_timeout_enabled": True,
            "llama_adaptive_timeout_base_seconds": 60,
            "llama_adaptive_timeout_per_token_seconds": 0.015,
            "llama_remote_request_timeout_base_seconds": 30,
            "llama_remote_request_timeout_per_token_seconds": 0.01,
            "llama_request_timeout": 1500,
        }
        body = self._make_body()
        local_to = _compute_request_timeout(config, body, remote=False)
        remote_to = _compute_request_timeout(config, body, remote=True)

        # Remote with shorter base should produce a shorter timeout
        assert remote_to.read < local_to.read, (
            f"Expected remote timeout ({remote_to.read}) < local ({local_to.read}) "
            "when remote base is shorter"
        )

    def test_remote_uses_remote_specific_per_token_when_configured(self):
        """AC1: remote=True uses llama_remote_request_timeout_per_token_seconds."""
        from proxy.router_helpers import _compute_request_timeout

        config = {
            "llama_adaptive_timeout_enabled": True,
            "llama_adaptive_timeout_base_seconds": 60,
            "llama_adaptive_timeout_per_token_seconds": 0.015,
            "llama_remote_request_timeout_base_seconds": 60,
            "llama_remote_request_timeout_per_token_seconds": 0.005,
            "llama_request_timeout": 1500,
        }
        body = self._make_body_large()
        local_to = _compute_request_timeout(config, body, remote=False)
        remote_to = _compute_request_timeout(config, body, remote=True)

        # Remote with lower per_token should produce a shorter timeout for large prompts
        assert remote_to.read < local_to.read, (
            f"Expected remote timeout ({remote_to.read}) < local ({local_to.read}) "
            "when remote per_token is lower"
        )

    # AC2: Fallback to local values when remote keys are not configured

    def test_remote_falls_back_to_local_when_no_remote_keys(self):
        """AC2: Without remote keys, remote=True falls back to local values."""
        from proxy.router_helpers import _compute_request_timeout

        config = {
            "llama_adaptive_timeout_enabled": True,
            "llama_adaptive_timeout_base_seconds": 60,
            "llama_adaptive_timeout_per_token_seconds": 0.015,
            "llama_request_timeout": 1500,
            # No remote-specific keys
        }
        body = self._make_body_large()
        local_to = _compute_request_timeout(config, body, remote=False)
        remote_to = _compute_request_timeout(config, body, remote=True)

        # Without remote keys, both should produce the same timeout
        assert abs(remote_to.read - local_to.read) < 0.001, (
            f"Expected remote timeout ({remote_to.read}) == local ({local_to.read}) "
            "when no remote-specific keys are configured"
        )

    def test_remote_falls_back_to_local_when_empty_dict(self):
        """AC2: Empty server_config falls back to defaults for both paths."""
        from proxy.router_helpers import _compute_request_timeout

        config = {
            "llama_adaptive_timeout_enabled": True,
        }
        body = self._make_body()
        local_to = _compute_request_timeout(config, body, remote=False)
        remote_to = _compute_request_timeout(config, body, remote=True)

        # Both should fall back to the same code defaults (60 base, 0.01 per_token)
        assert remote_to.read > 0
        assert local_to.read > 0
        assert abs(remote_to.read - local_to.read) < 0.001, (
            "Remote and local should use the same defaults when no config"
        )

    # AC3: Local path continues to use local keys

    def test_local_ignores_remote_keys(self):
        """AC3: Local path (remote=False) ignores remote-specific keys."""
        from proxy.router_helpers import _compute_request_timeout

        config = {
            "llama_adaptive_timeout_enabled": True,
            "llama_adaptive_timeout_base_seconds": 60,
            "llama_adaptive_timeout_per_token_seconds": 0.01,
            "llama_remote_request_timeout_base_seconds": 999,
            "llama_remote_request_timeout_per_token_seconds": 0.5,
            "llama_request_timeout": 1500,
        }
        body = self._make_body()
        to = _compute_request_timeout(config, body, remote=False)

        # Local path should use local per_token (0.01), not remote (0.5)
        expected = 60.0 + (0.01 * 1)  # ~1 token
        assert abs(to.read - expected) < 5.0, (
            f"Expected local timeout ~{expected}, got {to.read}"
        )

    # AC1+AC3: Remote and local produce different results with different configs

    def test_remote_and_local_independence(self):
        """AC1+AC3: Remote and local timeout configs operate independently."""
        from proxy.router_helpers import _compute_request_timeout

        config = {
            "llama_adaptive_timeout_enabled": True,
            "llama_adaptive_timeout_base_seconds": 60,
            "llama_adaptive_timeout_per_token_seconds": 0.015,
            "llama_remote_request_timeout_base_seconds": 30,
            "llama_remote_request_timeout_per_token_seconds": 0.005,
            "llama_request_timeout": 1500,
        }
        body = self._make_body_large(200_000)
        local_to = _compute_request_timeout(config, body, remote=False)
        remote_to = _compute_request_timeout(config, body, remote=True)

        # Both should be meaningful but different
        assert local_to.read > 0
        assert remote_to.read > 0
        assert remote_to.read < local_to.read, (
            f"Expected remote ({remote_to.read}) < local ({local_to.read}) "
            "with different configs"
        )

    # Adaptive disabled fallback

    def test_adaptive_disabled_uses_fixed_timeout(self):
        """When adaptive timeout is disabled, remote=True still uses fixed timeout."""
        from proxy.router_helpers import _compute_request_timeout

        config = {
            "llama_adaptive_timeout_enabled": False,
            "llama_request_timeout": 300,
        }
        body = self._make_body_large()
        to = _compute_request_timeout(config, body, remote=True)

        assert to.read == 300, (
            f"Expected fixed timeout 300 when adaptive disabled, got {to.read}"
        )

    def test_adaptive_disabled_local_uses_fixed_timeout(self):
        """When adaptive timeout is disabled, local path uses fixed timeout."""
        from proxy.router_helpers import _compute_request_timeout

        config = {
            "llama_adaptive_timeout_enabled": False,
            "llama_request_timeout": 300,
        }
        body = self._make_body_large()
        to = _compute_request_timeout(config, body, remote=False)

        assert to.read == 300, (
            f"Expected fixed timeout 300 when adaptive disabled, got {to.read}"
        )

    # Default per_token values same for remote and local at code level

    def test_remote_and_local_defaults_match_when_no_config(self):
        """When no config keys are set, remote and local defaults match.

        Both remote and local fall back to the same code defaults:
        base=60s, per_token=0.01s, max=1500s.
        """
        from proxy.router_helpers import _compute_request_timeout

        config = {
            "llama_adaptive_timeout_enabled": True,
        }
        body = self._make_body()
        local_to = _compute_request_timeout(config, body, remote=False)
        remote_to = _compute_request_timeout(config, body, remote=True)

        # Both should use the same defaults -> same timeout
        assert abs(local_to.read - remote_to.read) < 0.001, (
            f"Expected local ({local_to.read}) == remote ({remote_to.read}) "
            "with no config keys"
        )

    def test_remote_defaults_match_via_config(self):
        """When config has explicit local values, remote falls back to them."""
        from proxy.router_helpers import _compute_request_timeout

        config = {
            "llama_adaptive_timeout_enabled": True,
            "llama_adaptive_timeout_base_seconds": 120,
            "llama_adaptive_timeout_per_token_seconds": 0.02,
            "llama_request_timeout": 1500,
        }
        body = self._make_body_large()
        local_to = _compute_request_timeout(config, body, remote=False)
        remote_to = _compute_request_timeout(config, body, remote=True)

        # Remote falls back to local config values
        assert abs(local_to.read - remote_to.read) < 0.001, (
            f"Expected remote ({remote_to.read}) == local ({local_to.read}) "
            "when remote falls back to local config"
        )
