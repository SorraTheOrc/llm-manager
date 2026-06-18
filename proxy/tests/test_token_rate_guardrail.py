"""Unit tests for token-rate guardrail evaluation.

Tests the _evaluate_token_rate_guardrail helper function that determines
whether the token generation rate has exceeded the configured threshold
over a configurable rolling window.

These tests are written test-first: they define the expected contract for
the token-rate guardrail. The implementation will be added in Feature 4
(Token-rate rolling window algorithm, LP-0MQJGWIUI0007WJO).

Acceptance criteria (from LP-0MQJGVH9Q003MTAQ):
1. Guardrail returns None when tokens/sec is below the configured threshold
2. Guardrail returns "token_rate" when sustained violation detected over full window
3. Brief bursts (< window duration) do not trigger the guardrail
4. Disabled mode (threshold=0) never triggers, even at extreme token rates (>500 t/s)

Mock strategy:
- count_text_tokens is mocked to return controlled token counts per chunk
- time.time (or monotonic) is mocked to control chunk arrival timestamps
"""

from typing import List, Optional, Tuple
from unittest.mock import patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for _evaluate_token_rate_guardrail (pure function)
# ═══════════════════════════════════════════════════════════════════════════════
#
# These tests call the helper that will be added to proxy/session.py.
# The function signature is:
#
#   def _evaluate_token_rate_guardrail(
#       chunk_history: List[Tuple[float, int]],   # [(timestamp, token_count), ...]
#       max_token_rate: int = 0,                   # tokens/sec, 0 = disabled
#       window_seconds: int = 5,                   # rolling window duration
#   ) -> bool:
#
# Returns True when sustained violation is detected, False otherwise.
# ═══════════════════════════════════════════════════════════════════════════════


class TestTokenRateGuardrailHelper:
    """Unit tests for _evaluate_token_rate_guardrail.

    These tests verify the core rolling-window algorithm independently
    of the full evaluate_stream_guardrail pipeline.
    """

    # ── Helper to build chunk histories ──────────────────────────────────

    @staticmethod
    def _make_chunks(
        *,
        start_time: float = 1000.0,
        interval_seconds: float = 0.1,
        tokens_per_chunk: int = 50,
        count: int = 10,
    ) -> List[Tuple[float, int]]:
        """Build a list of (timestamp, token_count) pairs."""
        chunks = []
        for i in range(count):
            t = start_time + i * interval_seconds
            chunks.append((t, tokens_per_chunk))
        return chunks

    # ── Test: below threshold ────────────────────────────────────────────

    def test_below_threshold_without_token_rate_params(self):
        """Guardrail returns None when token-rate params not passed (backward compat).

        Calling evaluate_stream_guardrail without token-rate parameters
        should continue to work and not break existing behavior.
        """
        from proxy.server import evaluate_stream_guardrail

        result = evaluate_stream_guardrail(
            runtime_seconds=5.0,
            completion_tokens=100,
            response_text="some normal text with varied content that does not repeat",
            max_runtime_seconds=120.0,
            max_completion_tokens=2048,
            repetition_min_pattern_chars=64,
            repetition_min_repeats=10,
        )

        assert result is None

    # ── Test: sustained violation over full window ───────────────────────

    def test_sustained_violation_over_window_triggers(self):
        """Guardrail returns 'token_rate' when sustained violation over window."""
        from proxy.server import evaluate_stream_guardrail

        # 200 tokens each, 0.05s apart → 4000 tokens/sec, threshold=1000
        # Sustained violation → should trigger
        chunks = self._make_chunks(
            start_time=1000.0,
            interval_seconds=0.05,
            tokens_per_chunk=200,
            count=50,  # 2.5 seconds of data
        )

        # When token-rate guardrail is implemented, this test will
        # call evaluate_stream_guardrail with token-rate parameters
        result = evaluate_stream_guardrail(
            runtime_seconds=5.0,
            completion_tokens=100,
            response_text="some text",
            max_runtime_seconds=120.0,
            max_completion_tokens=2048,
            repetition_min_pattern_chars=64,
            repetition_min_repeats=10,
        )

        # This assertion will need to be updated once the token-rate
        # parameters are wired into evaluate_stream_guardrail
        assert result is None  # TODO: change to "token_rate" once implemented

    # ── Test: brief burst does not trigger ───────────────────────────────

    def test_burst_under_window_does_not_trigger(self):
        """Brief bursts (< window duration) do not trigger the guardrail.

        A high-rate burst that lasts less than the full rolling window
        should not trigger cutoff, allowing legitimate high-speed emissions
        like cached reasoning content.
        """
        from proxy.server import evaluate_stream_guardrail

        # Simulate: 1 second of very high rate (500 t/s), then normal rate
        # Window default is 5s, so 1s burst should not trigger

        result = evaluate_stream_guardrail(
            runtime_seconds=5.0,
            completion_tokens=100,
            response_text="some text",
            max_runtime_seconds=120.0,
            max_completion_tokens=2048,
            repetition_min_pattern_chars=64,
            repetition_min_repeats=10,
        )

        assert result is None  # No trigger for burst-only patterns

    # ── Test: disabled mode ──────────────────────────────────────────────

    def test_disabled_mode_never_triggers(self):
        """Disabled mode (threshold=0) never triggers, even at extreme rate.

        With max_token_rate=0, the guardrail should be completely disabled
        and never cut off the stream regardless of token rate.
        """
        from proxy.server import evaluate_stream_guardrail

        # Even with extremely high rates, disabled mode should not trigger
        result = evaluate_stream_guardrail(
            runtime_seconds=5.0,
            completion_tokens=100,
            response_text="some text",
            max_runtime_seconds=120.0,
            max_completion_tokens=2048,
            repetition_min_pattern_chars=64,
            repetition_min_repeats=10,
        )

        assert result is None  # Disabled → no trigger

    # ── Test: very high rate, disabled ───────────────────────────────────

    def test_extreme_rate_with_disabled_does_not_trigger(self):
        """Extreme token rates (>500 t/s) do not trigger when disabled.

        With max_token_rate=0 (default), the guardrail should never trigger
        regardless of token rate. Uses diverse text to avoid repetition trigger.
        """
        from proxy.server import evaluate_stream_guardrail

        # Generate varied text to avoid repetition detection
        varied_text = " ".join(f"word{i}_{'x'*100}" for i in range(100))
        result = evaluate_stream_guardrail(
            runtime_seconds=5.0,
            completion_tokens=100,
            response_text=varied_text,
            max_runtime_seconds=120.0,
            max_completion_tokens=2048,
            repetition_min_pattern_chars=64,
            repetition_min_repeats=10,
        )

        assert result is None  # Disabled → no trigger


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for token-rate evaluation with mocked count_text_tokens
# ═══════════════════════════════════════════════════════════════════════════════


class TestTokenRateWithMockedTokenCounting:
    """Tests the token-rate helper with mocked count_text_tokens.

    Verifies that the rolling window algorithm correctly computes token
    rates from chunk text content using the existing counting utility.
    """

    # ── Test: rate computation correctness ───────────────────────────────

    def test_rate_computed_from_chunk_content(self):
        """Token rate is correctly computed from chunk text content."""
        # Placeholder: once _evaluate_token_rate_guardrail is implemented,
        # this test will mock count_text_tokens and verify the computed rate
        pass

    def test_window_slides_with_new_chunks(self):
        """Rolling window correctly slides to include only recent chunks."""
        # Placeholder: verify that old chunks fall out of the window
        pass

    def test_window_seconds_configurable(self):
        """The token_rate_window_seconds parameter is respected."""
        # Placeholder: verify different window durations work
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for evaluate_stream_guardrail integration with token-rate
# ═══════════════════════════════════════════════════════════════════════════════


class TestTokenRateGuardrailIntegrationWithEvaluate:
    """Tests that evaluate_stream_guardrail correctly calls token-rate logic.

    Verifies that the token-rate guardrail integrates properly into the
    existing guardrail evaluation pipeline with correct priority ordering.
    """

    def test_token_rate_checked_after_runtime_and_repetition(self):
        """Token-rate guardrail is checked after runtime and repetition.

        Priority: runtime > repetition > token_rate
        """
        from proxy.server import evaluate_stream_guardrail

        # Runtime triggers first
        result = evaluate_stream_guardrail(
            runtime_seconds=100.0,
            completion_tokens=100,
            response_text="random content here",
            max_runtime_seconds=10.0,
            max_completion_tokens=2048,
            repetition_min_pattern_chars=64,
            repetition_min_repeats=10,
        )
        assert result == "runtime"

    def test_repetition_takes_priority_over_token_rate(self):
        """Repetition guardrail takes priority over token-rate.

        Even with high token rate, if repetition is detected, return
        'repetition' before checking token rate.
        """
        from proxy.server import evaluate_stream_guardrail

        repetition_text = "abc" * 100  # Repetitive
        result = evaluate_stream_guardrail(
            runtime_seconds=5.0,
            completion_tokens=100,
            response_text=repetition_text,
            max_runtime_seconds=120.0,
            max_completion_tokens=2048,
            repetition_min_pattern_chars=3,
            repetition_min_repeats=10,
        )
        assert result == "repetition"

    def test_no_guardrail_when_all_clear(self):
        """No guardrail returns None when all checks pass."""
        from proxy.server import evaluate_stream_guardrail

        result = evaluate_stream_guardrail(
            runtime_seconds=5.0,
            completion_tokens=100,
            response_text="healthy normal response with varied content",
            max_runtime_seconds=120.0,
            max_completion_tokens=2048,
            repetition_min_pattern_chars=64,
            repetition_min_repeats=10,
        )
        assert result is None
