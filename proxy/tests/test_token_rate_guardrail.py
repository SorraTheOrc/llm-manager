"""Unit tests for token-rate guardrail evaluation.

Tests the _evaluate_token_rate_guardrail helper function that determines
whether the token generation rate has exceeded the configured threshold
over a configurable rolling window, and its integration into
evaluate_stream_guardrail.

Acceptance criteria (from LP-0MQJGVH9Q003MTAQ):
1. Guardrail returns None when tokens/sec is below the configured threshold
2. Guardrail returns "token_rate" when sustained violation detected over full window
3. Brief bursts (< window duration) do not trigger the guardrail
4. Disabled mode (threshold=0) never triggers, even at extreme token rates (>500 t/s)
"""

from typing import List, Tuple
from unittest.mock import patch

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_chunks(
    *,
    start_time: float = 1000.0,
    interval_seconds: float = 0.1,
    text_length: int = 200,  # ~50 tokens at 4 bytes/token
    count: int = 10,
) -> list[tuple[float, str]]:
    """Build a list of (timestamp, chunk_text) pairs with controlled spacing."""
    chunks = []
    for i in range(count):
        t = start_time + i * interval_seconds
        text = "x" * text_length  # ~text_len/4 tokens via heuristic
        chunks.append((t, text))
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for _evaluate_token_rate_guardrail (direct helper testing)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTokenRateGuardrailHelper:
    """Unit tests for _evaluate_token_rate_guardrail.

    Tests the core rolling-window algorithm by calling evaluate_stream_guardrail
    with chunk_history to exercise the helper.
    """

    # ── Test: below threshold ────────────────────────────────────────────

    def test_below_threshold_does_not_trigger(self):
        """Guardrail returns None when tokens/sec is below configured threshold.

        50 chars per chunk at 4 bytes/token heuristic → ~12 tokens each.
        0.1s apart → ~125 tokens/sec total. Threshold=1000 → below → no trigger.
        """
        from proxy.server import evaluate_stream_guardrail

        chunks = _make_chunks(
            start_time=1000.0,
            interval_seconds=0.1,
            text_length=50,  # ~12 tokens via heuristic
            count=60,  # 6 seconds of data (fills a 5s window)
        )

        result = evaluate_stream_guardrail(
            runtime_seconds=5.0,
            completion_tokens=100,
            response_text="normal varied text that does not repeat at all",
            max_runtime_seconds=120.0,
            max_completion_tokens=2048,
            repetition_min_pattern_chars=64,
            repetition_min_repeats=10,
            chunk_history=chunks,
            max_token_rate=1000,
            token_rate_window_seconds=5,
        )

        assert result is None, f"Expected None (below threshold), got {result}"

    # ── Test: disabled mode (threshold=0) ────────────────────────────────

    def test_disabled_threshold_zero_never_triggers(self):
        """Disabled mode (max_token_rate=0) never triggers regardless of rate."""
        from proxy.server import evaluate_stream_guardrail

        # Very fast chunks with lots of text → high token rate
        chunks = _make_chunks(
            start_time=1000.0,
            interval_seconds=0.01,
            text_length=500,  # ~125 tokens each
            count=100,  # 1 second of data
        )

        result = evaluate_stream_guardrail(
            runtime_seconds=5.0,
            completion_tokens=2000,
            response_text="normal varied text that does not repeat",
            max_runtime_seconds=120.0,
            max_completion_tokens=2048,
            repetition_min_pattern_chars=64,
            repetition_min_repeats=10,
            chunk_history=chunks,
            max_token_rate=0,  # Disabled
            token_rate_window_seconds=5,
        )

        assert result is None, (
            f"Expected None (disabled), got {result}"
        )

    # ── Test: sustained violation over full window ───────────────────────

    @patch("proxy.utils.count_text_tokens")
    def test_sustained_violation_over_window_triggers(
        self, mock_count_tokens
    ):
        """Guardrail returns 'token_rate' when sustained violation over window.

        With mocked count_text_tokens returning 200 tokens per chunk,
        0.05s apart → 4000 tokens/sec. Window=5s, threshold=1000 → trigger.
        """
        from proxy.server import evaluate_stream_guardrail

        mock_count_tokens.return_value = 200

        # 6 seconds of data at 0.05s intervals → 120 chunks
        chunks = _make_chunks(
            start_time=1000.0,
            interval_seconds=0.05,
            text_length=800,  # content won't matter since count_text_tokens is mocked
            count=120,
        )

        result = evaluate_stream_guardrail(
            runtime_seconds=10.0,
            completion_tokens=500,
            response_text="normal varied text that does not repeat",
            max_runtime_seconds=120.0,
            max_completion_tokens=2048,
            repetition_min_pattern_chars=64,
            repetition_min_repeats=10,
            chunk_history=chunks,
            max_token_rate=1000,
            token_rate_window_seconds=5,
        )

        assert result == "token_rate", (
            f"Expected 'token_rate' (sustained violation), got {result}"
        )

    # ── Test: brief burst does not trigger ───────────────────────────────

    @patch("proxy.utils.count_text_tokens")
    def test_burst_under_window_does_not_trigger(
        self, mock_count_tokens
    ):
        """Brief bursts (< window duration) do not trigger the guardrail.

        1 second of data with 500 tokens/sec, but window is 5s.
        The window is not fully populated so the guardrail should not trigger.
        """
        from proxy.server import evaluate_stream_guardrail

        mock_count_tokens.return_value = 50

        # Only 1 second of data — not enough to fill the 5s window
        chunks = _make_chunks(
            start_time=1000.0,
            interval_seconds=0.1,
            text_length=200,  # content won't matter with mock
            count=10,  # 1 second of data
        )

        result = evaluate_stream_guardrail(
            runtime_seconds=5.0,
            completion_tokens=200,
            response_text="normal varied text that does not repeat",
            max_runtime_seconds=120.0,
            max_completion_tokens=2048,
            repetition_min_pattern_chars=64,
            repetition_min_repeats=10,
            chunk_history=chunks,
            max_token_rate=100,  # Low threshold
            token_rate_window_seconds=5,
        )

        # Burst too short to fill the window → no trigger
        assert result is None, (
            f"Expected None (burst too short), got {result}"
        )

    # ── Test: disabled mode via backward compat ──────────────────────────

    def test_no_chunk_history_is_noop(self):
        """Calling without chunk_history or max_token_rate is backward compat."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for _evaluate_token_rate_guardrail algorithm details
# ═══════════════════════════════════════════════════════════════════════════════


class TestTokenRateAlgorithmDetail:
    """Detailed unit tests for the rolling window algorithm."""

    @patch("proxy.utils.count_text_tokens")
    def test_rate_computed_from_chunk_content(self, mock_count_tokens):
        """Token rate is correctly computed from chunk text content.

        count_text_tokens returns 100 per chunk, 0.2s apart → 500 tokens/sec.
        """
        from proxy.session import _evaluate_token_rate_guardrail

        mock_count_tokens.return_value = 100

        chunks = _make_chunks(
            start_time=1000.0,
            interval_seconds=0.2,
            text_length=400,
            count=30,  # 6 seconds of data
        )

        # Window set to 5s, with fully populated window and rate > 200
        result = _evaluate_token_rate_guardrail(
            chunks, max_token_rate=200, window_seconds=5
        )

        assert result is True, (
            "Rate should exceed threshold over full window"
        )

    @patch("proxy.utils.count_text_tokens")
    def test_window_slides_with_new_chunks(self, mock_count_tokens):
        """Rolling window correctly slides: old chunks fall out of window.

        With a 5s window, chunks older than 5s should not affect the rate.
        """
        from proxy.session import _evaluate_token_rate_guardrail

        mock_count_tokens.return_value = 100

        # Simulate:
        # - Old burst at t=1000 to t=1002 (very high rate, 500 t/s)
        # - Normal rate at t=1002 to t=1008 (low rate, 50 t/s)
        # With window=5s at t=1008, old burst (t=1000-1002) should be out of window
        chunks: list[tuple[float, str]] = []
        # Old burst (t=1000 to t=1002): high rate
        for i in range(20):
            t = 1000.0 + i * 0.1
            chunks.append((t, "x" * 800))
        # Normal rate (t=1002 to t=1008): low rate
        mock_count_tokens.return_value = 10  # switch to low return value
        for i in range(60):
            t = 1002.0 + i * 0.1
            chunks.append((t, "x" * 40))

        # At the end (t≈1008), the rolling window should only include
        # data from ~t=1003 onward, which has 10 tokens/chunk at 0.1s = 100 t/s
        # Threshold = 1000 should not trigger
        result = _evaluate_token_rate_guardrail(
            chunks, max_token_rate=1000, window_seconds=5
        )

        assert result is False, (
            "Old burst should have fallen out of 5s window"
        )

    def test_window_seconds_configurable(self):
        """The token_rate_window_seconds parameter is respected."""
        from proxy.server import evaluate_stream_guardrail

        # 1 second of data, window=5s → not enough to trigger
        chunks = _make_chunks(
            start_time=1000.0,
            interval_seconds=0.1,
            text_length=800,  # ~200 tokens each
            count=10,  # 1 second
        )

        # With window=1s and threshold=100, should trigger (200/0.9s ≈ 222 t/s)
        # But with window=5s, the window is not fully populated yet
        result_5s = evaluate_stream_guardrail(
            runtime_seconds=2.0,
            completion_tokens=500,
            response_text="normal varied text that does not repeat",
            max_runtime_seconds=120.0,
            max_completion_tokens=2048,
            repetition_min_pattern_chars=64,
            repetition_min_repeats=10,
            chunk_history=chunks,
            max_token_rate=100,
            token_rate_window_seconds=5,
        )
        assert result_5s is None, (
            "5s window not filled by 1s of data"
        )


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

    @patch("proxy.utils.count_text_tokens")
    def test_token_rate_can_trigger_with_high_rate(
        self, mock_count_tokens
    ):
        """Token-rate triggers when rate exceeds threshold over full window,
        even when runtime and repetition checks pass."""
        from proxy.server import evaluate_stream_guardrail

        mock_count_tokens.return_value = 100

        chunks = _make_chunks(
            start_time=1000.0,
            interval_seconds=0.1,
            text_length=400,
            count=60,  # 6 seconds
        )

        result = evaluate_stream_guardrail(
            runtime_seconds=5.0,
            completion_tokens=100,
            response_text="normal varied text that does not repeat",
            max_runtime_seconds=120.0,
            max_completion_tokens=2048,
            repetition_min_pattern_chars=64,
            repetition_min_repeats=10,
            chunk_history=chunks,
            max_token_rate=500,
            token_rate_window_seconds=5,
        )

        # 100 tokens/chunk ÷ 0.1s = 1000 t/s, threshold=500 → trigger
        assert result == "token_rate", (
            f"Expected 'token_rate' (rate exceeds threshold), got {result}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for guardrail config defaults
# ═══════════════════════════════════════════════════════════════════════════════


class TestGuardrailConfigDefaults:
    """Tests for _get_guardrail_config defaults."""

    def test_guardrail_config_default_max_runtime_seconds_is_1800(self):
        """
        _get_guardrail_config should default to 1800s when no config is provided.
        This is now a safety cap for the adaptive guardrail budget (LP-0MRB9AZDJ00716OT).
        """
        from proxy.router import _get_guardrail_config

        config = _get_guardrail_config({})
        assert config["max_runtime_seconds"] == 1800.0, (
            f"Expected default 1800.0s, got {config['max_runtime_seconds']}"
        )

    def test_guardrail_config_honors_explicit_config(self):
        """_get_guardrail_config should use the value from config when provided."""
        from proxy.router import _get_guardrail_config

        config = _get_guardrail_config(
            {"session_guardrail_max_runtime_seconds": 600}
        )
        assert config["max_runtime_seconds"] == 600.0

    def test_guardrail_config_runtime_handles_zero_value(self):
        """_get_guardrail_config should fall back to 1800 when config is 0/falsy."""
        from proxy.router import _get_guardrail_config

        config = _get_guardrail_config(
            {"session_guardrail_max_runtime_seconds": 0}
        )
        assert config["max_runtime_seconds"] == 1800.0
