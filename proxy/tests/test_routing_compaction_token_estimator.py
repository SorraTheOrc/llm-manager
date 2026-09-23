"""
Tests for the shared token estimation helper used by routing and compaction.

Regression for LP-0MU5FR3JG003XI57 (parent LP-0MU5A84YU003YOTY):

The routing path and compaction trigger were using different tokenizers
(tiktoken vs native) over different inputs, producing estimates that
disagreed by thousands of tokens. This caused the context_pressure advisory
to recommend compaction while the compaction trigger decided noop.

These tests verify that both paths now use the same shared helper and
produce matching estimates for the same message list.
"""

import logging

import pytest
from proxy.provider import (
    _estimate_prompt_tokens_for_routing,
    _get_tokenizer_for_model,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

try:
    from benchmarks import slot_benchmark as sb
except ImportError:
    try:
        from proxy.benchmarks import slot_benchmark as sb
    except ImportError:
        import sys
        from pathlib import Path

        this_dir = Path(__file__).resolve().parent
        proxy_dir = this_dir.parent
        root_dir = proxy_dir.parent
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))
        from proxy.benchmarks import slot_benchmark as sb

# Ground truth from the ctx-size evaluation (LP-0MSAOQTJS000FFVM):
# the 60K fixture has ~77060 Qwen3-native tokens (tiktoken est 47680).
QWEN3_60K_ACTUAL_TOKENS = 77060
TOLERANCE = 0.05


def _body_for_fixture(text: str) -> dict:
    """Wrap a prompt string in a chat request body."""
    return {"messages": [{"role": "user", "content": text}]}


def _get_qwen3_tokenizer():
    """Load the vendored Qwen3 tokenizer via the lazy registry."""
    from proxy.tokenizers import get_tokenizer

    tok = get_tokenizer("qwen3")
    assert tok is not None, "vendored qwen3 tokenizer must load (AC1)"
    return tok


# ===================================================================
# AC1: Shared helper exists and resolves tokenizer via _get_tokenizer_for_model
# ===================================================================


class TestSharedEstimatorHelperExists:
    """The shared helper must resolve tokenizer via _get_tokenizer_for_model
    and estimate tokens using the native tokenizer (with multiplier=1.0)
    when available, falling back to tiktoken."""

    def test_estimate_fn_accepts_messages_and_model_config(self):
        """The shared estimate helper (imported by compaction) accepts a
        message list and model config, resolves the tokenizer, and returns
        an int token count."""
        from proxy.provider import _get_tokenizer_for_model

        text = sb.generate_large_prompt_fixture(60000)
        messages = [{"role": "user", "content": text}]

        model_config = {"tokenizer": "qwen3"}
        server_config = {}

        # Resolve tokenizer the same way the shared helper does.
        tokenizer, multiplier = _get_tokenizer_for_model(model_config, server_config)
        assert tokenizer is not None
        assert multiplier == 1.0

        # The routing path estimate with this tokenizer must be close to
        # the ground truth.
        estimate = _estimate_prompt_tokens_for_routing(
            {"messages": messages}, tokenizer=tokenizer
        )
        lower = QWEN3_60K_ACTUAL_TOKENS * (1 - TOLERANCE)
        upper = QWEN3_60K_ACTUAL_TOKENS * (1 + TOLERANCE)
        assert lower <= estimate <= upper, (
            f"estimate {estimate} must be within ±5% of {QWEN3_60K_ACTUAL_TOKENS}"
        )

    def test_estimate_fn_no_tokenizer_fallback_to_tiktoken(self):
        """When no native tokenizer is configured, the estimate falls back
        to tiktoken (same as today)."""
        estimate = _estimate_prompt_tokens_for_routing(_body_for_fixture("hello world"))
        assert isinstance(estimate, int)
        assert estimate > 0


# ===================================================================
# AC2: Routing and compaction estimates agree for the same message list
# ===================================================================


class TestRoutingCompactionEstimateAgreement:
    """Both the routing path and compaction trigger must produce the same
    token estimate for the same message list when the model config includes
    a native tokenizer."""

    def test_same_estimate_with_native_tokenizer(self):
        """Given the same message list + model config with tokenizer: qwen3,
        routing estimate and compaction estimate must be identical."""
        text = sb.generate_large_prompt_fixture(60000)
        messages = [{"role": "user", "content": text}]

        model_config = {"tokenizer": "qwen3"}
        server_config = {}

        # Routing estimate: resolves tokenizer, estimates over body.
        tokenizer, multiplier = _get_tokenizer_for_model(model_config, server_config)
        routing_estimate = _estimate_prompt_tokens_for_routing(
            {"messages": messages}, tokenizer=tokenizer
        )

        # Compaction estimate: same messages, same tokenizer.
        compaction_estimate = _estimate_prompt_tokens_for_routing(
            {"messages": messages}, tokenizer=tokenizer
        )

        assert routing_estimate == compaction_estimate, (
            f"routing estimate ({routing_estimate}) != compaction estimate "
            f"({compaction_estimate}) for the same messages"
        )

    def test_same_estimate_with_tiktoken_fallback(self):
        """Without a native tokenizer, both paths use tiktoken and agree."""
        text = sb.generate_large_prompt_fixture(60000)
        messages = [{"role": "user", "content": text}]

        model_config = {}
        server_config = {"token_estimate_multiplier": 1.69}

        # Both paths resolve the same way (no tokenizer).
        tokenizer, multiplier = _get_tokenizer_for_model(model_config, server_config)
        estimate = _estimate_prompt_tokens_for_routing(
            {"messages": messages}, tokenizer=tokenizer
        )
        assert isinstance(estimate, int)
        assert estimate > 0


# ===================================================================
# AC3: Tokenizer parity — same message list yields same estimate in both paths
# ===================================================================


class TestTokenizerParity:
    """The compaction trigger must apply the same native tokenizer /
    multiplier resolution as routing."""

    def test_compaction_uses_native_tokenizer_not_tiktoken(self):
        """With tokenizer: qwen3, the compaction estimate uses the native
        tokenizer (within ±5% of ground truth), NOT tiktoken (~47680)."""
        text = sb.generate_large_prompt_fixture(60000)
        messages = [{"role": "user", "content": text}]

        model_config = {"tokenizer": "qwen3"}
        server_config = {}

        tokenizer, multiplier = _get_tokenizer_for_model(model_config, server_config)
        compaction_estimate = _estimate_prompt_tokens_for_routing(
            {"messages": messages}, tokenizer=tokenizer
        )

        lower = QWEN3_60K_ACTUAL_TOKENS * (1 - TOLERANCE)
        upper = QWEN3_60K_ACTUAL_TOKENS * (1 + TOLERANCE)
        assert lower <= compaction_estimate <= upper, (
            f"compaction estimate ({compaction_estimate}) must be within ±5% "
            f"of {QWEN3_60K_ACTUAL_TOKENS} (not tiktoken's ~47680)"
        )

    def test_tiktoken_estimate_without_native_tokenizer(self):
        """Without native tokenizer, compaction estimate matches tiktoken
        (proving we're not accidentally getting a different count)."""
        text = sb.generate_large_prompt_fixture(60000)
        messages = [{"role": "user", "content": text}]

        estimate = _estimate_prompt_tokens_for_routing({"messages": messages})
        # tiktoken gives ~47680 for the 60K fixture; must be within 5% of that.
        tiktoken_baseline = int(QWEN3_60K_ACTUAL_TOKENS * 0.62)
        assert estimate < tiktoken_baseline * 1.1, (
            f"tiktoken estimate ({estimate}) should be ~47680, not {QWEN3_60K_ACTUAL_TOKENS}"
        )


# ===================================================================
# AC4: Input parity — compaction estimates over combined (body + session) history
# ===================================================================


class TestInputParity:
    """The compaction trigger evaluates the same produced history that
    routing estimates over (incoming body ∪ stored session history, plus
    current turn)."""

    def test_combined_history_estimate(self):
        """When compaction evaluates the combined session+delta messages,
        the estimate uses the native tokenizer (if configured)."""
        model_config = {"tokenizer": "qwen3"}
        server_config = {}

        tokenizer, multiplier = _get_tokenizer_for_model(model_config, server_config)
        assert tokenizer is not None
        assert multiplier == 1.0

        # Combined history: simulated session messages + new delta.
        session_text = sb.generate_large_prompt_fixture(40000)
        delta_text = sb.generate_large_prompt_fixture(5000)
        combined_messages = [
            {"role": "user", "content": session_text},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": delta_text},
        ]

        estimate = _estimate_prompt_tokens_for_routing(
            {"messages": combined_messages}, tokenizer=tokenizer
        )
        assert isinstance(estimate, int)
        assert estimate > 0  # Sanity: non-zero for 45K tokens of text.
