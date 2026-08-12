"""
Tests for Qwen3-native tokenizer integration into routing/persistence estimates.

Regression for LP-0MSEQ71IF0003FRT (follow-up to LP-0MSEGPO77005CYCQ AC1):

The routing tokenizer mismatch (tiktoken cl100k undercounts Qwen3 native
tokens ~1.69x for dense prose) was fixed near-term via the
``token_estimate_multiplier`` heuristic. This work replaces the multiplier
with the actual Qwen3-native tokenizer so routing and persistence estimates
match true tokens exactly.

These tests are expected to FAIL (red) against the current code and pass
after the vendored-tokenizer + shared-helper changes land
(LP-0MSMAHZRW006CKSF F1, LP-0MSMAI5E6009F3S7 F2, LP-0MSMAI5E0001O17Q F3).
"""

import logging

import pytest
from proxy.provider import (
    _estimate_prompt_tokens_for_routing,
    _get_token_estimate_multiplier,
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
        proxy_dir = this_dir.parent  # proxy/
        root_dir = proxy_dir.parent  # project root
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
# AC1: vendored tokenizer loads via the tokenizers library
# ===================================================================


class TestVendoredTokenizerLoads:
    """The vendored tokenizer.json must load without errors (AC1)."""

    def test_get_tokenizer_qwen3_returns_loaded_tokenizer(self):
        """get_tokenizer('qwen3') returns a working tokenizer, not None."""
        tok = _get_qwen3_tokenizer()
        ids = tok.encode("The quick brown fox jumps over the lazy dog.").ids
        assert len(ids) > 0

    def test_get_tokenizer_unknown_name_returns_none(self):
        """Unknown registry names return None (no crash in request path)."""
        from proxy.tokenizers import get_tokenizer

        assert get_tokenizer("does-not-exist") is None

    def test_get_tokenizer_load_failure_returns_none(self, monkeypatch):
        """A registry entry whose file is missing returns None + warning.

        Simulates a corrupted install (missing tokenizer.json) — the
        registry must degrade to None, never raising (AC2 fallback path).
        """
        from pathlib import Path

        import proxy.tokenizers as tk_mod

        monkeypatch.setattr(
            tk_mod,
            "TOKENIZER_REGISTRY",
            {"qwen3": Path("/nonexistent/qwen3/tokenizer.json")},
        )
        # The registry is lru_cached by name — clear it so the monkeypatched
        # (missing-file) registry entry is actually consulted.
        tk_mod._load_tokenizer.cache_clear()
        try:
            tok = tk_mod.get_tokenizer("qwen3")
        finally:
            tk_mod._load_tokenizer.cache_clear()
        assert tok is None


# ===================================================================
# AC2: 60K fixture estimates within ±5% of true Qwen3 tokens
# ===================================================================


class TestSixtyKFixtureNativeEstimate:
    """With the native tokenizer active, the 60K fixture estimates
    within ±5% of the actual 77060 Qwen3 tokens."""

    @pytest.fixture(scope="class")
    def fixture_body(self):
        text = sb.generate_large_prompt_fixture(60000)
        return _body_for_fixture(text)

    def test_native_tokenizer_estimate_within_5pct(self, fixture_body):
        """_estimate_prompt_tokens_for_routing with the qwen3 tokenizer
        lands within ±5% of 77060 (vs tiktoken's 47680, ~38% low)."""
        tok = _get_qwen3_tokenizer()
        estimate = _estimate_prompt_tokens_for_routing(fixture_body, tokenizer=tok)
        lower = QWEN3_60K_ACTUAL_TOKENS * (1 - TOLERANCE)
        upper = QWEN3_60K_ACTUAL_TOKENS * (1 + TOLERANCE)
        assert lower <= estimate <= upper, (
            f"native estimate {estimate} must be within ±5% of "
            f"{QWEN3_60K_ACTUAL_TOKENS} [{lower:.0f}, {upper:.0f}]"
        )

    def test_tiktoken_estimate_is_outside_5pct(self, fixture_body):
        """Sanity: the tiktoken cl100k estimate (~47680) is NOT within ±5%
        of 77060 — proving the native tokenizer matters (AC2 motivation)."""
        estimate = _estimate_prompt_tokens_for_routing(fixture_body)
        upper = QWEN3_60K_ACTUAL_TOKENS * (1 + TOLERANCE)
        assert estimate < upper * 0.7, (
            f"tiktoken estimate {estimate} must be well below the native "
            f"count (it undercounts Qwen3 ~1.69x)"
        )


# ===================================================================
# AC3: shared resolution chain (_get_tokenizer_for_model)
# ===================================================================


class TestTokenizerResolutionChain:
    """The shared helper resolves (tokenizer, multiplier) deterministically:

    1. ``tokenizer: qwen3`` (loads) → native tokenizer, multiplier forced 1.0
    2. no tokenizer named → (None, multiplier) — tiktoken + multiplier
    3. named tokenizer fails to load → warning + (None, multiplier)
    """

    def test_named_tokenizer_forces_multiplier_to_1(self):
        """tokenizer: qwen3 on the model entry → native tokenizer AND
        multiplier 1.0 (applying ~1.69x on top would over-count ~69%)."""
        from proxy.provider import _get_tokenizer_for_model

        tokenizer, multiplier = _get_tokenizer_for_model(
            {"tokenizer": "qwen3"}, {"token_estimate_multiplier": 1.69}
        )
        assert tokenizer is not None
        assert multiplier == 1.0

    def test_no_tokenizer_uses_multiplier(self):
        """No tokenizer named → (None, multiplier); per-model override wins,
        else server-level, else 1.0 (today's behavior, unchanged)."""
        from proxy.provider import _get_tokenizer_for_model

        tokenizer, multiplier = _get_tokenizer_for_model(
            {}, {"token_estimate_multiplier": 1.69}
        )
        assert tokenizer is None
        assert multiplier == 1.69

    def test_no_tokenizer_no_multiplier_defaults_1(self):
        """Neither tokenizer nor multiplier → (None, 1.0)."""
        from proxy.provider import _get_tokenizer_for_model

        tokenizer, multiplier = _get_tokenizer_for_model({}, {})
        assert tokenizer is None
        assert multiplier == 1.0

    def test_per_model_multiplier_override_wins_without_tokenizer(self):
        """Per-model token_estimate_multiplier still wins over server-level
        when no native tokenizer is configured (backward compat)."""
        from proxy.provider import _get_tokenizer_for_model

        tokenizer, multiplier = _get_tokenizer_for_model(
            {"token_estimate_multiplier": 2.0},
            {"token_estimate_multiplier": 1.69},
        )
        assert tokenizer is None
        assert multiplier == 2.0

    def test_failed_tokenizer_falls_back_with_warning(self, monkeypatch, caplog):
        """A named tokenizer that fails to load → warning + (None, multiplier)
        fallback so routing never crashes (AC2 'falling back to
        tiktoken+multiplier when unavailable')."""
        from proxy.provider import _get_tokenizer_for_model

        # Force the registry to fail loading the qwen3 tokenizer.
        def _boom(name):
            raise RuntimeError("simulated load failure")

        monkeypatch.setattr("proxy.tokenizers.get_tokenizer", _boom)
        with caplog.at_level(logging.WARNING, logger="proxy.provider"):
            tokenizer, multiplier = _get_tokenizer_for_model(
                {"tokenizer": "qwen3"}, {"token_estimate_multiplier": 1.69}
            )
        assert tokenizer is None
        assert multiplier == 1.69
        assert any("tokenizer" in r.message for r in caplog.records), (
            "expected a warning log about the tokenizer fallback"
        )

    def test_failed_tokenizer_returns_none_fallback(self, monkeypatch):
        """get_tokenizer returning None (missing file) → (None, multiplier)."""
        from proxy.provider import _get_tokenizer_for_model

        monkeypatch.setattr("proxy.tokenizers.get_tokenizer", lambda name: None)
        tokenizer, multiplier = _get_tokenizer_for_model(
            {"tokenizer": "qwen3"}, {"token_estimate_multiplier": 1.69}
        )
        assert tokenizer is None
        assert multiplier == 1.69


# ===================================================================
# AC3: routing and persistence estimates are identical (shared helper)
# ===================================================================


class TestRoutingPersistenceConsistency:
    """Routing (provider.py) and persistence (session.py) estimates must
    never disagree — both consume the same shared helper."""

    def test_identical_with_native_tokenizer(self):
        """Same body + model config with tokenizer: qwen3 → both paths give
        the same (native) estimate; multiplier is NOT applied on top."""
        from proxy.provider import _get_tokenizer_for_model
        from proxy.session import _estimate_slot_prompt_tokens

        text = sb.generate_large_prompt_fixture(60000)
        body = _body_for_fixture(text)
        server_config = {"token_estimate_multiplier": 1.69}
        model_config = {"tokenizer": "qwen3"}

        # Routing path: shared helper → native count, multiplier forced 1.0.
        tokenizer, multiplier = _get_tokenizer_for_model(model_config, server_config)
        routing_estimate = _estimate_prompt_tokens_for_routing(body, tokenizer=tokenizer)
        assert multiplier == 1.0

        # Persistence path: same shared helper.
        persistence_estimate = _estimate_slot_prompt_tokens(
            body, server_config, model_config
        )

        assert persistence_estimate == routing_estimate, (
            "routing and persistence estimates must be identical for the "
            "same body/config when a native tokenizer is active"
        )

    def test_identical_with_multiplier_fallback(self):
        """No tokenizer → both paths apply the SAME multiplier, so the
        estimates stay identical (multiplier heuristic consistency)."""
        from proxy.provider import _get_tokenizer_for_model
        from proxy.session import _estimate_slot_prompt_tokens

        text = sb.generate_large_prompt_fixture(60000)
        body = _body_for_fixture(text)
        server_config = {"token_estimate_multiplier": 1.69}
        model_config = {}

        tokenizer, multiplier = _get_tokenizer_for_model(model_config, server_config)
        routing_estimate = _estimate_prompt_tokens_for_routing(body, tokenizer=tokenizer)
        if multiplier != 1.0:
            routing_estimate = int(routing_estimate * multiplier)

        persistence_estimate = _estimate_slot_prompt_tokens(
            body, server_config, model_config
        )
        assert persistence_estimate == routing_estimate

    def test_no_multiplier_no_tokenizer_identical(self):
        """Neither configured → both paths return the raw tiktoken count."""
        from proxy.provider import _get_tokenizer_for_model
        from proxy.session import _estimate_slot_prompt_tokens

        text = sb.generate_large_prompt_fixture(60000)
        body = _body_for_fixture(text)

        tokenizer, multiplier = _get_tokenizer_for_model({}, {})
        routing_estimate = _estimate_prompt_tokens_for_routing(body, tokenizer=tokenizer)
        assert multiplier == 1.0

        persistence_estimate = _estimate_slot_prompt_tokens(body, {}, {})
        assert persistence_estimate == routing_estimate


# ===================================================================
# Config plumbing: per-model tokenizer + removal of server multiplier
# ===================================================================


class TestConfigTokenizerPlumbing:
    """The live config wires tokenizer: qwen3 onto the local Qwen3 models
    and drops the now-redundant server-level token_estimate_multiplier."""

    @pytest.mark.parametrize("config_file", ["config.yaml", "config-fast.yaml"])
    def test_local_qwen3_models_have_tokenizer(self, config_file):
        """plan/author/code (all local Qwen3) carry tokenizer: qwen3."""
        from pathlib import Path

        import yaml

        path = Path(__file__).resolve().parent.parent / config_file
        cfg = yaml.safe_load(path.read_text())
        for model in ("plan", "author", "code"):
            entry = cfg["models"][model]
            assert entry.get("tokenizer") == "qwen3", (
                f"{model} must reference tokenizer: qwen3 in {config_file}"
            )

    @pytest.mark.parametrize("config_file", ["config.yaml", "config-fast.yaml"])
    def test_server_level_multiplier_removed(self, config_file):
        """The server-level token_estimate_multiplier is gone — the native
        tokenizer replaces it (AC4: removing it causes no behavior change)."""
        from pathlib import Path

        import yaml

        path = Path(__file__).resolve().parent.parent / config_file
        cfg = yaml.safe_load(path.read_text())
        server = cfg.get("server", {})
        assert "token_estimate_multiplier" not in server, (
            f"server-level token_estimate_multiplier must be removed from "
            f"{config_file} — the native tokenizer replaces it"
        )

    def test_multiplier_helper_defaults_when_key_absent(self):
        """_get_token_estimate_multiplier stays functional without the key
        (defaults 1.0), so config removal is safe."""
        mult = _get_token_estimate_multiplier({"server": {}}, {})
        assert mult == 1.0
