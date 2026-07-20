"""
Unit and integration tests for the provider resolver module.

Tests for:
- resolve_name_to_ids(): Static mapping resolution for short names.
- resolve_audit_model(): Full resolution with fallback chain.
- validate_audit_models(): Startup validation (lenient & strict modes).
- Metrics: Unresolved counter tracking.
- Integration: Free-model fallback chain exercise.
"""

from unittest.mock import patch

import proxy.provider_resolver as resolver
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_unresolved_counts():
    """Reset the unresolved counter between tests."""
    resolver._unresolved_counts.clear()
    yield


# ---------------------------------------------------------------------------
# resolve_name_to_ids() — static mapping resolution
# ---------------------------------------------------------------------------


class TestResolveNameToIds:
    """Tests for the static mapping resolution function."""

    def test_deepseek_v4_flash_free(self):
        result = resolver.resolve_name_to_ids("deepseek-v4-flash-free")
        assert result == [
            "opencode/deepseek-v4-flash-free",
            "openrouter/openrouter/free",
            "opencode-go/deepseek-v4-flash",
        ]

    def test_deepseek_v4_flash(self):
        result = resolver.resolve_name_to_ids("deepseek-v4-flash")
        assert result == [
            "opencode/deepseek-v4-flash",
            "opencode-go/deepseek-v4-flash",
            "openrouter/deepseek/deepseek-v4-flash",
        ]

    def test_deepseek_v4_flash_wildcard(self):
        # deepseek-v4-flash-something should match the deepseek-v4-flash* rule
        result = resolver.resolve_name_to_ids("deepseek-v4-flash-pro")
        assert result == [
            "opencode/deepseek-v4-flash",
            "opencode-go/deepseek-v4-flash",
            "openrouter/deepseek/deepseek-v4-flash",
        ]

    def test_deepseek_prefix(self):
        # Generic deepseek prefix
        result = resolver.resolve_name_to_ids("deepseek-model")
        assert result == [
            "opencode/deepseek-v4-flash",
            "opencode-go/deepseek-v4-flash",
            "openrouter/deepseek/deepseek-v4-flash",
        ]

    def test_deepseek_v4_prefix(self):
        # Generic deepseek-v4 prefix
        result = resolver.resolve_name_to_ids("deepseek-v4-pro")
        assert result == [
            "opencode/deepseek-v4-flash",
            "opencode-go/deepseek-v4-flash",
            "openrouter/deepseek/deepseek-v4-flash",
        ]

    def test_openrouter_free(self):
        result = resolver.resolve_name_to_ids("openrouter/free")
        assert result == [
            "openrouter/openrouter/free",
            "opencode/deepseek-v4-flash-free",
            "opencode-go/deepseek-v4-flash",
        ]

    def test_openrouter_free_variants(self):
        for variant in ["openrouter-free", "free-model"]:
            result = resolver.resolve_name_to_ids(variant)
            assert result == [
                "openrouter/openrouter/free",
                "opencode/deepseek-v4-flash-free",
                "opencode-go/deepseek-v4-flash",
            ]

    def test_openrouter_deepseek(self):
        result = resolver.resolve_name_to_ids("openrouter/deepseek/deepseek-v4-flash")
        assert result == [
            "openrouter/deepseek/deepseek-v4-flash",
            "opencode/deepseek-v4-flash",
            "opencode-go/deepseek-v4-flash",
        ]

    def test_openrouter_generic_prefix(self):
        result = resolver.resolve_name_to_ids("openrouter/gpt-4")
        assert result == [
            "openrouter/openrouter/free",
            "opencode/deepseek-v4-flash-free",
            "opencode-go/deepseek-v4-flash",
        ]

    def test_opencode_go_prefix(self):
        result = resolver.resolve_name_to_ids("opencode-go/deepseek-v4-flash")
        assert result == [
            "opencode-go/deepseek-v4-flash",
        ]

    def test_opencode_prefix(self):
        result = resolver.resolve_name_to_ids("opencode/deepseek-v4-flash-free")
        assert result == [
            "opencode/deepseek-v4-flash-free",
        ]

    def test_already_qualified_pass_through(self):
        # Already-qualified names pass through
        result = resolver.resolve_name_to_ids("opencode/some-model")
        assert result == ["opencode/some-model"]

    def test_custom_name_fallback(self):
        # Unknown custom names fall back to opencode/ as default
        result = resolver.resolve_name_to_ids("my-custom-model")
        assert result == [
            "opencode/my-custom-model",
            "opencode-go/my-custom-model",
            "openrouter/my-custom-model",
        ]


# ---------------------------------------------------------------------------
# resolve_audit_model() — full resolution with fallback chain
# ---------------------------------------------------------------------------


class TestResolveAuditModel:
    """Tests for the full audit model resolution function."""

    def test_resolve_primary_only(self):
        result = resolver.resolve_audit_model("deepseek-v4-flash-free")
        assert len(result) >= 1
        assert "opencode/deepseek-v4-flash-free" in result

    def test_resolve_with_fallbacks(self):
        # Primary resolves, fallbacks not attempted
        result = resolver.resolve_audit_model(
            "deepseek-v4-flash-free",
            fallbacks=["free-model"],
        )
        assert len(result) >= 1

    def test_resolve_primary_fails_then_fallback_succeeds(self):
        # Patch resolve_name_to_ids so primary returns nothing
        resolver._original_resolve_name_to_ids = resolver.resolve_name_to_ids
        try:
            def mock_resolve(n):
                if n == "nonexistent-model":
                    return []
                return resolver._original_resolve_name_to_ids(n)

            with patch.object(resolver, "resolve_name_to_ids", side_effect=mock_resolve):
                result = resolver.resolve_audit_model(
                    "nonexistent-model",
                    fallbacks=["deepseek-v4-flash-free"],
                )
                assert len(result) >= 1
        finally:
            resolver.resolve_name_to_ids = resolver._original_resolve_name_to_ids
            delattr(resolver, "_original_resolve_name_to_ids")

    def test_resolve_primary_fails_all_fallbacks(self):
        with patch.object(
            resolver,
            "resolve_name_to_ids",
            return_value=[],
        ):
            result = resolver.resolve_audit_model(
                "nonexistent-model",
                fallbacks=["also-nonexistent"],
            )
            assert result == []

    def test_resolve_no_fallbacks_given(self):
        # Empty fallback list should not cause errors
        result = resolver.resolve_audit_model("deepseek-v4-flash-free", fallbacks=[])
        assert len(result) >= 1

    def test_resolve_none_fallbacks(self):
        # None fallbacks should not cause errors
        result = resolver.resolve_audit_model("deepseek-v4-flash-free", fallbacks=None)
        assert len(result) >= 1

    def test_free_model_fallback_chain(self):
        """Integration test: exercise the free-model fallback chain.

        When the primary is 'deepseek-v4-flash-free', the first candidate
        should be 'opencode/deepseek-v4-flash-free', followed by
        'openrouter/openrouter/free' and 'opencode-go/deepseek-v4-flash'.
        """
        result = resolver.resolve_audit_model(
            "deepseek-v4-flash-free",
            fallbacks=["openrouter/free", "deepseek-v4-flash"],
        )
        # Primary should resolve
        assert len(result) >= 1
        # First candidate should be from the primary mapping
        assert result[0] == "opencode/deepseek-v4-flash-free"

    def test_openrouter_free_as_fallback_chain(self):
        """Integration test: openrouter/free → deepseek-v4-flash-free → deepseek-v4-flash."""
        result = resolver.resolve_audit_model(
            "openrouter/free",
            fallbacks=["deepseek-v4-flash-free", "deepseek-v4-flash"],
        )
        assert len(result) >= 1
        assert result[0] == "openrouter/openrouter/free"


# ---------------------------------------------------------------------------
# validate_audit_models() — startup validation
# ---------------------------------------------------------------------------


class TestValidateAuditModels:
    """Tests for the startup validation function."""

    def test_no_audit_model_configured(self):
        # Strict mode should fail when no audit_model
        result = resolver.validate_audit_models({}, strict=True)
        assert not result["ok"]
        assert len(result["warnings"]) >= 1
        assert "No audit_model configured" in result["warnings"][0]
        assert result["resolved_ids"] == []

    def test_no_audit_model_configured_lenient(self):
        # Lenient mode should still pass (ok=True) when no audit_model
        result = resolver.validate_audit_models({}, strict=False)
        # When strict=False and no model is configured, ok depends on implementation
        # The current implementation returns ok=False only when strict=True
        # Actually looking at the code: it returns ok=False when no primary is configured
        # and strict=True, ok=True otherwise
        # Let me check the actual implementation again...
        # In the code: if not primary: returns ok=False if strict else True
        assert result["ok"] is True

    def test_no_audit_model_configured_strict(self):
        result = resolver.validate_audit_models({}, strict=True)
        assert not result["ok"]

    def test_valid_audit_model(self):
        config = {"audit_model": "deepseek-v4-flash-free"}
        result = resolver.validate_audit_models(config)
        assert result["ok"] is True
        assert result["primary"] == "deepseek-v4-flash-free"
        assert len(result["resolved_ids"]) >= 1

    def test_valid_with_fallbacks(self):
        config = {
            "audit_model": "deepseek-v4-flash-free",
            "audit_model_fallbacks": ["openrouter/free", "deepseek-v4-flash"],
        }
        result = resolver.validate_audit_models(config)
        assert result["ok"] is True
        assert len(result["resolved_ids"]) >= 1

    def test_invalid_model_lenient(self):
        with patch.object(
            resolver,
            "resolve_name_to_ids",
            return_value=[],
        ):
            result = resolver.validate_audit_models(
                {"audit_model": "nonexistent-model"},
                strict=False,
            )
            assert result["ok"] is True  # lenient
            assert len(result["warnings"]) >= 1

    def test_invalid_model_strict(self):
        with patch.object(
            resolver,
            "resolve_name_to_ids",
            return_value=[],
        ):
            result = resolver.validate_audit_models(
                {"audit_model": "nonexistent-model"},
                strict=True,
            )
            assert result["ok"] is False
            assert len(result["warnings"]) >= 1

    def test_unresolvable_primary_with_resolvable_fallbacks(self):
        """Primary fails, fallbacks resolve — should still be ok."""
        def mock_resolve(n):
            if n == "nonexistent":
                return []
            return resolver._original_resolve_name_to_ids(n)
        # Save the real function before patching
        resolver._original_resolve_name_to_ids = resolver.resolve_name_to_ids
        try:
            with patch.object(resolver, "resolve_name_to_ids", side_effect=mock_resolve):
                config = {
                    "audit_model": "nonexistent",
                    "audit_model_fallbacks": ["deepseek-v4-flash-free"],
                }
                result = resolver.validate_audit_models(config, strict=True)
                assert result["ok"] is True
                assert len(result["resolved_ids"]) >= 1
        finally:
            resolver.resolve_name_to_ids = resolver._original_resolve_name_to_ids
            delattr(resolver, "_original_resolve_name_to_ids")

    def test_unresolvable_primary_and_fallbacks_strict(self):
        with patch.object(
            resolver,
            "resolve_name_to_ids",
            return_value=[],
        ):
            config = {
                "audit_model": "nonexistent",
                "audit_model_fallbacks": ["also-nonexistent"],
            }
            result = resolver.validate_audit_models(config, strict=True)
            assert result["ok"] is False

    def test_return_structure(self):
        """Validate the return dict structure."""
        config = {"audit_model": "deepseek-v4-flash-free"}
        result = resolver.validate_audit_models(config)
        assert "ok" in result
        assert "primary" in result
        assert "resolved_ids" in result
        assert "warnings" in result
        assert isinstance(result["ok"], bool)
        assert isinstance(result["primary"], str)
        assert isinstance(result["resolved_ids"], list)
        assert isinstance(result["warnings"], list)


# ---------------------------------------------------------------------------
# Metrics — unresolved counter
# ---------------------------------------------------------------------------


class TestUnresolvedMetrics:
    """Tests for the unresolved name counter metric."""

    def test_no_unresolved_initially(self):
        assert resolver.get_unresolved_counts() == {}

    def test_unresolved_increments(self):
        with patch.object(
            resolver,
            "resolve_name_to_ids",
            side_effect=lambda n: [] if n == "bad" else resolver.resolve_name_to_ids(n),
        ):
            resolver.resolve_audit_model("bad", fallbacks=["bad"])
        counts = resolver.get_unresolved_counts()
        assert counts.get("bad", 0) >= 1

    def test_unresolved_returns_copy(self):
        resolver._unresolved_counts["test"] = 1
        copy = resolver.get_unresolved_counts()
        copy["test"] = 999
        assert resolver.get_unresolved_counts()["test"] == 1

    def test_multiple_unresolved_names(self):
        with patch.object(
            resolver,
            "resolve_name_to_ids",
            return_value=[],
        ):
            resolver.resolve_audit_model("bad1", fallbacks=["bad2"])
        counts = resolver.get_unresolved_counts()
        assert "bad1" in counts
        assert "bad2" in counts


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_name(self):
        result = resolver.resolve_name_to_ids("")
        # Should fall through to the catch-all
        assert result == [
            "opencode/",
            "opencode-go/",
            "openrouter/",
        ]

    def test_name_with_slash_passes_through(self):
        result = resolver.resolve_name_to_ids("some-provider/some-model")
        assert result == ["some-provider/some-model"]

    def test_opencode_go_with_path(self):
        result = resolver.resolve_name_to_ids("opencode-go/custom-model")
        assert result == ["opencode-go/custom-model"]

    def test_opencode_with_path(self):
        result = resolver.resolve_name_to_ids("opencode/custom-model")
        assert result == ["opencode/custom-model"]
