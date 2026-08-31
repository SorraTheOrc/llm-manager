"""Validation tests for Qwen thinking preservation in pi agent model configuration.

Tests verify that the canonical Qwen thinking overrides (tracked in
``config/pi-agent/qwen-thinking-overrides.json``) define thinking config for
all Qwen variants, and that the live ``~/.pi/agent/models.json`` (when present)
matches the canonical overrides.

This covers AC1 (thinkingFormat/thinkingLevelMap set), AC3 (all variants
reviewed), and AC4 (configuration-level validation that thinking is enabled).
Live transcript testing (AC4) is handled in proxy/tests/test_qwen_thinking_preserved.py.

Relevant work-item: LP-0MT5YLL36000ZYRT
"""

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CANONICAL_PATH = _REPO_ROOT / "config" / "pi-agent" / "qwen-thinking-overrides.json"
_LIVE_MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"


def _load_canonical() -> dict:
    """Load the canonical Qwen thinking overrides."""
    return json.loads(_CANONICAL_PATH.read_text(encoding="utf-8"))


def _load_live_models() -> dict | None:
    """Load the live models.json, returning None when absent."""
    if not _LIVE_MODELS_JSON.is_file():
        return None
    return json.loads(_LIVE_MODELS_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# AC1 + AC3: Canonical overrides validate
# ---------------------------------------------------------------------------

class TestCanonicalOverrides:
    """Validate the canonical Qwen thinking overrides file structure and content."""

    def test_canonical_file_exists(self):
        """The canonical overrides file must exist in the repo."""
        assert _CANONICAL_PATH.exists(), (
            f"Canonical Qwen thinking overrides not found at {_CANONICAL_PATH}"
        )

    def test_canonical_json_parses(self):
        """The canonical file must be valid JSON."""
        data = _load_canonical()
        assert isinstance(data, dict)

    def test_canonical_has_providers_key(self):
        """The canonical file must have a providers key."""
        data = _load_canonical()
        assert "providers" in data

    def test_canonical_opencode_overrides_exist(self):
        """opencode provider must have modelOverrides in canonical config."""
        data = _load_canonical()
        assert "opencode" in data["providers"]
        assert "modelOverrides" in data["providers"]["opencode"]

    def test_canonical_opencode_go_overrides_exist(self):
        """opencode-go provider must have modelOverrides in canonical config."""
        data = _load_canonical()
        assert "opencode-go" in data["providers"]
        assert "modelOverrides" in data["providers"]["opencode-go"]

    def test_canonical_opencode_qwen35_has_thinking_config(self):
        """opencode qwen3.5-plus must have thinkingLevelMap."""
        data = _load_canonical()
        overrides = data["providers"]["opencode"]["modelOverrides"]
        model = overrides["qwen3.5-plus"]
        assert "thinkingLevelMap" in model

    def test_canonical_opencode_qwen36_has_thinking_config(self):
        """opencode qwen3.6-plus must have thinkingLevelMap."""
        data = _load_canonical()
        overrides = data["providers"]["opencode"]["modelOverrides"]
        model = overrides["qwen3.6-plus"]
        assert "thinkingLevelMap" in model

    def test_canonical_opencode_go_qwen36_has_thinking_format(self):
        """opencode-go qwen3.6-plus must have thinkingFormat + thinkingLevelMap."""
        data = _load_canonical()
        overrides = data["providers"]["opencode-go"]["modelOverrides"]
        model = overrides["qwen3.6-plus"]
        assert "compat" in model
        assert model["compat"].get("thinkingFormat") == "qwen"
        assert "thinkingLevelMap" in model

    def test_canonical_opencode_go_qwen37_max_has_thinking_format(self):
        """opencode-go qwen3.7-max must have thinkingFormat + thinkingLevelMap."""
        data = _load_canonical()
        overrides = data["providers"]["opencode-go"]["modelOverrides"]
        model = overrides["qwen3.7-max"]
        assert model["compat"].get("thinkingFormat") == "qwen"
        assert "thinkingLevelMap" in model

    def test_canonical_opencode_go_qwen37_plus_has_thinking_format(self):
        """opencode-go qwen3.7-plus must have thinkingFormat + thinkingLevelMap."""
        data = _load_canonical()
        overrides = data["providers"]["opencode-go"]["modelOverrides"]
        model = overrides["qwen3.7-plus"]
        assert model["compat"].get("thinkingFormat") == "qwen"
        assert "thinkingLevelMap" in model

    def test_canonical_opencode_go_qwen38_max_has_thinking_format(self):
        """opencode-go qwen3.8-max must have thinkingFormat + thinkingLevelMap."""
        data = _load_canonical()
        overrides = data["providers"]["opencode-go"]["modelOverrides"]
        model = overrides["qwen3.8-max"]
        assert model["compat"].get("thinkingFormat") == "qwen"
        assert "thinkingLevelMap" in model

    def test_all_expected_qwen_variants_covered(self):
        """All Qwen variants mentioned in the work item AC3 must have overrides."""
        data = _load_canonical()
        overrides = data["providers"]["opencode-go"]["modelOverrides"]
        expected = {"qwen3.6-plus", "qwen3.7-max", "qwen3.7-plus", "qwen3.8-max"}
        assert set(overrides.keys()) == expected, (
            f"Expected opencode-go variants {expected}, got {set(overrides.keys())}"
        )

    def test_opencode_qwen_variants_covered(self):
        """opencode Qwen variants must have overrides."""
        data = _load_canonical()
        overrides = data["providers"]["opencode"]["modelOverrides"]
        expected = {"qwen3.5-plus", "qwen3.6-plus"}
        assert set(overrides.keys()) == expected, (
            f"Expected opencode variants {expected}, got {set(overrides.keys())}"
        )

    def test_thinking_level_map_has_off_high_max(self):
        """All thinkingLevelMap values must map off→null, high→high, max→max."""
        data = _load_canonical()
        for provider in ("opencode", "opencode-go"):
            for model_id, model in data["providers"][provider]["modelOverrides"].items():
                tlm = model["thinkingLevelMap"]
                assert tlm.get("off") is None, (
                    f"{provider}/{model_id}: thinkingLevelMap.off must be null"
                )
                assert tlm.get("high") == "high", (
                    f"{provider}/{model_id}: thinkingLevelMap.high must be 'high'"
                )
                assert tlm.get("max") == "max", (
                    f"{provider}/{model_id}: thinkingLevelMap.max must be 'max'"
                )


# ---------------------------------------------------------------------------
# AC1 + AC3: Live config parity (when live file present)
# ---------------------------------------------------------------------------

class TestLiveConfigParity:
    """Compare live ~/.pi/agent/models.json against canonical overrides.

    These tests are skipped when the live models.json is absent.
    """

    def test_live_config_exists(self):
        """The live models.json should exist on the operator's machine."""
        if not _LIVE_MODELS_JSON.is_file():
            pytest.skip("Live ~/.pi/agent/models.json not found (skipping live parity)")

    def test_live_config_has_opencode_overrides(self):
        """Live config must have opencode modelOverrides for Qwen."""
        live = _load_live_models()
        if live is None:
            pytest.skip("Live models.json not found")

        providers = live.get("providers", {})
        assert "opencode" in providers, "Live config missing opencode provider"
        assert "modelOverrides" in providers["opencode"], (
            "Live config missing opencode.modelOverrides"
        )
        overrides = providers["opencode"]["modelOverrides"]
        for variant in ("qwen3.5-plus", "qwen3.6-plus"):
            assert variant in overrides, f"Live config missing {variant}"

    def test_live_config_has_opencode_go_overrides(self):
        """Live config must have opencode-go modelOverrides for Qwen."""
        live = _load_live_models()
        if live is None:
            pytest.skip("Live models.json not found")

        providers = live.get("providers", {})
        assert "opencode-go" in providers, "Live config missing opencode-go provider"
        overrides = providers["opencode-go"]["modelOverrides"]
        for variant in ("qwen3.6-plus", "qwen3.7-max", "qwen3.7-plus", "qwen3.8-max"):
            assert variant in overrides, f"Live config missing {variant}"

    def test_live_config_opencode_qwen35_thinking(self):
        """Live opencode qwen3.5-plus must have thinking config."""
        live = _load_live_models()
        if live is None:
            pytest.skip("Live models.json not found")

        model = live["providers"]["opencode"]["modelOverrides"]["qwen3.5-plus"]
        assert "thinkingLevelMap" in model

    def test_live_config_opencode_go_qwen_thinking_format(self):
        """Live opencode-go qwen models must have thinkingFormat."""
        live = _load_live_models()
        if live is None:
            pytest.skip("Live models.json not found")

        for variant in ("qwen3.6-plus", "qwen3.7-max", "qwen3.7-plus", "qwen3.8-max"):
            model = live["providers"]["opencode-go"]["modelOverrides"][variant]
            compat = model.get("compat", {})
            assert compat.get("thinkingFormat") == "qwen", (
                f"Live opencode-go {variant}: expected thinkingFormat=qwen"
            )

    def test_live_config_matches_canonical_opencode(self):
        """Live opencode overrides must match canonical."""
        canonical = _load_canonical()
        live = _load_live_models()
        if live is None:
            pytest.skip("Live models.json not found")

        expected = canonical["providers"]["opencode"]["modelOverrides"]
        actual = live["providers"]["opencode"]["modelOverrides"]
        for variant, expected_model in expected.items():
            actual_model = actual.get(variant)
            assert actual_model is not None, f"Live missing {variant}"
            assert actual_model.get("thinkingLevelMap") == expected_model.get(
                "thinkingLevelMap"
            ), f"Live {variant} thinkingLevelMap mismatch"

    def test_live_config_matches_canonical_opencode_go(self):
        """Live opencode-go overrides must match canonical."""
        canonical = _load_canonical()
        live = _load_live_models()
        if live is None:
            pytest.skip("Live models.json not found")

        expected = canonical["providers"]["opencode-go"]["modelOverrides"]
        actual = live["providers"]["opencode-go"]["modelOverrides"]
        for variant, expected_model in expected.items():
            actual_model = actual.get(variant)
            assert actual_model is not None, f"Live missing {variant}"
            assert actual_model.get("thinkingLevelMap") == expected_model.get(
                "thinkingLevelMap"
            ), f"Live {variant} thinkingLevelMap mismatch"
            actual_compat = actual_model.get("compat", {})
            expected_compat = expected_model.get("compat", {})
            assert actual_compat.get("thinkingFormat") == expected_compat.get(
                "thinkingFormat"
            ), f"Live {variant} thinkingFormat mismatch"
