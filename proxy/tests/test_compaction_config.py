
# <!-- REFACTOR-LP-0MTGDB3D9007TWHA
# smell: formatting
# severity: high
# description: Module level import not at top of file
# -->
"""
Config parsing, validation, and defaults for proxy-side session compaction.

Feature: LP-0MTG6RW3L003X122 — compaction config (summarizer_model,
compaction_trigger_ratio, summarizer_ctx_size, summarizer_max_tokens).

This module verifies the config-reading helpers and startup validation that
ensure the compaction configuration is sane before the proxy starts.
"""
import pathlib

import pytest
from proxy.provider import (
    _DEFAULT_COMPACTION_TRIGGER_RATIO,
    _DEFAULT_SUMMARIZER_CTX_SIZE,
    _DEFAULT_SUMMARIZER_MAX_TOKENS,
    _SUMMARIZER_SYSTEM_PROMPT,
    compaction_config,
    validate_compaction_config,
)

# ===================================================================
# compaction_config helper
# ===================================================================


class TestCompactionConfigDefaults:
    """Default values when no compaction config is present."""

    def test_empty_config_returns_defaults(self):
        """No server section → all defaults applied."""
        cfg = {}
        c = compaction_config(cfg)
        assert c["trigger_ratio"] == _DEFAULT_COMPACTION_TRIGGER_RATIO
        assert c["summarizer_model_type"] == "local"
        assert c["summarizer_model_name"] == "Qwen3"
        assert c["summarizer_ctx_size"] == _DEFAULT_SUMMARIZER_CTX_SIZE
        assert c["summarizer_max_tokens"] == _DEFAULT_SUMMARIZER_MAX_TOKENS
        assert c["summarizer_system_prompt"] == _SUMMARIZER_SYSTEM_PROMPT

    def test_server_section_empty_returns_defaults(self):
        """server: {} → all defaults applied."""
        cfg = {"server": {}}
        c = compaction_config(cfg)
        assert c["trigger_ratio"] == _DEFAULT_COMPACTION_TRIGGER_RATIO
        assert c["summarizer_model_name"] == "Qwen3"
        assert c["summarizer_ctx_size"] == _DEFAULT_SUMMARIZER_CTX_SIZE
        assert c["summarizer_max_tokens"] == _DEFAULT_SUMMARIZER_MAX_TOKENS

    def test_flat_key_reads_trigger_ratio(self):
        """Top-level compaction_trigger_ratio is read."""
        cfg = {"compaction_trigger_ratio": 0.75}
        c = compaction_config(cfg)
        assert c["trigger_ratio"] == 0.75


class TestCompactionConfigOverrides:
    """Explicit config values override defaults."""

    def test_nested_trigger_ratio(self):
        """server.compaction_trigger_ratio is read."""
        cfg = {"server": {"compaction_trigger_ratio": 0.65}}
        c = compaction_config(cfg)
        assert c["trigger_ratio"] == 0.65

    def test_summarizer_model_overrides(self):
        """Custom summarizer_model values are read."""
        cfg = {
            "server": {
                "summarizer_model": {
                    "type": "local",
                    "llama_model": "Qwen3-Next",
                },
            },
        }
        c = compaction_config(cfg)
        assert c["summarizer_model_type"] == "local"
        assert c["summarizer_model_name"] == "Qwen3-Next"

    def test_summarizer_ctx_size_overrides(self):
        """server.summarizer_ctx_size is read."""
        cfg = {"server": {"summarizer_ctx_size": 4096}}
        c = compaction_config(cfg)
        assert c["summarizer_ctx_size"] == 4096

    def test_summarizer_max_tokens_overrides(self):
        """server.summarizer_max_tokens is read."""
        cfg = {"server": {"summarizer_max_tokens": 256}}
        c = compaction_config(cfg)
        assert c["summarizer_max_tokens"] == 256

    def test_full_custom_config(self):
        """All compaction settings can be customised together."""
        cfg = {
            "server": {
                "compaction_trigger_ratio": 0.70,
                "summarizer_model": {
                    "type": "local",
                    "llama_model": "Qwen3",
                },
                "summarizer_ctx_size": 8192,
                "summarizer_max_tokens": 512,
            },
        }
        c = compaction_config(cfg)
        assert c["trigger_ratio"] == 0.70
        assert c["summarizer_model_type"] == "local"
        assert c["summarizer_model_name"] == "Qwen3"
        assert c["summarizer_ctx_size"] == 8192
        assert c["summarizer_max_tokens"] == 512


class TestSummarizerSystemPrompt:
    """The dedicated system prompt for compaction summarisation."""

    def test_system_prompt_is_non_empty(self):
        """_SUMMARIZER_SYSTEM_PROMPT must be a non-empty string."""
        assert isinstance(_SUMMARIZER_SYSTEM_PROMPT, str)
        assert len(_SUMMARIZER_SYSTEM_PROMPT) > 0

    def test_config_returns_same_prompt(self):
        """compaction_config returns the same prompt via summarizer_system_prompt."""
        c = compaction_config({})
        assert c["summarizer_system_prompt"] is _SUMMARIZER_SYSTEM_PROMPT


# ===================================================================
# validate_compaction_config startup validation
# ===================================================================


class TestValidateCompactionConfig:
    """Startup validation of the compaction configuration."""

    def test_valid_defaults_no_problems(self):
        """Empty config passes validation (all defaults are valid)."""
        problems = validate_compaction_config({})
        assert problems == []

    def test_valid_full_config_no_problems(self):
        """Fully customised but valid config passes validation."""
        cfg = {
            "server": {
                "compaction_trigger_ratio": 0.70,
                "summarizer_model": {
                    "type": "local",
                    "llama_model": "Qwen3",
                },
                "summarizer_ctx_size": 8192,
                "summarizer_max_tokens": 512,
            },
        }
        problems = validate_compaction_config(cfg)
        assert problems == []

    def test_trigger_ratio_out_of_range_lower(self):
        """Negative trigger_ratio is FATAL."""
        cfg = {"server": {"compaction_trigger_ratio": -0.1}}
        problems = validate_compaction_config(cfg)
        assert any(
            "compaction_trigger_ratio" in p and p.startswith("FATAL:")
            for p in problems
        )

    def test_trigger_ratio_out_of_range_upper(self):
        """trigger_ratio > 1.0 is FATAL."""
        cfg = {"server": {"compaction_trigger_ratio": 1.1}}
        problems = validate_compaction_config(cfg)
        assert any(
            "compaction_trigger_ratio" in p and p.startswith("FATAL:")
            for p in problems
        )

    def test_trigger_ratio_zero_is_valid_but_warns(self):
        """trigger_ratio=0 is valid (disables compaction) but may warn."""
        cfg = {"server": {"compaction_trigger_ratio": 0}}
        problems = validate_compaction_config(cfg)
        # Zero is not FATAL — it just disables compaction
        assert not any(p.startswith("FATAL:") for p in problems)

    def test_explicit_empty_llama_model_is_fatal(self):
        """summarizer_model with explicit empty llama_model is FATAL."""
        cfg = {
            "server": {
                "summarizer_model": {"type": "local", "llama_model": ""},
            },
        }
        problems = validate_compaction_config(cfg)
        assert any(
            "llama_model" in p and p.startswith("FATAL:")
            for p in problems
        )

    def test_missing_llama_model_defaults_to_qwen3(self):
        """summarizer_model without llama_model defaults to Qwen3."""
        cfg = {
            "server": {
                "summarizer_model": {"type": "local"},
            },
        }
        c = compaction_config(cfg)
        assert c["summarizer_model_name"] == "Qwen3"

    def test_missing_summarizer_type_warns(self):
        """summarizer_model without type defaults to local."""
        cfg = {
            "server": {
                "summarizer_model": {"llama_model": "Qwen3"},
            },
        }
        problems = validate_compaction_config(cfg)
        # Missing type is not fatal — defaults to local
        assert not any(p.startswith("FATAL:") for p in problems)

    def test_invalid_summarizer_ctx_size_is_fatal(self):
        """Non-positive summarizer_ctx_size is FATAL."""
        cfg = {
            "server": {
                "summarizer_model": {"type": "local", "llama_model": "Qwen3"},
                "summarizer_ctx_size": 0,
            },
        }
        problems = validate_compaction_config(cfg)
        assert any(
            "summarizer_ctx_size" in p and p.startswith("FATAL:")
            for p in problems
        )

    def test_negative_summarizer_ctx_size_is_fatal(self):
        """Negative summarizer_ctx_size is FATAL."""
        cfg = {
            "server": {
                "summarizer_model": {"type": "local", "llama_model": "Qwen3"},
                "summarizer_ctx_size": -100,
            },
        }
        problems = validate_compaction_config(cfg)
        assert any(
            "summarizer_ctx_size" in p and p.startswith("FATAL:")
            for p in problems
        )

    def test_invalid_summarizer_max_tokens_is_fatal(self):
        """Non-positive summarizer_max_tokens is FATAL."""
        cfg = {
            "server": {
                "summarizer_model": {"type": "local", "llama_model": "Qwen3"},
                "summarizer_max_tokens": 0,
            },
        }
        problems = validate_compaction_config(cfg)
        assert any(
            "summarizer_max_tokens" in p and p.startswith("FATAL:")
            for p in problems
        )

    def test_trigger_ratio_string_is_handled_gracefully(self):
        """Non-numeric trigger_ratio defaults to default value."""
        cfg = {"server": {"compaction_trigger_ratio": "abc"}}
        c = compaction_config(cfg)
        # Should fall back to default without crashing
        assert c["trigger_ratio"] == _DEFAULT_COMPACTION_TRIGGER_RATIO

    def test_ctx_size_string_is_handled_gracefully(self):
        """Non-numeric summarizer_ctx_size falls back to default."""
        cfg = {
            "server": {
                "summarizer_model": {"type": "local", "llama_model": "Qwen3"},
                "summarizer_ctx_size": "invalid",
            },
        }
        c = compaction_config(cfg)
        assert c["summarizer_ctx_size"] == _DEFAULT_SUMMARIZER_CTX_SIZE

    def test_max_tokens_string_is_handled_gracefully(self):
        """Non-numeric summarizer_max_tokens falls back to default."""
        cfg = {
            "server": {
                "summarizer_model": {"type": "local", "llama_model": "Qwen3"},
                "summarizer_max_tokens": "not-a-number",
            },
        }
        c = compaction_config(cfg)
        assert c["summarizer_max_tokens"] == _DEFAULT_SUMMARIZER_MAX_TOKENS


# ===================================================================
# Live config validation
# ===================================================================


class TestLiveConfigsValidate:
    """Verify that the live config files pass compaction validation."""

    @pytest.mark.parametrize("config_file", [
        "config.yaml",
        "config-fast.yaml",
        "config-cheap.yaml",
    ])
    def test_live_configs_pass_compaction_validation(self, config_file):
        """All live config files validate without FATAL errors."""
        config_dir = pathlib.Path(__file__).parent.parent
        config_path = config_dir / config_file
        if not config_path.exists():
            pytest.skip(f"{config_file} not found")
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        problems = validate_compaction_config(cfg)
        fatal = [p for p in problems if p.startswith("FATAL:")]
        assert fatal == [], f"{config_file} has FATAL compaction issues: {fatal}"

    def test_compaction_trigger_ratio_is_070(self):
        """Verify the default trigger_ratio in config.yaml is 0.70."""
        import yaml
        config_path = pathlib.Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        c = compaction_config(cfg)
        assert c["trigger_ratio"] == 0.70

    def test_summarizer_uses_qwen3(self):
        """Verify the summariser defaults to Qwen3 in config.yaml."""
        import yaml
        config_path = pathlib.Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        c = compaction_config(cfg)
        assert c["summarizer_model_name"] == "Qwen3"

    def test_summarizer_ctx_size_is_8192(self):
        """Verify the summariser ctx-size is 8192 in config.yaml."""
        import yaml
        config_path = pathlib.Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        c = compaction_config(cfg)
        assert c["summarizer_ctx_size"] == 8192

    def test_summarizer_max_tokens_is_512(self):
        """Verify the summariser max tokens is 512 in config.yaml."""
        import yaml
        config_path = pathlib.Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        c = compaction_config(cfg)
        assert c["summarizer_max_tokens"] == 512
