
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
from proxy.compaction import (
    compaction_trigger_tokens,
    decide_session_compaction,
    should_compact_session,
)
from proxy.provider import (
    _DEFAULT_COMPACTION_TRIGGER_RATIO,
    _DEFAULT_SUMMARIZER_CTX_SIZE,
    _DEFAULT_SUMMARIZER_MAX_TOKENS,
    _DEFAULT_SUMMARIZER_REASONING_EFFORT,
    _DEFAULT_SUMMARIZER_RETRIES,
    _DEFAULT_SUMMARIZER_RETRY_DELAY_SECONDS,
    _DEFAULT_SUMMARIZER_TIMEOUT_SECONDS,
    _SUMMARIZATION_PROMPT,
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

    def test_system_prompt_matches_pi_role_and_guard_rails(self):
        """Pi verbatim: role + Do NOT continue/respond/ONLY-output guard rails."""
        assert _SUMMARIZER_SYSTEM_PROMPT.startswith(
            "You are a context summarization assistant."
        )
        assert "Do NOT continue the conversation." in _SUMMARIZER_SYSTEM_PROMPT
        assert "Do NOT respond to any questions" in _SUMMARIZER_SYSTEM_PROMPT
        assert "ONLY output the structured summary." in _SUMMARIZER_SYSTEM_PROMPT

    def test_system_prompt_has_no_format_template(self):
        """Structured sections live in the USER template, mirroring Pi's split.

        R1 (LP-0MTTPXHTI0081WIR) merged the format template into the system
        prompt; the R1-companion (LP-0MTTSL2AW000A5OG) restores Pi's exact
        system/user split, so no "## Goal"-style sections belong here.
        """
        assert "Use this EXACT format" not in _SUMMARIZER_SYSTEM_PROMPT
        assert "## Goal" not in _SUMMARIZER_SYSTEM_PROMPT
        assert "## Next Steps" not in _SUMMARIZER_SYSTEM_PROMPT

    def test_system_prompt_is_pi_verbatim(self):
        """Exact Pi SUMMARIZATION_SYSTEM_PROMPT text (guard against drift)."""
        expected = (
            "You are a context summarization assistant. Your task is to read a "
            "conversation between a user and an AI coding assistant, then produce "
            "a structured summary following the exact format specified.\n"
            "\n"
            "Do NOT continue the conversation. Do NOT respond to any questions "
            "in the conversation. ONLY output the structured summary."
        )
        assert _SUMMARIZER_SYSTEM_PROMPT == expected


class TestSummarizationPromptTemplate:
    """Pi's SUMMARIZATION_PROMPT — the structured format template delivered in
    the summarizer's USER message after the <conversation> serialization."""

    def test_template_is_non_empty(self):
        """_SUMMARIZATION_PROMPT must be a non-empty string."""
        assert isinstance(_SUMMARIZATION_PROMPT, str)
        assert len(_SUMMARIZATION_PROMPT) > 0

    def test_template_starts_with_conversation_reference(self):
        """Pi verbatim opening: the conversation is serialised above the template."""
        assert _SUMMARIZATION_PROMPT.startswith(
            "The messages above are a conversation to summarize."
        )

    def test_template_has_structured_sections(self):
        """All Pi structured sections are present."""
        for section in (
            "## Goal",
            "## Constraints & Preferences",
            "## Progress",
            "### Done",
            "### In Progress",
            "### Blocked",
            "## Key Decisions",
            "## Next Steps",
            "## Critical Context",
        ):
            assert section in _SUMMARIZATION_PROMPT

    def test_template_preserves_exact_references_instruction(self):
        """Verbatim Pi tail: keep paths/function names/errors exact."""
        assert (
            "Keep each section concise. Preserve exact file paths, function "
            "names, and error messages." in _SUMMARIZATION_PROMPT
        )


class TestCompactionRetryConfig:
    """Retry policy for transient summarizer failures (R3)."""

    def test_defaults(self):
        c = compaction_config({})
        assert c["summarizer_retries"] == _DEFAULT_SUMMARIZER_RETRIES
        assert (
            c["summarizer_retry_delay_seconds"]
            == _DEFAULT_SUMMARIZER_RETRY_DELAY_SECONDS
        )

    def test_server_overrides(self):
        cfg = {
            "server": {
                "compaction_summarizer_retries": 5,
                "compaction_summarizer_retry_delay_seconds": 1.5,
            }
        }
        c = compaction_config(cfg)
        assert c["summarizer_retries"] == 5
        assert c["summarizer_retry_delay_seconds"] == 1.5

    def test_flat_overrides(self):
        cfg = {"compaction_summarizer_retries": 0}
        c = compaction_config(cfg)
        assert c["summarizer_retries"] == 0  # 0 disables retries

    def test_explicit_zero_delay(self):
        cfg = {"server": {"compaction_summarizer_retry_delay_seconds": 0}}
        c = compaction_config(cfg)
        assert c["summarizer_retry_delay_seconds"] == 0.0

    def test_negative_values_clamped_to_zero(self):
        cfg = {
            "server": {
                "compaction_summarizer_retries": -3,
                "compaction_summarizer_retry_delay_seconds": -1,
            }
        }
        c = compaction_config(cfg)
        assert c["summarizer_retries"] == 0
        assert c["summarizer_retry_delay_seconds"] == 0.0

    def test_bad_types_fall_back_to_defaults(self):
        cfg = {
            "server": {
                "compaction_summarizer_retries": "lots",
                "compaction_summarizer_retry_delay_seconds": "soon",
            }
        }
        c = compaction_config(cfg)
        assert c["summarizer_retries"] == _DEFAULT_SUMMARIZER_RETRIES
        assert (
            c["summarizer_retry_delay_seconds"]
            == _DEFAULT_SUMMARIZER_RETRY_DELAY_SECONDS
        )


class TestCompactionTimeoutConfig:
    """HTTP timeout for the local summarizer (LP-0MU1RXEY10075TUU).

    The default is 600 s so the summarizer can wait for the single local
    slot to free up while a long generating request holds it; 30 s was
    too short and caused a 100% summarizer_failed rate.
    """

    def test_defaults(self):
        c = compaction_config({})
        assert c["summarizer_timeout_seconds"] == _DEFAULT_SUMMARIZER_TIMEOUT_SECONDS

    def test_server_override(self):
        cfg = {"server": {"compaction_summarizer_timeout": 120}}
        c = compaction_config(cfg)
        assert c["summarizer_timeout_seconds"] == 120.0

    def test_flat_override(self):
        cfg = {"compaction_summarizer_timeout": 45.5}
        c = compaction_config(cfg)
        assert c["summarizer_timeout_seconds"] == 45.5

    def test_values_clamped_to_minimum(self):
        # Sub-1s timeouts are nonsense for a slot wait; clamp to 1 s.
        cfg = {"server": {"compaction_summarizer_timeout": 0.1}}
        c = compaction_config(cfg)
        assert c["summarizer_timeout_seconds"] == 1.0

    def test_bad_type_falls_back_to_default(self):
        cfg = {"server": {"compaction_summarizer_timeout": "soon"}}
        c = compaction_config(cfg)
        assert c["summarizer_timeout_seconds"] == _DEFAULT_SUMMARIZER_TIMEOUT_SECONDS


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


class TestSummarizerReasoningConfig:
    """Reasoning suppression for the compaction summarizer (LP-0MU58PBRD004OV1I)."""

    def test_defaults(self):
        """Default effort is 'minimal' and local thinking is disabled."""
        c = compaction_config({})
        assert c["summarizer_reasoning_effort"] == _DEFAULT_SUMMARIZER_REASONING_EFFORT
        assert c["summarizer_disable_thinking"] is True

    def test_reasoning_effort_override(self):
        c = compaction_config({"server": {"summarizer_reasoning_effort": "low"}})
        assert c["summarizer_reasoning_effort"] == "low"

    def test_reasoning_effort_flat_key(self):
        c = compaction_config({"summarizer_reasoning_effort": "medium"})
        assert c["summarizer_reasoning_effort"] == "medium"

    def test_reasoning_effort_explicit_null_disables_override(self):
        """An explicit null means 'use the upstream default', not the default."""
        c = compaction_config({"server": {"summarizer_reasoning_effort": None}})
        assert c["summarizer_reasoning_effort"] is None

    def test_reasoning_effort_blank_becomes_none(self):
        c = compaction_config({"server": {"summarizer_reasoning_effort": "  "}})
        assert c["summarizer_reasoning_effort"] is None

    def test_disable_thinking_override_false(self):
        c = compaction_config({"server": {"summarizer_disable_thinking": False}})
        assert c["summarizer_disable_thinking"] is False

    def test_disable_thinking_non_bool_falls_back_to_default(self):
        c = compaction_config({"server": {"summarizer_disable_thinking": "yes"}})
        assert c["summarizer_disable_thinking"] is True


# ===================================================================
# Trigger threshold consistency after estimator unification
# (LP-0MU5FU2YG009KN57, parent LP-0MU5A84YU003YOTY)
# ===================================================================


def _fast_schedule() -> dict:
    """3-slot fast schedule: per-slot 83,285 → trigger 58,300."""
    return {
        "server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
            "compaction_trigger_ratio": 0.70,
        }
    }


def _cheap_schedule() -> dict:
    """2-slot cheap schedule: per-slot 61,440 → trigger 43,008 (≈43K)."""
    return {
        "server": {
            "local_model_ctx_size": 131072,
            "session_slot_pool_size": 2,
            "compaction_trigger_ratio": 0.70,
        }
    }


def _compactable_messages() -> list:
    """System + first user + two whole turns (so compaction has work to do)."""
    return [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "FIRST"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]


def _stub_summarizer(middle_messages, previous_summary=None) -> str:
    return "SUMMARY"


class TestTriggerThresholdConsistency:
    """The operator-approved trigger constants are re-verified against the
    now-unified estimator (LP-0MU5FU2YG009KN57 AC2/AC3).

    The estimator unification changed only *which* tokenizer the compaction
    path uses (tiktoken → the same native Qwen3 tokenizer routing already
    used). The documented derivations remain valid: the constants were tuned
    from routing-time ``estimated_tokens``, i.e. the unified estimate.
    """

    def test_fast_trigger_still_resolves_58300(self):
        # 0.70 × (262144//3 − 4096 = 83285) = 58299.5 → round half-up 58300.
        assert compaction_trigger_tokens("fast", _fast_schedule()) == 58300

    def test_cheap_trigger_still_resolves_43008(self):
        # 0.70 × (131072//2 − 4096 = 61440) = 43008.
        assert compaction_trigger_tokens("cheap", _cheap_schedule()) == 43008

    def test_should_compact_boundary_is_strictly_above(self):
        # 58,300 itself does NOT fire; 58,301 does (unchanged semantics).
        assert should_compact_session(58300, "fast", _fast_schedule()) is False
        assert should_compact_session(58301, "fast", _fast_schedule()) is True
        assert should_compact_session(43008, "cheap", _cheap_schedule()) is False
        assert should_compact_session(43009, "cheap", _cheap_schedule()) is True

    def test_unified_estimate_does_not_under_count_vs_legacy_tiktoken(self):
        """AC1: no compaction-frequency regression.

        The thresholds were derived from the native (Qwen3) routing estimate.
        The legacy compaction path used tiktoken, which undercounts dense
        prose, so it under-compacted.  On the canonical dense-prose fixture
        the unified estimate is >= the legacy tiktoken estimate — the fix can
        only compact *more* sessions, never fewer.
        """
        from proxy.provider import (
            _estimate_prompt_tokens_for_routing,
            _get_tokenizer_for_model,
        )
        try:
            from benchmarks import slot_benchmark as sb
        except ImportError:  # pragma: no cover - package layout fallback
            from proxy.benchmarks import slot_benchmark as sb

        messages = [{"role": "user", "content": sb.generate_large_prompt_fixture(46000)}]
        tokenizer, multiplier = _get_tokenizer_for_model(
            {"tokenizer": "qwen3"}, _fast_schedule()
        )
        assert tokenizer is not None and multiplier == 1.0
        unified = _estimate_prompt_tokens_for_routing(
            {"messages": messages}, tokenizer=tokenizer
        )
        legacy_tiktoken = _estimate_prompt_tokens_for_routing({"messages": messages})
        assert unified >= legacy_tiktoken, (
            f"unified estimate ({unified}) must not fall below the legacy "
            f"tiktoken estimate ({legacy_tiktoken}) — sessions that compacted "
            f"before must still compact"
        )

    @pytest.mark.parametrize(
        "config_factory,trigger",
        [(_fast_schedule, 58300), (_cheap_schedule, 43008)],
        ids=["fast-58300", "cheap-43008"],
    )
    def test_decision_boundary_noop_below_and_compact_above(self, config_factory, trigger):
        """AC4: a session *at* the exact trigger is a ``below_trigger``
        noop; one token above it produces a real compaction decision."""
        messages = _compactable_messages()

        at_trigger = decide_session_compaction(
            messages,
            config_factory(),
            "fast",
            summarizer=_stub_summarizer,
            estimate_tokens=lambda _msgs: trigger,
            session_id="sess-boundary-at",
        )
        assert at_trigger["action"] == "noop"
        assert at_trigger["reason"] == "below_trigger"
        assert at_trigger["estimated_before"] == trigger

        above_trigger = decide_session_compaction(
            messages,
            config_factory(),
            "fast",
            summarizer=_stub_summarizer,
            estimate_tokens=lambda _msgs: trigger + 1,
            session_id="sess-boundary-above",
        )
        assert above_trigger["reason"] != "below_trigger"
        assert above_trigger["action"] in ("compact", "remote_with_guidance")
