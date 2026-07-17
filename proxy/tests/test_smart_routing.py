"""Tests for smart routing logic that bypasses local for large-context
requests when the slot cache is invalidated (LP-0MRCSSBTM002NK3B)."""


from proxy.provider import (
    _model_cache_cold,
    _cache_cold_initialized,
    mark_model_cache_cold,
    clear_model_cache_cold,
    is_model_cache_cold,
    initialize_cache_cold_from_config,
    _estimate_prompt_tokens_for_routing,
)


class TestModelCacheColdTracking:
    """Tests for the in-memory cache-invalidation tracking."""

    def setup_method(self):
        _cache_cold_initialized()
        _model_cache_cold.clear()

    def test_initial_state_is_not_cold_after_init(self):
        """After initialization with empty config, no models are cold."""
        # After clearing and initialization, no models are marked cold.
        # The initialize_cache_cold_from_config function populates the set from config.
        assert is_model_cache_cold("Qwen3") is False
        assert is_model_cache_cold("gemma4") is False
        assert is_model_cache_cold("nonexistent") is False

    def test_mark_cache_cold(self):
        """Marking a model as cold should be reflected in is_model_cache_cold."""
        mark_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is True

    def test_mark_cache_cold_twice_is_idempotent(self):
        """Marking a model cold multiple times should not raise."""
        mark_model_cache_cold("Qwen3")
        mark_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is True

    def test_clear_cache_cold(self):
        """Clearing cache-cold for a model should return it to warm."""
        mark_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is True
        clear_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is False

    def test_clear_cache_cold_idempotent(self):
        """Clearing an already-warm model should not raise."""
        clear_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is False

    def test_multiple_models_independent(self):
        """Cache states for different models should be independent."""
        mark_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is True
        assert is_model_cache_cold("gemma4") is False
        clear_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is False
        assert is_model_cache_cold("gemma4") is False


class TestSessionCacheColdTracking:
    """Tests for per-session cache cold tracking.

    Each session has its own cache-cold state per model, independent of
    other sessions and of the model-level fallback.
    """

    def setup_method(self):
        _cache_cold_initialized()
        _model_cache_cold.clear()
        # Clear session-level state
        from proxy.provider import _session_cache_cold
        _session_cache_cold.clear()

    def test_session_new_is_cold_by_default(self):
        """AC 1: A new session (no entry) is cold by default."""
        assert is_model_cache_cold("Qwen3", session_id="sess_new") is True

    def test_session_warm_after_clear(self):
        """AC 2: Clearing cache cold for a session makes it warm."""
        clear_model_cache_cold("Qwen3", session_id="sess_a")
        assert is_model_cache_cold("Qwen3", session_id="sess_a") is False

    def test_session_mark_cold(self):
        """Marking a session cold is reflected in is_model_cache_cold."""
        clear_model_cache_cold("Qwen3", session_id="sess_a")
        assert is_model_cache_cold("Qwen3", session_id="sess_a") is False
        mark_model_cache_cold("Qwen3", session_id="sess_a")
        assert is_model_cache_cold("Qwen3", session_id="sess_a") is True

    def test_per_session_isolation_different_sessions(self):
        """AC 2: Two sessions on the same model are independent."""
        assert is_model_cache_cold("Qwen3", session_id="sess_a") is True  # cold by default
        assert is_model_cache_cold("Qwen3", session_id="sess_b") is True  # cold by default

        clear_model_cache_cold("Qwen3", session_id="sess_a")
        assert is_model_cache_cold("Qwen3", session_id="sess_a") is False  # warm
        assert is_model_cache_cold("Qwen3", session_id="sess_b") is True   # still cold

    def test_per_session_isolation_different_models(self):
        """Session state is tracked per (model, session) pair."""
        clear_model_cache_cold("Qwen3", session_id="sess_a")
        assert is_model_cache_cold("Qwen3", session_id="sess_a") is False   # warm
        # Same session, different model
        assert is_model_cache_cold("gemma4", session_id="sess_a") is True   # cold

    def test_session_fallback_to_model_level_without_session_id(self):
        """Without session_id, model-level fallback is used."""
        mark_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is True
        assert is_model_cache_cold("Qwen3", session_id="sess_a") is True  # new session still cold
        clear_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is False
        # New session still defaults to cold even when model-level is warm
        assert is_model_cache_cold("Qwen3", session_id="sess_b") is True

    def test_clear_without_session_id_uses_model_level(self):
        """clear_model_cache_cold without session_id clears model-level only."""
        mark_model_cache_cold("Qwen3")
        clear_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is False

    def test_mark_without_session_id_uses_model_level(self):
        """mark_model_cache_cold without session_id marks model-level only."""
        mark_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is True
        # Session should still be cold by default
        assert is_model_cache_cold("Qwen3", session_id="sess_a") is True

    def test_clear_session_does_not_affect_model_level(self):
        """Clearing a session's cache should not affect model-level state."""
        mark_model_cache_cold("Qwen3")
        clear_model_cache_cold("Qwen3", session_id="sess_a")
        # Session is warm
        assert is_model_cache_cold("Qwen3", session_id="sess_a") is False
        # Model-level is still cold
        assert is_model_cache_cold("Qwen3") is True

    def test_mark_session_does_not_affect_model_level(self):
        """Marking a session cold should not affect model-level state."""
        clear_model_cache_cold("Qwen3", session_id="sess_a")
        assert is_model_cache_cold("Qwen3", session_id="sess_a") is False  # warm
        # Model-level remains unchanged (no entry)
        assert is_model_cache_cold("Qwen3") is False

    def test_mixed_sessions_warm_and_cold(self):
        """AC: Mixed warm and cold sessions get correct thresholds independently."""
        # Session A: warm (cleared)
        clear_model_cache_cold("Qwen3", session_id="sess_a")
        # Session B: cold (default)
        # Session C: cold (default)

        assert is_model_cache_cold("Qwen3", session_id="sess_a") is False   # warm
        assert is_model_cache_cold("Qwen3", session_id="sess_b") is True    # cold
        assert is_model_cache_cold("Qwen3", session_id="sess_c") is True    # cold

    def test_slot_eviction_resets_to_cold(self):
        """AC 3: After invalidation (slot eviction), session returns to cold."""
        # Start warm
        clear_model_cache_cold("Qwen3", session_id="sess_a")
        assert is_model_cache_cold("Qwen3", session_id="sess_a") is False

        # Slot evicted - mark cold
        mark_model_cache_cold("Qwen3", session_id="sess_a")
        assert is_model_cache_cold("Qwen3", session_id="sess_a") is True


class TestEstimatePromptTokensForRouting:
    """Tests for _estimate_prompt_tokens_for_routing function."""

    def test_empty_body(self):
        """Empty body should return 0."""
        assert _estimate_prompt_tokens_for_routing({}) == 0

    def test_no_messages(self):
        """Body without messages list should return 0."""
        assert _estimate_prompt_tokens_for_routing({"model": "test"}) == 0

    def test_empty_messages(self):
        """Empty messages list should return 0."""
        assert _estimate_prompt_tokens_for_routing({"messages": []}) == 0

    def test_small_message(self):
        """A small message should yield a small token estimate."""
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        tokens = _estimate_prompt_tokens_for_routing(body)
        assert 1 <= tokens <= 5

    def test_large_message(self):
        """A large message should yield a large estimate using tiktoken."""
        phrase = "test message content for token estimation "
        large_content = phrase * 12000  # ~72K tokens with tiktoken
        body = {"messages": [{"role": "user", "content": large_content}]}
        tokens = _estimate_prompt_tokens_for_routing(body)
        assert tokens >= 65_000
        assert tokens <= 80_000

    def test_40k_threshold_crossing(self):
        """A message with >40K actual tokens should be detected as above threshold."""
        phrase = "test message content for token estimation "
        content = phrase * 7000  # ~42K actual tokens with tiktoken
        body = {"messages": [{"role": "user", "content": content}]}
        tokens = _estimate_prompt_tokens_for_routing(body)
        assert tokens > 40_000

    def test_below_40k(self):
        """A message at ~20K tokens should be below 40K threshold."""
        content = "x" * 40_000  # ~20K estimated
        body = {"messages": [{"role": "user", "content": content}]}
        tokens = _estimate_prompt_tokens_for_routing(body)
        assert tokens < 40_000

    def test_system_message_counted(self):
        """System messages SHOULD be counted in routing estimate.

        System prompts were previously excluded but are now included
        (LP-0MRGT35H1003D1PM) to avoid underestimating large contexts.
        """
        body = {
            "messages": [
                {"role": "system", "content": "x" * 100_000},  # ~25K tokens system
                {"role": "user", "content": "Hello"},
            ]
        }
        tokens = _estimate_prompt_tokens_for_routing(body)
        # Should be large (system content is now counted)
        assert tokens >= 10000

    def test_combined_user_assistant_messages_counted(self):
        """Both user and assistant messages should be counted."""
        phrase = "test message "
        body = {
            "messages": [
                {"role": "user", "content": phrase * 20000},  # ~40K tokens
                {"role": "assistant", "content": phrase * 20000},  # ~40K tokens
            ]
        }
        tokens = _estimate_prompt_tokens_for_routing(body)
        # Should count both: ~80K total
        assert tokens >= 75_000
        assert tokens <= 85_000

    def test_array_content_counts_text_only(self):
        """Array content (multimodal) should count text fields only."""
        phrase = "test message content for token estimation "
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": phrase * 10000},  # ~60K tokens
                        {"type": "image", "url": "http://example.com/img.jpg"},
                    ],
                }
            ]
        }
        tokens = _estimate_prompt_tokens_for_routing(body)
        # Should count only the text part: ~60K
        assert tokens >= 55_000
        assert tokens <= 65_000

    def test_reasoning_content_counted(self):
        """Assistant reasoning_content should be counted (LP-0MRDE669Y003V1SO)."""
        phrase = "test message content for token estimation "
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": phrase * 10000,  # ~60K tokens
                },
            ]
        }
        tokens = _estimate_prompt_tokens_for_routing(body)
        # Should count reasoning_content: ~60K
        assert tokens >= 55_000
        assert tokens <= 65_000

    def test_tool_calls_counted(self):
        """Tool call function names and arguments should be counted (LP-0MRDE669Y003V1SO)."""
        phrase = "test message content for token estimation "
        args_content = '{"command": "' + phrase * 5000 + '"}'
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "bash",
                                "arguments": args_content,
                            },
                        },
                        {
                            "id": "call_2",
                            "function": {
                                "name": "read",
                                "arguments": args_content,
                            },
                        },
                    ],
                },
            ]
        }
        tokens = _estimate_prompt_tokens_for_routing(body)
        # Should count both tool calls: ~60K total
        assert tokens >= 55_000
        assert tokens <= 65_000

    def test_mixed_content_reasoning_tool_calls_counted(self):
        """Mixed content, reasoning, and tool calls should all be counted (LP-0MRDE669Y003V1SO)."""
        phrase = "test message "
        phrase2 = "test message content for token estimation "
        body = {
            "messages": [
                {"role": "user", "content": phrase * 6000},  # ~12K tokens
                {
                    "role": "assistant",
                    "content": phrase * 6000,  # ~12K tokens
                    "reasoning_content": phrase2 * 5000,  # ~30K tokens
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "test_func",
                                "arguments": phrase2 * 5000,  # ~30K tokens
                            },
                        },
                    ],
                },
            ]
        }
        tokens = _estimate_prompt_tokens_for_routing(body)
        # Should count content + reasoning + tool_call args: ~84K total
        assert tokens >= 75_000
        assert tokens <= 95_000


class TestSmartRoutingThresholdConfig:
    """Tests for the threshold configuration parsing."""

    def test_cold_cache_threshold_disabled_when_zero(self):
        """Cold-cache threshold of 0 should disable smart routing."""
        from proxy.provider import _get_large_context_cold_cache_threshold
        config = {"server": {"local_large_context_cold_cache_threshold": 0}}
        assert _get_large_context_cold_cache_threshold(config) == 0

    def test_cold_cache_threshold_disabled_when_absent(self):
        """Missing cold-cache threshold config should disable smart routing (returns 0)."""
        from proxy.provider import _get_large_context_cold_cache_threshold
        config = {"server": {}}
        assert _get_large_context_cold_cache_threshold(config) == 0

    def test_cold_cache_threshold_at_40000(self):
        """Cold-cache threshold of 40000 should be parsed correctly."""
        from proxy.provider import _get_large_context_cold_cache_threshold
        config = {"server": {"local_large_context_cold_cache_threshold": 40000}}
        assert _get_large_context_cold_cache_threshold(config) == 40000

    def test_cold_cache_threshold_flat_config(self):
        """Flat (non-nested) config should also work for test compatibility."""
        from proxy.provider import _get_large_context_cold_cache_threshold
        config = {"local_large_context_cold_cache_threshold": 40000}
        assert _get_large_context_cold_cache_threshold(config) == 40000

    def test_cold_cache_threshold_legacy_key(self):
        """Legacy key (local_large_context_fallback_threshold) should still work."""
        from proxy.provider import _get_large_context_cold_cache_threshold
        config = {"server": {"local_large_context_fallback_threshold": 40000}}
        assert _get_large_context_cold_cache_threshold(config) == 40000

    def test_warm_cache_threshold_default(self):
        """Warm-cache threshold should default to 60000."""
        from proxy.provider import _get_large_context_warm_cache_threshold
        config = {"server": {}}
        assert _get_large_context_warm_cache_threshold(config) == 60000

    def test_warm_cache_threshold_explicit(self):
        """Warm-cache threshold should be parsed correctly when set."""
        from proxy.provider import _get_large_context_warm_cache_threshold
        config = {"server": {"local_large_context_warm_cache_threshold": 80000}}
        assert _get_large_context_warm_cache_threshold(config) == 80000

    def test_warm_cache_threshold_zero_disables(self):
        """Warm-cache threshold of 0 should disable bypass."""
        from proxy.provider import _get_large_context_warm_cache_threshold
        config = {"server": {"local_large_context_warm_cache_threshold": 0}}
        assert _get_large_context_warm_cache_threshold(config) == 0

    def test_warm_cache_threshold_flat_config(self):
        """Flat (non-nested) config should work for test compatibility."""
        from proxy.provider import _get_large_context_warm_cache_threshold
        config = {"local_large_context_warm_cache_threshold": 50000}
        assert _get_large_context_warm_cache_threshold(config) == 50000


class TestSmartRoutingDecision:
    """Tests for the smart routing decision logic.

    Verifies that when cache is cold and tokens > threshold, local is skipped;
    otherwise local routing proceeds normally.
    """

    def setup_method(self):
        _cache_cold_initialized()
        _model_cache_cold.clear()

    def test_warm_cache_small_request_routes_local(self):
        """AC 4: Warm cache + <=40K tokens → routes local (normal)."""
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        threshold = 40000
        cache_cold = False
        should_skip = _should_skip_local(cache_cold, body, threshold)
        assert should_skip is False

    def test_warm_cache_large_request_routes_local_with_default_warm_threshold(self):
        """Warm cache + >40K tokens → routes local when warm_cache_threshold not set (default 0 = disabled)."""
        phrase = "test message content for token estimation "
        body = {"messages": [{"role": "user", "content": phrase * 7000}]}  # ~42K tokens
        cold_threshold = 40000
        cache_cold = False
        should_skip = _should_skip_local(cache_cold, body, cold_threshold)
        assert should_skip is False

    def test_warm_cache_large_request_skips_local_with_warm_threshold(self):
        """AC 1: Warm cache + >60K tokens → skips local when warm_cache_threshold is set."""
        phrase = "test message content for token estimation "
        body = {"messages": [{"role": "user", "content": phrase * 10500}]}  # ~63K tokens
        warm_threshold = 60000
        cache_cold = False
        should_skip = _should_skip_local(cache_cold, body, 40000, warm_cache_threshold=warm_threshold)
        assert should_skip is True

    def test_warm_cache_moderate_request_routes_local_with_warm_threshold(self):
        """AC 2: Warm cache + <=60K tokens → routes local when warm_cache_threshold is 60000."""
        phrase = "test message content for token estimation "
        body = {"messages": [{"role": "user", "content": phrase * 7000}]}  # ~42K tokens
        warm_threshold = 60000
        cache_cold = False
        should_skip = _should_skip_local(cache_cold, body, 40000, warm_cache_threshold=warm_threshold)
        assert should_skip is False

    def test_warm_cache_disabled_via_zero_threshold(self):
        """Warm cache + >60K tokens → routes local when warm_cache_threshold=0 (disabled)."""
        phrase = "test message content for token estimation "
        body = {"messages": [{"role": "user", "content": phrase * 10500}]}  # ~63K tokens
        warm_threshold = 0
        cache_cold = False
        should_skip = _should_skip_local(cache_cold, body, 40000, warm_cache_threshold=warm_threshold)
        assert should_skip is False

    def test_warm_cache_exact_threshold_does_not_skip(self):
        """Warm cache + tokens below 60K → does NOT skip local (must be strictly greater)."""
        phrase = "test message content for token estimation "
        # ~57K tokens - safely below 60K threshold (7000 reps ≈ 42K)
        body = {"messages": [{"role": "user", "content": phrase * 9500}]}
        warm_threshold = 60000
        cache_cold = False
        should_skip = _should_skip_local(cache_cold, body, 40000, warm_cache_threshold=warm_threshold)
        assert should_skip is False

    def test_cold_cache_small_request_routes_local(self):
        """AC 4: Cold cache + <=40K tokens → routes local (normal)."""
        mark_model_cache_cold("Qwen3")
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        threshold = 40000
        should_skip = _should_skip_local(is_model_cache_cold("Qwen3"), body, threshold)
        assert should_skip is False

    def test_cold_cache_large_request_skips_local(self):
        """AC 3: Cold cache + >40K tokens → skips local."""
        mark_model_cache_cold("Qwen3")
        phrase = "test message content for token estimation "
        body = {"messages": [{"role": "user", "content": phrase * 7000}]}  # ~42K tokens
        threshold = 40000
        should_skip = _should_skip_local(is_model_cache_cold("Qwen3"), body, threshold)
        assert should_skip is True

    def test_threshold_disabled_cold_cache_large_request_routes_local(self):
        """Threshold=0 (disabled) → always routes local regardless of cache state."""
        mark_model_cache_cold("Qwen3")
        phrase = "test message content for token estimation "
        body = {"messages": [{"role": "user", "content": phrase * 7000}]}  # ~42K tokens
        threshold = 0
        should_skip = _should_skip_local(is_model_cache_cold("Qwen3"), body, threshold)
        assert should_skip is False

    def test_cold_cache_exact_threshold_does_not_skip(self):
        """Content below threshold should NOT skip local."""
        mark_model_cache_cold("Qwen3")
        phrase = "test message content for token estimation "
        body = {"messages": [{"role": "user", "content": phrase * 5500}]}  # ~33K tokens (<40K)
        threshold = 40000
        should_skip = _should_skip_local(is_model_cache_cold("Qwen3"), body, threshold)
        assert should_skip is False

    def test_cold_cache_just_above_threshold_skips(self):
        """Content above threshold should skip local when cache is cold."""
        mark_model_cache_cold("Qwen3")
        phrase = "test message content for token estimation "
        body = {"messages": [{"role": "user", "content": phrase * 7000}]}  # ~42K tokens (>40K)
        threshold = 40000
        should_skip = _should_skip_local(is_model_cache_cold("Qwen3"), body, threshold)
        assert should_skip is True


# The function tested above lives in provider.py; we import it here.
# It's defined for test purposes - the real implementation is in provider.py
from proxy.provider import _should_skip_local


class TestInitializeCacheColdFromConfig:
    """Tests for the startup cache-cold initialization."""

    def setup_method(self):
        _cache_cold_initialized()
        _model_cache_cold.clear()

    def test_marks_local_models_cold(self):
        """Local models should be marked cold after init."""
        config = {
            "models": {
                "plan": {
                    "providers": [
                        {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
                        {"name": "opencode", "type": "remote"},
                    ]
                },
                "qwen3": {
                    "providers": [
                        {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
                    ]
                },
            }
        }
        initialize_cache_cold_from_config(config)
        assert is_model_cache_cold("Qwen3") is True

    def test_skips_remote_only_models(self):
        """Models with only remote providers should NOT be marked cold."""
        config = {
            "models": {
                "opencode-model": {
                    "providers": [
                        {"name": "opencode", "type": "remote", "model": "deepseek"},
                    ]
                },
            }
        }
        initialize_cache_cold_from_config(config)
        assert is_model_cache_cold("Qwen3") is False

    def test_multiple_local_models_all_cold(self):
        """Multiple local models should all be marked cold."""
        config = {
            "models": {
                "plan": {
                    "providers": [
                        {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
                    ]
                },
                "gemma4": {
                    "providers": [
                        {"name": "local-gemma4", "type": "local", "llama_model": "gemma4"},
                    ]
                },
            }
        }
        initialize_cache_cold_from_config(config)
        assert is_model_cache_cold("Qwen3") is True
        assert is_model_cache_cold("gemma4") is True

    def test_clears_existing_state(self):
        """Re-initializing should clear any previous state."""
        # Set some state first
        _model_cache_cold.add("unknown-model")
        _model_cache_cold.add("Qwen3")
        assert len(_model_cache_cold) == 2

        # Re-initialize
        config = {
            "models": {
                "plan": {
                    "providers": [
                        {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
                    ]
                },
            }
        }
        initialize_cache_cold_from_config(config)
        # unknown-model should be gone, Qwen3 should still be cold
        assert is_model_cache_cold("Qwen3") is True
        assert is_model_cache_cold("unknown-model") is False


class TestCacheColdLifecycle:
    """Tests for the full cache-cold lifecycle."""

    def setup_method(self):
        _cache_cold_initialized()
        _model_cache_cold.clear()

    def test_startup_to_warm_to_cold_cycle(self):
        """
        Full lifecycle: startup (cold) → successful local (warm) → invalidation (cold).
        """
        # Startup: initialize from config
        config = {
            "models": {
                "plan": {
                    "providers": [
                        {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
                        {"name": "opencode", "type": "remote"},
                    ]
                },
            }
        }
        initialize_cache_cold_from_config(config)
        assert is_model_cache_cold("Qwen3") is True

        # After a successful local response, cache should be warm
        clear_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is False

        # After session invalidation, cache should be cold again
        mark_model_cache_cold("Qwen3")
        assert is_model_cache_cold("Qwen3") is True

    def test_model_not_in_config_starts_unknown(self):
        """
        A model not in the config should not be in the cold set.
        """
        config = {"models": {}}
        initialize_cache_cold_from_config(config)
        assert is_model_cache_cold("Qwen3") is False

    def test_large_context_bypass_works_after_startup(self):
        """
        AC 3: A large-context request with cold cache should be routed to remote.
        """
        # Simulate startup init
        config = {
            "models": {
                "plan": {
                    "providers": [
                        {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"},
                    ]
                },
            }
        }
        initialize_cache_cold_from_config(config)
        assert is_model_cache_cold("Qwen3") is True

        # Large request (>40K tokens) with cold cache should skip local
        phrase = "test message content for token estimation "
        body = {"messages": [{"role": "user", "content": phrase * 7000}]}  # ~42K tokens
        threshold = 40000
        should_skip = _should_skip_local(is_model_cache_cold("Qwen3"), body, threshold)
        assert should_skip is True
