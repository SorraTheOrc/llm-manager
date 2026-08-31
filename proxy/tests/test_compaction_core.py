"""
Proxy-side proactive session compaction — core summarization strategy.

Feature: LP-0MTGBOOJY004OWPI (child of LP-0MTCWE8NG003P0SD).

Verifies the *summarize-first* compaction primitive:

- Trigger thresholds resolve from ``compaction_trigger_ratio`` × the
  effective per-slot context (fast ≈58.3K, cheap ≈43K).
- Per-mode compaction targets (fast ≤ 38K, cheap ≤ 30K).
- Retention invariant: system prompt + very first user prompt verbatim in
  every compacted output (AC1); summary injected between the retained first
  prompt and the recent turns (AC5).
- Non-compactable sessions (summarizer unavailable) resolve to an explicit
  ``remote_with_guidance`` action — never silent (AC4).
- Deterministic output composes with slot save/restore (AC7).

The token estimator is injected so tests are deterministic and independent
of the Qwen3 tokenizer; production wiring passes the routing estimator.
"""
import pytest
from proxy.compaction import (
    CHEAP_COMPACTION_TARGET_TOKENS,
    FAST_COMPACTION_TARGET_TOKENS,
    compaction_target_tokens,
    compaction_trigger_tokens,
    estimate_session_tokens,
    pair_turns,
    plan_session_compaction,
    should_compact_session,
)

# ===================================================================
# Test profiles — mirror config-fast.yaml / config-cheap.yaml
# ===================================================================
# fast  : local_model_ctx_size 262144, session_slot_pool_size 3
#         per_slot = 262144//3 - 4096 = 83285
#         trigger = round-half-up(0.70 × 83285) = 58300
# cheap : local_model_ctx_size 131072, session_slot_pool_size 2
#         per_slot = 131072//2 - 4096 = 61440
#         trigger = round-half-up(0.70 × 61440) = 43008 (spec: ≈43K)


def fast_config(**overrides) -> dict:
    cfg = {
        "server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
            "compaction_trigger_ratio": 0.70,
        }
    }
    server = cfg["server"]
    for key, value in overrides.items():
        if key == "compaction_trigger_ratio":
            server["compaction_trigger_ratio"] = value
        elif key == "local_model_ctx_size":
            server["local_model_ctx_size"] = value
        elif key == "session_slot_pool_size":
            server["session_slot_pool_size"] = value
        else:
            server[key] = value
    return cfg


def cheap_config(**overrides) -> dict:
    return fast_config(local_model_ctx_size=131072, session_slot_pool_size=2, **overrides)


# -------------------------------------------------------------------
# Token estimator (deterministic): each message costs 1000 tokens plus
# one token per 4 content chars. Sessions cross the 58.3K / 43K triggers
# with realistic message counts, without multi-MB fixture strings.
# -------------------------------------------------------------------
_EST_PER_MESSAGE = 1000


def counting_estimator(messages) -> int:
    return _EST_PER_MESSAGE * len(messages) + sum(
        len(str(m.get("content", ""))) for m in messages
    )


def make_session(
    num_turns: int,
    *,
    system_content: str = "SYSTEM_PROMPT",
    first_user_content: str = "FIRST_USER_PROMPT",
) -> list[dict]:
    """Build a chat transcript: system + first user turn + ``num_turns``
    additional (user, assistant) turns."""
    msgs = [{"role": "system", "content": system_content}]
    msgs.append({"role": "user", "content": first_user_content})
    msgs.append({"role": "assistant", "content": "FIRST_ASSISTANT_REPLY"})
    for i in range(1, num_turns + 1):
        msgs.append({"role": "user", "content": f"user_q_{i}"})
        msgs.append({"role": "assistant", "content": f"assistant_a_{i}"})
    return msgs


def fixed_summarizer(middle_messages) -> str:
    """Deterministic summarizer spy: returns a fixed concise summary."""
    return f"MIDDLE_SUMMARY: folded {len(middle_messages)} messages."


# ===================================================================
# Trigger computation
# ===================================================================


class TestTriggerComputation:
    def test_fast_trigger_resolves_58300(self):
        # AC2: fast trigger = 0.70 × (262144//3 - 4096 = 83285) = 58299.5
        # rounded half-up → 58300.
        assert compaction_trigger_tokens("fast", fast_config()) == 58300

    def test_cheap_trigger_resolves_43008(self):
        # AC3: cheap trigger = 0.70 × (131072//2 - 4096 = 61440) = 43008
        # (the spec writes ≈43K).
        assert compaction_trigger_tokens("cheap", cheap_config()) == 43008

    def test_trigger_scales_with_ratio(self):
        cfg = fast_config(compaction_trigger_ratio=0.9)
        # round-half-up(0.9 × 83285 = 74956.5) = 74957
        assert compaction_trigger_tokens("fast", cfg) == 74957

    def test_trigger_zero_when_ratio_zero(self):
        assert (
            compaction_trigger_tokens("fast", fast_config(compaction_trigger_ratio=0))
            == 0
        )

    def test_trigger_zero_when_ctx_size_zero(self):
        assert (
            compaction_trigger_tokens("fast", fast_config(local_model_ctx_size=0)) == 0
        )

    def test_trigger_respects_slot_pool_size(self):
        # 262144//4 - 4096 = 61440 → 0.70 × 61440 = 43008
        assert (
            compaction_trigger_tokens("fast", fast_config(session_slot_pool_size=4))
            == 43008
        )

    def test_should_compact_fast_fires_above_58300(self):
        # AC2: fires at est_tokens > 58,300 — 58,300 itself does NOT fire.
        assert should_compact_session(58300, "fast", fast_config()) is False
        assert should_compact_session(58301, "fast", fast_config()) is True

    def test_should_compact_cheap_fires_above_43008(self):
        assert should_compact_session(43008, "cheap", cheap_config()) is False
        assert should_compact_session(43009, "cheap", cheap_config()) is True

    def test_should_compact_false_when_disabled(self):
        cfg = fast_config(compaction_trigger_ratio=0)
        assert should_compact_session(10_000_000, "fast", cfg) is False


class TestTargetTokens:
    def test_target_constants(self):
        assert FAST_COMPACTION_TARGET_TOKENS == 38000
        assert CHEAP_COMPACTION_TARGET_TOKENS == 30000

    def test_target_tokens_by_mode(self):
        assert compaction_target_tokens("fast") == 38000
        assert compaction_target_tokens("cheap") == 30000
        assert compaction_target_tokens("unknown_mode") == 30000


# ===================================================================
# No-op paths
# ===================================================================


class TestNoopPaths:
    def test_noop_below_trigger(self):
        messages = make_session(5)  # 13 msgs × 1000 → 13K < 58.3K
        result = plan_session_compaction(
            messages, fast_config(), "fast", summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
        )
        assert result["action"] == "noop"
        assert result["reason"] == "below_trigger"
        assert result["messages"] is messages
        assert result["estimated_before"] == result["estimated_after"]
        assert result["summary_text"] is None
        assert result["turns_summarized"] == 0

    def test_noop_when_trigger_disabled(self):
        messages = make_session(80)  # 163 msgs → 163K > trigger
        cfg = fast_config(compaction_trigger_ratio=0)
        result = plan_session_compaction(
            messages, cfg, "fast", summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
        )
        assert result["action"] == "noop"
        assert result["reason"] == "compaction_disabled"

    def test_noop_when_only_system_messages(self):
        # System-only transcript; nothing to compact even over the trigger.
        messages = [
            {"role": "system", "content": f"rule_{i}"} for i in range(80)
        ]  # 80K > 58.3K
        result = plan_session_compaction(
            messages, fast_config(), "fast", summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
        )
        assert result["action"] == "noop"
        assert result["reason"] == "no_user_message"
        assert result["messages"] is messages

    def test_empty_messages_noop(self):
        result = plan_session_compaction(
            [], fast_config(), "fast", summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
        )
        assert result["action"] == "noop"
        assert result["messages"] == []

    def test_noop_keeps_original_objects(self):
        messages = make_session(3)
        result = plan_session_compaction(
            messages, fast_config(), "fast", summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
        )
        assert result["messages"] is messages


# ===================================================================
# Retention invariant (AC1) + summary placement (AC5)
# ===================================================================


class TestRetentionInvariant:
    def _compact_fast(self, messages):
        return plan_session_compaction(
            messages, fast_config(), "fast", summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
        )

    def test_fast_retains_system_and_first_user_verbatim(self):
        messages = make_session(60)  # 123 msgs → 123K > 58.3K
        system_msg, first_user_msg = messages[0], messages[1]
        result = self._compact_fast(messages)
        compacted = result["messages"]
        assert result["action"] == "compact"
        # Same dict objects — verbatim retention, not re-serialization.
        assert compacted[0] is system_msg
        assert compacted[1] is first_user_msg
        assert compacted[0]["role"] == "system"
        assert compacted[1]["content"] == "FIRST_USER_PROMPT"

    def test_cheap_retains_system_and_first_user_verbatim(self):
        messages = make_session(60)  # 123K > 43K
        result = plan_session_compaction(
            messages, cheap_config(), "cheap", summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
        )
        compacted = result["messages"]
        assert result["action"] == "compact"
        assert compacted[0] is messages[0]
        assert compacted[1] is messages[1]
        assert compacted[1]["content"] == "FIRST_USER_PROMPT"

    def test_summary_injected_below_retained_first_prompt(self):
        # AC5: summary is the first message after the retained first prompt.
        result = self._compact_fast(make_session(60))
        compacted = result["messages"]
        assert compacted[1]["role"] == "user"
        assert compacted[1]["content"] == "FIRST_USER_PROMPT"
        summary_msg = compacted[2]
        assert summary_msg["role"] == "user"
        assert summary_msg["content"].startswith(
            "The conversation history before this point was compacted"
        )
        assert "MIDDLE_SUMMARY" in summary_msg["content"]

    def test_recent_turns_follow_summary_in_order(self):
        # The newest turns appear after the summary, in original order.
        result = self._compact_fast(make_session(60))
        compacted = result["messages"]
        tail = [m["content"] for m in compacted[3:]]
        # The last retained user/assistant pair is the newest turn (60).
        # tail alternates user_q_/assistant_a_ for turns 44..60 (17 turns).
        assert "user_q_60" in tail
        assert "assistant_a_60" in tail
        last_q = max(i for i, c in enumerate(tail) if c.startswith("user_q_"))
        # The newest user question is the final user message.
        assert tail[-1] == "assistant_a_60"
        assert tail[last_q] == "user_q_60"
        assert tail[last_q + 1] == "assistant_a_60"


# ===================================================================
# Fast mode (AC2)
# ===================================================================


class TestFastMode:
    def test_fast_compacts_under_38k_target(self):
        result = plan_session_compaction(
            make_session(60), fast_config(), "fast", summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
        )
        assert result["action"] == "compact"
        assert result["reason"] == "compacted_within_target"
        assert result["estimated_before"] > 58300
        assert result["estimated_after"] <= 38000

    def test_fast_summarizes_middle(self):
        result = plan_session_compaction(
            make_session(60), fast_config(), "fast", summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
        )
        assert result["turns_summarized"] > 0
        assert result["recent_turns_kept"] > 0
        compacted = [m["content"] for m in result["messages"]]
        # Early middle content (e.g. q_2) is folded into the summary — it
        # must NOT appear verbatim in the compacted output.
        assert "user_q_2" not in compacted
        assert "MIDDLE_SUMMARY" in " ".join(compacted)

    def test_fast_recent_window_respects_target(self):
        result = plan_session_compaction(
            make_session(60), fast_config(), "fast", summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
        )
        # 3 retained msgs + k whole turns ≤ 38 msgs (38,000 tokens @ 1000/msg).
        # 60 pairs + the first assistant reply = 61 turns after first user.
        assert result["recent_turns_kept"] == 17
        assert result["turns_summarized"] == 61 - 17

    def test_fast_summarizer_receives_middle_messages(self):
        captured = {}

        def spy_summarizer(middle_messages):
            captured["middle"] = list(middle_messages)
            return "SPY_SUMMARY"

        plan_session_compaction(
            make_session(60), fast_config(), "fast", summarizer=spy_summarizer,
            estimate_tokens=counting_estimator,
        )
        middle = captured["middle"]
        middle_contents = [m.get("content") for m in middle]
        # First user prompt and newest turns are NOT in the middle input.
        assert "FIRST_USER_PROMPT" not in middle_contents
        assert "user_q_60" not in middle_contents
        assert "user_q_2" in middle_contents
        assert "FIRST_ASSISTANT_REPLY" in middle_contents


# ===================================================================
# Cheap mode (AC3, AC5)
# ===================================================================


class TestCheapMode:
    def _compact_cheap(self, num_turns=60):
        messages = make_session(num_turns)  # 123K > 43K
        return plan_session_compaction(
            messages, cheap_config(), "cheap", summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
        )

    def test_cheap_compacts_under_30k_target(self):
        result = self._compact_cheap()
        assert result["action"] == "compact"
        assert result["reason"] == "compacted_within_target"
        assert result["estimated_before"] > 43008
        assert result["estimated_after"] <= 30000

    def test_cheap_recent_window_respects_target(self):
        result = self._compact_cheap()
        # 3 retained msgs + k whole turns ≤ 30 msgs (30,000 tokens).
        # 60 pairs + the first assistant reply = 61 turns after first user.
        assert result["recent_turns_kept"] == 13
        assert result["turns_summarized"] == 61 - 13

    def test_cheap_summary_injected_below_first_prompt(self):
        result = self._compact_cheap()
        compacted = result["messages"]
        assert compacted[1]["content"] == "FIRST_USER_PROMPT"
        assert compacted[2]["content"].startswith(
            "The conversation history before this point was compacted"
        )


# ===================================================================
# Non-compactable → remote with guidance (AC4)
# ===================================================================


class TestNonCompactable:
    def test_summarizer_unavailable_routes_remote_with_guidance(self):
        # Cheap mode, over-trigger, summarizer unavailable — the dispatcher
        # must route remote with guidance, NEVER silent.
        messages = make_session(60)
        result = plan_session_compaction(
            messages, cheap_config(), "cheap", summarizer=None,
            estimate_tokens=counting_estimator,
        )
        assert result["action"] == "remote_with_guidance"
        assert result["reason"] == "summarizer_unavailable"
        assert result["messages"] is messages  # untouched
        assert result["estimated_before"] == result["estimated_after"]

    def test_summarizer_unavailable_also_in_fast(self):
        messages = make_session(60)
        result = plan_session_compaction(
            messages, fast_config(), "fast", summarizer=None,
            estimate_tokens=counting_estimator,
        )
        assert result["action"] == "remote_with_guidance"
        assert result["reason"] == "summarizer_unavailable"

    def test_non_compactable_never_silent_below_trigger(self):
        # Even well below the trigger with no summarizer, the result is an
        # explicit noop (never an implicit fallback), and compacted output
        # is never produced without a summary.
        messages = make_session(3)
        result = plan_session_compaction(
            messages, cheap_config(), "cheap", summarizer=None,
            estimate_tokens=counting_estimator,
        )
        assert result["action"] == "noop"
        assert result["reason"] == "below_trigger"


# ===================================================================
# Budget enforcement (AC6)
# ===================================================================


class TestBudgetEnforcement:
    def test_over_budget_reported_when_retention_exceeds_target(self):
        # Pathological retention (huge system prompt) cannot fit under the
        # target even with zero recent turns — the core reports it; the
        # backstop child drops by whole turns.
        messages = make_session(10)
        messages[0]["content"] = "HUGE_SYSTEM_" + "x" * 250000  # ~250K tokens
        result = plan_session_compaction(
            messages, fast_config(), "fast", summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
        )
        assert result["action"] == "compact"
        assert result["reason"] == "compacted_over_budget"
        assert result["recent_turns_kept"] == 0
        assert result["turns_summarized"] == 11  # all turns + first reply
        assert result["estimated_after"] > 38000
        # Retention invariant still holds under over-budget compaction.
        compacted = result["messages"]
        assert compacted[0]["content"].startswith("HUGE_SYSTEM_")
        assert compacted[1]["content"] == "FIRST_USER_PROMPT"

    def test_estimate_session_tokens_uses_injected_estimator(self):
        messages = make_session(10)
        est = estimate_session_tokens(messages, estimate_tokens=counting_estimator)
        assert est == counting_estimator(messages)

    def test_plan_reports_trigger_and_target(self):
        result = plan_session_compaction(
            make_session(60), fast_config(), "fast", summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
        )
        assert result["trigger_tokens"] == 58300
        assert result["target_tokens"] == 38000
        assert result["mode"] == "fast"


# ===================================================================
# Turn pairing + determinism (AC7 / slot save-restore compose)
# ===================================================================


class TestPairingAndDeterminism:
    def test_pair_turns_groups_user_and_assistant(self):
        turns = pair_turns(
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
                {"role": "assistant", "content": "d"},
            ]
        )
        assert len(turns) == 2
        assert [m["content"] for m in turns[0]] == ["a", "b"]
        assert [m["content"] for m in turns[1]] == ["c", "d"]

    def test_pair_turns_leading_assistant_starts_first_turn(self):
        turns = pair_turns(
            [
                {"role": "assistant", "content": "orphan"},
                {"role": "user", "content": "a"},
            ]
        )
        # Lossless: the orphan is preserved as the first turn's opener.
        assert len(turns) == 2
        assert turns[0][0]["content"] == "orphan"

    def test_compaction_is_deterministic(self):
        messages = make_session(60)
        args = (fast_config(), "fast")
        first = plan_session_compaction(
            messages, *args, summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
        )
        second = plan_session_compaction(
            messages, *args, summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
        )
        assert first["messages"] == second["messages"]
        assert first == second

    def test_compacted_messages_preserve_dict_shape(self):
        result = plan_session_compaction(
            make_session(60), fast_config(), "fast", summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
        )
        for msg in result["messages"]:
            assert set(msg.keys()) == {"role", "content"}
            assert msg["role"] in ("system", "user", "assistant")
