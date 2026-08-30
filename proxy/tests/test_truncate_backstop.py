"""
Truncate backstop for both modes.

Feature: LP-0MTGBOYJX006KVN8 (child of LP-0MTCWE8NG003P0SD).

When summarization alone cannot bring the session under its per-mode
budget, the backstop drops the oldest whole recent turns — a logged safety
net, never the primary mechanism.

- AC1: only triggers when the summary path still leaves the session over
  budget (no-op otherwise).
- AC2: whole turns dropped — never split a turn; never the system prompt
  or the first user prompt (and never the summary marker).
- AC3: every backstop event reports turns dropped and remaining budget.
- AC4: unit tests for backstop boundaries, turn-split prohibition, and
  first-prompt protection.

Chains on top of ``plan_session_compaction`` (core child): when that plan
resolves to ``compacted_over_budget``, the dispatcher (integration child)
passes the compacted list here to truncate; ``backstop=True`` on the plan
chains automatically.
"""
import pytest
from proxy.compaction import (
    _SUMMARY_MARKER,
    _SUMMARY_MARKER_END,
    plan_session_compaction,
    truncate_backstop,
)

_EST_PER_MESSAGE = 1000


def counting_estimator(messages) -> int:
    return _EST_PER_MESSAGE * len(messages) + sum(
        len(str(m.get("content", ""))) for m in messages
    )


def fast_config() -> dict:
    return {
        "server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
            "compaction_trigger_ratio": 0.70,
        }
    }


def _summary_content(text: str = "SUMMARY") -> str:
    return f"{_SUMMARY_MARKER}{text}{_SUMMARY_MARKER_END}"


def over_budget_compacted(num_recent_turns: int = 40) -> list[dict]:
    """A compacted-shaped list whose estimate exceeds the fast target
    (38,000 tokens @ 1000/msg): protected front + ``num_recent_turns``
    whole (user, assistant) turns."""
    msgs = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "FIRST_USER"},
        {"role": "user", "content": _summary_content()},
    ]
    for i in range(num_recent_turns):
        msgs.append({"role": "user", "content": f"q_{i}"})
        msgs.append({"role": "assistant", "content": f"a_{i}"})
    return msgs


def fixed_summarizer(middle_messages) -> str:
    return f"MIDDLE_SUMMARY: folded {len(middle_messages)} messages."


# ===================================================================
# AC1 — trigger only when over budget
# ===================================================================


class TestTriggerBoundary:
    def test_noop_when_within_target(self):
        msgs = over_budget_compacted(5)  # 13 msgs ≈ 13K ≤ 38K
        result = truncate_backstop(
            msgs, 38000, estimate_tokens=counting_estimator
        )
        assert result["action"] == "noop"
        assert result["dropped_turns"] == 0
        assert result["messages"] is msgs  # untouched

    def test_noop_at_exact_target(self):
        msgs = over_budget_compacted(5)
        est = counting_estimator(msgs)
        result = truncate_backstop(msgs, est, estimate_tokens=counting_estimator)
        assert result["action"] == "noop"
        assert result["remaining_budget"] == 0


# ===================================================================
# AC2 — whole turns only; protected front intact
# ===================================================================


class TestWholeTurnDropping:
    def test_drops_oldest_whole_turns_until_budget(self):
        msgs = over_budget_compacted(40)  # 83 msgs ≈ 84K > 38K
        result = truncate_backstop(msgs, 38000, estimate_tokens=counting_estimator)
        assert result["action"] == "dropped"
        assert counting_estimator(result["messages"]) <= 38000
        # 3 protected msgs + k whole turns ≤ 38 msgs → 17 turns kept,
        # 23 dropped (oldest first).
        assert result["dropped_turns"] == 23
        assert result["messages"][0]["content"] == "SYSTEM"
        assert result["messages"][1]["content"] == "FIRST_USER"

    def test_never_drops_system_first_or_summary(self):
        msgs = over_budget_compacted(40)
        result = truncate_backstop(msgs, 38000, estimate_tokens=counting_estimator)
        out = result["messages"]
        # Protected front (system, first user, summary marker) is intact and
        # still at the head of the list, in order.
        assert out[0]["role"] == "system"
        assert out[1]["role"] == "user"
        assert out[1]["content"] == "FIRST_USER"
        assert out[2]["content"].startswith(_SUMMARY_MARKER)
        assert any(
            m.get("role") == "system" for m in out
        )

    def test_never_splits_a_turn(self):
        msgs = over_budget_compacted(40)
        result = truncate_backstop(msgs, 38000, estimate_tokens=counting_estimator)
        out = result["messages"]
        assert result["dropped_turns"] == 23
        # Later turns (q_23..q_39 / a_23..a_39) remain as whole alternating
        # pairs — no orphaned half of a turn.
        region = out[3:]
        assert len(region) == 17 * 2
        for i in range(0, len(region), 2):
            assert region[i]["role"] == "user"
            assert region[i + 1]["role"] == "assistant"
        assert region[0]["content"] == "q_23"
        assert region[-1]["content"] == "a_39"

    def test_oldest_recent_dropped_newest_kept(self):
        msgs = over_budget_compacted(40)
        result = truncate_backstop(msgs, 38000, estimate_tokens=counting_estimator)
        contents = [m["content"] for m in result["messages"]]
        # The oldest surviving turn is the newest-kept boundary; all dropped
        # old turns are gone.
        assert "q_0" not in contents
        assert "a_0" not in contents
        assert "q_22" not in contents  # 23rd dropped is q_22..a_22
        assert "q_23" in contents


# ===================================================================
# AC3 — event reporting (turns dropped, remaining budget)
# ===================================================================


class TestEventReporting:
    def test_reports_turns_dropped_and_remaining_budget(self):
        msgs = over_budget_compacted(40)
        result = truncate_backstop(msgs, 38000, estimate_tokens=counting_estimator)
        assert result["dropped_turns"] == 23
        assert result["dropped_messages"] == 46
        assert result["estimated_before"] > 38000
        assert result["estimated_after"] <= 38000
        assert result["remaining_budget"] == 38000 - result["estimated_after"]
        assert result["remaining_budget"] >= 0

    def test_exhausted_reports_no_path_to_budget(self):
        # Protected front alone exceeds the budget — nothing droppable left.
        msgs = [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "FIRST_USER"},
            {"role": "user", "content": _summary_content("Z" * 50000)},
        ]
        result = truncate_backstop(msgs, 38000, estimate_tokens=counting_estimator)
        assert result["action"] == "exhausted"
        assert result["dropped_turns"] == 0
        assert result["remaining_budget"] < 0


# ===================================================================
# AC4 + chaining with the summary path
# ===================================================================


class TestChaining:
    def test_plan_backstop_flag_chains_when_summary_over_budget(self):
        # Summarizer output larger than the selection placeholder blows the
        # budget after selection → backstop drops the oldest recent turns.
        def big_summarizer(middle_messages):
            return "B" * 20000

        result = plan_session_compaction(
            make_session(60), fast_config(), "fast",
            summarizer=big_summarizer,
            estimate_tokens=counting_estimator,
            backstop=True,
        )
        assert result["action"] == "compact"
        assert result["reason"] == "backstop_dropped"
        assert result["backstop_dropped_turns"] > 0
        assert result["estimated_after"] <= 38000

    def test_plan_backstop_off_keeps_over_budget_reason(self):
        # Default plan (backstop off) still reports compacted_over_budget.
        def big_summarizer(middle_messages):
            return "B" * 20000

        result = plan_session_compaction(
            make_session(60), fast_config(), "fast",
            summarizer=big_summarizer,
            estimate_tokens=counting_estimator,
            backstop=False,
        )
        assert result["reason"] == "compacted_over_budget"

    def test_plan_backstop_exhausted_when_retention_unavoidable(self):
        # Pathological retention — even dropping every recent turn cannot
        # reach the target; the plan reports backstop_exhausted so the
        # dispatcher can route remote with guidance.
        messages = make_session(10)
        messages[0]["content"] = "HUGE_SYSTEM_" + "x" * 250000

        def fixed(middle_messages):
            return "S"

        result = plan_session_compaction(
            messages, fast_config(), "fast", summarizer=fixed,
            estimate_tokens=counting_estimator, backstop=True,
        )
        assert result["reason"] == "backstop_exhausted"

    def test_backstop_flag_never_fires_within_target(self):
        result = plan_session_compaction(
            make_session(5), fast_config(), "fast",
            summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
            backstop=True,
        )
        assert result["action"] == "noop"
        assert result["reason"] == "below_trigger"


def make_session(num_turns: int) -> list[dict]:
    msgs = [{"role": "system", "content": "SYSTEM_PROMPT"}]
    msgs.append({"role": "user", "content": "FIRST_USER_PROMPT"})
    msgs.append({"role": "assistant", "content": "FIRST_ASSISTANT_REPLY"})
    for i in range(1, num_turns + 1):
        msgs.append({"role": "user", "content": f"user_q_{i}"})
        msgs.append({"role": "assistant", "content": f"assistant_a_{i}"})
    return msgs
