"""
Integration & end-to-end verification.

Feature: LP-0MTGBQ01A000ZFT9 (final child of LP-0MTCWE8NG003P0SD).

Verifies the complete compaction flow end-to-end across fast/cheap modes,
the retention invariant, and all fallback paths:

- AC1 fast mode full flow: trigger → summarize → dispatch (≤ 38K)
- AC2 cheap mode full flow incl. non-compactable routing
- AC3 dry-run mode: zero behavior change
- AC4 retention invariant across all modes
- AC5 parent ACs verified here
- AC6 full suite green (checked by the finish gate, not this file)
"""
import pytest
from proxy.compaction import (
    CompactionChurnCollector,
    decide_session_compaction,
    estimate_session_tokens,
)

_EST_PER_MESSAGE = 1000


def counting_estimator(messages) -> int:
    return _EST_PER_MESSAGE * len(messages) + sum(
        len(str(m.get("content", ""))) for m in messages
    )


def raw_config(ctx_size: int, slots: int, **extra) -> dict:
    cfg = {
        "server": {
            "local_model_ctx_size": ctx_size,
            "session_slot_pool_size": slots,
            "compaction_trigger_ratio": 0.70,
        }
    }
    cfg["server"].update(extra)
    return cfg


FAST = raw_config(262144, 3)          # per-slot 83,285 → trigger 58,300
CHEAP = raw_config(131072, 2)         # per-slot 61,440 → trigger 43,008
FAST_DRY = raw_config(262144, 3, compaction_dry_run=True)
CHEAP_DRY = raw_config(131072, 2, compaction_dry_run=True)


def fixed_summarizer(middle_messages) -> str:
    return f"MIDDLE_SUMMARY: folded {len(middle_messages)} messages."


def make_session(num_turns: int) -> list[dict]:
    msgs = [{"role": "system", "content": "SYSTEM_PROMPT"}]
    msgs.append({"role": "user", "content": "FIRST_USER_PROMPT"})
    msgs.append({"role": "assistant", "content": "FIRST_ASSISTANT_REPLY"})
    for i in range(1, num_turns + 1):
        msgs.append({"role": "user", "content": f"user_q_{i}"})
        msgs.append({"role": "assistant", "content": f"assistant_a_{i}"})
    return msgs


# ===================================================================
# AC1 — fast mode: trigger → summarize → dispatch
# ===================================================================


class TestFastFlow:
    def test_trigger_fires_and_dispatch_gets_compacted_history(self):
        messages = make_session(60)  # est ~124K > 58,300 trigger
        decision = decide_session_compaction(
            messages, FAST, "fast",
            summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
            session_id="sess-fast-e2e",
        )
        assert decision["action"] == "compact"
        assert decision["applied"] is True
        assert decision["dry_run"] is False
        assert decision["estimated_before"] > decision["trigger_tokens"]
        assert decision["estimated_after"] <= decision["target_tokens"]  # ≤ 38K
        assert decision["estimated_after"] <= 38000

    def test_dispatch_history_has_summary_below_first_prompt(self):
        decision = decide_session_compaction(
            make_session(60), FAST, "fast",
            summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
            session_id="sess-fast-e2e",
        )
        dispatch_messages = decision["messages"]
        # Layout: [system, first_user (verbatim), summary marker, recent...]
        assert dispatch_messages[0]["role"] == "system"
        assert dispatch_messages[1]["role"] == "user"
        assert dispatch_messages[1]["content"] == "FIRST_USER_PROMPT"
        summary = dispatch_messages[2]
        assert summary["role"] == "user"
        assert "MIDDLE_SUMMARY" in summary["content"]
        assert "The conversation history before this point was compacted" in (
            summary["content"]
        )
        assert summary["content"].endswith("</summary>")

    def test_below_trigger_never_compacts(self):
        messages = make_session(3)
        decision = decide_session_compaction(
            messages, FAST, "fast",
            summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
            session_id="sess-fast-e2e",
        )
        assert decision["action"] == "noop"
        assert decision["applied"] is False
        assert decision["messages"] is messages  # untouched list
        # No event log noise when below trigger (caplog verified separately).


# ===================================================================
# AC2 — cheap mode: trigger → summarize → dispatch (+ non-compactable)
# ===================================================================


class TestCheapFlow:
    def test_trigger_fires_and_dispatch_gets_compacted_history(self):
        messages = make_session(60)  # est ~124K > 43,008 trigger
        decision = decide_session_compaction(
            messages, CHEAP, "cheap",
            summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
            session_id="sess-cheap-e2e",
        )
        assert decision["action"] == "compact"
        assert decision["applied"] is True
        assert decision["estimated_after"] <= decision["target_tokens"]  # ≤ 30K
        assert decision["estimated_after"] <= 30000

    def test_non_compactable_routes_remote_with_guidance(self):
        # Summarizer unavailable → session cannot be compacted → the
        # dispatcher must route remote WITH guidance, never dispatch local
        # near-full-slot (and never silently).
        messages = make_session(60)
        decision = decide_session_compaction(
            messages, CHEAP, "cheap",
            summarizer=None,  # unavailable
            estimate_tokens=counting_estimator,
            session_id="sess-cheap-e2e",
        )
        assert decision["action"] == "remote_with_guidance"
        assert decision["reason"] == "summarizer_unavailable"
        assert decision["applied"] is False  # never safe to dispatch local
        assert decision["messages"] is messages  # untouched


# ===================================================================
# AC3 — dry-run mode: zero behavior change
# ===================================================================


class TestDryRunEndToEnd:
    def test_zero_dispatch_change_identity(self):
        messages = make_session(60)
        decision = decide_session_compaction(
            messages, FAST_DRY, "fast",
            summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
            session_id="sess-dry-e2e",
        )
        assert decision["dry_run"] is True
        assert decision["applied"] is False
        # The exact same list object is used for dispatch — nothing swapped.
        assert decision["messages"] is messages

    def test_advisory_log_with_dry_run_flag(self, caplog):
        with caplog.at_level("INFO", logger="proxy.compaction"):
            decide_session_compaction(
                make_session(60), FAST_DRY, "fast",
                summarizer=fixed_summarizer,
                estimate_tokens=counting_estimator,
                session_id="sess-dry-e2e",
            )
        rec = next(r for r in caplog.records if r.name == "proxy.compaction")
        assert rec.getMessage().startswith("compaction_event")
        assert "dry_run=True" in rec.getMessage()

    def test_dry_run_records_churn(self):
        collector = CompactionChurnCollector()
        decide_session_compaction(
            make_session(60), FAST_DRY, "fast",
            summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
            session_id="sess-dry-e2e",
            churn_collector=collector,
        )
        assert collector.churn_counts() == {"sess-dry-e2e": 1}

    def test_dry_run_noop_emits_no_event_and_no_churn(self, caplog):
        collector = CompactionChurnCollector()
        messages = make_session(3)
        with caplog.at_level("INFO", logger="proxy.compaction"):
            decision = decide_session_compaction(
                messages, FAST_DRY, "fast",
                summarizer=fixed_summarizer,
                estimate_tokens=counting_estimator,
                session_id="sess-dry-e2e",
                churn_collector=collector,
            )
        assert decision["action"] == "noop"
        assert decision["messages"] is messages
        assert not [r for r in caplog.records if r.name == "proxy.compaction"]
        assert collector.churn_counts() == {}


# ===================================================================
# AC4 — retention invariant across all modes
# ===================================================================


class TestRetentionInvariant:
    def _retained(self, decision):
        return decision["messages"][0], decision["messages"][1]

    def test_fast_keeps_system_and_first_prompt_verbatim(self):
        messages = make_session(60)
        system, first_user = messages[0], messages[1]
        decision = decide_session_compaction(
            messages, FAST, "fast",
            summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
            session_id="sess-ret-fast",
        )
        kept_system, kept_first = self._retained(decision)
        assert kept_system is system  # same dict object, never re-serialized
        assert kept_first is first_user
        assert kept_system == system
        assert kept_first == first_user

    def test_cheap_keeps_system_and_first_prompt_verbatim(self):
        messages = make_session(60)
        system, first_user = messages[0], messages[1]
        decision = decide_session_compaction(
            messages, CHEAP, "cheap",
            summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
            session_id="sess-ret-cheap",
        )
        kept_system, kept_first = self._retained(decision)
        assert kept_system is system
        assert kept_first is first_user

    def test_backstop_never_touches_system_or_first_prompt(self):
        messages = make_session(60)
        system, first_user = messages[0], messages[1]

        def big_summarizer(middle_messages):
            return "B" * 20000

        decision = decide_session_compaction(
            messages, FAST, "fast",
            summarizer=big_summarizer,
            estimate_tokens=counting_estimator,
            session_id="sess-ret-backstop",
        )
        assert decision["reason"] == "backstop_dropped"
        kept_system, kept_first = self._retained(decision)
        assert kept_system is system
        assert kept_first is first_user

    def test_dry_run_keeps_original_verbatim(self):
        messages = make_session(60)
        decision = decide_session_compaction(
            messages, FAST_DRY, "fast",
            summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
            session_id="sess-ret-dry",
        )
        assert decision["messages"] is messages
        assert [m for m in decision["messages"]] == messages


# ===================================================================
# AC2 — backstop whole-turn drops (never splits turns, never silent)
# ===================================================================


class TestBackstopFlow:
    def test_backstop_drops_whole_turns_only(self):
        decision = decide_session_compaction(
            make_session(60), FAST, "fast",
            summarizer=lambda m: "B" * 20000,
            estimate_tokens=counting_estimator,
            session_id="sess-backstop",
        )
        assert decision["reason"] == "backstop_dropped"
        assert decision["applied"] is True
        assert decision["backstop_dropped_turns"] > 0
        msgs = decision["messages"]
        # Protected front: system, first user prompt, summary marker.
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "FIRST_USER_PROMPT"
        assert msgs[2]["role"] == "user"
        assert msgs[2]["content"].startswith(
            "The conversation history before this point was compacted"
        )
        # Recent turns are whole: every user turn is followed by its reply.
        for i in range(3, len(msgs)):
            assert msgs[i]["role"] != "user" or (
                i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant"
            )
        assert msgs[-1]["role"] == "assistant"

    def test_backstop_exhausted_still_never_remote_silently(self):
        # Huge system prompt alone above target → nothing droppable;
        # dispatcher escalates (backstop_exhausted) rather than silently
        # dropping the protected prefix. The summary path output is kept
        # (never the original full history), still over budget.
        messages = make_session(10)
        messages[0]["content"] = "HUGE_SYSTEM_" + "x" * 250000
        decision = decide_session_compaction(
            messages, FAST, "fast",
            summarizer=fixed_summarizer,
            estimate_tokens=counting_estimator,
            session_id="sess-backstop",
        )
        assert decision["reason"] == "backstop_exhausted"
        msgs = decision["messages"]
        # Protected prefix intact + summary marker present.
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == messages[0]["content"]
        assert msgs[1] is messages[1]
        assert msgs[2]["content"].startswith(
            "The conversation history before this point was compacted"
        )
        # Still over budget → dispatcher must escalate, never dispatch
        # local near-full-slot.
        assert decision["estimated_after"] > decision["target_tokens"]
        assert decision["backstop_dropped_turns"] == 0


# ===================================================================
# AC5 — structured logging on all paths (parent AC5)
# ===================================================================


class TestLoggingFields:
    def test_compact_event_fields(self, caplog):
        with caplog.at_level("INFO", logger="proxy.compaction"):
            decide_session_compaction(
                make_session(60), FAST, "fast",
                summarizer=fixed_summarizer,
                estimate_tokens=counting_estimator,
                session_id="sess-logging-e2e",
            )
        rec = next(r for r in caplog.records if r.name == "proxy.compaction")
        line = rec.getMessage()
        assert line.startswith("compaction_event")
        for field in (
            "session=", "mode=", "action=", "reason=", "pre_tokens=",
            "post_tokens=", "turns_summarized=", "turns_dropped=",
            "summary_tokens=", "dry_run=",
        ):
            assert field in line

    def test_non_compactable_logged_at_warning(self, caplog):
        with caplog.at_level("INFO", logger="proxy.compaction"):
            decide_session_compaction(
                make_session(60), CHEAP, "cheap",
                summarizer=None,
                estimate_tokens=counting_estimator,
                session_id="sess-logging-e2e",
            )
        rec = next(r for r in caplog.records if r.name == "proxy.compaction")
        assert rec.levelno == 30  # WARNING — never silent
        assert "remote_with_guidance" in rec.getMessage()


# ===================================================================
# AC6 — churn stats end-to-end
# ===================================================================


class TestChurnEndToEnd:
    def test_churn_recorded_per_session(self):
        collector = CompactionChurnCollector()
        for i in range(4):
            decide_session_compaction(
                make_session(60), FAST_DRY, "fast",
                summarizer=fixed_summarizer,
                estimate_tokens=counting_estimator,
                session_id=f"sess-{i}",
                churn_collector=collector,
            )
        counts = collector.churn_counts()
        assert counts == {f"sess-{i}": 1 for i in range(4)}

    def test_churn_rate_target(self):
        collector = CompactionChurnCollector()
        for _ in range(3):
            decide_session_compaction(
                make_session(60), FAST_DRY, "fast",
                summarizer=fixed_summarizer,
                estimate_tokens=counting_estimator,
                session_id="sess-hot",
                churn_collector=collector,
            )
        report = collector.churn_report()
        assert report["sess-hot"]["rate_per_hour"] == 3.0
        assert report["sess-hot"]["exceeds_target"] is True


# ===================================================================
# Production seam — session-persistence wiring (router._update_session)
# ===================================================================


class FakeSessionManager:
    """Minimal stand-in for srv.session_manager used by the router seam."""

    def __init__(self):
        self.stored = {}
        self.update_calls = []

    async def update_messages(self, session_id, messages):
        self.stored[session_id] = list(messages)
        self.update_calls.append((session_id, list(messages)))

    async def get(self, session_id):
        return None


class FakeSrv:
    def __init__(self, config, manager=None):
        self.config = config
        self.session_manager = manager or FakeSessionManager()
        self.logger = __import__("logging").getLogger("test.fakesrv")


class TestDispatchSeam:
    def test_dry_run_seam_persists_original(self):
        from proxy.router_helpers import _evaluate_session_compaction

        srv = FakeSrv(FAST_DRY)
        messages = make_session(60)
        decision = _evaluate_session_compaction(
            srv, "sess-seam", messages, "fast",
            summarizer=fixed_summarizer, estimate_tokens=counting_estimator,
        )
        assert decision["applied"] is False
        assert decision["messages"] is messages  # persisted unchanged

    def test_live_seam_returns_compacted_for_persistence(self):
        from proxy.router_helpers import _evaluate_session_compaction

        srv = FakeSrv(FAST)
        messages = make_session(60)
        decision = _evaluate_session_compaction(
            srv, "sess-seam", messages, "fast",
            summarizer=fixed_summarizer, estimate_tokens=counting_estimator,
        )
        assert decision["applied"] is True
        assert decision["messages"] is not messages
        assert estimate_session_tokens(decision["messages"]) <= 38000
        # Retention: first stored message is the same system dict object.
        assert decision["messages"][0] is messages[0]
