"""
Warn-only dry-run mode.

Feature: LP-0MTGBPICV003JMXI (child of LP-0MTCWE8NG003P0SD).

Deploy phase: advisory logging only — zero dispatch behavior change — for
validation of the churn rate (< 1 compaction/session/hour).

- AC1: ``compaction_dry_run`` / ``compaction.dry_run`` config flag enables
  advisory mode; when enabled, the proxy logs what WOULD happen
  (would-summarize / would-drop) without changing dispatch.
- AC2: churn stats collected and logged (compactions per session per hour).
- AC3: unit tests verify dry-run produces no dispatch changes and emits
  advisory logs.
"""
import threading
import time

import pytest
from proxy.compaction import (
    CompactionChurnCollector,
    is_compaction_dry_run,
    log_compaction_event,
    plan_session_compaction,
    run_dry_run_plan,
    truncate_backstop,
)

_EST_PER_MESSAGE = 1000


def counting_estimator(messages) -> int:
    return _EST_PER_MESSAGE * len(messages) + sum(
        len(str(m.get("content", ""))) for m in messages
    )


def fast_config(**overrides) -> dict:
    cfg = {
        "server": {
            "local_model_ctx_size": 262144,
            "session_slot_pool_size": 3,
            "compaction_trigger_ratio": 0.70,
        }
    }
    if "dry_run" in overrides:
        cfg["server"]["compaction_dry_run"] = overrides["dry_run"]
    return cfg


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
# AC1 — dry-run flag resolution
# ===================================================================


class TestDryRunFlag:
    def test_defaults_to_false(self):
        assert is_compaction_dry_run({}) is False
        assert is_compaction_dry_run({"server": {}}) is False

    def test_flat_key_enables(self):
        assert is_compaction_dry_run({"server": {"compaction_dry_run": True}}) is True
        assert is_compaction_dry_run({"compaction_dry_run": True}) is True

    def test_nested_spec_key_enables(self):
        # Spec wording: `compaction.dry_run: true`.
        assert (
            is_compaction_dry_run({"server": {"compaction": {"dry_run": True}}})
            is True
        )

    def test_explicit_false_wins(self):
        assert (
            is_compaction_dry_run(
                {"server": {"compaction_dry_run": False}}
            )
            is False
        )
        assert (
            is_compaction_dry_run(
                {"server": {"compaction_dry_run": False, "compaction": {"dry_run": True}}}
            )
            is False
        )


# ===================================================================
# AC2 — churn stats collected and logged
# ===================================================================


class TestChurnCollector:
    def test_record_counts_events(self):
        clock = [1000.0]

        def now_fn():
            return clock[0]

        collector = CompactionChurnCollector(now_fn=now_fn)
        collector.record("session-a")
        collector.record("session-a")
        collector.record("session-b")
        assert collector.churn_counts(window_seconds=3600) == {
            "session-a": 2,
            "session-b": 1,
        }

    def test_expired_events_excluded(self):
        clock = [0.0]

        def now_fn():
            return clock[0]

        collector = CompactionChurnCollector(now_fn=now_fn)
        collector.record("session-a")  # at t=0
        clock[0] = 3610.0
        collector.record("session-a")  # at t=3610
        # 1-hour window from now=t3610 excludes the t=0 event.
        assert collector.churn_counts(window_seconds=3600) == {"session-a": 1}
        # 2-hour window includes both.
        assert collector.churn_counts(window_seconds=7200) == {"session-a": 2}

    def test_rate_per_hour_and_target(self):
        collector = CompactionChurnCollector()
        for _ in range(3):
            collector.record("session-a")
        # Under-target session (≤ 1/hour).
        collector.record("session-b")
        report = collector.churn_report(window_seconds=3600, target_rate=1.0)
        stats = report["session-a"]
        assert stats["count"] == 3
        assert stats["rate_per_hour"] == 3.0
        assert stats["exceeds_target"] is True
        assert report["session-b"]["rate_per_hour"] == 1.0
        assert report["session-b"]["exceeds_target"] is False

    def test_log_churn_report_emits_structured_lines(self, caplog):
        collector = CompactionChurnCollector()
        collector.record("sess-1234567890")
        with caplog.at_level("WARNING", logger="proxy.compaction"):
            report = collector.log_churn_report()
        assert report["sess-1234567890"]["count"] == 1
        assert any(
            r.name == "proxy.compaction" and "compaction_churn" in r.getMessage()
            for r in caplog.records
        )

    def test_thread_safe_recording(self):
        collector = CompactionChurnCollector()
        threads = []
        for _ in range(4):
            t = threading.Thread(
                target=lambda: [collector.record("shared") for _ in range(25)]
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert collector.churn_counts() == {"shared": 100}


# ===================================================================
# AC1/AC3 — dry-run advisories without dispatch change
# ===================================================================


class TestDryRunAdvisory:
    def test_dry_run_plans_but_never_mutates_input(self):
        messages = make_session(60)
        snapshot = [dict(m) for m in messages]
        result = run_dry_run_plan(
            messages, fast_config(), "fast",
            summarizer=fixed_summarizer, estimate_tokens=counting_estimator,
            session_id="sess-1234567890",
        )
        assert result["action"] != "noop"  # would-compact
        # Original objects untouched (dispatch unchanged).
        assert messages == snapshot
        assert result["messages"] is not messages  # never swapped in place

    def test_dry_run_logs_with_dry_run_flag(self, caplog):
        messages = make_session(60)
        with caplog.at_level("INFO", logger="proxy.compaction"):
            result = run_dry_run_plan(
                messages, fast_config(), "fast",
                summarizer=fixed_summarizer, estimate_tokens=counting_estimator,
                session_id="sess-1234567890",
            )
        rec = next(
            r for r in caplog.records if r.name == "proxy.compaction"
        )
        assert rec.getMessage().startswith("compaction_event")
        assert "dry_run=True" in rec.getMessage()
        assert result["action"] == "compact"

    def test_dry_run_surfaces_would_drop(self, caplog):
        # Backstop chain: dry-run shows would-drop even though nothing is
        # applied to the session.
        def big_summarizer(middle_messages):
            return "B" * 20000

        messages = make_session(60)
        result = run_dry_run_plan(
            messages, fast_config(), "fast",
            summarizer=big_summarizer, estimate_tokens=counting_estimator,
            session_id="sess-1234567890",
        )
        assert result["reason"] == "backstop_dropped"
        assert result["backstop_dropped_turns"] > 0

    def test_dry_run_below_trigger_logs_nothing(self, caplog):
        messages = make_session(3)  # below trigger
        with caplog.at_level("INFO", logger="proxy.compaction"):
            result = run_dry_run_plan(
                messages, fast_config(), "fast",
                summarizer=fixed_summarizer, estimate_tokens=counting_estimator,
                session_id="sess-1234567890",
            )
        assert result["action"] == "noop"
        # Dry-run log_compaction_event skips non-events (no advisory noise).
        assert not [
            r for r in caplog.records if r.name == "proxy.compaction"
        ]

    def test_compose_collector_with_dry_run(self, caplog):
        # End-to-end advisory loop: plan → log → churn record.
        collector = CompactionChurnCollector()
        messages = make_session(60)
        result = run_dry_run_plan(
            messages, fast_config(), "fast",
            summarizer=fixed_summarizer, estimate_tokens=counting_estimator,
            session_id="sess-1234567890",
        )
        collector.record("sess-1234567890")
        assert collector.churn_counts() == {"sess-1234567890": 1}
        assert result["action"] == "compact"
