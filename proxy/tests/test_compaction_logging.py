"""
Compaction event logging.

Feature: LP-0MTGBP8DX003R5ZO (child of LP-0MTCWE8NG003P0SD).

Every compaction event — summary path, backstop, dry-run advisory, remote
fallback — MUST emit one structured log entry (AC2: no silent drops) with
the full field set (AC1):

- Session ID (truncated per codebase convention)
- Pre-compaction estimated tokens
- Post-compaction estimated tokens
- Turns summarized
- Turns dropped (if backstop fired)
- Summary length in tokens
- Mode (fast/cheap)
- Dry-run flag (true/false)

``log_compaction_event`` maps a ``plan_session_compaction`` result (plus
session id / dry-run flag) onto that structured entry, emits it, and
returns the field dict so the dry-run child can collect churn stats from
the same data.
"""
import pytest
from proxy.compaction import (
    log_compaction_event,
    plan_session_compaction,
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


def cheap_config() -> dict:
    return {
        "server": {
            "local_model_ctx_size": 131072,
            "session_slot_pool_size": 2,
            "compaction_trigger_ratio": 0.70,
        }
    }


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


_REQUIRED_FIELDS = {
    "session",
    "mode",
    "action",
    "reason",
    "pre_tokens",
    "post_tokens",
    "turns_summarized",
    "turns_dropped",
    "summary_tokens",
    "dry_run",
}


def _emit(plan_result, *, session_id="sess-123456", logger_obj=None, **kw):
    return log_compaction_event(
        plan_result,
        session_id=session_id,
        logger_obj=logger_obj,
        estimate_tokens=counting_estimator,
        **kw,
    )


# ===================================================================
# AC1 — full structured field set on every compaction path
# ===================================================================


class TestFieldCompleteness:
    def test_compact_within_target_has_all_fields(self, caplog):
        result = plan_session_compaction(
            make_session(60), fast_config(), "fast",
            summarizer=fixed_summarizer, estimate_tokens=counting_estimator,
        )  # action=compact, reason=compacted_within_target
        with caplog.at_level("INFO", logger="proxy.compaction"):
            fields = _emit(result)
        assert fields is not None
        assert set(fields.keys()) == _REQUIRED_FIELDS
        assert fields["session"] == "sess-123"
        assert fields["mode"] == "fast"
        assert fields["action"] == "compact"
        assert fields["reason"] == "compacted_within_target"
        assert fields["pre_tokens"] == result["estimated_before"]
        assert fields["post_tokens"] == result["estimated_after"]
        assert fields["turns_summarized"] == result["turns_summarized"]
        assert fields["turns_dropped"] == 0
        assert fields["summary_tokens"] > 0
        assert fields["dry_run"] is False

    def test_backstop_dropped_has_turns_dropped(self, caplog):
        def big_summarizer(middle_messages):
            return "B" * 20000

        result = plan_session_compaction(
            make_session(60), fast_config(), "fast",
            summarizer=big_summarizer, estimate_tokens=counting_estimator,
            backstop=True,
        )  # reason=backstop_dropped
        with caplog.at_level("INFO", logger="proxy.compaction"):
            fields = _emit(result)
        assert fields["action"] == "compact"
        assert fields["reason"] == "backstop_dropped"
        assert fields["turns_dropped"] == result["backstop_dropped_turns"]
        assert fields["turns_dropped"] > 0
        assert fields["post_tokens"] <= 38000

    def test_remote_fallback_has_all_fields(self, caplog):
        result = plan_session_compaction(
            make_session(60), cheap_config(), "cheap",
            summarizer=None, estimate_tokens=counting_estimator,
        )  # action=remote_with_guidance
        with caplog.at_level("INFO", logger="proxy.compaction"):
            fields = _emit(result)
        assert fields["action"] == "remote_with_guidance"
        assert fields["reason"] == "summarizer_unavailable"
        assert fields["turns_summarized"] == 0
        assert fields["pre_tokens"] == fields["post_tokens"]

    def test_over_budget_summary_path_emitted(self, caplog):
        messages = make_session(10)
        messages[0]["content"] = "HUGE_SYSTEM_" + "x" * 250000
        result = plan_session_compaction(
            messages, fast_config(), "fast",
            summarizer=fixed_summarizer, estimate_tokens=counting_estimator,
        )  # reason=compacted_over_budget (backstop off)
        with caplog.at_level("INFO", logger="proxy.compaction"):
            fields = _emit(result)
        assert fields["reason"] == "compacted_over_budget"
        assert fields["post_tokens"] > 38000


# ===================================================================
# AC2 — no silent drops
# ===================================================================


class TestNoSilentDrops:
    def test_log_emitted_for_every_compaction_path(self, caplog):
        paths = []
        # Summary path (fast).
        paths.append(
            plan_session_compaction(
                make_session(60), fast_config(), "fast",
                summarizer=fixed_summarizer, estimate_tokens=counting_estimator,
            )
        )
        # Backstop path.
        def big_summarizer(middle_messages):
            return "B" * 20000

        paths.append(
            plan_session_compaction(
                make_session(60), fast_config(), "fast",
                summarizer=big_summarizer, estimate_tokens=counting_estimator,
                backstop=True,
            )
        )
        # Remote fallback (non-compactable).
        paths.append(
            plan_session_compaction(
                make_session(60), cheap_config(), "cheap",
                summarizer=None, estimate_tokens=counting_estimator,
            )
        )
        with caplog.at_level("INFO", logger="proxy.compaction"):
            for result in paths:
                _emit(result)
        emitted = [
            r.getMessage() for r in caplog.records if r.name == "proxy.compaction"
        ]
        assert len([e for e in emitted if e.startswith("compaction_event")]) == 3

    def test_noop_is_not_an_event(self, caplog):
        result = plan_session_compaction(
            make_session(3), fast_config(), "fast",
            summarizer=fixed_summarizer, estimate_tokens=counting_estimator,
        )  # below_trigger
        with caplog.at_level("INFO", logger="proxy.compaction"):
            fields = _emit(result)
        assert fields is None
        assert not [
            r for r in caplog.records if r.name == "proxy.compaction"
        ]


# ===================================================================
# Dry-run flag + summary token accounting
# ===================================================================


class TestDryRunAndSummaryTokens:
    def test_dry_run_flag_true(self, caplog):
        result = plan_session_compaction(
            make_session(60), fast_config(), "fast",
            summarizer=fixed_summarizer, estimate_tokens=counting_estimator,
        )
        with caplog.at_level("INFO", logger="proxy.compaction"):
            fields = _emit(result, dry_run=True)
        assert fields["dry_run"] is True

    def test_summary_tokens_computed_from_estimator(self):
        result = plan_session_compaction(
            make_session(60), fast_config(), "fast",
            summarizer=fixed_summarizer, estimate_tokens=counting_estimator,
        )
        fields = _emit(result)
        # summary_tokens = est of a single summary message.
        est = counting_estimator(
            [{"role": "user", "content": result["summary_text"]}]
        )
        assert fields["summary_tokens"] == est

    def test_no_summary_means_zero_summary_tokens(self):
        result = plan_session_compaction(
            make_session(60), cheap_config(), "cheap",
            summarizer=None, estimate_tokens=counting_estimator,
        )  # remote_with_guidance — no summary produced
        fields = _emit(result)
        assert fields["summary_tokens"] == 0

    def test_returns_fields_for_dry_run_churn(self):
        result = plan_session_compaction(
            make_session(60), fast_config(), "fast",
            summarizer=fixed_summarizer, estimate_tokens=counting_estimator,
        )
        fields = _emit(result, dry_run=True)
        # Churn metrics derivable from the returned dict.
        assert fields["pre_tokens"] - fields["post_tokens"] > 0
        assert fields["turns_summarized"] > 0


class TestLevelSelection:
    def test_dropped_and_exhausted_logged_at_warning(self, caplog):
        def big_summarizer(middle_messages):
            return "B" * 20000

        result = plan_session_compaction(
            make_session(60), fast_config(), "fast",
            summarizer=big_summarizer, estimate_tokens=counting_estimator,
            backstop=True,
        )
        with caplog.at_level("INFO", logger="proxy.compaction"):
            _emit(result)
        rec = next(
            r for r in caplog.records if r.name == "proxy.compaction"
        )
        assert rec.levelno == 30  # WARNING

    def test_tidy_compaction_logged_at_info(self, caplog):
        result = plan_session_compaction(
            make_session(60), fast_config(), "fast",
            summarizer=fixed_summarizer, estimate_tokens=counting_estimator,
        )
        with caplog.at_level("INFO", logger="proxy.compaction"):
            _emit(result)
        rec = next(
            r for r in caplog.records if r.name == "proxy.compaction"
        )
        assert rec.levelno == 20  # INFO
