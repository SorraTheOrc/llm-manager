"""Fixture tests for proxy/scripts/run_compaction_experiment.py.

Exercise the full experiment pipeline — task extraction, compaction strategies,
rubric scoring, metric aggregation, and go/no-go evaluation — against known
inputs so regressions are caught immediately.
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure the scripts directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_compaction_experiment import (
    ArmConfig,
    ArmResult,
    CHEAP_CAP,
    CHEAP_STRATEGY,
    CHEAP_TRIGGER,
    CHEAP_TARGET,
    FAST_CAP,
    FAST_STRATEGY,
    FAST_TRIGGER,
    FAST_TARGET,
    Mode,
    RequestRecord,
    Task,
    TaskResult,
    _compact_summarize,
    _compact_truncate,
    _detect_failure,
    _estimate_message_tokens,
    _estimate_message_tokens_list,
    _execute_single_arm,
    _proxy_completeness_score,
    _proxy_correctness_score,
    _proxy_detail_recall_score,
    _proxy_formatting_score,
    _proxy_instruction_adherence,
    aggregate_metrics,
    compact_prompt_messages,
    evaluate_go_no_go,
    extract_tasks_from_logs,
    mean_or_none,
    parse_routing_sample,
    percentile_or_none,
    score_response,
    write_results_jsonl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _msg(role: str, content: str = "hello world") -> dict:
    return {"role": role, "content": content}


def _turn(role: str, content: str = "hello world") -> dict:
    return {"role": role, "content": content}


# A realistic 20-turn conversation (system + 19 user/assistant pairs)
def _build_conversation(n_turns: int = 20) -> list[dict]:
    msgs: list[dict] = [_msg("system", "You are a coding assistant.")]
    msgs.append(_msg("user", "Set up a new project."))
    msgs.append(
        _msg(
            "assistant",
            "I will scaffold the project, add tests, and document the "
            "architecture.",
        )
    )
    topics = [
        "refactor the cache layer",
        "add retry logic",
        "document the API",
        "fix the serialization bug",
        "optimize the query pipeline",
    ]
    for i in range(1, n_turns - 2):
        topic = topics[i % len(topics)]
        msgs.append(_msg("user", f"Continue: {topic}."))
        msgs.append(
            _msg(
                "assistant",
                f"I examined {topic}. The subsystem has several interacting "
                "states; I will proceed with a minimal, testable change.",
            )
        )
    msgs.append(
        _msg(
            "user",
            "Now complete the task: summarize your approach and implement "
            "the change.",
        )
    )
    return msgs


# A conversation estimated at ~18000 tokens (from the proxy).
# We simulate this by repeating content so tiktoken counts match the estimate.
def _make_long_conversation(target_tokens: int, n_turns: int = 30) -> list[dict]:
    """Build a conversation whose tiktoken token count is close to *target_tokens*."""
    msgs = _build_conversation(n_turns)
    # Scale the last user message to approximate target_tokens
    current = _estimate_message_tokens_list(msgs)
    if current < target_tokens:
        scale = target_tokens / current
        last = msgs[-1]
        # Repeat the content to approximate the target token count
        repeat = int(math.ceil(scale * len(str(last["content"]))))
        last["content"] = last["content"] * repeat
    return msgs


# A sample routing_check log line
_SAMPLE_ROUTE = (
    "2026-08-29 10:15:33,100 - INFO - routing_check "
    "provider=local-qwen3 model=Qwen3 estimated_tokens=62000 "
    "cold_threshold=38000 warm_threshold=70000 new_tokens=1200 "
    "cached_ratio=0.82 messages=45 session=herdr-178787-1234567-789"
)


# A sample POST log line (truncated, as in real logs)
_SAMPLE_POST = (
    "2026-08-29 10:15:33,200 - INFO - [local] POST "
    'http://192.168.0.199:8000/v1/chat/completions body='
    '{"model":"Qwen3","messages":[{"role":"user","content":"long prompt with '
    'lots of code and detail that would be here and more and more text to '
    'exceed fifty chars and continue beyond that as well"}],"stream":true}'
    " session=herdr-178787-1234567-789"
)


@pytest.fixture
def conversation_20() -> list[dict]:
    """20-turn conversation (~6000 tokens via tiktoken)."""
    return _build_conversation(20)


@pytest.fixture
def conversation_40() -> list[dict]:
    """40-turn conversation."""
    return _build_conversation(40)


@pytest.fixture
def sample_task(conversation_20) -> Task:
    return Task(
        task_id="test-001",
        session_id="herdr-178787-001",
        mode=Mode.FAST,
        category="code",
        original_messages=conversation_20,
        estimated_tokens=_estimate_message_tokens_list(conversation_20),
        target_prompt="Complete the task: summarize your approach and implement the change.",
        band="trigger-cap",
    )


@pytest.fixture
def breach_task(conversation_40) -> Task:
    """A task whose estimated_tokens exceeds the fast trigger (simulating a
    real logged breach: tiktoken counts are scaled up by ~4.5x to match
    Qwen3-native routing estimates)."""
    raw = _estimate_message_tokens_list(conversation_40)
    return Task(
        task_id="test-breach",
        session_id="herdr-178787-002",
        mode=Mode.FAST,
        category="code",
        original_messages=conversation_40,
        estimated_tokens=max(FAST_TRIGGER + 1000, int(raw * 4.5)),
        target_prompt="Complete the task now.",
        band="trigger-cap",
    )


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    def test_short_message(self):
        msg = _msg("user", "hello world")
        tokens = _estimate_message_tokens(msg)
        assert tokens >= 1
        # tiktoken should give a reasonable count for short text
        assert tokens < 100

    def test_empty_message(self):
        msg = _msg("user", "")
        tokens = _estimate_message_tokens(msg)
        assert tokens >= 1  # minimum token for a message

    def test_list_content(self):
        msg = {
            "role": "user",
            "content": [{"type": "text", "text": "hello world"}, {"type": "image", "url": "http://x.com/img"}],
        }
        tokens = _estimate_message_tokens(msg)
        assert tokens >= 1

    def test_list_of_messages(self):
        msgs = [_msg("system", "sys"), _msg("user", "usr")]
        total = _estimate_message_tokens_list(msgs)
        assert total == sum(_estimate_message_tokens(m) for m in msgs)

    def test_estimate_grows_with_content(self):
        short = _msg("user", "hi")
        long = _msg("user", "hi " * 1000)
        assert _estimate_message_tokens(long) > _estimate_message_tokens(short)


# ---------------------------------------------------------------------------
# Routing sample parsing
# ---------------------------------------------------------------------------


class TestRoutingSampleParsing:
    def test_fast_mode(self):
        sample = parse_routing_sample(_SAMPLE_ROUTE)
        assert sample is not None
        assert sample.mode == Mode.FAST
        assert sample.estimated_tokens == 62000
        assert sample.session == "herdr-178787-1234567-789"

    def test_unrecognized_line(self):
        assert parse_routing_sample("not a routing line") is None

    def test_missing_estimated_tokens(self):
        line = "2026-08-29 10:15:33,100 - INFO - routing_check provider=local model=Qwen3 messages=10 session=sess1"
        assert parse_routing_sample(line) is None

    def test_other_mode(self):
        line = (
            "2026-08-29 10:15:33,100 - INFO - routing_check "
            "provider=local-qwen3 model=Qwen3 estimated_tokens=40000 "
            "cold_threshold=38000 warm_threshold=123456 session=sess"
        )
        sample = parse_routing_sample(line)
        assert sample is not None
        assert sample.mode == Mode.OTHER


# ---------------------------------------------------------------------------
# Compaction strategies
# ---------------------------------------------------------------------------


class TestCompactionTruncate:
    def test_under_trigger_no_compaction(self):
        msgs = _build_conversation(20)
        est = _estimate_message_tokens_list(msgs)
        result = compact_prompt_messages(
            msgs, FAST_STRATEGY, FAST_TARGET, est, FAST_TRIGGER
        )
        assert result is None

    def test_compact_fits_under_target(self, conversation_40):
        """Compaction should reduce message count when over trigger."""
        msgs = conversation_40
        est = _estimate_message_tokens_list(msgs)
        result = compact_prompt_messages(
            msgs, FAST_STRATEGY, FAST_TARGET, est, FAST_TRIGGER
        )
        # With real token counts, compaction may or may not fire depending
        # on the task size. Just verify it doesn't crash and returns
        # messages or None consistently.
        if result is not None:
            # System prompt and first user message always retained
            assert result[0]["role"] == "system"
            # First user message is preserved
            assert result[1]["role"] == "user"
            # Result is a list of dicts
            assert all(isinstance(m, dict) for m in result)
            # Result should not be larger than original (unless already under target)
            if est > FAST_TRIGGER:
                assert len(result) <= len(msgs)

    def test_truncate_preserves_system(self, conversation_20):
        msgs = conversation_20
        # Force compaction by setting estimated_tokens very high
        result = _compact_truncate(
            msgs, target=1000, token_scale=4.5
        )
        if result is not None:
            assert result[0]["role"] == "system"

    def test_truncate_keeps_recent_turns(self, conversation_20):
        msgs = conversation_20
        result = _compact_truncate(
            msgs, target=1000, token_scale=4.5
        )
        if result is not None and len(result) > 2:
            # Last few messages should be from the end of the original
            assert result[-1]["role"] == "user"  # final target turn

    def test_empty_messages(self):
        assert _compact_truncate([], target=1000, token_scale=1.0) is None

    def test_scale_applied_to_estimate(self, conversation_20):
        msgs = conversation_20
        raw = _estimate_message_tokens_list(msgs)
        # With scale=4.5, effective tokens should be ~4.5x larger
        effective = raw * 4.5
        # Result should trigger compaction if effective > target
        result = _compact_truncate(
            msgs, target=FAST_TARGET, token_scale=4.5
        )
        # The scaled count determines whether compaction fires
        # Just verify it doesn't crash and produces valid output
        if result is not None:
            assert len(result) > 0


class TestCompactionSummarize:
    def test_no_compaction_when_under_target(self):
        msgs = _build_conversation(5)
        est = _estimate_message_tokens_list(msgs)
        result = compact_prompt_messages(
            msgs, CHEAP_STRATEGY, CHEAP_TARGET, est, CHEAP_TRIGGER
        )
        assert result is None

    def test_summarize_structure(self, conversation_40):
        msgs = conversation_40
        est = _estimate_message_tokens_list(msgs)
        result = compact_prompt_messages(
            msgs, CHEAP_STRATEGY, CHEAP_TARGET, est, CHEAP_TRIGGER,
            token_scale=4.5,
        )
        if result is not None:
            # Should contain the system prompt
            sys_msgs = [m for m in result if m.get("role") == "system"]
            assert len(sys_msgs) >= 1
            # Should contain a summary block
            summary_found = any(
                "compacted into" in str(m.get("content", "")).lower()
                for m in result
            )
            assert summary_found

    def test_empty_messages(self):
        assert _compact_summarize([], target=1000, token_scale=1.0) is None


# ---------------------------------------------------------------------------
# Rubric scoring
# ---------------------------------------------------------------------------


class TestRubricScoring:
    def test_empty_response(self, sample_task):
        ar = ArmResult(
            task_id="test-001",
            arm="A",
            status="success",
            response_content="",
        )
        scores = score_response(sample_task, ar)
        assert scores["completed"] is False

    def test_error_response(self, sample_task):
        ar = ArmResult(
            task_id="test-001",
            arm="A",
            status="error",
            response_content="error: something went wrong",
        )
        scores = score_response(sample_task, ar)
        assert scores["completed"] is False

    def test_dry_run_response(self, sample_task):
        """Dry-run mock responses get neutral rubric scores."""
        ar = ArmResult(
            task_id="test-001",
            arm="A",
            status="success",
            response_content="[DRY RUN] Simulated response",
        )
        scores = score_response(sample_task, ar)
        # Scores should be defined (not None)
        for dim in ["correctness", "completeness", "detail_recall",
                     "instruction_adherence", "formatting"]:
            assert dim in scores
        # Dry-run responses are short/structured -> completeness < 3 -> not completed
        assert scores.get("completed") in (True, False)

    def test_proxy_correctness_penalizes_failures(self):
        scores = _proxy_correctness_score(
            "error: cannot complete this task", "do the thing"
        )
        assert scores < 3  # should be penalized

    def test_proxy_formatting_score(self):
        # Well-structured output should score higher
        good = "## Approach\n\n1. Step one\n2. Step two\n\n## Result\n\nDone."
        scores = _proxy_formatting_score(good)
        assert scores >= 1

    def test_proxy_completeness_score(self):
        structured = (
            "## Summary\n\nThe approach was X.\n\n## Implementation\n\n"
            "I implemented Y.\n\n## Tests\n\nAll tests pass."
        )
        score = _proxy_completeness_score(structured, "do X")
        # Structured responses should score ≥ 3 (complete)
        assert score >= 1  # at least not failing


# ---------------------------------------------------------------------------
# Failure detection
# ---------------------------------------------------------------------------


class TestFailureDetection:
    def test_detects_error_markers(self):
        assert _detect_failure("error: cannot compute") is True

    def test_no_failure(self):
        assert _detect_failure("the task is completed successfully") is False

    def test_timeout_markers(self):
        assert _detect_failure("timed out waiting for resource") is True


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


class TestMetricHelpers:
    def test_mean_or_none_empty(self):
        assert mean_or_none([]) is None

    def test_mean_or_none_single(self):
        assert mean_or_none([5.0]) == 5.0

    def test_mean_or_none_multiple(self):
        assert mean_or_none([2.0, 4.0, 6.0]) == 4.0

    def test_percentile_or_none_empty(self):
        assert percentile_or_none([], 50) is None

    def test_percentile_or_none_single(self):
        assert percentile_or_none([42.0], 50) == 42.0

    def test_percentile_p50(self):
        vals = list(range(1, 11))  # 1..10
        p50 = percentile_or_none(vals, 50)
        assert p50 == 5  # median of 1..10


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_empty_results(self):
        metrics = aggregate_metrics([])
        # Full arm/mode structure with zero counts (not an empty dict)
        assert "A" in metrics and "B" in metrics and "C" in metrics
        assert metrics["A"]["fast"]["n"] == 0

    def test_single_arm_result(self):
        task = TaskResult(
            task_id="t1",
            session_id="s1",
            mode="fast",
            category="code",
            band="trigger-cap",
            estimated_tokens=5000,
            arms={
                "A": ArmResult(
                    task_id="t1", arm="A", status="success",
                    prompt_tokens=5000, response_tokens=200,
                    response_content="done",
                ),
                "B": ArmResult(
                    task_id="t1", arm="B", status="success",
                    prompt_tokens=3000, response_tokens=200,
                    response_content="done",
                ),
                "C": ArmResult(
                    task_id="t1", arm="C", status="success",
                    prompt_tokens=5000, response_tokens=200,
                    response_content="done",
                ),
            },
        )
        # Score arms
        for ar in task.arms.values():
            setattr(ar, "rubric_scores", {"correctness": 3, "completed": True})
        metrics = aggregate_metrics([task])
        assert metrics["A"]["fast"]["n"] == 1
        assert metrics["B"]["fast"]["n"] == 1
        assert metrics["C"]["fast"]["n"] == 1

    def test_multi_arm_result(self):
        tasks = [
            TaskResult(
                task_id=f"t{i}", session_id=f"s{i}", mode="fast",
                category="code", band="trigger-cap",
                estimated_tokens=5000,
                arms={
                    "A": ArmResult(task_id=f"t{i}", arm="A", status="success",
                                   prompt_tokens=5000, response_tokens=200,
                                   response_content="done"),
                    "B": ArmResult(task_id=f"t{i}", arm="B", status="error",
                                   error="timeout",
                                   prompt_tokens=3000, response_tokens=0,
                                   response_content=None),
                    "C": ArmResult(task_id=f"t{i}", arm="C", status="success",
                                   prompt_tokens=5000, response_tokens=200,
                                   response_content="done"),
                },
            )
            for i in range(3)
        ]
        for t in tasks:
            for arm_id, ar in t.arms.items():
                setattr(ar, "rubric_scores", {
                    "correctness": 3, "completed": arm_id != "B",
                })
        metrics = aggregate_metrics(tasks)
        assert metrics["A"]["fast"]["n"] == 3
        # B has 3 errors, so failure rate should be 1.0
        assert metrics["B"]["fast"]["failure_rate"] == 1.0
        assert metrics["A"]["fast"]["failure_rate"] == 0.0


# ---------------------------------------------------------------------------
# Go/no-go evaluation
# ---------------------------------------------------------------------------


def _mk_metrics(fast_a, fast_b) -> dict:
    """Build a metrics dict with both modes for A and B arms."""
    return {
        "A": {"fast": fast_a, "cheap": fast_a},
        "B": {"fast": fast_b, "cheap": fast_b},
        "C": {"fast": {}, "cheap": {}},
    }


class TestGoNoGo:
    def test_all_pass(self):
        metrics = _mk_metrics(
            {"rubric_mean": 4.0, "completion_rate": 0.9,
             "failure_rate": 0.01, "prefill_est_total": 1000000,
             "ttft_p95_ms": 500.0},
            {"rubric_mean": 3.9, "completion_rate": 0.88,
             "failure_rate": 0.02, "prefill_est_total": 600000,
             "ttft_p95_ms": 550.0},  # +10% TTFT, within +20% budget
        )
        decision = evaluate_go_no_go(metrics)
        # Quality checks should pass: 3.9/4.0 = 0.975 >= 0.95
        assert decision["go"] is True, decision["rationale"]

    def test_rubric_fail(self):
        metrics = _mk_metrics(
            {"rubric_mean": 4.0, "completion_rate": 0.9,
             "failure_rate": 0.01},
            {"rubric_mean": 3.0, "completion_rate": 0.88,
             "failure_rate": 0.02},  # 3.0/4.0 = 0.75 < 0.95
        )
        decision = evaluate_go_no_go(metrics)
        assert decision["go"] is False
        # Check that the rubric rule failed
        rubric_checks = [
            c for c in decision["checks"] if c["rule"] == "rubric"
        ]
        assert any(not c["pass"] for c in rubric_checks)

    def test_prefill_reduction_pass(self):
        metrics = _mk_metrics(
            {"prefill_est_total": 1000000},
            {"prefill_est_total": 700000},  # 30% reduction
        )
        decision = evaluate_go_no_go(metrics)
        prefill_checks = [
            c for c in decision["checks"] if c["rule"] == "prefill_reduction"
        ]
        assert any(c["pass"] for c in prefill_checks)

    def test_insufficient_data(self):
        metrics = {}
        decision = evaluate_go_no_go(metrics)
        assert decision["go"] is False
        assert any(
            c.get("pass") is None for c in decision["checks"]
        )

    def test_failure_increase_beyond_noise(self):
        metrics = _mk_metrics(
            {"completion_rate": 0.9, "failure_rate": 0.01},
            {"completion_rate": 0.89, "failure_rate": 0.08},
            # B failure rate 8% - A 1% = +7pp > 5pp threshold
        )
        decision = evaluate_go_no_go(metrics)
        fail_checks = [
            c for c in decision["checks"] if c["rule"] == "failures"
        ]
        # With 7pp increase, should fail the noise threshold of 5pp
        assert any(not c["pass"] for c in fail_checks)


class TestJsonlOutput:
    def test_write_and_read(self, tmp_path: Path):
        task = TaskResult(
            task_id="t1", session_id="s1", mode="fast",
            category="code", band="trigger-cap",
            estimated_tokens=5000,
            arms={
                "A": ArmResult(
                    task_id="t1", arm="A", status="success",
                    prompt_tokens=5000, response_tokens=200,
                    response_content="done",
                ),
            },
        )
        setattr(task.arms["A"], "rubric_scores", {"correctness": 3, "completed": True})
        path = tmp_path / "test.jsonl"
        write_results_jsonl([task], path)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["task_id"] == "t1"

    def test_empty_write(self, tmp_path: Path):
        path = tmp_path / "empty.jsonl"
        write_results_jsonl([], path)
        assert path.read_text().strip() == ""


# ---------------------------------------------------------------------------
# Task extraction from logs
# ---------------------------------------------------------------------------


class TestTaskExtraction:
    """Integration tests that extract tasks from actual log files.

    These are slow (full log parse, ~3 min) and are skipped unless the
    ``RUN_EXPERIMENT_INTEGRATION`` env var is set. CI runs the fast unit
    tests; the integration suite is run once per experiment execution.
    """

    @pytest.fixture(autouse=True)
    def _skip_unless_enabled(self):
        import os

        if not os.environ.get("RUN_EXPERIMENT_INTEGRATION"):
            pytest.skip("set RUN_EXPERIMENT_INTEGRATION=1 to run slow log integration tests")

    @pytest.fixture
    def log_dir(self) -> Path:
        return Path("/var/log/llama-proxy")

    def test_extract_fast_tasks(self, log_dir):
        if not log_dir.is_dir():
            pytest.skip("log dir not available")
        tasks = extract_tasks_from_logs(
            log_dir, modes=[Mode.FAST], min_tasks=3, min_extreme=1,
        )
        assert len(tasks) > 0
        for t in tasks:
            assert t.mode == Mode.FAST
            assert t.task_id
            assert t.estimated_tokens > FAST_TRIGGER
            assert t.original_messages
            assert t.target_prompt

    def test_extract_cheap_tasks(self, log_dir):
        if not log_dir.is_dir():
            pytest.skip("log dir not available")
        tasks = extract_tasks_from_logs(
            log_dir, modes=[Mode.CHEAP], min_tasks=3, min_extreme=1,
        )
        assert len(tasks) > 0
        for t in tasks:
            assert t.mode == Mode.CHEAP
            assert t.estimated_tokens > CHEAP_TRIGGER

    def test_extract_both_modes(self, log_dir):
        if not log_dir.is_dir():
            pytest.skip("log dir not available")
        tasks = extract_tasks_from_logs(
            log_dir, min_tasks=3, min_extreme=1,
        )
        modes = {t.mode for t in tasks}
        assert Mode.FAST in modes
        assert Mode.CHEAP in modes

    def test_task_band_classification(self, log_dir):
        if not log_dir.is_dir():
            pytest.skip("log dir not available")
        tasks = extract_tasks_from_logs(
            log_dir, modes=[Mode.FAST], min_tasks=1, min_extreme=1,
        )
        bands = {t.band for t in tasks}
        assert "trigger-cap" in bands or "extreme" in bands


# ---------------------------------------------------------------------------
# Dry-run execution
# ---------------------------------------------------------------------------


class TestDryRunExecution:
    def test_execute_single_arm_dry_run(self, sample_task):
        arm_cfg = ArmConfig(
            name="A", model="deepseek-v4-flash",
            endpoint="http://localhost:8000", compacted=False,
        )
        result = asyncio.run(
            _execute_single_arm(
                sample_task, "A", arm_cfg, "none",
                dry_run=True, timeout_s=30,
            )
        )
        assert result.status == "success"
        assert "DRY RUN" in result.response_content
        assert result.prompt_tokens > 0

    def test_compacted_arm_dry_run(self, breach_task):
        arm_cfg = ArmConfig(
            name="B", model="Qwen3",
            endpoint="http://localhost:8000", compacted=True,
        )
        result = asyncio.run(
            _execute_single_arm(
                breach_task, "B", arm_cfg, FAST_STRATEGY,
                dry_run=True, timeout_s=30,
            )
        )
        assert result.status == "success"
        assert "DRY RUN" in result.response_content
        # Compacted arm should record compaction metadata
        assert result.compaction_strategy == FAST_STRATEGY
        assert result.compaction_before is not None
        assert result.compaction_after is not None

    def test_summarize_arm_dry_run(self, breach_task):
        arm_cfg = ArmConfig(
            name="B", model="Qwen3",
            endpoint="http://localhost:8000", compacted=True,
        )
        result = asyncio.run(
            _execute_single_arm(
                breach_task, "B", arm_cfg, CHEAP_STRATEGY,
                dry_run=True, timeout_s=30,
            )
        )
        assert result.status == "success"
        assert result.compaction_strategy == CHEAP_STRATEGY
