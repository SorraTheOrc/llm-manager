"""
Tests for the slot-count benchmark runner (proxy/benchmarks/slot_benchmark.py).

Validates prompt construction, config manipulation, result aggregation,
and CLI argument parsing. Does NOT require a running llama-server or proxy.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

try:
    from benchmarks import slot_benchmark as sb
except ImportError:
    try:
        from proxy.benchmarks import slot_benchmark as sb
    except ImportError:
        import sys

        this_dir = Path(__file__).resolve().parent
        proxy_dir = this_dir.parent  # proxy/
        root_dir = proxy_dir.parent  # project root
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))
        from proxy.benchmarks import slot_benchmark as sb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_config(tmp_path: Path) -> Path:
    """Create a temporary config.yaml with session_slot_pool_size set to 6."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "server:\n"
        "  session_slot_pool_size: 6\n"
        "  port: 8000\n"
    )
    # Point the benchmark at our temp config
    with patch.object(sb, "CONFIG_YAML", cfg):
        yield cfg


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

class TestBuildPrompts:
    """Verify the split prompt architecture: shared ~8k + varied ~8k per request."""

    def test_shared_content_is_string(self):
        content = sb.build_shared_content()
        assert isinstance(content, str)
        assert len(content) > 0
        assert "PART 1" in content
        assert "PART 2" in content

    def test_shared_content_approx_3k_tokens(self):
        content = sb.build_shared_content()
        # ~3k tokens of English prose ≈ ~6-30k chars
        assert 6_000 <= len(content) <= 40_000, (
            f"Expected ~3k tokens of shared content, got {len(content)} chars"
        )

    def test_varied_content_differs_per_index(self):
        c0 = sb.build_varied_content(0)
        c1 = sb.build_varied_content(1)
        assert c0 != c1, "Each index should produce unique varied content"
        assert "---PART 2 END---" in c0

    def test_varied_content_approx_3k_tokens(self):
        content = sb.build_varied_content(0)
        # ~1000 numbers + commas + closing marker ≈ 4-20k chars
        assert 4_000 <= len(content) <= 20_000, (
            f"Expected ~3k tokens worth of numbers, got {len(content)} chars"
        )
        # Verify it contains the expected number of commas (= numbers - 1)
        comma_count = content.count(",")
        assert 800 <= comma_count <= 1200, f"Expected ~1000 commas, got {comma_count}"

    def test_varied_content_contains_only_numbers(self):
        content = sb.build_varied_content(0)
        # Strip the closing marker, split on commas, verify each part is a number
        data_part = content.replace("---PART 2 END---", "").strip(",")
        numbers = data_part.split(",")
        assert len(numbers) >= 800, f"Expected hundreds of numbers, got {len(numbers)}"
        for n in numbers[:10]:
            assert n.strip().isdigit(), f"Expected number, got: {n}"

    def test_varied_content_reproducible(self):
        """Same index produces same content (seeded RNG)."""
        c0a = sb.build_varied_content(0)
        c0b = sb.build_varied_content(0)
        assert c0a == c0b

    def test_build_user_prompt(self):
        prompt = sb.build_user_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50
        assert "count" in prompt.lower() or "Total:" in prompt


# ---------------------------------------------------------------------------
# Config manipulation
# ---------------------------------------------------------------------------

class TestConfigManipulation:
    """Verify config.yaml read/write and slot-pool-size replacement."""

    def test_read_config(self, temp_config):
        content = sb._read_config()
        assert "session_slot_pool_size: 6" in content

    def test_replace_slot_pool_size_increases(self):
        content = "server:\n  session_slot_pool_size: 6\n"
        updated = sb._replace_slot_pool_size(content, 12)
        assert "session_slot_pool_size: 12" in updated

    def test_replace_slot_pool_size_decreases(self):
        content = "server:\n  session_slot_pool_size: 6\n"
        updated = sb._replace_slot_pool_size(content, 1)
        assert "session_slot_pool_size: 1" in updated

    def test_replace_slot_pool_size_preserves_other_config(self):
        content = (
            "server:\n"
            "  session_slot_pool_size: 6\n"
            "  port: 8000\n"
            "  llama_router_mode: true\n"
        )
        updated = sb._replace_slot_pool_size(content, 4)
        assert "port: 8000" in updated
        assert "llama_router_mode: true" in updated

    def test_set_slot_count(self, temp_config):
        sb.set_slot_count(12)
        content = sb._read_config()
        assert "session_slot_pool_size: 12" in content


# ---------------------------------------------------------------------------
# RequestResult dataclass
# ---------------------------------------------------------------------------

class TestRequestResult:
    """Verify RequestResult construction and serialization."""

    def test_create_completed(self):
        r = sb.RequestResult(
            request_index=0,
            status="completed",
            total_duration_seconds=600.0,
            time_to_first_token_seconds=2.5,
            prompt_tokens=16_000,
            completion_tokens=18_000,
            tokens_per_second=30.0,
        )
        assert r.status == "completed"
        assert r.prompt_tokens == 16_000

    def test_create_error(self):
        r = sb.RequestResult(
            request_index=1,
            status="error",
            total_duration_seconds=60.0,
            error="Timeout: connection reset",
        )
        assert r.status == "error"
        assert "Timeout" in r.error

    def test_to_dict_round_trip(self):
        r = sb.RequestResult(
            request_index=0,
            status="completed",
            total_duration_seconds=600.123,
            time_to_first_token_seconds=2.456,
            prompt_tokens=16_000,
            completion_tokens=18_000,
            tokens_per_second=30.01,
            resolved_model="local-qwen3/Qwen3",
        )
        d = r.to_dict()
        assert d["total_duration_seconds"] == 600.123
        assert d["time_to_first_token_seconds"] == 2.456
        assert d["tokens_per_second"] == 30.01
        assert d["resolved_model"] == "local-qwen3/Qwen3"

    def test_to_dict_error_no_ttft(self):
        r = sb.RequestResult(
            request_index=0,
            status="error",
            total_duration_seconds=10.0,
            error="fail",
        )
        d = r.to_dict()
        assert d["time_to_first_token_seconds"] is None
        assert d["resolved_model"] is None


# ---------------------------------------------------------------------------
# SlotRunResult and summary
# ---------------------------------------------------------------------------

class TestSlotRunResult:
    """Verify SlotRunResult aggregation and summary."""

    def test_summary_empty(self):
        r = sb.SlotRunResult(slot_count=6)
        s = r.summary()
        assert s["slot_count"] == 6
        assert s["completed"] == 0
        assert s["errors"] == 0
        assert s["avg_duration"] is None

    def test_summary_with_completed_requests(self):
        r = sb.SlotRunResult(slot_count=6)
        r.results = [
            sb.RequestResult(0, "completed", 600.0, 2.0, 16000, 18000, 30.0),
            sb.RequestResult(1, "completed", 750.0, 3.0, 16000, 20000, 26.7),
        ]
        s = r.summary()
        assert s["completed"] == 2
        assert s["errors"] == 0
        assert s["avg_duration"] == 675.0  # (600 + 750) / 2
        assert s["avg_ttft"] == 2.5
        assert s["total_completion_tokens"] == 38_000

    def test_summary_with_errors(self):
        r = sb.SlotRunResult(slot_count=4)
        r.results = [
            sb.RequestResult(0, "completed", 600.0, 2.0, 16000, 18000, 30.0),
            sb.RequestResult(1, "error", 60.0, error="timeout"),
            sb.RequestResult(2, "completed", 700.0, 2.5, 16000, 19000, 27.1),
        ]
        s = r.summary()
        assert s["completed"] == 2
        assert s["errors"] == 1

    def test_to_dict_round_trip(self):
        r = sb.SlotRunResult(slot_count=6)
        r.results = [
            sb.RequestResult(0, "completed", 600.0, 2.0, 16000, 18000, 30.0),
        ]
        r.start_time = "2026-07-22T12:00:00Z"
        r.end_time = "2026-07-22T12:10:00Z"
        d = r.to_dict()
        assert d["config"]["slot_count"] == 6
        assert len(d["requests"]) == 1
        assert d["summary"]["completed"] == 1
        assert d["summary"]["avg_duration"] == 600.0


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class TestReportGeneration:
    """Verify generate_report produces valid JSON and a readable summary."""

    def test_generate_report_writes_files(self, tmp_path):
        results = [
            sb.SlotRunResult(slot_count=1),
            sb.SlotRunResult(slot_count=6),
        ]
        results[0].results = [
            sb.RequestResult(0, "completed", 600.0, 2.0, 16000, 18000, 30.0),
        ]
        results[1].results = [
            sb.RequestResult(0, "completed", 700.0, 3.0, 16000, 20000, 28.6),
        ]

        report_path = sb.generate_report(results, tmp_path)
        json_path = tmp_path / "slot_benchmark_report.json"
        summary_path = tmp_path / "slot_benchmark_summary.txt"

        assert report_path == json_path
        assert json_path.exists()
        assert summary_path.exists()

        # Validate JSON structure
        with open(json_path) as f:
            data = json.load(f)
        assert "benchmark" in data
        assert "runs" in data
        assert len(data["runs"]) == 2
        assert data["runs"][0]["slot_count"] == 1
        assert data["runs"][1]["slot_count"] == 6

    def test_summary_text_contains_table(self, tmp_path):
        results = [
            sb.SlotRunResult(slot_count=6),
        ]
        results[0].results = [
            sb.RequestResult(0, "completed", 600.0, 2.0, 16000, 18000, 30.0),
        ]
        sb.generate_report(results, tmp_path)

        summary = (tmp_path / "slot_benchmark_summary.txt").read_text()
        assert "SLOT-COUNT BENCHMARK REPORT" in summary
        assert "Slots" in summary  # table header
        assert "6" in summary
        assert "600" in summary


# ---------------------------------------------------------------------------
# Prompt validation
# ---------------------------------------------------------------------------

class TestCombinedPrompt:
    """Validation that shared + varied content together hit ~16k tokens."""

    def test_combined_total_char_count(self):
        shared = sb.build_shared_content()
        varied = sb.build_varied_content(0)
        combined = shared + varied
        # Combined: shared + varied content
        assert 10_000 <= len(combined) <= 60_000, (
            f"Expected combined content, got {len(combined)} chars"
        )

    def test_combined_differs_across_indices(self):
        c0 = sb.build_shared_content() + sb.build_varied_content(0)
        c1 = sb.build_shared_content() + sb.build_varied_content(1)
        assert c0 != c1, "Different indices should produce different combined content"


# ---------------------------------------------------------------------------
# Slot counts constant
# ---------------------------------------------------------------------------

class TestSlotCounts:
    """Verify SLOT_COUNTS covers meaningful values."""

    def test_slot_counts_include_current(self):
        assert 6 in sb.SLOT_COUNTS, "Should include current production value"

    def test_slot_counts_include_lower_and_higher(self):
        assert any(s < 6 for s in sb.SLOT_COUNTS), "Should test below current"
        assert any(s > 6 for s in sb.SLOT_COUNTS), "Should test above current"

    def test_slot_counts_sorted(self):
        assert sb.SLOT_COUNTS == sorted(sb.SLOT_COUNTS)


# ---------------------------------------------------------------------------
# Configurable workload parameters (concurrency / prompt size)
# ---------------------------------------------------------------------------

class TestWorkloadParameters:
    """CLI parameters for concurrency and prompt size (AC1)."""

    def test_cli_parses_concurrency_and_prompt_tokens(self):
        args = sb.parse_args([
            "--slots", "1", "--concurrency", "3", "--prompt-tokens", "2000",
        ])
        assert args.concurrency == 3
        assert args.prompt_tokens == 2000

    def test_cli_defaults_preserve_existing_behaviour(self):
        args = sb.parse_args(["--slots", "1"])
        assert args.concurrency == sb.TOTAL_REQUESTS_PER_RUN
        assert args.prompt_tokens == sb.TARGET_PROMPT_TOKENS

    def test_health_sampling_on_by_default_with_opt_out(self):
        assert sb.parse_args(["--slots", "1"]).no_health_samples is False
        assert sb.parse_args(["--slots", "1", "--no-health-samples"]).no_health_samples is True

    def test_all_flag_still_accepted(self):
        args = sb.parse_args(["--all"])
        assert args.all is True
        assert args.concurrency == sb.TOTAL_REQUESTS_PER_RUN
        assert args.prompt_tokens == sb.TARGET_PROMPT_TOKENS

    def test_prompt_tokens_scales_shared_and_varied_content(self):
        small = sb.build_shared_content(prompt_tokens=2_000)
        large = sb.build_shared_content(prompt_tokens=12_000)
        assert len(small) < len(large)

        varied_small = sb.build_varied_content(0, prompt_tokens=2_000)
        varied_large = sb.build_varied_content(0, prompt_tokens=12_000)
        assert varied_small.count(",") < varied_large.count(",")

    def test_prompt_tokens_default_matches_global_target(self):
        # Default must reproduce the historical ~6k split exactly.
        assert sb.build_varied_content(0).count(",") == 999

    def test_run_metadata_records_workload(self):
        r = sb.SlotRunResult(slot_count=2, concurrency=3, prompt_tokens_target=2_000)
        d = r.to_dict()
        assert d["config"]["concurrency"] == 3
        assert d["config"]["prompt_tokens_target"] == 2_000


# ---------------------------------------------------------------------------
# Contention metric derivation
# ---------------------------------------------------------------------------

TS = "2026-08-23 01:18:32,000"


def _dispatch_line(ts: str, wait: float, session: str = "s-1") -> str:
    return (
        f"{ts} INFO router.routing: contention_queue_dispatch "
        f"provider=local session={session} queued_duration={wait:.2f}s policy=queue depth=0"
    )


def _fallback_line(ts: str, wait: float, session: str = "s-1") -> str:
    return (
        f"{ts} INFO router.routing: contention_queue_fallback_after_queue "
        f"provider=local session={session} queued_duration={wait:.2f}s"
    )


class TestLocalStreamShare:
    def test_mixed_results(self):
        results = [
            sb.RequestResult(0, "completed", 1.0, resolved_model="local/Qwen3"),
            sb.RequestResult(1, "completed", 1.0, resolved_model="local-qwen3/Qwen3"),
            sb.RequestResult(2, "completed", 1.0, resolved_model="deepseek/v4"),
        ]
        local, observed, share, reason = sb.derive_local_stream_share(results)
        assert (local, observed) == (2, 3)
        assert share == pytest.approx(2 / 3)
        assert reason is None

    def test_ignores_errors_without_resolved_model(self):
        results = [
            sb.RequestResult(0, "completed", 1.0, resolved_model="local/Qwen3"),
            sb.RequestResult(1, "error", 1.0, error="timeout"),
        ]
        local, observed, share, reason = sb.derive_local_stream_share(results)
        assert (local, observed) == (1, 1)
        assert share == 1.0
        assert reason is None

    def test_no_resolved_models_reports_reason(self):
        results = [sb.RequestResult(0, "error", 1.0, error="timeout")]
        local, observed, share, reason = sb.derive_local_stream_share(results)
        assert share is None
        assert observed == 0
        assert reason


class TestQuarantineEpisodes:
    def _snap(self, reachable: bool, healing: bool = False, running: bool = True):
        return {
            "llama_server_running": running,
            "backend_reachable": reachable,
            "self_healing_in_progress": healing,
        }

    def test_counts_available_to_unavailable_transitions(self):
        samples = [
            self._snap(True),
            self._snap(False),
            self._snap(False),
            self._snap(True),
            self._snap(True),
            self._snap(False),
        ]
        count, reason = sb.count_quarantine_episodes(samples)
        assert count == 2
        assert reason is None

    def test_self_healing_counts_as_unavailable(self):
        samples = [self._snap(True), self._snap(True, healing=True), self._snap(True)]
        count, reason = sb.count_quarantine_episodes(samples)
        assert count == 1
        assert reason is None

    def test_unknown_samples_are_skipped(self):
        samples = [self._snap(True), {}, self._snap(False)]
        count, reason = sb.count_quarantine_episodes(samples)
        assert count == 1
        assert reason is None

    def test_no_samples_reports_reason(self):
        count, reason = sb.count_quarantine_episodes([])
        assert count is None
        assert reason

    def test_all_unknown_samples_reports_reason(self):
        # None sentinels (network failures) carry no availability signal.
        count, reason = sb.count_quarantine_episodes([None, None])
        assert count is None
        assert reason


class TestBuildContentionMetrics:
    def test_derives_queue_metrics_from_logs(self, tmp_path: Path):
        log = tmp_path / "proxy.log"
        log.write_text(
            _dispatch_line(TS, 5.0) + "\n"
            + _dispatch_line(TS, 2.0) + "\n"
            + _fallback_line(TS, 60.0) + "\n"
        )
        metrics = sb.build_contention_metrics(
            [sb.RequestResult(0, "completed", 1.0, resolved_model="local/Qwen3")],
            log_files=[str(log)],
            start_ts=0.0,
            end_ts=2_000_000_000.0,
            health_snapshots=[{
                "llama_server_running": True,
                "backend_reachable": True,
                "self_healing_in_progress": False,
            }],
        )
        assert metrics.queue_timeout_count == 1
        assert metrics.queue_dispatch_count == 2
        assert metrics.queue_timeout_rate == pytest.approx(1 / 3)
        assert metrics.local_stream_share == 1.0
        assert metrics.quarantine_episode_count == 0
        assert metrics.unavailable == {}

    def test_missing_logs_reports_null_and_reason(self):
        metrics = sb.build_contention_metrics(
            [], log_files=["/nonexistent/proxy.log*"], start_ts=0.0, end_ts=1.0,
        )
        d = metrics.to_dict()
        assert d["queue_timeout_count"] is None
        assert d["queue_timeout_rate"] is None
        assert "queue_timeout" in d["unavailable"]
        assert d["local_stream_share"] is None
        assert "local_stream_share" in d["unavailable"]
        assert d["quarantine_episode_count"] is None
        assert "quarantine_episodes" in d["unavailable"]

    def test_no_queue_events_counts_zero_but_rate_null(self, tmp_path: Path):
        log = tmp_path / "proxy.log"
        log.write_text("2026-08-23 01:18:32,000 INFO router.routing: unrelated x=1\n")
        metrics = sb.build_contention_metrics(
            [], log_files=[str(log)], start_ts=0.0, end_ts=2_000_000_000.0,
        )
        assert metrics.queue_timeout_count == 0
        assert metrics.queue_dispatch_count == 0
        assert metrics.queue_timeout_rate is None
        assert metrics.unavailable.get("queue_timeout")

    def test_metrics_surface_in_summary_and_dict(self, tmp_path: Path):
        log = tmp_path / "proxy.log"
        log.write_text(_fallback_line(TS, 60.0) + "\n")
        metrics = sb.build_contention_metrics(
            [sb.RequestResult(0, "completed", 1.0, resolved_model="local/Qwen3")],
            log_files=[str(log)], start_ts=0.0, end_ts=2_000_000_000.0,
        )
        r = sb.SlotRunResult(slot_count=2, contention=metrics)
        s = r.summary()
        assert s["contention"]["queue_timeout_count"] == 1
        assert s["contention"]["local_stream_share"] == 1.0
        assert s["contention"]["quarantine_episode_count"] is None
        assert r.to_dict()["contention"]["queue_timeout_count"] == 1
        assert r.to_dict()["summary"]["contention"]["queue_timeout_count"] == 1


class TestReportContentionMetrics:
    def test_summary_text_includes_contention_section(self, tmp_path: Path):
        run = sb.SlotRunResult(
            slot_count=2,
            contention=sb.ContentionMetrics(
                queue_timeout_count=3,
                queue_dispatch_count=7,
                queue_timeout_rate=0.3,
                local_stream_count=4,
                observed_stream_count=5,
                local_stream_share=0.8,
                quarantine_episode_count=1,
            ),
        )
        run.results = [sb.RequestResult(0, "completed", 600.0, 2.0, 16000, 18000, 30.0)]
        sb.generate_report([run], tmp_path)
        summary = (tmp_path / "slot_benchmark_summary.txt").read_text()
        assert "CONTENTION METRICS" in summary
        assert "queue-timeout" in summary.lower()
        assert "local stream share" in summary.lower()
        assert "quarantine" in summary.lower()

    def test_absent_metrics_render_reason(self, tmp_path: Path):
        run = sb.SlotRunResult(
            slot_count=2,
            contention=sb.ContentionMetrics(
                unavailable={
                    "queue_timeout": "no proxy log files matched",
                    "local_stream_share": "no completed requests reported a resolved model",
                    "quarantine_episodes": "no health snapshots collected",
                },
            ),
        )
        sb.generate_report([run], tmp_path)
        summary = (tmp_path / "slot_benchmark_summary.txt").read_text()
        assert "no proxy log files matched" in summary
        assert "no health snapshots collected" in summary
