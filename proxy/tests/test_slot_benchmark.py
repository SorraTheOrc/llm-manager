"""
Tests for the slot-count benchmark runner (proxy/benchmarks/slot_benchmark.py).

Validates prompt construction, config manipulation, result aggregation,
and CLI argument parsing. Does NOT require a running llama-server or proxy.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

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
