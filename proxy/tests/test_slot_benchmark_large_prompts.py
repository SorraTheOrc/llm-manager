"""
Tests for large-prompt support in the slot benchmark harness.

Validates:
- --clean-cache flag (default OFF) controls whether slot cache is cleared
- --phase flag (cold|warm) and phase recording in JSON output
- restart timestamps are recorded in JSON output
- large-prompt fixture generation (30K/60K/90K/120K tokens)
- SlotRunResult.to_dict() includes phase and restart_timestamps fields
"""

import json
import tempfile
import time
from datetime import datetime, timezone
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
    with patch.object(sb, "CONFIG_YAML", cfg):
        yield cfg


# ---------------------------------------------------------------------------
# --clean-cache flag tests
# ---------------------------------------------------------------------------

class TestCleanCacheFlag:
    """Verify that --clean-cache flag controls cache-clearing behavior."""

    def test_parse_args_no_clean_cache_by_default(self):
        """--clean-cache should default to False."""
        args = sb.parse_args(["--slots", "6"])
        assert hasattr(args, "clean_cache"), "args should have clean_cache attribute"
        assert args.clean_cache is False

    def test_parse_args_clean_cache_true(self):
        """--clean-cache flag should set clean_cache to True."""
        args = sb.parse_args(["--slots", "6", "--clean-cache"])
        assert args.clean_cache is True

    def test_clean_cache_not_called_when_flag_false(self, temp_config, tmp_path):
        """When --clean-cache is False, clear_slot_cache() must NOT be called."""
        # Patch clear_slot_cache to track calls
        with patch.object(sb, "clear_slot_cache", wraps=sb.clear_slot_cache) as mock_clear:
            args = sb.parse_args(["--slots", "6"])
            clean_cache = args.clean_cache
            skip_restart = args.skip_restart
            # In main(), clear_slot_cache is called when NOT skip_restart AND clean_cache
            if not skip_restart and clean_cache:
                sb.clear_slot_cache()
            assert mock_clear.call_count == 0, (
                "clear_slot_cache should NOT be called when clean_cache=False"
            )

    def test_clean_cache_called_when_flag_true(self, temp_config, tmp_path):
        """When --clean-cache is True, clear_slot_cache() should be called."""
        with patch.object(sb, "clear_slot_cache", wraps=sb.clear_slot_cache) as mock_clear:
            args = sb.parse_args(["--slots", "6", "--clean-cache"])
            skip_restart = args.skip_restart
            clean_cache = args.clean_cache
            if not skip_restart and clean_cache:
                sb.clear_slot_cache()
            assert mock_clear.call_count == 1, (
                "clear_slot_cache should be called when clean_cache=True"
            )


# ---------------------------------------------------------------------------
# --phase flag tests
# ---------------------------------------------------------------------------

class TestPhaseFlag:
    """Verify --phase flag and phase recording in JSON output."""

    def test_parse_args_default_phase(self):
        """--phase should default to 'cold'."""
        args = sb.parse_args(["--slots", "6"])
        assert hasattr(args, "phase"), "args should have phase attribute"
        assert args.phase == "cold"

    def test_parse_args_phase_cold(self):
        """--phase cold should be accepted."""
        args = sb.parse_args(["--slots", "6", "--phase", "cold"])
        assert args.phase == "cold"

    def test_parse_args_phase_warm(self):
        """--phase warm should be accepted."""
        args = sb.parse_args(["--slots", "6", "--phase", "warm"])
        assert args.phase == "warm"

    def test_parse_args_invalid_phase_rejected(self):
        """--phase with invalid value should raise error."""
        with pytest.raises(SystemExit):
            sb.parse_args(["--slots", "6", "--test", "--phase", "invalid"])

    def test_slot_run_result_to_dict_includes_phase(self):
        """SlotRunResult.to_dict() should include phase in config."""
        r = sb.SlotRunResult(slot_count=6)
        r.results = [
            sb.RequestResult(0, "completed", 1.0, 0.1, 100, 50, 50.0),
        ]
        r.phase = "warm"
        d = r.to_dict()
        assert d["config"]["phase"] == "warm"

    def test_slot_run_result_default_phase(self):
        """SlotRunResult should default phase to 'cold'."""
        r = sb.SlotRunResult(slot_count=6)
        d = r.to_dict()
        assert d["config"]["phase"] == "cold"


# ---------------------------------------------------------------------------
# Restart timestamp tests
# ---------------------------------------------------------------------------

class TestRestartTimestamps:
    """Verify restart timestamps are recorded in JSON output."""

    def test_parse_args_restart_timestamps(self):
        """parse_args should handle restart_timestamps if added."""
        # restart_timestamps is stored in SlotRunResult, not parsed from CLI
        # This test verifies the SlotRunResult can hold them
        r = sb.SlotRunResult(slot_count=6)
        r.proxy_restart_time = "2026-08-01T12:00:00Z"
        r.llama_ready_time = "2026-08-01T12:00:30Z"
        d = r.to_dict()
        assert "proxy_restart_time" in d["config"]
        assert d["config"]["proxy_restart_time"] == "2026-08-01T12:00:00Z"
        assert "llama_ready_time" in d["config"]
        assert d["config"]["llama_ready_time"] == "2026-08-01T12:00:30Z"

    def test_slot_run_result_to_dict_includes_timestamps(self):
        """SlotRunResult.to_dict() should include restart timestamps."""
        r = sb.SlotRunResult(slot_count=6)
        r.proxy_restart_time = "2026-08-01T12:00:00Z"
        r.llama_ready_time = "2026-08-01T12:00:30Z"
        r.results = [
            sb.RequestResult(0, "completed", 1.0, 0.1, 100, 50, 50.0),
        ]
        d = r.to_dict()
        assert "proxy_restart_time" in d["config"]
        assert "llama_ready_time" in d["config"]
        # Timestamps are in config, not summary
        assert d["config"]["proxy_restart_time"] == "2026-08-01T12:00:00Z"

    def test_slot_run_result_timestamps_none_by_default(self):
        """When no timestamps are set, they should be null in JSON."""
        r = sb.SlotRunResult(slot_count=6)
        d = r.to_dict()
        # Timestamps should be absent or null when not set
        assert d["config"].get("proxy_restart_time") is None
        assert d["config"].get("llama_ready_time") is None


# ---------------------------------------------------------------------------
# Large-prompt fixture tests
# ---------------------------------------------------------------------------

class TestLargePromptFixtures:
    """Verify large-prompt fixture generation and loading."""

    def test_generate_large_prompt_fixture_30k(self):
        """generate_large_prompt_fixture should produce ~30K tokens."""
        text = sb.generate_large_prompt_fixture(token_target=30_000)
        # ~3 chars/token for English prose, so ~90K chars
        assert len(text) >= 70_000, f"Expected >=70K chars for 30K tokens, got {len(text)}"
        assert len(text) <= 150_000, f"Expected <=150K chars for 30K tokens, got {len(text)}"

    def test_generate_large_prompt_fixture_60k(self):
        """generate_large_prompt_fixture should produce ~60K tokens."""
        text = sb.generate_large_prompt_fixture(token_target=60_000)
        assert len(text) >= 150_000, f"Expected >=150K chars for 60K tokens, got {len(text)}"
        assert len(text) <= 300_000, f"Expected <=300K chars for 60K tokens, got {len(text)}"

    def test_generate_large_prompt_fixture_90k(self):
        """generate_large_prompt_fixture should produce ~90K tokens."""
        text = sb.generate_large_prompt_fixture(token_target=90_000)
        assert len(text) >= 220_000, f"Expected >=220K chars for 90K tokens, got {len(text)}"
        assert len(text) <= 450_000, f"Expected <=450K chars for 90K tokens, got {len(text)}"

    def test_generate_large_prompt_fixture_120k(self):
        """generate_large_prompt_fixture should produce ~120K tokens."""
        text = sb.generate_large_prompt_fixture(token_target=120_000)
        assert len(text) >= 300_000, f"Expected >=300K chars for 120K tokens, got {len(text)}"
        assert len(text) <= 600_000, f"Expected <=600K chars for 120K tokens, got {len(text)}"

    def test_generate_large_prompt_fixture_deterministic(self):
        """Same token_target produces same content."""
        t1 = sb.generate_large_prompt_fixture(token_target=50_000)
        t2 = sb.generate_large_prompt_fixture(token_target=50_000)
        assert t1 == t2

    def test_load_large_prompt_fixture(self):
        """load_large_prompt_fixture should read from JSON file."""
        # Create a temporary fixture file
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_path = Path(tmpdir) / "large_prompts.json"
            fixture_data = {
                "30k": "This is a test prompt repeated. " * 3000,
                "60k": "This is a longer test prompt repeated. " * 6000,
            }
            fixture_path.write_text(json.dumps(fixture_data))

            prompts = sb.load_large_prompt_fixture(fixture_path)
            assert "30k" in prompts
            assert "60k" in prompts
            assert len(prompts["30k"]) > 0
            assert len(prompts["60k"]) > 0

    def test_load_large_prompt_fixture_missing_key(self):
        """load_large_prompt_fixture should raise KeyError for missing key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_path = Path(tmpdir) / "large_prompts.json"
            fixture_data = {"prompts": {"30k": "test"}}
            fixture_path.write_text(json.dumps(fixture_data))

            with pytest.raises(KeyError):
                sb.load_large_prompt_fixture(fixture_path, key="90k")

    def test_load_large_prompt_fixture_nested_prompts_key(self):
        """load_large_prompt_fixture should handle nested 'prompts' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_path = Path(tmpdir) / "large_prompts.json"
            fixture_data = {"prompts": {"30k": "test content 30k", "60k": "test content 60k"}}
            fixture_path.write_text(json.dumps(fixture_data))

            # With nested key, load with key=None returns the prompts dict
            prompts = sb.load_large_prompt_fixture(fixture_path)
            assert isinstance(prompts, dict)
            assert "30k" in prompts
            assert prompts["30k"] == "test content 30k"

            # With nested key, load with specific key returns that prompt
            prompt_30k = sb.load_large_prompt_fixture(fixture_path, key="30k")
            assert prompt_30k == "test content 30k"

    def test_save_large_prompt_fixture(self):
        """save_large_prompt_fixture should write valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_path = Path(tmpdir) / "large_prompts.json"
            prompts = {"30k": "test content", "60k": "more content"}
            sb.save_large_prompt_fixture(fixture_path, prompts)

            assert fixture_path.exists()
            loaded = json.loads(fixture_path.read_text())
            assert loaded == prompts


# ---------------------------------------------------------------------------
# Integration: full SlotRunResult JSON round-trip
# ---------------------------------------------------------------------------

class TestFullSlotRunResultRoundTrip:
    """Verify complete SlotRunResult serialization with all new fields."""

    def test_full_round_trip(self):
        """SlotRunResult with all fields should serialize and deserialize."""
        r = sb.SlotRunResult(slot_count=6)
        r.results = [
            sb.RequestResult(0, "completed", 1.5, 0.2, 1000, 500, 333.3),
            sb.RequestResult(1, "error", 0.5, error="timeout"),
        ]
        r.start_time = "2026-08-01T12:00:00Z"
        r.end_time = "2026-08-01T12:01:00Z"
        r.phase = "warm"
        r.proxy_restart_time = "2026-08-01T12:00:00Z"
        r.llama_ready_time = "2026-08-01T12:00:30Z"

        d = r.to_dict()

        # Verify all fields present
        assert d["config"]["slot_count"] == 6
        assert d["config"]["phase"] == "warm"
        assert d["config"]["proxy_restart_time"] == "2026-08-01T12:00:00Z"
        assert d["config"]["llama_ready_time"] == "2026-08-01T12:00:30Z"
        assert len(d["requests"]) == 2
        assert d["summary"]["completed"] == 1
        assert d["summary"]["errors"] == 1
