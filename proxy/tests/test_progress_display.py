import re
import sys
from proxy import server


def test_format_progress_basic():
    s = server.format_progress(26988, 40494, 0.658083, model_name="Qwen3", slot_id=2, tokens_per_sec=45.2)
    assert s.startswith("\r"), "progress string must start with carriage return for in-place updates"
    assert "Processing 26988/40494 tokens (65%)" in s
    assert "[slot:2 Qwen3]" in s, "model/slot prefix must be present"
    assert "@ 45.2 tok/s" in s, "tokens-per-second must be present"
    assert "\x1b[2m" in s, "ANSI DIM (\x1b[2m) must be present"


def test_format_progress_zero_tokens():
    s = server.format_progress(0, 1000, 0.0, model_name="gemma4", slot_id=0, tokens_per_sec=None)
    assert "Processing 0/1000 tokens (0%)" in s
    assert "[slot:0 gemma4]" in s
    assert "@ --.- tok/s" in s, "None TPS should show placeholder"


def test_format_progress_100_percent():
    s = server.format_progress(50000, 50000, 1.0, model_name="Qwen3", slot_id=3, tokens_per_sec=123.4)
    assert "(100%)" in s
    assert "[slot:3 Qwen3]" in s
    assert "@ 123.4 tok/s" in s


def test_format_progress_percentage_truncation():
    # 0.659 -> 65%
    s = server.format_progress(10, 1000, 0.659, model_name="default", slot_id=1)
    assert "(65%)" in s


def test_format_progress_ansi_reset():
    # Ensure the string ends with reset code so subsequent output isn't dimmed
    s = server.format_progress(1, 2, 0.5, model_name="test", slot_id=1, tokens_per_sec=99.9)
    assert s.endswith("\x1b[0m"), "progress string should reset ANSI formatting at end"


def test_format_progress_default_model_name():
    """When model_name is not provided, defaults to 'unknown'."""
    s = server.format_progress(10, 100, 0.5)
    assert "[slot:0 unknown]" in s


def test_format_progress_tps_one_decimal():
    """TPS should be formatted to one decimal place."""
    s = server.format_progress(100, 200, 0.5, model_name="m", slot_id=1, tokens_per_sec=45.678)
    assert "@ 45.7 tok/s" in s


def test_format_progress_tps_whole_number():
    """Whole number TPS should still show one decimal."""
    s = server.format_progress(100, 200, 0.5, model_name="m", slot_id=1, tokens_per_sec=50.0)
    assert "@ 50.0 tok/s" in s


def test_format_progress_tps_very_small():
    """Very small TPS should show properly."""
    s = server.format_progress(1, 200, 0.005, model_name="slow", slot_id=0, tokens_per_sec=0.3)
    assert "@ 0.3 tok/s" in s


def test_format_progress_tps_none_when_elapsed_zero():
    """When tokens_per_sec is None (first progress update), show placeholder."""
    s = server.format_progress(50, 500, 0.1, model_name="Qwen3", slot_id=1, tokens_per_sec=None)
    assert "@ --.- tok/s" in s


def test_format_progress_dim_wraps_entire_line():
    """The entire line — prefix, body, and tps suffix — is wrapped in dim formatting."""
    s = server.format_progress(100, 200, 0.5, model_name="Qwen3", slot_id=1, tokens_per_sec=45.2)
    # The dim sequence should be at the very start (after \r) and reset at the very end
    # Pattern: \r\x1b[2m[slot:...] Processing ... @ ... tok/s\x1b[0m
    assert s.startswith("\r\x1b[2m")
    assert s.endswith("\x1b[0m")
    # Check that all content is between dim and reset
    # \r is 1 char, \x1b[2m is 4 chars = 5 chars prefix
    content = s[5:]  # Strip \r\x1b[2m
    assert content.startswith("[slot:")
    assert content.endswith("\x1b[0m")
    assert "tok/s" in content[:-4]  # Before the reset sequence
