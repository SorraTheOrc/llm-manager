import re
import sys
from proxy import server


def test_format_progress_basic():
    s = server.format_progress(26988, 40494, 0.658083)
    assert s.startswith("\r"), "progress string must start with carriage return for in-place updates"
    assert "Processing 26988/40494 tokens (65%)" in s
    assert "\x1b[2m" in s, "ANSI DIM (\x1b[2m) must be present"


def test_format_progress_zero_tokens():
    s = server.format_progress(0, 1000, 0.0)
    assert "Processing 0/1000 tokens (0%)" in s


def test_format_progress_100_percent():
    s = server.format_progress(50000, 50000, 1.0)
    assert "(100%)" in s


def test_format_progress_percentage_truncation():
    # 0.659 -> 65%
    s = server.format_progress(10, 1000, 0.659)
    assert "(65%)" in s


def test_format_progress_ansi_reset():
    # Ensure the string ends with reset code so subsequent output isn't dimmed
    s = server.format_progress(1, 2, 0.5)
    assert s.endswith("\x1b[0m"), "progress string should reset ANSI formatting at end"
