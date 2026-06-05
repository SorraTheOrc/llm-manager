"""Tests for prompt processing progress line parsing.

Validates extraction of `n_tokens` and `progress` from llama-server stdout
log lines in the format::

    slot N : prompt processing progress, n_tokens = 26988, progress = 0.658083
"""

import pytest
from proxy.server import extract_progress_data


# ---------------------------------------------------------------------------
# Valid format tests
# ---------------------------------------------------------------------------

class TestExtractProgressDataValid:
    """Standard llama-server progress lines should parse correctly."""

    def test_standard_format(self):
        line = "slot 1 : prompt processing progress, n_tokens = 26988, progress = 0.658083"
        result = extract_progress_data(line)
        assert result == (26988, 0.658083)

    def test_another_standard_line(self):
        line = "slot 3 : prompt processing progress, n_tokens = 100, progress = 0.012500"
        result = extract_progress_data(line)
        assert result == (100, 0.0125)

    def test_single_digit_tokens(self):
        line = "slot 0 : prompt processing progress, n_tokens = 5, progress = 0.000100"
        result = extract_progress_data(line)
        assert result == (5, 0.0001)

    def test_large_token_count(self):
        line = "slot 2 : prompt processing progress, n_tokens = 999999, progress = 0.999900"
        result = extract_progress_data(line)
        assert result == (999999, 0.9999)

    def test_progress_rounds_to_one(self):
        line = "slot 1 : prompt processing progress, n_tokens = 50000, progress = 1.000000"
        result = extract_progress_data(line)
        assert result == (50000, 1.0)

    def test_zero_progress(self):
        line = "slot 1 : prompt processing progress, n_tokens = 0, progress = 0.000000"
        result = extract_progress_data(line)
        assert result == (0, 0.0)

    def test_partial_digits_in_progress(self):
        line = "slot 2 : prompt processing progress, n_tokens = 3333, progress = 0.5"
        result = extract_progress_data(line)
        assert result == (3333, 0.5)

    def test_progress_with_many_decimal_places(self):
        line = "slot 1 : prompt processing progress, n_tokens = 42, progress = 0.123456789"
        result = extract_progress_data(line)
        assert result == (42, 0.123456789)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestExtractProgressDataEdgeCases:
    """Edge cases: missing fields, extra whitespace, Unicode."""

    def test_missing_n_tokens(self):
        """Line with progress but no n_tokens — should return None."""
        line = "slot 1 : prompt processing progress, progress = 0.5"
        result = extract_progress_data(line)
        assert result is None

    def test_missing_progress(self):
        """Line with n_tokens but no progress — should return None."""
        line = "slot 1 : prompt processing progress, n_tokens = 100"
        result = extract_progress_data(line)
        assert result is None

    def test_extra_whitespace_around_values(self):
        """Extra spaces between tokens and values."""
        line = "slot 1 : prompt processing progress,  n_tokens  =  26988  ,  progress  =  0.658083  "
        result = extract_progress_data(line)
        assert result == (26988, 0.658083)

    def test_unicode_in_line(self):
        """Line with Unicode characters should not crash."""
        line = "slot 1 : prompt processing progress, n_tokens = 26988, progress = 0.658083 — 日本語"
        result = extract_progress_data(line)
        assert result == (26988, 0.658083)

    def test_unicode_before_progress(self):
        """Unicode text before progress line."""
        line = "日本語 slot 1 : prompt processing progress, n_tokens = 26988, progress = 0.658083"
        result = extract_progress_data(line)
        assert result == (26988, 0.658083)

    def test_only_progress_keyword(self):
        """A line that mentions 'progress' but is not a progress line."""
        line = "slot 1 : prompt processing progress (this is a test message)"
        result = extract_progress_data(line)
        assert result is None


# ---------------------------------------------------------------------------
# Malformed input tests
# ---------------------------------------------------------------------------

class TestExtractProgressDataMalformed:
    """Malformed or unexpected input should return None."""

    def test_empty_string(self):
        result = extract_progress_data("")
        assert result is None

    def test_empty_string_spaces(self):
        result = extract_progress_data("   ")
        assert result is None

    def test_non_numeric_n_tokens(self):
        line = "slot 1 : prompt processing progress, n_tokens = abc, progress = 0.5"
        result = extract_progress_data(line)
        assert result is None

    def test_non_numeric_progress(self):
        line = "slot 1 : prompt processing progress, n_tokens = 100, progress = xyz"
        result = extract_progress_data(line)
        assert result is None

    def test_partial_line_missing_comma(self):
        line = "slot 1 : prompt processing progress n_tokens = 100 progress = 0.5"
        result = extract_progress_data(line)
        assert result is None

    def test_reversed_field_order(self):
        """Progress before n_tokens — field order may vary."""
        line = "slot 1 : prompt processing progress, progress = 0.5, n_tokens = 100"
        result = extract_progress_data(line)
        assert result == (100, 0.5)

    def test_completely_unrelated_line(self):
        line = "slot 1 : starting up"
        result = extract_progress_data(line)
        assert result is None

    def test_null_like_input(self):
        """Passing a non-string should not crash — return None."""
        result = extract_progress_data(None)  # type: ignore[arg-type]
        assert result is None


# ---------------------------------------------------------------------------
# Return type validation
# ---------------------------------------------------------------------------

class TestExtractProgressDataReturnType:
    """Verify that the returned tuple has the correct types."""

    def test_returns_int_and_float(self):
        line = "slot 1 : prompt processing progress, n_tokens = 26988, progress = 0.658083"
        result = extract_progress_data(line)
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)

    def test_returns_none_for_invalid(self):
        result = extract_progress_data("not a progress line")
        assert result is None
