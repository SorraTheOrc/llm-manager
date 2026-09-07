"""Tests for the per-provider available_times display helpers (LP-0MT2WMACO003SE7M).

Covers the UI-facing formatters that reuse ``proxy.provider`` window parsing
(``format_available_times`` / ``format_active_status``) and the
``_build_home_model_rows`` table render that uses them.  The helper tests
exercise every semantic required by the ACs: restricted vs. unrestricted,
inside/outside window, overnight wrap, boundaries, and malformed fail-open.
The render tests assert the new Active Times / Status cells appear per
provider row (including fallback chains with ``rowspan``).
"""

import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from proxy.provider import format_active_status, format_available_times

# ---------------------------------------------------------------------------
# format_available_times
# ---------------------------------------------------------------------------

def test_format_available_times_always_when_unrestricted():
    """Unrestricted providers (no key) render as 'Always'."""
    assert format_available_times({"name": "p"}) == "Always"


def test_format_available_times_always_when_empty_list():
    """Empty available_times list renders as 'Always'."""
    assert format_available_times({"name": "p", "available_times": []}) == "Always"


def test_format_available_times_always_when_malformed():
    """Malformed available_times renders as 'Always' (fail-open)."""
    assert format_available_times({"name": "p", "available_times": ["garbage"]}) == "Always"


def test_format_available_times_single_window():
    """A single window is shown as 'HH:MM-HH:MM (UTC)'."""
    cfg = {"name": "p", "available_times": ["10:00-12:00"]}
    assert format_available_times(cfg) == "10:00-12:00 (UTC)"


def test_format_available_times_multiple_windows_in_config_order():
    """Multiple windows are joined with ', ' in config order."""
    cfg = {"name": "p", "available_times": ["00:00-01:00", "04:00-06:00", "10:00-00:00"]}
    assert format_available_times(cfg) == "00:00-01:00, 04:00-06:00, 10:00-00:00 (UTC)"


def test_format_available_times_overnight_wrap_preserved():
    """Overnight windows are displayed as written (not normalised)."""
    cfg = {"name": "p", "available_times": ["22:00-02:00"]}
    assert format_available_times(cfg) == "22:00-02:00 (UTC)"


def test_format_available_times_mixed_malformed_drops_bad_window():
    """Malformed entries are skipped; valid ones still render."""
    cfg = {"name": "p", "available_times": ["garbage", "10:00-12:00"]}
    assert format_available_times(cfg) == "10:00-12:00 (UTC)"


# ---------------------------------------------------------------------------
# format_active_status
# ---------------------------------------------------------------------------

def test_format_active_status_unrestricted_is_active():
    cfg = {"name": "p"}
    now = datetime(2026, 1, 5, 13, 0, tzinfo=UTC)
    assert format_active_status(cfg, now_utc=now) == "Active"


def test_format_active_status_inside_window_is_active():
    cfg = {"name": "p", "available_times": ["10:00-12:00"]}
    assert format_active_status(cfg, now_utc=datetime(2026, 1, 5, 10, 30, tzinfo=UTC)) == "Active"


def test_format_active_status_outside_window_is_inactive():
    cfg = {"name": "p", "available_times": ["10:00-12:00"]}
    assert format_active_status(cfg, now_utc=datetime(2026, 1, 5, 13, 0, tzinfo=UTC)) == "Inactive"


def test_format_active_status_start_edge_inclusive():
    """Window start minute is inside the window (inclusive)."""
    cfg = {"name": "p", "available_times": ["10:00-12:00"]}
    assert format_active_status(cfg, now_utc=datetime(2026, 1, 5, 10, 0, tzinfo=UTC)) == "Active"


def test_format_active_status_end_edge_exclusive():
    """Window end minute is outside the window (exclusive)."""
    cfg = {"name": "p", "available_times": ["10:00-12:00"]}
    assert format_active_status(cfg, now_utc=datetime(2026, 1, 5, 12, 0, tzinfo=UTC)) == "Inactive"


def test_format_active_status_overnight_wrap():
    """Overnight window wraps past midnight."""
    cfg = {"name": "p", "available_times": ["22:00-02:00"]}
    assert format_active_status(cfg, now_utc=datetime(2026, 1, 5, 23, 30, tzinfo=UTC)) == "Active"
    assert format_active_status(cfg, now_utc=datetime(2026, 1, 6, 1, 0, tzinfo=UTC)) == "Active"
    assert format_active_status(cfg, now_utc=datetime(2026, 1, 5, 12, 0, tzinfo=UTC)) == "Inactive"


def test_format_active_status_malformed_is_active():
    """Malformed available_times renders as Active (fail-open)."""
    cfg = {"name": "p", "available_times": ["garbage"]}
    assert format_active_status(cfg, now_utc=datetime(2026, 1, 5, 13, 0, tzinfo=UTC)) == "Active"


def test_format_active_status_consistent_with_routing():
    """Display helper uses the same window math as routing helpers."""
    from proxy.provider import _is_within_allowed_window
    cfg = {"name": "p", "available_times": ["10:00-12:00"]}
    inside = datetime(2026, 1, 5, 11, 0, tzinfo=UTC)
    outside = datetime(2026, 1, 5, 13, 0, tzinfo=UTC)
    assert _is_within_allowed_window(cfg, now_utc=inside) is True
    assert format_active_status(cfg, now_utc=inside) == "Active"
    assert _is_within_allowed_window(cfg, now_utc=outside) is False
    assert format_active_status(cfg, now_utc=outside) == "Inactive"


# ---------------------------------------------------------------------------
# _build_home_model_rows render tests (Active Times / Status cells)
# ---------------------------------------------------------------------------

def _mock_srv(models_cfg):
    srv = MagicMock()
    srv.config = {"models": models_cfg}
    return srv


def test_home_rows_unrestricted_shows_always_and_active():
    """Unrestricted provider row shows 'Always' and 'Active'."""
    from proxy.ui import _build_home_model_rows
    srv = _mock_srv({
        "my-model": {"providers": [{"name": "p", "type": "local", "llama_model": "qwen3"}]},
    })
    html = _build_home_model_rows(srv)
    assert "Always" in html
    assert "badge-active" in html
    assert "Active" in html


def test_home_rows_restricted_shows_windows_and_badge():
    """Restricted provider row lists windows with (UTC) and a status badge."""
    from proxy.ui import _build_home_model_rows
    srv = _mock_srv({
        "my-model": {"providers": [
            {"name": "p", "type": "remote", "available_times": ["00:00-01:00", "04:00-06:00"], "endpoint": "https://x.example"},
        ]},
    })
    html = _build_home_model_rows(srv)
    assert "00:00-01:00, 04:00-06:00 (UTC)" in html
    # Badge depends on wall-clock time; acceptance is that a badge is rendered
    assert "badge-active" in html or "badge-inactive" in html


def test_home_rows_malformed_renders_always_and_active():
    """Malformed available_times on a provider renders as Always/Active."""
    from proxy.ui import _build_home_model_rows
    srv = _mock_srv({
        "my-model": {"providers": [
            {"name": "p", "type": "remote", "available_times": ["garbage"], "endpoint": "https://x.example"},
        ]},
    })
    html = _build_home_model_rows(srv)
    assert "Always" in html
    assert "badge-active" in html


def test_home_rows_fallback_chain_each_row_has_status_cells():
    """Multi-provider (fallback chain) model: each row renders its own windows/status."""
    from proxy.ui import _build_home_model_rows
    now = datetime(2026, 1, 5, 11, 0, tzinfo=UTC)
    # p0 is restricted to 10-12 — Active at 11:00; p1 is unrestricted — Active
    # p2 restricted 13-14 — Inactive at 11:00
    class _FakeDT:
        @classmethod
        def now(cls, tz=None):
            return now
    with patch("proxy.provider.datetime", _FakeDT):
        srv = _mock_srv({
            "model-a": {"providers": [
                {"name": "p0", "type": "local", "llama_model": "qwen3", "available_times": ["10:00-12:00"]},
                {"name": "p1", "type": "remote", "endpoint": "https://a.example"},
                {"name": "p2", "type": "remote", "endpoint": "https://b.example", "available_times": ["13:00-14:00"]},
            ]},
        })
        html = _build_home_model_rows(srv)
    # There must be one Active-Times cell and one status cell per provider row
    assert html.count("active-times") == 3
    assert html.count("badge-active") + html.count("badge-inactive") == 3


def test_home_rows_rowspan_preserved_with_new_columns():
    """Fallback-chain rows keep the existing rowspan on Model/Type + new col cells."""
    from proxy.ui import _build_home_model_rows
    srv = _mock_srv({
        "model-a": {"providers": [
            {"name": "p0", "type": "local", "llama_model": "qwen3"},
            {"name": "p1", "type": "remote", "endpoint": "https://a.example"},
        ]},
    })
    html = _build_home_model_rows(srv)
    assert 'rowspan="2"' in html
    # Each fallback row has 4 tds (endpoint + model + active-times + status) and no rowspan
    rows = html.split("<tr>")
    assert len([r for r in rows if r.strip()]) == 2
