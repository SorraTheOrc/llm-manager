"""Tests for the F4 KV-cache headroom table (LP-0MSC95W3T000CCYC).

Validates the per-config per-slot KV-memory accounting emitted by
``proxy/benchmarks/kv_memory_table.py`` against the measured values
recorded in the F4 findings comment (LP-C0MSEGJO5A006TV05):

- KV total is total-ctx bound, not slot-count bound (q8_0, 10 layers):
  131072 total ctx -> 1362.7 MiB; 262144 -> ~2720-2725 MiB.
- Per-slot KV = per-token KV cost x per-slot ctx (454 MiB/slot at 6x43.7K,
  1360 MiB/slot at 2x131K).
- ~87 GiB available; headroom ~60 GiB at every config.
"""

import json

import pytest


def _import_module():
    import sys
    from pathlib import Path

    try:
        import proxy.benchmarks.kv_memory_table as m
        return m
    except ImportError:
        pass
    try:
        from benchmarks import kv_memory_table as m
        return m
    except ImportError:
        pass
    this_dir = Path(__file__).resolve().parent
    root_dir = this_dir.parent.parent  # repo root
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    try:
        import proxy.benchmarks.kv_memory_table as m
        return m
    except ImportError:
        return None


M = _import_module()
pytestmark = pytest.mark.skipif(M is None, reason="kv_memory_table module not importable")


def _samples(module=M):
    return {row["config"]: row for row in module.build_table()}


def test_defines_f4_constants():
    """Constants must exist and match the measured F4 values."""
    # KV q8_0: 131072 total ctx -> 1362.7 MiB (measured, llama-server logs)
    assert M.KV_MIB_PER_TOKEN == pytest.approx(1362.7 / 131072, rel=1e-3)
    assert M.MODEL_GIB == pytest.approx(24.7, rel=1e-3)
    assert M.AVAILABLE_GIB == pytest.approx(87.0, rel=1e-3)


def test_all_five_candidate_configs_present():
    """The audit-required table covers every evaluated config."""
    table = M.build_table()
    labels = {row["config"] for row in table}
    assert {"8x32.8K", "6x43.7K", "4x65.5K", "3x87.4K", "2x131K"} <= labels
    # Live baseline is included for comparison.
    assert any("3x43.7K" in label for label in labels)


def test_per_slot_kv_matches_measured_values():
    """Per-slot KV must reproduce the measured F4 splits (454 @6x43.7K, 1360 @2x131K)."""
    samples = _samples()
    assert samples["6x43.7K"]["per_slot_kv_mib"] == pytest.approx(454, abs=3)
    assert samples["2x131K"]["per_slot_kv_mib"] == pytest.approx(1360, abs=3)
    # 4 total slots x ~65.5K -> double the 8-slot per-slot share.
    assert samples["8x32.8K"]["per_slot_kv_mib"] == pytest.approx(
        samples["4x65.5K"]["per_slot_kv_mib"] / 2, abs=2
    )


def test_total_kv_is_ctx_bound_not_slot_bound():
    """All 262144-ctx configs share the same total KV (~2720-2725 MiB)."""
    samples = _samples()
    totals = [samples[label]["total_kv_mib"] for label in ("8x32.8K", "6x43.7K", "4x65.5K", "3x87.4K", "2x131K")]
    for t in totals:
        assert 2720 <= t <= 2726, f"total KV {t} outside measured 2720-2725 MiB range"
    assert max(totals) - min(totals) <= 6  # ctx-bound: identical across slot splits
    # 131072-ctx baseline total is half of the 262144-ctx total.
    baseline = next(row for label, row in _samples().items() if "3x43.7K" in label)
    assert baseline["total_kv_mib"] == pytest.approx(1362.7, abs=3)


def test_headroom_at_least_59_gib_everywhere():
    """~87 GiB available minus model+KV leaves >= ~59 GiB at every config."""
    for row in M.build_table():
        assert row["headroom_gib"] >= 59.0, row


def test_baseline_headroom_matches_f4_table():
    """Live 3x43.7K baseline headroom ~61.0 GiB (F4 comment table)."""
    baseline = next(row for row in M.build_table() if "3x43.7K" in row["config"])
    assert baseline["headroom_gib"] == pytest.approx(61.0, abs=0.2)


def test_json_output_round_trips_table():
    """The --json CLI output carries the same rows as build_table()."""
    out = M._render_json(M.build_table())
    data = json.loads(out)
    assert len(data["configs"]) == len(M.build_table())
    assert data["configs"][0]["config"]
    assert "per_slot_kv_mib" in data["configs"][0]


def test_markdown_table_has_headers_and_rows():
    """The markdown table is a proper per-config table (audit F4 AC3)."""
    md = M._render_markdown(M.build_table())
    assert "| Config" in md
    assert "| per-slot KV" in md
    assert "| headroom" in md
    assert md.count("| ---") >= 1
