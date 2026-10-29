"""Offline tests for the per-band snapshot report (LP-0MVOG8TK6002EWZU).

The report is a pure function of two harness run documents, so everything here
is offline: no GPU, no llama-server, no proxy. The assertions cover the numbers
the parent evaluation is decided on — under-load deltas, timeout-exceeded
counts, the 4-retained-snapshot footprint projection against the
``cleanup-slot-cache.sh`` cap, and the token-based benefit verdict — plus the
guard that this analysis tool stays out of the production import graph.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

try:
    from benchmarks import kv_memory_table as kvt
    from benchmarks import slot_snapshot_band_report as r
    from benchmarks import slot_snapshot_benchmark as harness
except ImportError:
    try:
        from proxy.benchmarks import kv_memory_table as kvt
        from proxy.benchmarks import slot_snapshot_band_report as r
        from proxy.benchmarks import slot_snapshot_benchmark as harness
    except ImportError:
        this_dir = Path(__file__).resolve().parent
        root_dir = this_dir.parent.parent
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))
        from proxy.benchmarks import kv_memory_table as kvt
        from proxy.benchmarks import slot_snapshot_band_report as r
        from proxy.benchmarks import slot_snapshot_benchmark as harness

MIB = 1024**2
GIB = 1024**3
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def make_record(
    band: str,
    prompt_tokens: int,
    *,
    timeout_seconds: float | None = None,
    save_seconds: float | None = 1.0,
    restore_seconds: float | None = 10.0,
    save_outcome: str = "success",
    restore_outcome: str = "success",
    save_error: str | None = None,
    restore_error: str | None = None,
    snapshot_bytes: int | None = None,
    cached_tokens: int | None = None,
    populate_seconds: float | None = 40.0,
    snapshot_holds_band: bool | None = None,
) -> dict:
    """A harness record shaped exactly like ``SnapshotRecord.to_json()``."""
    if timeout_seconds is None:
        timeout_seconds = min(3.0 + 0.0015 * prompt_tokens, 60.0)
    if snapshot_bytes is None and restore_outcome != "failed":
        snapshot_bytes = int(kvt.KV_MIB_PER_TOKEN * prompt_tokens * MIB)
    return {
        "band": band,
        "target_tokens": prompt_tokens,
        "session_id": f"snapshot-benchmark-{band}",
        "slot_id": 1,
        "salt": "salt",
        "prompt_tokens_estimated": prompt_tokens,
        "prompt_total_tokens": prompt_tokens,
        "prompt_processed_tokens": (
            None if cached_tokens is None else prompt_tokens - cached_tokens
        ),
        "cached_tokens": cached_tokens,
        "cached_ratio": (
            None if cached_tokens is None else cached_tokens / prompt_tokens
        ),
        "populate_seconds": populate_seconds,
        "populate_outcome": (
            harness.OUTCOME_NOT_REQUESTED
            if populate_seconds is None
            else harness.OUTCOME_SUCCESS
        ),
        "populate_error": None,
        "populate_prompt_total_tokens": prompt_tokens,
        "timeout_seconds": timeout_seconds,
        "save_seconds": save_seconds,
        "save_outcome": save_outcome,
        "save_error": save_error,
        "save_within_timeout": (
            None
            if save_seconds is None
            else save_outcome == "success" and save_seconds <= timeout_seconds
        ),
        "restore_seconds": restore_seconds,
        "restore_outcome": restore_outcome,
        "restore_error": restore_error,
        "restore_within_timeout": (
            None
            if restore_seconds is None
            else restore_outcome == "success"
            and restore_seconds <= timeout_seconds
        ),
        "snapshot_bytes": snapshot_bytes,
        "expected_snapshot_bytes": 0,
        "snapshot_size_ratio": None,
        "snapshot_size_in_tolerance": None,
        "snapshot_holds_band": snapshot_holds_band,
        "under_load": False,
        "verified": cached_tokens is not None,
    }


def make_run(
    records: list[dict],
    *,
    under_load: bool = False,
    populate_slot: bool = True,
    slot_id: int = 1,
    verify_slot_id: int | None = 1,
    load_slot_id: int | None = None,
) -> dict:
    """A harness run document with the header fields the report reads."""
    return {
        "tool": "slot_snapshot_benchmark",
        "schema_version": 3,
        "sentinel": harness.SNAPSHOT_BENCHMARK_SENTINEL,
        "generated_at": "2026-10-28T00:00:00+00:00",
        "dry_run": False,
        "under_load": under_load,
        "verify_chat": True,
        "populate_slot": populate_slot,
        "slot_id": slot_id,
        "verify_slot_id": verify_slot_id,
        "load_slot_id": load_slot_id,
        "populate_timeout_seconds": harness.POPULATE_TIMEOUT_SECONDS,
        "base_url": "http://127.0.0.1:41395",
        "hit_definition": "cached_tokens = prompt_total_tokens - timings.prompt_n",
        "timeout_model": {
            "base_seconds": 3.0,
            "per_token_seconds": 0.0015,
            "cap_seconds": 60.0,
            "cap_tokens": 38_000.0,
        },
        "bands": [],
        "records": records,
        "totals": {},
    }


# ---------------------------------------------------------------------------
# Cleanup-script caps (the footprint comparison's source of truth)
# ---------------------------------------------------------------------------


def test_repository_cleanup_script_matches_the_documented_defaults():
    """Drift guard: the report's documented defaults must equal the script's."""
    caps = r.parse_cleanup_caps(REPO_ROOT / "scripts" / "cleanup-slot-cache.sh")
    assert caps == r.DEFAULT_CLEANUP_CAPS


def test_cleanup_caps_are_parsed_from_an_overridden_script(tmp_path):
    script = tmp_path / "cleanup-slot-cache.sh"
    script.write_text(
        "#!/usr/bin/env bash\nMAX_AGE_DAYS=14\nMAX_SIZE_GB=50\nMIN_AGE_DAYS=2\n",
        encoding="utf-8",
    )
    caps = r.parse_cleanup_caps(script)
    assert caps == {"max_age_days": 14.0, "max_size_gb": 50.0, "min_age_days": 2.0}


def test_missing_cleanup_script_falls_back_to_the_documented_defaults(tmp_path):
    caps = r.parse_cleanup_caps(tmp_path / "absent.sh")
    assert caps == r.DEFAULT_CLEANUP_CAPS


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_load_run_rejects_a_foreign_document(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"tool": "slot_benchmark", "records": []}), "utf-8")
    with pytest.raises(r.ReportError):
        r.load_run(path)


def test_load_run_rejects_missing_and_malformed_documents(tmp_path):
    with pytest.raises(r.ReportError):
        r.load_run(tmp_path / "absent.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(r.ReportError):
        r.load_run(broken)
    header_only = tmp_path / "header.json"
    header_only.write_text(
        json.dumps({"tool": "slot_snapshot_benchmark"}), encoding="utf-8"
    )
    with pytest.raises(r.ReportError):
        r.load_run(header_only)


# ---------------------------------------------------------------------------
# Under-load delta and timeout-exceeded counting (AC2)
# ---------------------------------------------------------------------------


def test_band_row_reports_the_under_load_delta_per_band():
    unloaded = make_record("25k", 25_000, save_seconds=2.0, restore_seconds=10.0)
    loaded = make_record("25k", 25_000, save_seconds=5.5, restore_seconds=16.0)
    row = r.band_row("25k", unloaded, loaded)
    assert row["save_delta_seconds"] == pytest.approx(3.5)
    assert row["restore_delta_seconds"] == pytest.approx(6.0)
    assert row["save_delta_ratio"] == pytest.approx(1.75)
    assert row["restore_delta_ratio"] == pytest.approx(0.6)
    assert row["unloaded"]["restore_seconds"] == 10.0
    assert row["under_load"]["restore_seconds"] == 16.0


def test_delta_is_null_when_a_run_is_missing_the_band():
    row = r.band_row("10k", make_record("10k", 10_000), None)
    assert row["restore_delta_seconds"] is None
    assert row["save_delta_seconds"] is None


def test_timeout_exceeded_counts_only_failures_that_used_the_budget():
    """A timeout is a failure at the budget; a fast 5xx is a plain failure."""
    unloaded = make_record(
        "60k+",
        60_000,
        timeout_seconds=60.0,
        restore_seconds=60.0,
        restore_outcome="failed",
        restore_error="ReadTimeout",
        snapshot_bytes=None,
        cached_tokens=None,
    )
    loaded = make_record(
        "60k+",
        60_000,
        timeout_seconds=60.0,
        save_seconds=0.2,
        save_outcome="failed",
        save_error="HTTP 503: slot busy",
        restore_seconds=None,
        restore_outcome="skipped",
        snapshot_bytes=None,
        cached_tokens=None,
    )
    row = r.band_row("60k+", unloaded, loaded)
    assert row["timeout_exceeded_operations"] == 1
    assert row["failed_operations"] == 2
    assert row["verdict"] == "restore-failed"


def test_build_summary_totals_the_timeout_exceeded_operations():
    unloaded = make_run(
        [
            make_record("10k", 10_000, save_seconds=0.5, restore_seconds=2.0,
                        cached_tokens=9_900),
            make_record(
                "60k+",
                60_000,
                timeout_seconds=60.0,
                restore_seconds=60.0,
                restore_outcome="failed",
                restore_error="ReadTimeout",
                snapshot_bytes=None,
            ),
        ]
    )
    loaded = make_run(
        [
            make_record("10k", 10_000, save_seconds=1.0, restore_seconds=4.0,
                        cached_tokens=9_800),
            make_record(
                "60k+",
                60_000,
                timeout_seconds=60.0,
                save_seconds=60.0,
                save_outcome="failed",
                save_error="ReadTimeout",
                restore_seconds=None,
                restore_outcome="skipped",
                snapshot_bytes=None,
            ),
        ],
        under_load=True,
    )
    summary = r.build_summary(unloaded, loaded)
    assert summary["totals"]["bands"] == 2
    assert summary["totals"]["timeout_exceeded_operations"] == 2


# ---------------------------------------------------------------------------
# Footprint projection against the cleanup cap (AC3)
# ---------------------------------------------------------------------------


def test_projection_scales_one_snapshot_to_four_retained():
    unloaded = make_run([make_record("40k", 40_000, snapshot_bytes=500 * MIB)])
    loaded = make_run(
        [make_record("40k", 40_000, snapshot_bytes=500 * MIB)],
        under_load=True,
    )
    summary = r.build_summary(unloaded, loaded)
    entry = summary["footprint"]["per_band"][0]
    assert entry["snapshot_bytes"] == 500 * MIB
    assert entry["projected_bytes_retained"] == 4 * 500 * MIB
    cap_bytes = int(r.DEFAULT_CLEANUP_CAPS["max_size_gb"] * GIB)
    assert entry["retained_snapshots_until_cap"] == cap_bytes // (500 * MIB)
    assert entry["cleanup_cap_fraction"] == pytest.approx(
        4 * 500 * MIB / cap_bytes
    )
    assert summary["footprint"]["projected_bytes_retained"] == 4 * 500 * MIB


def test_footprint_totals_sum_every_band_and_state_the_cap_semantics():
    records = [
        make_record("10k", 10_000, snapshot_bytes=100 * MIB),
        make_record("25k", 25_000, snapshot_bytes=250 * MIB),
    ]
    summary = r.build_summary(make_run(records), make_run(records, under_load=True))
    footprint = summary["footprint"]
    assert footprint["single_snapshot_total_bytes"] == 350 * MIB
    assert footprint["projected_bytes_retained"] == 4 * 350 * MIB
    assert footprint["cleanup_caps"] == r.DEFAULT_CLEANUP_CAPS
    assert "slot_<session>.bin" in footprint["cleanup_semantics"]
    assert footprint["measured_bands"] == 2


def test_expected_snapshot_bytes_uses_the_shared_kv_table_constant():
    assert r.expected_snapshot_bytes(1_000) == int(
        kvt.KV_MIB_PER_TOKEN * 1_000 * MIB
    )
    assert r.expected_snapshot_bytes(None) == 0
    assert r.expected_snapshot_bytes(0) == 0


# ---------------------------------------------------------------------------
# Token-based benefit verdict (AC4)
# ---------------------------------------------------------------------------


def test_verdict_is_plausible_when_the_restore_fits_and_delivers_the_prefix():
    unloaded = make_record("25k", 25_000, restore_seconds=20.0, cached_tokens=24_500)
    loaded = make_record("25k", 25_000, restore_seconds=32.0, cached_tokens=24_000)
    row = r.band_row("25k", unloaded, loaded)
    assert row["timeout_seconds"] == pytest.approx(40.5)
    assert row["verdict"] == "plausible"
    assert row["cached_ratio"] == pytest.approx(24_500 / 25_000)
    assert row["restore_budget_headroom_seconds"] == pytest.approx(40.5 - 32.0)


def test_verdict_is_latency_bound_when_a_restore_exceeds_the_cap():
    unloaded = make_record("60k+", 60_000, restore_seconds=45.0, cached_tokens=59_000)
    loaded = make_record("60k+", 60_000, restore_seconds=61.0, cached_tokens=59_000)
    row = r.band_row("60k+", unloaded, loaded)
    assert row["timeout_seconds"] == pytest.approx(60.0)
    assert row["verdict"] == "latency-bound"


def test_verdict_is_prefix_not_delivered_when_the_restore_reprocessed_everything():
    unloaded = make_record("40k", 40_000, restore_seconds=5.0, cached_tokens=0)
    row = r.band_row("40k", unloaded, None)
    assert row["verdict"] == "prefix-not-delivered"


def test_verdict_is_unknown_without_a_token_measurement():
    unloaded = make_record("10k", 10_000, restore_seconds=2.0, cached_tokens=None)
    row = r.band_row("10k", unloaded, None)
    assert row["verdict"] == "unknown"


def test_summary_partitions_bands_by_verdict():
    unloaded = make_run(
        [
            make_record("10k", 10_000, restore_seconds=2.0, cached_tokens=9_900),
            make_record("60k+", 60_000, restore_seconds=70.0, cached_tokens=59_000),
            make_record(
                "25k",
                25_000,
                restore_seconds=5.0,
                restore_outcome="failed",
                restore_error="HTTP 503",
                snapshot_bytes=None,
            ),
        ]
    )
    loaded = make_run(unloaded["records"], under_load=True)
    summary = r.build_summary(unloaded, loaded)
    assert summary["totals"]["plausible_bands"] == ["10k"]
    assert summary["totals"]["latency_bound_bands"] == ["60k+"]
    assert summary["totals"]["restore_failed_bands"] == ["25k"]


def test_a_snapshot_that_does_not_hold_the_band_is_never_plausible():
    """A fast restore of the *wrong* KV must not read as a benefit.

    The first full run saved 3 878 KV cells for a 40k prompt (the slot was
    replaced between populate and save), yet the restore was fast and its
    bytes were small — without this check the row would have looked like an
    excellent, cheap, fast band.
    """
    contaminated = make_record(
        "40k",
        40_000,
        restore_seconds=0.03,
        cached_tokens=39_900,
        snapshot_holds_band=False,
    )
    row = r.band_row("40k", contaminated, None)
    assert row["verdict"] == "snapshot-not-band"

    summary = r.build_summary(
        make_run([contaminated]), make_run([contaminated], under_load=True)
    )
    assert summary["totals"]["snapshot_not_band_bands"] == ["40k"]
    assert summary["totals"]["plausible_bands"] == []
    assert "snapshot-not-band" in r.render_markdown(summary)


def test_a_snapshot_that_holds_the_band_is_classified_normally():
    row = r.band_row(
        "40k",
        make_record(
            "40k", 40_000, restore_seconds=2.0, cached_tokens=39_000,
            snapshot_holds_band=True,
        ),
        None,
    )
    assert row["verdict"] == "plausible"


# ---------------------------------------------------------------------------
# Rendering and CLI
# ---------------------------------------------------------------------------


def test_render_markdown_covers_every_band_and_the_caveats():
    unloaded = make_run(
        [
            make_record("10k", 10_000, restore_seconds=2.0, cached_tokens=9_900),
            make_record("25k", 25_000, restore_seconds=20.0, cached_tokens=24_000),
        ]
    )
    loaded = make_run(unloaded["records"], under_load=True, load_slot_id=0)
    markdown = r.render_markdown(r.build_summary(unloaded, loaded))
    for band in ("10k", "25k"):
        assert band in markdown
    assert "Save / restore cost per band" in markdown
    assert "Disk footprint per band" in markdown
    assert "Restore effectiveness" in markdown
    # The caveats that make the numbers honest must survive into the report.
    assert "--populate-slot" in markdown
    assert "RAM prompt cache" in markdown
    assert "one concurrent stream" in markdown
    assert "Measurements bypass the proxy" in markdown
    # Fidelity caveats (mode lock / dispatch lease) are required by the AC.
    assert "No mode lock is held" in markdown
    assert "No live dispatch lease is taken" in markdown
    assert "The measured slot is destroyed" in markdown
    assert "Instance ports churn" in markdown
    # The under-load load slot is shown, so a reader can verify "another slot".
    assert "| under_load |" in markdown


def test_cli_writes_summary_json_and_markdown(tmp_path):
    unloaded_path = tmp_path / "unloaded.json"
    loaded_path = tmp_path / "under_load.json"
    unloaded_path.write_text(
        json.dumps(make_run([make_record("10k", 10_000, cached_tokens=9_900)])),        encoding="utf-8",
    )
    loaded_path.write_text(
        json.dumps(
            make_run(
                [make_record("10k", 10_000, cached_tokens=9_800)],
                under_load=True,
                load_slot_id=0,
            )
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    markdown_path = out_dir / "README.md"
    exit_code = r.main(
        [
            "--unloaded",
            str(unloaded_path),
            "--under-load",
            str(loaded_path),
            "--output-dir",
            str(out_dir),
            "--markdown",
            str(markdown_path),
            "--json",
        ]
    )
    assert exit_code == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["tool"] == "slot_snapshot_band_report"
    assert summary["sentinel"] == r.SNAPSHOT_REPORT_SENTINEL
    assert summary["bands"][0]["band"] == "10k"
    assert summary["runs"]["under_load"]["load_slot_id"] == 0
    assert markdown_path.exists()
    assert "Slot snapshot: cost per prompt-size band" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_cli_rejects_a_foreign_run_document(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"tool": "nope", "records": []}), encoding="utf-8")
    exit_code = r.main(["--unloaded", str(bad), "--under-load", str(bad)])
    assert exit_code == 2
    assert "not a slot_snapshot_benchmark result" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Guard: reporting is an analysis tool, never production code
# ---------------------------------------------------------------------------


def test_report_is_not_imported_by_any_production_module():
    production_modules = (
        "proxy.session",
        "proxy.router",
        "proxy.handlers",
        "proxy.prefill_eta",
        "proxy.lifecycle",
        "proxy.metrics",
    )
    for name in production_modules:
        module = importlib.import_module(name)
        assert not hasattr(module, r.SNAPSHOT_REPORT_SENTINEL), (
            f"{name} imported the snapshot band report — it is an analysis "
            "tool and must stay out of the production import graph"
        )
