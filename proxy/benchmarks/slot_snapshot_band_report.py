#!/usr/bin/env python3
"""Per-band cost / footprint / effectiveness report for snapshot measurements.

Consumes two result documents written by
:mod:`proxy.benchmarks.slot_snapshot_benchmark` — one **unloaded** run and one
**under load** run — and derives the decision-grade summary the parent
evaluation (``LP-0MVN3KPEC007WWL5`` AC1) and the go/no-go decision record need:

* per band, the save and restore wall time in both runs, the **delta** the
  second streaming session adds, and how many save/restore operations
  **exceeded** their adaptive timeout budget;
* the measured snapshot footprint per band, projected to
  :data:`RETAINED_SNAPSHOTS_PROJECTION` retained snapshots, cross-checked
  against ``proxy/benchmarks/kv_memory_table.py`` and compared with the size
  cap of ``scripts/cleanup-slot-cache.sh``;
* the token-based restore effectiveness
  (``cached_tokens = prompt_total - timings.prompt_n``) and therefore a per
  band verdict on whether the band can plausibly benefit *inside* the 60 s
  timeout cap.

This module is **reporting only** — it issues no HTTP, touches no
configuration, and does not re-measure anything. It is an analysis tool like
the harness it reads: no production module may import it
(:data:`SNAPSHOT_REPORT_SENTINEL` guards that).

Usage
-----
::

    # Summarise two runs (writes summary.json + prints the markdown report)
    python3 -m proxy.benchmarks.slot_snapshot_band_report \
        --unloaded proxy/benchmarks/results/<run>/unloaded.json \
        --under-load proxy/benchmarks/results/<run>/under_load.json \
        --output-dir proxy/benchmarks/results/<run> \
        --markdown proxy/benchmarks/results/<run>/README.md

    # Machine-readable only
    python3 -m proxy.benchmarks.slot_snapshot_band_report ... --json

Nothing here runs on the GPU: ``--dry-run``-style planning is unnecessary
because the report is a pure function of the two JSON documents.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from proxy.benchmarks.kv_memory_table import KV_MIB_PER_TOKEN
except ImportError:  # pragma: no cover - depends on the import root
    try:
        from benchmarks.kv_memory_table import KV_MIB_PER_TOKEN
    except ImportError:  # pragma: no cover - exercised when run as a script
        from kv_memory_table import KV_MIB_PER_TOKEN

#: Copied into any module that imports this report, so the AC5-style guard
#: test can prove the report stayed out of the production import graph.
SNAPSHOT_REPORT_SENTINEL = "slot-snapshot-band-report"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEANUP_SCRIPT = PROJECT_ROOT / "scripts" / "cleanup-slot-cache.sh"

GIB = 1024**3
MIB = 1024**2

#: How many retained snapshots the parent decision is about (``pool 2 -> 4``).
RETAINED_SNAPSHOTS_PROJECTION = 4

#: A band "can plausibly benefit" only if the restore actually delivers the
#: prefix: at least this fraction of the prompt must not be reprocessed.
BENEFIT_MIN_CACHED_RATIO = 0.5

#: Documented defaults of ``scripts/cleanup-slot-cache.sh``. The live values are
#: parsed from the script (never hard-coded as the source of truth) so a change
#: to the script shows up in the next report instead of silently drifting.
DEFAULT_CLEANUP_CAPS: dict[str, float] = {
    "max_age_days": 7.0,
    "max_size_gb": 20.0,
    "min_age_days": 1.0,
}

_CLEANUP_DEFAULT_PATTERNS = {
    "max_age_days": re.compile(r"^\s*MAX_AGE_DAYS=(\d+(?:\.\d+)?)\s*$", re.MULTILINE),
    "max_size_gb": re.compile(r"^\s*MAX_SIZE_GB=(\d+(?:\.\d+)?)\s*$", re.MULTILINE),
    "min_age_days": re.compile(r"^\s*MIN_AGE_DAYS=(\d+(?:\.\d+)?)\s*$", re.MULTILINE),
}


class ReportError(ValueError):
    """A run document is missing, malformed, or is not a harness result."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def parse_cleanup_caps(script_path: Path | str | None = None) -> dict[str, float]:
    """Read the ``cleanup-slot-cache.sh`` retention/size defaults.

    Fail-open to :data:`DEFAULT_CLEANUP_CAPS` when the script is unreadable, so a
    report can still be produced from committed run artefacts in a checkout that
    lacks the script.
    """
    path = Path(script_path) if script_path is not None else CLEANUP_SCRIPT
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return dict(DEFAULT_CLEANUP_CAPS)
    caps = dict(DEFAULT_CLEANUP_CAPS)
    for key, pattern in _CLEANUP_DEFAULT_PATTERNS.items():
        match = pattern.search(text)
        if match:
            caps[key] = float(match.group(1))
    return caps


def load_run(path: Path | str) -> dict[str, Any]:
    """Load and validate one harness result document."""
    run_path = Path(path)
    try:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReportError(f"cannot read run document {run_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReportError(f"run document {run_path} is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportError(f"run document {run_path} is not a JSON object")
    if payload.get("tool") != "slot_snapshot_benchmark":
        raise ReportError(
            f"run document {run_path} is not a slot_snapshot_benchmark result "
            f"(tool={payload.get('tool')!r})"
        )
    if not isinstance(payload.get("records"), list):
        raise ReportError(f"run document {run_path} has no records list")
    return payload


def records_by_band(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a run's records by band name (last record wins)."""
    return {
        str(record.get("band")): record
        for record in run.get("records", [])
        if isinstance(record, dict)
    }


# ---------------------------------------------------------------------------
# Derived per-band numbers
# ---------------------------------------------------------------------------


def expected_snapshot_bytes(prompt_tokens: int | None) -> int:
    """Expected snapshot bytes from the shared KV-memory table constant.

    ``KV_MIB_PER_TOKEN`` is the measured q8_0 per-total-ctx-cell KV footprint
    used by the F4 headroom table. The same order-of-magnitude cross-check the
    harness records, restated here so the report can show the ratio for a band
    whose record predates the cross-check field.
    """
    if not prompt_tokens:
        return 0
    return int(KV_MIB_PER_TOKEN * int(prompt_tokens) * MIB)


def _is_timeout_exceeded(record: dict[str, Any], prefix: str) -> bool:
    """Whether *prefix* (``save``/``restore``) failed having used its budget.

    A timeout is a failure whose elapsed time reached the issued timeout (the
    client aborted at the budget). A failure *below* the budget (a 5xx, a busy
    slot rejection) is counted in ``failed_operations`` instead — conflating
    the two would overstate the timeout rate.
    """
    if record.get(f"{prefix}_outcome") != "failed":
        return False
    elapsed = record.get(f"{prefix}_seconds")
    timeout = record.get("timeout_seconds")
    if elapsed is None or timeout is None:
        return False
    return float(elapsed) >= float(timeout)


def _best_cached_ratio(records: Sequence[dict[str, Any]]) -> float | None:
    ratios = [
        float(record["cached_ratio"])
        for record in records
        if isinstance(record.get("cached_ratio"), (int, float))
    ]
    return max(ratios) if ratios else None


def _delta(loaded: float | None, unloaded: float | None) -> float | None:
    if loaded is None or unloaded is None:
        return None
    return round(float(loaded) - float(unloaded), 3)


def _ratio_delta(loaded: float | None, unloaded: float | None) -> float | None:
    if loaded is None or unloaded is None or float(unloaded) == 0:
        return None
    return round((float(loaded) - float(unloaded)) / float(unloaded), 4)


def classify_band(
    *,
    restore_seconds: list[float | None],
    timeout_seconds: float | None,
    cached_ratio: float | None,
    restore_failed: int,
    snapshot_holds_band: bool | None = None,
) -> str:
    """Classify whether a band can plausibly benefit inside the timeout cap."""
    if snapshot_holds_band is False:
        # The saved KV is not this band's prompt (it was replaced between the
        # populate and the save), so every number in the row describes
        # something else. Checked before anything else so a contaminated band
        # can never be reported as a measurement.
        return "snapshot-not-band"
    if restore_failed:
        return "restore-failed"
    if timeout_seconds is None:
        return "unknown"
    measured = [value for value in restore_seconds if value is not None]
    if not measured:
        return "unknown"
    if max(measured) > float(timeout_seconds):
        return "latency-bound"
    if cached_ratio is None:
        return "unknown"
    if cached_ratio >= BENEFIT_MIN_CACHED_RATIO:
        return "plausible"
    return "prefix-not-delivered"


def _queue_seconds(record: dict[str, Any], prefix: str) -> float | None:
    """Wall-clock time spent *waiting* rather than in the server's own work.

    ``wall - server_ms/1000``: a positive value is time the request spent
    queued behind an in-flight task (prefill on another slot), which is the
    difference between "the mechanism is slow" and "the server was busy".
    """
    wall = record.get(f"{prefix}_seconds")
    server_ms = record.get(f"{prefix}_server_ms")
    if not isinstance(wall, (int, float)) or not isinstance(server_ms, (int, float)):
        return None
    return round(max(0.0, float(wall) - float(server_ms) / 1000.0), 3)


def _run_view(record: dict[str, Any] | None) -> dict[str, Any]:
    """The per-run fields the report shows for one band."""
    if record is None:
        return {}
    return {
        "populate_seconds": record.get("populate_seconds"),
        "erase_outcome": record.get("erase_outcome"),
        "save_seconds": record.get("save_seconds"),
        "save_outcome": record.get("save_outcome"),
        "save_server_ms": record.get("save_server_ms"),
        "save_queue_seconds": _queue_seconds(record, "save"),
        "restore_seconds": record.get("restore_seconds"),
        "restore_outcome": record.get("restore_outcome"),
        "restore_server_ms": record.get("restore_server_ms"),
        "restore_queue_seconds": _queue_seconds(record, "restore"),
    }


def band_row(
    band: str,
    unloaded: dict[str, Any] | None,
    loaded: dict[str, Any] | None,
) -> dict[str, Any]:
    """One band's report row (unloaded vs under load, footprint, verdict)."""
    present = [record for record in (unloaded, loaded) if record is not None]
    if not present:
        return {"band": band, "verdict": "missing"}

    primary = unloaded or loaded or {}
    assert primary is not None  # for type checkers; guarded by ``present``
    prompt_tokens = primary.get("prompt_total_tokens")
    timeout_seconds = primary.get("timeout_seconds")
    snapshot_bytes = next(
        (
            int(record["snapshot_bytes"])
            for record in present
            if isinstance(record.get("snapshot_bytes"), int)
        ),
        None,
    )
    cached_ratio = _best_cached_ratio(present)
    cached_tokens = next(
        (
            int(record["cached_tokens"])
            for record in present
            if isinstance(record.get("cached_tokens"), int)
        ),
        None,
    )
    restore_seconds = [
        None
        if record.get("restore_seconds") is None
        else float(record["restore_seconds"])
        for record in present
    ]
    restore_failed = sum(
        1 for record in present if record.get("restore_outcome") == "failed"
    )
    timeout_exceeded = sum(
        _is_timeout_exceeded(record, prefix)
        for record in present
        for prefix in ("save", "restore")
    )
    failed_operations = sum(
        record.get(f"{prefix}_outcome") == "failed"
        for record in present
        for prefix in ("save", "restore")
    )

    expected_bytes = expected_snapshot_bytes(
        prompt_tokens if isinstance(prompt_tokens, int) else None
    )
    projected_bytes = (
        None
        if snapshot_bytes is None
        else snapshot_bytes * RETAINED_SNAPSHOTS_PROJECTION
    )

    return {
        "band": band,
        "target_tokens": primary.get("target_tokens"),
        "prompt_total_tokens": prompt_tokens,
        "timeout_seconds": timeout_seconds,
        "unloaded": _run_view(unloaded),
        "under_load": _run_view(loaded),
        "save_delta_seconds": _delta(
            (loaded or {}).get("save_seconds"), (unloaded or {}).get("save_seconds")
        ),
        "restore_delta_seconds": _delta(
            (loaded or {}).get("restore_seconds"),
            (unloaded or {}).get("restore_seconds"),
        ),
        "save_delta_ratio": _ratio_delta(
            (loaded or {}).get("save_seconds"), (unloaded or {}).get("save_seconds")
        ),
        "restore_delta_ratio": _ratio_delta(
            (loaded or {}).get("restore_seconds"),
            (unloaded or {}).get("restore_seconds"),
        ),
        "timeout_exceeded_operations": timeout_exceeded,
        "failed_operations": failed_operations,
        "snapshot_bytes": snapshot_bytes,
        "snapshot_bytes_on_disk": next(
            (
                record["snapshot_bytes_on_disk"]
                for record in present
                if isinstance(record.get("snapshot_bytes_on_disk"), int)
            ),
            None,
        ),
        "expected_snapshot_bytes": expected_bytes,
        "snapshot_size_ratio": (
            None
            if snapshot_bytes is None or expected_bytes <= 0
            else round(snapshot_bytes / expected_bytes, 4)
        ),
        "projected_bytes_retained": projected_bytes,
        "cached_tokens": cached_tokens,
        "cached_ratio": cached_ratio,
        "restore_budget_headroom_seconds": (
            None
            if timeout_seconds is None or not restore_seconds
            else round(
                float(timeout_seconds) - max(value for value in restore_seconds if value is not None),
                3,
            )
            if any(value is not None for value in restore_seconds)
            else None
        ),
        "verdict": classify_band(
            restore_seconds=restore_seconds,
            timeout_seconds=timeout_seconds,
            cached_ratio=cached_ratio,
            restore_failed=restore_failed,
            snapshot_holds_band=next(
                (
                    record["snapshot_holds_band"]
                    for record in present
                    if isinstance(record.get("snapshot_holds_band"), bool)
                ),
                None,
            ),
        ),
    }


def build_summary(
    unloaded_run: dict[str, Any],
    under_load_run: dict[str, Any],
    *,
    cleanup_caps: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build the full summary document from the two harness runs."""
    caps = dict(cleanup_caps or parse_cleanup_caps())
    unloaded_records = records_by_band(unloaded_run)
    loaded_records = records_by_band(under_load_run)
    bands = list(dict.fromkeys([*unloaded_records, *loaded_records]))
    rows = [
        band_row(band, unloaded_records.get(band), loaded_records.get(band))
        for band in bands
    ]

    cap_bytes = int(caps["max_size_gb"] * GIB)
    measured = [row for row in rows if isinstance(row.get("snapshot_bytes"), int)]
    single_snapshot_bytes = sum(row["snapshot_bytes"] for row in measured)
    projected_bytes = sum(
        row["projected_bytes_retained"] for row in measured
    )
    per_band_retention = [
        {
            "band": row["band"],
            "snapshot_bytes": row["snapshot_bytes"],
            "snapshot_mib": round(row["snapshot_bytes"] / MIB, 1),
            "projected_bytes_retained": row["projected_bytes_retained"],
            "projected_gib_retained": round(
                row["projected_bytes_retained"] / GIB, 2
            ),
            "cleanup_cap_fraction": (
                round(row["projected_bytes_retained"] / cap_bytes, 8)
                if cap_bytes
                else None
            ),
            "retained_snapshots_until_cap": (
                int(cap_bytes // row["snapshot_bytes"])
                if row["snapshot_bytes"]
                else None
            ),
        }
        for row in measured
    ]

    return {
        "tool": "slot_snapshot_band_report",
        "schema_version": 1,
        "sentinel": SNAPSHOT_REPORT_SENTINEL,
        "generated_at": datetime.now(UTC).isoformat(),
        "retained_snapshots_projection": RETAINED_SNAPSHOTS_PROJECTION,
        "benefit_min_cached_ratio": BENEFIT_MIN_CACHED_RATIO,
        "kv_mib_per_token": KV_MIB_PER_TOKEN,
        "runs": {
            "unloaded": _run_header(unloaded_run),
            "under_load": _run_header(under_load_run),
        },
        "bands": rows,
        "footprint": {
            "cleanup_caps": caps,
            "cleanup_cap_bytes": cap_bytes,
            "cleanup_cap_gib": round(cap_bytes / GIB, 2),
            "measured_bands": len(measured),
            "single_snapshot_total_bytes": single_snapshot_bytes,
            "single_snapshot_total_mib": round(single_snapshot_bytes / MIB, 1),
            "projected_bytes_retained": projected_bytes,
            "projected_gib_retained": round(projected_bytes / GIB, 2),
            "projected_cleanup_cap_fraction": (
                round(projected_bytes / cap_bytes, 8) if cap_bytes else None
            ),
            "per_band": per_band_retention,
            "cleanup_semantics": (
                "cleanup-slot-cache.sh keeps one snapshot per session "
                "(slot_<session>.bin rewritten on every save), deletes files "
                "older than max_age_days, and if the directory still exceeds "
                "max_size_gb deletes the oldest files that are older than "
                "min_age_days. It caps the whole slot-cache directory, not a "
                "per-band quota."
            ),
        },
        "totals": {
            "bands": len(rows),
            "timeout_exceeded_operations": sum(
                row.get("timeout_exceeded_operations", 0) for row in rows
            ),
            "failed_operations": sum(row.get("failed_operations", 0) for row in rows),
            "plausible_bands": [row["band"] for row in rows if row["verdict"] == "plausible"],
            "latency_bound_bands": [
                row["band"] for row in rows if row["verdict"] == "latency-bound"
            ],
            "restore_failed_bands": [
                row["band"] for row in rows if row["verdict"] == "restore-failed"
            ],
            "snapshot_not_band_bands": [
                row["band"] for row in rows if row["verdict"] == "snapshot-not-band"
            ],
            "unknown_bands": [
                row["band"]
                for row in rows
                if row["verdict"] in ("unknown", "missing", "prefix-not-delivered")
            ],
        },
    }


def _run_header(run: dict[str, Any]) -> dict[str, Any]:
    return {
        key: run.get(key)
        for key in (
            "generated_at",
            "base_url",
            "under_load",
            "verify_chat",
            "populate_slot",
            "slot_id",
            "verify_slot_id",
            "load_slot_id",
            "timeout_model",
            "totals",
        )
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _seconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):.2f}"


def _signed(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):+.2f}"


def _pct(value: Any, digits: int = 1) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _ms(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):.1f}"


def _bytes_label(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    value = float(value)
    for unit, scale in (("GiB", GIB), ("MiB", MIB), ("KiB", 1024)):
        if value >= scale:
            return f"{value / scale:.1f} {unit}"
    return f"{value:.0f} B"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the human-readable report (committed next to the raw runs)."""
    runs = summary["runs"]
    footprint = summary["footprint"]
    totals = summary["totals"]
    lines: list[str] = []
    lines.append("# Slot snapshot: cost per prompt-size band (unloaded and under load)")
    lines.append("")
    lines.append(
        "Generated by `proxy/benchmarks/slot_snapshot_band_report.py` from two "
        "`slot_snapshot_benchmark` runs. Machine-readable summary: "
        "`summary.json`; raw runs: `unloaded.json`, `under_load.json`."
    )
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    lines.append("| run | generated | populate slot | measured slot | verify slot | load slot |")
    lines.append("|---|---|---|---|---|---|")
    for name in ("unloaded", "under_load"):
        run = runs.get(name) or {}
        lines.append(
            "| {name} | {generated} | {populate} | {slot} | {verify} | {load} |".format(
                name=name,
                generated=run.get("generated_at") or "—",
                populate="yes" if run.get("populate_slot") else "no",
                slot=run.get("slot_id"),
                verify=run.get("verify_slot_id")
                if run.get("verify_slot_id") is not None
                else "any",
                load=run.get("load_slot_id")
                if run.get("load_slot_id") is not None
                else "any",
            )
        )
    lines.append("")
    lines.append("## Save / restore cost per band")
    lines.append("")
    lines.append(
        "| band | prompt tokens | timeout (s) | fill (s) | save (s) | restore (s) "
        "| save under load (s) | restore under load (s) | Δ save (s) | Δ restore (s) "
        "| save/restore outcome | timeouts exceeded |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"
    )
    for row in summary["bands"]:
        unloaded = row.get("unloaded") or {}
        loaded = row.get("under_load") or {}
        lines.append(
            "| {band} | {tokens} | {timeout} | {fill} | {save} | {restore} | "
            "{save_l} | {restore_l} | {d_save} | {d_restore} | {outcomes} | {timeouts} |".format(
                band=row["band"],
                tokens=row.get("prompt_total_tokens"),
                timeout=_seconds(row.get("timeout_seconds")),
                fill=_seconds(unloaded.get("populate_seconds")),
                save=_seconds(unloaded.get("save_seconds")),
                restore=_seconds(unloaded.get("restore_seconds")),
                save_l=_seconds(loaded.get("save_seconds")),
                restore_l=_seconds(loaded.get("restore_seconds")),
                d_save=_signed(row.get("save_delta_seconds")),
                d_restore=_signed(row.get("restore_delta_seconds")),
                outcomes="{}/{} then {}/{}".format(
                    unloaded.get("save_outcome"),
                    unloaded.get("restore_outcome"),
                    loaded.get("save_outcome"),
                    loaded.get("restore_outcome"),
                ),
                timeouts=row.get("timeout_exceeded_operations"),
            )
        )
    lines.append("")
    lines.append("## Timeout attribution (server work vs queueing)")
    lines.append("")
    lines.append(
        "`server ms` is llama-server's own `timings.save_ms`/`restore_ms`; "
        "`queued s` is wall clock minus that (time waiting behind an in-flight "
        "task on another slot). A timeout with a large `queued s` is the server "
        "being busy, not the snapshot mechanism being slow."
    )
    lines.append("")
    lines.append(
        "| band | save server (ms) | save queued (s) | restore server (ms) "
        "| restore queued (s) | under-load save server (ms) | under-load save queued (s) "
        "| under-load restore server (ms) | under-load restore queued (s) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["bands"]:
        unloaded = row.get("unloaded") or {}
        loaded = row.get("under_load") or {}
        lines.append(
            "| {band} | {ss} | {sq} | {rs} | {rq} | {ss_l} | {sq_l} | {rs_l} | {rq_l} |".format(
                band=row["band"],
                ss=_ms(unloaded.get("save_server_ms")),
                sq=_seconds(unloaded.get("save_queue_seconds")),
                rs=_ms(unloaded.get("restore_server_ms")),
                rq=_seconds(unloaded.get("restore_queue_seconds")),
                ss_l=_ms(loaded.get("save_server_ms")),
                sq_l=_seconds(loaded.get("save_queue_seconds")),
                rs_l=_ms(loaded.get("restore_server_ms")),
                rq_l=_seconds(loaded.get("restore_queue_seconds")),
            )
        )
    lines.append("")
    lines.append("## Disk footprint per band")
    lines.append("")
    lines.append(
        "| band | snapshot | on disk | expected (kv_memory_table) | measured/expected "
        "| projected, {n} retained | share of cleanup cap | snapshots until cap |".format(
            n=summary["retained_snapshots_projection"]
        )
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    per_band = {entry["band"]: entry for entry in footprint["per_band"]}
    for row in summary["bands"]:
        entry = per_band.get(row["band"])
        if entry is None:
            lines.append(
                f"| {row['band']} | — | — | — | — | — | — | — |"
            )
            continue
        lines.append(
            "| {band} | {snapshot} | {on_disk} | {expected} | {ratio} | {projected} | {share} | {until} |".format(
                band=row["band"],
                snapshot=_bytes_label(entry["snapshot_bytes"]),
                on_disk=_bytes_label(row.get("snapshot_bytes_on_disk")),
                expected=_bytes_label(row.get("expected_snapshot_bytes")),
                ratio="—"
                if row.get("snapshot_size_ratio") is None
                else f"{row['snapshot_size_ratio']:.2f}x",
                projected=_bytes_label(entry["projected_bytes_retained"]),
                share=_pct(entry.get("cleanup_cap_fraction"), 4),
                until=entry.get("retained_snapshots_until_cap"),
            )
        )
    lines.append("")
    lines.append(
        "Cleanup cap: **{cap} GiB** (`max_size_gb`), age rule {age} days, "
        "protect-recent {protect} days. All {bands} bands retained at {n} "
        "snapshots each = **{total}**, {share} of the cap.".format(
            cap=footprint["cleanup_cap_gib"],
            age=footprint["cleanup_caps"].get("max_age_days"),
            protect=footprint["cleanup_caps"].get("min_age_days"),
            bands=footprint["measured_bands"],
            n=summary["retained_snapshots_projection"],
            total=_bytes_label(footprint["projected_bytes_retained"]),
            share=_pct(footprint.get("projected_cleanup_cap_fraction"), 4),
        )
    )
    lines.append("")
    lines.append("## Restore effectiveness and per-band verdict")
    lines.append("")
    lines.append(
        "| band | cached tokens | cached ratio | restore headroom (s) | verdict |"
    )
    lines.append("|---|---:|---:|---:|---|")
    for row in summary["bands"]:
        lines.append(
            "| {band} | {tokens} | {ratio} | {headroom} | {verdict} |".format(
                band=row["band"],
                tokens=row.get("cached_tokens")
                if row.get("cached_tokens") is not None
                else "—",
                ratio=_pct(row.get("cached_ratio")),
                headroom=_signed(row.get("restore_budget_headroom_seconds")),
                verdict=row.get("verdict"),
            )
        )
    lines.append("")
    lines.append(
        "Verdict `plausible` = every restore finished inside its adaptive "
        "timeout **and** at least {min_ratio} of the prompt was served from the "
        "restored KV. `latency-bound` = a restore exceeded its timeout budget. "
        "`restore-failed` = a restore did not complete at all. "
        "`snapshot-not-band` = the saved KV did not hold this band's prompt "
        "(the slot was replaced between populate and save), so the row is "
        "recorded as invalid rather than read as a measurement.".format(
            min_ratio=_pct(summary["benefit_min_cached_ratio"], 0)
        )
    )
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- bands measured: {totals['bands']}")
    lines.append(
        f"- timeout-exceeded save/restore operations: "
        f"{totals['timeout_exceeded_operations']}"
    )
    lines.append(f"- failed save/restore operations: {totals['failed_operations']}")
    lines.append(
        f"- bands that can plausibly benefit inside the cap: "
        f"{', '.join(totals['plausible_bands']) or 'none'}"
    )
    lines.append(
        f"- latency-bound bands: {', '.join(totals['latency_bound_bands']) or 'none'}"
    )
    lines.append(
        "- invalid (snapshot did not hold the band's prompt): "
        f"{', '.join(totals['snapshot_not_band_bands']) or 'none'}"
    )
    lines.append(
        f"- unknown / prefix-not-delivered bands: "
        f"{', '.join(totals['unknown_bands']) or 'none'}"
    )
    lines.append("")
    lines.append("## Reading the numbers")
    lines.append("")
    lines.append(
        "- **Populate before save is mandatory for these numbers.** Without it "
        "the save captures an almost-empty slot, so `snapshot_bytes` and "
        "`cached_tokens` say nothing about a prompt-sized session. Both runs "
        "here use `--populate-slot`, with the measured slot pinned."
    )
    lines.append(
        "- **`cached_tokens` cannot separate a restored KV from llama-server's "
        "RAM prompt cache** (`--cache-ram`): both surface as a small "
        "`timings.prompt_n`. The token metric proves the prefix was not "
        "recomputed; the *latency* comparison against the adaptive timeout is "
        "what decides whether the band benefits inside the cap."
    )
    lines.append(
        "- **\"Under load\" is one concurrent stream** pinned to the other "
        "slot. It reproduces \"saves issue while another slot is processing\", "
        "not the full production contention mix (that is the bake-off replay)."
    )
    lines.append(
        "- **Measurements bypass the proxy.** Save/restore go straight to "
        "llama-server, so `session_slot_skip_when_busy`, the context cap, the "
        "dispatch lease and the failure cooldown are *not* in the measured "
        "path; the replay/log-analysis children cover those exclusions."
    )
    lines.append(
        "- **Timeout attribution matters.** A wall-clock timeout is reported "
        "next to llama-server's own `timings.save_ms`/`restore_ms`. Where the "
        "gap is the whole budget, the operation waited for a busy server "
        "(another slot's prefill), which is a *scheduling* limit rather than a "
        "snapshot-mechanism limit."
    )
    lines.append("")
    lines.append("## Fidelity caveats")
    lines.append("")
    lines.append(
        "Recorded explicitly, following the convention of "
        "`docs/slot-count-verdict-lp-0mv9ungeb009xuz5.md`: a number is only "
        "usable once the condition it was taken under is stated."
    )
    lines.append("")
    lines.append(
        "- **No mode lock is held.** Save/restore are issued straight at "
        "llama-server; the run takes no mode-switch lock "
        "(`LP-0MVCBMLP7009MN81`). A mode switch during the window restarts the "
        "backend and would void the affected band, so the time each band was "
        "measured is recorded above (`Runs` table) for correlation against the "
        "proxy log. It is a recording, not a protection."
    )
    lines.append(
        "- **No live dispatch lease is taken.** The proxy's per-endpoint "
        "dispatch lease / slot-ownership registry is not consulted and cannot "
        "see this harness, so a live session can be dispatched onto the "
        "measured slot mid-band. Nothing here proves the host was quiet; the "
        "`queued s` column in *Timeout attribution* is the observable that "
        "exposes such interference (wall clock above the server's own "
        "`save_ms`/`restore_ms`)."
    )
    lines.append(
        "- **The measured slot is destroyed.** `--populate-slot` erases and "
        "then prefills the measured slot, so any live session that owned that "
        "slot loses its cached KV and re-prefills on return. That is a real "
        "cost of measuring on the production host, not a neutral probe."
    )
    lines.append(
        "- **Instance ports churn; the router port does not.** The runs above "
        "address `base_url` = `{base}`. On the production host the per-model "
        "llama-server instances are recycled by the router onto new ephemeral "
        "ports, so a direct instance URL goes stale mid-investigation; slot "
        "actions therefore carry the model name and go through the router, "
        "which is also the path the proxy itself uses.".format(
            base=(runs.get("unloaded") or {}).get("base_url")
            or (runs.get("under_load") or {}).get("base_url")
            or "—"
        )
    )
    lines.append("")
    return "\n".join(lines)


def write_report(
    summary: dict[str, Any],
    output_dir: Path | str | None = None,
    markdown_path: Path | str | None = None,
) -> tuple[Path | None, Path | None]:
    """Write ``summary.json`` and optionally the markdown report."""
    json_path = None
    md_path = None
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / "summary.json"
        json_path.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    if markdown_path is not None:
        md_path = Path(markdown_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(summary), encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise two slot_snapshot_benchmark runs (unloaded and under "
            "load) into a per-band cost / footprint / effectiveness report."
        )
    )
    parser.add_argument(
        "--unloaded", required=True, help="Path to the unloaded run JSON"
    )
    parser.add_argument(
        "--under-load", required=True, help="Path to the under-load run JSON"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write summary.json into (default: none)",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="Path to write the markdown report to (default: none)",
    )
    parser.add_argument(
        "--cleanup-script",
        default=None,
        help=(
            "Path to cleanup-slot-cache.sh to read the retention/size defaults "
            "from (default: the repository copy)"
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the summary JSON on stdout"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        unloaded = load_run(args.unloaded)
        loaded = load_run(args.under_load)
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = build_summary(
        unloaded,
        loaded,
        cleanup_caps=parse_cleanup_caps(args.cleanup_script),
    )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(render_markdown(summary))
    json_path, md_path = write_report(
        summary, args.output_dir, args.markdown
    )
    if json_path is not None and not args.json:
        print(f"\nsummary written to {json_path}")
    if md_path is not None and not args.json:
        print(f"report written to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
