#!/usr/bin/env python3
"""Artefact contract for the diurnally matched pre-change baseline record.

LP-0MVV29XDW006VTWK restates the abandoned-queue-wait baseline on a 24 h
pre-change window that **ends exactly at the contended-pricing deploy**
(`52085f09` served from `2026-10-29 00:10:18` UTC). Two properties make that
restatement different from the historical 19:00→19:00 baseline, and both are
invisible in a prose-only document:

1. **the window boundaries are the deploy boundary** — a window that starts or
   ends elsewhere is not diurnally matched to the post-change record and may
   include post-change lines, so the "pre-change code" label would be false;
2. **the numbers are the harness's, not typed by hand** — the totals must
   decompose arithmetically over the per-band and per-trigger splits the
   harness emits, so a hand-edited JSON is caught here rather than in an
   operator's analysis.

The document is also pinned to the artefacts it quotes: its relative links must
resolve, the committed share must appear in the text, and it must name every
log archive the committed JSON says contributed (the window's early hours were
pruned by 7-day retention before the record was committed, so the coverage
behind a number is part of the contract — `LP-0MVV1V213002QAIR`).

No production log, no network: the committed artefact is read as data.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs" / "dev"
DOC_PATH = DOCS_DIR / "prefill-admission-conversion-diurnal-baseline.md"
ARTEFACT_PATH = DOCS_DIR / "prefill-admission-conversion-diurnal-baseline.json"

#: The deploy boundary: ``52085f09`` started serving at this instant, and the
#: window must end here so that every line inside it is pre-change code.
DEPLOY_AT = datetime(2026, 10, 29, 0, 10, 18)
WINDOW_START = datetime(2026, 10, 28, 0, 10, 18)
WINDOW_HOURS = 24


def _artefact() -> dict:
    return json.loads(ARTEFACT_PATH.read_text())


def test_window_is_the_deploy_anchored_pre_change_window():
    """The artefact measures exactly the diurnally matched pre-change window."""
    document = _artefact()
    since = datetime.fromisoformat(document["since"])
    until = datetime.fromisoformat(document["until"])

    assert since == WINDOW_START, (
        f"window starts {since}, not the deploy-anchored pre-change instant "
        f"{WINDOW_START}"
    )
    assert until == DEPLOY_AT, (
        f"window ends {until}, not the deploy boundary {DEPLOY_AT}: a window "
        "ending elsewhere is not fully pre-change code"
    )
    assert (until - since).total_seconds() == WINDOW_HOURS * 3600, (
        f"window is {(until - since).total_seconds() / 3600} h, not "
        f"{WINDOW_HOURS} h — the per-hour rate columns would not be comparable"
    )


def test_totals_decompose_over_bands_and_triggers():
    """Every published total is the sum the harness would recompute.

    Pins "no hand-typed numbers": the share is the harness's round(4) of
    abandoned / outcomes, the totals are the band sums, and the trigger split
    (positive evidence only — the pre-change shape carries no
    ``trigger=concurrency_limit`` token) adds back up.
    """
    document = _artefact()
    totals = document["totals"]
    bands = document["bands"]

    assert totals["queued_waits"] == (
        totals["queue_dispatched"] + totals["queue_abandoned"]
    )
    assert totals["abandoned_share"] == round(
        totals["queue_abandoned"] / totals["queued_waits"], 4
    )

    for key in ("admit", "reject"):
        assert totals["admission_verdicts"][key] == sum(
            band["admission_verdicts"][key] for band in bands.values()
        ), f"total {key} verdicts do not equal the per-band sum"

    for total_key, band_key in (
        ("queue_dispatched", "queue_dispatched"),
        ("queue_abandoned", "queue_abandoned"),
        ("queued_waits", "queued_waits"),
        ("queue_wait_seconds", "queue_wait_seconds"),
        ("evictions", "evictions"),
    ):
        band_sum = sum(band[band_key] for band in bands.values())
        if total_key == "queue_wait_seconds":
            assert abs(totals[total_key] - band_sum) < 0.5, total_key
        else:
            assert totals[total_key] == band_sum, (
                f"{total_key} is {totals[total_key]}, bands sum to {band_sum}"
            )

    for split in ("queue_dispatched_by_trigger", "queue_abandoned_by_trigger"):
        per_band: dict[str, int] = {}
        for band in bands.values():
            for trigger, count in band[split].items():
                per_band[trigger] = per_band.get(trigger, 0) + count
        assert per_band == totals[split], f"{split} does not match the band sum"

    # The enqueue denominator is auditable: priced enqueues + unbanded ones.
    assert document["queued_waits_enqueued"] == (
        document["queued_waits_enqueued_unbanded"]
        + sum(band["queue_enqueued"] for band in bands.values())
    )
    assert document["queued_waits_enqueued"] >= totals["queued_waits"]


def test_contributing_archives_are_a_subset_of_the_scanned_inventory():
    """Provenance is intact: what contributed was actually scanned."""
    document = _artefact()
    contributing = document["log_files"]
    scanned = set(document["log_files_scanned"])

    assert contributing, "the artefact records no contributing log archive"
    assert set(contributing) <= scanned, (
        "contributing archives missing from log_files_scanned: "
        f"{sorted(set(contributing) - scanned)}"
    )
    assert document["queue_banding"], "the queue-wait banding provenance is gone"


def test_document_quotes_the_committed_share_and_names_its_archives():
    """The doc quotes the artefact's own numbers and the files behind them."""
    text = DOC_PATH.read_text()
    document = _artefact()
    share = f"{document['totals']['abandoned_share'] * 100:.1f} %"

    assert share in text, (
        f"the doc does not quote the committed abandoned share {share} "
        "(a hand-edited share would go unnoticed)"
    )
    for name in document["log_files"]:
        assert name in text, (
            f"the doc does not name contributing archive {name}: an operator "
            "cannot tell which slice of the window the numbers cover"
        )


def test_document_links_resolve():
    """Every relative doc/artefact link in the record points at a real file."""
    text = DOC_PATH.read_text()
    links = re.findall(r"\]\(([^)#\s]+\.(?:md|json))\)", text)

    assert links, "the record no longer links any doc or artefact"
    for link in links:
        if link.startswith(("http://", "https://")):
            continue
        target = (DOC_PATH.parent / link).resolve()
        assert target.is_file(), f"broken link: {link}"


def test_retained_historical_reference_and_bar_are_stated():
    """The record keeps the 36.9 % reference and does not move the 85 % bar."""
    text = DOC_PATH.read_text()

    assert "36.9 %" in text, (
        "the historical 19:00→19:00 reference (36.9 %) must be retained "
        "alongside the diurnally matched baseline"
    )
    assert "85 %" in text, (
        "the parent's literal 85 % bar must be stated as unchanged by this "
        "restatement"
    )
    assert "52.6 %" in text, (
        "the diurnally matched working baseline (52.6 %) is missing from the "
        "record"
    )
