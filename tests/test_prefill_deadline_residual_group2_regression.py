#!/usr/bin/env python3
"""Artefact contract for the group-2 regression check (parent AC6).

LP-0MW8EAV91002W5IN answers one question of the group-2 validation: did group 2
buy its residual reduction by stalling more `prefill_deadline` traffic, or by
collapsing local coverage? Both readings are only meaningful if the two windows
are the *committed* ones and the numbers are the tools', not typed by hand, so
this module pins four properties of
``docs/dev/prefill-deadline-residual-group2-regression.json`` and the
"Regression check" section of the verdict report that quotes it:

1. **the windows are the committed analyser windows** — a comparison against a
   different group-1 window (or a different group-1 tool) is not a regression
   comparison, so both boundaries are read back from the committed datasets;
2. **the `prefill_deadline` figures are the analyser datasets'** — the count,
   the user/audit split, the user-class rate and the combined (b2) figure must
   match the committed datasets, so a hand-edited artefact fails here;
3. **the coverage arithmetic holds** — local share decomposes over the
   local/remote request counts, the fallback rate over the fallback events, the
   admission refusals over their per-reason split, and the producer's threshold
   (local share >= group-1 share - 5 pp) is recomputed rather than trusted;
4. **the document quotes the verdicts it computes** — the met/not-met
   statements, the threshold floor and the honest `no_progress_watchdog`
   counterpoint must survive in the report, because a regression check that
   silently drops its caveat is worse than none.

No production log and no network: the committed artefact and datasets are read
as data.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs" / "dev"
ARTEFACT_PATH = DOCS_DIR / "prefill-deadline-residual-group2-regression.json"
VERDICT_PATH = DOCS_DIR / "prefill-deadline-residual-group2-verdict.md"
GROUP2_DATASET_PATH = DOCS_DIR / "prefill-deadline-residual-group2-24h.json"
GROUP1_DATASET_PATH = REPO_ROOT / "proxy" / "docs" / "validation" / "group1-prefill-deadline-residual.json"

#: Producer decision 2026-11-08 (Appendix Q2): the allowed drop in local share.
LOCAL_SHARE_ALLOWANCE_PP = 5.0


def _artefact() -> dict:
    return json.loads(ARTEFACT_PATH.read_text())


def _dataset(path: Path) -> dict:
    return json.loads(path.read_text())


def _counts(dataset: dict) -> dict:
    """The `prefill_deadline` and combined (b2) figures from an analyser run."""
    hours = dataset["window"]["hours"]
    split = dataset["reason_split"]["prefill_deadline"]
    deficit = dataset["reason_split"]["measured_cache_deficit"]
    combined_user = split["user"] + deficit["user"]
    return {
        "count": dataset["evictions"]["by_reason"].get("prefill_deadline", 0),
        "user": split["user"],
        "audit": split["audit"],
        "user_per_hour": round(split["user_per_hour"], 3),
        "combined_b2_user": combined_user,
        "combined_b2_user_per_hour": round(combined_user / hours, 3),
    }


def test_the_two_windows_are_the_committed_analyser_windows():
    """Both windows are the datasets' own boundaries, not a re-chosen span."""
    artefact = _artefact()
    for window, path in (
        ("group1", GROUP1_DATASET_PATH),
        ("group2", GROUP2_DATASET_PATH),
    ):
        committed = _dataset(path)["window"]
        assert artefact["windows"][window] == {
            "start": committed["start"],
            "end": committed["end"],
            "hours": committed["hours"],
        }, (
            f"{window} window is {artefact['windows'][window]}, not the committed "
            f"analyser window {committed} — a comparison across a different "
            "window is not a regression comparison"
        )


def test_prefill_deadline_figures_are_the_analyser_datasets_figures():
    """The counted reason, its user/audit split and (b2) come from the runs."""
    artefact = _artefact()["prefill_deadline"]
    for window, path in (
        ("group1", GROUP1_DATASET_PATH),
        ("group2", GROUP2_DATASET_PATH),
    ):
        expected = _counts(_dataset(path))
        recorded = {key: artefact[window][key] for key in expected}
        assert recorded == expected, (
            f"{window} figures {recorded} do not match the committed analyser "
            f"dataset {path.name} ({expected}) — the artefact must be the "
            "tool's output, not a hand-typed restatement"
        )


def test_prefill_deadline_did_not_increase():
    """AC1's met/not-met test: the count must not rise beyond group 1."""
    artefact = _artefact()["prefill_deadline"]
    group1 = artefact["group1"]["count"]
    group2 = artefact["group2"]["count"]

    assert artefact["increased"] is (group2 > group1), (
        f"`increased` is {artefact['increased']} but the counts are "
        f"{group1} -> {group2}"
    )
    assert artefact["increased"] is False, (
        f"`prefill_deadline` rose {group1} -> {group2}: group 2 bought its "
        "residual reduction by stalling more deadline traffic (AC1 not met)"
    )
    assert artefact["group2"]["user_per_hour"] <= artefact["group1"]["user_per_hour"]


def test_coverage_arithmetic_decomposes_over_the_usage_counts():
    """Local share, fallback rate and refusals must decompose arithmetically."""
    coverage = _artefact()["local_coverage"]
    hours = _artefact()["windows"]
    for window in ("group1", "group2"):
        row = coverage[window]
        total = row["total_requests"]
        assert row["local_requests"] + row["remote_requests"] == total, (
            f"{window}: local + remote != requests — a request is missing or "
            "double-counted"
        )
        assert row["local_share_pct"] == round(row["local_requests"] / total * 100, 2), (
            f"{window}: local share is not local_requests / total_requests"
        )
        assert row["remote_fallback_rate_pct"] == round(
            row["remote_fallback_events"] / total * 100, 2
        ), f"{window}: fallback rate is not fallback_events / total_requests"
        assert sum(row["admission_refusals_by_reason"].values()) == row[
            "admission_refusals"
        ], f"{window}: the per-reason refusals do not sum to the refusal count"
        assert row["prefill_budget_refusals"] == row["admission_refusals_by_reason"][
            "prefill_budget_exceeded"
        ], f"{window}: the budget refusals disagree with the per-reason split"
        assert row["admission_refusals_per_hour"] == round(
            row["admission_refusals"] / hours[window]["hours"], 2
        ), f"{window}: refusals per hour is not refusals / window hours"


def test_no_local_coverage_collapse_against_the_producer_threshold():
    """AC2's met/not-met test, recomputed from the two local shares."""
    artefact = _artefact()
    threshold = artefact["threshold"]
    group1 = artefact["local_coverage"]["group1"]["local_share_pct"]
    group2 = artefact["local_coverage"]["group2"]["local_share_pct"]
    floor = round(group1 - LOCAL_SHARE_ALLOWANCE_PP, 2)

    assert threshold["group1_local_share_pct"] == group1
    assert threshold["group2_local_share_pct"] == group2
    assert threshold["floor_pct"] == floor, (
        f"the floor is {threshold['floor_pct']} %, not group 1 minus "
        f"{LOCAL_SHARE_ALLOWANCE_PP:g} pp ({floor} %)"
    )
    assert threshold["delta_pp"] == round(group2 - group1, 2)
    assert threshold["margin_above_floor_pp"] == round(group2 - floor, 2)
    assert threshold["met"] is (group2 >= floor), (
        f"`met` is {threshold['met']} but {group2} % vs the {floor} % floor says "
        f"{group2 >= floor}"
    )
    assert threshold["met"] is True, (
        f"local share fell {group1} % -> {group2} %, below the {floor} % floor: "
        "group 2 collapsed local coverage (AC2 not met)"
    )


def test_the_report_quotes_the_verdicts_and_the_honest_counterpoint():
    """The committed prose must carry both verdicts and the disclosed caveat."""
    text = VERDICT_PATH.read_text()
    # Prose wraps at 80 columns, so the verdict sentences are matched against a
    # whitespace-normalised copy; the table row below is single-spaced already.
    flat = " ".join(text.split())

    assert "## 9. Regression check" in text, (
        "the verdict report has no 'Regression check' section (AC3)"
    )
    assert "prefill-deadline-residual-group2-regression.json" in text, (
        "the regression section does not link its committed evidence artefact"
    )

    artefact = _artefact()
    group1 = artefact["prefill_deadline"]["group1"]["count"]
    group2 = artefact["prefill_deadline"]["group2"]["count"]
    assert f"| `prefill_deadline` evictions (all classes) | {group1} | {group2} |" in text, (
        f"the report does not quote the counted reason ({group1} -> {group2})"
    )

    threshold = artefact["threshold"]
    assert f"{threshold['group2_local_share_pct']:.2f} %" in text
    assert f"{threshold['floor_pct']:.2f} %" in text

    assert "AC1 verdict: MET — the count did not increase" in flat
    assert "AC2 verdict: MET — no local-coverage collapse" in flat

    # The honest counterpoint: the early-abort class rose even though the
    # counted reason fell, so the regression reading cannot be quoted alone.
    group2_counts = _dataset(GROUP2_DATASET_PATH)["evictions"]["by_reason"]
    group1_counts = _dataset(GROUP1_DATASET_PATH)["evictions"]["by_reason"]
    assert group2_counts["no_progress_watchdog"] > group1_counts["no_progress_watchdog"]
    assert "no_progress_watchdog" in text
    assert f"{group1_counts['no_progress_watchdog']} → {group2_counts['no_progress_watchdog']}" in text, (
        "the report no longer discloses the rising no_progress_watchdog class"
    )
