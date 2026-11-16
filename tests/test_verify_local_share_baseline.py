#!/usr/bin/env python3
"""Tests for the local-share ceiling verifier (LP-0MWCEGJDP009WURD).

The verifier exists to stop the spike's two "local share" figures (~1.6 % and
~5.8 %) being swapped: they use different denominators. These tests pin the
parsing contract, both derivations, the reconciliation verdict and the failure
modes, all against the 2026-11-01 figures the committed baseline records — no
live log dependency, no sleeps.

Covered:

- parsing a ``proxy-usage-analysis`` report (including the nested
  ``### Tokens — by model`` table) into the measured basis;
- both share definitions and their ceilings, and the identity that makes them
  differ (prompt tokens per stream, local vs remote);
- the ceiling scaling with the window length and the delivered rate;
- every published claim reproducing for the committed basis, and the one claim
  that cannot be decided (the 4.2 M "10 % of ~42 M") being reported as
  ``unverified`` rather than silently passing;
- refutation: a tampered basis (doubled busy seconds) fails ``--check``;
- loud failure on a malformed report (missing section, inconsistent fallback
  totals) — never a silent zero;
- the independent ``standard_sessions.csv`` cross-check;
- the committed artefact staying in step with the script (drift guard).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_local_share_baseline import (  # noqa: E402
    PUBLISHED,
    PUBLISHED_CEILING_SECONDS,
    ReportFormatError,
    all_ok,
    build_report,
    check,
    cross_check,
    derive,
    main,
    parse_report,
    parse_sessions_csv,
    verdicts,
)

ARTEFACT = REPO_ROOT / "docs" / "dev" / "local-share-baseline-2026-11-01.json"

#: A trimmed replica of the real report's structure and figures.
REPORT_TEXT = """\
# Proxy Usage Analysis Report

- Generated: 2026-11-01 14:07:11
- Window: 2026-11-01 13:12:48 → 2026-11-01 14:00:03 (0.8h)
- Sessions in window: **58** | Requests: **1788** (local 46 / remote 1742)
- Fallback events: **1667** (93.2% of requests) | dispatch denied: 106

## Total tokens

| Kind | Count | % of total |
|---|---|---|
| Input (prompt) | 125,577,780 | 99.3% |
| Output (completion) | 841,445 | 0.7% |
| Total | **126,419,225** | 100.0% |

### Tokens — by model

| Provider | Model | Input | % input | Output | % output | Total | % total |
|---|---|---|---|---|---|---|---|
| OpenCode | deepseek-v4.1-flash | 124,699,872 | 99.3% | 836,418 | 99.4% | 125,536,290 | 99.3% |
| local | Qwen3 | 877,908 | 0.7% | 5,027 | 0.6% | 882,935 | 0.7% |

## Local model utilization

| Metric | Total | Standard | Low |
|---|---|---|---|
| Busy time — upper bound (dispatch → finish) | 20m 23s (43.1%) | 20m 23s (43.1%) | 0s |
| Decode-only busy — lower bound (`Stream started` → finish) | 4m 53s (10.3%) | 4m 53s (10.3%) | 0s |
| Streams served | 46 | - | - |

## Fallback reasons

| Reason | Total | % of fallbacks | Standard | Low |
|---|---|---|---|---|
| large_context_bypass | 547 | 32.8% | 547 (32.8%) | 0 (0.0%) |
| resident_prefix_insufficient | 411 | 24.7% | 411 (24.7%) | 0 (0.0%) |
| local_admission_guard | 302 | 18.1% | 302 (18.1%) | 0 (0.0%) |
| context_too_large | 276 | 16.6% | 276 (16.6%) | 0 (0.0%) |
| local_concurrency_limit | 67 | 4.0% | 67 (4.0%) | 0 (0.0%) |
| local_lease_active | 39 | 2.3% | 39 (2.3%) | 0 (0.0%) |
| post_deadline_terminal | 10 | 0.6% | 10 (0.6%) | 0 (0.0%) |
| tenant_limit | 9 | 0.5% | 9 (0.5%) | 0 (0.0%) |
| ReadTimeout | 4 | 0.2% | 4 (0.2%) | 0 (0.0%) |
| empty_response | 2 | 0.1% | 2 (0.1%) | 0 (0.0%) |

## Error analysis

| Error type | Status | Count | Evidence excerpt |
|---|---|---|---|
| slot_save failed | - | 5 | `...` |
"""


def _report_json() -> dict:
    return json.loads(ARTEFACT.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_report_extracts_the_measured_basis() -> None:
    figures = parse_report(REPORT_TEXT)

    assert figures["window_start"] == "2026-11-01 13:12:48"
    assert figures["window_end"] == "2026-11-01 14:00:03"
    assert figures["window_seconds"] == 2835.0  # 47.25 min, not the quoted 0.8 h
    assert figures["sessions"] == 58
    assert figures["requests"] == 1788
    assert figures["local_requests"] == 46
    assert figures["remote_requests"] == 1742
    assert figures["offered_prompt_tokens"] == 125_577_780
    # The nested `### Tokens — by model` table must not be mistaken for the
    # Total-tokens table, and vice versa.
    assert figures["local_prompt_tokens"] == 877_908
    assert figures["local_completion_tokens"] == 5_027
    assert figures["local_total_tokens"] == 882_935
    assert figures["local_streams"] == 46
    assert figures["local_busy_seconds"] == 1223.0  # "20m 23s"
    assert figures["fallback_total"] == 1667
    assert figures["fallback_counts"]["local_admission_guard"] == 302


def test_parse_report_fails_loudly_when_a_section_is_missing() -> None:
    text = REPORT_TEXT.replace("## Local model utilization", "## Local utilisation")
    with pytest.raises(ReportFormatError, match="Local model utilization"):
        parse_report(text)


def test_parse_report_fails_loudly_when_fallback_totals_disagree() -> None:
    text = REPORT_TEXT.replace("Fallback events: **1667**", "Fallback events: **1699**")
    with pytest.raises(ReportFormatError, match="sum to 1667"):
        parse_report(text)


# ---------------------------------------------------------------------------
# derivation: the two denominators
# ---------------------------------------------------------------------------


def test_both_share_definitions_are_derived_from_their_own_denominator() -> None:
    derived = derive(parse_report(REPORT_TEXT))

    # Prompt-token share: denominator is offered prompt tokens.
    assert derived["local_token_share_pct"] == pytest.approx(0.699, abs=1e-3)
    assert derived["ceiling_token_share_pct"] == pytest.approx(1.621, abs=1e-3)

    # Stream share: denominator is offered streams, capacity converted with the
    # current *local* mean prompt size per stream.
    assert derived["local_stream_share_pct"] == pytest.approx(2.573, abs=1e-3)
    assert derived["ceiling_stream_share_pct"] == pytest.approx(5.964, abs=1e-3)

    # The two ceilings differ by exactly the prompt-size mix — offered streams
    # average 70 234 tokens against the local 19 085 — that identity is the
    # reconciliation, so it is pinned rather than assumed.
    assert derived["local_prompt_tokens_per_stream"] == pytest.approx(19_084.96, abs=0.01)
    assert derived["offered_prompt_tokens_per_stream"] == pytest.approx(70_233.66, abs=0.01)
    assert derived["prompt_size_ratio_remote_over_local"] == pytest.approx(3.751, abs=1e-3)
    assert (
        derived["ceiling_stream_share_pct"] / derived["ceiling_token_share_pct"]
        == pytest.approx(
            derived["offered_prompt_tokens_per_stream"]
            / derived["local_prompt_tokens_per_stream"],
            rel=1e-9,
        )
    )


def test_ceiling_scales_with_window_length_and_delivered_rate() -> None:
    figures = parse_report(REPORT_TEXT)
    derived = derive(figures)

    # Ceiling = window seconds x delivered rate (the serialised-gate model).
    assert derived["ceiling_tokens"] == pytest.approx(
        figures["window_seconds"] * (figures["local_prompt_tokens"] / figures["local_busy_seconds"]),
        rel=1e-12,
    )
    assert derived["headroom_factor"] == pytest.approx(2.318, abs=1e-3)

    # Halving busy seconds doubles the rate, so the headroom factor doubles.
    faster = dict(figures, local_busy_seconds=figures["local_busy_seconds"] / 2)
    assert derive(faster)["headroom_factor"] == pytest.approx(
        2 * derived["headroom_factor"], rel=1e-12
    )

    # The published 2 820 s basis is recorded, not substituted for the window.
    assert PUBLISHED_CEILING_SECONDS == 2820.0
    assert derived["ceiling_tokens_published_2820s"] == pytest.approx(2_024_285, abs=2)


def test_mix_assumptions_bracket_the_stream_ceiling() -> None:
    derived = derive(parse_report(REPORT_TEXT))

    # Constant local mix is the optimistic end; the offered-stream mean lands
    # back on the token ceiling; the refused-band midpoint sits between.
    assert (
        derived["ceiling_stream_share_pct_offered_mix"]
        < derived["ceiling_stream_share_pct_refusal_mix"]
        < derived["ceiling_stream_share_pct"]
    )
    assert derived["ceiling_stream_share_pct_offered_mix"] == pytest.approx(
        derived["ceiling_token_share_pct"], abs=1e-6
    )


def test_ten_percent_requirement_is_derived_per_definition() -> None:
    derived = derive(parse_report(REPORT_TEXT))

    assert derived["needed_for_10pct_token_share_tokens"] == pytest.approx(12_557_778, abs=1)
    assert derived["needed_for_10pct_token_share_factor"] == pytest.approx(6.171, abs=1e-3)
    assert derived["needed_for_10pct_stream_share_tokens"] == pytest.approx(3_412_390, abs=1)
    assert derived["needed_for_10pct_stream_share_factor"] == pytest.approx(1.677, abs=1e-3)


# ---------------------------------------------------------------------------
# claim checks
# ---------------------------------------------------------------------------


def test_every_decidable_published_claim_reproduces() -> None:
    report = build_report(parse_report(REPORT_TEXT))
    checks = {entry["claim"]: entry for entry in report["checks"]}

    for claim in (
        "offered prompt load ≈ 44 300 tok/s",
        "local delivered rate ≈ 718 tok/s",
        "prompt-token share ceiling ≈ 1.6 %",
        "stream-share ceiling ≈ 5.8 % (constant local prompt mix)",
    ):
        assert checks[claim]["verdict"] == "reproduces", claim

    for reason, share in PUBLISHED["refusal_shares_pct"].items():
        entry = checks[f"refusal share {reason} = {share} %"]
        assert entry["verdict"] == "reproduces", reason
        assert entry["reproduced"] == pytest.approx(share, abs=0.05)

    assert all_ok(report)


def test_the_four_point_two_million_claim_is_reported_unverified() -> None:
    report = build_report(parse_report(REPORT_TEXT))
    unverified = [e for e in report["checks"] if e["verdict"] == "unverified"]

    assert [e["claim"] for e in unverified] == [
        "≥10 % share needs ~4.2 M local prompt tokens (10 % of ~42 M)"
    ]
    reproduced = unverified[0]["reproduced"]
    # 4.2 M is 2.06x capacity (so "more than 2x" holds) but 10 % of neither
    # denominator the window contains.
    assert reproduced["factor_vs_single_gate_capacity"] == pytest.approx(2.06, abs=0.01)
    assert reproduced["implied_token_share_pct_of_offered"] == pytest.approx(3.345, abs=1e-3)
    assert reproduced["token_share_requirement_tokens"] == pytest.approx(12_557_778, abs=1)
    assert reproduced["stream_share_requirement_tokens"] == pytest.approx(3_412_390, abs=1)
    # An undecidable claim never fails the run, but it is never silent either.
    assert all_ok(report)


def test_a_tampered_rate_refutes_the_published_claims() -> None:
    figures = dict(parse_report(REPORT_TEXT), local_busy_seconds=2446.0)
    report = build_report(figures)
    checks = {entry["claim"]: entry for entry in report["checks"]}

    assert checks["local delivered rate ≈ 718 tok/s"]["verdict"] == "refuted"
    assert checks["prompt-token share ceiling ≈ 1.6 %"]["verdict"] == "refuted"
    assert not all_ok(report)


def test_check_makes_the_exit_status_meaningful(tmp_path: Path) -> None:
    tampered = _report_json()
    tampered["measured"]["local_prompt_tokens"] = 1_877_908  # impossible: > offered
    path = tmp_path / "basis.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    assert main(["--basis", str(path), "--check"]) == 1
    assert main(["--basis", str(path)]) == 0  # reporting only: exit 0
    assert main(["--report", str(tmp_path / "absent.md"), "--check"]) == 2


# ---------------------------------------------------------------------------
# cross-check, verdicts and the committed artefact
# ---------------------------------------------------------------------------


def test_csv_cross_check_refutes_a_mismatched_header() -> None:
    figures = parse_report(REPORT_TEXT)
    agreed = {
        "sessions": 58,
        "requests": 1788,
        "local_requests": 46,
        "remote_requests": 1742,
    }
    assert all(entry["ok"] for entry in cross_check(figures, agreed))

    disagreeing = dict(agreed, requests=1790)
    report = build_report(figures, sessions=disagreeing)
    assert [e["claim"] for e in report["cross_checks"] if not e["ok"]] == [
        "header requests matches standard_sessions.csv sum"
    ]
    assert not all_ok(report)


def test_parse_sessions_csv_sums_the_per_session_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "standard_sessions.csv"
    csv_path.write_text(
        "session_id,local_requests,remote_requests\n"
        "s1,10,30\n"
        "s2,0,5\n"
        "s3,2,\n",
        encoding="utf-8",
    )
    assert parse_sessions_csv(csv_path) == {
        "sessions": 3,
        "requests": 47,
        "local_requests": 12,
        "remote_requests": 35,
    }


def test_verdicts_name_both_definitions_and_their_assumptions() -> None:
    report = build_report(parse_report(REPORT_TEXT))
    result = verdicts(report)

    assert result["token_definition"]["ceiling_pct"] == pytest.approx(1.621, abs=1e-3)
    assert result["stream_definition"]["ceiling_pct"] == pytest.approx(5.964, abs=1e-3)
    assert "19085 tokens" in result["stream_definition"]["assumption"]
    assert "locally servable" in result["stream_definition"]["assumption"]
    assert result["published_stream_ceiling_pct"] == 5.8
    assert result["headroom_factor"] == pytest.approx(2.318, abs=1e-3)

    # The denominator the published 5.8 % implies is the *stream-equivalent*
    # one (streams x local mean prompt size), not an uncached-token one — that
    # is what identifies the streams-vs-tokens mixup as the cause.
    assert result["implied_denominator_tokens_for_published_stream_ceiling"] == pytest.approx(
        34_901_466, rel=1e-4
    )
    assert result["stream_equivalent_denominator_tokens"] == pytest.approx(
        34_123_900, rel=1e-4
    )
    assert (
        result["implied_denominator_tokens_for_published_stream_ceiling"]
        / result["stream_equivalent_denominator_tokens"]
        == pytest.approx(1.0, abs=0.03)
    )


def test_committed_artefact_verifies_and_matches_the_script() -> None:
    assert ARTEFACT.exists(), f"missing committed baseline artefact: {ARTEFACT}"
    committed = _report_json()

    # The artefact must still verify end to end.
    assert main(["--basis", str(ARTEFACT), "--check"]) == 0

    # Drift guard: re-deriving from the recorded measured figures must
    # reproduce the recorded derived figures exactly, so a later change to the
    # script cannot leave a stale artefact behind.
    rebuilt = build_report(committed["measured"])
    for key, value in committed["derived"].items():
        assert rebuilt["derived"][key] == pytest.approx(value, rel=1e-12), key

    assert committed["window"]["start"] == "2026-11-01 13:12:48"
    assert committed["window"]["seconds"] == 2835.0
    assert all(entry["ok"] for entry in committed["cross_checks"])
