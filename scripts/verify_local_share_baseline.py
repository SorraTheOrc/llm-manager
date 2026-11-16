#!/usr/bin/env python3
"""Verify the local-share ceiling arithmetic on the 2026-11-01 measured basis.

Baseline / verification harness for **LP-0MWCEGJDP009WURD** (spike parent
LP-0MVXY8ZBZ0003042, AC1b/AC3 support). This is a *pure arithmetic analyser*:
it reads a ``proxy-usage-analysis`` report (or the committed basis artefact)
and re-derives every figure the spike quotes. It never changes proxy
behaviour, never reads a live slot and never needs the proxy running.

Why the tool exists
-------------------
The spike headline says one serialised prefill gate "caps local share at
~5.8 %", while the same measured basis divides to ~1.6 %. Two different
**denominators** were in play:

- *prompt-token share* — delivered / offered **prompt tokens** (the denominator
  the 44 300 tok/s and 718 tok/s figures already use);
- *stream share* — locally served streams / offered **streams**, obtained by
  dividing the same token capacity by the *current local* mean prompt size per
  stream (``local_prompt_tokens / local_streams``).

Neither is wrong; quoting one label ("local share") for both is. This tool
computes both, states the assumption each needs, and reports the headroom
factor and the ≥10 % requirement under each definition, so a downstream option
comparison cannot silently swap denominators.

Inputs
------
``--report PATH``
    A ``proxy-usage-analysis`` ``report.md`` for the window of record (or a
    freshly regenerated one for the same window). Every figure is parsed out of
    the report, so nothing is hand-copied.
``--basis PATH``
    A committed basis artefact (e.g.
    ``docs/dev/local-share-ceiling-baseline-2026-11-01.md``'s JSON sibling) whose
    recorded arithmetic is re-checked. Use this when the source report has
    rotated out of retention.

``--check`` makes the exit status meaningful: 0 when every recorded claim
reproduces inside its stated tolerance, 1 otherwise. ``--json`` emits the whole
artefact (the same shape the committed baseline uses) instead of the human
summary.

Usage::

    python3 scripts/verify_local_share_baseline.py \\
        --report ~/proxy-usage-reports/postfix-thrash-gate/report.md --json

    python3 scripts/verify_local_share_baseline.py \\
        --basis docs/dev/local-share-ceiling-baseline-2026-11-01.json --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

#: The window of record for the spike's measured basis.
WINDOW_START = "2026-11-01 13:12:48"
WINDOW_END = "2026-11-01 14:00:03"

#: Seconds of gate time the published ceiling extrapolation used (47.0 min).
#: The window is 2 835 s = 47.25 min; the 15 s difference is recorded as a
#: rounding deltas, not silently corrected away, because the published figures
#: are reproduced against the inputs that produced them.
PUBLISHED_CEILING_SECONDS = 2820.0

#: The intake brief's published figures, each verified (or refuted) below.
PUBLISHED = {
    "offered_prompt_tok_s": 44300.0,
    "local_rate_tok_s": 718.0,
    "local_token_share_pct": 1.6,
    "local_stream_share_pct": 5.8,
    "needed_for_10pct_tokens": 4_200_000.0,
    "needed_for_10pct_denominator": 42_000_000.0,
    "refusal_shares_pct": {
        "large_context_bypass": 32.8,
        "resident_prefix_insufficient": 24.7,
        "local_admission_guard": 18.1,
        "context_too_large": 16.6,
    },
}

#: Tolerances. Chosen from the precision the source report carries (whole
#: tokens, 0.1 percentage points), not from what makes a check pass.
TOL_TOK_S_REL = 0.001          # 0.1 % relative — report figures are rounded
TOL_SHARE_PP = 0.05            # 0.05 pp — refusal shares are quoted to 0.1 pp
TOL_CEILING_PP = 0.20          # 0.20 pp — the ceiling is quoted to 0.1 pp

#: Midpoint of the dominant *refused* priced-new-token band (20k-32k), used for
#: the mix-corrected stream ceiling. The band table is on uncached new tokens;
#: the constant-mix figure is on full prompt tokens, so the two are reported
#: as a range and labelled, never averaged.
REFUSED_BAND_MIDPOINT_TOKENS = 26_000.0


class ReportFormatError(RuntimeError):
    """The input report is missing a section this tool derives from."""


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def _num(value: str) -> float:
    """Parse ``877,908`` / ``20m 23s`` / ``0.8h`` style values to a number."""
    text = value.strip().replace(",", "")
    text = text.replace("**", "").strip()
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*%", text)
    if match:
        return float(match.group(1))
    return float(text)


def _duration_seconds(value: str) -> float:
    """Parse a report duration (``20m 23s`` / ``1h 02m`` / ``45s``) to seconds."""
    text = value.strip().replace("**", "")
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", text)
    if not parts:
        raise ReportFormatError(f"unparseable duration: {value!r}")
    scale = {"h": 3600.0, "m": 60.0, "s": 1.0}
    return sum(float(amount) * scale[unit] for amount, unit in parts)


def _require(pattern: str, text: str, what: str, flags: int = 0) -> re.Match:
    match = re.search(pattern, text, flags)
    if match is None:
        raise ReportFormatError(f"report is missing {what} (pattern {pattern!r})")
    return match


def _section(text: str, heading: str) -> str:
    """Return the body of a ``##``/``###`` heading up to the next peer heading.

    The report nests some tables one level down (``### Tokens — by model``), so
    the level is read from the heading itself and the body ends at the next
    heading of the same or a higher level.
    """
    found = re.search(
        rf"^(#{{2,6}}) {re.escape(heading)}\s*$", text, flags=re.MULTILINE
    )
    if found is None:
        raise ReportFormatError(f"report is missing the {heading!r} section")
    level = len(found.group(1))
    body = text[found.end() :]
    end = re.search(rf"^#{{2,{level}}} ", body, flags=re.MULTILINE)
    return body[: end.start()] if end else body


def _table_rows(section: str) -> list[list[str]]:
    """Split a markdown section's ``| a | b |`` rows into cell lists."""
    rows: list[list[str]] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= set("-: ") and cell for cell in cells):
            continue  # separator row
        rows.append(cells)
    return rows


def _row(rows: list[list[str]], first_cell: str) -> list[str]:
    for cells in rows:
        if cells and cells[0].replace("**", "").strip() == first_cell:
            return cells
    raise ReportFormatError(f"report table is missing the {first_cell!r} row")


def parse_report(text: str) -> dict[str, Any]:
    """Extract the measured basis from a proxy-usage-analysis ``report.md``."""
    window = _require(
        r"^- Window: ([\d: \-]+) → ([\d: \-]+) \(([\d.]+)h\)",
        text,
        "the window header",
        flags=re.MULTILINE,
    )
    header = _require(
        r"^- Sessions in window: \**(\d+)\** \| Requests: \**(\d+)\** "
        r"\(local (\d+) / remote (\d+)\)",
        text,
        "the sessions/requests header",
        flags=re.MULTILINE,
    )
    generated = _require(
        r"^- Generated: ([\d: \-]+)", text, "the generated timestamp", flags=re.MULTILINE
    )

    totals = _table_rows(_section(text, "Total tokens"))
    prompt = _row(totals, "Input (prompt)")
    completion = _row(totals, "Output (completion)")

    by_model = _table_rows(_section(text, "Tokens — by model"))
    local_tokens = _row(by_model, "local")
    if len(local_tokens) < 7:
        raise ReportFormatError("the by-model local row is missing token columns")

    utilization = _table_rows(_section(text, "Local model utilization"))
    busy = _row(utilization, "Busy time — upper bound (dispatch → finish)")
    streams = _row(utilization, "Streams served")

    fallbacks = _table_rows(_section(text, "Fallback reasons"))
    fallback_counts: dict[str, int] = {}
    total_row: int | None = None
    for cells in fallbacks[1:]:
        if cells[0].replace("**", "").strip() == "Reason":
            continue
        try:
            count = int(_num(cells[1]))
        except (ValueError, IndexError):
            continue
        if cells[0] == "Total":
            total_row = count
        else:
            fallback_counts[cells[0]] = count

    # The report states the fallback total in its header and (usually) as a
    # table footer. Prefer the header, keep the row sum, and fail loudly when
    # they disagree — a silent mismatch would move every refusal share.
    header_fallbacks = re.search(
        r"^- Fallback events: \**(\d+)\**", text, flags=re.MULTILINE
    )
    counted = sum(fallback_counts.values())
    if header_fallbacks is not None:
        fallback_total = int(header_fallbacks.group(1))
        if total_row is not None and total_row != fallback_total:
            raise ReportFormatError(
                f"fallback total row {total_row} disagrees with the header {fallback_total}"
            )
    elif total_row is not None:
        fallback_total = total_row
    else:
        fallback_total = counted
    if counted != fallback_total:
        raise ReportFormatError(
            f"fallback reason rows sum to {counted}, report total is {fallback_total}"
        )

    start = datetime.strptime(window.group(1).strip(), "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(window.group(2).strip(), "%Y-%m-%d %H:%M:%S")

    return {
        "report_generated_at": generated.group(1).strip(),
        "window_start": window.group(1).strip(),
        "window_end": window.group(2).strip(),
        "window_hours_reported": float(window.group(3)),
        "window_seconds": (end - start).total_seconds(),
        "sessions": int(header.group(1)),
        "requests": int(header.group(2)),
        "local_requests": int(header.group(3)),
        "remote_requests": int(header.group(4)),
        "offered_prompt_tokens": int(_num(prompt[1])),
        "offered_completion_tokens": int(_num(completion[1])),
        "local_prompt_tokens": int(_num(local_tokens[2])),
        "local_completion_tokens": int(_num(local_tokens[4])),
        "local_total_tokens": int(_num(local_tokens[6])),
        "local_streams": int(_num(streams[1])),
        "local_busy_seconds": _duration_seconds(busy[1]),
        "fallback_total": fallback_total,
        "fallback_counts": fallback_counts,
    }


# ---------------------------------------------------------------------------
# derivation
# ---------------------------------------------------------------------------


def _pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole else 0.0


def derive(figures: dict[str, Any]) -> dict[str, Any]:
    """Derive both share definitions, the ceilings and the headroom factors."""
    window_seconds = float(figures["window_seconds"])
    offered = float(figures["offered_prompt_tokens"])
    local_prompt = float(figures["local_prompt_tokens"])
    local_streams = float(figures["local_streams"])
    busy_seconds = float(figures["local_busy_seconds"])
    requests = float(figures["requests"])
    remote_prompt = offered - local_prompt
    remote_streams = float(figures["remote_requests"])

    local_rate = local_prompt / busy_seconds
    offered_rate = offered / window_seconds

    ceiling_tokens = window_seconds * local_rate
    ceiling_published = PUBLISHED_CEILING_SECONDS * local_rate

    local_prompt_per_stream = local_prompt / local_streams
    remote_prompt_per_stream = remote_prompt / remote_streams if remote_streams else 0.0
    offered_prompt_per_stream = offered / requests if requests else 0.0

    stream_ceiling = ceiling_tokens / local_prompt_per_stream if local_prompt_per_stream else 0.0
    stream_ceiling_published = (
        ceiling_published / local_prompt_per_stream if local_prompt_per_stream else 0.0
    )
    stream_ceiling_refusal_mix = (
        ceiling_tokens / REFUSED_BAND_MIDPOINT_TOKENS if REFUSED_BAND_MIDPOINT_TOKENS else 0.0
    )
    stream_ceiling_offered_mix = (
        ceiling_tokens / offered_prompt_per_stream if offered_prompt_per_stream else 0.0
    )

    refusal_shares = {
        reason: _pct(count, figures["fallback_total"])
        for reason, count in figures["fallback_counts"].items()
    }

    return {
        "offered_prompt_tok_s": offered_rate,
        "local_rate_tok_s": local_rate,
        "local_token_share_pct": _pct(local_prompt, offered),
        "local_stream_share_pct": _pct(local_streams, requests),
        "local_prompt_tokens_per_stream": local_prompt_per_stream,
        "remote_prompt_tokens_per_stream": remote_prompt_per_stream,
        "offered_prompt_tokens_per_stream": offered_prompt_per_stream,
        "prompt_size_ratio_remote_over_local": (
            remote_prompt_per_stream / local_prompt_per_stream
            if local_prompt_per_stream
            else 0.0
        ),
        "ceiling_tokens": ceiling_tokens,
        "ceiling_tokens_published_2820s": ceiling_published,
        "headroom_factor": ceiling_tokens / local_prompt if local_prompt else 0.0,
        "ceiling_token_share_pct": _pct(ceiling_tokens, offered),
        "ceiling_stream_share_pct": _pct(stream_ceiling, requests),
        "ceiling_stream_share_pct_published_2820s": _pct(
            stream_ceiling_published, requests
        ),
        "ceiling_stream_share_pct_refusal_mix": _pct(stream_ceiling_refusal_mix, requests),
        "ceiling_stream_share_pct_offered_mix": _pct(stream_ceiling_offered_mix, requests),
        "needed_for_10pct_token_share_tokens": 0.10 * offered,
        "needed_for_10pct_token_share_factor": (0.10 * offered) / ceiling_tokens,
        "needed_for_10pct_stream_share_tokens": (
            0.10 * requests * local_prompt_per_stream
        ),
        "needed_for_10pct_stream_share_factor": (
            (0.10 * requests * local_prompt_per_stream) / ceiling_tokens
        ),
        "refusal_shares_pct": refusal_shares,
    }


def check(
    figures: dict[str, Any], derived: dict[str, Any]
) -> list[dict[str, Any]]:
    """Verify each published claim; return one entry per assertion.

    A check is ``ok`` when the published figure reproduces inside its stated
    tolerance, ``refuted`` when the arithmetic contradicts it, and ``unverified``
    when the retained artefact cannot decide it (an explicit outcome, never a
    silent pass).
    """
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, expected: Any, actual: Any, tolerance: Any, note: str = "") -> None:
        entry: dict[str, Any] = {
            "claim": name,
            "published": expected,
            "reproduced": actual,
            "tolerance": tolerance,
            "ok": ok,
            "verdict": "reproduces" if ok else "refuted",
        }
        if note:
            entry["note"] = note
        checks.append(entry)

    def within_rel(actual: float, expected: float, tol_rel: float) -> bool:
        return abs(actual - expected) <= abs(expected) * tol_rel

    def within_pp(actual: float, expected: float, tol_pp: float) -> bool:
        return abs(actual - expected) <= tol_pp

    add(
        "offered prompt load ≈ 44 300 tok/s",
        within_rel(derived["offered_prompt_tok_s"], PUBLISHED["offered_prompt_tok_s"], TOL_TOK_S_REL),
        PUBLISHED["offered_prompt_tok_s"],
        round(derived["offered_prompt_tok_s"], 2),
        f"{TOL_TOK_S_REL:.3%} relative",
    )
    add(
        "local delivered rate ≈ 718 tok/s",
        within_rel(derived["local_rate_tok_s"], PUBLISHED["local_rate_tok_s"], TOL_TOK_S_REL),
        PUBLISHED["local_rate_tok_s"],
        round(derived["local_rate_tok_s"], 2),
        f"{TOL_TOK_S_REL:.3%} relative",
    )
    add(
        "prompt-token share ceiling ≈ 1.6 %",
        within_pp(derived["ceiling_token_share_pct"], PUBLISHED["local_token_share_pct"], TOL_CEILING_PP),
        PUBLISHED["local_token_share_pct"],
        round(derived["ceiling_token_share_pct"], 3),
        f"{TOL_CEILING_PP} pp",
        "denominator: offered prompt tokens",
    )
    add(
        "stream-share ceiling ≈ 5.8 % (constant local prompt mix)",
        within_pp(
            derived["ceiling_stream_share_pct_published_2820s"],
            PUBLISHED["local_stream_share_pct"],
            TOL_CEILING_PP,
        ),
        PUBLISHED["local_stream_share_pct"],
        round(derived["ceiling_stream_share_pct_published_2820s"], 3),
        f"{TOL_CEILING_PP} pp",
        "denominator: offered streams, capacity divided by the current local mean prompt size",
    )

    for reason, expected in PUBLISHED["refusal_shares_pct"].items():
        actual = derived["refusal_shares_pct"].get(reason)
        if actual is None:
            checks.append(
                {
                    "claim": f"refusal share {reason} = {expected} %",
                    "published": expected,
                    "reproduced": None,
                    "tolerance": f"{TOL_SHARE_PP} pp",
                    "ok": False,
                    "verdict": "unverified",
                    "note": "reason absent from the retained report's fallback table",
                }
            )
            continue
        add(
            f"refusal share {reason} = {expected} %",
            within_pp(actual, expected, TOL_SHARE_PP),
            expected,
            round(actual, 3),
            f"{TOL_SHARE_PP} pp",
        )

    needed = PUBLISHED["needed_for_10pct_tokens"]
    implied_denominator = needed / 0.10
    offered = float(figures["offered_prompt_tokens"])
    implied_share = _pct(needed, offered)
    checks.append(
        {
            "claim": "≥10 % share needs ~4.2 M local prompt tokens (10 % of ~42 M)",
            "published": needed,
            "reproduced": {
                "token_share_requirement_tokens": round(
                    derived["needed_for_10pct_token_share_tokens"], 1
                ),
                "stream_share_requirement_tokens": round(
                    derived["needed_for_10pct_stream_share_tokens"], 1
                ),
                "implied_denominator_tokens": implied_denominator,
                "implied_token_share_pct_of_offered": round(implied_share, 3),
                "offered_prompt_tokens": offered,
                "factor_vs_single_gate_capacity": round(
                    needed / derived["ceiling_tokens"], 3
                ),
            },
            "tolerance": "exact denominator match",
            "ok": False,
            "verdict": "unverified",
            "note": (
                "no denominator in the window equals ~42 M: 10 % of the offered "
                f"{offered / 1e6:,.2f} M is "
                f"{derived['needed_for_10pct_token_share_tokens'] / 1e6:,.2f} M and 10 % of "
                "the offered streams at the current local mix is "
                f"{derived['needed_for_10pct_stream_share_tokens'] / 1e6:,.2f} M; 4.2 M is "
                f"{needed / derived['ceiling_tokens']:,.2f}x the single-gate capacity, so the "
                "'more than 2x' claim holds, but it is not 10 % of any denominator this "
                "window contains"
            ),
        }
    )

    return checks


def parse_sessions_csv(path: Path) -> dict[str, int]:
    """Sum the per-session request counts from a report's ``standard_sessions.csv``.

    The CSV is written beside the report by the same run, so it is an
    independent corroboration of the header totals (sessions / requests /
    local / remote) rather than a re-parse of the same prose.
    """
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "sessions": len(rows),
        "requests": sum(int(row.get("local_requests") or 0) + int(row.get("remote_requests") or 0) for row in rows),
        "local_requests": sum(int(row.get("local_requests") or 0) for row in rows),
        "remote_requests": sum(int(row.get("remote_requests") or 0) for row in rows),
    }


def cross_check(figures: dict[str, Any], sessions: dict[str, int]) -> list[dict[str, Any]]:
    """Compare the header totals with the per-session CSV sums."""
    checks: list[dict[str, Any]] = []
    for key in ("sessions", "requests", "local_requests", "remote_requests"):
        expected = figures[key]
        actual = sessions[key]
        checks.append(
            {
                "claim": f"header {key} matches standard_sessions.csv sum",
                "published": expected,
                "reproduced": actual,
                "ok": expected == actual,
                "verdict": "reproduces" if expected == actual else "refuted",
            }
        )
    return checks


def build_report(
    figures: dict[str, Any],
    report_path: str | None = None,
    sessions: dict[str, int] | None = None,
) -> dict[str, Any]:
    derived = derive(figures)
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "Re-derive the local-share ceiling from the 2026-11-01 measured basis "
            "and pin which denominator each published share uses."
        ),
        "report_path": report_path,
        "window": {
            "start": WINDOW_START,
            "end": WINDOW_END,
            "seconds": figures["window_seconds"],
            "hours": figures["window_seconds"] / 3600.0,
        },
        "measured": figures,
        "derived": derived,
        "cross_checks": cross_check(figures, sessions) if sessions else [],
        "checks": check(figures, derived),
        "published_ceiling_seconds": PUBLISHED_CEILING_SECONDS,
    }


def all_ok(report: dict[str, Any]) -> bool:
    """True when no recorded claim is ``refuted``.

    ``unverified`` is an explicit, reported outcome (the retained artefact
    cannot decide the claim); only a contradiction fails the run.
    """
    entries = list(report["checks"]) + list(report.get("cross_checks") or [])
    return all(entry["verdict"] != "refuted" for entry in entries)


def load_basis(path: Path) -> dict[str, Any]:
    """Load a committed basis artefact and re-derive from its measured figures."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "measured" not in data:
        raise ReportFormatError(f"{path} carries no 'measured' block")
    return data


def verdicts(report: dict[str, Any]) -> dict[str, Any]:
    """The reconciliation statements the spike needs, as machine-readable data."""
    derived = report["derived"]
    figures = report["measured"]
    return {
        "token_definition": {
            "label": "local share of offered prompt tokens",
            "ceiling_pct": round(derived["ceiling_token_share_pct"], 3),
            "assumption": "none beyond the measured rate",
        },
        "stream_definition": {
            "label": "local share of streams (current local prompt mix)",
            "ceiling_pct": round(derived["ceiling_stream_share_pct"], 3),
            "assumption": (
                "every offered stream is locally servable and the newly served "
                "streams carry the current local mean prompt size "
                f"({derived['local_prompt_tokens_per_stream']:.0f} tokens)"
            ),
        },
        "stream_definition_refusal_mix_pct": round(
            derived["ceiling_stream_share_pct_refusal_mix"], 3
        ),
        "stream_definition_offered_mix_pct": round(
            derived["ceiling_stream_share_pct_offered_mix"], 3
        ),
        "published_stream_ceiling_pct": PUBLISHED["local_stream_share_pct"],
        "implied_denominator_tokens_for_published_stream_ceiling": (
            derived["ceiling_tokens_published_2820s"]
            / (PUBLISHED["local_stream_share_pct"] / 100.0)
        ),
        "stream_equivalent_denominator_tokens": (
            figures["requests"] * derived["local_prompt_tokens_per_stream"]
        ),
        "headroom_factor": round(derived["headroom_factor"], 3),
        "prompt_size_ratio_remote_over_local": round(
            derived["prompt_size_ratio_remote_over_local"], 3
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render(report: dict[str, Any]) -> str:
    figures = report["measured"]
    derived = report["derived"]
    lines = [
        "Local-share ceiling verification — 2026-11-01 measured basis",
        "=" * 66,
        f"window            {report['window']['start']} → {report['window']['end']}"
        f"  ({report['window']['seconds']:,.0f} s)",
        f"requests          {_fmt(figures['requests'])}"
        f"  (local {_fmt(figures['local_requests'])})",
        f"sessions          {_fmt(figures['sessions'])}",
        "",
        "Measured basis",
        f"  offered prompt load {_fmt(derived['offered_prompt_tok_s'], 1)} tok/s",
        f"  local delivered     {_fmt(figures['local_prompt_tokens'])} prompt tokens"
        f" in {_fmt(figures['local_busy_seconds'], 0)} busy s",
        f"  local rate          {_fmt(derived['local_rate_tok_s'], 2)} tok/s",
        "",
        "Two share definitions (the reconciliation)",
        f"  prompt-token share  actual {_fmt(derived['local_token_share_pct'])} %"
        f"  ceiling {_fmt(derived['ceiling_token_share_pct'])} %",
        f"  stream share        actual {_fmt(derived['local_stream_share_pct'])} %"
        f"  ceiling {_fmt(derived['ceiling_stream_share_pct'])} %"
        f" (constant local mix)",
        f"    mix-corrected stream ceiling: {_fmt(derived['ceiling_stream_share_pct_refusal_mix'])} %"
        f" at the refused-band midpoint,"
        f" {_fmt(derived['ceiling_stream_share_pct_offered_mix'])} % at the all-stream mean",
        f"  headroom factor     {_fmt(derived['headroom_factor'], 2)}x"
        f"  (published 2 820 s basis:"
        f" {_fmt(PUBLISHED_CEILING_SECONDS * derived['local_rate_tok_s'], 0)} tokens)",
        "",
        "≥10 % requirement",
        f"  token share:  {_fmt(derived['needed_for_10pct_token_share_tokens'], 0)} tokens"
        f" = {_fmt(derived['needed_for_10pct_token_share_factor'], 2)}x capacity",
        f"  stream share: {_fmt(derived['needed_for_10pct_stream_share_tokens'], 0)} tokens"
        f" = {_fmt(derived['needed_for_10pct_stream_share_factor'], 2)}x capacity",
        "",
        "Claim checks",
    ]
    for entry in report["checks"]:
        marker = {"reproduces": "PASS", "refuted": "FAIL", "unverified": "N/A "}[
            entry["verdict"]
        ]
        lines.append(f"  [{marker}] {entry['claim']}")
        if entry.get("note"):
            lines.append(f"          {entry['note']}")
    if report.get("cross_checks"):
        lines.append("")
        lines.append("Independent cross-checks (standard_sessions.csv)")
        for entry in report["cross_checks"]:
            marker = "PASS" if entry["ok"] else "FAIL"
            lines.append(
                f"  [{marker}] {entry['claim']}: header {entry['published']:,}"
                f" vs csv {entry['reproduced']:,}"
            )
    lines.append("")
    lines.append(
        "verdict: " + ("all checked claims reproduce" if all_ok(report) else "CONTRADICTION FOUND")
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-derive the 2026-11-01 local-share ceiling from a proxy-usage "
            "report (or a committed basis artefact) and pin the denominator."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--report", type=Path, help="proxy-usage-analysis report.md to parse"
    )
    source.add_argument(
        "--basis", type=Path, help="committed basis JSON artefact to re-verify"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 when a published claim is contradicted",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.basis is not None:
        try:
            basis = load_basis(args.basis)
        except (OSError, ValueError, ReportFormatError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        report = build_report(basis["measured"], report_path=basis.get("report_path"))
        # The recorded cross-checks cannot be recomputed without the report's
        # CSV, so they are carried over as recorded evidence — ``--check``
        # still fails if a recorded one was bad.
        report["cross_checks"] = list(basis.get("cross_checks") or [])
    elif args.report is not None:
        try:
            text = args.report.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot read report {args.report}: {exc}", file=sys.stderr)
            return 2
        try:
            figures = parse_report(text)
        except ReportFormatError as exc:
            print(f"error: {args.report}: {exc}", file=sys.stderr)
            return 2
        csv_path = args.report.parent / "standard_sessions.csv"
        sessions = None
        if csv_path.exists():
            try:
                sessions = parse_sessions_csv(csv_path)
            except (OSError, ValueError) as exc:
                print(f"error: cannot read {csv_path}: {exc}", file=sys.stderr)
                return 2
        report = build_report(figures, report_path=str(args.report), sessions=sessions)
    else:
        print(
            "error: pass --report PATH (a proxy-usage-analysis report) or "
            "--basis PATH (a committed basis artefact)",
            file=sys.stderr,
        )
        return 2

    report["verdicts"] = verdicts(report)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=False))
    else:
        print(render(report))

    if args.check and not all_ok(report):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
