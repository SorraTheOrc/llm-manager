"""Error taxonomy for the typed deterministic-400 errors.

LP-0MW959HRZ00231FN (T4, parent LP-0MW0KZU86008PD8J AC5). The proxy answers a
byte-identical resend of a history it has already rejected with a typed
``deterministic_client_error`` 400 (LP-0MW95BDJQ002DSVI) and, once the backoff
ladder is exhausted, a terminal ``session_wedged`` 400
(LP-0MW95BE7X00574OL). Each response increments its own Prometheus counter
reason *and* logs exactly one WARNING line
(``proxy/proxy/provider.py:_build_typed_400_response`` /
``_build_session_wedged_response``). The log line is the only evidence the
usage report has: the Prometheus counters are not part of the analysed logs.

These tests pin the taxonomy contract (parent AC5): both are their own error
categories in the report's Error analysis section, attributed to the provider,
model and session that produced them — never folded into the provider-outage
bucket (``provider_exhausted``), because a client resending a rejected history
is not a provider failure and must not be reported as one.
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import aggregation  # noqa: E402
import bucketing  # noqa: E402
import log_parser  # noqa: E402
import reporting  # noqa: E402

WINDOW_START = datetime(2026, 11, 11, 9, 0, 0)
WINDOW_END = datetime(2026, 11, 11, 11, 0, 0)
TS = "2026-11-11 10:00:00,123"
TS_LATER = "2026-11-11 10:05:00,123"
SESSION = "herdr-1794407977-1476922-937"

TYPED_LABEL = "deterministic client error (identical history)"
WEDGED_LABEL = "session wedged (terminal)"


def _analyse(lines: list[str]):
    events = [
        e for e in (log_parser.parse_log_line(ln) for ln in lines) if e is not None
    ]
    schedule = bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])
    return aggregation.aggregate(events, WINDOW_START, WINDOW_END, schedule)


def _typed_line(
    session: str = SESSION,
    provider: str = "local-qwen3",
    resend_count: int = 2,
    retry_after: int = 60,
    ts: str = TS,
) -> str:
    """The line ``_build_typed_400_response`` logs when it answers a resend."""
    return (
        f"{ts} - WARNING - Deterministic client error: provider={provider} "
        f"model=v1/chat/completions session={session} "
        f"fingerprint=1a2b3c4d5e6f resend_count={resend_count} "
        f"retry_after={retry_after}s \u2014 returning typed HTTP 400 without "
        "remote fallback or chain hold (this exact history is rejected every "
        "time; compaction may help)"
    )


def _wedged_line(
    session: str = SESSION,
    provider: str = "local-qwen3",
    resend_count: int = 5,
    retry_after: int = 900,
    ts: str = TS_LATER,
) -> str:
    """The line ``_build_session_wedged_response`` logs when the breaker trips."""
    return (
        f"{ts} - WARNING - Session wedged: provider={provider} "
        f"model=v1/chat/completions session={session} "
        f"fingerprint=1a2b3c4d5e6f resend_count={resend_count} "
        f"retry_after={retry_after}s \u2014 this history has been resent after "
        "every backoff window; returning the terminal session_wedged error "
        "(compaction is the only way forward)"
    )


def _exhausted_line(ts: str = TS) -> str:
    """A provider-outage line: the bucket the new categories must stay out of."""
    return (
        f"{ts} - WARNING - All providers exhausted for "
        "model=v1/chat/completions; unavailable={'local-qwen3': 59}; "
        "usage_limit_reset=None"
    )


# ---------------------------------------------------------------------------
# TAC3 — parsing: each new error is its own event kind
# ---------------------------------------------------------------------------


def test_typed_deterministic_error_parses_as_its_own_error_kind():
    event = log_parser.parse_log_line(_typed_line())

    assert event is not None, "the typed-400 warning must not be dropped"
    assert event.kind == "deterministic_client_error"
    assert event.provider == "local-qwen3"
    assert event.model == "v1/chat/completions"
    assert event.session == SESSION
    assert event.retry_after == 60, "the advertised wait is the classifier's clock"


def test_session_wedged_parses_as_its_own_error_kind():
    event = log_parser.parse_log_line(_wedged_line())

    assert event is not None, "the terminal-breaker warning must not be dropped"
    assert event.kind == "session_wedged"
    assert event.provider == "local-qwen3"
    assert event.model == "v1/chat/completions"
    assert event.session == SESSION
    assert event.retry_after == 900


# ---------------------------------------------------------------------------
# TAC3 — taxonomy: separate categories, separate counts
# ---------------------------------------------------------------------------


def test_the_two_categories_are_counted_separately_from_provider_exhaustion():
    summary = _analyse(
        [
            _typed_line(),
            _typed_line(ts=TS_LATER),
            _wedged_line(),
            _exhausted_line(),
            _exhausted_line(ts=TS_LATER),
        ]
    )

    assert summary.error_counts == Counter(
        {"deterministic_client_error": 2, "session_wedged": 1}
    ), "the deterministic-400 errors are their own taxonomy categories"
    assert len(summary.provider_exhausted_events) == 2, (
        "provider exhaustion keeps counting only 'All providers exhausted' lines"
    )


def test_attribution_reaches_the_provider_model_breakdown():
    summary = _analyse([_typed_line(), _wedged_line()])

    assert summary.error_provider_model_counts == Counter(
        {
            ("deterministic_client_error", "local-qwen3", "v1/chat/completions"): 1,
            ("session_wedged", "local-qwen3", "v1/chat/completions"): 1,
        }
    )


def test_json_summary_reports_the_categories_separately():
    payload = reporting.summary_to_json(_analyse([_typed_line(), _wedged_line()]))

    assert payload["errors_by_type"]["deterministic_client_error"] == 1
    assert payload["errors_by_type"]["session_wedged"] == 1
    assert payload["provider_exhausted_errors"] == 0, (
        "a typed 400 is not a provider outage"
    )


def test_report_renders_a_distinct_row_per_category():
    report = reporting.build_report(
        _analyse([_typed_line(), _typed_line(ts=TS_LATER), _wedged_line(), _exhausted_line()]),
        {},
    )
    taxonomy = report.split("### Provider/model breakdown")[0]

    assert "## Error analysis" in taxonomy
    assert f"| {TYPED_LABEL} | - | 2 |" in taxonomy, taxonomy
    assert f"| {WEDGED_LABEL} | - | 1 |" in taxonomy, taxonomy
    assert "| Provider exhaustion | 1 |" in report, (
        "the provider-outage row is reported elsewhere and counts only its own lines"
    )
