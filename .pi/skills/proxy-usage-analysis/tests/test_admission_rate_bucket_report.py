"""Contended-pricing surface in the usage report (LP-0MVS9YKHL007UBBZ).

Parent LP-0MVS8T6F4006DGG3 prices contended admissions against a contended
rate bucket and rebases the occupant term on the occupant's remaining new
tokens. Those decisions are only *observable* if the standard report surfaces
them, so this module verifies the additive surface:

- AC1 the ``local_admission_guard`` parser carries ``rate_bucket``,
  ``eta_occupancy_s``, ``new_tokens`` and ``contended_samples``, including a
  ``solitary_fallback`` bucket and a pre-change line with none of the fields;
- AC2 the JSON summary splits the admit/reject verdicts by ``rate_bucket`` and
  by the conversion harness's new-token bands, with per-band admission →
  first-byte conversion counts and absent bands omitted;
- AC3 pre-change line shapes parse exactly as before;
- AC5 (report) the markdown report renders the banded/bucketed verdict table.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
# The conversion harness lives in the repo's ``scripts/`` directory, not in the
# skill. It is loaded by file path (never by package name: the skill's own
# ``scripts/`` directory shadows the name ``scripts`` on ``sys.path``).
REPO_ROOT = Path(__file__).resolve().parents[4]
HARNESS_PATH = REPO_ROOT / "scripts" / "prefill_admission_conversion.py"

import aggregation  # noqa: E402
import bucketing  # noqa: E402
import log_parser  # noqa: E402
import reporting  # noqa: E402

WINDOW_START = datetime(2026, 10, 28, 9, 0, 0)
WINDOW_END = datetime(2026, 10, 28, 11, 0, 0)
TS = "2026-10-28 10:00:00,123"


def _load_conversion_harness():
    """Import the repo's admission-conversion harness module.

    The harness imports ``scripts.lib.proxy_logs``, which resolves to the
    repository's ``scripts/`` namespace package only when no other ``sys.path``
    entry provides a *regular* ``scripts`` package. The skill's own
    ``scripts/__init__.py`` does exactly that during the skill's pytest run, so
    those shadowing entries are dropped, the repository root is put first, and
    both the path and the ``scripts*``/``lib*`` module entries are restored
    afterwards — loading the harness cannot leak state into the skill's
    imports.
    """

    def _is_harness_namespace(name: str) -> bool:
        return name in ("scripts", "lib") or name.startswith(("scripts.", "lib."))

    def _shadows_repo_scripts(entry: str) -> bool:
        """Whether *entry* provides a regular ``scripts`` package elsewhere."""
        if not entry:
            return False
        root = Path(entry).resolve()
        return root != REPO_ROOT and (root / "scripts" / "__init__.py").is_file()

    saved_path = list(sys.path)
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if _is_harness_namespace(name)
    }
    for name in saved_modules:
        del sys.modules[name]
    sys.path[:] = [str(REPO_ROOT)] + [
        entry for entry in saved_path if not _shadows_repo_scripts(entry)
    ]
    try:
        spec = importlib.util.spec_from_file_location(
            "prefill_admission_conversion", HARNESS_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # The harness uses ``from __future__ import annotations``: its
        # dataclasses resolve string annotations through ``sys.modules``, so
        # the module must be registered while it executes.
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
        return module
    finally:
        sys.path[:] = saved_path
        for name in [n for n in sys.modules if _is_harness_namespace(n)]:
            del sys.modules[name]
        sys.modules.update(saved_modules)


SESSION_A = "herdr-1793223953-985232-14121"
SESSION_B = "herdr-1793223953-985232-14122"
SESSION_C = "herdr-1793223953-985232-14123"
SESSION_D = "herdr-1793223953-985232-14124"


def _guard_line(
    *,
    session: str = SESSION_A,
    decision: str = "admit",
    reason: str = "within_budget",
    new_tokens: int | None = 12_000,
    rate_bucket: str | None = "solitary",
    contended_samples: int | None = 0,
    eta_occupancy: float | None = 0.0,
    ts: str = TS,
) -> str:
    """A ``local_admission_guard`` line in the post-change (contended) shape."""
    tokens = f" new_tokens={new_tokens}" if new_tokens is not None else ""
    bucket = ""
    if rate_bucket is not None:
        bucket += f" rate_bucket={rate_bucket}"
        if contended_samples is not None:
            bucket += f" contended_samples={contended_samples}"
    occupancy = (
        f" eta_occupancy_s={eta_occupancy:.1f}" if eta_occupancy is not None else ""
    )
    return (
        f"{ts} - INFO - local_admission_guard decision={decision} reason={reason} "
        f"enforced=true eta_prefill_s=41.2{occupancy} projected_wait_s=41.2 "
        "budget_s=60.0 deadline_s=120.0 deadline_budget_s=114.0"
        f"{tokens} budget_tokens=40000{bucket} busy_slots=1 in_flight_prefills=0 "
        "free_slot=false priority=medium policy=reject model=Qwen3 samples=12 "
        f"provider=local-qwen3 session={session}"
    )


#: The pre-change guard shape: none of the contended-pricing fields exist yet.
PRE_CHANGE_GUARD = (
    "2026-10-28 10:00:00,123 - INFO - local_admission_guard decision=reject "
    "reason=eta_exceeds_budget enforced=true eta_prefill_s=468.0 "
    "projected_wait_s=468.0 budget_s=120.0 busy_slots=1 in_flight_prefills=1 "
    "free_slot=false priority=low policy=reject model=Qwen3 samples=12 "
    f"provider=local-qwen3 session={SESSION_B}"
)


def _first_byte_line(session: str, ts: str = TS) -> str:
    return (
        f"{ts} - INFO - dispatch_first_byte_ms=8123.4 "
        "dispatch_to_first_byte_ms=8120.1 cache_state=cold cached_ratio=0.000 "
        f"queue_wait_ms=3.3 session={session} model=Qwen3 priority=high "
        "estimated_tokens=22726"
    )


def _remote_ttft_line(session: str, ts: str = TS) -> str:
    return (
        f"{ts} - INFO - ttft_seconds=1.234 session={session} "
        "provider=opencode-go model=deepseek-v4-flash"
    )


def _analyse(lines: list[str]):
    events = [
        e for e in (log_parser.parse_log_line(ln) for ln in lines) if e is not None
    ]
    schedule = bucketing.schedule_from_entries([("23:59", 8), ("10:00", 6)])
    return aggregation.aggregate(events, WINDOW_START, WINDOW_END, schedule)


# ---------------------------------------------------------------------------
# AC1 — the decision log's new fields are parsed
# ---------------------------------------------------------------------------

def test_guard_line_carries_contended_pricing_fields():
    event = log_parser.parse_log_line(
        _guard_line(
            decision="reject",
            reason="eta_exceeds_budget",
            new_tokens=6_659,
            rate_bucket="contended",
            contended_samples=6,
            eta_occupancy=16.0,
        )
    )

    assert event is not None
    assert event.kind == "local_admission_guard"
    assert event.rate_bucket == "contended"
    assert event.contended_samples == 6
    assert event.eta_occupancy_s == "16.0"
    assert event.new_tokens == 6_659
    # The pre-existing guard fields keep their values (AC3).
    assert (event.decision, event.reason, event.enforced) == (
        "reject",
        "eta_exceeds_budget",
        "true",
    )
    assert (event.eta_prefill_s, event.budget_s) == ("41.2", "60.0")
    assert (event.busy_slots, event.in_flight_prefills) == (1, 0)


def test_solitary_fallback_bucket_parses():
    event = log_parser.parse_log_line(
        _guard_line(
            rate_bucket="solitary_fallback",
            contended_samples=0,
            eta_occupancy=3.5,
        )
    )

    assert event is not None
    assert event.rate_bucket == "solitary_fallback"
    assert event.contended_samples == 0
    assert event.eta_occupancy_s == "3.5"


def test_pre_change_guard_line_leaves_new_fields_unset():
    event = log_parser.parse_log_line(PRE_CHANGE_GUARD)

    assert event is not None
    assert event.kind == "local_admission_guard"
    assert event.rate_bucket is None
    assert event.contended_samples is None
    assert event.eta_occupancy_s is None
    assert event.new_tokens is None
    # Everything the pre-change parser already extracted still parses.
    assert event.decision == "reject"
    assert event.reason == "eta_exceeds_budget"
    assert event.eta_prefill_s == "468.0"
    assert event.budget_s == "120.0"


def test_local_and_remote_first_byte_lines_are_distinguished():
    local = log_parser.parse_log_line(_first_byte_line(SESSION_A))
    remote = log_parser.parse_log_line(_remote_ttft_line(SESSION_A))

    assert local is not None and remote is not None
    assert local.kind == remote.kind == "ttft"
    assert local.ttft_source == "local"
    assert remote.ttft_source == "remote"


# ---------------------------------------------------------------------------
# AC2 — verdicts split by rate bucket and by conversion-harness band
# ---------------------------------------------------------------------------

def test_verdicts_are_split_by_rate_bucket():
    summary = _analyse(
        [
            _guard_line(rate_bucket="contended", decision="admit"),
            _guard_line(rate_bucket="contended", decision="reject"),
            _guard_line(rate_bucket="contended", decision="reject"),
            _guard_line(rate_bucket="solitary", decision="admit"),
            _guard_line(rate_bucket="solitary_fallback", decision="reject"),
        ]
    )

    payload = reporting.summary_to_json(summary)

    assert payload["local_admission_by_rate_bucket"] == {
        "contended": {"admit": 1, "reject": 2},
        "solitary": {"admit": 1, "reject": 0},
        "solitary_fallback": {"admit": 0, "reject": 1},
    }
    # Additive only: the pre-existing verdict counters keep their meaning.
    assert payload["local_admission_admit"] == 2
    assert payload["local_admission_reject"] == 3


def test_verdicts_are_split_by_conversion_harness_band():
    summary = _analyse(
        [
            _guard_line(new_tokens=5_000),
            _guard_line(new_tokens=7_999),
            _guard_line(new_tokens=8_000),
            _guard_line(new_tokens=25_000),
            _guard_line(new_tokens=32_000),
            _guard_line(new_tokens=70_000),
            PRE_CHANGE_GUARD,
        ]
    )

    bands = reporting.summary_to_json(summary)["local_admission_by_token_band"]

    assert bands["<8k"]["admit"] == 2
    assert bands["8k-20k"]["admit"] == 1
    assert bands["20k-32k"]["admit"] == 1
    assert bands["32k-64k"]["admit"] == 1
    assert bands[">=64k"]["admit"] == 1
    # A pre-change line has no priced token count: it is reported as unknown
    # rather than dropped or bucketed as <8k.
    assert bands["unknown"]["reject"] == 1
    # Bands with no verdict are absent, not zero-filled.
    assert set(bands) == {"<8k", "8k-20k", "20k-32k", "32k-64k", ">=64k", "unknown"}


def test_absent_buckets_are_omitted_not_forced():
    summary = _analyse([_guard_line(rate_bucket="solitary", new_tokens=12_000)])

    payload = reporting.summary_to_json(summary)

    assert set(payload["local_admission_by_rate_bucket"]) == {"solitary"}
    assert set(payload["local_admission_by_token_band"]) == {"8k-20k"}


def test_band_conversion_join_counts_admit_to_first_byte():
    summary = _analyse(
        [
            # A: admitted inside 8k-20k and a local first byte follows.
            _guard_line(session=SESSION_A, new_tokens=12_000),
            _first_byte_line(SESSION_A, ts="2026-10-28 10:00:09,123"),
            # B: admitted, but only a *remote* first byte follows.
            _guard_line(session=SESSION_B, new_tokens=12_000),
            _remote_ttft_line(SESSION_B, ts="2026-10-28 10:00:05,123"),
            # C: admitted inside>=64k with no first byte at all.
            _guard_line(session=SESSION_C, new_tokens=70_000),
        ]
    )

    bands = reporting.summary_to_json(summary)["local_admission_by_token_band"]

    assert bands["8k-20k"]["converted"] == 1
    assert bands["8k-20k"]["unconverted"] == 1
    assert bands["8k-20k"]["conversion_share"] == pytest.approx(0.5)
    assert bands[">=64k"]["converted"] == 0
    assert bands[">=64k"]["unconverted"] == 1
    assert bands[">=64k"]["conversion_share"] == pytest.approx(0.0)


def test_first_byte_without_an_admit_is_not_joined():
    summary = _analyse([_first_byte_line(SESSION_D)])

    payload = reporting.summary_to_json(summary)

    assert payload["local_admission_by_rate_bucket"] == {}
    assert payload["local_admission_by_token_band"] == {}


def test_json_output_is_serialisable_and_empty_when_no_guard_events():
    payload = reporting.summary_to_json(_analyse([]))

    assert payload["local_admission_by_rate_bucket"] == {}
    assert payload["local_admission_by_token_band"] == {}
    json.dumps(payload)


def test_bands_match_the_conversion_harness_contract():
    """The report bands must be the harness's bands (AC2 says "the bands the
    manual harness uses"), so drift between the two is a test failure."""
    if not HARNESS_PATH.is_file():
        pytest.skip(f"conversion harness not present at {HARNESS_PATH}")
    harness = _load_conversion_harness()

    assert aggregation.ADMISSION_TOKEN_BANDS == harness.BANDS
    assert aggregation.UNKNOWN_BAND_LABEL == harness.UNKNOWN_BAND
    for tokens in (0, 7_999, 8_000, 19_999, 20_000, 31_999, 32_000, 63_999, 64_000):
        assert aggregation.admission_token_band(tokens) == harness.band_for(tokens)
    assert aggregation.admission_token_band(None) == harness.band_for(None)


# ---------------------------------------------------------------------------
# AC2/AC5 — rendered report section
# ---------------------------------------------------------------------------

def test_report_renders_banded_and_bucketed_verdict_table():
    summary = _analyse(
        [
            _guard_line(
                session=SESSION_A,
                decision="reject",
                reason="eta_exceeds_budget",
                new_tokens=6_659,
                rate_bucket="contended",
                contended_samples=6,
                eta_occupancy=16.0,
            ),
            _guard_line(session=SESSION_C, new_tokens=70_000, rate_bucket="contended"),
            _guard_line(
                session=SESSION_D,
                new_tokens=25_000,
                rate_bucket="solitary_fallback",
                contended_samples=0,
            ),
            _first_byte_line(SESSION_C, ts="2026-10-28 10:00:30,123"),
        ]
    )

    report = reporting.build_report(summary, {})

    assert "## Admission verdicts by rate bucket and prompt band" in report
    section = report.split("## Admission verdicts by rate bucket and prompt band", 1)[1]
    section = section.split("\n## ", 1)[0]
    assert "| Rate bucket | Admitted | Refused | Refused share |" in section
    assert "| contended | 1 | 1 | 50.0% |" in section
    assert "| solitary_fallback | 1 | 0 | 0.0% |" in section
    assert "| Band | Admitted | Refused | Conversions | Unconverted | Conversion share |" in section
    assert "| <8k | 0 | 1 |" in section
    assert "| >=64k | 1 | 0 | 1 | 0 |" in section
    assert "| 20k-32k | 1 | 0 | 0 | 1 |" in section


def test_report_omits_the_section_without_guard_events():
    report = reporting.build_report(_analyse([]), {})

    assert "## Admission verdicts by rate bucket and prompt band" not in report
