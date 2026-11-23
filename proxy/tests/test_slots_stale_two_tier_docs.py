"""Doc contract for the two-tier `slots_stale` framing (LP-0MVVXGN2C004O28B).

F7 closes parent AC4 by reconciling the freshness narrative. Three things
could silently drift back, so they are pinned here:

1. **Schema version.** The payload example and ``proxy/README.md`` must state
   the *delivered* version (8), not the historical 5/7 — a stale example is
   how the previous mismatch crept in.
2. **Two-tier framing.** ``slots_stale`` at the 300 s ceiling is an
   *unusably-old-or-undatable* bound, never a freshness claim; freshness is the
   reported snapshot-age distribution; no hard freshness SLO is introduced and
   the revisit trigger (with both gating items) is recorded. This is the
   producer decision of 2026-11-22 (LP-0MWRXH13L000UXSA, option 3).
3. **Figures come from the recorded measurement.** Every rate in the section
   must be derivable from
   ``docs/dev/slots-stale-postfix-window-2026-11-18-to-19.json`` — no invented
   numbers, and no "open question" left behind once the decision is recorded.

Mirrors ``test_slots_stale_reason_docs.py`` (doc/code drift pins).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "proxy" / "docs" / "per-slot-status-verification.md"
README = REPO_ROOT / "proxy" / "README.md"
OPERATOR_DOC = REPO_ROOT / "docs" / "local-slots-observability.md"
RECORD = REPO_ROOT / "docs" / "dev" / "slots-stale-postfix-window-2026-11-18-to-19.md"
RECORD_JSON = REPO_ROOT / "docs" / "dev" / "slots-stale-postfix-window-2026-11-18-to-19.json"

SCHEMA_HEADING = "## Payload (new fields)"
OUTCOME_TITLE = "Measured outcome — post-fix 24 h window (LP-0MWLZ68SF0052E7Z)"


def _text(path: Path) -> str:
    assert path.exists(), f"{path} is missing"
    return path.read_text(encoding="utf-8")


def _payload_example(text: str) -> str:
    """The ```json fenced block that follows the payload heading."""
    after = text.split(SCHEMA_HEADING, 1)[1]
    block = after.split("```json", 1)[1]
    return block.split("```", 1)[0]


def _outcome_section(text: str) -> str:
    start = text.index(OUTCOME_TITLE)
    rest = text[start + len(OUTCOME_TITLE) :]
    # next heading of the same or higher level terminates the section
    terminator = re.search(r"\n## ", rest)
    section = rest if terminator is None else rest[: terminator.start()]
    # emphasis markers would split a quoted figure (``max **424.20 s**``)
    return section.replace("*", "")


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(_text(RECORD_JSON))


class TestSchemaVersionIsCurrent:
    def test_payload_example_uses_the_delivered_version(self):
        example = _payload_example(_text(DOC))
        assert '"status_schema_version": 8' in example, (
            "the payload example in proxy/docs/per-slot-status-verification.md "
            "must show the delivered status_schema_version (8)"
        )

    def test_payload_example_has_no_historical_version(self):
        example = _payload_example(_text(DOC))
        for stale in ('"status_schema_version": 5', '"status_schema_version": 7'):
            assert stale not in example, (
                f"{stale} in the payload example presents a superseded schema version as current"
            )

    def test_payload_example_carries_the_schema_8_fields(self):
        example = _payload_example(_text(DOC))
        for field in (
            "slots_stale_reason",
            "slots_snapshot_age_seconds",
            "slots_snapshot_source",
        ):
            assert f'"{field}"' in example, f"payload example is missing {field}"

    def test_readme_states_the_current_version(self):
        readme = _text(README)
        assert "currently `8`" in readme, "proxy/README.md must state the current status_schema_version (8)"
        assert "currently `7`" not in readme, "proxy/README.md still presents status_schema_version 7 as current"

    def test_documented_version_matches_the_recorded_build(self, record):
        documented = record["deployed_build"]["status_schema_version_in_window"]
        assert documented == 8
        assert '"status_schema_version": 8' in _payload_example(_text(DOC))
        assert "currently `8`" in _text(README)


class TestTwoTierFraming:
    def test_stale_flag_is_labelled_a_bound_not_a_freshness_claim(self):
        section = _outcome_section(_text(DOC))
        assert "unusably-old-or-undatable" in section, (
            "the measured-outcome section must describe slots_stale at the "
            "300 s ceiling as an unusably-old-or-undatable bound"
        )
        assert "never a freshness claim" in section

    def test_distribution_is_the_number_to_watch(self):
        section = _outcome_section(_text(DOC))
        assert "the number to watch" in section.lower()
        assert "reported not targeted" in section.lower()

    def test_no_hard_freshness_slo_is_introduced(self):
        section = _outcome_section(_text(DOC))
        assert "none today" in section.lower(), "the section must record that no hard freshness SLO exists today"

    def test_revisit_trigger_is_recorded_with_both_gates(self):
        section = _outcome_section(_text(DOC))
        assert "Revisit trigger" in section
        assert "LP-0MWRXFO0I005UBVM" in section  # undatable < 0.5 % gate
        assert "LP-0MWRXGPO3004GXD7" in section  # nightly age series gate
        assert "0.5 %" in section
        assert "14 days" in section
        assert "relabelling test" in section

    def test_the_question_is_no_longer_open(self):
        doc = _text(DOC)
        assert "open, non-blocking producer question" not in doc.lower(), (
            "the ceiling choice is decided (2026-11-22, option 3) and must not still be described as an open question"
        )
        assert "LP-0MWRXH13L000UXSA" in doc
        assert "LP-0MWRXH13L000UXSA" in _text(RECORD)


class TestFiguresMatchTheRecordedMeasurement:
    def test_stale_rate_matches_the_as_configured_reading(self, record):
        reading = record["as_configured_reading"]
        assert reading["verdict_under_5_percent"] is True
        rate = round(reading["slots_stale"]["rate"] * 100, 2)
        assert rate == 4.88
        section = _outcome_section(_text(DOC))
        assert "4.88 %" in section
        assert f"{reading['slots_stale']['count']:,}" in section  # 608
        assert f"{reading['status_polls']:,}" in section  # 12,450

    def test_tightened_reading_matches_the_distribution_run(self, record):
        tightened = record["tightened_reading"]
        two_second = next(t for t in tightened["thresholds"] if t["threshold_seconds"] == 2.0)
        assert round(two_second["rate_of_datable"] * 100, 2) == 31.60
        section = _outcome_section(_text(DOC))
        assert "31.60 %" in section
        assert "31.6 %" in section  # the same figure, quoted once in prose

    def test_age_quantiles_match_the_distribution_run(self, record):
        age = record["tightened_reading"]["age_seconds"]
        section = _outcome_section(_text(DOC))
        for key in ("p50", "p90", "p99", "max"):
            quoted = f"{age[key]:.2f} s"
            assert quoted in section, (
                f"age {key} {quoted} from the measurement record is not quoted in the measured-outcome section"
            )

    def test_undatable_share_matches_the_record(self, record):
        reading = record["as_configured_reading"]
        undatable = reading["stale_composition"]["undatable"]
        polls = reading["status_polls"]
        share = round(undatable / polls * 100, 2)
        assert share == 4.47
        section = _outcome_section(_text(DOC))
        assert "4.47 %" in section


class TestRederivationIsExact:
    def test_both_reproducing_commands_are_given_verbatim(self):
        section = _outcome_section(_text(DOC))
        assert "python3 scripts/capacity_signal_verification.py" in section
        assert "--since '2026-11-18 10:25:00' --until '2026-11-19 10:25:00'" in section
        assert "python3 scripts/slots_snapshot_age_distribution.py" in section
        assert "--threshold-seconds 2" in section

    def test_window_bounds_match_the_record(self, record):
        window = record["window"]
        section = _outcome_section(_text(DOC))
        assert window["since_local"] in section
        assert window["until_local"] in section
        assert window["duration_hours"] == 24.0


class TestNoStaleCurrentBehaviourClaims:
    def test_pre_f3_serialised_split_is_not_presented_as_current(self):
        doc = _text(DOC)
        assert "the handler issues counts and detail as two sequential bounded" not in doc, (
            "the serialised counts/detail split is the pre-F3 shape and must not be described in the present tense"
        )
        assert "Neither shape is current" in doc, (
            "the profile conclusion must say the poll-blocking/fan-out shape is not the delivered behaviour"
        )

    def test_delivered_read_path_is_named(self):
        doc = _text(DOC)
        assert "Read path delivered (LP-0MVVXGLTM008ROJ9)" in doc
        assert "one shared snapshot" in doc

    def test_operator_contract_is_cross_referenced_not_duplicated(self):
        doc = _text(DOC)
        assert "../../docs/local-slots-observability.md" in doc, (
            "per-slot-status-verification.md must cross-reference the operator "
            "staleness contract instead of restating it"
        )
        assert OPERATOR_DOC.exists()
        operator = _text(OPERATOR_DOC)
        for token in ("local_slots_max_snapshot_age_seconds", "slots_stale_reason"):
            assert token in operator

    def test_operator_doc_does_not_call_the_bound_fresh(self):
        operator = _text(OPERATOR_DOC)
        assert "not stale by this bound" in operator, (
            "docs/local-slots-observability.md must say an inside-ceiling "
            "snapshot is not stale by the bound, not that it is fresh"
        )
        assert "inside the ceiling is\nfresh" not in operator
        assert "no hard freshness SLO exists today" in operator
        assert "unusably-old-or-undatable" in operator


class TestRecordDocCarriesTheDecision:
    def test_record_is_no_longer_waiting_on_a_question(self):
        record_md = _text(RECORD)
        assert "was **decided on 2026-11-22**" in record_md
        assert "Decision note (2026-11-22)" in record_md
        assert "option 3" in record_md

    def test_record_points_at_the_applied_text(self):
        record_md = _text(RECORD)
        assert "per-slot-status-verification.md" in record_md
        assert "LP-0MVVXGN2C004O28B" in record_md
