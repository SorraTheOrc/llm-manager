"""Release-context classification of post-acceptance evictions (V1, AC3).

LP-0MWM0CP4B001088M AC3: after the umbrella ``no_progress_watchdog`` label was
split (LP-0MWKQJHSM000CI6W), the window must classify **every** post-acceptance
abort by the emitted reason, ``task_started``, ``progress_observed`` and the
queue-wait bucket — with the never-started share made explicit and comparable
with the group-2 baseline (20 of 184 evictions carrying
``llama_task_id_reason=task_id_not_started``).

The contract pinned here:

- the classification consumes only fields the ``local_eviction_after_acceptance``
  line already carries (the four pre-existing keys plus the three release-context
  fields the split added), so no analyser change can move what the proxy emits;
- a window logged **before** the split (no release-context fields) degrades
  honestly to ``unknown`` — absence must never be read as ``task_started=no``;
- ``task_started=no`` is split by whether the slot was resolvable at all, so a
  reader cannot mistake "not attributable" for "the backend task never started";
- the never-started share uses the same ``llama_task_id_reason`` field the
  group-2 baseline counted, keeping the two figures comparable.
"""

from __future__ import annotations

import importlib
import sys
from datetime import datetime
from pathlib import Path

import pytest

WINDOW_START = datetime(2026, 11, 20, 0, 0, 0)
WINDOW_END = datetime(2026, 11, 20, 1, 0, 0)

#: Four eviction lines, one per shape AC3 must separate.
#:
#: 1. eta-exceeded, task resolved (``task_started=yes``), progress measured,
#:    no queue wait, priced the whole prompt as new compute (1000/1000).
#: 2. stalled, audit traffic, slot known but task unresolved
#:    (``llama_task_id_reason=task_id_not_started``), no progress, 60 s.
#: 3. umbrella fallback, task not attributable because the slot is unknown,
#:    12.5 s of queue wait, half the prompt priced as new compute.
#: 4. a pre-split line (no release-context fields at all) at the deadline.
FIXTURE_LINES = [
    "2026-11-20 00:05:00,000 - ERROR - local_eviction_after_acceptance "
    "session=u-eta reason=prefill_eta_exceeded estimated_tokens=1000 "
    "new_tokens=1000 eta_prefill_s=10.0 elapsed_s=30.0 slot_id=1 "
    "llama_task_id=42 endpoint=local task_started=yes "
    "progress_observed=yes queue_wait_s=0.0",
    "2026-11-20 00:10:00,000 - ERROR - local_eviction_after_acceptance "
    "session=audit-a reason=prefill_stalled estimated_tokens=2000 "
    "new_tokens=0 eta_prefill_s=unknown elapsed_s=60.0 slot_id=0 "
    "llama_task_id=unknown endpoint=local "
    "llama_task_id_reason=task_id_not_started task_started=no "
    "progress_observed=no queue_wait_s=unknown",
    "2026-11-20 00:20:00,000 - ERROR - local_eviction_after_acceptance "
    "session=u-umb reason=no_progress_watchdog estimated_tokens=1000 "
    "new_tokens=500 eta_prefill_s=5.0 elapsed_s=20.0 slot_id=unknown "
    "llama_task_id=unknown endpoint=local task_started=no "
    "progress_observed=no queue_wait_s=12.5",
    "2026-11-20 00:30:00,000 - ERROR - local_eviction_after_acceptance "
    "session=u-legacy reason=prefill_deadline estimated_tokens=2000 "
    "new_tokens=2000 eta_prefill_s=unknown elapsed_s=180.0 slot_id=unknown "
    "llama_task_id=unknown endpoint=local",
]


@pytest.fixture(name="script")
def _script():
    """Import the analysis script from ``proxy/scripts`` (no package there)."""
    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module("analyse_prefill_deadline_residual")


@pytest.fixture(name="log")
def _log(script):
    return script.parse_window(list(FIXTURE_LINES), WINDOW_START, WINDOW_END)


@pytest.fixture(name="split")
def _split(script, log):
    return script.release_context_split(log.evictions, 1.0)


# ---------------------------------------------------------------------------
# Parsing — the three release-context fields
# ---------------------------------------------------------------------------


def test_parser_reads_task_started_progress_and_queue_wait(script, log):
    first, second, third = log.evictions[0], log.evictions[1], log.evictions[2]
    assert (first.task_started, first.progress_observed) == (True, True)
    assert first.queue_wait_s == pytest.approx(0.0)
    assert (second.task_started, second.progress_observed) == (False, False)
    assert second.queue_wait_s is None
    assert second.task_id_reason == "task_id_not_started"
    assert (third.task_started, third.progress_observed) == (False, False)
    assert third.queue_wait_s == pytest.approx(12.5)


def test_a_pre_split_line_keeps_unknown_context(log):
    """Absence is ``None`` — never silently coerced to ``no``."""
    legacy = log.evictions[3]
    assert legacy.task_started is None
    assert legacy.progress_observed is None
    assert legacy.queue_wait_s is None
    assert legacy.task_id_reason is None


# ---------------------------------------------------------------------------
# Aggregates — counts and Σ elapsed_s, built by hand from the fixture
# ---------------------------------------------------------------------------


def test_event_count_and_stall_total_are_the_hand_computed_ones(split):
    assert split["events"] == 4
    assert split["hours"] == pytest.approx(1.0)
    assert split["stall_seconds"] == pytest.approx(290.0)


def test_reason_rows_carry_count_share_and_stall(split):
    rows = split["by_reason"]
    assert set(rows) == {
        "prefill_eta_exceeded",
        "prefill_stalled",
        "no_progress_watchdog",
        "prefill_deadline",
    }
    eta = rows["prefill_eta_exceeded"]
    assert eta["events"] == 1
    assert eta["share"] == pytest.approx(0.25)
    assert eta["stall_seconds"] == pytest.approx(30.0)
    assert eta["stall_seconds_per_hour"] == pytest.approx(30.0)
    assert (eta["user"], eta["audit"]) == (1, 0)
    stalled = rows["prefill_stalled"]
    assert (stalled["user"], stalled["audit"]) == (0, 1)
    assert stalled["stall_seconds"] == pytest.approx(60.0)


def test_task_started_split_counts_the_legacy_line_as_unknown(split):
    rows = split["by_task_started"]
    assert rows["yes"]["events"] == 1
    assert rows["no"]["events"] == 2
    assert rows["unknown"]["events"] == 1
    assert rows["no"]["stall_seconds"] == pytest.approx(80.0)
    assert rows["unknown"]["share"] == pytest.approx(0.25)


def test_progress_observed_split(split):
    rows = split["by_progress_observed"]
    assert rows["yes"]["events"] == 1
    assert rows["no"]["events"] == 2
    assert rows["unknown"]["events"] == 1


def test_queue_wait_buckets(split):
    rows = split["by_queue_wait_bucket"]
    assert rows["zero"]["events"] == 1
    assert rows["unknown"]["events"] == 2
    assert rows["(5, 30]"]["events"] == 1
    assert rows["(5, 30]"]["stall_seconds"] == pytest.approx(20.0)
    # Every bucket is always present, so a consumer never has to guess.
    assert set(rows) == {"zero", "(0, 1]", "(1, 5]", "(5, 30]", "> 30", "unknown"}


def test_never_started_share_uses_the_group2_comparable_field(split):
    never = split["never_started"]
    assert never["basis"] == "llama_task_id_reason=task_id_not_started"
    assert never["events"] == 1
    assert never["share"] == pytest.approx(0.25)
    assert (never["user"], never["audit"]) == (0, 1)
    assert never["stall_seconds"] == pytest.approx(60.0)
    assert never["by_task_id_reason"] == {"task_id_not_started": 1}


def test_task_started_no_is_split_by_slot_attributability(split):
    """``task_started=no`` must not be read as "the task never started"."""
    rows = split["task_started_no_by_slot"]
    assert rows["slot_known"]["events"] == 1
    assert rows["slot_unknown"]["events"] == 1
    assert rows["slot_known"]["by_task_id_reason"] == {"task_id_not_started": 1}


def test_whole_prompt_priced_as_new_compute(split):
    priced = split["priced_as_new_compute"]
    assert priced["events"] == 2
    assert priced["share"] == pytest.approx(0.5)


def test_pre_split_window_never_infers_a_release_context(script):
    lines = [FIXTURE_LINES[3]]
    log = script.parse_window(lines, WINDOW_START, WINDOW_END)
    split = script.release_context_split(log.evictions, 1.0)
    assert split["events_with_release_context"] == 0
    assert split["events_without_release_context"] == 1
    assert split["by_task_started"]["unknown"]["events"] == 1
    assert split["by_task_started"]["no"]["events"] == 0
    assert split["by_queue_wait_bucket"]["unknown"]["events"] == 1


def test_empty_population_is_well_formed(script):
    split = script.release_context_split([], 0.0)
    assert split["events"] == 0
    assert split["stall_seconds"] == 0.0
    assert split["by_task_started"]["yes"]["events"] == 0
    assert split["never_started"]["share"] == 0.0


def test_report_includes_the_classification_block(script, log):
    report = script.build_report(log, WINDOW_START, WINDOW_END)
    block = report["release_context_split"]
    assert block["events"] == 4
    assert block["source_fields"] == list(script.RELEASE_CONTEXT_SOURCE_FIELDS)
    assert block["queue_wait_buckets"] == list(script.QUEUE_WAIT_BUCKET_LABELS)
