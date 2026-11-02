"""Pin the release→backend-cancel joiner contract (LP-0MVW8S6SR003MBP5).

The sibling implementation item LP-0MVW8S7FN001B2Y1 ("Release→cancel
accounting in the residual analyser") makes every proxy release of an accepted
local dispatch attributable to a backend task lifecycle. This module pins that
joiner's **contract** against a hermetic fixture pair (a proxy-log slice and
the matching ``llama-server.log`` slice covering the same fake window), so
"the proxy released a dispatch and llama-server kept prefilling" becomes a
measured, per-dispatch fact instead of the subtraction of two unrelated
counters.

The joiner is implemented by the sibling implementation item
LP-0MVW8S7FN001B2Y1 ("Release→cancel accounting in the residual analyser");
the assertions below were pinned as ``xfail(strict=True)`` until it landed and
the markers have been removed — the tests now assert the contract directly.
One test — ``test_fixture_pair_is_hermetic_and_covers_all_shapes`` — verifies
the fixture pair itself (AC1) rather than the joiner.

Joiner contract pinned here
---------------------------

A **release** is one ``local_eviction_after_acceptance`` line (the proxy
releasing an *accepted* local dispatch early). The ``lease_released`` and
``abandoned_slot_recorded`` lines of the same millisecond are companions of
that event, not additional releases.

The joiner emits, per window, one row per release (AC2) plus the window
aggregates (impl AC1). Row fields:

- ``session``, ``reason`` — the release's session id and eviction reason;
- ``released_at`` — the proxy release timestamp (``%Y-%m-%d %H:%M:%S``);
- ``task_id`` — the joined ``id_task``, or ``None`` when the release is
  unjoined;
- ``unjoined`` / ``unjoined_reason`` — whether no backend task could be
  joined, and a machine-readable reason (e.g. ``cancel_without_task_id``);
- ``cancel_observed`` — whether a backend cancel event lands in the
  release's segment (a ``stop: cancel task, id_task = N`` line for the joined
  task, or an operator ``http client error: Connection handling canceled``
  event that carries no task id);
- ``progress_lines_after_cancel`` — the number of ``prompt processing
  progress`` lines for the joined task **after** its cancel line;
- ``tokens_evaluated_after_cancel`` — the tokens evaluated after the cancel,
  counted from the progress lines (the last ``n_tokens`` minus the ``n_tokens``
  at the last progress before the cancel) — **never** derived from the
  hard-coded ``ABANDONED_PREFILL_RATE_TOK_S`` constant (AC5);
- ``cancel_to_stop_s`` — the cancel→``stop processing`` latency in seconds:
  ``0.0`` when nothing was evaluated after the cancel, ``None`` when no cancel
  was observed, otherwise ``> 0`` and monotone in
  ``tokens_evaluated_after_cancel`` (the seconds basis must come from observed
  data — the fixture's own ``prompt eval time`` lines — not the constant);
- ``prompt_eval_time_emitted`` — whether the joined task still emitted a
  ``prompt eval time`` line after its cancel (H1 discriminator).

Window aggregates (impl AC1): ``releases_total``,
``releases_with_observed_cancel``, ``releases_without_explained_backend_stop``
(``{count, share}`` — a release whose joined lifecycle shows the backend
stopped processing is *explained*, with or without a cancel),
``releases_unjoined`` (``{count, reasons}``), ``cancel_to_stop_seconds``
(p10/p50/p90 distribution), ``tokens_evaluated_after_cancel``
(``{total, per_release}``), and the per-release ``releases`` list.

Fixture ground truth (window 2026-01-01 00:00:00 → 00:04:00):

+---------+-------------+----------------------+---------------------+----------+---------------+------------------+--------------+--------------+------------+
| release | session     | reason               | released_at         | task_id  | cancel seen   | progress after    | tokens after  | eval emitted | unjoined?  |
|         |             |                      |                     |          |               | cancel            | cancel       |              |            |
+=========+=============+======================+=====================+==========+===============+==================+==============+==============+============+
| (a)     | user-alpha  | prefill_deadline     | 2026-01-01 00:00:00 | 41       | yes           | 2                | 10000        | no           | no         |
+---------+-------------+----------------------+---------------------+----------+---------------+------------------+--------------+--------------+------------+
| (b)     | user-bravo  | prefill_deadline     | 2026-01-01 00:01:00 | 43       | **no**        | 0                | 0            | no           | no         |
+---------+-------------+----------------------+---------------------+----------+---------------+------------------+--------------+--------------+------------+
| (c)     | user-charlie| measured_cache_deficit| 2026-01-01 00:02:00| (none)   | yes (no id)   | 0                | 0            | no           | **yes**    |
+---------+-------------+----------------------+---------------------+----------+---------------+------------------+--------------+--------------+------------+
| (d)     | user-delta  | prefill_deadline     | 2026-01-01 00:03:00 | 42       | yes           | 0                | 0            | **yes**     | no         |
+---------+-------------+----------------------+---------------------+----------+---------------+------------------+--------------+--------------+------------+

No live ``llama-server``, no GPU, no network and no ``/var/log`` access are
needed: the tests read only the two fixture files through the analyser's
existing ``build_report(..., llama_log=...)`` surface (AC6).
"""

from __future__ import annotations

import importlib
import sys
from datetime import datetime
from pathlib import Path

import pytest

PROXY_FIXTURE = (
    Path(__file__).parent / "fixtures" / "abandoned_prefill_release_cancel_proxy.log"
)
LLAMA_FIXTURE = (
    Path(__file__).parent / "fixtures" / "abandoned_prefill_release_cancel_llama.log"
)
WINDOW_START = datetime(2026, 1, 1, 0, 0, 0)
WINDOW_END = datetime(2026, 1, 1, 0, 4, 0)

#: The sibling implementation item that delivered the joiner (formerly pinned
#: here with ``xfail`` markers). See docs on the parent LP-0MVVEQW5Y00962T2 AC2.
IMPL_ITEM = "LP-0MVW8S7FN001B2Y1"


@pytest.fixture(name="script")
def _script():
    """Import the analysis script from ``proxy/scripts`` (no package there)."""
    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module("analyse_prefill_deadline_residual")


@pytest.fixture(name="report")
def _report(script):
    """The analyser dataset over the fixture pair.

    ``build_report`` already accepts ``llama_log=`` today (the optional
    backend accounting cross-check), so this fixture runs against the current
    analyser; only the ``release_cancel_accounting`` block it must carry is
    missing until the implementation item lands.
    """
    proxy_log = script.parse_window(
        script.read_log_lines([PROXY_FIXTURE]), WINDOW_START, WINDOW_END
    )
    return script.build_report(
        proxy_log, WINDOW_START, WINDOW_END, llama_log=str(LLAMA_FIXTURE)
    )


def _rows(report) -> list[dict]:
    """The per-release rows of the joiner contract (raises until impl lands)."""
    return report["release_cancel_accounting"]["releases"]


def _row_for(report, session: str) -> dict:
    rows = [row for row in _rows(report) if row["session"] == session]
    assert len(rows) == 1, f"expected exactly one release row for {session}, got {len(rows)}"
    return rows[0]


# ---------------------------------------------------------------------------
# AC1 — the fixture pair itself (not xfail: these assertions already hold)
# ---------------------------------------------------------------------------


def test_fixture_pair_is_hermetic_and_covers_all_shapes(script):
    """AC1: the committed fixture pair contains all four shapes and is parseable.

    This test verifies the *fixture artifacts* — it is deliberately not
    xfail-marked: the fixtures exist the moment their item lands. The four
    shapes are exercised behaviourally (not textually) by the joiner tests
    below.
    """
    proxy_lines = script.read_log_lines([PROXY_FIXTURE])
    llama_text = LLAMA_FIXTURE.read_text()

    # Every line in the proxy fixture parses inside the fake window, and the
    # window holds exactly the four post-acceptance releases.
    proxy_log = script.parse_window(proxy_lines, WINDOW_START, WINDOW_END)
    assert proxy_log.lines_in_window == 12
    assert len(proxy_log.evictions) == 4
    sessions = {e.session for e in proxy_log.evictions}
    assert sessions == {"user-alpha", "user-bravo", "user-charlie", "user-delta"}

    # Real proxy release line shapes (field order, slot_id=unknown,
    # llama_task_id=unknown).
    assert "lease_released session=user-alpha reason=prefill_deadline " in "\n".join(proxy_lines)
    assert "local_eviction_after_acceptance session=" in "\n".join(proxy_lines)
    assert "slot_id=unknown llama_task_id=unknown endpoint=http://localhost:8080" in "\n".join(
        proxy_lines
    )
    assert "abandoned_slot_recorded session=" in "\n".join(proxy_lines)

    # Shape (a): release -> backend cancel -> progress lines -> stop processing.
    assert "srv          stop: cancel task, id_task = 41" in llama_text
    assert "prompt processing progress, n_tokens = 45000" in llama_text
    assert "slot      release: id  0 | task 41 | stop processing" in llama_text
    # Shape (b): release with NO observed backend cancel for its task.
    assert "slot      release: id  1 | task 43 | stop processing" in llama_text
    assert "stop: cancel task, id_task = 43" not in llama_text
    # Shape (c): a cancel that exists but carries no joinable task id.
    assert "srv    operator(): http client error: Connection handling canceled" in llama_text
    # Shape (d): a cancelled task that still emitted prompt eval time.
    assert "srv          stop: cancel task, id_task = 42" in llama_text
    assert "prompt eval time" in llama_text


# ---------------------------------------------------------------------------
# AC2 — per-release columns
# ---------------------------------------------------------------------------


def test_every_release_reports_the_required_columns(report):
    """AC2: per-release columns — reason, timestamp, task, cancel, latency."""
    expected = {
        "user-alpha": {
            "reason": "prefill_deadline",
            "released_at": "2026-01-01 00:00:00",
            "task_id": 41,
            "unjoined": False,
            "unjoined_reason": None,
            "cancel_observed": True,
            "progress_lines_after_cancel": 2,
            "tokens_evaluated_after_cancel": 10000,
            "prompt_eval_time_emitted": False,
        },
        "user-bravo": {
            "reason": "prefill_deadline",
            "released_at": "2026-01-01 00:01:00",
            "task_id": 43,
            "unjoined": False,
            "unjoined_reason": None,
            "cancel_observed": False,
            "progress_lines_after_cancel": 0,
            "tokens_evaluated_after_cancel": 0,
            "prompt_eval_time_emitted": False,
        },
        "user-charlie": {
            "reason": "measured_cache_deficit",
            "released_at": "2026-01-01 00:02:00",
            "task_id": None,
            "unjoined": True,
            "unjoined_reason": "cancel_without_task_id",
            "cancel_observed": True,
            "progress_lines_after_cancel": 0,
            "tokens_evaluated_after_cancel": 0,
            "prompt_eval_time_emitted": False,
        },
        "user-delta": {
            "reason": "prefill_deadline",
            "released_at": "2026-01-01 00:03:00",
            "task_id": 42,
            "unjoined": False,
            "unjoined_reason": None,
            "cancel_observed": True,
            "progress_lines_after_cancel": 0,
            "tokens_evaluated_after_cancel": 0,
            "prompt_eval_time_emitted": True,
        },
    }
    reported = {row["session"]: row for row in _rows(report)}
    assert set(reported) == set(expected)
    for session, values in expected.items():
        for column, value in values.items():
            assert reported[session][column] == value, (
                f"release {session} column {column!r}: expected {value!r}, "
                f"got {reported[session].get(column)!r}"
            )


def test_cancel_to_stop_latency_is_in_seconds_per_release(report):
    """AC2: the cancel→stop latency column — 0.0 for an immediate stop, no
    fabricated latency when no cancel was observed."""
    alpha = _row_for(report, "user-alpha")  # 10000 tokens evaluated after its cancel
    assert alpha["cancel_to_stop_s"] is not None
    assert alpha["cancel_to_stop_s"] > 0.0
    delta = _row_for(report, "user-delta")  # nothing evaluated after its cancel
    assert delta["cancel_to_stop_s"] == 0.0
    # A release with no observed cancel reports no latency at all.
    assert _row_for(report, "user-bravo")["cancel_to_stop_s"] is None
    assert _row_for(report, "user-charlie")["cancel_to_stop_s"] is None


# ---------------------------------------------------------------------------
# AC3 — nothing is dropped
# ---------------------------------------------------------------------------


def test_every_release_appears_exactly_once_unjoined_is_counted(report):
    """AC3: joined + unjoined sums to the release count; unjoined carries a
    machine-readable reason and is never dropped."""
    rows = _rows(report)
    assert len(rows) == 4
    sessions = [row["session"] for row in rows]
    assert len(set(sessions)) == len(sessions), "a release must appear exactly once"
    unjoined = [row for row in rows if row["unjoined"]]
    assert len(unjoined) == 1
    assert unjoined[0]["session"] == "user-charlie"
    assert isinstance(unjoined[0]["unjoined_reason"], str) and unjoined[0]["unjoined_reason"]
    accounting = report["release_cancel_accounting"]
    assert accounting["releases_unjoined"]["count"] == 1
    assert accounting["releases_total"] == 4
    assert accounting["releases_total"] == len(rows)
    joined = len(rows) - len(unjoined)
    assert joined + accounting["releases_unjoined"]["count"] == accounting["releases_total"]


# ---------------------------------------------------------------------------
# AC4 — no fabricated cancels
# ---------------------------------------------------------------------------


def test_release_without_backend_cancel_is_not_fabricated(report):
    """AC4: shape (b) reports no cancel. A naive time-window join that marks
    everything cancelled because a cancel line exists elsewhere in the window
    cannot pass."""
    bravo = _row_for(report, "user-bravo")
    assert bravo["cancel_observed"] is False
    assert bravo["progress_lines_after_cancel"] == 0
    accounting = report["release_cancel_accounting"]
    with_cancel = [row for row in accounting["releases"] if row["cancel_observed"]]
    assert accounting["releases_with_observed_cancel"] == 3
    assert len(with_cancel) == accounting["releases_with_observed_cancel"]


# ---------------------------------------------------------------------------
# AC5 — distributions, not point estimates
# ---------------------------------------------------------------------------


def test_cancel_to_stop_latency_is_reported_as_a_distribution(report):
    """AC5: the cancel→stop latency is p10/p50/p90 over the per-release values
    (here exactly two: 0.0 for the immediate stop and the joined (a) value)."""
    dist = report["release_cancel_accounting"]["cancel_to_stop_seconds"]
    assert dist["count"] == 2
    for point in ("p10", "p50", "p90", "min", "max"):
        assert isinstance(dist[point], (int, float))
    assert dist["min"] == 0.0
    alpha = _row_for(report, "user-alpha")["cancel_to_stop_s"]
    assert dist["max"] == alpha
    assert dist["p10"] <= dist["p50"] <= dist["p90"]
    assert dist["p10"] >= dist["min"]
    assert dist["p90"] <= dist["max"]


def test_tokens_after_cancel_are_counted_from_progress_lines(report):
    """AC5: tokens evaluated after the cancel come from the fixture's progress
    lines — 10000 for shape (a) (50000 − 40000) — not from the
    ``ABANDONED_PREFILL_RATE_TOK_S`` constant."""
    tokens = report["release_cancel_accounting"]["tokens_evaluated_after_cancel"]
    assert tokens["total"] == 10000
    per_release = {
        row["session"]: row["tokens_evaluated_after_cancel"]
        for row in _rows(report)
    }
    assert per_release == {
        "user-alpha": 10000,
        "user-bravo": 0,
        "user-charlie": 0,
        "user-delta": 0,
    }
    assert tokens["per_release"] == per_release


# ---------------------------------------------------------------------------
# Window aggregates (impl AC1) at the fixture ground truth — pinned so
# the sibling implementation cannot report a different fixture than this one
# ---------------------------------------------------------------------------


def test_window_aggregates_match_the_fixture_ground_truth(report):
    accounting = report["release_cancel_accounting"]
    assert accounting["releases_total"] == 4
    assert accounting["releases_with_observed_cancel"] == 3
    # Only user-charlie's release has no explained backend stop: (b)'s task was
    # stopped (a slot release line), so it is explained even without a cancel.
    assert accounting["releases_without_explained_backend_stop"] == {"count": 1, "share": 0.25}
    assert accounting["releases_unjoined"] == {"count": 1, "reasons": {"cancel_without_task_id": 1}}
