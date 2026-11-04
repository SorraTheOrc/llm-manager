"""Pin the measured abandoned-prefill window model (LP-0MVW8S74G004MFAQ).

The parent LP-0MVVEQW5Y00962T2 needs the 58/60 overlap figure and the 667 s
median backlog restated on a measured basis: the analyser's
``abandoned_prefill_windows()`` currently prices the *whole* remaining prompt
at ``new tokens / ABANDONED_PREFILL_RATE_TOK_S`` (43.1 tok/s), which is the
source of the 667 s median and the 58/60 figure. The sibling implementation
item LP-0MVW8S7FN001B2Y1 re-points the window **end** at the observed backend
stop (cancel → ``slot release: … stop processing``) whenever one exists,
keeping the legacy token/rate estimator as an explicitly labelled fallback.

This module pins that corrected model's **contract** against a hermetic
fixture pair (a proxy-log slice and the matching ``llama-server.log`` slice
covering the same fake window). Every assertion is on the *report surface*
(``build_report(..., llama_log=...)``), never on private helper structure, so
the implementation item is free to refactor ``abandoned_prefill_windows()`` /
``backlog_evidence()`` internals as long as the documented columns below hold.

The corrected window model is implemented by the sibling implementation item
LP-0MVW8S7FN001B2Y1; the assertions below were pinned as
``xfail(strict=True)`` until it landed and the markers have been removed — the
tests now assert the contract directly. (The same red-first pattern was used
for the release→cancel joiner contract, LP-0MVW8S6SR003MBP5.) Two tests are
not xfail: the fixture-integrity test (AC6) verifies the fixture artifacts and
the legacy-compat test pins the pre-existing ``events``/``total`` overlap keys
that must survive the dual-basis change (AC6).

Window-model contract pinned here
---------------------------------

Report surface (all produced from on-disk log slices only; AC6):

``report["abandoned_prefill_windows"]`` — the corrected window model:

- ``legacy_rate_tok_s`` — ``ABANDONED_PREFILL_RATE_TOK_S`` always exported,
  so the legacy basis stays computable forever (backwards comparability);
- ``basis`` — the basis the report's overlap verdict uses: ``"measured_stop"``
  when the backend coverage is established, else ``"legacy_rate_estimate"``;
- ``measured_available`` — whether the llama-server log supplied
  cancel→stop evidence for the window;
- ``rows`` — one row per released dispatch, keyed by the release timestamp:
  - ``start`` / ``end`` — ISO ``%Y-%m-%d %H:%M:%S``; ``start`` is the proxy
    release timestamp, ``end`` is the **observed stop** time when
    ``bounded``, otherwise the legacy estimate;
  - ``duration_s`` — ``end - start`` in seconds;
  - ``bounded`` — whether a backend stop was observed for the release
    (a cancel→``stop processing`` pair);
  - ``basis`` — ``"measured_stop"`` or ``"legacy_rate_estimate"``, i.e. the
    basis that produced **this** row's ``end``;
  - ``estimated`` — ``True`` exactly when the row's duration is the legacy
    token/rate estimate (the two bases are never mixed silently);
  - ``tokens`` — the tokens priced into the window (observed after the cancel
    when bounded, else the release's ``new_tokens``);
  - ``measured_stop_s`` — the observed cancel→stop latency in seconds
    (``None`` when not bounded);
  - ``legacy_duration_s`` — ``tokens / legacy_rate_tok_s``, always present
    for comparison.

``report["attribution"]["abandoned_prefill_overlap"]`` — dual-basis reporting
(AC3). The backwards-compatible ``events``/``total`` keys keep the **verdict**
basis (so the existing ``test_prefill_deadline_residual_analysis.py`` keeps
passing unchanged); ``basis`` states which one the verdict uses, and the
``measured`` / ``legacy`` blocks carry both counts:

- ``events`` — overlap count on the verdict basis;
- ``total`` — number of ``prefill_deadline`` events;
- ``basis`` — ``"measured_stop"`` | ``"legacy_rate_estimate"``;
- ``measured`` — ``{"events", "total", "basis": "measured_stop", "available"}``
  (``events`` is ``None`` when ``available`` is false — never invented);
- ``legacy`` — ``{"events", "total", "basis": "legacy_rate_estimate"}``.

``report["attribution"]["zero_progress_while_draining"]`` — AC5, the second
half of the parent's AC3, reported separately from the share:

- ``count`` — accepted dispatches that expired their lease (post-acceptance
  evictions) with no ``prefill_progress_measured`` for their session inside
  the dispatch window, whose dispatch start still falls inside an abandoned
  window on the verdict basis;
- ``total`` — all post-acceptance lease expiries in the window;
- ``basis`` — the basis the count uses.

Per-event evidence (``report["events"][i]["evidence"]``) keeps
``abandoned_prefill_overlap`` / ``abandoned_prefill_remaining_s`` on the
verdict basis and adds ``abandoned_prefill_basis``,
``abandoned_prefill_overlap_measured`` and ``abandoned_prefill_overlap_legacy``
so a reader can re-derive AC4's discriminator.

Fixture ground truth (window 2026-02-01 00:00:00 → 00:10:00):

+-----+-----------+---------------------+--------------------+---------+-----------------+------------------------------------------+
| row | session   | released_at         | measured (bounded) | legacy  | dispatch_start  | overlap (measured / legacy)               |
+=====+===========+=====================+====================+=========+=================+==========================================+
| W1  | user-one  | 2026-02-01 00:00:00 | end 00:00:16       | 118.79s | 23:57:59.9 (prev)| false / false                             |
+-----+-----------+---------------------+--------------------+---------+-----------------+------------------------------------------+
| W4  | user-one  | 2026-02-01 00:02:05 | unbounded          | 696.06s | 00:00:05         | **true** / true  (retry before stop)       |
+-----+-----------+---------------------+--------------------+---------+-----------------+------------------------------------------+
| W3  | user-one  | 2026-02-01 00:02:30 | unbounded          | 696.06s | 00:00:30         | **false** / true (retry after stop)        |
+-----+-----------+---------------------+--------------------+---------+-----------------+------------------------------------------+
| W2  | user-two  | 2026-02-01 00:05:00 | unbounded          | 129.93s | 00:03:00         | **false** / true (inside W4/W3 legacy ends)|
+-----+-----------+---------------------+--------------------+---------+-----------------+------------------------------------------+

Measured overlap count 1/4 (W4 only); legacy overlap count 3/4 (W4, W3, W2);
``zero_progress_while_draining.count == 1`` (only W4's dispatch starts inside
W1's *measured* window). The two bases deliberately disagree on W3, so AC3's
dual-basis reporting is observable, not a tautology.

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
    Path(__file__).parent / "fixtures" / "abandoned_prefill_windows_measured_proxy.log"
)
LLAMA_FIXTURE = (
    Path(__file__).parent / "fixtures" / "abandoned_prefill_windows_measured_llama.log"
)
WINDOW_START = datetime(2026, 2, 1, 0, 0, 0)
WINDOW_END = datetime(2026, 2, 1, 0, 10, 0)

#: The sibling implementation item that delivered the measured window basis
#: (formerly pinned here with ``xfail`` markers). See docs on the parent
#: LP-0MVVEQW5Y00962T2 AC2/AC3 (measurement half).
IMPL_ITEM = "LP-0MVW8S7FN001B2Y1"

#: Ground truth durations for the fixture (computed from the fixture's own
#: numbers so the tests never re-derive them from the analyser's constant).
LEGACY_RATE = 43.1
W1_LEGACY_DURATION_S = 5120 / LEGACY_RATE  # 118.793…
W2_LEGACY_DURATION_S = 5600 / LEGACY_RATE  # 129.930…
RETRY_NEW_TOKENS = 30000
RETRY_LEGACY_DURATION_S = RETRY_NEW_TOKENS / LEGACY_RATE  # 696.056…
W1_MEASURED_DURATION_S = 10000 / 625.0  # 16.0 — observed rate from the fixture


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
    analyser; only the ``abandoned_prefill_windows`` block and the dual-basis
    overlap reporting it must carry are missing until the implementation item
    lands.
    """
    proxy_log = script.parse_window(
        script.read_log_lines([PROXY_FIXTURE]), WINDOW_START, WINDOW_END
    )
    return script.build_report(
        proxy_log, WINDOW_START, WINDOW_END, llama_log=str(LLAMA_FIXTURE)
    )


@pytest.fixture(name="report_without_llama")
def _report_without_llama(script):
    """The same window measured without a backend log: legacy basis only."""
    proxy_log = script.parse_window(
        script.read_log_lines([PROXY_FIXTURE]), WINDOW_START, WINDOW_END
    )
    return script.build_report(proxy_log, WINDOW_START, WINDOW_END)


def _window_rows(report) -> list[dict]:
    """The corrected-window-model rows (raises until impl lands)."""
    return report["abandoned_prefill_windows"]["rows"]


def _row_at(report, released_at: str) -> dict:
    rows = [row for row in _window_rows(report) if row["start"] == released_at]
    assert len(rows) == 1, (
        f"expected exactly one window row released at {released_at}, "
        f"got {[(r['start'], r['session']) for r in rows]}"
    )
    return rows[0]


def _evidence_row(report, ts: str) -> dict:
    """The per-event evidence for the ``prefill_deadline`` eviction at *ts*."""
    rows = [row for row in report["events"] if row["ts"] == ts]
    assert len(rows) == 1, f"expected one event at {ts}, got {len(rows)}"
    return rows[0]["evidence"]


# ---------------------------------------------------------------------------
# AC6 — the fixture pair itself (not xfail: these assertions already hold)
# ---------------------------------------------------------------------------


def test_fixture_pair_is_hermetic_and_covers_all_shapes(script):
    """AC6: the committed fixture pair parses and contains all four shapes.

    This test verifies the *fixture artifacts* — it is deliberately not
    xfail-marked: the fixtures exist the moment their item lands. The window
    model shapes (measured stop, unbounded, retry before stop, retry after
    stop) are exercised behaviourally by the contract tests below.
    """
    proxy_lines = script.read_log_lines([PROXY_FIXTURE])
    llama_text = LLAMA_FIXTURE.read_text()

    # Every line in the proxy fixture parses inside the fake window and the
    # window holds exactly the four prefill_deadline releases.
    proxy_log = script.parse_window(proxy_lines, WINDOW_START, WINDOW_END)
    assert proxy_log.lines_in_window == 4
    assert len(proxy_log.evictions) == 4
    assert {e.reason for e in proxy_log.evictions} == {"prefill_deadline"}
    assert {e.session for e in proxy_log.evictions} == {"user-one", "user-two"}

    # Real proxy release line shape (field order, slot_id=unknown,
    # llama_task_id=unknown).
    assert "local_eviction_after_acceptance session=" in "\n".join(proxy_lines)
    assert "slot_id=unknown llama_task_id=unknown endpoint=http://localhost:8080" in "\n".join(
        proxy_lines
    )

    # Shape pair (a): release -> cancel -> progress after -> stop processing.
    assert "stop: cancel task, id_task = 51" in llama_text
    assert "prompt processing progress, n_tokens = 50000" in llama_text
    assert "slot      release: id  0 | task 51 | stop processing" in llama_text
    # Observed rate the measured 16.0 s window comes from.
    assert "625.00 tokens per second" in llama_text
    # Shapes (b/c/d): no cancel and no stop for the retry/unbounded segments.
    assert "stop: cancel task, id_task = 52" not in llama_text
    assert "stop: cancel task, id_task = 53" not in llama_text
    assert "stop: cancel task, id_task = 54" not in llama_text
    stop_lines = [ln for ln in llama_text.splitlines() if ln.lstrip().startswith("[") and "stop processing:" in ln]
    assert stop_lines == ["[54819] slot      release: id  0 | task 51 | stop processing: n_tokens = 50000, truncated = 0"]


# ---------------------------------------------------------------------------
# AC1 — window end comes from the observed stop
# ---------------------------------------------------------------------------


def test_bounded_window_end_is_the_observed_stop(report):
    """AC1: W1's window ends at the observed stop, not the token/rate estimate.

    Task 51 was cancelled and stopped 16.0 s after the release (10000 tokens
    after the cancel at the fixture's observed 625 tok/s), so the window is
    [00:00:00, 00:00:16]. The row must record which basis produced it.
    """
    row = _row_at(report, "2026-02-01 00:00:00")
    assert row["bounded"] is True
    assert row["basis"] == "measured_stop"
    assert row["estimated"] is False
    assert row["end"] == "2026-02-01 00:00:16"
    assert row["duration_s"] == pytest.approx(W1_MEASURED_DURATION_S, abs=1.0)
    assert row["measured_stop_s"] == pytest.approx(W1_MEASURED_DURATION_S, abs=1.0)
    # The duration is NOT the token/rate estimate for this release.
    assert row["legacy_duration_s"] == pytest.approx(W1_LEGACY_DURATION_S, abs=1.0)
    assert abs(row["duration_s"] - row["legacy_duration_s"]) > 20.0


# ---------------------------------------------------------------------------
# AC2 — unbounded windows are labelled, not invented
# ---------------------------------------------------------------------------


def test_unbounded_window_is_explicitly_legacy_estimate(report):
    """AC2: no stop observed => bounded=false and the legacy token/rate
    estimate carries an explicit estimate label; the two bases are never mixed
    silently."""
    row = _row_at(report, "2026-02-01 00:05:00")  # user-two, task 54 never stops
    assert row["bounded"] is False
    assert row["basis"] == "legacy_rate_estimate"
    assert row["estimated"] is True
    assert row["measured_stop_s"] is None
    assert row["duration_s"] == pytest.approx(W2_LEGACY_DURATION_S, abs=1.0)
    assert row["legacy_duration_s"] == pytest.approx(W2_LEGACY_DURATION_S, abs=1.0)

    # The retries are unbounded too (their backend tasks never stop).
    for released_at in ("2026-02-01 00:02:05", "2026-02-01 00:02:30"):
        retry = _row_at(report, released_at)
        assert retry["bounded"] is False
        assert retry["basis"] == "legacy_rate_estimate"
        assert retry["estimated"] is True
        assert retry["duration_s"] == pytest.approx(RETRY_LEGACY_DURATION_S, abs=1.0)


def test_no_backend_log_degrades_to_legacy_basis_without_inventing(report_without_llama):
    """AC2/AC3: without a llama-server log every window is an unbounded legacy
    estimate and the report says the measured basis is unavailable — it never
    invents a measured overlap."""
    windows = report_without_llama["abandoned_prefill_windows"]
    assert windows["basis"] == "legacy_rate_estimate"
    assert windows["measured_available"] is False
    assert all(row["bounded"] is False for row in windows["rows"])
    assert all(row["basis"] == "legacy_rate_estimate" for row in windows["rows"])

    overlap = report_without_llama["attribution"]["abandoned_prefill_overlap"]
    assert overlap["basis"] == "legacy_rate_estimate"
    assert overlap["events"] == 3  # W4, W3, W2 on the legacy basis
    assert overlap["legacy"]["events"] == 3
    assert overlap["measured"]["available"] is False
    assert overlap["measured"]["events"] is None  # unmeasured, not zero


# ---------------------------------------------------------------------------
# AC3 — dual-basis reporting
# ---------------------------------------------------------------------------


def test_both_overlap_bases_are_reported_and_the_verdict_basis_stated(report, script):
    """AC3: the 58/60-style overlap is reported on the measured AND the legacy
    basis, and the report states which one the verdict uses (here measured,
    because the llama log covers the window)."""
    windows = report["abandoned_prefill_windows"]
    assert windows["legacy_rate_tok_s"] == script.ABANDONED_PREFILL_RATE_TOK_S == LEGACY_RATE
    assert windows["measured_available"] is True
    assert windows["basis"] == "measured_stop"

    overlap = report["attribution"]["abandoned_prefill_overlap"]
    assert overlap["total"] == 4
    assert overlap["basis"] == "measured_stop"
    # W4 starts inside W1's measured window; W3 and W2 start after it.
    assert overlap["measured"] == {
        "events": 1,
        "total": 4,
        "basis": "measured_stop",
        "available": True,
    }
    # On the legacy basis, W4, W3 AND W2 all fall inside an estimated window.
    assert overlap["legacy"] == {
        "events": 3,
        "total": 4,
        "basis": "legacy_rate_estimate",
        "available": True,
    }
    # The verdict key keeps the measured count: 1/4, not 3/4.
    assert overlap["events"] == 1


# ---------------------------------------------------------------------------
# AC4 — a retry inside a cancelled-and-stopped window is not "starved"
# ---------------------------------------------------------------------------


def test_retry_before_the_observed_stop_is_inside_the_measured_window(report):
    """AC4: the retry dispatched at 00:00:05 starts before W1's observed stop
    (00:00:16), so it IS inside the measured abandoned window."""
    evidence = _evidence_row(report, "2026-02-01 00:02:05")
    assert evidence["abandoned_prefill_basis"] == "measured_stop"
    assert evidence["abandoned_prefill_overlap_measured"] is True
    assert evidence["abandoned_prefill_overlap"] is True  # verdict = measured


def test_retry_after_the_observed_stop_is_not_counted_as_starved(report):
    """AC4: the retry dispatched at 00:00:30 starts after W1's observed stop
    (00:00:16). It must NOT be counted as starved on the measured basis — even
    though the legacy estimator (window to 00:01:58.79) still covers it."""
    evidence = _evidence_row(report, "2026-02-01 00:02:30")
    assert evidence["abandoned_prefill_basis"] == "measured_stop"
    assert evidence["abandoned_prefill_overlap_measured"] is False
    assert evidence["abandoned_prefill_overlap_legacy"] is True
    assert evidence["abandoned_prefill_overlap"] is False  # verdict = measured


# ---------------------------------------------------------------------------
# AC5 — zero-progress while draining is counted separately from the share
# ---------------------------------------------------------------------------


def test_zero_progress_lease_inside_a_window_is_counted(report):
    """AC5: of the four zero-progress lease expiries, exactly one (W4, whose
    dispatch starts inside W1's measured window) is a starvation; the count is
    reported separately from the overlap share."""
    draining = report["attribution"]["zero_progress_while_draining"]
    assert draining["basis"] == "measured_stop"
    assert draining["total"] == 4
    assert draining["count"] == 1

    # Nothing in the fixture reports measured progress at all.
    assert all("prefill_progress_measured" not in line for line in PROXY_FIXTURE.read_text().splitlines())


# ---------------------------------------------------------------------------
# Legacy compatibility — the existing analyser contract keeps working
# ---------------------------------------------------------------------------


def test_legacy_overlap_columns_keep_the_backwards_compatible_keys(report):
    """The pre-existing overlap keys (events/total) must remain present so
    ``test_prefill_deadline_residual_analysis.py`` keeps passing unchanged
    (AC6). This holds as soon as the dual-basis block lands; it is left
    xfail-free on purpose — the legacy contract is not new behaviour."""
    overlap = report["attribution"]["abandoned_prefill_overlap"]
    assert "total" in overlap
    assert "events" in overlap
