"""Interim 120 s post-deadline prefill-thrash cooldown (LP-0MVWC8JCG007GAPW).

The measured-cache early abort (`LP-0MVR7IN0O000DPXN`) fires ~0.1 s into a
dispatch and then armed a **600 s** per-session thrash cooldown. For a
pin-local (``X-Local-Preference: local-preferred``) client — which by contract
receives ``429 Local admission refused`` instead of a remote fallback — that
0.1 s GPU saving became a ten-minute outage. Measured 2026-10-31
(``herdr-1793416360-4155786-6690``): 330 aborts across 28 sessions, 249 of them
at ``deficit_ratio=1.0``, and 113 refusals for that one session.

The operator approved reducing the cooldown to **120 s** (option B): two client
retry intervals, which still suppresses the retry storm the cooldown exists for
but caps a pinned session's outage at two minutes.

Acceptance criteria covered here:

- AC1: the shipped ``proxy/config.yaml`` value is 120 and the abort/budget
  knobs it must not disturb keep their values.
- AC2: the armed window and the ``cooldown_s=`` log field are driven by the
  configured value, never by a literal.
- AC3: each abort still arms exactly one window (the thrash defence is
  shortened, not weakened).
- AC5: the module default and the routing doc state the new value and say it is
  to be re-tuned once the benefit gate lands (``LP-0MVWBD91Z004U135``).

Behaviour is asserted through the real ``proxy_to_local`` abort path by reusing
the measured-cache-abort wiring harness (no second copy of that scaffolding).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import test_measured_cache_abort_wiring as wiring
import yaml

from proxy import router_helpers as helpers

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "proxy" / "config.yaml"
ROUTING_DOC = REPO_ROOT / "proxy" / "docs" / "routing.md"

#: The value this item ships (operator decision 2026-10-31).
SHIPPED_COOLDOWN_SECONDS = 120


def _server_section() -> dict:
    with CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)["server"]


def _started_lines() -> list[str]:
    """Every ``prefill_thrash_cooldown_started`` line the abort path logged."""
    return [
        line
        for line in wiring._all_log_lines("info")
        if line.startswith("prefill_thrash_cooldown_started")
    ]


# ---------------------------------------------------------------------------
# AC1 — the shipped value, and the knobs it must not move
# ---------------------------------------------------------------------------


def test_shipped_config_arms_a_two_minute_cooldown():
    """AC1: the deployed config arms 120 s, not the 600 s it replaced."""
    assert _server_section()["local_prefill_thrash_cooldown_seconds"] == (
        SHIPPED_COOLDOWN_SECONDS
    )


def test_shipped_config_leaves_the_abort_and_budget_knobs_untouched():
    """AC1: this is a one-value change — the abort keeps its full strength.

    Shortening the cooldown must not be smuggled in via a weakened abort: the
    measured-cache early abort stays enabled with its 2 000-token bound and
    0.25 deficit floor, and the prefill budget/deadline are unchanged.
    """
    server = _server_section()
    assert server["local_measured_cache_abort_enabled"] is True
    assert server["local_measured_cache_abort_bound_tokens"] == 2_000
    assert server["local_measured_cache_abort_min_deficit_ratio"] == 0.25
    assert server["local_prefill_deadline_seconds"] == 120
    assert server["local_cold_prefill_budget_tokens"] == 25_000


def test_module_default_cannot_drift_from_the_shipped_value():
    """AC1: a missing config key must not reintroduce the rejected 600 s.

    ``_get_prefill_thrash_cooldown_seconds`` falls back to the module constant
    when the key is absent (e.g. a partial config in a deployment or a test
    harness). If that constant disagreed with ``config.yaml`` a config omission
    would silently restore the value this item removed.
    """
    assert helpers.DEFAULT_PREFILL_THRASH_COOLDOWN_SECONDS == float(
        _server_section()["local_prefill_thrash_cooldown_seconds"]
    )


# ---------------------------------------------------------------------------
# AC2 — the window and its log field follow the configured value
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_armed_window_and_log_follow_the_configured_value(monkeypatch):
    """AC2: a distinctive configured value is what the abort actually arms.

    A hard-coded 600 (or 120) anywhere in the arm/log path fails this: the
    window length and the ``cooldown_s=`` field must both report 45.
    """
    _, _, _ = await wiring._drive_local(
        [
            wiring._fake_warm_progress(),
            wiring._CONTENT_CHUNK,
            wiring._FINISH_CHUNK,
            wiring._DONE,
        ],
        monkeypatch=monkeypatch,
        server_config_extra={"local_prefill_thrash_cooldown_seconds": 45},
    )

    started = _started_lines()
    assert len(started) == 1, f"expected one arming line, got {started}"
    assert "cooldown_s=45" in started[0], started[0]

    remaining = helpers._prefill_thrash_cooldown_remaining(
        wiring.SESSION, wiring.server.config
    )
    assert 40 < remaining <= 45, (
        f"the armed window must be the configured 45 s, got {remaining}"
    )


# ---------------------------------------------------------------------------
# AC3 — the defence is shortened, never weakened
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_config_arms_exactly_one_two_minute_window_per_abort(
    monkeypatch,
):
    """AC3: an abort still arms one 120 s window (no double-arm, no silence).

    The harness config omits the key, so this also pins the module-default
    path an unconfigured deployment would take.
    """
    _, received, _ = await wiring._drive_local(
        [
            wiring._fake_warm_progress(),
            wiring._CONTENT_CHUNK,
            wiring._FINISH_CHUNK,
            wiring._DONE,
        ],
        monkeypatch=monkeypatch,
    )

    # The abort itself is unchanged: no content escaped and local was released.
    assert received == b""
    started = _started_lines()
    assert len(started) == 1, f"expected exactly one arming line, got {started}"
    assert "cooldown_s=120" in started[0], started[0]
    assert helpers._prefill_thrash_cooldown_remaining(
        wiring.SESSION, wiring.server.config
    ) > 0


# ---------------------------------------------------------------------------
# AC5 — the docs carry the decision and the condition for re-tuning it upward
# ---------------------------------------------------------------------------


def test_routing_doc_records_the_new_default_and_the_re_tune_condition():
    """AC5: 120 s must not read as permanent — the benefit gate supersedes it."""
    section = ROUTING_DOC.read_text()
    start = section.index("#### Post-deadline prefill-thrash cooldown")
    end = section.index("#### Regression coverage for the prefill-thrash wedge")
    thrash_doc = section[start:end]

    assert "120" in thrash_doc, "the new default is not documented"
    assert "LP-0MVWBD91Z004U135" in thrash_doc, (
        "the doc must name the benefit-gate item that supersedes this value"
    )
