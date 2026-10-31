"""Documentation-drift tests for the "contention queue = straggler net" docs.

Work item LP-0MVT114HU009UNML (docs child of LP-0MVIKPRMD000Y97R). The queue's
*behaviour* is unit-tested in ``test_contention_queue.py``; this module pins the
**shipped-config ↔ documentation** axis, which nothing else covers:

- the shipped caps (``contention_queue_max_wait_seconds`` /
  ``contention_queue_max_depth``) are the values
  ``proxy/docs/routing.md`` and ``README.md`` state. The expected numbers are
  **derived from ``proxy/config.yaml``**, not hard-coded, so a future sweep
  cannot silently leave the docs stale — the test fails instead;
- the docs frame the queue as a bounded **straggler safety net** whose primary
  cooperative path is a reservation (``POST /llama/local/reserve``,
  ``X-Local-Reservation``, ``X-Local-Preference``), not a wait in the queue;
- the record of the 60 → 30 / 8 → 2 change keeps its measurement windows, the
  three flip criteria and the two-line rollback, so the decision can be
  re-judged — and reverted — from the documentation alone;
- no contradictory queue value survives in a touched file: a stale ``120s`` /
  ``depth 8`` / ``60 s`` claim is exactly the drift this test exists to catch.

Pattern: ``test_idle_admit_config_docs.py`` (shipped-configuration drift pin).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "proxy" / "config.yaml"
ROUTING_DOC = REPO_ROOT / "proxy" / "docs" / "routing.md"
README = REPO_ROOT / "README.md"
SWEEP_RUNBOOK = REPO_ROOT / "docs" / "dev" / "contention-queue-sweep.md"

#: Values the sweep (LP-0MVT1146V007WXA4) replaced. They survive only in the
#: documented rollback, which is why they are named here rather than derived.
ROLLBACK_WAIT_SECONDS = 60
ROLLBACK_MAX_DEPTH = 8

#: A bigger/longer queue reverts the straggler-net decision, which needs its own
#: measurement window and flip verdict — never a silent config edit.
STRAGGLER_NET_MAX_WAIT_SECONDS = 60
STRAGGLER_NET_MAX_DEPTH = 4

#: Stale *queue* claims — each pattern asserts a queue value that is no longer
#: shipped. Every pattern is queue-scoped on purpose: `120s` alone also appears
#: in unrelated places (the upstream idle timeout), which is not this test's
#: business. Rollback YAML (``contention_queue_max_wait_seconds: 60``) is not
#: matched either: it states the value to restore, not the value in force.
STALE_QUEUE_VALUE_PATTERNS = (
    r"\bdepth 8\b",
    r"\bdepth of 8\b",
    r"`8`\s*\|\s*`120s`",
    r"\bwait 120s?\b",
    r"\bwait 60s\b",
    r"\bwait 60 s\b",
    r"\bwait 45s?\b",
    r"contention_queue_max_wait_seconds`?\s*\((?:60|120)\b",
    r"contention_queue_max_depth`?\s*\((?:8|4)\b",
)


def _server_config() -> dict:
    with open(CONFIG_PATH) as handle:
        config = yaml.safe_load(handle) or {}
    return config["server"]


def _shipped_caps() -> tuple[int, int]:
    """Return ``(max_wait_seconds, max_depth)`` exactly as shipped."""
    server = _server_config()
    return (
        int(server["contention_queue_max_wait_seconds"]),
        int(server["contention_queue_max_depth"]),
    )


def test_shipped_config_stays_a_bounded_straggler_net():
    """The live caps keep the queue a net, not a parking lot."""
    wait, depth = _shipped_caps()
    assert wait <= STRAGGLER_NET_MAX_WAIT_SECONDS
    assert depth <= STRAGGLER_NET_MAX_DEPTH


def test_config_comment_frames_the_queue_as_a_straggler_net():
    """AC1: the shipped config's own comment tells the same story as the docs."""
    text = CONFIG_PATH.read_text()
    assert "STRAGGLER SAFETY NET" in text
    assert "parking lot" in text
    # The same measurement windows and the same two-line rollback.
    assert "2026-10-22 19:00" in text
    assert "2026-10-30 05:00" in text
    assert "flip criteria" in text
    assert f"contention_queue_max_wait_seconds: {ROLLBACK_WAIT_SECONDS}" in text
    assert f"contention_queue_max_depth: {ROLLBACK_MAX_DEPTH}" in text


def test_routing_doc_states_the_shipped_caps():
    routing = ROUTING_DOC.read_text()
    wait, depth = _shipped_caps()
    # Config reference block: the copyable YAML an operator would apply.
    assert f"contention_queue_max_wait_seconds: {wait}" in routing
    assert f"contention_queue_max_depth: {depth}" in routing
    # Per-priority table rows (low / medium in the shipped table).
    assert routing.count(f"| `{depth}` | `{wait}s` |") == 2


def test_readme_states_the_shipped_caps():
    readme = README.read_text()
    wait, depth = _shipped_caps()
    # The per-priority shipped-defaults table (low and medium rows).
    assert readme.count(f"| `{depth}` | `{wait}s` |") == 2
    assert "contention_queue_max_wait_seconds" in readme
    assert "contention_queue_max_depth" in readme


def test_routing_doc_frames_reservations_as_the_primary_cooperative_path():
    routing = ROUTING_DOC.read_text()
    assert "straggler safety net" in routing
    assert "POST /llama/local/reserve" in routing
    assert "X-Local-Reservation" in routing
    assert "X-Local-Preference" in routing
    # The queue is the fallback path for non-cooperating clients, not the
    # mechanism a well-behaved dispatcher is expected to use.
    assert "primary cooperative path" in routing


def test_readme_frames_the_queue_as_a_straggler_net_and_points_at_the_docs():
    readme = README.read_text()
    assert "straggler net" in readme
    assert "llama/local/reserve" in readme
    assert "X-Local-Reservation" in readme
    # The reader is sent to the canonical semantics + evidence.
    assert "proxy/docs/routing.md" in readme
    assert "docs/dev/contention-queue-sweep.md" in readme


def test_routing_doc_records_the_sweep_evidence_criteria_and_rollback():
    routing = ROUTING_DOC.read_text()
    # Measurement windows (the two the decision rests on).
    assert "2026-10-22" in routing
    assert "2026-10-30 05:00" in routing
    assert "3.3 h intake window" in routing
    assert "24 h pre-flip baseline" in routing
    # The three flip criteria, each with its bar.
    assert "dead wait per hour falls by **>= 50 %**" in routing
    assert ">= 20 %" in routing
    assert "< 20 %" in routing
    # The two-line rollback restores the pre-sweep pair.
    assert f"contention_queue_max_wait_seconds: {ROLLBACK_WAIT_SECONDS}" in routing
    assert f"contention_queue_max_depth: {ROLLBACK_MAX_DEPTH}" in routing
    assert "proxy restart" in routing
    # Traceability to the epic that owns the decision.
    assert "LP-0MVIKPRMD000Y97R" in routing


def test_routing_doc_points_at_the_measurement_runbook():
    routing = ROUTING_DOC.read_text()
    assert SWEEP_RUNBOOK.exists()
    assert "docs/dev/contention-queue-sweep.md" in routing


def test_no_stale_queue_values_remain_in_the_touched_docs():
    """AC7: a stale 120s / depth 8 / 60 s *queue* claim must not survive."""
    wait, depth = _shipped_caps()
    for path in (ROUTING_DOC, README, CONFIG_PATH):
        text = path.read_text()
        for pattern in STALE_QUEUE_VALUE_PATTERNS:
            matches = re.findall(pattern, text)
            assert not matches, (
                f"{path.relative_to(REPO_ROOT)} still claims a stale queue value "
                f"{pattern!r}: {matches[:3]} (shipped: {wait} s / depth {depth})"
            )
        # Every YAML assignment of a queue cap states either the shipped value or
        # the documented rollback target — never a third, half-remembered number.
        cap_assignments = {
            "contention_queue_max_wait_seconds": (wait, ROLLBACK_WAIT_SECONDS),
            "contention_queue_max_depth": (depth, ROLLBACK_MAX_DEPTH),
        }
        for key, (shipped, rollback) in cap_assignments.items():
            stated = {int(value) for value in re.findall(rf"{key}:\s*(\d+)", text)}
            assert stated <= {shipped, rollback}, (
                f"{path.relative_to(REPO_ROOT)} states {key} value(s) "
                f"{sorted(stated - {shipped, rollback})} — neither the shipped "
                f"{shipped} nor the rollback target {rollback}"
            )
        # A queue sentence must never state a wait budget other than the shipped
        # one (a line quoting the rollback target is the one exemption).
        for line in text.splitlines():
            if "contention_queue" not in line:
                continue
            if f"contention_queue_max_wait_seconds: {ROLLBACK_WAIT_SECONDS}" in line:
                continue  # documented rollback target
            assert not re.search(r"\b(?:45|60|120)\s*s\b", line), (
                f"{path.relative_to(REPO_ROOT)} states a stale wait budget: {line!r}"
            )
