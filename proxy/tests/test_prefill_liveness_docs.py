"""Documentation-contract pin for prefill-marker liveness (LP-0MWL5KMUW002441T).

Parent: **LP-0MWKWGLRT003AUG4** — *Orphaned ``local_prefill_in_flight`` marker
closes the local route for every session*, AC7 (documentation).

The behaviour is unit-tested by the three implementation items; this module
pins the **shipped code ↔ operator documentation** axis, which nothing else
covers:

- ``proxy/docs/routing.md`` prices the prefill cap on the *live* count, states
  the per-deletion pop invariant, and documents the bounded reclaim sweep — and
  no longer says an inactive/unexpired record backs the count;
- the ``local_dispatch_denied`` evidence block explains ``prefill_in_flight``
  as the gate's own live count (so it agrees with
  ``/llama/local/status``'s ``local_prefills_in_flight``) and names
  ``prefill_marker_reclaimed`` as the recovery signal;
- the motivating 2026-11-17 measurement and the manual
  ``POST /v1/leases/release`` fallback stay recorded, in both the routing doc
  and the endpoint's README section;
- the doc names the helpers the code actually exports, so a rename or a
  change of priced quantity fails here instead of silently invalidating the
  prose.

Pattern: ``test_contention_queue_docs.py`` (shipped-fact documentation pin).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTING_DOC = REPO_ROOT / "proxy" / "docs" / "routing.md"
README = REPO_ROOT / "proxy" / "README.md"
ROUTER_HELPERS = REPO_ROOT / "proxy" / "proxy" / "router_helpers.py"
SERVER = REPO_ROOT / "proxy" / "proxy" / "server.py"

#: The stale wording the docs change replaced: the live count is filtered on an
#: ``active`` record, not on an unexpired one (an inactive post-request lease
#: still has a future ``expires_at`` and must not back the count).
STALE_FILTER_CLAIM = "active/unexpired"


def _section(text: str, heading: str, next_heading: str) -> str:
    """Return the prose between *heading* and *next_heading* (exclusive)."""
    start = text.index(heading)
    end = text.index(next_heading, start)
    return text[start:end]


def _flat(text: str) -> str:
    """Collapse whitespace so a phrase assertion ignores markdown line wraps."""
    return " ".join(text.split())


def _liveness_section() -> str:
    return _section(
        ROUTING_DOC.read_text(encoding="utf-8"),
        "### Stale records never block (AC1)",
        "### The denial line carries its evidence (AC2)",
    )


def _evidence_section() -> str:
    text = ROUTING_DOC.read_text(encoding="utf-8")
    return _section(
        text,
        "### The denial line carries its evidence (AC2)",
        "### Divergence is logged, not shipped silently (AC3)",
    )


def test_docs_price_the_cap_on_the_live_count_and_name_the_shipped_helper():
    """AC1: the priced quantity in the prose is the one the gate computes."""
    section = _liveness_section()
    assert "_get_live_prefill_in_flight_count" in section, (
        "routing.md must name the liveness-filtered helper that prices the cap"
    )
    assert "`active` dispatch record" in section, (
        "routing.md must state that only an active record backs a live prefill"
    )
    # Shipped code: the gate really prices the live helper...
    helpers = ROUTER_HELPERS.read_text(encoding="utf-8")
    assert "_prefill_count = _get_live_prefill_in_flight_count(srv)" in helpers
    # ...and the live helper really filters on ``active``.
    live_helper = helpers[
        helpers.index("def _get_live_prefill_in_flight_count") : helpers.index(
            "def _pop_prefill_marker"
        )
    ]
    assert 'record.get("active")' in live_helper, (
        "the live count must filter on the record's active flag; the docs cite it"
    )


def test_docs_state_the_deletion_invariant_and_the_bounded_sweep():
    """AC1+AC2: pop invariant and one-tick reclaim are documented as shipped."""
    section = _liveness_section()
    assert "_pop_prefill_marker" in section, (
        "the shared pop helper must be named where the invariant is stated"
    )
    for deletion_path in (
        "_prune_stale_dispatch_records",
        "client-disconnect",
        "session invalidation",
        "`POST /v1/leases/release`",
    ):
        assert deletion_path in section, (
            f"routing.md must cover the {deletion_path} deletion path"
        )
    assert "_reclaim_orphaned_prefill_markers" in section
    assert "_dispatch_cleanup_loop" in section
    assert "prefill_marker_reclaimed session=<id> reason=orphaned" in section, (
        "the operator-facing recovery log token must be documented verbatim"
    )
    assert "no proxy restart" in _flat(section), (
        "the recovery bound (one tick, no restart) must be stated"
    )
    # Shipped code emits exactly that token in the cleanup loop.
    server = SERVER.read_text(encoding="utf-8")
    assert "prefill_marker_reclaimed session=%s reason=orphaned" in server


def test_docs_record_the_incident_and_keep_the_manual_fallback():
    """AC1+AC2: the motivating measurement and the manual release survive."""
    section = _liveness_section()
    assert "2026-11-17" in section
    assert "14:24:42" in section
    assert "01a1e9e4-90be-70d7-a899-aa39b5c753ca" in section, (
        "the leaked session id makes the measurement attributable"
    )
    assert "records_total=0" in section and "prefill_in_flight=1" in section
    assert "llama_available_slots=2" in section
    assert "`POST /v1/leases/release" in section, (
        "the manual release must stay documented as the fallback"
    )
    readme = README.read_text(encoding="utf-8")
    lease_release = _section(readme, "#### Lease Release", "#### Dispatch Lease")
    assert "local_prefill_in_flight" in lease_release, (
        "the endpoint's README section must say it pops the marker"
    )
    assert "prefill_marker_reclaimed" in lease_release, (
        "the README must name the automatic reclaim signal beside the fallback"
    )


def test_evidence_block_describes_the_liveness_filtered_count():
    """AC2: the denial-line field is the live count and cannot disagree."""
    section = _evidence_section()
    assert "`prefill_in_flight`" in section
    assert "_get_live_prefill_in_flight_count" in section, (
        "the field's meaning must be tied to the helper that computes it"
    )
    assert "local_prefills_in_flight" in section, (
        "the status-payload cross-check must be stated"
    )
    assert "cannot contradict the status payload" in _flat(section)


def test_no_stale_raw_or_unexpired_filter_claim_survives():
    """AC1+AC3: the replaced wording is gone from the routing doc."""
    text = ROUTING_DOC.read_text(encoding="utf-8")
    assert STALE_FILTER_CLAIM not in text, (
        "the count filters on `active`, not on `active/unexpired`"
    )
    assert re.search(
        r"raw `_get_prefill_in_flight_count` is retained for", text
    ), "the raw count must be documented as observability-only in the cap prose"
