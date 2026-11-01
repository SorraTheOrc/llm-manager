"""Documentation-drift pins for the non-quarantining deficit detour.

LP-0MVWFKJ71006U6K2 (parent LP-0MVWEZCB0006WPBQ AC6, plus the AC10/AC11
documentation constraints).

The documentation *is* the deliverable here, so each assertion compares the
shipped prose with the code that enforces it rather than freezing a copy of the
prose:

* the documented ``/llama/local/status`` route-field defaults (``true`` /
  ``null`` / ``null``) are what
  :func:`proxy.observability.local_route_availability` actually returns with
  nothing armed, and the documented field names are that helper's own output
  keys (minus the log-only failure count);
* the documented ``local_route_reason`` tokens are the ones that helper emits
  and every one of them is a canonical
  :data:`proxy.admission_reserve.LOCAL_FALLBACK_REASON_TOKENS` member;
* the documented starvation-detector defaults are what
  :func:`proxy.local_route_starvation.starved_warning_settings` resolves from an
  empty config, so a retune cannot leave the docs behind;
* the documented non-quarantining claim is true of the handler: a deficit-marked
  local 503 is deferred with no entry quarantine while an unmarked local 503 and
  a marked *remote* response keep the old behaviour;
* the documented abandoned-slot release is the shipped one: the only
  abandoned-slot TTL constant is the 900 s registry bound (no per-reason
  shorter TTL exists), and the annotation is cleared only from a fresh
  ``/slots`` snapshot.
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import proxy.provider as provider
import pytest
from fastapi import Response

from proxy import admission_reserve, local_route_starvation, observability, router_helpers

ROUTING_DOC = Path(__file__).resolve().parents[1] / "docs" / "routing.md"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "proxy" / "config.yaml"

STARVATION_HEADING = "## Local-route starvation detector"
COLD_START_HEADING = "## Local cold-start immunity"
DEFICIT_HEADING = "#### Measured cache-deficit early abort"
ABANDONMENT_HEADING = "#### Abandoned local dispatches"
RELEASE_HEADING = "#### Release, not a short TTL, for the deficit reason"

#: The documented route-availability fields, in table order.
ROUTE_FIELDS = (
    "local_route_available",
    "local_route_reason",
    "local_route_cooldown_remaining_s",
)

#: The one snapshot key the status payload does not publish (log-only evidence).
LOG_ONLY_ROUTE_FIELD = "local_route_failure_count"

DEFICIT_REASON = "measured_cache_deficit"


def _doc() -> str:
    return ROUTING_DOC.read_text(encoding="utf-8")


def _section(text: str, heading: str, next_prefix: str) -> str:
    """Heading to the next heading of the same or a higher level."""
    start = text.index(heading)
    rest = text[start + len(heading) :]
    return rest.split(f"\n{next_prefix}", 1)[0]


def _starvation_section() -> str:
    return _section(_doc(), STARVATION_HEADING, "## ")


def _cold_start_section() -> str:
    return _section(_doc(), COLD_START_HEADING, "## ")


def _deficit_section() -> str:
    return _section(_doc(), DEFICIT_HEADING, "#### ")


def _abandonment_section() -> str:
    return _section(_doc(), ABANDONMENT_HEADING, "#### ")


def _release_section() -> str:
    return _section(_doc(), RELEASE_HEADING, "#### ")


def _config_abort_block() -> str:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    start = text.index("# Measured cache-deficit early abort")
    end = text.index("  local_measured_cache_abort_min_deficit_ratio:", start)
    return text[start : text.index("\n", end)]


def _route_field_rows() -> dict[str, list[str]]:
    """The published-route-field table, keyed by field name."""
    rows: dict[str, list[str]] = {}
    for line in _starvation_section().splitlines():
        stripped = line.strip()
        if not stripped.startswith("| `local_route_"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        field = re.search(r"`([^`]+)`", cells[0]).group(1)
        rows[field] = cells
    return rows


def _yaml_knobs(section: str) -> dict[str, str]:
    """``key: value`` pairs from the fenced yaml block in *section*."""
    knobs: dict[str, str] = {}
    in_yaml = False
    for line in section.splitlines():
        if line.strip().startswith("```"):
            in_yaml = line.strip().lower().startswith("```yaml")
            continue
        if not in_yaml:
            continue
        match = re.match(r"([a-z][a-z0-9_]*):\s*([^#]+)", line.strip())
        if match:
            knobs[match.group(1)] = match.group(2).strip()
    return knobs


@pytest.fixture(autouse=True)
def _clean_state():
    provider._provider_unavailable_until.clear()
    local_route_starvation.reset_local_route_starvation_tracker()
    yield
    provider._provider_unavailable_until.clear()
    local_route_starvation.reset_local_route_starvation_tracker()


def _local_config(entry: dict | None = None) -> dict:
    local = {"name": "local-qwen3", "type": "local", "llama_model": "Qwen3"}
    if entry:
        local.update(entry)
    return {"models": {"hybrid": {"providers": [local]}}}


# ---------------------------------------------------------------------------
# AC10: the route-availability field table
# ---------------------------------------------------------------------------


def test_documented_route_fields_are_exactly_the_helper_output_keys():
    """The table names every published route field and nothing invented."""
    rows = _route_field_rows()
    published = set(observability.local_route_availability({}, model=None)) - {
        LOG_ONLY_ROUTE_FIELD
    }

    assert set(rows) == published == set(ROUTE_FIELDS), (
        "routing.md's route-field table must document exactly the published "
        f"fields; missing={sorted(published - set(rows))} "
        f"invented={sorted(set(rows) - published)}"
    )
    for field, cells in rows.items():
        assert len(cells) >= 4, f"{field} row must carry type/default/meaning cells"
        assert cells[3], f"{field} row must explain what the field means"


def test_documented_absent_safe_values_match_the_shipped_snapshot():
    """``true`` / ``null`` / ``null`` is what the helper really returns."""
    snapshot = observability.local_route_availability({}, model=None)

    assert snapshot["local_route_available"] is True
    assert snapshot["local_route_reason"] is None
    assert snapshot["local_route_cooldown_remaining_s"] is None

    rows = _route_field_rows()
    assert rows["local_route_available"][2] == "`true`"
    assert rows["local_route_reason"][2] == "`null`"
    assert rows["local_route_cooldown_remaining_s"][2] == "`null`"


def test_documented_route_reason_tokens_are_the_emitted_canonical_ones():
    """Both documented reason tokens are emitted by the helper and canonical."""
    provider._provider_unavailable_until["local-qwen3"] = time.time() + 45
    armed = observability.local_route_availability(
        _local_config(), model="local-qwen3"
    )
    provider._provider_unavailable_until.clear()

    now = datetime.now(UTC)
    window = (
        f"{(now + timedelta(minutes=30)).strftime('%H:%M')}-"
        f"{(now + timedelta(minutes=31)).strftime('%H:%M')}"
    )
    outside = observability.local_route_availability(
        _local_config({"available_times": [window]}), model="local-qwen3"
    )

    assert armed["local_route_available"] is False
    assert armed["local_route_reason"] == "local_provider_cooldown"
    assert armed["local_route_cooldown_remaining_s"] >= 1
    assert outside["local_route_available"] is False
    assert outside["local_route_reason"] == "outside_available_window"
    assert outside["local_route_cooldown_remaining_s"] is None, (
        "a window is not a cooldown, so no countdown may be fabricated"
    )

    documented = _route_field_rows()["local_route_reason"][3]
    for token in (armed["local_route_reason"], outside["local_route_reason"]):
        assert token in admission_reserve.LOCAL_FALLBACK_REASON_TOKENS, (
            f"{token} must stay a canonical fallback-reason token"
        )
        assert f"`{token}`" in documented, (
            f"the local_route_reason row must document {token}"
        )


# ---------------------------------------------------------------------------
# AC11: the starvation detector's documented meaning and throttle
# ---------------------------------------------------------------------------


def test_documented_starvation_defaults_match_the_resolver():
    """The documented knobs resolve to the documented defaults."""
    knobs = _yaml_knobs(_starvation_section())
    resolved = local_route_starvation.starved_warning_settings({})

    assert float(knobs["local_route_starvation_window_seconds"]) == pytest.approx(
        resolved["window_seconds"]
    )
    assert float(
        knobs["local_route_starvation_min_armed_fraction"]
    ) == pytest.approx(resolved["min_armed_fraction"])
    assert int(knobs["local_route_starvation_min_arms"]) == resolved["min_arms"]
    assert float(
        knobs["local_route_starvation_throttle_seconds"]
    ) == pytest.approx(resolved["throttle_seconds"])

    # The prose and the code agree on the shipped throttle (10 minutes).
    assert resolved["throttle_seconds"] == pytest.approx(
        local_route_starvation.DEFAULT_THROTTLE_SECONDS
    )
    assert resolved["throttle_seconds"] == 600.0
    assert resolved["min_arms"] == 2
    assert "10 minutes" in _starvation_section(), (
        "the section must state the throttle in operator units"
    )


def test_documented_starvation_reason_tokens_are_the_recorded_ones():
    """Every token in the reasons table is a reason the detector is fed with."""
    section = _starvation_section()
    table = section.split("The `reasons` tokens are the canonical routing", 1)[1]
    tokens = {
        match.group(1)
        for match in re.finditer(r"^\| `([a-z][a-z0-9_]*)` \|", table, re.MULTILINE)
    }

    recordable = {
        local_route_starvation.LOCAL_ROUTE_REASON_PROVIDER_COOLDOWN,
        local_route_starvation.LOCAL_ROUTE_REASON_MEASURED_CACHE_DEFICIT,
        local_route_starvation.LOCAL_ROUTE_REASON_PREFILL_THRASH,
    }
    assert tokens == recordable, (
        "the reasons table must equal the reasons the tracker is fed with; "
        f"missing={sorted(recordable - tokens)} "
        f"invented={sorted(tokens - recordable)}"
    )
    assert (
        local_route_starvation.LOCAL_ROUTE_REASON_MEASURED_CACHE_DEFICIT
        in admission_reserve.LOCAL_FALLBACK_REASON_TOKENS
    ), "the deficit arm reason must stay a canonical fallback-reason token"


# ---------------------------------------------------------------------------
# AC6: the detour is documented as non-quarantining, and it really is
# ---------------------------------------------------------------------------


def _marked_deficit_503() -> Response:
    response = Response(
        content=b'{"error": "measured_cache_deficit"}',
        status_code=503,
        media_type="application/json",
    )
    setattr(
        response,
        admission_reserve.LOCAL_DISPATCH_REASON_ATTR,
        DEFICIT_REASON,
    )
    return response


def test_documented_non_quarantining_claim_matches_the_handler():
    """A marked local 503 is deferred; unmarked and remote keep quarantining."""
    assert (
        provider._local_dispatch_deferral_reason(_marked_deficit_503(), "local")
        == provider.SKIP_REASON_MEASURED_CACHE_DEFICIT
    ), "the deficit detour must be deferred, not quarantined"

    unmarked = Response(content=b"{}", status_code=503)
    assert provider._local_dispatch_deferral_reason(unmarked, "local") is None, (
        "an unmarked local 503 must keep the pre-change entry quarantine"
    )
    assert (
        provider._local_dispatch_deferral_reason(_marked_deficit_503(), "remote")
        is None
    ), "the deferral is local-only; a remote 503 is never deferred by it"

    # A header-only marker must survive, so the deferral does not depend on the
    # in-process attribute.
    header_only = Response(
        content=b"{}",
        status_code=503,
        headers={admission_reserve.HEADER_LOCAL_DISPATCH_REASON: DEFICIT_REASON},
    )
    assert (
        provider._local_dispatch_deferral_reason(header_only, "local")
        == provider.SKIP_REASON_MEASURED_CACHE_DEFICIT
    )


def test_both_documents_state_the_detour_is_not_a_quarantine():
    """The claim lives beside the abort keys and beside the immunity note."""
    config = " ".join(_config_abort_block().split())
    deficit = " ".join(_deficit_section().split())
    cold_start = " ".join(_cold_start_section().split())

    for name, text in (("config.yaml", config), ("routing.md", deficit)):
        assert "NOT A QUARANTINE" in text or "does not quarantine" in text, (
            f"{name} must state the detour is non-quarantining"
        )
        assert "not on local" in text, (
            f"{name} must say the release is not a verdict on local's health"
        )
        assert "local_prefill_thrash_cooldown_seconds" in text, (
            f"{name} must scope the remaining refusal to the per-session window"
        )
        assert DEFICIT_REASON in text

    assert "LP-0MVWFKESK0005G61" in cold_start, (
        "the cold-start immunity section must carry the reason-scoped sibling note"
    )
    assert "local_dispatch_deferral" in cold_start
    assert "local_prefill_thrash_cooldown_seconds" in cold_start


# ---------------------------------------------------------------------------
# AC4: the abandoned-slot release / no-short-TTL semantics
# ---------------------------------------------------------------------------


def test_documented_release_semantics_match_the_shipped_constants():
    """900 s is only the registry bound; no per-reason TTL exists."""
    ttl_names = {
        name
        for name in dir(router_helpers)
        if "ABANDONED_SLOT_TTL" in name and name.isupper()
    }
    assert ttl_names == {"DEFAULT_ABANDONED_SLOT_TTL_SECONDS"}, (
        "a per-reason abandoned-slot TTL must be documented here if added; "
        f"found={sorted(ttl_names)}"
    )
    assert router_helpers.DEFAULT_ABANDONED_SLOT_TTL_SECONDS == 900.0

    release = " ".join(_release_section().split())
    assert "900 s" in release
    assert "not the release trigger" in release, (
        "the docs must state that the TTL is a bound, not the release trigger"
    )
    assert "no per-reason TTL exists" in release
    assert "fresh" in release and "`/slots`" in release, (
        "the docs must name the observed-free release evidence"
    )
    assert "LP-0MVUTYHG7001S6BM" in release, (
        "the docs must explain why a short TTL was rejected (the ghost gap)"
    )
    assert DEFICIT_REASON in release


def test_config_thrash_cooldown_note_keeps_its_scope():
    """The AC2 scope note still says the window is per-session/per-prompt."""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    key = "  local_prefill_thrash_cooldown_seconds:"
    start = text.rindex("  # Post-deadline prefill-thrash cooldown", 0, text.index(key))
    block = " ".join(text[start : text.index(key)].split())

    assert "per session AND per prompt fingerprint" in block, (
        "the thrash-cooldown note must keep the per-session/per-prompt scope"
    )
    assert "OTHER sessions" in block


def test_abandonment_table_carries_the_deficit_reason():
    """The abandonment table documents the deficit marker and the release rule."""
    abandonment = _abandonment_section()

    assert f"`{DEFICIT_REASON}`" in abandonment, (
        "the abandonment table must document the measured_cache_deficit marker"
    )
    doc = _doc()
    assert doc.index(RELEASE_HEADING) > doc.index(ABANDONMENT_HEADING), (
        "the release section must sit with the abandonment semantics"
    )
