"""Docs drift pins for the finished-but-not-yet-released occupancy rule.

Parent LP-0MWMSQR3O002YNPW (AC6); implementation item LP-0MWNHBGUF007BRWR.

The prose is the deliverable here, so the assertions tie the shipped
documentation to the predicate that enforces it rather than freezing a copy:

* the "Routing occupancy counts other sessions, not your own lease" section
  states the finished-but-not-yet-released window (terminal chunk → teardown),
  the streaming-token marker keyed on the dispatch credit token, and that a
  record holding at least one streaming dispatch token is genuine in-flight
  work;
* the retired ``active``-keyed wording ("only an idle lease is excluded / an
  active record always counts") appears nowhere in ``proxy/docs/`` or
  ``proxy/proxy/``;
* the ``remote-ok`` never-queues statement stays accurate — this change removes
  a false refusal, it does not add waiting.
"""

from __future__ import annotations

from pathlib import Path

ROUTING_DOC = Path(__file__).resolve().parents[1] / "docs" / "routing.md"
PROXY_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROXY_ROOT / "docs"
CODE_DIR = PROXY_ROOT / "proxy"

SECTION_HEADING = "## Routing occupancy counts other sessions, not your own lease"
NEXT_HEADING = "## Dispatch-lease occupancy and the denial line"

#: Wording that stated the retired, ``active``-keyed rule. It is false of the
#: shipped predicate — a finished-but-not-yet-released record is ``active`` and
#: is *not* counted — so it must not survive in any doc or source comment.
STALE_OLD_RULE_WORDING = (
    "only while it is an idle lease",
    "A record that is still `active` is genuine in-flight work",
    "an active record always counts",
    "only an idle lease is excluded",
    "Only an **inactive** record is excluded",
)


def _doc() -> str:
    return ROUTING_DOC.read_text(encoding="utf-8")


def _section() -> str:
    """The occupancy section, heading to the next ``## `` heading."""
    text = _doc()
    start = text.index(SECTION_HEADING)
    rest = text[start + len(SECTION_HEADING) :]
    return rest.split(f"\n{NEXT_HEADING}", 1)[0]


def test_occupancy_section_states_the_finished_window() -> None:
    """The terminal-chunk → teardown window is documented explicitly."""
    section = " ".join(_section().split())

    assert "finished-but-not-yet-released" in section, (
        "the occupancy section must name the finished-but-not-yet-released "
        "shape"
    )
    assert "terminal chunk" in section, (
        "the window starts at the terminal chunk and must say so"
    )
    assert "teardown" in section, "the window ends at request teardown"


def test_occupancy_section_names_the_streaming_marker() -> None:
    """How the window is handled: the credit-keyed streaming-token marker."""
    section = " ".join(_section().split())

    assert "streaming_dispatch_tokens" in section, (
        "the marker field must be named so the next reader can find it"
    )
    assert "dispatch credit token" in section, (
        "the marker is keyed on the dispatch credit token, not the wall clock"
    )
    assert "_mark_local_dispatch_stream_finished" in section, (
        "the terminal-chunk removal path must be named"
    )
    assert "_claim_local_dispatch_credit" in section, (
        "the teardown/prune path must be named"
    )


def test_genuine_in_flight_work_is_a_streaming_token() -> None:
    """What still counts: a record holding at least one streaming token."""
    section = " ".join(_section().split())

    assert "at least one" in section, (
        "the docs must say a record with >=1 streaming token still counts"
    )
    assert "streaming dispatch token" in section
    assert "genuine in-flight work" in section, (
        "the docs must keep the genuine-in-flight-work distinction explicit"
    )


def test_remote_ok_still_never_queues() -> None:
    """Removing a false refusal must not be documented as adding waiting."""
    text = " ".join(_doc().split())

    assert "`remote-ok` (the default preference) still never queues" in text


def test_retired_old_rule_wording_is_gone() -> None:
    """No stale statement of the ``active``-keyed rule survives."""
    offenders: list[str] = []
    for path in sorted(DOCS_DIR.rglob("*.md")) + sorted(CODE_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for phrase in STALE_OLD_RULE_WORDING:
            if phrase in text:
                offenders.append(f"{path}: {phrase!r}")

    assert not offenders, (
        "the retired active-keyed occupancy rule must not survive in "
        f"proxy/docs or proxy/proxy: {offenders}"
    )
