#!/usr/bin/env python3
"""Artefact-contract tests for the prefill-recovery verification docs.

LP-0MVSAMS5O006OAIS (AC1/AC4) fixes the *measurement gate* of the
post-deadline re-dispatch fix (LP-0MVSAMN14001DMTY) in documentation:
``docs/dev/prefill-recovery-verification.md`` names the JSON fields an
operator must read (`ac1.wedge_denials`, `ac1.max_unproductive_hold_seconds`,
`ac2.max_release_seconds`, `ac2.unreleased_sessions`,
`ac3.local_finishes_after_last_release`, `coverage.verdict_met`) and links the
committed measurement artefacts.

These tests pin that contract against the artefacts `--measure` actually
produced, so a renamed field, a renamed artefact or a broken relative link
fails here instead of leaving an operator with an unreadable gate. They also
pin the honesty rule the work item insists on: while the fix is **pushed but
not deployed**, no artefact may be presented as the post-fix verification, so
the document must carry the explicit PENDING status until a post-deploy 24 h
artefact exists — and must stop saying PENDING once one does.

No production log, no network: the committed artefacts are read as data.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs" / "dev"
DOC_PATH = DOCS_DIR / "prefill-recovery-verification.md"

#: Artefacts of this fix committed alongside the doc (baseline + partial).
ARTEFACT_GLOB = "prefill-recovery-verification-post-deadline-*.json"

#: The post-deploy artefact the AC1 gate waits for (follow-up work item).
POST_DEPLOY_GLOB = "prefill-recovery-verification-post-deadline-24h-*.json"

#: Field paths the doc's gate table tells an operator to read, with the
#: Python type the measurement emits for each.
GATE_FIELDS = {
    "coverage.verdict_met": bool,
    "ac1.wedge_denials": int,
    "ac1.verdict_met": bool,
    "ac2.max_release_seconds": (int, float),
    "ac2.unreleased_sessions": list,
    "ac3.local_finishes_after_last_release": int,
    "all_verdicts_met": bool,
}

#: Nullable numeric field: ``None`` when the window contains no denials.
NULLABLE_NUMERIC_FIELDS = ("ac1.max_unproductive_hold_seconds",)


def _resolve(document: dict, dotted: str):
    node = document
    for part in dotted.split("."):
        assert isinstance(node, dict), f"{dotted}: {part} is not in a mapping"
        assert part in node, f"{dotted}: missing key {part!r}"
        node = node[part]
    return node


def _artefacts() -> list[Path]:
    paths = sorted(DOCS_DIR.glob(ARTEFACT_GLOB))
    assert paths, f"no measurement artefacts matched {ARTEFACT_GLOB}"
    return paths


def test_committed_artefacts_carry_the_documented_gate_fields():
    """Every committed artefact parses and exposes the documented gate fields."""
    for path in _artefacts():
        document = json.loads(path.read_text())

        window = document["window"]
        assert datetime.strptime(window["since"], "%Y-%m-%d %H:%M:%S")
        assert datetime.strptime(window["until"], "%Y-%m-%d %H:%M:%S")
        assert window["since"] < window["until"]

        for dotted, expected_type in GATE_FIELDS.items():
            value = _resolve(document, dotted)
            assert isinstance(value, expected_type), (
                f"{path.name}: {dotted} is {type(value).__name__}, not {expected_type}"
            )

        for dotted in NULLABLE_NUMERIC_FIELDS:
            value = _resolve(document, dotted)
            assert value is None or isinstance(value, (int, float)), f"{path.name}: {dotted} is {type(value).__name__}"


def test_doc_links_resolve_to_committed_artefacts():
    """Every relative ``.json`` link in the doc points at a file on disk."""
    text = DOC_PATH.read_text()
    links = re.findall(r"\]\(([^)#\s]+\.json)\)", text)
    assert links, "the doc no longer links any measurement artefact"
    for link in links:
        target = (DOC_PATH.parent / link).resolve()
        assert target.is_file(), f"broken artefact link: {link}"


def test_pending_gate_status_tracks_the_post_deploy_artefact():
    """The doc says PENDING exactly while the post-deploy window is missing.

    AC1 stays open until a >=24 h **post-deploy** measurement exists, so the
    PENDING marker must not be dropped early (that would present a pre-deploy
    window as fix verification — the misattribution this slice exists to
    prevent) and must not survive the artefact that closes the gate.
    """
    text = DOC_PATH.read_text()
    pending = "status: PENDING" in text
    post_deploy = sorted(DOCS_DIR.glob(POST_DEPLOY_GLOB))

    if post_deploy:
        assert not pending, (
            "a post-deploy artefact "
            f"({post_deploy[0].name}) exists — replace the PENDING status "
            "with the measured verdict"
        )
    else:
        assert pending, (
            "no post-deploy measurement artefact exists yet, so the doc must "
            "record the AC1 gate as status: PENDING (never silently waived)"
        )
