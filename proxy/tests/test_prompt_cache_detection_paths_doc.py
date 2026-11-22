"""Drift guard for the warm-prefix detection-path assessment.

LP-0MWRQVM5L006RC05 rests on one document section: *Detection paths assessed*
in ``docs/dev/prompt-cache-tuning.md``. The item's acceptance criteria require
that section to keep

1. **all three** candidate paths — (a) llama-server cache metrics on ``/metrics``
   or ``/slots``, (b) an asynchronous warm-prefix sampler, (c) charging the
   observed outcome back into the calibration window — each with the measurement
   it needs, the cost it adds to the admission path and a verdict;
2. **one** recommended path, with the follow-up work items it was filed as, and
   the conditional prerequisites named; and
3. the evidence section that the verdicts rest on pointing at the assessment,
   so a reader arriving at the measurement can find the decision.

Why pin prose at all: the assessment is the evidence that closes the third
acceptance criterion of the parent item (LP-0MWRQ6ZID000HZAZ AC3), and the
follow-up ids are the only durable record of what was decided. A later edit that
silently drops a path — or the verdict that closed it — would leave the parent
claiming an assessment it no longer carries. The assertions therefore check the
*structure and the facts*, never the exact wording: rewording is free, deleting a
path, a verdict or a follow-up is not.

Nothing here contacts llama-server, the proxy or the network: the surface facts
live in the document, and the shipped-geometry check reads ``models.ini``.
"""

from __future__ import annotations

import configparser
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = REPO_ROOT / "docs/dev/prompt-cache-tuning.md"
MODELS_INI = REPO_ROOT / "models.ini"

#: The assessment heading that anchors the section under test.
ASSESSMENT_HEADING = "## Detection paths assessed (LP-0MWRQVM5L006RC05)"

#: The evidence heading the assessment must sit *after* (it consumes it).
EVIDENCE_HEADING = "## What the proxy cannot see"

#: The canonical verdict vocabulary (the item's AC wording).
VERDICTS = ("not viable", "viable with conditions", "viable")

#: Candidate path letters and a needle that identifies each path's subject.
PATHS = {
    "a": ("(a)", ("/metrics", "/slots")),
    "b": ("(b)", ("sampler", "asynchronous")),
    "c": ("(c)", ("calibration window",)),
}

#: A work-item id, as filed in this project.
WI_ID_RE = re.compile(r"\bLP-[0-9A-Z]{12,}\b")

#: The recommended path's follow-up and its blocking prerequisite.
RECOMMENDED_FOLLOW_UP = "LP-0MWRWBPFV00339XP"
BLOCKING_PREREQUISITE = "LP-0MWRWBJPP001Q5JX"

#: The conditional prerequisite already on the books (coded-coefficient fix).
CONDITIONAL_PREREQUISITE = "LP-0MWL0VHOY0065EI8"


def _doc() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def _section(heading: str) -> str:
    """Return the prose from *heading* to the next level-2 heading."""
    text = _doc()
    start = text.find(heading)
    assert start >= 0, f"{DOC_PATH.name} has no section {heading!r}"
    rest = text[start + len(heading):]
    end = rest.find("\n## ")
    return heading + (rest if end < 0 else rest[:end])


def _subsection(section: str, marker: str) -> str:
    """Return the level-3 block starting with *marker* (e.g. ``### (b)``)."""
    start = section.find(f"### {marker}")
    assert start >= 0, f"subsection {marker!r} missing from the assessment"
    rest = section[start + len(marker) + 4:]
    end = rest.find("\n### ")
    return section[start: start + len(marker) + 4] + (rest if end < 0 else rest[:end])


def _shipped_cache_ram() -> str | None:
    """The ``cache-ram`` value the shipped ``models.ini`` sets for ``Qwen3``."""
    parser = configparser.ConfigParser()
    assert parser.read(MODELS_INI), f"{MODELS_INI} is unreadable"
    for section in parser.sections():
        if section.lower() == "qwen3" and parser.has_option(section, "cache-ram"):
            return parser[section]["cache-ram"]
    return None


def test_the_assessment_sits_after_the_evidence_it_rests_on():
    """The verdicts are placed beside the measurement they consume."""
    text = _doc()
    evidence_at = text.find(EVIDENCE_HEADING)
    assessment_at = text.find(ASSESSMENT_HEADING)
    assert evidence_at >= 0, "the prompt-cache evidence section was removed"
    assert assessment_at >= 0, "the detection-path assessment section was removed"
    assert assessment_at > evidence_at, (
        "the assessment must follow the evidence section it relies on, so the "
        "figures it quotes are read before the verdict they support"
    )


def test_the_evidence_section_points_forward_to_the_assessment():
    """A reader of the measurement can reach the decision from it."""
    evidence = _section(EVIDENCE_HEADING)
    assert "Detection paths assessed" in evidence, (
        "the prompt-cache evidence section no longer links the detection-path "
        "assessment, so the measurement and the decision have drifted apart"
    )
    assert "LP-0MWRQVM5L006RC05" in evidence, (
        "the evidence section must name the assessment item it hands off to"
    )


def test_every_candidate_path_keeps_its_measurement_cost_and_verdict():
    """All three paths stay assessed, each with the three required facts."""
    section = _section(ASSESSMENT_HEADING)
    for letter, (marker, subject_needles) in PATHS.items():
        block = _subsection(section, marker)
        lowered = block.lower()
        for needle in subject_needles:
            assert needle in lowered, (
                f"path ({letter}) no longer names {needle!r}; the assessment "
                "must keep its subject identifiable"
            )
        assert "need" in lowered, (
            f"path ({letter}) must record the measurement it needs"
        )
        assert "cost" in lowered, (
            f"path ({letter}) must record the cost/latency it adds to admission"
        )
        assert any(verdict in lowered for verdict in VERDICTS), (
            f"path ({letter}) must carry an explicit viability verdict "
            f"(one of {VERDICTS})"
        )


def test_one_path_is_recommended_and_two_paths_are_closed_with_evidence():
    """The verdicts still match the shipped cache and the recorded findings."""
    section = _section(ASSESSMENT_HEADING)
    path_a = _subsection(section, "(a)")
    path_b = _subsection(section, "(b)")
    path_c = _subsection(section, "(c)")

    # (a) is closed by the backend surface, not by effort: the document must
    # quote the unknown-flag refusal and the absence of a cache-hit counter.
    assert "501" in path_a and "--metrics" in path_a
    assert "no cache-hit counter" in path_a.lower()

    # (b) is the measured-evidence path: it names the capture it reads and the
    # tokenizer-free fingerprint machinery it reuses.
    assert "measured_prefill" in path_b
    assert "message_fingerprints" in path_b

    # (c) is anchored to the two call sites whose distance makes it free.
    assert "record_dispatch_sample" in path_c
    assert "get_measured_prefill" in path_c

    assert "recommend" in section.lower(), "the assessment must recommend a path"


def test_the_recommended_path_and_its_prerequisites_are_linked():
    """The follow-ups the parent item promises are still named in the document."""
    section = _section(ASSESSMENT_HEADING)
    recommendation = _subsection(section, "Recommendation")
    ids = set(WI_ID_RE.findall(section))
    for expected in (
        RECOMMENDED_FOLLOW_UP,
        BLOCKING_PREREQUISITE,
        CONDITIONAL_PREREQUISITE,
    ):
        assert expected in ids, (
            f"the assessment no longer links {expected}; the follow-up work "
            "filed for the recommended path (and its prerequisites) must stay "
            "discoverable from the evidence document"
        )
    assert RECOMMENDED_FOLLOW_UP in recommendation
    assert "path (b)" in recommendation.lower()


def test_the_assessment_rests_on_the_shipped_cross_session_cache():
    """The object under assessment is real: a non-zero ``cache-ram`` cap."""
    value = _shipped_cache_ram()
    assert value is not None, "models.ini no longer configures [Qwen3] cache-ram"
    assert int(value) > 0, (
        f"[Qwen3] cache-ram is {value}; with no cross-session prompt cache "
        "there is no warmth to detect"
    )
    assert "cache-ram" in _section(ASSESSMENT_HEADING).lower()
