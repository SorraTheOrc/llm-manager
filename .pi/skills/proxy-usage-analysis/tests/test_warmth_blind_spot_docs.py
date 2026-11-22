"""Documentation pin for the prompt-cache warmth blind spot in the usage report.

LP-0MWRQVFDU005WN76 (parent LP-0MWRQ6ZID000HZAZ). This is the **test-first**
child: the prose lands in LP-0MWRQVG4C002C5YV, so the assertions here are the
contract that child must satisfy.

What is pinned, and why it matters
----------------------------------
The 24 h report's ``prefill_eta_accuracy`` row and its M3b copy currently read
as a pure calibration score. They are not: the admission guard prices
``est_tokens * cached_ratio`` where ``cached_ratio`` is a per-``(model,
session)`` ratio, while llama-server's ``--cache-ram`` prompt cache is shared
across sessions. Warmth becomes knowable only *after* the fact, so part of the
over-prediction (the observed 863 % median) is **unobservable prompt-cache
warmth**, not miscalibration. An operator who reads it as calibration error
retunes the fit, which cannot recover an input the ETA never sees.

The three surfaces an operator or maintainer reads are pinned separately:

1. the ``## Limitations`` section of ``REFERENCE.md`` (the named limitation);
2. the ``prefill_eta_accuracy`` metric definition in ``REFERENCE.md`` (the
   caveat on the metric itself);
3. the **rendered** M3b copy in ``scripts/reporting.py`` — asserted on the
   report text the section builder emits, not on a source snapshot, so a
   sentence quoted only in a comment cannot satisfy it.

Red-first ordering (LP-0MWJOUV5P007R799 convention)
---------------------------------------------------
Each shipped-prose assertion was marked ``xfail(strict=True)`` with a reason
naming the documentation child, so the dev-push gate stayed green while the red
witness was recorded and each strict marker turned red (XPASS) once the prose
landed — the markers were removed by LP-0MWRQVG4C002C5YV, the documentation
child. Do not weaken an assertion to keep it green.

The checkers assert *facts* (which cache, priced how, knowable when), so the
prose may be rewritten freely; the canonical wordings below are offered so the
documentation child can satisfy the contract without guesswork.
"""

from __future__ import annotations

import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import reporting  # noqa: E402

REFERENCE_DOC = SKILL_ROOT / "REFERENCE.md"

#: The documentation child whose prose satisfies this contract.
DOCS_CHILD = "LP-0MWRQVG4C002C5YV"

#: A compliant ``## Limitations`` bullet (the named limitation).
CANONICAL_LIMITATION_BULLET = """\
- **The ETA prices per-session KV residency, not llama-server's cross-session
  prompt cache (LP-0MWRQ6ZID000HZAZ).** ``cached_ratio`` is a per-(model,
  session) ratio — the share of the prompt served from the slot's own KV —
  while llama-server's ``--cache-ram`` prompt cache is shared *across* sessions.
  A dispatch logged with ``cached_ratio=0.0`` can therefore still be **warm**
  and fast to first byte, and the cache is unpriced input: warmth is
  unobservable at admission and only knowable post hoc.
"""

#: A compliant caveat for the metric definition and the M3b copy.
CANONICAL_METRIC_CAVEAT = (
    "Part of the over-prediction is unobservable prompt-cache warmth, not "
    "calibration error: the ETA prices per-session KV residency while "
    "llama-server's cross-session prompt cache can still serve a dispatch "
    "logged as cold, so a large error is not the calibration to retune."
)

_MARKDOWN_NOISE = re.compile(r"[`*>]+")
_WHITESPACE = re.compile(r"\s+")

_CROSS_SESSION_RE = re.compile(r"cross[- ]session|across sessions")
_PER_SESSION_RE = re.compile(r"per[- ]\(?model,? session\)?|per[- ]session")
_UNPRICED_RE = re.compile(
    r"cannot see|can ?not see|can't see|invisible|unobservable|blind spot"
    r"|unpriced|not priced|never priced"
)
_OVER_PREDICT_RE = re.compile(r"over[- ]predict")
_PROMPT_CACHE_RE = re.compile(r"prompt[- ]cache")
_PARTIAL_RE = re.compile(r"\bpart\b|partly|portion|some of|much of|most of")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """Lower-case, de-emphasised, single-spaced text (wording-agnostic)."""
    return _WHITESPACE.sub(" ", _MARKDOWN_NOISE.sub("", text)).strip().lower()


# ---------------------------------------------------------------------------
# Extractors (guarded by their own test so a moved heading fails loudly)
# ---------------------------------------------------------------------------

def _heading_section(text: str, heading: str) -> str:
    """The body under *heading*, up to the next ``## `` heading (or EOF)."""
    start = text.index(heading)
    body = text[start + len(heading):]
    stop = re.search(r"\n## ", body)
    return body[: stop.start()] if stop else body


def _limitations_section() -> str:
    return _heading_section(_read(REFERENCE_DOC), "## Limitations")


def _limitation_bullets(section: str) -> list[str]:
    return [part for part in re.split(r"\n(?=- \*\*)", section) if part.strip()]


def _metric_definition() -> str:
    """The ``prefill_eta_accuracy`` definition bullet, up to the next item.

    The bullet is the list's last entry, so the boundary may be the next
    ``- **`` item or the next heading (``###``/``##``); without it the
    assertion would read the rest of the file and could be satisfied by prose
    that is not about this metric.
    """
    text = _read(REFERENCE_DOC)
    start = text.index("- **`prefill_eta_accuracy`**")
    body = text[start:]
    stop = re.search(r"\n(?:- \*\*|#{2,} )", body)
    return body[: stop.start()] if stop else body


class _StubSummary:
    """The minimum an ``AnalysisResult``-shaped object needs to render M1-M3b.

    M3b is static copy, so no matched pair is required: an empty window still
    emits the block. Rendering the real section builder (rather than reading a
    source snapshot) is what makes "the copy states the caveat" mean the copy an
    operator actually sees.
    """

    window_start = datetime(2026, 11, 22, tzinfo=UTC)
    window_end = window_start + timedelta(hours=24)
    dispatch_denied_count = 1
    denial_attributions: list = []
    dispatch_denied_events: list = []
    local_admission_events: list = []
    status_request_events: list = []
    ttft_events: list = []
    busy = None


def _render_m3b() -> str:
    lines: list[str] = []
    reporting._append_contention_prediction_section(
        lines.append, _StubSummary(), None
    )
    rendered = "\n".join(lines)
    assert "### M3b" in rendered, (
        "the report no longer renders an M3b section; the copy this pin asserts "
        "has moved or been removed"
    )
    return rendered.split("### M3b", 1)[1]


# ---------------------------------------------------------------------------
# The blind spot contract (REFERENCE Limitations)
# ---------------------------------------------------------------------------

def _missing_blind_spot_claims(text: str) -> list[str]:
    """Return the contract facts *text* does not state (empty list == compliant)."""
    flat = _flat(text)
    missing: list[str] = []
    if not _CROSS_SESSION_RE.search(flat):
        missing.append(
            "names llama-server's prompt cache as cross-session (say "
            "'cross-session', as opposed to the per-session KV the ETA prices)"
        )
    if "cache-ram" not in flat:
        missing.append(
            "names the shipped `--cache-ram` prompt-cache surface the ETA "
            "cannot see"
        )
    if not _PER_SESSION_RE.search(flat):
        missing.append(
            "states the priced quantity as per-session KV residency (say "
            "'per-session' / 'per-(model, session)')"
        )
    if not _UNPRICED_RE.search(flat):
        missing.append(
            "states the cache is invisible at admission (say 'cannot see', "
            "'unobservable' or 'unpriced')"
        )
    if not ("cached_ratio" in flat and "0.0" in flat and "warm" in flat):
        missing.append(
            "states that a `cached_ratio=0.0` dispatch can still be warm"
        )
    return missing


def _missing_limitation_claims(section: str) -> list[str]:
    """Facts missing from the *single* ``## Limitations`` bullet that should say it.

    Requiring one bullet keeps the statement a named limitation rather than
    five facts scattered across the section.
    """
    candidates = [
        bullet
        for bullet in _limitation_bullets(section)
        if "cached_ratio" in _flat(bullet) or "prompt cache" in _flat(bullet)
    ]
    if not candidates:
        return [
            "no `## Limitations` bullet names the prompt cache or "
            "`cached_ratio` at all"
        ]
    missing_by_bullet = {bullet: _missing_blind_spot_claims(bullet) for bullet in candidates}
    return min(missing_by_bullet.values(), key=len)


# ---------------------------------------------------------------------------
# The over-prediction caveat contract (metric definition + M3b copy)
# ---------------------------------------------------------------------------

def _missing_over_prediction_claims(text: str) -> list[str]:
    """Return the caveat facts *text* does not state (empty list == compliant)."""
    flat = _flat(text)
    missing: list[str] = []
    if not _OVER_PREDICT_RE.search(flat):
        missing.append("names the error as an over-prediction")
    if not _PARTIAL_RE.search(flat):
        missing.append(
            "says only *part* of the over-prediction is explained (say 'part of')"
        )
    if not _UNPRICED_RE.search(flat):
        missing.append(
            "states that warmth is unobservable (say 'unobservable' / "
            "'cannot see')"
        )
    if not _PROMPT_CACHE_RE.search(flat):
        missing.append("names the prompt cache as the missing input")
    if "warm" not in flat:
        missing.append("says the unobserved cache is *warm* (warmth)")
    if "calibration" not in flat:
        missing.append(
            "contrasts it with calibration error, so 863 % is not read as "
            "miscalibration to retune"
        )
    return missing


def _assert_states_caveat(name: str, text: str, missing: list[str]) -> None:
    assert not missing, (
        f"{name} does not state that part of the over-prediction is "
        f"unobservable prompt-cache warmth.\n"
        f"  file: {name}\n"
        f"  missing facts:\n"
        + "\n".join(f"    - {claim}" for claim in missing)
        + f"\n  the prose lands in {DOCS_CHILD}; a compliant example is:\n"
        f"    {CANONICAL_METRIC_CAVEAT}"
    )


# ---------------------------------------------------------------------------
# The shipped surfaces
# ---------------------------------------------------------------------------

def test_the_limitations_section_states_the_blind_spot():
    """AC2: a reader of ``## Limitations`` learns the ETA cannot see warmth."""
    section = _limitations_section()
    missing = _missing_limitation_claims(section)
    assert not missing, (
        "REFERENCE.md `## Limitations` does not carry the cross-session "
        "prompt-cache blind spot in a single bullet.\n"
        "  missing facts:\n"
        + "\n".join(f"    - {claim}" for claim in missing)
        + f"\n  the prose lands in {DOCS_CHILD}; a compliant bullet is:\n"
        + "\n".join(f"    {line}" for line in CANONICAL_LIMITATION_BULLET.splitlines())
    )


def test_the_prefill_eta_accuracy_definition_states_the_caveat():
    """AC4: the metric definition says part of the error is unobservable warmth."""
    name = "REFERENCE.md `prefill_eta_accuracy` definition"
    _assert_states_caveat(
        name, _metric_definition(), _missing_over_prediction_claims(_metric_definition())
    )


def test_the_m3b_copy_states_the_caveat():
    """AC3: the rendered M3b copy does not read as pure calibration error."""
    rendered = _render_m3b()
    _assert_states_caveat(
        "scripts/reporting.py (rendered M3b copy)",
        rendered,
        _missing_over_prediction_claims(rendered),
    )


# ---------------------------------------------------------------------------
# Extractor guards: a moved heading must fail here, not silently pass
# ---------------------------------------------------------------------------

def test_the_extractors_find_the_shipped_sections():
    limitations = _limitations_section()
    assert len(_limitation_bullets(limitations)) >= 3, (
        "REFERENCE.md `## Limitations` no longer parses into bullets"
    )

    definition = _metric_definition()
    assert "`prefill_eta_accuracy`" in definition
    assert "median" in definition.lower()

    m3b = _render_m3b()
    assert "prompt" in m3b.lower() or "Predicted" in m3b
    assert len(m3b.splitlines()) >= 8, "the M3b render is unexpectedly short"


# ---------------------------------------------------------------------------
# Checker mutation tests (AC5): every assertion fails on the reduced framing
# ---------------------------------------------------------------------------

def test_the_limit_at_checker_accepts_the_canonical_wording():
    """The contract is satisfiable: the offered bullet states every fact."""
    assert _missing_blind_spot_claims(CANONICAL_LIMITATION_BULLET) == []
    assert (
        _missing_limitation_claims(
            "## Limitations\n\n" + CANONICAL_LIMITATION_BULLET + "\n"
        )
        == []
    )


def test_the_blind_spot_checker_rejects_a_bare_calibration_error_framing():
    """AC5: reducing the limitation to "calibration is off" fails."""
    bare = (
        "- **Prefill ETA calibration (LP-0MVGR2OXT0096OGK M3).** The published\n"
        "  ETA is compared against the served prefill durations; the remaining\n"
        "  gap is a calibration error to tune with the conservatism factor.\n"
    )
    section = "## Limitations\n\n" + bare + "\n"

    missing = _missing_limitation_claims(section)

    assert missing
    assert any("prompt cache or `cached_ratio`" in claim for claim in missing)


def test_the_blind_spot_checker_rejects_a_deleted_statement():
    """AC5: deleting the cache sentences leaves the checker red."""
    truncated = CANONICAL_LIMITATION_BULLET.replace(
        "  A dispatch logged with ``cached_ratio=0.0`` can therefore still be **warm**\n"
        "  and fast to first byte, and the cache is unpriced input: warmth is\n"
        "  unobservable at admission and only knowable post hoc.\n",
        "",
    )

    assert truncated != CANONICAL_LIMITATION_BULLET
    assert any(
        "cached_ratio=0.0" in claim
        for claim in _missing_blind_spot_claims(truncated)
    )


def test_the_caveat_checker_accepts_the_canonical_wording():
    """The caveat contract is satisfiable as written."""
    assert _missing_over_prediction_claims(CANONICAL_METRIC_CAVEAT) == []


def test_the_caveat_checker_rejects_a_deleted_caveat():
    """AC5: the metric definition without the caveat is still a red assertion."""
    bare_definition = (
        "- **`prefill_eta_accuracy`** — median **absolute relative error** "
        "between the admission guard's priced ETA and the observed first byte. "
        "The registry direction is lower-is-better, so a smaller error is a "
        "better prediction."
    )

    missing = _missing_over_prediction_claims(bare_definition)

    assert missing
    assert any("over-prediction" in claim for claim in missing)
    assert any("prompt cache" in claim for claim in missing)


def test_the_caveat_checker_rejects_a_deleted_m3b_sentence():
    """AC5: a mutation of the rendered M3b copy is detected, not tolerated.

    Before the sentence lands this is the shipped render (already incomplete);
    after it lands the mutation strips the warmth claim and must still fail.
    """
    rendered = _render_m3b()
    mutated = re.sub(r"(?i)\bunobservable\b", "", rendered)
    mutated = re.sub(r"(?i)\bwarm\b", "", mutated)
    mutated = re.sub(r"(?i)prompt[- ]cache", "", mutated)

    missing = _missing_over_prediction_claims(mutated)

    assert missing
    assert any("unobservable" in claim for claim in missing)
    assert any("prompt cache" in claim for claim in missing)
