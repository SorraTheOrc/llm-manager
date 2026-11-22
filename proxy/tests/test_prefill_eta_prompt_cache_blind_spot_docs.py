"""Documentation pin for the prefill ETA's cross-session prompt-cache blind spot.

LP-0MWRQVFDU005WN76 (parent LP-0MWRQ6ZID000HZAZ). This is the **test-first**
child: the prose lands in LP-0MWRQVG4C002C5YV, so the assertions here are the
contract that child must satisfy.

Why this is a safety pin and not a formatting check
---------------------------------------------------
``prefill_eta`` prices ``est_tokens * cached_ratio`` as "coded" (served from the
prompt cache) where ``cached_ratio`` is a per-``(model, session)`` ratio, while
llama-server's ``--cache-ram`` prompt cache is shared **across** sessions
(``models.ini`` ``[Qwen3] cache-ram = 40960``; ``docs/dev/prompt-cache-tuning.md``).
Warmth is only knowable *after* the fact (``update_cached_ratio`` writes turn N's
usage, ``record_measured_divergence`` corrects turn N+1), so a dispatch logged
with ``cached_ratio=0.0`` can still be warm and first-byte fast.

Without that stated, a maintainer reading the ETA contract reasonably concludes
the ETA prices *the* prompt cache — and then "fixes" the over-prediction by
retuning the calibration, which cannot help, because the missing term is
unobservable at admission. The blind spot therefore has to be a **named
limitation** in the module documentation, not an inference from the code.

Red-first ordering (LP-0MWJOUV5P007R799 convention)
---------------------------------------------------
The assertion that fails against today's documentation is marked
``xfail(strict=True)`` with a reason naming the documentation child. The
dev-push gate therefore stays green while the red witness is recorded; once the
prose lands the strict marker turns red (XPASS) and forces its removal. Do not
weaken the assertion to make it green — land the prose instead.

The checker accepts any wording that states the facts; the canonical paragraph
below is offered so the documentation child can satisfy the contract without
guesswork.
"""

from __future__ import annotations

import configparser
import re
from pathlib import Path

import pytest

from proxy import prefill_eta

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_INI = REPO_ROOT / "models.ini"

#: The documentation child whose prose satisfies this contract.
DOCS_CHILD = "LP-0MWRQVG4C002C5YV"

#: A compliant statement of the limitation. The documentation child may reword
#: it — the checker asserts the *facts*, not this snapshot — but every fact it
#: carries is required.
CANONICAL_BLIND_SPOT_PARAGRAPH = """\
What the ETA cannot see
-----------------------

The ETA prices per-session KV residency, not llama-server's cross-session
prompt cache. ``cached_ratio`` is a per-(model, session) ratio (the share of the
prompt served from the slot's own KV), while llama-server's ``--cache-ram``
prompt cache is shared *across* sessions, so a dispatch logged with
``cached_ratio=0.0`` can still be **warm** and fast to first byte. The
``--cache-ram`` prompt cache is unpriced input: its warmth is unobservable at
admission (there is no synchronous ``/slots`` probe on the hot path), so the
ETA cannot cost it.
"""

#: The reduction the contract exists to prevent: the blind spot rewritten as a
#: bare "calibration error" an operator would try to tune away.
BARE_CALIBRATION_FRAMING = """\
The ETA is fitted from observed local dispatches. The error between the priced
ETA and the observed first byte is a calibration error, so the conservatism
factor and the percentile knobs are the levers for tuning it down.
"""

#: Underscores are *not* punctuation here: ``cached_ratio`` is an identifier.
_MARKDOWN_NOISE = re.compile(r"[`*>]+")
_WHITESPACE = re.compile(r"\s+")

_CROSS_SESSION_RE = re.compile(r"cross[- ]session|across sessions")
_PER_SESSION_RE = re.compile(r"per[- ]\(?model,? session\)?|per[- ]session")
_UNPRICED_RE = re.compile(
    r"cannot see|can ?not see|can't see|invisible|unobservable|blind spot"
    r"|unpriced|not priced|never priced"
)


def _flat(text: str) -> str:
    """Lower-case, de-emphasised, single-spaced text (wording-agnostic)."""
    return _WHITESPACE.sub(" ", _MARKDOWN_NOISE.sub("", text)).strip().lower()


def _missing_blind_spot_claims(text: str) -> list[str]:
    """Return the contract facts *text* does not state (empty list == compliant).

    Facts, not phrases: each names what a reader must learn, so the prose may be
    rewritten freely as long as it still says all of them.
    """
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


def _assert_names_blind_spot(name: str, text: str) -> None:
    """Fail with the file and every missing fact (self-correcting message)."""
    missing = _missing_blind_spot_claims(text)
    assert not missing, (
        f"{name} does not state the cross-session prompt-cache blind spot.\n"
        f"  file: {name}\n"
        f"  missing facts:\n"
        + "\n".join(f"    - {claim}" for claim in missing)
        + f"\n  the prose lands in {DOCS_CHILD}; a compliant example is:\n"
        + "\n".join(f"    {line}" for line in CANONICAL_BLIND_SPOT_PARAGRAPH.splitlines())
    )


def _shipped_cache_ram_sections() -> dict[str, str]:
    """Every ``models.ini`` section that sets ``cache-ram``, value as written."""
    parser = configparser.ConfigParser()
    parser.read(MODELS_INI, encoding="utf-8")
    return {
        section: parser[section]["cache-ram"]
        for section in parser.sections()
        if parser.has_option(section, "cache-ram")
    }


# ---------------------------------------------------------------------------
# Premise: the cache the prose names really is configured and cross-session
# ---------------------------------------------------------------------------

def test_the_shipped_model_configures_a_cross_session_prompt_cache():
    """The named limitation has a real object: a non-zero ``--cache-ram`` cap.

    If llama-server's RAM prompt cache were disabled the blind spot would not
    exist, so documentation asserting it would be wrong. This keeps the premise
    tied to the shipped ``models.ini`` rather than to prose.
    """
    sections = _shipped_cache_ram_sections()

    assert sections, "models.ini configures no `cache-ram` prompt cache"
    for section, value in sections.items():
        assert float(value) > 0, (
            f"models.ini section [{section}] sets cache-ram = {value}; the "
            "cross-session prompt cache the documentation names is disabled"
        )


# ---------------------------------------------------------------------------
# The shipped module documentation
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "the limitation is prose and lands in "
        f"{DOCS_CHILD}; this is the red-first witness (LP-0MWRQVFDU005WN76)"
    ),
)
def test_the_module_documentation_names_the_cross_session_prompt_cache():
    """AC1/AC2: the ETA contract states the input it cannot price.

    Red today, strict-XPASS once the prose lands (the marker then fails and is
    removed by the documentation child).
    """
    assert prefill_eta.__doc__, "proxy.proxy.prefill_eta has no module docstring"
    _assert_names_blind_spot(
        "proxy/proxy/prefill_eta.py (module docstring)", prefill_eta.__doc__
    )


# ---------------------------------------------------------------------------
# The checker itself: mutation tests (AC5)
# ---------------------------------------------------------------------------

def test_the_checker_accepts_the_canonical_wording():
    """The contract is satisfiable: the offered paragraph states every fact."""
    assert _missing_blind_spot_claims(CANONICAL_BLIND_SPOT_PARAGRAPH) == []


def test_the_checker_rejects_a_bare_calibration_error_framing():
    """AC5: rewriting the blind spot as "a calibration error" must fail.

    This is the reduction the pin exists for — an operator who reads only the
    calibration framing retunes the fit, which cannot recover an input the ETA
    never sees.
    """
    missing = _missing_blind_spot_claims(BARE_CALIBRATION_FRAMING)

    assert "calibration error" in BARE_CALIBRATION_FRAMING.lower()
    assert missing, "a bare calibration-error framing must not satisfy the contract"
    assert any("cross-session" in claim for claim in missing)
    assert any("cache-ram" in claim for claim in missing)


def test_the_checker_rejects_a_deleted_limitation():
    """AC5: deleting the statement leaves the checker red, not silently green."""
    without_limitation = (
        "Prefill-ETA calibration and the local admission guard.\n\n"
        "The guard costs the projected prefill from observed local dispatches\n"
        "and compares it with the caller's wait budget.\n"
    )

    missing = _missing_blind_spot_claims(without_limitation)

    assert missing
    assert any("cached_ratio=0.0" in claim for claim in missing)


def test_the_checker_rejects_a_partially_reduced_statement():
    """Naming the cache but pricing it correctly (no ``cached_ratio=0.0``) fails.

    A docstring may say "the ETA prices the cross-session prompt cache" and be
    wrong in the opposite direction; a truncated edit must not pass either.
    """
    reduced = CANONICAL_BLIND_SPOT_PARAGRAPH.replace(
        "so a dispatch logged with\n``cached_ratio=0.0`` can still be **warm** and fast to first byte. ",
        "",
    )

    assert reduced != CANONICAL_BLIND_SPOT_PARAGRAPH
    assert any("cached_ratio=0.0" in claim for claim in _missing_blind_spot_claims(reduced))
