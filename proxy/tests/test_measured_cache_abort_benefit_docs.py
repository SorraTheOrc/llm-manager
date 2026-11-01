"""Documentation-drift tests for the measured-cache benefit gate.

LP-0MVWDZ2DV007M57U (parent LP-0MVWBD91Z004U135 AC9, plus the "no new vocabulary
elsewhere" half of AC5).

Why this is a safety test, not a formatting check
-------------------------------------------------
The benefit gate is the difference between "a mid-size cold prompt is served
locally" and "a pin-local client is refused for minutes": an unjustified abort
returns a retryable 503 that a ``local-preferred`` request may not detour, so
the contract turns it into a retryable ``429``, and the abort also arms
``local_prefill_thrash_cooldown_seconds``. The gate's behaviour is therefore
governed by two numbers an operator reads out of the documentation — the
**comparator** (``budget_safety_fraction x local_prefill_deadline_seconds``,
114 s shipped) and the fact that the admission **wait** budget is *not* that
comparator — and by one new piece of vocabulary (``within_eta_budget``).

Documentation that drifts from the shipped config is how a future maintainer
re-tunes the abort back into refusing a client that could have been served, so
these rows pin the shipped prose **and** re-derive every claim from the code
that enforces it:

* the documented comparator is the number
  :func:`proxy.prefill_eta.prefill_deadline_budget_seconds` resolves from the
  shipped ``proxy/config.yaml`` (114.0 s), and it really is derived from
  ``local_prefill_deadline_seconds`` (shortening the deadline moves it);
* the documented "the admission wait budget is **not** the comparator" claim is
  enforced: changing ``contention_queue_max_wait_seconds`` leaves the resolved
  budget untouched;
* the documented continue token is
  :data:`proxy.measured_cache_abort.DETAIL_WITHIN_ETA_BUDGET`, and the recorded
  incident row really continues with it under the shipped config;
* the documented feature rollback (``local_measured_cache_abort_enabled: 0``)
  really disables the gate for the population it exists for;
* the ``detail`` vocabulary enumerated in ``routing.md`` is exactly the set the
  decision module can emit — no invented tokens, and exactly one reason token;
* every config key named in the documentation exists under ``server:``.

The assertions describe the *shipped* documentation, so they stay valid after
any deliberate retuning: they compare the prose against the config and the
readers, never against a frozen literal (the 114 s figure is asserted as a
*derived* value, with the shipped inputs named).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from proxy.measured_cache_abort import (
    DETAIL_ABORT,
    DETAIL_BELOW_RATIO_FLOOR,
    DETAIL_DISABLED,
    DETAIL_ERROR,
    DETAIL_INVALID_MEASUREMENT,
    DETAIL_INVALID_PRICED,
    DETAIL_NO_MEASUREMENT,
    DETAIL_WITHIN_BOUND,
    DETAIL_WITHIN_ETA_BUDGET,
    MEASURED_CACHE_DEFICIT_REASON,
    AbortSettings,
    decide_measured_cache_abort,
)
from proxy.prefill_eta import prefill_deadline_budget_seconds
from proxy.prompt_progress import PromptProgress

from proxy import measured_cache_abort, prefill_eta

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "proxy" / "config.yaml"
ROUTING_DOC = REPO_ROOT / "proxy" / "docs" / "routing.md"

#: The model the incident pane was dispatched to (``routing_check
#: provider=local-qwen3``), i.e. the calibration key the gate prices with.
MODELLED_MODEL = "local-qwen3"

#: The recorded incident measurement: 14 380 tokens of real cold prefill,
#: admitted on a priced 48 new tokens.
INCIDENT_TOTAL = 14_380
INCIDENT_PRICED = 48

#: The population the abort exists for: a huge cold prompt priced as warm. Its
#: measured ETA (218.4 s at the shipped fit) exceeds the comparator, so the
#: documented abort verdict stays reachable.
HUGE_TOTAL = 56_000
HUGE_PRICED = 100

#: Shipped comparator inputs, named in the documentation.
DEADLINE_KEY = "local_prefill_deadline_seconds"
SAFETY_FRACTION_KEY = "local_admission_guard_budget_safety_fraction"

#: The admission wait budget the documentation explicitly rules out.
WAIT_BUDGET_KEY = "contention_queue_max_wait_seconds"

#: Config keys the documented rollbacks and knobs are named by.
ENABLED_KEY = "local_measured_cache_abort_enabled"
RATIO_KEY = "local_measured_cache_abort_min_deficit_ratio"
BOUND_KEY = "local_measured_cache_abort_bound_tokens"
COOLDOWN_KEY = "local_prefill_thrash_cooldown_seconds"

#: Every ``detail`` the decision module can emit (module constants, not prose).
DECISION_DETAILS = {
    DETAIL_ABORT,
    DETAIL_DISABLED,
    DETAIL_NO_MEASUREMENT,
    DETAIL_INVALID_MEASUREMENT,
    DETAIL_INVALID_PRICED,
    DETAIL_BELOW_RATIO_FLOOR,
    DETAIL_WITHIN_BOUND,
    DETAIL_WITHIN_ETA_BUDGET,
    DETAIL_ERROR,
}

#: Shapes that unambiguously name a *config key* (same convention as
#: ``test_resident_prefix_config_docs.py``): an explicit ``server.<knob>``
#: reference, a backticked ``<knob>: <value>``, or an indented ``<knob>:`` line
#: inside a fenced YAML block. Log events (``local_admission_guard``) and
#: state names (``local_eviction_after_acceptance``) therefore never count.
_SERVER_REF_RE = re.compile(r"server\.([a-z][a-z0-9_]*)")
_BACKTICK_KEY_RE = re.compile(r"`([a-z][a-z0-9_]*):\s[^`]*`")
_INDENTED_KEY_RE = re.compile(r"^\s{2,}([a-z][a-z0-9_]*):", re.MULTILINE)
_DETAIL_EQ_RE = re.compile(r"detail=([a-z][a-z0-9_]*)")
_DETAIL_TABLE_RE = re.compile(r"^\| `([a-z][a-z0-9_]*)` \|", re.MULTILINE)
_REASON_EQ_RE = re.compile(r"reason=([a-z][a-z0-9_]*)")

#: Backticked ``name: value`` shapes in this prose that are *request/stream
#: fields*, not ``server:`` config knobs — the one upstream key the proxy adds
#: to a copy of the client body. Excluded from the config-key cross-check.
NON_CONFIG_FIELDS = frozenset({"return_progress"})


@pytest.fixture(autouse=True)
def _clean_calibration():
    """Price from the uncalibrated shipped fit (no leaked dispatch samples)."""
    prefill_eta.reset()
    yield
    prefill_eta.reset()


def _shipped_server() -> dict:
    with open(CONFIG_PATH) as handle:
        return (yaml.safe_load(handle) or {})["server"]


def _shipped_config(**overrides) -> dict:
    server = dict(_shipped_server())
    server.update(overrides)
    return {"server": server}


def _progress(total: int, cache: int) -> PromptProgress:
    return PromptProgress(total=total, cache=cache, processed=0, time_ms=0.0)


def _settings(config: dict | None = None) -> AbortSettings:
    return AbortSettings.from_config(
        _shipped_config() if config is None else config, model=MODELLED_MODEL
    )


# ---------------------------------------------------------------------------
# Documentation blocks under test
# ---------------------------------------------------------------------------

def _config_abort_block() -> str:
    """The measured-cache abort comment block from ``proxy/config.yaml``.

    Bounded by its banner and by the last key of the group, so the block always
    carries both the prose and the three shipped keys it documents.
    """
    text = CONFIG_PATH.read_text(encoding="utf-8")
    marker = "# Measured cache-deficit early abort"
    assert marker in text, "config.yaml must document the measured-cache abort"
    start = text.index(marker)
    end = text.index(f"  {RATIO_KEY}:", start)
    end = text.index("\n", end)
    return text[start:end]


def _routing_section() -> str:
    """The measured-cache early-abort section from ``proxy/docs/routing.md``."""
    text = ROUTING_DOC.read_text(encoding="utf-8")
    marker = "#### Measured cache-deficit early abort"
    assert marker in text, "routing.md must document the measured-cache abort"
    return text.split(marker, 1)[1].split("\n#### ", 1)[0]


def test_both_documents_are_reachable_and_non_empty():
    """Guard the extractors themselves (a moved heading must fail loudly)."""
    assert len(_config_abort_block().splitlines()) > 40
    assert len(_routing_section().splitlines()) > 40


# ---------------------------------------------------------------------------
# The comparator: documented numbers == the enforced ones
# ---------------------------------------------------------------------------

def test_documented_comparator_is_the_shipped_derived_budget():
    """AC2/AC3: 114 s is ``fraction x deadline`` from the shipped config."""
    server = _shipped_server()
    budget = prefill_deadline_budget_seconds({"server": server})
    expected = float(server[SAFETY_FRACTION_KEY]) * float(server[DEADLINE_KEY])

    assert budget == pytest.approx(expected) == pytest.approx(114.0)

    rendered = f"{budget:g} s"
    for name, text in (
        ("config.yaml", _config_abort_block()),
        ("routing.md", _routing_section()),
    ):
        assert rendered in text, f"{name} must state the shipped comparator ({rendered})"
        for key in (SAFETY_FRACTION_KEY, DEADLINE_KEY):
            assert key in text, f"{name} must name the comparator input {key}"


def test_documented_comparator_inputs_are_the_ones_the_gate_reads():
    """The prose names the real inputs: the deadline moves the budget."""
    assert prefill_deadline_budget_seconds(
        _shipped_config(**{DEADLINE_KEY: 60})
    ) == pytest.approx(57.0)
    # ... and the bare-deadline fallback the docs describe (fraction 0).
    assert prefill_deadline_budget_seconds(
        _shipped_config(**{SAFETY_FRACTION_KEY: 0})
    ) == pytest.approx(float(_shipped_server()[DEADLINE_KEY]))


def test_documented_wait_budget_is_not_the_comparator():
    """AC2: the admission wait budget really is not an input."""
    baseline = prefill_deadline_budget_seconds({"server": _shipped_server()})

    assert prefill_deadline_budget_seconds(
        _shipped_config(**{WAIT_BUDGET_KEY: 300})
    ) == pytest.approx(baseline)

    for name, text in (
        ("config.yaml", _config_abort_block()),
        ("routing.md", _routing_section()),
    ):
        assert WAIT_BUDGET_KEY in text, (
            f"{name} must name the wait-budget knob it rules out"
        )
        assert "not the comparator" in text.replace("*", "").lower(), (
            f"{name} must state that the wait budget is not the comparator"
        )


def test_documented_abort_verdict_stays_reachable_for_the_huge_prompt():
    """AC2: the gate narrows the abort, it does not disable it."""
    decision = decide_measured_cache_abort(
        _progress(HUGE_TOTAL, 0), HUGE_PRICED, _settings()
    )

    assert decision.abort is True
    assert decision.reason == MEASURED_CACHE_DEFICIT_REASON
    assert decision.detail == DETAIL_ABORT
    assert decision.budget_s == pytest.approx(114.0)
    assert decision.measured_eta_s > decision.budget_s


# ---------------------------------------------------------------------------
# The pinned-client guarantee and the continue token
# ---------------------------------------------------------------------------

def test_documented_continue_token_is_the_implemented_one():
    """AC3/AC4: ``within_eta_budget`` is real for the incident row."""
    decision = decide_measured_cache_abort(
        _progress(INCIDENT_TOTAL, 0), INCIDENT_PRICED, _settings()
    )

    assert DETAIL_WITHIN_ETA_BUDGET == "within_eta_budget"
    assert decision.abort is False
    assert decision.reason is None
    assert decision.detail == DETAIL_WITHIN_ETA_BUDGET
    assert decision.measured_eta_s < decision.budget_s

    assert f"`{DETAIL_WITHIN_ETA_BUDGET}`" in _routing_section(), (
        "routing.md must name the continue token the decision emits"
    )


def test_documented_pinned_client_guarantee_is_stated():
    """AC1/AC3: a pinned dispatch inside budget is never aborted."""
    routing = _routing_section()
    config = _config_abort_block()

    for name, text in (("routing.md", routing), ("config.yaml", config)):
        assert "local-preferred" in text, f"{name} must name the pinned class"
        assert "never aborted" in text or "never refuses a pinned" in text, (
            f"{name} must state that a pinned dispatch inside budget is never aborted"
        )
    # The consequence the guarantee exists for: the 503/429 + cooldown shape.
    assert "429" in config
    assert COOLDOWN_KEY in config


def test_documented_feature_rollback_disables_the_gate():
    """AC2: ``local_measured_cache_abort_enabled: 0`` is the documented rollback."""
    disabled = _settings(_shipped_config(**{ENABLED_KEY: 0}))
    decision = decide_measured_cache_abort(
        _progress(HUGE_TOTAL, 0), HUGE_PRICED, disabled
    )

    assert decision.abort is False
    assert decision.detail == DETAIL_DISABLED
    assert f"`{ENABLED_KEY}: 0`" in _config_abort_block(), (
        "config.yaml must name the feature rollback exactly as the reader reads it"
    )


# ---------------------------------------------------------------------------
# Vocabulary: exactly one new ``detail`` token, no new reason/header/body
# ---------------------------------------------------------------------------

def test_documented_detail_vocabulary_is_the_implemented_one():
    """AC4: no invented ``detail`` tokens, and none of them undocumented."""
    section = _routing_section()

    documented = set(_DETAIL_TABLE_RE.findall(section)) | set(
        _DETAIL_EQ_RE.findall(section)
    )
    assert documented == DECISION_DETAILS, (
        "routing.md's detail vocabulary must equal the decision module's; "
        f"missing={sorted(DECISION_DETAILS - documented)} "
        f"invented={sorted(documented - DECISION_DETAILS)}"
    )


def test_documented_reason_token_is_unchanged():
    """AC4: the abort keeps its single pre-existing reason token."""
    reasons = set(_REASON_EQ_RE.findall(_routing_section()))

    assert reasons == {MEASURED_CACHE_DEFICIT_REASON}


def test_documentation_introduces_no_response_header():
    """AC4: the refusal surface belongs to LP-0MVSS435600370D6, not here."""
    section = _routing_section()

    assert "X-" not in section, "the gate must not document a new response header"
    assert "refusal body" in section, (
        "routing.md must say the refusal body is unchanged/owned elsewhere"
    )


# ---------------------------------------------------------------------------
# Consistency: every config key named in the docs exists under ``server:``
# ---------------------------------------------------------------------------

def _documented_knobs(text: str) -> set[str]:
    knobs = set(_SERVER_REF_RE.findall(text))
    knobs |= set(_BACKTICK_KEY_RE.findall(text))
    in_yaml = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_yaml = line.strip().lower().startswith("```yaml")
            continue
        if in_yaml:
            match = _INDENTED_KEY_RE.match(line)
            if match:
                knobs.add(match.group(1))
    return knobs


def test_every_config_key_named_in_the_docs_exists_under_server():
    server = _shipped_server()
    documented: set[str] = set()
    for text in (_config_abort_block(), _routing_section()):
        documented |= _documented_knobs(text)
    documented -= NON_CONFIG_FIELDS

    assert documented, "the extractors found no config keys (prose moved/broken?)"
    missing = sorted(k for k in documented if k not in server)
    assert missing == [], f"documented but absent from config.yaml: {missing}"
    assert ENABLED_KEY in documented


def test_documented_abort_knobs_and_rollbacks_are_named():
    """The gate's inputs and both rollbacks are named, and really exist."""
    server = _shipped_server()
    text = "\n".join((_config_abort_block(), _routing_section()))

    for key in (
        ENABLED_KEY,
        RATIO_KEY,
        BOUND_KEY,
        DEADLINE_KEY,
        SAFETY_FRACTION_KEY,
        COOLDOWN_KEY,
        WAIT_BUDGET_KEY,
    ):
        assert key in text, f"{key} must be documented"
        assert key in server, f"{key} must exist under server: in config.yaml"
