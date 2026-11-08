"""Documentation-drift tests for the residency-aware admission knobs.

LP-0MVN1IQTF0037WTG AC8. The three new knobs decide whether a large prompt may
be served locally, so a doc/config drift here is not cosmetic: a knob documented
in one place and shipped under another name (or shipped without a documented
``0``-disables semantics) is how a "safe rollback" turns into an outage.

Covered:

- ``proxy/config.yaml`` declares all three knobs with the values the readers
  resolve, and the readers agree with the shipped file (no accidental
  defaulting);
- the ``0``-disables semantics are real *for the shipped reader*, not just
  claimed in a comment: the ladder admits on the disabled path;
- the tenancy default really is the slot count;
- the three design documents describe the five-rung ladder in order and state
  the "read the cached snapshot, never a synchronous ``/slots`` call"
  constraint with its ``LP-0MUXTW4MD003TVIC`` rationale;
- every knob named in those documents exists in ``proxy/config.yaml`` under
  ``server:`` and every new knob is named in at least one of them (no
  documented-but-absent or shipped-but-undocumented knob).

These assertions describe the *shipped* documentation, not a temporary
experiment, so they stay valid after any retuning of the defaults.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from proxy.provider import (
    MEASUREMENT_VOCABULARY,
    RESIDENCY_SOURCE_MEASURED,
    RESIDENCY_SOURCE_MEASURED_LAST_DISPATCH,
    RESIDENCY_SOURCE_NON_RESIDENT,
    RESIDENCY_SOURCE_SESSION_SLOT,
    RESIDENCY_SOURCE_UNKNOWN,
    SKIP_REASON_UNMEASURED_COLD,
    THRESHOLD_APPLIED_VOCABULARY,
    _effective_resident_prefix_min_ratio,
    _unmeasured_residency_cold_enabled,
)
from proxy.router_helpers import (
    DEFAULT_TENANCY_IDLE_SECONDS,
    resolve_session_tenants,
    resolve_tenancy_idle_seconds,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "proxy" / "config.yaml"
ROUTING_DOC = REPO_ROOT / "proxy" / "docs" / "routing.md"
CACHED_RATIO_DOC = REPO_ROOT / "docs" / "dev" / "cached-ratio-prefix-credit.md"
SLOT_DESIGN_DOC = REPO_ROOT / "docs" / "dev" / "slot-management-design.md"

NEW_KNOBS = (
    "local_resident_prefix_min_ratio",
    "local_session_tenants",
    "local_session_tenancy_idle_seconds",
)

#: Single rollback switch for the unmeasured-cold enforcement
#: (LP-0MW81DJZ60049X5W), documented beside the residency knobs above.
UNMEASURED_COLD_KEY = "local_unmeasured_residency_cold_enabled"

DOC_PATHS = (ROUTING_DOC, CACHED_RATIO_DOC, SLOT_DESIGN_DOC)


def _server_config() -> dict:
    with open(CONFIG_PATH) as handle:
        config = yaml.safe_load(handle) or {}
    return config["server"]


# ---------------------------------------------------------------------------
# Shipped config vs. the readers that consume it
# ---------------------------------------------------------------------------

def test_config_declares_every_new_knob():
    server = _server_config()

    assert isinstance(server["local_resident_prefix_min_ratio"], (int, float))
    assert isinstance(server["local_session_tenants"], int)
    assert isinstance(server["local_session_tenancy_idle_seconds"], (int, float))


def test_ratio_reader_agrees_with_the_shipped_value():
    server = _server_config()

    assert _effective_resident_prefix_min_ratio({"server": server}) == pytest.approx(
        float(server["local_resident_prefix_min_ratio"])
    )


def test_ratio_zero_disables_the_gate_in_the_reader():
    """The documented ``0`` rollback is the reader's behaviour, not a comment."""
    assert _effective_resident_prefix_min_ratio(
        {"server": {"local_resident_prefix_min_ratio": 0}}
    ) == 0.0


def test_unmeasured_cold_rollback_is_documented_and_shipped_enabled():
    """AC1/AC3 (LP-0MW81DJZ60049X5W): the rollback ships documented and ON.

    The unmeasured-cold enforcement's single rollback switch must be visible in
    the shipped config (an operator reads ``proxy/config.yaml``, not the
    source), its documented default must be the intended post-rollout
    behaviour, and the reader must agree with the shipped value.
    """
    server = _server_config()
    text = CONFIG_PATH.read_text()

    assert server[UNMEASURED_COLD_KEY] is True
    assert _unmeasured_residency_cold_enabled({"server": server}) is True
    # The documented rollback is the reader's behaviour, not just a comment.
    assert (
        _unmeasured_residency_cold_enabled(
            {"server": {UNMEASURED_COLD_KEY: False}}
        )
        is False
    )
    # The rollback statement sits in the knob's own comment block, so an
    # operator looking at the key sees how to undo it (AC1).
    index = text.index(f"{UNMEASURED_COLD_KEY}:")
    block = text[max(0, index - 4000):index].lower()
    assert "rollback" in block
    assert "predicted" in block


def test_tenancy_default_is_the_shipped_slot_count():
    server = _server_config()

    assert resolve_session_tenants({"server": server}) == int(
        server["session_slot_pool_size"]
    )
    assert resolve_session_tenants({"server": {"local_session_tenants": 0}}) == 0


def test_tenancy_idle_reader_agrees_with_the_shipped_value():
    server = _server_config()

    assert resolve_tenancy_idle_seconds({"server": server}) == pytest.approx(
        float(server["local_session_tenancy_idle_seconds"])
    )


def test_tenancy_idle_default_is_the_dispatch_lease():
    """Documented derivation: the adaptive dispatch lease, bounded by the watchdog."""
    assert (
        resolve_tenancy_idle_seconds({"server": {}})
        == DEFAULT_TENANCY_IDLE_SECONDS
    )
    assert DEFAULT_TENANCY_IDLE_SECONDS <= 180.0


# ---------------------------------------------------------------------------
# Design docs describe the shipped ladder
# ---------------------------------------------------------------------------

def _routing_section() -> str:
    text = ROUTING_DOC.read_text(encoding="utf-8")
    marker = "## Residency-aware admission ladder"
    assert marker in text, "routing.md must document the admission ladder"
    return text.split(marker, 1)[1].split("\n## ", 1)[0]


def test_routing_doc_describes_the_five_rungs_in_order():
    section = _routing_section()

    positions = [
        section.index("Hard context cap"),
        section.index("Resident-prefix gate"),
        section.index("Session tenancy"),
        section.index("Live-priced cold check"),
        section.index("Idle-exempt override"),
    ]
    assert positions == sorted(positions), "rungs must be documented in ladder order"


def test_routing_doc_states_the_cached_snapshot_constraint():
    section = _routing_section()

    assert "never issue a synchronous `/slots` call" in section.lower()
    assert "LP-0MUXTW4MD003TVIC" in section
    assert "_last_slot_details_cache" in section
    assert "Fail-open" in section or "fail-open" in section


def test_routing_doc_names_every_new_knob_and_the_skip_reasons():
    section = _routing_section()

    for knob in NEW_KNOBS:
        assert knob in section, f"{knob} missing from the ladder docs"
    for reason in (
        "resident_prefix_insufficient",
        "tenant_limit",
        "large_context_bypass",
        "context_too_large",
    ):
        assert reason in section, f"skip reason {reason} undocumented"


def test_slot_design_doc_records_the_shipped_approach():
    text = SLOT_DESIGN_DOC.read_text(encoding="utf-8")

    assert "## 6. Shipped" in text
    for token in ("proxy/residency.py", "local_session_tenants", "local_eviction"):
        assert token in text
    assert "LP-0MUXTW4MD003TVIC" in text


def test_prefix_credit_doc_scopes_the_prediction():
    """The prediction's remaining role (fail-open + sub-threshold) is documented."""
    text = CACHED_RATIO_DOC.read_text(encoding="utf-8")

    assert "Where the prediction still matters" in text
    assert "residency_source=unknown" in text
    assert "local_resident_prefix_min_ratio" in text


# ---------------------------------------------------------------------------
# Consistency: docs <-> shipped config
# ---------------------------------------------------------------------------

def _documented_knobs(text: str) -> set[str]:
    """Extract *config knob* references from prose and YAML fences.

    Only two shapes count: an explicit ``server.<knob>`` reference, and a
    ``<knob>:`` key written inside a fenced YAML block. That keeps log-event
    names (``local_admission_guard``) and internal state (``local_active``) out
    of the check — they are documented behaviour, not config knobs.
    """
    knobs = set(re.findall(r"server\.([a-z0-9_]+)", text))
    in_yaml = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_yaml = line.strip().lower().startswith("```yaml")
            continue
        if in_yaml:
            match = re.match(r"^\s{2,}([a-z][a-z0-9_]*):", line)
            if match:
                knobs.add(match.group(1))
    return knobs


def _config_known_knobs() -> set[str]:
    """Every ``local_*`` knob name the shipped config refers to.

    Includes knobs documented as commented-out defaults (``# local_…: …``) —
    those are shipped documentation even when the key is left at its code
    default.
    """
    return set(re.findall(r"\b(local_[a-z0-9_]+)\b", CONFIG_PATH.read_text()))


def test_no_documented_but_absent_local_knob():
    known = _config_known_knobs()
    documented: set[str] = set()
    for path in DOC_PATHS:
        documented |= _documented_knobs(path.read_text(encoding="utf-8"))
    documented = {k for k in documented if k.startswith("local_")}

    missing = sorted(k for k in documented if k not in known)
    assert missing == [], f"documented but absent from config.yaml: {missing}"


def test_every_new_knob_is_documented_somewhere():
    text = "\n".join(path.read_text(encoding="utf-8") for path in DOC_PATHS)

    undocumented = [knob for knob in NEW_KNOBS if knob not in text]
    assert undocumented == [], f"shipped but undocumented: {undocumented}"


def test_config_comments_explain_defaults_and_zero_semantics():
    """Each knob's block names its default and its disable/fallback semantics."""
    text = CONFIG_PATH.read_text(encoding="utf-8")

    for knob in NEW_KNOBS:
        index = text.index(f"{knob}:")
        block = text[max(0, index - 2500):index]
        assert "`0`" in block or " 0 " in block or "0 disables" in block, (
            f"{knob} must document its 0-disables/fallback semantics"
        )
        assert "default" in block.lower(), f"{knob} must document its default"


# ---------------------------------------------------------------------------
# Staged rollout documentation (LP-0MVN4LC7N005XH7P)
# ---------------------------------------------------------------------------

ROLLOUT_DOC = REPO_ROOT / "docs" / "dev" / "resident-prefix-rollout.md"


def test_rollout_doc_documents_every_knob_and_both_rollback_switches():
    """The operator's rollout/rollback runbook stays in step with the knobs."""
    text = ROLLOUT_DOC.read_text(encoding="utf-8")

    for knob in NEW_KNOBS:
        assert knob in text, f"{knob} missing from the rollout doc"
    # Both documented 0-disables rollbacks.
    assert "local_resident_prefix_min_ratio: 0" in text
    assert "local_session_tenants: 0" in text
    # The observation contract: absence of the report section is a finding.
    assert "## Residency-aware admission" in text
    # Out-of-scope knobs the rollout must not disturb.
    for knob in (
        "local_prefill_deadline_seconds",
        "local_max_concurrent_prefills",
        "session_slot_pool_size",
    ):
        assert knob in text, f"{knob} must be named as unchanged"


def test_rollout_doc_does_not_promise_out_of_scope_changes():
    """No slot-count increase ships here (evaluated separately)."""
    text = ROLLOUT_DOC.read_text(encoding="utf-8")

    assert "LP-0MVN3KPEC007WWL5" in text
    assert "deliberately does not change" in text


# ---------------------------------------------------------------------------
# Measurement provenance documentation (LP-0MVT2QW9P0054LX2)
# ---------------------------------------------------------------------------

RESIDENCY_SOURCES = (
    RESIDENCY_SOURCE_MEASURED,
    RESIDENCY_SOURCE_MEASURED_LAST_DISPATCH,
    RESIDENCY_SOURCE_SESSION_SLOT,
    RESIDENCY_SOURCE_NON_RESIDENT,
    RESIDENCY_SOURCE_UNKNOWN,
)


def _observability_grammar_line() -> str:
    """The ``routing_resident_prefix`` grammar line from ``routing.md``."""
    for line in ROUTING_DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("`routing_resident_prefix "):
            return line
    raise AssertionError("routing.md must show the routing_resident_prefix grammar")


def test_routing_grammar_line_matches_the_shipped_vocabularies():
    """The logged grammar cannot drift from the constants that produce it."""
    line = _observability_grammar_line()

    assert " measurement=" in line
    # Additive field: after the source it qualifies, before the thresholds.
    assert (
        line.index("residency_source=")
        < line.index("measurement=")
        < line.index("min_ratio=")
    )
    for value in MEASUREMENT_VOCABULARY:
        assert value in line, f"measurement={value} missing from the grammar"
    for source in RESIDENCY_SOURCES:
        assert source in line, f"residency_source={source} missing from the grammar"


def test_measurement_vocabulary_is_documented_where_operators_look():
    """Every shipped unmeasured reason is documented, not just the grammar."""
    for path in (ROUTING_DOC, ROLLOUT_DOC):
        text = path.read_text(encoding="utf-8")
        missing = [value for value in MEASUREMENT_VOCABULARY if value not in text]
        assert missing == [], f"{path.name} does not document {missing}"


def test_threshold_applied_fields_are_documented_as_an_additive_suffix():
    """``threshold_applied``/``threshold_value`` (LP-0MW81DHPH00650NY).

    The fields explain *which* threshold decided a >38k admission, so the
    operator-facing grammar must show them — and must show them **after**
    ``session=``, because appending is what keeps the existing positional
    consumers of the line working (parent AC3).
    """
    line = _observability_grammar_line()

    assert "threshold_applied=" in line
    assert "threshold_value=" in line
    assert (
        line.index(" session=")
        < line.index("threshold_applied=")
        < line.index("threshold_value=")
    )
    for value in THRESHOLD_APPLIED_VOCABULARY:
        assert value in line, f"threshold_applied={value} missing from the grammar"


def test_rollout_doc_explains_what_to_read_and_the_residual_gap():
    """The runbook says what to grep and which limitation is expected."""
    text = ROLLOUT_DOC.read_text(encoding="utf-8")

    assert "## What to read" in text
    assert "measurement=" in text
    # The residual limitation: buffered dispatches emit no prompt_progress.
    assert "no `prompt_progress`" in text
    assert "unavailable:no_prior_measurement" in text
    # The capture rollback relabels resident decisions, intentionally.
    assert "local_measured_cache_abort_enabled: 0" in text
    assert "unavailable:capture_disabled" in text
    assert "not a regression" in text


def test_slot_design_doc_records_the_measured_last_dispatch_rung():
    """The design doc places the new rung and states its invalidation rule."""
    text = SLOT_DESIGN_DOC.read_text(encoding="utf-8")
    section = text.split("## 6. Shipped", 1)[1]

    assert "measured_prefill" in section
    assert RESIDENCY_SOURCE_MEASURED_LAST_DISPATCH in section
    # Between the live-snapshot rung and the ownership-only rung.
    assert (
        section.index(RESIDENCY_SOURCE_MEASURED)
        < section.index(RESIDENCY_SOURCE_MEASURED_LAST_DISPATCH)
        < section.index(RESIDENCY_SOURCE_SESSION_SLOT)
    )
    for event in ("eviction", "slot release/reassignment", "model switch"):
        assert event in section, f"invalidation trigger {event!r} undocumented"
    # Accuracy: the value is an upper bound, never a promise.
    assert "upper bound" in section
    assert "decode" in section


# ---------------------------------------------------------------------------
# Unmeasured is cold: the rule, its thresholds and its rollback
# (LP-0MW81DUBK005QRFQ; rule LP-0MW81DFML006O5KL, rollback LP-0MW81DJZ60049X5W)
# ---------------------------------------------------------------------------

ROUTER_DOC = REPO_ROOT / "docs" / "llama-router.md"
README_DOC = REPO_ROOT / "proxy" / "README.md"

#: The skip reason the enforcement emits (``provider.SKIP_REASON_UNMEASURED_COLD``).
UNMEASURED_COLD_DENIAL = SKIP_REASON_UNMEASURED_COLD

ALL_RULE_DOCS = (ROUTING_DOC, ROUTER_DOC, README_DOC, ROLLOUT_DOC, CACHED_RATIO_DOC)

#: Phrases that described the retired fail-open path. They must not come back:
#: re-introducing one would tell an operator that an unmeasured prompt is still
#: admitted on the predicted ratio, which is exactly the 2026-11-07 failure.
RETIRED_FAIL_OPEN_CLAIMS = (
    "degrades to the predicted ratio rather than denying local on absent data",
    "degrade to the predicted-ratio decision",
    "→ today's predicted-ratio decision",
)


def _threshold_forms(value: int) -> tuple[str, ...]:
    """The ways a token count is written in the docs (plain, comma, spaced)."""
    grouped = f"{value:,}"
    return (str(value), grouped, grouped.replace(",", " "))


def _documents_threshold(text: str, value: int) -> bool:
    return any(form in text for form in _threshold_forms(value))


def _shipped_thresholds() -> tuple[int, int]:
    """The cold/warm token thresholds as the shipped config declares them."""
    server = _server_config()
    return (
        int(server["local_large_context_cold_cache_threshold"]),
        int(server["local_large_context_warm_cache_threshold"]),
    )


def test_config_documents_the_unmeasured_cold_rule_its_thresholds_and_rollback():
    """AC1: the operator-facing config states the rule beside the flag."""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    index = text.index(f"{UNMEASURED_COLD_KEY}:")
    block = text[max(0, index - 4000):index]

    assert UNMEASURED_COLD_DENIAL in block, (
        "the flag's comment block must name the skip reason it produces"
    )
    assert "residency_source=unknown" in block, (
        "the comment must state which residency basis counts as unmeasured"
    )
    assert "local_large_context_cold_cache_threshold" in block, (
        "the comment must tie the rule to the cold threshold it applies"
    )
    assert "ROLLBACK" in block and "byte-for-byte" in block, (
        "the comment must state the rollback and that it is byte-for-byte"
    )


def test_routing_doc_documents_the_unmeasured_cold_rule_and_rollback():
    section = _routing_section()

    assert UNMEASURED_COLD_DENIAL in section
    assert UNMEASURED_COLD_KEY in section
    assert "residency_source=unknown" in section
    # The rule is stated against both shipped thresholds, so an operator can
    # see which number governs a measured vs unmeasured prompt.
    for value in _shipped_thresholds():
        assert _documents_threshold(section, value), (
            f"the ladder docs must state the {value}-token threshold"
        )
    # Not idle-overridable, and the final local override is named.
    assert "IDLE_OVERRIDABLE_SKIP_REASONS" in section
    assert "local_last_resort" in section


def test_routing_doc_exit_table_carries_the_unmeasured_cold_denial():
    """The client-visible exit token is documented, and as a refusal."""
    text = ROUTING_DOC.read_text(encoding="utf-8")
    rows: list[list[str]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        token = re.search(r"`([^`]+)`", cells[1])
        if token and token.group(1) == UNMEASURED_COLD_DENIAL:
            rows.append(cells)

    assert len(rows) == 1, (
        "routing.md's exit→token table must carry exactly one row for "
        f"{UNMEASURED_COLD_DENIAL}; got {len(rows)}"
    )
    cells = rows[0]
    assert cells[2].strip("*` ") == "refusing", (
        "the unmeasured-cold denial is never idle-overridable, so its exit row "
        f"must be a refusal; got {cells[2]!r}"
    )
    assert f"reason={UNMEASURED_COLD_DENIAL}" in cells[3], (
        "the exit row must show the routing_skip_local line it produces"
    )


def test_router_doc_documents_the_unmeasured_cold_rule_and_rollback():
    """docs/llama-router.md is the operator's cold/warm entry point."""
    text = ROUTER_DOC.read_text(encoding="utf-8")

    assert UNMEASURED_COLD_DENIAL in text
    assert UNMEASURED_COLD_KEY in text
    assert "residency_source=unknown" in text
    for value in _shipped_thresholds():
        assert _documents_threshold(text, value), (
            f"llama-router.md must state the {value}-token threshold"
        )
    # The rollback is stated as a value the operator can set, not just named.
    assert f"{UNMEASURED_COLD_KEY}:" in text.replace("`", ""), (
        "the rollback must be shown as a settable key, not only mentioned"
    )
    assert "admit_predicted" in text and "skip_predicted" in text, (
        "the rollback must state the decision values it restores"
    )


def test_readme_documents_the_unmeasured_cold_rule_and_rollback():
    """README is where the cold/warm thresholds are explained to operators."""
    text = README_DOC.read_text(encoding="utf-8")

    assert UNMEASURED_COLD_DENIAL in text
    assert UNMEASURED_COLD_KEY in text
    assert "residency_source=unknown" in text
    for value in _shipped_thresholds():
        assert _documents_threshold(text, value), (
            f"proxy/README.md must state the {value}-token threshold"
        )


def test_rollout_doc_documents_the_unmeasured_cold_rule_and_rollback():
    """The rollout runbook is what an operator reads while the rule is armed."""
    text = ROLLOUT_DOC.read_text(encoding="utf-8")

    assert UNMEASURED_COLD_DENIAL in text
    assert UNMEASURED_COLD_KEY in text
    # It must not still claim the gate fail-opens on an unmeasured basis.
    assert "the gate fails open" not in text


def test_no_doc_reintroduces_a_retired_fail_open_claim():
    """The fail-open wording the enforcement retired cannot drift back in."""
    stale: list[str] = []
    for path in ALL_RULE_DOCS:
        text = path.read_text(encoding="utf-8")
        for claim in RETIRED_FAIL_OPEN_CLAIMS:
            if claim in text:
                stale.append(f"{path.relative_to(REPO_ROOT)}: {claim!r}")

    assert stale == [], (
        "these passages still describe the retired fail-open path "
        f"(LP-0MW81DFML006O5KL): {stale}"
    )
