"""Fidelity, provenance and arithmetic checks for the single-slot prediction.

LP-0MVTV6G0N0053V0G requires a request-level throughput prediction for a
1 x 262,144 slot geometry in which **every** conclusion names its mechanism,
its sourced baseline (with the measurement window and the source document), the
metric that would confirm it and the Arm A / Arm B comparison that would settle
it — including tokens processed locally with an explicit token-attribution rule,
the uncontended-evidence caveat, and a statement of what must be re-measured.

A prose document cannot be trusted to keep those promises by itself, so this
module makes them executable:

1. **Register shape.** §5 carries one row per required effect, and each row
   cites a mechanism ``path:line``, a window plus registered figures, an
   arm-metric name, and both arms.
2. **No unsourced number.** Every figure id mentioned in §5 must be registered
   in ``docs/dev/single-slot-throughput-baseline-2026-10-31.json`` and its
   rendered value must appear in the text — so editing a number in one place
   without the other fails the suite.
3. **Register integrity and arithmetic.** Every registered figure declares its
   provenance class and its source (or formula plus inputs), its rendered form
   agrees with its value, and the derived estimates recompute from their inputs.
4. **Production geometry.** The warm ceilings the prediction relies on are the
   values production computes at 1 and at 2 slots, with and without the §1.3
   co-change — not hand arithmetic.
5. **The AC3/AC4/AC5/AC6 statements are present**, with the provenance counts
   and caveat figures the ACs name.

No llama-server, GPU, network or host log access is required: the register is
the committed artifact and the geometry is resolved through the production
helpers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from config_test_utils import geometry_config
from proxy.provider import (
    _effective_large_context_thresholds,
    effective_per_slot_threshold,
)

#: Repo root (``proxy/tests/`` → ``proxy/`` → repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]

DOC_PATH = REPO_ROOT / "docs/dev/single-slot-impact.md"
REGISTER_PATH = REPO_ROOT / "docs/dev/single-slot-throughput-baseline-2026-10-31.json"

#: Geometry the prediction is written against (``proxy/config.yaml``).
TOTAL_CTX = 262_144

#: The §1.3 co-change that makes the extra per-slot context reachable.
ARM_B_ECONOMIC_RATIO = 0.9844
ARM_B_WARM_CAP = 258_048

#: Provenance classes the register may use.
PROVENANCE_CLASSES = ("measured", "derived-estimate", "config-derived")

#: Window labels the prediction must be able to cite.
WINDOW_LABELS = ("W1", "W2", "W3", "W4", "W5", "W6", "W7")

#: Effects the parent AC requires, in register order.
REQUIRED_EFFECTS = ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8")

#: Metric names the A/B captures (``arm.json``) or their archived analyzer
#: report expose. A register row's metric cell must name at least one of these.
ARM_METRICS = (
    "turns_served_locally",
    "diverts_by_reason",
    "context_ceiling_diverts",
    "local_busy_pct",
    "avg_concurrency",
    "peak_concurrency",
    "queue_waits",
    "fallback_after_queue",
    "queue_wait_seconds",
    "local_dispatch_first_byte",
    "ttft_all_streams",
    "prefill_slot_seconds",
    "decode_slot_seconds",
    "local_concurrency_limit",
    "slot_save_failed",
    "Decode speed",
)

#: ``path:line`` citation used for the mechanism column.
_MECHANISM_CITATION = re.compile(r"[\w./-]+\.(?:py|yaml|md|ini):\d+")

#: Suffix multipliers for rendered values such as ``11.5M``.
_SUFFIXES = {"": 1.0, "M": 1_000_000.0, "k": 1_000.0}


def _doc_text() -> str:
    assert DOC_PATH.is_file(), f"single-slot impact report missing: {DOC_PATH}"
    return DOC_PATH.read_text(encoding="utf-8")


def _section(start: str, end: str) -> str:
    """Return the text between the *start* heading and the *end* heading."""
    text = _doc_text()
    begin = text.find(start)
    assert begin != -1, f"report is missing the required section {start!r}"
    finish = text.find(end, begin)
    assert finish != -1, f"section {start!r} is not terminated by {end!r}"
    return text[begin:finish]


def _prediction() -> str:
    """The whole §5 throughput-prediction section."""
    return _section("## 5. Throughput prediction", "## 6. Metrics gap list")


def _register() -> dict:
    assert REGISTER_PATH.is_file(), f"baseline register missing: {REGISTER_PATH}"
    return json.loads(REGISTER_PATH.read_text(encoding="utf-8"))


def _entries() -> dict[str, dict]:
    """Every registered figure and derived estimate, keyed by id."""
    reg = _register()
    entries: dict[str, dict] = {}
    entries.update(reg["figures"])
    entries.update(reg["derived"])
    return entries


def _parse_rendered(rendered: str) -> float:
    """Parse a register ``rendered`` string into a float.

    Handles plain numbers, thousands separators, ``M``/``k`` suffixes and
    duration forms such as ``2h 26m`` or ``44m 57s``.
    """
    text = rendered.strip().replace(",", "")
    match = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h\s*)?(?:(\d+(?:\.\d+)?)m\s*)?(?:(\d+(?:\.\d+)?)s)?", text)
    if match and any(match.groups()):
        hours, minutes, seconds = (float(g) if g else 0.0 for g in match.groups())
        return hours * 3600 + minutes * 60 + seconds
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([Mk]?)", text)
    assert match, f"cannot parse rendered value {rendered!r}"
    return float(match.group(1)) * _SUFFIXES[match.group(2)]


def _register_rows() -> list[dict[str, str]]:
    """Parse the §5.3 prediction register into per-effect dicts."""
    block = _section("### 5.3 Prediction register", "### 5.4")
    rows: list[dict[str, str]] = []
    for line in block.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 7 or not re.fullmatch(r"E\d+", cells[0]):
            continue
        rows.append(
            {
                "effect": cells[0],
                "name": cells[1],
                "mechanism": cells[2],
                "baseline": cells[3],
                "prediction": cells[4],
                "metric": cells[5],
                "ab": cells[6],
            }
        )
    return rows


def _value(entries: dict[str, dict], figure_id: str) -> float:
    """Resolve a registered figure id, including ``window.key`` paths."""
    if figure_id in entries:
        return float(entries[figure_id]["value"])
    assert "." in figure_id, f"unregistered figure id {figure_id!r}"
    section, key = figure_id.split(".", 1)
    return float(_register()[section][key])


class TestPredictionSectionExists:
    """§5 is filled in, not a reservation."""

    def test_section_is_written(self) -> None:
        section = _prediction()
        assert "reserved for" not in section, (
            "§5 still says the section is reserved; the throughput-prediction slice (LP-0MVTV6G0N0053V0G) must fill it"
        )
        assert len(section) > 8_000, "§5 is too short to be a request-level analysis"

    def test_register_is_committed_next_to_the_report(self) -> None:
        section = _prediction()
        assert REGISTER_PATH.name in section, (
            f"§5 must name the committed baseline register it quotes ({REGISTER_PATH.name})"
        )


class TestPredictionRegister:
    """AC1/AC2: one complete row per effect, each with all four citations."""

    def test_every_required_effect_has_a_row(self) -> None:
        effects = [row["effect"] for row in _register_rows()]
        assert effects == list(REQUIRED_EFFECTS), (
            f"§5.3 must register exactly the required effects in order {list(REQUIRED_EFFECTS)}; found {effects}"
        )

    def test_every_row_is_populated(self) -> None:
        incomplete = [
            row["effect"]
            for row in _register_rows()
            if not all(row[key] for key in ("name", "mechanism", "baseline", "prediction", "metric", "ab"))
        ]
        assert not incomplete, f"register rows missing a field: {incomplete}"

    @pytest.mark.parametrize("effect", REQUIRED_EFFECTS)
    def test_row_cites_mechanism_in_code_or_config(self, effect: str) -> None:
        row = _row(effect)
        assert _MECHANISM_CITATION.search(row["mechanism"]), (
            f"{effect} must cite a mechanism as path:line; got {row['mechanism']!r}"
        )

    @pytest.mark.parametrize("effect", REQUIRED_EFFECTS)
    def test_row_cites_a_window_and_registered_figures(self, effect: str) -> None:
        row = _row(effect)
        entries = _entries()
        windows = [w for w in WINDOW_LABELS if re.search(rf"\b{w}\b", row["baseline"])]
        assert windows, (
            f"{effect} baseline must name its measurement window ({', '.join(WINDOW_LABELS)}); got {row['baseline']!r}"
        )
        figures = _figure_refs(row["baseline"])
        assert figures, f"{effect} baseline names no registered figure: {row['baseline']!r}"
        unknown = [f for f in figures if f not in entries]
        assert not unknown, f"{effect} baseline cites unregistered figures: {unknown}"

    @pytest.mark.parametrize("effect", REQUIRED_EFFECTS)
    def test_row_names_a_confirming_metric(self, effect: str) -> None:
        row = _row(effect)
        named = [m for m in ARM_METRICS if m in row["metric"]]
        assert named, (
            f"{effect} must name a metric the A/B can report ({', '.join(ARM_METRICS[:5])}, ...); got {row['metric']!r}"
        )

    @pytest.mark.parametrize("effect", REQUIRED_EFFECTS)
    def test_row_compares_both_arms(self, effect: str) -> None:
        row = _row(effect)
        assert "Arm A" in row["ab"] and "Arm B" in row["ab"], (
            f"{effect} must state the Arm A vs Arm B comparison; got {row['ab']!r}"
        )

    def test_every_required_effect_is_covered_by_name(self) -> None:
        """The eight required effects must be the AC's effects, not filler."""
        names = " ".join(row["name"].lower() for row in _register_rows())
        for needle in (
            "turns served locally",
            "prompt tokens",
            "completion tokens",
            "first-byte",
            "decode",
            "head-of-line",
            "queue waits",
            "re-prefill",
            "headroom",
        ):
            assert needle in names, f"prediction register has no effect for {needle!r}"


def _row(effect: str) -> dict[str, str]:
    """Return the register row for *effect*."""
    matches = [row for row in _register_rows() if row["effect"] == effect]
    assert matches, f"{effect} has no register row"
    return matches[0]


def _figure_refs(cell: str) -> list[str]:
    """Registered figure ids named in *cell* (``W1.local_turns`` / ``D3``)."""
    entries = _entries()
    return [figure_id for figure_id in entries if re.search(re.escape(figure_id) + r"(?![0-9A-Za-z_.])", cell)]


class TestNoUnsourcedNumbers:
    """AC2/AC3/AC5: §5 quotes only registered figures, and quotes them intact."""

    def test_every_quoted_figure_is_registered_and_consistent(self) -> None:
        section = _prediction()
        entries = _entries()
        quoted = [figure_id for figure_id in entries if re.search(re.escape(figure_id) + r"(?![0-9A-Za-z_.])", section)]
        assert len(quoted) >= 30, (
            "§5 should quote a substantial number of registered figures; "
            f"only {len(quoted)} found — is the register wired to the text?"
        )
        stale = [
            (figure_id, entries[figure_id]["rendered"])
            for figure_id in quoted
            if entries[figure_id]["rendered"] not in section
        ]
        assert not stale, (
            f"§5 quotes these figures but not their registered value {stale}; update the text and the register together"
        )

    def test_register_ids_are_unique_and_prefixed(self) -> None:
        entries = _entries()
        for figure_id in entries:
            assert re.fullmatch(r"(?:W\d|D\d+)\.?[A-Za-z0-9_.]*", figure_id), (
                f"unexpected register id shape: {figure_id!r}"
            )


class TestRegisterIntegrity:
    """AC3/AC6: provenance, sources and arithmetic of the register itself."""

    def test_every_figure_declares_a_provenance_class(self) -> None:
        bad = [
            (figure_id, entry.get("provenance"))
            for figure_id, entry in _entries().items()
            if entry.get("provenance") not in PROVENANCE_CLASSES
        ]
        assert not bad, f"figures without a canonical provenance class: {bad}"

    def test_every_figure_declares_a_source_or_formula(self) -> None:
        bad = [
            figure_id
            for figure_id, entry in _entries().items()
            if not (entry.get("source") or (entry.get("formula") and entry.get("inputs")))
        ]
        assert not bad, f"figures with neither a source nor a formula: {bad}"

    def test_derived_estimates_name_their_inputs(self) -> None:
        entries = _entries()
        bad = []
        for figure_id, entry in entries.items():
            if entry["provenance"] != "derived-estimate":
                continue
            inputs = entry.get("inputs") or []
            if not inputs:
                bad.append((figure_id, "no inputs"))
                continue
            for input_id in inputs:
                if input_id.startswith("geometry."):
                    continue
                if input_id not in entries:
                    bad.append((figure_id, f"unknown input {input_id}"))
        assert not bad, f"derived estimates with unresolved inputs: {bad}"

    def test_rendered_values_agree_with_values(self) -> None:
        bad = []
        for figure_id, entry in _entries().items():
            parsed = _parse_rendered(entry["rendered"])
            value = float(entry["value"])
            tolerance = max(0.06, abs(value) * 0.005)
            if abs(parsed - value) > tolerance:
                bad.append((figure_id, entry["rendered"], value))
        assert not bad, f"rendered form disagrees with the recorded value: {bad}"

    def test_windows_record_span_and_or_reason(self) -> None:
        windows = _register()["windows"]
        assert set(windows) == set(WINDOW_LABELS)
        for label, window in windows.items():
            assert window["source_document"], f"{label} has no source document"
            assert window["retained"], f"{label} does not say whether it is retained"
            if window.get("start") is None or window.get("end") is None:
                assert window.get("retention_note"), f"{label} has no span, so it must explain why"


class TestDerivedArithmetic:
    """AC3/AC6: the estimates recompute from the measured figures."""

    def test_local_turn_overlap_bound(self) -> None:
        entries = _entries()
        overlap = _value(entries, "W1.compute_slot_seconds") - _value(entries, "W1.busy_seconds")
        assert overlap == pytest.approx(_value(entries, "D1.overlap_slot_seconds"), abs=1)
        assert overlap / _value(entries, "W1.busy_seconds") * 100 == pytest.approx(
            _value(entries, "D2.overlap_share_of_busy_pct"), abs=0.06
        )
        turns_at_risk = 2 * overlap / _value(entries, "W1.avg_stream_duration_seconds")
        assert turns_at_risk == pytest.approx(_value(entries, "D3.turns_at_risk"), abs=0.5)
        assert turns_at_risk / _value(entries, "W1.local_turns") * 100 == pytest.approx(
            _value(entries, "D4.turns_at_risk_pct"), abs=0.1
        )

    def test_single_slot_headroom(self) -> None:
        entries = _entries()
        used = _value(entries, "W1.compute_slot_seconds") / 86_400 * 100
        assert used == pytest.approx(_value(entries, "D5.single_slot_capacity_use_pct"), abs=0.06)
        assert 100 - used == pytest.approx(_value(entries, "D6.single_slot_headroom_pct"), abs=0.06)

    def test_per_turn_token_means(self) -> None:
        entries = _entries()
        prompt = _value(entries, "W1.local_prompt_tokens") / _value(entries, "W1.local_turns")
        completion = _value(entries, "W1.local_completion_tokens") / _value(entries, "W1.local_turns")
        assert prompt == pytest.approx(_value(entries, "D7.prompt_tokens_per_local_turn"), rel=0.005)
        assert completion == pytest.approx(_value(entries, "D8.completion_tokens_per_local_turn"), abs=0.5)

    def test_worst_case_reprefill_increment(self) -> None:
        entries = _entries()
        extra = _value(entries, "W2.local_prompt_tokens") * _value(entries, "W7.steal_pct_pool1_two_sessions") / 100
        assert extra == pytest.approx(_value(entries, "D9.worst_case_extra_reprefill_tokens"), rel=0.005)
        assert extra / _value(entries, "W5.reprefill_pool_tokens_per_day") * 100 == (
            pytest.approx(_value(entries, "D10.reprefill_pool_increase_pct"), abs=0.1)
        )

    def test_ceiling_divert_conversion_capacity(self) -> None:
        entries = _entries()
        ceiling_diverts = _value(entries, "W1.ceiling_diverts")
        assert ceiling_diverts == _value(entries, "W1.divert_context_too_large") + _value(
            entries, "W1.divert_large_context_bypass"
        )
        cold_volume = ceiling_diverts * 120_000 * _value(entries, "D18.cold_share_of_prompts_pct") / 100
        assert cold_volume == pytest.approx(_value(entries, "D20.cold_token_lower_bound"), rel=0.005)
        capacity = 86_400 * _value(entries, "W3.solo_new_tok_s_p50")
        assert capacity == pytest.approx(_value(entries, "D21.single_slot_prefill_capacity_tokens"), rel=0.005)
        assert capacity / cold_volume * 100 == pytest.approx(
            _value(entries, "D22.convertible_share_of_ceiling_pool_pct"), abs=0.06
        )
        assert capacity / 120_000 == pytest.approx(_value(entries, "D23.convertible_turns_per_day"), abs=0.5)
        assert 120_000 / _value(entries, "W3.solo_new_tok_s_p50") == pytest.approx(
            _value(entries, "D24.first_byte_per_converted_prompt_seconds"), abs=0.5
        )

    def test_local_token_totals_add_up(self) -> None:
        for window in ("W1", "W2"):
            entries = _entries()
            total = _value(entries, f"{window}.local_prompt_tokens") + _value(
                entries, f"{window}.local_completion_tokens"
            )
            assert total == pytest.approx(_value(entries, f"{window}.local_total_tokens"), rel=0.0001)


class TestProductionGeometry:
    """AC2/AC5: the thresholds the prediction leans on come from production."""

    def test_warm_ceiling_at_both_geometries_without_cochange(self) -> None:
        _, warm_1 = _effective_large_context_thresholds(geometry_config(1))
        _, warm_2 = _effective_large_context_thresholds(geometry_config(2))
        entries = _entries()
        assert warm_1 == warm_2 == _value(entries, "D17.warm_ceiling_arm_b_without_cochange")
        assert warm_2 == _value(entries, "D15.warm_ceiling_arm_a")
        assert effective_per_slot_threshold(TOTAL_CTX, 1) == 258_048
        assert effective_per_slot_threshold(TOTAL_CTX, 2) == 126_976

    def test_warm_ceiling_at_one_slot_with_cochange(self) -> None:
        config = geometry_config(1)
        config["server"]["local_large_context_economic_ceiling_ratio"] = ARM_B_ECONOMIC_RATIO
        config["server"]["local_large_context_warm_cache_threshold"] = ARM_B_WARM_CAP
        _, warm = _effective_large_context_thresholds(config)
        entries = _entries()
        assert warm == _value(entries, "D16.warm_ceiling_arm_b_with_cochange") == 258_048

    def test_register_geometry_block_matches_the_helpers(self) -> None:
        geometry = _register()["geometry"]
        for arm, slots in (("arm_a", 2), ("arm_b", 1)):
            config = geometry_config(slots)
            if arm == "arm_b":
                config["server"]["local_large_context_economic_ceiling_ratio"] = ARM_B_ECONOMIC_RATIO
                config["server"]["local_large_context_warm_cache_threshold"] = ARM_B_WARM_CAP
            _, warm = _effective_large_context_thresholds(config)
            assert geometry[arm]["effective_warm_ceiling"] == warm, f"{arm} warm ceiling in the register is stale"
            assert geometry[arm]["physical_clamp"] == effective_per_slot_threshold(TOTAL_CTX, slots)


class TestAcceptanceStatements:
    """AC3/AC4/AC5/AC6 statements are present with their figures."""

    def test_token_provenance_is_stated(self) -> None:
        provenance = _section("### 5.5 Token provenance", "### 5.6")
        assert "Stream finished" in provenance, (
            "the authoritative token source (Stream finished tokens= lines) must be named"
        )
        assert "tokens=missing" in provenance, "the report must state that non-reporting lines carry tokens=missing"
        # AC3 requires the count of lines lacking token evidence.
        entries = _entries()
        for figure_id in (
            "W2.lines_without_usage",
            "W2.finished_lines",
            "W2.local_lines_without_usage",
            "W2.local_turns",
        ):
            rendered = entries[figure_id]["rendered"]
            assert rendered in provenance, (
                f"the token-evidence count {figure_id} ({rendered}) must be quoted in the provenance subsection"
            )
        # The local-versus-remote split rule must be stated.
        lowered = provenance.lower()
        assert "local turn" in lowered and "remote" in lowered and "% local turns" in provenance

    def test_uncontended_caveat_is_stated(self) -> None:
        caveat = _section("### 5.6 The uncontended-evidence caveat", "### 5.7")
        entries = _entries()
        for figure_id in ("W1.idle_pct", "W1.avg_concurrency_while_busy", "W2.idle_pct"):
            rendered = entries[figure_id]["rendered"]
            assert rendered in caveat, f"the caveat must quote {figure_id} ({rendered})"
        assert "under-observed" in caveat, "the caveat must say the serialisation cost is under-observed today"
        assert "may be presented as a measured single-slot result" in caveat, (
            "the caveat must forbid presenting the prediction as a measured result"
        )

    def test_conditional_knobs_match_the_switch_list(self) -> None:
        """AC5: every static knob in §1.3 is named as a condition in §5.7."""
        verdict = _section("### 1.3", "### 1.4")
        conditional = _section("### 5.7", "### 5.8")
        keys = re.findall(r"^(?P<key>[a-z_]+):\s+\S+\s+->", verdict, re.MULTILINE)
        assert keys, "could not parse the §1.3 switch list"
        missing = [key for key in keys if key not in conditional]
        assert not missing, f"§5.7 does not state that these static knobs condition the prediction: {missing}"

    def test_remeasurement_table_names_confirming_measurements(self) -> None:
        block = _section("### 5.8 What must be re-measured", "### 5.9")
        rows = [
            [c.strip() for c in line.strip().strip("|").split("|")]
            for line in block.splitlines()
            if line.startswith("|")
        ][2:]
        rows = [r for r in rows if len(r) == 3]
        assert len(rows) >= 5, (
            f"§5.8 must list the claims that depend on a baseline needing re-measurement; found {len(rows)} rows"
        )
        thin = [r[0] for r in rows if not (r[1] and r[2])]
        assert not thin, f"§5.8 rows without a reason or a confirming measurement: {thin}"

    def test_prediction_is_not_presented_as_the_geometry_verdict(self) -> None:
        section = _prediction()
        assert "desk prediction" in section.lower() or "not a measurement" in section.lower()
        assert "docs/dev/slot-geometry-ab.md" in section, (
            "§5 must point at the A/B protocol that owns the geometry verdict"
        )
