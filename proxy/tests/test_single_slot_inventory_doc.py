"""Doc-fidelity and production-geometry checks for the single-slot inventory.

LP-0MVTV6FGT000A19P (AC2/AC5) requires that the inventory table in
``docs/dev/single-slot-impact.md`` records, for **every** mechanism and metric
on the local dispatch path, its value at 2 slots, its value at 1 slot, one of
the three canonical classifications and ``file:line`` evidence — and that the
derived numbers are the numbers production computes rather than hand
arithmetic. The parent (LP-0MVTF26NP003KXKJ AC1/AC2) is only satisfied if a
future operator can trust that table without re-deriving it.

Four things are asserted here:

1. **Coverage.** Every mechanism the parent's AC1 names is present in the
   table, so a mechanism cannot be dropped from the audit silently.
2. **Row shape.** Every row carries all four fields, its classification is one
   of the three canonical values, and its evidence names a ``path:line``
   citation or an endpoint.
3. **Fidelity + behaviour.** The four routing-threshold rows equal the output
   of the production helpers (``effective_per_slot_threshold``,
   ``_effective_large_context_thresholds``, ``compute_hard_routing_cap``) at 1
   and at 2 slots, using the same ``geometry_config`` fixture the pinned
   derivation tests use. If the helper or the shipped config changes, this
   fails loudly instead of leaving the report stale.
4. **Switch list.** Every ``key: old -> new`` entry in the verdict still
   matches the shipped ``proxy/config.yaml``, so the report's instruction list
   cannot drift away from the config it describes.

Nothing here fabricates a runtime measurement: no llama-server, GPU or network
access is required.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from config_test_utils import geometry_config, load_profile
from proxy.provider import (
    _effective_large_context_thresholds,
    compute_hard_routing_cap,
    effective_per_slot_threshold,
)

#: Repo root (``proxy/tests/`` → ``proxy/`` → repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]

DOC_PATH = REPO_ROOT / "docs/dev/single-slot-impact.md"

#: Geometry the report is written against (``proxy/config.yaml:1431``).
TOTAL_CTX = 262_144

#: The three canonical classifications (report §2).
CLASSIFICATIONS = ("derived-auto", "static-needs-manual-change", "not-applicable")

#: Which inventory section each required mechanism must appear in. Values are
#: case-insensitive needles matched against the row's *mechanism* cell, so an
#: unrelated mention in prose cannot satisfy the check.
REQUIRED_MECHANISMS = {
    "per-slot context": ("per-slot context",),
    "physical per-slot clamp": ("physical per-slot clamp",),
    "economic ceiling": ("economic ceiling",),
    "absolute warm cap": ("absolute warm cap",),
    "cold threshold": ("cold", "threshold"),
    "hard-routing cap": ("hard-routing cap",),
    "local_session_tenants": ("local_session_tenants",),
    "dispatch lease pool (max_local)": ("max_local",),
    "local_max_concurrent_prefills": ("local_max_concurrent_prefills",),
    "contention-queue wait": ("contention_queue_max_wait_seconds",),
    "contention-queue depth": ("contention_queue_max_depth",),
    "contention-queue lease grace": ("contention_queue_lease_grace_seconds",),
    "contention-queue ETA ordering": ("eta_ordering",),
    "slot-exhaustion retries": (
        "local_slot_exhaustion_retry_attempts",
        "_retry_delay_seconds",
    ),
    "/health": ("/health",),
    "/llama/local/status": ("/llama/local/status",),
    "per-slot detail": ("per-slot detail",),
    "llama_slots_* counters": ("llama_slots_",),
    "herdr consumption of the status endpoint": ("herdr",),
    "local busy %": ("local busy %",),
    "turns served locally": ("turns served locally",),
    "input/output tokens processed locally": ("tokens processed locally",),
    "diverts by reason": ("diverts by reason",),
    "queue waits": ("queue waits",),
    "dispatch-to-first-byte": ("dispatch-to-first-byte",),
    "prefill slot-seconds": ("prefill slot-seconds",),
    "decode slot-seconds": ("decode slot-seconds",),
    "decode tok/s": ("decode tok/s",),
}

#: Evidence must look like a citation: ``path:line`` (optionally a range) or an
#: HTTP endpoint path.
_EVIDENCE_CITATION = re.compile(r"[\w./-]+\.(?:py|yaml|md|ini):\d+")
_EVIDENCE_ENDPOINT = re.compile(r"`/(?:health|llama/local/status|slots)`")

#: ``key: <old> -> <new>`` lines in the verdict's fenced blocks.
_SWITCH_LINE = re.compile(
    r"^(?P<key>[a-z_]+):\s+(?P<old>\S+)\s+->\s+(?P<new>\S+)$", re.MULTILINE
)


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


def _table_rows() -> list[dict[str, str]]:
    """Parse every inventory row out of §3.

    Rows are ``| <n> | mechanism | @2 | @1 | classification | evidence |``.
    Header and separator rows are skipped, as are the §4 provenance tables
    (which are outside §3).
    """
    rows: list[dict[str, str]] = []
    for line in _section("## 3. Inventory", "## 4. Method and provenance").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or not cells[0].isdigit():
            continue
        rows.append(
            {
                "row": cells[0],
                "mechanism": cells[1],
                "at_2_slots": cells[2],
                "at_1_slot": cells[3],
                # The verdict cell is rendered as `` `derived-auto` `` so the
                # verdict reads as a value; normalise before comparing.
                "classification": cells[4].strip("`"),
                "evidence": cells[5],
            }
        )
    assert rows, "could not parse any inventory rows out of §3"
    return rows


def _row_matching(*needles: str) -> dict[str, str]:
    """Return the single inventory row whose mechanism names every needle."""
    matches = [
        row
        for row in _table_rows()
        if all(n.lower() in row["mechanism"].lower() for n in needles)
    ]
    assert len(matches) == 1, (
        f"expected exactly one inventory row matching {needles!r}, "
        f"found {[m['row'] for m in matches]}"
    )
    return matches[0]


def _first_int(cell: str) -> int:
    """Extract the first integer from a value cell (``"0 (disabled)"`` → 0)."""
    match = re.search(r"\d[\d,]*", cell)
    assert match, f"no number in inventory value cell {cell!r}"
    return int(match.group(0).replace(",", ""))


def _geometry(slots: int) -> dict:
    """Shipped server block at *slots* slots (ctx held at 262,144)."""
    return geometry_config(slots)


class TestInventoryCoverage:
    """AC1: the table lists every mechanism on the local dispatch path."""

    @pytest.mark.parametrize("label", sorted(REQUIRED_MECHANISMS))
    def test_required_mechanism_is_inventoried(self, label: str) -> None:
        """A mechanism named by the ACs must have its own inventory row."""
        mechanisms = [row["mechanism"].lower() for row in _table_rows()]
        needles = REQUIRED_MECHANISMS[label]
        assert any(all(n in m for n in needles) for m in mechanisms), (
            f"required mechanism {label!r} has no inventory row in "
            f"{DOC_PATH.name}; looked for {needles!r} in the mechanism column"
        )


class TestRowShape:
    """AC2: every row records all four fields with a canonical verdict."""

    def test_every_row_has_four_populated_fields(self) -> None:
        incomplete = [
            row["row"]
            for row in _table_rows()
            if not (
                row["mechanism"]
                and row["at_2_slots"]
                and row["at_1_slot"]
                and row["classification"]
                and row["evidence"]
            )
        ]
        assert not incomplete, f"inventory rows missing a required field: {incomplete}"

    def test_every_classification_is_canonical(self) -> None:
        bad = [
            (row["row"], row["classification"])
            for row in _table_rows()
            if row["classification"] not in CLASSIFICATIONS
        ]
        assert not bad, (
            f"non-canonical classification(s) {bad}; "
            f"allowed: {', '.join(CLASSIFICATIONS)}"
        )

    def test_every_row_cites_evidence(self) -> None:
        uncited = [
            (row["row"], row["evidence"])
            for row in _table_rows()
            if not (
                _EVIDENCE_CITATION.search(row["evidence"])
                or _EVIDENCE_ENDPOINT.search(row["evidence"])
            )
        ]
        assert not uncited, (
            "inventory rows without a file:line (or endpoint) citation: "
            f"{uncited}"
        )

    def test_verdict_states_counts_matching_the_table(self) -> None:
        """The headline counts must equal the table's classification tally."""
        rows = _table_rows()
        tally = {name: 0 for name in CLASSIFICATIONS}
        for row in rows:
            tally[row["classification"]] += 1

        verdict = _section("### 1.1 Headline counts", "### 1.2")
        assert f"**{len(rows)} rows**" in verdict, (
            f"verdict must state the table's {len(rows)} rows; "
            f"re-counted classifications: {tally}"
        )
        for name, count in tally.items():
            # The verdict renders "17 auto-adjust", "5 are static", "19 are not
            # affected" — assert the number appears next to its classification.
            assert str(count) in verdict, (
                f"verdict omits the {name} count ({count}); table tally: {tally}"
            )


class TestDerivedThresholdFidelity:
    """AC2/AC5: the derived numbers are production's, not hand arithmetic."""

    @pytest.mark.parametrize("slots", [1, 2])
    def test_physical_clamp_row_matches_helper(self, slots: int) -> None:
        row = _row_matching("physical per-slot clamp")
        cell = row["at_2_slots"] if slots == 2 else row["at_1_slot"]
        assert _first_int(cell) == effective_per_slot_threshold(TOTAL_CTX, slots), (
            f"physical per-slot clamp row is stale at {slots} slot(s)"
        )

    @pytest.mark.parametrize("slots", [1, 2])
    def test_effective_warm_ceiling_row_matches_helper(self, slots: int) -> None:
        _, warm = _effective_large_context_thresholds(_geometry(slots))
        row = _row_matching("effective warm ceiling")
        cell = row["at_2_slots"] if slots == 2 else row["at_1_slot"]
        assert _first_int(cell) == warm, (
            f"effective warm ceiling row is stale at {slots} slot(s): "
            f"production says {warm}"
        )

    @pytest.mark.parametrize("slots", [1, 2])
    def test_cold_threshold_row_matches_helper(self, slots: int) -> None:
        cold, _ = _effective_large_context_thresholds(_geometry(slots))
        row = _row_matching("cold")
        cell = row["at_2_slots"] if slots == 2 else row["at_1_slot"]
        assert _first_int(cell) == cold, (
            f"cold threshold row is stale at {slots} slot(s): "
            f"production says {cold}"
        )

    @pytest.mark.parametrize("slots", [1, 2])
    def test_hard_routing_cap_row_matches_helper(self, slots: int) -> None:
        cap = compute_hard_routing_cap("fast", _geometry(slots))
        # The trailing parenthesis keeps this off the effective-warm-ceiling
        # row, whose mechanism text also mentions the hard-routing cap.
        row = _row_matching("hard-routing cap (")
        cell = row["at_2_slots"] if slots == 2 else row["at_1_slot"]
        assert _first_int(cell) == cap, (
            f"hard-routing cap row is stale at {slots} slot(s): "
            f"production says {cap} (ratio keys retired)"
        )

    def test_double_per_slot_context_routes_nothing_extra(self) -> None:
        """AC4: the warm ceiling is identical at both geometries.

        The key sentence of the report: the physical clamp doubles to 258,048
        but the warm ceiling stays at 120,000, so doubling the per-slot context
        routes nothing extra until the warm cap and the ratio move together.
        """
        _, warm_1 = _effective_large_context_thresholds(_geometry(1))
        _, warm_2 = _effective_large_context_thresholds(_geometry(2))
        clamp_1 = effective_per_slot_threshold(TOTAL_CTX, 1)

        assert warm_1 == warm_2, (
            "effective warm ceiling must be geometry-invariant: "
            f"got {warm_1} at 1 slot and {warm_2} at 2 slots"
        )
        assert warm_1 < clamp_1, (
            "at 1 slot the physical clamp must NOT be the binding term: "
            f"warm {warm_1} vs clamp {clamp_1}"
        )

        statement = _section("### 1.4", "## 2. Classification vocabulary")
        assert "routes nothing extra on its own" in statement, (
            "§1.4 must state explicitly that doubling the per-slot context "
            "routes nothing extra on its own (AC4)"
        )


class TestSwitchKeyList:
    """AC3: the ``key: old -> new`` list still matches the shipped config."""

    def _switch_pairs(self) -> list[tuple[str, str, str]]:
        verdict = _section("### 1.3", "### 1.4")
        pairs = [
            (m.group("key"), m.group("old"), m.group("new"))
            for m in _SWITCH_LINE.finditer(verdict)
        ]
        assert pairs, "verdict §1.3 lists no `key: old -> new` entries"
        return pairs

    def test_every_listed_key_exists_in_the_shipped_config(self) -> None:
        server_cfg = load_profile("config.yaml")["server"]
        missing = [key for key, _old, _new in self._switch_pairs() if key not in server_cfg]
        assert not missing, (
            f"verdict names config keys that are not in proxy/config.yaml: {missing}"
        )

    def test_every_listed_old_value_matches_the_shipped_config(self) -> None:
        """A renamed or retuned constant must fail this, not the reader."""
        server_cfg = load_profile("config.yaml")["server"]
        stale = []
        for key, old, _new in self._switch_pairs():
            current = server_cfg[key]
            matches = (
                current == float(old)
                if isinstance(current, float)
                else str(current) == old
            )
            if not matches:
                stale.append((key, old, current))
        assert not stale, (
            "verdict §1.3 `old` values no longer match proxy/config.yaml "
            f"(key, documented old, shipped value): {stale}"
        )

    def test_the_switch_moves_geometry_warm_cap_ratio_and_tenants(self) -> None:
        """The four keys a coherent switch needs, and nothing else.

        ``session_slot_pool_size`` is the switch; the warm cap and the economic
        ratio are the documented co-change (raising the cap alone is a no-op,
        LP-0MVHSET0A008VU95); ``local_session_tenants`` is explicit, so it does
        not follow the pool the way its default would (AC3).
        """
        changed = {key for key, old, new in self._switch_pairs() if old != new}
        assert changed == {
            "session_slot_pool_size",
            "local_large_context_economic_ceiling_ratio",
            "local_large_context_warm_cache_threshold",
            "local_session_tenants",
        }, f"unexpected set of changed keys in the verdict: {sorted(changed)}"

    def test_unchanged_keys_are_documented_as_unchanged(self) -> None:
        """Cold and total context must be listed and must not move."""
        unchanged = {
            key: (old, new) for key, old, new in self._switch_pairs() if old == new
        }
        assert set(unchanged) == {
            "local_large_context_cold_cache_threshold",
            "local_model_ctx_size",
        }, f"unexpected unchanged-key list: {sorted(unchanged)}"

        server_cfg = load_profile("config.yaml")["server"]
        assert str(server_cfg["local_large_context_cold_cache_threshold"]) == (
            unchanged["local_large_context_cold_cache_threshold"][0]
        )
        assert str(server_cfg["local_model_ctx_size"]) == (
            unchanged["local_model_ctx_size"][0]
        )
