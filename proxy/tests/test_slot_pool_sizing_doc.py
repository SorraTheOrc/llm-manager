"""Doc-consistency and production-geometry tests for the pool-4 sizing doc.

LP-0MVOG8T9F0031JP5 (parent LP-0MVN3KPEC007WWL5 AC2/AC6) requires that the
per-slot context table in
``docs/dev/slot-snapshot-pool-sizing-LP-0MVN3KPEC007WWL5.md`` is **produced by
the same helper production uses** (``proxy.provider.effective_per_slot_threshold``)
rather than hand-computed, and that the doc's ``file:line`` citations still
resolve to the behaviour they claim.

Three things are asserted here:

1. **Behaviour (production geometry).** The helper returns 126,976 / 83,285 /
   61,440 for pool 2 / 3 / 4 at ``local_model_ctx_size = 262,144``, the gap from
   the raw division is exactly ``_LOCAL_ROUTING_OUTPUT_HEADROOM``, and the
   routing warm clamp (``_effective_large_context_thresholds``) agrees with it —
   i.e. the doc's numbers are the numbers the router and the persistence gate
   actually use, not a documentation artefact.
2. **Doc table fidelity.** Every row of the pool-2/3/4 table in the doc parses
   to the helper's output. If someone edits the table without re-running the
   helper (or the helper changes), this fails.
3. **Citation drift.** Every ``file:line`` the doc relies on is checked to still
   point at the line it cites (the named symbol / setting is present). A moved
   line fails loudly instead of leaving a silently stale citation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from proxy.provider import (
    _LOCAL_ROUTING_OUTPUT_HEADROOM,
    _effective_large_context_thresholds,
    effective_per_slot_threshold,
)

#: Repo root (``proxy/tests/`` → ``proxy/`` → repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: How far a cited symbol may drift from the line the doc quotes before the
#: citation is considered stale. ``dev`` is a shared branch: unrelated commits
#: land under a doc's citations and shift them by a few dozen lines without the
#: cited behaviour changing at all. A window keeps the check honest (a symbol
#: that moves far away, or disappears, still fails) without reddening the suite
#: for a refactor in another file.
CITATION_DRIFT_TOLERANCE = 60

DOC_PATH = REPO_ROOT / "docs/dev/slot-snapshot-pool-sizing-LP-0MVN3KPEC007WWL5.md"

#: The doc's stated production geometry.
TOTAL_CTX = 262_144
POOLS = (2, 3, 4)

#: Expected raw division and clamp, in the doc's order.
EXPECTED_RAW = {2: 131_072, 3: 87_381, 4: 65_536}
EXPECTED_CLAMP = {2: 126_976, 3: 83_285, 4: 61_440}


def _doc_text() -> str:
    assert DOC_PATH.is_file(), f"sizing doc missing: {DOC_PATH}"
    return DOC_PATH.read_text(encoding="utf-8")


def _resolve_citation(
    path: Path, line_no: int, needle: str
) -> tuple[int | None, int]:
    """Locate *needle* in *path* and measure its distance from *line_no*.

    Returns ``(nearest_line, drift)``, or ``(None, -1)`` when the needle does
    not appear anywhere in the file (the cited behaviour was removed).
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = [n for n, text in enumerate(lines, start=1) if needle in text]
    if not hits:
        return None, -1
    nearest = min(hits, key=lambda hit: abs(hit - line_no))
    return nearest, abs(nearest - line_no)


def _table_rows() -> dict[int, tuple[int, int]]:
    """Parse the pool-2/3/4 geometry table out of the doc.

    Returns ``{pool: (raw_per_slot_ctx, clamp)}``. The table is identified by
    its header, so an unrelated table elsewhere in the doc cannot satisfy the
    assertion.
    """
    rows: dict[int, tuple[int, int]] = {}
    in_table = False
    for line in _doc_text().splitlines():
        if line.startswith("| Pool N |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cells or not cells[0].isdigit():
            continue  # separator / header separator row
        pool = int(cells[0])
        raw = int(cells[1].replace(",", ""))
        # The clamp cell is bolded in the doc: **126,976**
        clamp = int(re.sub(r"[^0-9]", "", cells[2]))
        rows[pool] = (raw, clamp)
    assert rows, "could not parse the per-slot geometry table from the doc"
    return rows


class TestProductionGeometry:
    """The numbers the doc quotes are the numbers production computes."""

    @pytest.mark.parametrize("pool", POOLS)
    def test_raw_per_slot_is_ctx_divided_by_pool(self, pool: int) -> None:
        """llama-server divides ``--ctx-size`` by ``--parallel``: raw per-slot."""
        assert TOTAL_CTX // pool == EXPECTED_RAW[pool]
        # And the helper agrees that its input is the same raw division.
        assert effective_per_slot_threshold(TOTAL_CTX, pool) == (
            TOTAL_CTX // pool - _LOCAL_ROUTING_OUTPUT_HEADROOM
        )

    @pytest.mark.parametrize("pool", POOLS)
    def test_helper_returns_the_documented_clamp(self, pool: int) -> None:
        """``effective_per_slot_threshold`` is the routing/persistence clamp."""
        assert effective_per_slot_threshold(TOTAL_CTX, pool) == EXPECTED_CLAMP[pool]

    def test_clamp_is_exactly_headroom_below_raw(self) -> None:
        """The 4,096 headroom is the whole difference between 65,536 and 61,440.

        This is the doc's central warning: the headline "65,536 at pool 4" is
        not the usable context.
        """
        assert _LOCAL_ROUTING_OUTPUT_HEADROOM == 4_096
        for pool in POOLS:
            assert EXPECTED_RAW[pool] - EXPECTED_CLAMP[pool] == 4_096

    @pytest.mark.parametrize("pool", POOLS)
    def test_routing_warm_ceiling_matches_the_clamp(self, pool: int) -> None:
        """The warm routing ceiling is clamped to the same per-slot value.

        A configured warm threshold above the clamp must come back clamped, so
        the router, the persistence cap and the doc all use one number.
        """
        config = {
            "server": {
                "local_large_context_cold_cache_threshold": 38_000,
                "local_large_context_warm_cache_threshold": 250_000,
                "local_model_ctx_size": TOTAL_CTX,
                "session_slot_pool_size": pool,
            }
        }
        cold, warm = _effective_large_context_thresholds(config)
        assert warm == EXPECTED_CLAMP[pool]
        assert cold == 38_000  # cold is economic, never clamped

    def test_pool_4_erodes_the_headroom_above_the_thrash_window_prompt(self) -> None:
        """Why candidate A is not free: pool 4 shrinks the headroom 6.3x.

        The 2026-10-24 thrash window's dominant prompt is 49,103 tokens
        (LP-0MVN1IQTF0037WTG). It still fits inside the pool-4 clamp (61,440),
        but the headroom above it collapses from 77,873 tokens at pool 2 to
        12,337 at pool 4, so any prompt growth falls off the local path. Assert
        the numbers so a future ctx-size change re-opens the discussion instead
        of silently tightening the clamp.
        """
        pool2 = effective_per_slot_threshold(TOTAL_CTX, 2)
        pool4 = effective_per_slot_threshold(TOTAL_CTX, 4)
        dominant_prompt = 49_103
        assert dominant_prompt < pool4 < pool2
        assert pool4 - dominant_prompt == 12_337
        assert pool2 - dominant_prompt == 77_873
        assert round((pool2 - dominant_prompt) / (pool4 - dominant_prompt), 1) == 6.3


class TestDocTableFidelity:
    """The doc's table is the helper's output, not a hand calculation."""

    def test_table_has_one_row_per_candidate_pool(self) -> None:
        rows = _table_rows()
        assert sorted(rows) == list(POOLS)

    @pytest.mark.parametrize("pool", POOLS)
    def test_doc_row_matches_helper_output(self, pool: int) -> None:
        raw, clamp = _table_rows()[pool]
        assert raw == TOTAL_CTX // pool
        assert clamp == effective_per_slot_threshold(TOTAL_CTX, pool)

    def test_doc_states_the_derivation_inputs(self) -> None:
        """The table is meaningless unless the doc names its inputs."""
        text = _doc_text()
        assert "262,144" in text
        assert "4,096" in text
        assert "effective_per_slot_threshold" in text


class TestCitationsStillResolve:
    """Every ``file:line`` the doc leans on still points at the cited behaviour.

    Prevents a refactor from leaving the doc with silently wrong citations.
    """

    #: (relative path, 1-based line, needle that must be present on that line)
    CITATIONS: tuple[tuple[str, int, str], ...] = (
        # 1 — pool size exported as --parallel
        ("proxy/config.yaml", 1680, "session_slot_pool_size: 2"),
        ("proxy/proxy/lifecycle.py", 1604, "LLAMA_PARALLEL"),
        ("start-llama.sh", 239, "LLAMA_PARALLEL:=1"),
        ("start-llama.sh", 252, "--parallel"),
        ("start-llama.sh", 501, "LLAMA_PARALLEL:=1"),
        ("start-llama.sh", 511, "-np"),
        ("start-llama.sh", 514, "--parallel"),
        # 2/3 — per-slot context + routing clamp
        ("proxy/proxy/provider.py", 1881, "_LOCAL_ROUTING_OUTPUT_HEADROOM = 4096"),
        ("proxy/proxy/provider.py", 1890, "def effective_per_slot_threshold"),
        ("proxy/proxy/provider.py", 2002, "def _effective_large_context_thresholds"),
        ("proxy/proxy/provider.py", 2045, "effective_per_slot_threshold"),
        ("proxy/proxy/provider.py", 2209, "effective_per_slot_threshold"),
        # 4 — persistence cap derived from the same clamp
        ("proxy/proxy/provider.py", 2166, "def compute_hard_routing_cap"),
        ("proxy/proxy/session.py", 1305, "effective_per_slot_threshold"),
        # 5 — dispatch-lease cap
        ("proxy/proxy/router.py", 213, "def _get_local_max_concurrent_queries"),
        ("proxy/proxy/router.py", 1457, "_try_acquire_local_dispatch"),
        ("proxy/proxy/router_helpers.py", 3474, "async def _try_acquire_local_dispatch"),
        ("proxy/proxy/provider.py", 6464, "session_slot_pool_size"),
        # 6 — slot-ID space
        ("proxy/proxy/session.py", 1062, "def _slot_id_for_session"),
        # 7 — persistence on/off gate + eviction planner
        ("proxy/proxy/session.py", 488, "def _slot_persistence_enabled"),
        ("proxy/proxy/session.py", 1261, "session_slot_pool_size"),
        ("proxy/proxy/session.py", 848, "def plan_eviction_snapshot"),
        # 8 — tenancy default
        ("proxy/proxy/router_helpers.py", 3125, "def _default_tenants"),
        ("proxy/config.yaml", 1373, "session_slot_pool_size"),
        ("proxy/config.yaml", 1381, "local_session_tenants: 2"),
        # 9 — capacity admission + status fail-open
        ("proxy/proxy/capacity_admission.py", 124, "def _slots"),
        ("proxy/proxy/handlers.py", 426, "session_slot_pool_size"),
        ("proxy/proxy/handlers.py", 656, "session_slot_pool_size"),
        ("proxy/proxy/handlers.py", 702, "session_slot_pool_size"),
        ("proxy/proxy/handlers.py", 829, "session_slot_pool_size"),
        # geometry-independent retention + per-session snapshot filenames
        ("proxy/proxy/session.py", 1202, "def _slot_filename_for_session"),
        ("proxy/proxy/session.py", 1204, "slot_{safe_id}.bin"),
        ("scripts/cleanup-slot-cache.sh", 27, "MAX_AGE_DAYS=7"),
        ("scripts/cleanup-slot-cache.sh", 28, "MAX_SIZE_GB=20"),
        # persistence suppression gates named in the bake-off metrics
        ("proxy/proxy/session.py", 562, "def _slot_in_failure_cooldown"),
        ("proxy/proxy/session.py", 667, "def _slot_persistence_skip_when_busy"),
    )

    @pytest.mark.parametrize("rel_path,line_no,needle", CITATIONS)
    def test_citation_resolves(self, rel_path: str, line_no: int, needle: str) -> None:
        path = REPO_ROOT / rel_path
        assert path.is_file(), f"cited file missing: {rel_path}"
        nearest, drift = _resolve_citation(path, line_no, needle)
        assert nearest is not None, (
            f"{rel_path} no longer contains {needle!r} — the sizing doc cites it at "
            f"{rel_path}:{line_no}; update the doc (behaviour moved or removed)"
        )
        assert drift <= CITATION_DRIFT_TOLERANCE, (
            f"{rel_path}:{line_no} is stale: {needle!r} now lives at line {nearest} "
            f"(drift {drift} > {CITATION_DRIFT_TOLERANCE}) — update the sizing doc"
        )


class TestCitationDriftGuard:
    """The drift guard above must actually be able to fail.

    A citation check nobody has seen fail is indistinguishable from a no-op;
    these exercise it against synthetic sources so the tolerance window is a
    tested property rather than a hopeful constant.
    """

    def test_exact_citation_has_zero_drift(self, tmp_path: Path) -> None:
        src = tmp_path / "mod.py"
        src.write_text("a = 1\ndef target():\n    pass\n", encoding="utf-8")
        nearest, drift = _resolve_citation(src, 2, "def target")
        assert (nearest, drift) == (2, 0)

    def test_small_drift_is_tolerated(self, tmp_path: Path) -> None:
        src = tmp_path / "mod.py"
        src.write_text("\n" * 20 + "def target():\n    pass\n", encoding="utf-8")
        nearest, drift = _resolve_citation(src, 2, "def target")
        assert nearest == 21
        assert drift <= CITATION_DRIFT_TOLERANCE

    def test_large_drift_is_reported(self, tmp_path: Path) -> None:
        src = tmp_path / "mod.py"
        offset = CITATION_DRIFT_TOLERANCE + 10
        src.write_text("\n" * offset + "def target():\n    pass\n", encoding="utf-8")
        nearest, drift = _resolve_citation(src, 1, "def target")
        assert nearest == offset + 1
        assert drift > CITATION_DRIFT_TOLERANCE

    def test_missing_symbol_is_reported(self, tmp_path: Path) -> None:
        src = tmp_path / "mod.py"
        src.write_text("a = 1\n", encoding="utf-8")
        nearest, _ = _resolve_citation(src, 1, "def target")
        assert nearest is None
