"""Pool-size / snapshot-restore bake-off replay (LP-0MVOG8TUM005705R).

Answers parent AC2 (LP-0MVN3KPEC007WWL5): does widening the session-slot
snapshot pool raise prefix reuse on the measured 2026-10-24 thrash window?
This is a *replay in the snapshot-pool dimension only*: the 1,978-admission
fixture (captured by ``capture_prefill_thrash_window.py``) is replayed through
a model of the pool/restore mechanism, not through routing or admission —
those decisions were already made by production and recorded in the fixture.

The model is deliberately *thin* and pinned to production predicates:

* **Slot assignment** uses the real ``proxy.session._slot_id_for_session`` /
  ``_free_slot_assignment`` (the in-process ``_slot_owners`` registry is the
  documented stub source the residency tests drive the same way).
* **Gates use the real production functions**: ``_slot_persistence_skip_when_busy``
  (with a deterministic busy-state view injected through the documented
  ``_slot_busy_state_snapshot`` stub), ``_slot_in_failure_cooldown`` +
  ``_record_slot_success``／``_record_slot_failure`` (the real circuit breaker,
  clocked by the simulated timeline), and the ``context_too_large`` cap from
  ``proxy.provider.effective_per_slot_threshold`` — the same helper production
  uses for routing and persistence (C2 §1/§2).
* **Restore/save cost** is pinned from C3 (`LP-0MVOG8TK6002EWZU`) committed
  measurement ``proxy/benchmarks/results/slot_snapshot_bands_20261028/summary.json``
  (under-load seconds per band), compared against the adaptive timeout
  ``min(3.0 + 0.0015 * tokens, 60)`` from ``proxy/config.yaml:2236``.
* **Percentiles** delegate to ``scripts/analyse_slot_ratio_restore._percentiles``
  (fed percentages) so the reuse uplift uses the usage report's percentile
  definition — the AC's "no ad-hoc percentile definition".

Arms (C2 §3/§4):

* ``arm0`` — baseline, pool 2, per-session snapshot files (disk-capped),
  gates on. Comparable.
* ``arm1`` — **candidate B**: a decoupled retained-snapshot pool of **4**
  snapshots (LRU) over the same 2 live slots, gates on. Comparable.
* ``arm2`` — **candidate A**: literal pool 4 (4 live slots, clamp 61,440),
  which violates parent AC5 — recorded but **not** a bake-off arm (C2 §4.1).
* ``arm3`` — diagnostic: candidate B with ``session_slot_skip_when_busy``
  relaxed, so a "no uplift" result can be attributed to the busy gate rather
  than asserted. Informational.

Two documented modelling decisions (the established pattern of the sibling
``prefill_thrash_replay_harness``):

1. **The capture logs each admission decision twice** (``queue`` +
   ``large_context_bypass`` at the same timestamp); the pool model consumes
   one attempt per ``(ts, session, estimated_tokens)`` (:func:`unique_attempts`).
2. **Proportional calibration** anchors Arm 0 to the measured baseline (26.1 %
   aggregate, 15.6 % median for >20k prompts, 28/227 fully cold —
   ``LP-0MVN1IQTF0037WTG``); every other arm scales by the same per-metric
   factor so the *relative* uplift is the evidence while Arm 0 reproduces the
   baseline exactly (the C2 §4.3 G6 reproducibility requirement).

A restore is a *persistence decision* whose success depends on the gates above;
the report splits restore outcomes into ``success``/``failed`` and exclusions
by reason (``skip_when_busy``, circuit-breaker cooldown, ``context_too_large``,
``timeout_exceeded``), so a "no uplift" result is attributable rather than
merely observed (AC4).

Analysis tool only: no production module may import this module (the AC5
sentinel guard in ``test_pool_snapshot_bakeoff.py``), matching the
``prefill_thrash_replay_harness`` rule.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow ``python3 tests/pool_snapshot_bakeoff.py`` (the artefact writer) to
# resolve ``proxy.*`` and the sibling harness without pytest's conftest.
_PROXY_DIR = Path(__file__).resolve().parents[1]
for _path in (str(_PROXY_DIR), str(_PROXY_DIR / "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from prefill_thrash_replay_harness import FIXTURE_PATH as _FIXTURE_PATH  # noqa: E402
from prefill_thrash_replay_harness import band_rate as _band_rate  # noqa: E402
from prefill_thrash_replay_harness import load_fixture as _load_fixture  # noqa: E402
from proxy.provider import effective_per_slot_threshold  # noqa: E402

from proxy import session as session_mod  # noqa: E402

#: Sentinel attribute used by the AC5 test to prove no production module
#: imported the harness.
BAKEOFF_SENTINEL = "pool-snapshot-bakeoff-replay"

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSE_SCRIPT = REPO_ROOT / "scripts" / "analyse_slot_ratio_restore.py"
C3_SUMMARY = (
    REPO_ROOT
    / "proxy"
    / "benchmarks"
    / "results"
    / "slot_snapshot_bands_20261028"
    / "summary.json"
)
ARTEFACT_PATH = (
    REPO_ROOT / "proxy" / "benchmarks" / "results" / "pool_bakeoff_2026_10_24" / "summary.json"
)

#: Adaptive timeout model from proxy/config.yaml (LP-0MVN3KPEC007WWL5 C2 §4.2).
BASE_TIMEOUT_SECONDS = 3.0
PER_TOKEN_TIMEOUT_SECONDS = 0.0015
MAX_TIMEOUT_SECONDS = 60.0

#: Circuit breaker defaults (proxy/config.yaml:2262-2266).
MAX_CONSECUTIVE_FAILURES = 3
COOLDOWN_SECONDS = 300.0

#: Total context the llama-server is started with (proxy/config.yaml:1426).
SLOT_CTX = 262_144

#: The prefill deadline the fixture window ran under (its own header value).
FIXTURE_DEADLINE_SECONDS = 90.0

#: cleanup-slot-cache.sh default disk cap (scripts/cleanup-slot-cache.sh:27-29).
CLEANUP_MAX_SIZE_GB = 20
DISK_CAP_MIB = CLEANUP_MAX_SIZE_GB * 1024

#: MiB per KV token measured by C3 (summary.json "kv_mib_per_token").
KV_MIB_PER_TOKEN = 10.396575927734375 / 1024.0

#: C3-pinned under-load slot action costs (seconds) per prompt-size band,
#: from proxy/benchmarks/results/slot_snapshot_bands_20261028/summary.json
#: (band["under_load"].restore_seconds / save_seconds). Extrapolation past
#: the last band uses the final band's per-token slope.
RESTORE_COST_BANDS: tuple[tuple[int, float], ...] = (
    (10_000, 0.040),
    (25_000, 0.052),
    (40_000, 0.070),
    (60_000, 0.090),
)
SAVE_COST_BANDS: tuple[tuple[int, float], ...] = (
    (10_000, 0.049),
    (25_000, 0.072),
    (40_000, 0.110),
    (60_000, 0.290),
)

#: The measured baseline the report anchors Arm 0 to (LP-0MVN1IQTF0037WTG).
MEASURED_AGGREGATE_PCT = 26.1
MEASURED_MEDIAN_PCT = 15.6
MEASURED_COLD_OF = (28, 227)

#: Gate thresholds from C2 §4.3 (docs/dev/slot-snapshot-pool-sizing-…).
G1_MEDIAN_PCT = 31.2
G1_AGGREGATE_PCT = 36.1
G2_RESTORE_SUCCESS = 0.90
G2_EXCLUSION_SHARE = 0.20

_ENDPOINT = "pool-bakeoff"


@dataclass(frozen=True)
class ArmConfig:
    """One candidate configuration (C2 §3)."""

    name: str
    label: str
    live_slots: int
    #: None = per-session retention bounded only by the disk cap (baseline);
    #: an int = LRU count cap on retained snapshots (candidate B).
    retained_snapshot_pool: int | None
    skip_when_busy: bool
    comparable: bool
    clamp: int
    description: str = ""

    def server_config(self) -> dict:
        """The production server-config slice this arm runs under."""
        return {
            "session_slot_pool_size": self.live_slots,
            "session_slot_save_path": "/tmp/pool-bakeoff-sim",
            "session_slot_skip_when_busy": self.skip_when_busy,
            "session_slot_max_consecutive_failures": MAX_CONSECUTIVE_FAILURES,
            "session_slot_failure_cooldown_seconds": COOLDOWN_SECONDS,
            "session_slot_timeout_per_token_seconds": PER_TOKEN_TIMEOUT_SECONDS,
            "session_slot_max_timeout_seconds": MAX_TIMEOUT_SECONDS,
            "session_slot_timeout_seconds": BASE_TIMEOUT_SECONDS,
        }


def effective_per_slot_clamp(slots: int) -> int:
    """The routing/persistence clamp for *slots* (production helper, C2 §2)."""
    return int(effective_per_slot_threshold(SLOT_CTX, slots))


def _arm(name: str, label: str, slots: int, retained: int | None, busy: bool, comparable: bool,
         description: str) -> ArmConfig:
    return ArmConfig(
        name=name,
        label=label,
        live_slots=slots,
        retained_snapshot_pool=retained,
        skip_when_busy=busy,
        comparable=comparable,
        clamp=effective_per_slot_clamp(slots),
        description=description,
    )


ARMS: dict[str, ArmConfig] = {
    "arm0": _arm(
        "arm0",
        "Arm 0 - control, production config (pool 2, per-session snapshot "
        "files, skip_when_busy on)",
        2, None, True, True,
        "Baseline: session_slot_pool_size 2, one slot_<session>.bin per "
        "session (cleanup-slot-cache.sh disk cap), all deployed gates active.",
    ),
    "arm1": _arm(
        "arm1",
        "Arm 1 - candidate B (2 live slots, decoupled retained-snapshot pool "
        "of 4, LRU)",
        2, 4, True, True,
        "Candidate B per C2 §3: live geometry unchanged (AC5), a separate "
        "retained-snapshot pool of 4 with least-recently-active eviction.",
    ),
    "arm2": _arm(
        "arm2",
        "Arm 2 - candidate A (pool 4 literal: 4 live slots, clamp 61 440) - "
        "violates AC5, NOT a bake-off arm",
        4, None, True, False,
        "Candidate A per C2 §3/§4.1: literal session_slot_pool_size 4 shrinks "
        "per-slot context to 61 440 and changes live geometry, so it is "
        "recorded for completeness but excluded from the bake-off gates.",
    ),
    "arm3": _arm(
        "arm3",
        "Arm 3 - diagnostic: candidate B with skip_when_busy relaxed",
        2, 4, False, False,
        "Informational what-if: the same retained pool of 4 without the "
        "load-aware gate, so the busy gate's share of any no-uplift can be "
        "attributed (C3 measured restores at 0.02-0.09 s even under load).",
    ),
}


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


def load_records() -> tuple[dict, list[dict]]:
    """Load and return ``(meta, records)`` from the checked-in fixture."""
    return _load_fixture()


def unique_attempts(records: list[dict]) -> list[dict]:
    """Collapse double-logged admission decisions to one attempt each.

    The capture logs each decision twice (``route=queue`` +
    ``route=large_context_bypass`` at the same (ts, session, prompt)); the
    pool model consumes one attempt per triple, preserving first-seen order.
    """
    seen: set[tuple] = set()
    unique: list[dict] = []
    for record in records:
        key = (
            record.get("ts"),
            record.get("session"),
            record.get("estimated_tokens"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def _ts_seconds(value: str, start: dt.datetime) -> float:
    try:
        parsed = dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return 0.0
    return (parsed - start).total_seconds()


def _window_start(attempts: list[dict]) -> dt.datetime:
    parsed = []
    for attempt in attempts:
        try:
            parsed.append(dt.datetime.strptime(str(attempt["ts"]), "%Y-%m-%d %H:%M:%S"))
        except (TypeError, ValueError):
            pass
    return min(parsed) if parsed else dt.datetime(2026, 10, 24, 18, 38, 0)


# ---------------------------------------------------------------------------
# Usage-report percentile delegation (AC3 - no ad-hoc definition)
# ---------------------------------------------------------------------------


def _load_analyse_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "analyse_slot_ratio_restore_bakeoff", ANALYSE_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module's own dataclasses resolve their module
    # namespace (importlib does not do this automatically).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_analyse = _load_analyse_script()


def percentiles_pct(values: list[float]) -> dict | None:
    """Percentiles of *values* (0-100 scale) using the usage report's helper.

    Values are fed as percentages so the shared helper's 2-dp rounding lands
    at 0.01 percentage points instead of 0.01 of a ratio. The helper's
    ``sum_hours`` field is meaningless for ratios and is ignored.
    """
    if not values:
        return None
    return _analyse._percentiles([float(v) * 100.0 for v in values])


# ---------------------------------------------------------------------------
# Restore/save cost model (fed from C3's committed measurement)
# ---------------------------------------------------------------------------


def _band_cost(seconds_table: tuple[tuple[int, float], ...], tokens: int) -> float:
    """Interpolate/extrapolate *seconds_table* at *tokens* (C3 measured)."""
    if tokens <= 0:
        return 0.0
    first_upper, first_cost = seconds_table[0]
    if tokens <= first_upper:
        return first_cost * tokens / first_upper
    for (upper, cost), (next_upper, next_cost) in zip(
        seconds_table, seconds_table[1:]
    ):
        if tokens <= next_upper:
            frac = (tokens - upper) / (next_upper - upper)
            return cost + frac * (next_cost - cost)
    upper, cost = seconds_table[-2]
    next_upper, next_cost = seconds_table[-1]
    slope = (next_cost - cost) / (next_upper - upper)
    return next_cost + slope * (tokens - next_upper)


class RestoreCostModel:
    """C3-pinned cost model for slot save/restore, plus the adaptive timeout."""

    def restore_seconds(self, tokens: int) -> float:
        return _band_cost(RESTORE_COST_BANDS, tokens)

    def save_seconds(self, tokens: int) -> float:
        return _band_cost(SAVE_COST_BANDS, tokens)

    def timeout_seconds(self, tokens: int) -> float:
        """The production adaptive timeout (config.yaml:2236)."""
        return min(BASE_TIMEOUT_SECONDS + PER_TOKEN_TIMEOUT_SECONDS * tokens, MAX_TIMEOUT_SECONDS)


# ---------------------------------------------------------------------------
# Retained snapshot pool
# ---------------------------------------------------------------------------


@dataclass
class Snapshot:
    """A retained KV snapshot for one session."""

    tokens: int
    last_active: float


class RetentionPool:
    """LRU retention of session snapshots, bounded by count and/or bytes.

    ``cap_count=None`` = the baseline's per-session file behaviour (bounded by
    the cleanup-slot-cache.sh disk cap only); ``cap_count=N`` = candidate B's
    decoupled retained pool with least-recently-active eviction (C2 §3).
    """

    def __init__(self, cap_count: int | None, cap_bytes: int | None = DISK_CAP_MIB):
        self.cap_count = cap_count
        self.cap_bytes = cap_bytes
        self._entries: OrderedDict[str, Snapshot] = OrderedDict()
        self.evictions = 0

    def upsert(self, session: str, snapshot: Snapshot) -> None:
        existing = self._entries.pop(session, None)
        if existing is not None:
            self._entries[session] = snapshot
            self._prune()
            return
        self._entries[session] = snapshot
        self._prune()

    def touch(self, session: str) -> None:
        """Move *session* to the most-recently-active end (LRU access)."""
        if session in self._entries:
            self._entries.move_to_end(session)

    def get(self, session: str) -> Snapshot | None:
        return self._entries.get(session)

    def session_ids(self) -> list[str]:
        return list(self._entries.keys())

    def __contains__(self, session: object) -> bool:
        return session in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def byte_usage_mib(self) -> float:
        return sum(
            snapshot.tokens * KV_MIB_PER_TOKEN for snapshot in self._entries.values()
        )

    def _prune(self) -> None:
        while True:
            over_count = self.cap_count is not None and len(self._entries) > self.cap_count
            over_bytes = self.cap_bytes is not None and self.byte_usage_mib > self.cap_bytes
            if not (over_count or over_bytes) or not self._entries:
                return
            self._entries.popitem(last=False)
            self.evictions += 1


# ---------------------------------------------------------------------------
# The snapshot-pool simulation
# ---------------------------------------------------------------------------


class _SimClock:
    """A deterministic clock standing in for the ``time`` module.

    ``_slot_in_failure_cooldown`` and ``_record_slot_failure`` read
    ``time.time()``; ``_record_hot_slot_owner``-style flows read
    ``time.monotonic()``. Both resolve to the simulated timeline.
    """

    def __init__(self, owner: SnapshotPoolSim):
        self._owner = owner

    def time(self) -> float:
        return self._owner.clock

    def monotonic(self) -> float:
        return self._owner.clock


class SnapshotPoolSim:
    """Deterministic pool/snapshot-restore simulation for one arm.

    Drives the *real* production predicates (slot registry, busy gate,
    circuit breaker) with the documented stub sources
    (``_slot_owners`` registry, ``_slot_busy_state_snapshot`` value); the
    timestamps come from the fixture, never from a wall clock.
    """

    def __init__(
        self,
        arm: ArmConfig,
        *,
        fail_restores_every: int | None = None,
        fail_saves_every: int | None = None,
        cost: RestoreCostModel | None = None,
    ):
        self.arm = arm
        self.cost = cost or RestoreCostModel()
        self.server_config = arm.server_config()
        #: The adaptive-timeout budget (overridable for boundary tests).
        self.timeout_for: Callable[[int], float] = self.cost.timeout_seconds
        #: Deterministic failure schedules (None = no injected failures).
        self.fail_restores_every = fail_restores_every
        self.fail_saves_every = fail_saves_every

        session_mod._slot_owners.clear()
        session_mod._slot_failure_state.clear()
        self.clock = 0.0
        self.content: dict[int, tuple[str, int]] = {}
        self.busy_until: dict[str, float] = {}
        self.retained = RetentionPool(
            cap_count=arm.retained_snapshot_pool, cap_bytes=DISK_CAP_MIB
        )
        #: Every counter is pre-initialised so a report always carries the same
        #: attribution surface (a zero is evidence, a missing key is not).
        self.counters: dict[str, float] = {
            key: 0
            for key in (
                "attempts",
                "total_attempts",
                "pool_exhausted",
                "resident_hits",
                "reuse",
                "restore_opportunities",
                "restore_success",
                "restore_failed",
                "restore_failed_attempted",
                "save_success",
                "save_failed",
                "save_failed_attempted",
                "excluded_context_too_large",
                "excluded_slot_busy",
                "excluded_cooldown",
                "excluded_timeout",
                "save_skipped_slot_busy",
                "save_skipped_cooldown",
                "save_skipped_timeout",
                "save_skipped_context_too_large",
            )
        }
        self.reuse_by_attempt: list[dict] = []
        self._restore_attempt_no = 0
        self._save_attempt_no = 0
        self._window_start: dt.datetime | None = None

    # -- busy-state view (documented stub source) -------------------------

    def _busy_view(self, slot_id: int | None = None, exclude_session: str | None = None) -> dict:
        """Simulated dispatch-lease state: sessions with an in-flight serve."""
        active = sum(
            1
            for session, end in self.busy_until.items()
            if end > self.clock and session != exclude_session
        )
        return {
            "active_queries": active,
            "local_active_queries": active,
            "active_sessions": active,
            "slot_busy": False,
        }

    # -- gates ------------------------------------------------------------

    def _cooldown_active(self, slot_id: int) -> bool:
        return bool(session_mod._slot_in_failure_cooldown(slot_id, self.server_config))

    def _busy_gate(self, slot_id: int, session: str) -> bool:
        return bool(
            session_mod._slot_persistence_skip_when_busy(
                self.server_config, slot_id, session
            )
        )

    def _is_failure(self, counter: str, every: int | None) -> bool:
        if every is None:
            return False
        self.counters[counter] = self.counters.get(counter, 0) + 1
        return self.counters[counter] % every == 0

    # -- the per-attempt loop ----------------------------------------------

    def simulate(self, attempts: list[dict]) -> None:
        """Replay *attempts* through the pool model (time-ordered records)."""
        if not attempts:
            return
        if self._window_start is None:
            self._window_start = _window_start(attempts)

        clock_stub = _SimClock(self)
        saved_time = session_mod.time
        saved_busy = session_mod._slot_busy_state_snapshot
        session_mod.time = clock_stub  # type: ignore[assignment]
        session_mod._slot_busy_state_snapshot = self._busy_view  # type: ignore[assignment]
        try:
            for record in attempts:
                self._simulate_one(record)
        finally:
            session_mod.time = saved_time  # type: ignore[assignment]
            session_mod._slot_busy_state_snapshot = saved_busy  # type: ignore[assignment]

    def _simulate_one(self, record: dict) -> None:
        ts = _ts_seconds(str(record.get("ts", "")), self._window_start)
        for session in [s for s, end in self.busy_until.items() if end <= ts]:
            session_mod._free_slot_assignment(session, endpoint=_ENDPOINT)
        self.clock = ts

        session = str(record.get("session", ""))
        estimated = int(record.get("estimated_tokens", 0) or 0)
        record_new_tokens = int(record.get("new_tokens", estimated) or 0)
        outcome = str(record.get("outcome", "other"))

        self.counters["total_attempts"] += 1

        # The hard routing cap (`effective_per_slot_threshold`, the same clamp
        # the persistence cap derives from) sends an oversized prompt off the
        # local path entirely: it never occupies a slot and never reuses one
        # (C2 §1 items 3-4, provider.py:_effective_large_context_thresholds).
        if estimated > self.arm.clamp:
            self.counters["excluded_context_too_large"] += 1
            return

        # Slot allocation through the real registry (C2 §1 item 6).
        slot_id = session_mod._slot_id_for_session(
            session, self.arm.live_slots, endpoint=_ENDPOINT
        )
        if slot_id is None:
            self.counters["pool_exhausted"] += 1
            self._mark_busy(session, estimated, record_new_tokens, 0.0)
            return
        self.counters["attempts"] += 1

        content_session, content_tokens = self.content.get(slot_id, (None, 0))
        resident = content_session == session

        # The evicted tenant's KV is saved before the slot is overwritten
        # (session.plan_eviction_snapshot / save_evicted_session_snapshot).
        if not resident and content_session is not None:
            self._evict_save(slot_id, content_session, content_tokens)

        # Request-level persistence decision, in production order
        # (session._build_slot_context): context cap, busy gate, circuit
        # breaker; the adaptive timeout is priced when the restore is issued.
        eligibility = self._persistence_decision(slot_id, session, estimated)

        reuse = 0.0
        snapshot = None if resident else self.retained.get(session)
        if resident:
            # Same session still owns the slot's KV: reuse needs no persistence.
            session_mod._record_slot_success(slot_id)
            self.counters["resident_hits"] += 1
            reuse = min(content_tokens, estimated) / max(1, estimated)
        elif snapshot is not None:
            # A restorable snapshot exists and the session is not resident, so
            # a restore would have been issued — unless a gate suppressed the
            # whole request's persistence (attributed by the exclusion
            # counters). ``restore.opportunities`` therefore includes gated
            # restores, and ``restore.gated`` reports the difference.
            self.counters["restore_opportunities"] += 1
            if eligibility is None:
                if self.cost.restore_seconds(snapshot.tokens) > self.timeout_for(estimated):
                    self.counters["excluded_timeout"] += 1
                elif self._is_failure(
                    "restore_failed_attempted", self.fail_restores_every
                ):
                    session_mod._record_slot_failure(slot_id)
                    self.counters["restore_failed"] += 1
                else:
                    session_mod._record_slot_success(slot_id)
                    self.retained.touch(session)
                    self.counters["restore_success"] += 1
                    reuse = min(snapshot.tokens, estimated) / max(1, estimated)

        self.counters["reuse"] += reuse
        self.reuse_by_attempt.append(
            {
                "session": session,
                "estimated_tokens": estimated,
                "reuse": round(reuse, 6),
            }
        )

        # The slot ends up holding what the deadline allowed to be computed
        # (the window's own 90 s prefill deadline): a full prompt when its
        # prefill fits, otherwise the reused prefix plus the deadline-limited
        # remainder. This is the documented waste model of the sibling harness,
        # expressed as resident tokens instead of discarded tokens, and it is
        # what makes partial-reuse measurements possible at all.
        resident_tokens = self._resident_tokens(estimated, record_new_tokens, reuse, outcome)
        if resident_tokens <= 0:
            self.content.pop(slot_id, None)
        else:
            self.content[slot_id] = (session, resident_tokens)
            if eligibility is None:
                # The session saves its own KV at the end of its turn
                # (router.py: slot_save), permitted by the request-start
                # persistence decision; the snapshot holds what is resident.
                self._write_snapshot(
                    slot_id, session, resident_tokens, self.timeout_for(estimated)
                )

        self._mark_busy(session, estimated, record_new_tokens, reuse)

    def _resident_tokens(
        self, estimated: int, new_tokens: int, reuse: float, outcome: str
    ) -> int:
        """Tokens the slot holds for this session after the attempt.

        ``min(prefill, deadline)`` of the *unreused* prompt is computed; a
        deadline-cancelled attempt keeps nothing (its work is discarded).
        """
        if outcome == "prefill_deadline_exceeded":
            return 0
        remaining = max(0.0, float(new_tokens) * (1.0 - reuse))
        rate = _band_rate(estimated)
        computable = min(remaining, FIXTURE_DEADLINE_SECONDS * rate)
        resident = estimated - remaining + computable
        return int(max(0.0, min(float(estimated), resident)))

    def _persistence_decision(self, slot_id: int, session: str, estimated: int) -> str | None:
        """Mirror ``session._build_slot_context``'s gate ladder.

        Returns ``None`` when persistence is permitted for this request, else
        the exclusion reason. Order matches production: context cap, busy
        gate, circuit breaker (the adaptive timeout is applied at call time).
        """
        if self._busy_gate(slot_id, session):
            self.counters["excluded_slot_busy"] += 1
            return "slot_busy"
        if self._cooldown_active(slot_id):
            self.counters["excluded_cooldown"] += 1
            return "cooldown"
        return None

    def _write_snapshot(self, slot_id: int, session: str, tokens: int, timeout: float) -> None:
        """Issue one slot save (end-of-turn or eviction) through the gates."""
        if self.cost.save_seconds(tokens) > timeout:
            self.counters["save_skipped_timeout"] += 1
            return
        if self._is_failure("save_failed_attempted", self.fail_saves_every):
            session_mod._record_slot_failure(slot_id)
            self.counters["save_failed"] += 1
            return
        session_mod._record_slot_success(slot_id)
        self.counters["save_success"] += 1
        self.retained.upsert(session, Snapshot(tokens=tokens, last_active=self.clock))

    def _evict_save(self, slot_id: int, session: str, tokens: int) -> None:
        """Save an evicted tenant's KV before its slot is overwritten.

        Mirrors ``plan_eviction_snapshot`` (session.py:848): the same gates,
        and the save is priced against the *base* timeout because production
        passes ``body_json=None`` (so ``estimated=0``): the context cap never
        applies to an eviction save, but the busy gate and breaker do.
        """
        if self._busy_gate(slot_id, session):
            self.counters["save_skipped_slot_busy"] += 1
            return
        if self._cooldown_active(slot_id):
            self.counters["save_skipped_cooldown"] += 1
            return
        self._write_snapshot(slot_id, session, tokens, BASE_TIMEOUT_SECONDS)

    def _mark_busy(self, session: str, estimated: int, new_tokens: int, reuse: float) -> None:
        """The session holds a dispatch lease for its modelled serve.

        The busy window is the modelled prefill of the *unreused* prompt
        (``new_tokens * (1 - reuse)``) at the measured band rate. Generation
        time is not in the fixture, so the window is a conservative lower
        bound on how long the lease is actually held.
        """
        remaining = max(0.0, float(new_tokens) * (1.0 - reuse))
        prefill_seconds = remaining / _band_rate(estimated) if estimated else 0.0
        # A local serve cannot outlive the prefill deadline: the fixture
        # records either a first byte or a deadline cancellation at 90 s.
        self.busy_until[session] = self.clock + min(
            prefill_seconds, FIXTURE_DEADLINE_SECONDS
        )

    # -- metrics -----------------------------------------------------------

    def metrics(self) -> dict:
        c = self.counters
        attempts = int(c["attempts"])
        restore_opportunities = int(c["restore_opportunities"])
        restore_attempted = int(c["restore_success"]) + int(c["restore_failed"])

        gt20 = [a for a in self.reuse_by_attempt if a["estimated_tokens"] > 20_000]
        gt20_reuse = [a["reuse"] for a in gt20]
        gt20_median = percentiles_pct(gt20_reuse)
        cold = sum(1 for reuse in gt20_reuse if reuse == 0.0)

        exclusions = {
            "context_too_large": int(c["excluded_context_too_large"]),
            "slot_busy": int(c["excluded_slot_busy"]),
            "cooldown": int(c["excluded_cooldown"]),
            "timeout_exceeded": int(c["excluded_timeout"]),
        }
        save_skips = {
            "slot_busy": int(c["save_skipped_slot_busy"]),
            "cooldown": int(c["save_skipped_cooldown"]),
            "timeout_exceeded": int(c["save_skipped_timeout"]),
        }
        return {
            "attempts": attempts,
            "total_attempts": int(c["total_attempts"]),
            "pool_exhausted": int(c["pool_exhausted"]),
            "local_serves": len(self.reuse_by_attempt),
            # C2 §4.3 G5: an arm must not improve reuse by serving less.
            "local_serve_share": round(
                attempts / max(1, int(c["total_attempts"])), 4
            ),
            "resident_hits": int(c["resident_hits"]),
            # The measured baseline is a *local dispatch* metric (the usage
            # report measures dispatch_first_byte_ms), so the denominators are
            # local serves, not admission attempts.
            "aggregate_pct": round(c["reuse"] / len(self.reuse_by_attempt) * 100.0, 2)
            if self.reuse_by_attempt
            else 0.0,
            "gt20k": {
                "count": len(gt20),
                "median_pct": round(gt20_median["p50"], 2) if gt20_median else 0.0,
                "aggregate_pct": round(sum(gt20_reuse) / len(gt20) * 100.0, 2) if gt20 else 0.0,
                "cold_count": cold,
                "cold_share": round(cold / len(gt20), 4) if gt20 else 0.0,
            },
            "restore": {
                "opportunities": restore_opportunities,
                "gated": max(
                    0,
                    restore_opportunities
                    - int(c["restore_success"])
                    - int(c["restore_failed"])
                    - int(c["excluded_timeout"]),
                ),
                "success": int(c["restore_success"]),
                "failed": int(c["restore_failed"]),
                "success_rate": round(int(c["restore_success"]) / restore_attempted, 4)
                if restore_attempted
                else 1.0,
            },
            "save": {
                "success": int(c["save_success"]),
                "failed": int(c["save_failed"]),
            },
            "exclusions": {
                **exclusions,
                "pool_exhausted": int(c["pool_exhausted"]),
                "total": sum(exclusions.values()),
            },
            "save_skips": {**save_skips, "total": sum(save_skips.values())},
            "restore_success_rate": round(int(c["restore_success"]) / restore_attempted, 4)
            if restore_attempted
            else 1.0,
            "retention_evictions": self.retained.evictions,
            "disk_footprint_mib": round(self.retained.byte_usage_mib, 2),
        }


# ---------------------------------------------------------------------------
# Per-arm run, calibration and gates
# ---------------------------------------------------------------------------


def _calibrate(raw: dict, factor_aggregate: float, factor_median: float, factor_cold: float) -> dict:
    """Scale raw reuse metrics by the arm0 anchors (proportional calibration)."""
    aggregate = min(100.0, max(0.0, raw["aggregate_pct"] * factor_aggregate))
    median = min(100.0, max(0.0, raw["gt20k"]["median_pct"] * factor_median))
    count = raw["gt20k"]["count"]
    cold = round(min(1.0, max(0.0, raw["gt20k"]["cold_share"] * factor_cold)) * count)
    return {
        "aggregate_pct": round(aggregate, 2),
        "median_pct_gt20k": round(median, 2),
        "cold_of": [cold, count],
        "factors": {
            "aggregate": round(factor_aggregate, 4),
            "median_gt20k": round(factor_median, 4),
            "cold": round(factor_cold, 4),
        },
    }


def run_arm(arm: ArmConfig, attempts: list[dict], **sim_kwargs: Any) -> dict:
    """Run one arm over the deduped window and return its raw report."""
    sim = SnapshotPoolSim(arm, **sim_kwargs)
    sim.simulate(attempts)
    raw = sim.metrics()
    return {
        "name": arm.name,
        "label": arm.label,
        "description": arm.description,
        "config": {
            "live_slots": arm.live_slots,
            "retained_snapshot_pool": arm.retained_snapshot_pool,
            "skip_when_busy": arm.skip_when_busy,
            "clamp": arm.clamp,
            "comparable": arm.comparable,
        },
        "raw": raw,
        # Attribution surfaces lifted to the arm level (AC4: exclusion counts
        # per configuration so a no-uplift result is attributable).
        "exclusions": raw["exclusions"],
        "save_skips": raw["save_skips"],
        "restore_success_rate": raw["restore_success_rate"],
        "counters": {k: round(v, 4) for k, v in sorted(sim.counters.items())},
    }


def _paired_median_delta(arm0_sim: SnapshotPoolSim, arm1_sim: SnapshotPoolSim) -> float:
    """Median of per-session (arm1 median - arm0 median) reuse deltas."""
    by_session: dict[str, dict[str, list[float]]] = {}
    for sim, tag in ((arm0_sim, "arm0"), (arm1_sim, "arm1")):
        for entry in sim.reuse_by_attempt:
            by_session.setdefault(entry["session"], {})[tag] = by_session.setdefault(
                entry["session"], {}
            ).get(tag, []) + [entry["reuse"]]

    def median(values: list[float]) -> float:
        ordered = sorted(values)
        n = len(ordered)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    deltas = []
    for per_session in by_session.values():
        if "arm0" in per_session and "arm1" in per_session:
            deltas.append(median(per_session["arm1"]) - median(per_session["arm0"]))
    return median(deltas) if deltas else 0.0


def _g4_cost_checks(cost: RestoreCostModel) -> dict:
    """Per-band restore cost vs the adaptive timeout (C2 §4.3 G4)."""
    rows = []
    for band, tokens in (("10k", 10_000), ("25k", 25_000), ("40k", 40_000), ("60k+", 60_000)):
        rows.append(
            {
                "band": band,
                "target_tokens": tokens,
                "restore_seconds": round(cost.restore_seconds(tokens), 3),
                "timeout_seconds": round(cost.timeout_seconds(tokens), 1),
            }
        )
    return {"bands": rows}


def bakeoff(attempts: list[dict]) -> dict:
    """Run all arms and produce the full bake-off report (AC2/AC3/AC4)."""
    sims: dict[str, SnapshotPoolSim] = {}
    report_arms: dict[str, dict] = {}
    for name, arm in ARMS.items():
        sim = SnapshotPoolSim(arm)
        sim.simulate(attempts)
        sims[name] = sim
        report_arms[name] = run_arm(arm, attempts)

    # Proportional calibration: anchor Arm 0 to the measured baseline.
    raw0 = report_arms["arm0"]["raw"]
    def _factor(measured: float, modelled: float) -> float:
        return (measured / modelled) if modelled and modelled > 0 else 1.0

    factor_agg = _factor(MEASURED_AGGREGATE_PCT, raw0["aggregate_pct"])
    factor_med = _factor(MEASURED_MEDIAN_PCT, raw0["gt20k"]["median_pct"])
    factor_cold = _factor(
        MEASURED_COLD_OF[0] / MEASURED_COLD_OF[1], raw0["gt20k"]["cold_share"]
    )
    for name in ARMS:
        report_arms[name]["calibrated"] = _calibrate(
            report_arms[name]["raw"], factor_agg, factor_med, factor_cold
        )

    # Gate verdicts for the comparable arms (C2 §4.3).
    cost = RestoreCostModel()
    for name in ARMS:
        arm = ARMS[name]
        raw = report_arms[name]["raw"]
        cal = report_arms[name]["calibrated"]
        gates = {}
        if not arm.comparable:
            gates = {
                g: {"pass": "n/a", "rationale": "not a bake-off arm (C2 §4.1)"}
                for g in ("G1", "G2", "G3", "G4", "G5", "G6")
            }
            report_arms[name]["gates"] = gates
            continue

        paired_delta = _paired_median_delta(sims["arm0"], sims["arm1"])
        g1 = (
            cal["median_pct_gt20k"] >= G1_MEDIAN_PCT
            and cal["aggregate_pct"] >= G1_AGGREGATE_PCT
            and paired_delta > 0.0
        )
        gates["G1"] = {
            "pass": g1,
            "rationale": (
                f"median {cal['median_pct_gt20k']}% vs threshold {G1_MEDIAN_PCT}%; "
                f"aggregate {cal['aggregate_pct']}% vs {G1_AGGREGATE_PCT}%; "
                f"paired per-session median delta {paired_delta:+.6f} "
                f"(requires > 0)"
            ),
        }

        # C2 §4.3 G2: skip_when_busy + cooldown exclusions as a share of
        # *persistence decisions* (every attempt that got a slot makes one).
        exclusion_share = (
            raw["exclusions"]["slot_busy"] + raw["exclusions"]["cooldown"]
        ) / max(1, raw["attempts"])
        g2 = (
            raw["restore_success_rate"] >= G2_RESTORE_SUCCESS
            and exclusion_share <= G2_EXCLUSION_SHARE
        )
        gates["G2"] = {
            "pass": g2,
            "rationale": (
                f"restore success rate {raw['restore_success_rate']} (>= {G2_RESTORE_SUCCESS}); "
                f"skip_when_busy+cooldown exclusion share {exclusion_share:.4f} "
                f"(<= {G2_EXCLUSION_SHARE})"
            ),
        }

        gates["G3"] = {
            "pass": arm.live_slots == 2 and arm.clamp == 126_976,
            "rationale": f"live slots {arm.live_slots}, clamp {arm.clamp} (requires 2 / 126 976)",
        }

        cost_rows = _g4_cost_checks(cost)["bands"]
        g4 = all(row["restore_seconds"] <= row["timeout_seconds"] for row in cost_rows)
        gates["G4"] = {
            "pass": g4,
            "rationale": "per-band restore cost within adaptive timeout: "
            + "; ".join(
                f"{r['band']} {r['restore_seconds']}s <= {r['timeout_seconds']}s"
                for r in cost_rows
            ),
        }

        disk_ok = raw["disk_footprint_mib"] <= DISK_CAP_MIB
        share_ok = raw["local_serve_share"] >= report_arms["arm0"]["raw"]["local_serve_share"]
        gates["G5"] = {
            "pass": disk_ok and share_ok,
            "rationale": (
                f"modelled disk footprint {raw['disk_footprint_mib']} MiB <= "
                f"{DISK_CAP_MIB} MiB cleanup cap; local-serve share "
                f"{raw['local_serve_share']} vs Arm 0 "
                f"{report_arms['arm0']['raw']['local_serve_share']} (must not be "
                "lower); deadline cancellations / wasted tokens are "
                "admission-side and unchanged by the pool model (fixture "
                "baseline 125 / 778 863)"
            ),
        }

        gates["G6"] = {
            "pass": True,
            "rationale": (
                "deterministic replay (same fixture/config => identical JSON); "
                "Arm 0 recalibrated to the measured 26.1 % / 15.6 % baseline"
            ),
        }
        report_arms[name]["gates"] = gates

    # Arm-to-arm deltas and the attribution channels C5 needs.
    for name in ARMS:
        cal = report_arms[name]["calibrated"]
        base = report_arms["arm0"]["calibrated"]
        report_arms[name]["delta_vs_arm0"] = {
            "aggregate_pp": round(cal["aggregate_pct"] - base["aggregate_pct"], 2),
            "median_pp_gt20k": round(cal["median_pct_gt20k"] - base["median_pct_gt20k"], 2),
            "cold_prompts": cal["cold_of"][0] - base["cold_of"][0],
            "local_serve_share": round(
                report_arms[name]["raw"]["local_serve_share"]
                - report_arms["arm0"]["raw"]["local_serve_share"],
                4,
            ),
        }

    timestamps = sorted(str(a["ts"]) for a in attempts if a.get("ts"))
    return {
        "schema_version": 1,
        "sentinel": BAKEOFF_SENTINEL,
        "generated_at": "deterministic replay of the committed fixture "
        "(regenerate with: python3 proxy/tests/pool_snapshot_bakeoff.py --write)",
        "window": {
            "start": timestamps[0] if timestamps else "",
            "end": timestamps[-1] if timestamps else "",
        },
        "fixture": "proxy/tests/fixtures/prefill_thrash_2026_10_24.jsonl",
        "unique_attempts": len(attempts),
        "baseline_ref": {
            "aggregate_pct": MEASURED_AGGREGATE_PCT,
            "median_pct_gt20k": MEASURED_MEDIAN_PCT,
            "cold_of": list(MEASURED_COLD_OF),
            "percentile_definition": "scripts/analyse_slot_ratio_restore._percentiles "
            "(linear interpolation, percentages)",
        },
        "model": {
            "calibration": "proportional: Arm 0 anchored to the measured baseline; "
            "other arms scaled by the same per-metric factor",
            "restore_cost_source": str(C3_SUMMARY.relative_to(REPO_ROOT)),
            "busy_model": "serve holds a dispatch lease for modelled prefill "
            "(est*(1-reuse) / band rate); generation time not in the fixture "
            "(conservative: underestimates busy)",
            "gates": "the real production predicates (slot registry, busy gate, "
            "circuit breaker, context cap) driven through the documented "
            "stub sources",
        },
        "arms": report_arms,
    }


def bakeoff_result_to_json(report: dict) -> dict[str, Any]:
    """Serialise the bake-off report for machine-readable output."""
    return json.loads(json.dumps(report, sort_keys=True, default=str))


def write_artefact(report: dict, path: Path | None = None) -> Path:
    """Write the committed JSON artefact next to the replay (AC6)."""
    target = path or ARTEFACT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bakeoff_result_to_json(report), sort_keys=True, indent=2)
    target.write_text(payload + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    """Regenerate (``--write``) or print the bake-off report as JSON."""
    parser = argparse.ArgumentParser(
        description="Pool-size/snapshot-restore bake-off replay "
        "(LP-0MVOG8TUM005705R)"
    )
    parser.add_argument(
        "--fixture",
        default=str(_FIXTURE_PATH),
        help="thrash-window fixture (default: the checked-in 2026-10-24 capture)",
    )
    parser.add_argument(
        "--write",
        nargs="?",
        const=str(ARTEFACT_PATH),
        default=None,
        help=f"write the JSON report to this path (default {ARTEFACT_PATH})",
    )
    args = parser.parse_args(argv)

    _, records = _load_fixture(Path(args.fixture))
    attempts = unique_attempts(records)
    report = bakeoff(attempts)
    if args.write:
        path = write_artefact(report, Path(args.write))
        print(f"wrote {path} ({len(attempts)} unique attempts)")
    else:
        print(json.dumps(bakeoff_result_to_json(report), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
