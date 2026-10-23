"""Fast/cheap session bucketing derived from the proxy configuration.

Three attribution sources feed the fast/cheap split, in precedence order
(LP-0MVKSXXIR006OBI2):

1. **Mode timeline** (historical) — ``Mode scheduler: applied scheduled mode
   <mode>`` / ``Grandfathering: ... (current=<mode>)`` log lines reconstruct
   the operating mode active at any timestamp, and that mode names the bucket
   (:class:`ModeScheduleMap`). Retired in the unified config, but archived
   windows still carry the evidence, so it always wins when present.
2. **Priority window** — the unified ``proxy/config.yaml`` routes cheap
   semantics through ``server.priority_window`` (local time, midnight
   wrapping): every request inside the window is capped to ``low`` priority.
   When the logs show no mode transitions, the window's time-of-day bounds
   are the fast/cheap schedule (:class:`PriorityWindow`).
3. **Slot schedule** — the legacy fallback: the config's ``slot_schedule``
   entries, labelled by slot count (fewest slots = "fast", most = "cheap").

Terminology note (LP-0MSLMYEEU002IBH6): the operating modes are **fast**
(cloud-backed; the old "day" period) and **cheap** (local, low priority; the
old "night" period; the 1-slot local pool, same models as fast,
LP-0MSMIPPJI007GU9N). The slot-count → label mapping of the legacy path is
unchanged: the period(s) with the fewest slots are labelled "fast", the
period(s) with the most "cheap"; equal counts collapse to a single "fast"
bucket.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass(order=True)
class ModeTransition:
    """A single operating-mode transition with optional attribution.

    ``source``, ``actor``, and ``reason`` are populated when the log line
    carries attribution markers (LP-0MVC0S6U5003SEDJ) and stay ``None``
    for historical unattributed lines.
    """

    ts: datetime
    mode: str = field(compare=False)  # "fast" | "cheap"
    source: str | None = field(default=None, compare=False)
    actor: str | None = field(default=None, compare=False)
    reason: str | None = field(default=None, compare=False)

MINUTES_PER_DAY = 24 * 60


def minute_of_day(dt: datetime) -> float:
    """Fractional minutes from midnight (seconds contribute a fraction)."""
    return dt.hour * 60 + dt.minute + dt.second / 60.0


def parse_window_minutes(value: object, default: float) -> float:
    """Parse a ``"HH:MM"`` clock bound to minutes from midnight.

    Local time, matching ``proxy/proxy/priority.py::in_cheap_window`` (no UTC
    conversion). Invalid/missing values fall back to ``default``.
    """
    try:
        hh, mm = str(value).strip().split(":")[:2]
        hours, minutes = int(hh), int(mm)
    except (ValueError, AttributeError, TypeError):
        return float(default)
    if not (0 <= hours < 24 and 0 <= minutes < 60):
        return float(default)
    return float(hours * 60 + minutes)


PRIORITY_WINDOW_SOURCE = "server.priority_window"


@dataclass(frozen=True)
class PriorityWindow:
    """The proxy's ``server.priority_window`` cheap window (local time).

    Inside the window the unified config routes every request as ``low``
    priority (LP-0MVDT94F0003VHUA AC1); outside it requests keep their own
    priority. For historical bucketing that makes the window the fast/cheap
    schedule: inside = cheap, outside = fast. ``start_minutes`` is inclusive,
    ``end_minutes`` exclusive, and ``end <= start`` wraps past midnight.
    """

    start_minutes: float
    end_minutes: float
    source: str = PRIORITY_WINDOW_SOURCE

    @property
    def wraps(self) -> bool:
        return self.end_minutes <= self.start_minutes

    def contains(self, minutes: float) -> bool:
        if self.start_minutes == self.end_minutes:
            return False  # zero-length window is never cheap
        if self.wraps:
            return minutes >= self.start_minutes or minutes < self.end_minutes
        return self.start_minutes <= minutes < self.end_minutes

    def segments(self) -> list[tuple[float, float, str]]:
        """``(start, end, label)`` segments covering the whole day.

        A midnight-wrapping window is split into two cheap segments plus the
        intervening fast segment so every returned segment is non-wrapping
        (``SlotPeriod.contains`` then never needs the wrap branch).
        """
        start, end = self.start_minutes, self.end_minutes
        if start == end:
            return [(0.0, float(MINUTES_PER_DAY), "fast")]
        if not self.wraps:
            out: list[tuple[float, float, str]] = []
            if start > 0:
                out.append((0.0, start, "fast"))
            out.append((start, end, "cheap"))
            if end < MINUTES_PER_DAY:
                out.append((end, float(MINUTES_PER_DAY), "fast"))
            return out
        out = []
        if end > 0:
            out.append((0.0, end, "cheap"))
        out.append((end, start, "fast"))
        if start < MINUTES_PER_DAY:
            out.append((start, float(MINUTES_PER_DAY), "cheap"))
        return out

    @classmethod
    def from_config(cls, config: dict | None) -> PriorityWindow | None:
        """Build from a parsed config's ``priority_window`` block.

        Accepts both the canonical normalised form produced by
        ``config_loader`` (``start_minutes``/``end_minutes``) and a raw
        ``{"enabled": bool, "start": "HH:MM", "end": "HH:MM"}`` block.
        Returns ``None`` when the window is absent or disabled, so the caller
        falls through to the legacy slot-schedule attribution (AC4).
        """
        if not isinstance(config, dict):
            return None
        window = config.get("priority_window")
        if not isinstance(window, dict):
            return None
        enabled = window.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in ("true", "yes", "1")
        if not enabled:
            return None
        start = window.get("start_minutes")
        end = window.get("end_minutes")
        start_m = float(start) if start is not None else parse_window_minutes(window.get("start"), 60)
        end_m = float(end) if end is not None else parse_window_minutes(window.get("end"), 600)
        return cls(start_minutes=start_m, end_minutes=end_m)


@dataclass
class SlotPeriod:
    """A contiguous window of the day with a constant slot count.

    ``start_minutes`` is inclusive, ``end_minutes`` exclusive; when
    ``end_minutes <= start_minutes`` the period wraps past midnight.
    ``ctx_size`` is the per-period context window (from the schedule entry's
    ``ctx_size``, LP-0MSMZOAJW002UR2A); ``None`` when the entry did not
    pin one (the profile's global ``local_model_ctx_size`` then applies).
    """

    start_minutes: float
    end_minutes: float
    slots: int
    label: str  # "fast" | "cheap"
    ctx_size: int | None = None

    def contains(self, minutes: float) -> bool:
        if self.end_minutes > self.start_minutes:
            return self.start_minutes <= minutes < self.end_minutes
        # Wraps past midnight.
        return minutes >= self.start_minutes or minutes < self.end_minutes


@dataclass
class SlotSchedule:
    periods: list[SlotPeriod]
    fast_slots: int | None
    cheap_slots: int | None
    source: str  # "config" | "default"

    def period_for(self, dt: datetime) -> SlotPeriod:
        minutes = minute_of_day(dt)
        for p in self.periods:
            if p.contains(minutes):
                return p
        return self.periods[-1]


def schedule_from_entries(entries: Sequence[tuple[str, int] | tuple[str, int, int | None]]) -> SlotSchedule:
    """Build a schedule from ``(HH:MM, slots[, ctx_size])`` transition entries.

    Each entry's slot count applies from its time until the next entry
    (midnight wrapping: the last entry's count applies from 00:00 until the
    first entry). Periods are labelled by slot count: the period(s) with the
    minimum count are "fast", all others "cheap". If every period has the
    same count there is a single "fast" bucket.
    """
    parsed = []
    for item in entries:
        time_str, slots = item[0], int(item[1])
        ctx_size = int(item[2]) if len(item) > 2 and item[2] is not None else None
        hh, mm = time_str.split(":")
        parsed.append((int(hh) * 60 + int(mm), int(slots), ctx_size))
    parsed.sort(key=lambda t: t[0])

    if not parsed:
        return SlotSchedule(periods=[], fast_slots=None, cheap_slots=None, source="default")

    boundaries = [t[0] for t in parsed]
    counts = [t[1] for t in parsed]
    ctxs = [t[2] for t in parsed]

    min_count = min(counts)
    max_count = max(counts)

    periods: list[SlotPeriod] = []
    # Count active before the first boundary is the last entry's (midnight wrap).
    prev_start = 0.0
    prev_slots = counts[-1]
    prev_ctx = ctxs[-1]
    for boundary, slots, ctx in zip(boundaries, counts, ctxs):
        if boundary > prev_start:
            periods.append(
                SlotPeriod(prev_start, float(boundary), prev_slots, _label(prev_slots, min_count, max_count), ctx_size=prev_ctx)
            )
        prev_start = float(boundary)
        prev_slots = slots
        prev_ctx = ctx
    if prev_start < MINUTES_PER_DAY:
        periods.append(
            SlotPeriod(prev_start, float(MINUTES_PER_DAY), prev_slots, _label(prev_slots, min_count, max_count), ctx_size=prev_ctx)
        )

    fast_slots = min_count if min_count != max_count else min_count
    cheap_slots = max_count if min_count != max_count else None
    return SlotSchedule(periods=periods, fast_slots=fast_slots, cheap_slots=cheap_slots, source="config")


def _label(slots: int, min_count: int, max_count: int) -> str:
    if max_count == min_count:
        return "fast"
    return "fast" if slots == min_count else "cheap"


def schedule_from_config(config: dict | None, default_slots: int | None) -> SlotSchedule:
    """Derive the schedule from a parsed proxy config dict.

    If ``slot_schedule.enabled`` is false or entries are missing, fall back to
    a single "fast" bucket using ``default_slots`` (typically
    ``session_slot_pool_size``).
    """
    entries: list[tuple[str, int]] = []
    ctx_by_time: dict[str, int] = {}
    if config:
        schedule = config.get("slot_schedule") or {}
        enabled = schedule.get("enabled", True)
        if enabled:
            for e in schedule.get("entries") or []:
                if isinstance(e, dict):
                    entries.append((str(e.get("time")), int(e.get("slots", 0))))
                elif isinstance(e, (tuple, list)) and len(e) == 2:
                    entries.append((str(e[0]), int(e[1])))
            ctx_by_time = schedule.get("ctx_by_time") or {}
    if not entries:
        slots = default_slots if default_slots is not None else 0
        return SlotSchedule(
            periods=[SlotPeriod(0.0, float(MINUTES_PER_DAY), slots, "fast")],
            fast_slots=slots if slots else None,
            cheap_slots=None,
            source="default",
        )
    return schedule_from_entries(
        [(t, s, ctx_by_time.get(t)) for t, s in entries]
    )


def bucket_for_time(schedule: SlotSchedule, dt: datetime) -> SlotPeriod:
    """Return the fast/cheap period containing ``dt``."""
    return schedule.period_for(dt)


@dataclass
class ModeScheduleMap:
    """Fast/cheap schedules per operating mode plus the mode timeline.

    The proxy runs one of two operating modes (fast / cheap, each with its
    own config profile: ``config-fast.yaml`` / ``config-cheap.yaml``), each
    with its own ``slot_schedule`` (LP-0MSM5K4TX004MICX). The mode active at
    any timestamp is reconstructed from ``Mode scheduler: applied scheduled
    mode <mode>`` transitions (scheduled) plus the ``Grandfathering:
    enabled; ... (current=<mode>)`` marker (manual ``POST /admin/set-mode``
    switches, LP-0MT1EE315007AKXG) parsed out of the logs
    (LP-0MSPZUD4G007IYGH) — so sessions that ran during cheap hours are
    bucketed with the cheap profile even when the analysis itself runs in
    fast mode (the old behaviour bucketed every session by the
    analysis-time config profile).

    Sessions are bucketed by the mode active at their start: the bucket
    label is the mode name (``fast``/``cheap``), and slots/ctx come from that
    mode's config profile (e.g. 2 slots / 262144 ctx for cheap, 3 slots /
    131072 for fast).
    """

    schedules: dict[str, SlotSchedule]
    ctx_sizes: dict[str, int | None]
    default_mode: str
    transitions: list[ModeTransition] = field(default_factory=list)
    # Cheap window from the unified config (LP-0MVKSXXIR006OBI2). ``None``
    # when the priority window is absent or disabled, which keeps the legacy
    # slot-schedule attribution for historical windows.
    priority_window: PriorityWindow | None = field(default=None)

    @classmethod
    def from_profiles(
        cls,
        profiles: dict[str, dict | None],
        analysis_mode: str | None,
        default_slots: int | None,
        priority_window: PriorityWindow | None = None,
    ) -> ModeScheduleMap:
        """Build the per-mode schedules from the parsed config profiles.

        ``profiles`` maps mode name (``fast``/``cheap``) to its parsed config
        (``None`` when the profile file is absent — the default config's
        schedule then applies). ``default_slots`` is the slot-pool fallback
        used when a profile has no slot schedule.
        """
        default_cfg = (profiles or {}).get("default")
        fallback = schedule_from_config(default_cfg, default_slots)
        fallback_ctx = (default_cfg or {}).get("local_model_ctx_size") if default_cfg else None
        schedules = {"fast": fallback, "cheap": fallback}
        ctx_sizes = {"fast": fallback_ctx, "cheap": fallback_ctx}
        for mode in ("fast", "cheap"):
            cfg = (profiles or {}).get(mode)
            if cfg is None:
                continue
            slots = cfg.get("session_slot_pool_size")
            if slots is None:
                slots = default_slots
            schedules[mode] = schedule_from_config(cfg, slots)
            ctx_sizes[mode] = cfg.get("local_model_ctx_size")
        return cls(
            schedules=schedules,
            ctx_sizes=ctx_sizes,
            default_mode=analysis_mode or "fast",
            transitions=[],
            priority_window=priority_window,
        )

    def schedule_for_mode(self, mode: str) -> SlotSchedule:
        """Return the schedule for ``mode`` (falling back to the default mode's
        schedule when the mode has no profile)."""
        s = self.schedules.get(mode)
        if s is not None:
            return s
        return self.schedules.get(self.default_mode) or next(iter(self.schedules.values()))

    def ctx_for(self, mode: str) -> int | None:
        """Global context size of ``mode``'s profile (per-period ctx overrides)."""
        return self.ctx_sizes.get(mode)

    def mode_at(self, dt: datetime) -> str:
        """The mode active at ``dt``: the most recent transition at/before it;
        sessions before the earliest observed transition use the analysis-time
        mode (the documented fallback)."""
        for tr in reversed(self.transitions):
            ts = tr.ts if isinstance(tr, ModeTransition) else tr[0]
            mode = tr.mode if isinstance(tr, ModeTransition) else tr[1]
            if ts <= dt:
                return mode
        return self.default_mode

    @property
    def profile_source(self) -> str | None:
        """Which source attributes fast/cheap for this window.

        ``"mode_schedule"`` when the logs carried mode-timeline evidence
        (historical windows), ``"priority_window"`` when the configured
        ``server.priority_window`` supplies the schedule, ``None`` when
        neither applies (legacy slot-schedule attribution).
        """
        if self.transitions:
            return "mode_schedule"
        if self.priority_window is not None:
            return PRIORITY_WINDOW_SOURCE
        return None

    def priority_periods(self) -> list[SlotPeriod]:
        """Fast/cheap periods derived from ``server.priority_window``.

        Slots / per-period ctx come from the default mode's slot schedule at
        the period's midpoint: the unified config has one profile, so the
        priority window only re-labels the time axis (cheap inside the
        window), it does not change the resource figures. Empty when no
        window is configured (or it is disabled).
        """
        window = self.priority_window
        if window is None:
            return []
        template = self.schedule_for_mode(self.default_mode)
        reference_day = datetime(2000, 1, 1)
        periods: list[SlotPeriod] = []
        for start, end, label in window.segments():
            midpoint = reference_day + timedelta(minutes=(start + end) / 2.0)
            base = template.period_for(midpoint)
            periods.append(
                SlotPeriod(start, end, base.slots, label, ctx_size=base.ctx_size)
            )
        return periods

    def priority_period_for(self, dt: datetime) -> SlotPeriod | None:
        """The priority-window period containing ``dt`` (``None`` if no window)."""
        minutes = minute_of_day(dt)
        for period in self.priority_periods():
            if period.contains(minutes):
                return period
        return None

    def period_for(self, dt: datetime) -> SlotPeriod:
        """Return the slot period of the mode active at ``dt``, labelled with
        the mode name (``fast``/``cheap``) instead of the intra-schedule
        slot-count comparison, so cheap-mode hours always yield a "cheap"
        period with the cheap profile's slots/ctx."""
        mode = self.mode_at(dt)
        period = self.schedule_for_mode(mode).period_for(dt)
        return SlotPeriod(
            period.start_minutes,
            period.end_minutes,
            period.slots,
            mode,
            ctx_size=period.ctx_size,
        )

    @property
    def all_periods(self) -> list[SlotPeriod]:
        """Union of every mode schedule's periods (used as segment-boundary
        candidates so busy-time attribution splits at both profiles'
        transitions)."""
        seen: set[tuple[float, float]] = set()
        out: list[SlotPeriod] = []
        for schedule in self.schedules.values():
            for p in schedule.periods:
                key = (p.start_minutes, p.end_minutes)
                if key not in seen:
                    seen.add(key)
                    out.append(p)
        return out


def bucket_periods(
    schedule: SlotSchedule | None, mode_map: ModeScheduleMap | None
) -> list[SlotPeriod]:
    """Period boundary candidates for fast/cheap attribution.

    Precedence (LP-0MVKSXXIR006OBI2): mode timeline when the logs show mode
    transitions, then the configured priority window, then the legacy
    slot-schedule periods.
    """
    if mode_map is not None:
        if mode_map.transitions:
            return mode_map.all_periods
        priority = mode_map.priority_periods()
        if priority:
            return priority
    if schedule is not None and schedule.periods:
        return schedule.periods
    return []


def bucket_period(
    dt: datetime, schedule: SlotSchedule | None, mode_map: ModeScheduleMap | None
) -> SlotPeriod | None:
    """The fast/cheap period covering ``dt`` (``None`` when nothing applies)."""
    if mode_map is not None:
        if mode_map.transitions:
            return mode_map.period_for(dt)
        priority = mode_map.priority_period_for(dt)
        if priority is not None:
            return priority
    if schedule is not None and schedule.periods:
        return schedule.period_for(dt)
    return None


def bucket_label(
    dt: datetime, schedule: SlotSchedule | None, mode_map: ModeScheduleMap | None
) -> str:
    """The fast/cheap label for ``dt`` (``"fast"`` when nothing applies)."""
    period = bucket_period(dt, schedule, mode_map)
    return period.label if period is not None else "fast"
