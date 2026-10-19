"""Proactive per-account quota state for ``opencode-*`` providers.

This is the tracer bullet for the quota-aware surge epic
(LP-0MVF4FIP1001RD8P, shipped as LP-0MVF7KEG7007C8MF). It gives the proxy an
authoritative, proactive view of each OpenCode Go/Go Plus account's remaining
allowance instead of learning about exhaustion reactively from a
``GoUsageLimitError``.

The authoritative source is ``GET {endpoint}/v1/usage`` (Bearer key,
``cache-control: no-store``) — response headers and bodies carry no quota
information (see the epic's discovery findings). The endpoint takes no model
selector: **quota is per API key (per account), not per model or endpoint**.

What this module provides
-------------------------

- :class:`QuotaWindow` / :class:`QuotaSnapshot` — immutable window values and
  the derived account state (per-window ``headroom``, ``effective_headroom``,
  ``seconds_to_reset``, ``exhausted``, ``last_success_at``, ``last_status``).
- :func:`get_quota_snapshot` — a non-blocking cache read for routing callers.
- :func:`quota_behaviour` / :class:`QuotaBehaviour` — the single predicate
  answering "may quota state influence routing right now?" (enabled /
  dry-run / suspended-on-stale-state / disabled by the kill switch),
  consulted by every quota decision site (LP-0MVF7KPB9009U7CG).
- :func:`refresh_quota_accounts` — an interval-gated, single-flighted
  background refresh for a set of provider entries.
- An injectable module-level transport (:data:`_quota_usage_transport`,
  mirroring :data:`proxy.provider._usage_limit_probe_transport`) and an
  injectable clock (:data:`_now`) so tests never touch the network or the
  wall clock.
- :func:`quota_poll_loop` / :func:`start_quota_poller` /
  :func:`stop_quota_poller` — the background task wired into startup by
  :mod:`proxy.lifecycle` / :mod:`proxy.server`.

Derivation rules (child ACs / epic R2)
--------------------------------------

- ``headroom = 100 - percent`` per window, clamped to ``[0, 100]``.
- ``effective_headroom`` is the **minimum** of the three window headrooms —
  the monthly cap is irrelevant if the rolling or weekly window is spent.
- A window whose ``status`` is anything other than ``"ok"`` (or is entirely
  absent) is treated conservatively as blocking, contributing ``0`` headroom.
- A missing or unparseable ``resetsAt`` yields ``seconds_to_reset <= 0`` —
  never a guessed value.
- The account-level ``seconds_to_reset`` reports the **monthly** window (the
  allowance period that resets per account); the rolling/weekly windows reset
  on much shorter cadences and are exposed per window for callers that need
  them.

Fail-open
---------

A poll error (500, timeout, malformed body, missing key) never raises out of
:func:`refresh_quota_accounts`: the previous values are kept intact, the
snapshot's ``last_status`` is set to the error reason, and a WARN is logged
(rate-limited so a dead endpoint cannot flood the log). A failed poll never
delays a client request because client-facing reads go through
:func:`get_quota_snapshot`, which only touches the in-memory cache.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

import httpx

from proxy import quota_pacing

logger = logging.getLogger("llama-proxy.quota_state")

#: The three quota windows returned by ``/v1/usage``, in nesting order.
WINDOW_NAMES: tuple[str, ...] = ("rolling", "weekly", "monthly")

#: Default poll interval (``server.quota_poll_interval_seconds``).
DEFAULT_POLL_INTERVAL_SECONDS = 900.0

#: Timeout for one ``/v1/usage`` request.
DEFAULT_USAGE_TIMEOUT_SECONDS = 10.0

#: Minimum gap between failure WARNs for the same account.
WARN_MIN_INTERVAL_SECONDS = 300.0

#: ``server.quota_exhaustion_percent`` default: the quota hard stop threshold
#: (LP-0MVF7KO8C005NB59). An account at or above this monthly percentage — or
#: whose ``effective_headroom`` is at or below the equivalent headroom — is
#: skipped before any upstream request is attempted.
DEFAULT_QUOTA_EXHAUSTION_PERCENT = 99.5

#: Inclusive clamp bounds for an out-of-range ``quota_exhaustion_percent``.
#: The value is fail-safe: a mistyped threshold can never disable the guard
#: (the lower bound keeps the skip conservative) nor exceed the percentage
#: scale (the upper bound).
MIN_QUOTA_EXHAUSTION_PERCENT = 1.0
MAX_QUOTA_EXHAUSTION_PERCENT = 100.0

#: One-shot guard for the invalid-``quota_exhaustion_percent`` WARN.
_QUOTA_EXHAUSTION_PERCENT_WARNED = False

#: ``server.quota_state_max_age_seconds`` default: once the last successful
#: poll for an account is older than this, the quota state is treated as
#: **stale** and surge is suspended for that account (fail-safe: no bypass, no
#: quota-driven re-ordering or skip — routing reverts to the configured
#: ``available_times`` and config order).
DEFAULT_QUOTA_STATE_MAX_AGE_SECONDS = 3600.0

#: The four modes :func:`quota_behaviour` can report.
MODE_ENABLED = "enabled"
MODE_DRY_RUN = "dry_run"
MODE_SUSPENDED = "suspended"
MODE_DISABLED = "disabled"

#: Only ``opencode-*`` entries are quota-managed by default.
_OPENCODE_PROVIDER_PREFIX = "opencode"


class QuotaPollError(Exception):
    """Raised by the default transport / parser on an unusable poll result."""


# ---------------------------------------------------------------------------
# Injectable transport and clock
# ---------------------------------------------------------------------------

#: Injectable transport: ``(provider_cfg) -> parsed /v1/usage JSON body``.
#: It must raise on any failure. Tests set this to avoid the network a poll
#: never needs to touch.
_quota_usage_transport: Callable[[dict], Awaitable[dict]] | None = None


def _default_now() -> datetime:
    """Wall clock in UTC (the module's default, injectable for tests)."""
    return datetime.now(UTC)


_now: Callable[[], datetime] = _default_now


def set_quota_transport(
    transport: Callable[[dict], Awaitable[dict]] | None,
) -> None:
    """Install (or clear) the injectable ``/v1/usage`` transport."""
    global _quota_usage_transport
    _quota_usage_transport = transport


def set_clock(now: Callable[[], datetime] | None) -> None:
    """Install (or reset) the injectable wall clock."""
    global _now
    _now = now or _default_now


# ---------------------------------------------------------------------------
# Derived state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuotaWindow:
    """One window (``rolling`` / ``weekly`` / ``monthly``) of a quota report.

    ``percent`` is the percentage of the window's allowance already consumed.
    ``present`` is ``False`` when the upstream payload omitted the window
    entirely; ``status`` is the upstream status string (``"ok"`` when healthy).
    """

    percent: float = 100.0
    resets_at: datetime | None = None
    status: str = "ok"
    present: bool = True

    @property
    def blocking(self) -> bool:
        """True when this window must be treated conservatively as spent."""
        return (not self.present) or self.status != "ok"

    @property
    def headroom(self) -> float:
        """Remaining allowance percentage (``0`` when blocking)."""
        if self.blocking:
            return 0.0
        return max(0.0, min(100.0, 100.0 - self.percent))

    def seconds_to_reset_at(self, now: datetime) -> float:
        """Seconds until ``resets_at``, or ``0`` when missing / in the past."""
        if self.resets_at is None:
            return 0.0
        remaining = (self.resets_at - now).total_seconds()
        return remaining if remaining > 0 else 0.0

    @property
    def seconds_to_reset(self) -> float:
        """Seconds until this window resets, using the module clock."""
        return self.seconds_to_reset_at(_now())


@dataclass(frozen=True)
class QuotaSnapshot:
    """The last known authoritative quota state for one account."""

    account: str
    rolling: QuotaWindow
    weekly: QuotaWindow
    monthly: QuotaWindow
    last_success_at: datetime
    last_status: str = "ok"

    def window(self, name: str) -> QuotaWindow:
        """Return the :class:`QuotaWindow` for ``name`` (one of WINDOW_NAMES)."""
        if name not in WINDOW_NAMES:
            raise KeyError(name)
        return getattr(self, name)

    @property
    def windows(self) -> dict[str, QuotaWindow]:
        """All three windows keyed by name."""
        return {name: getattr(self, name) for name in WINDOW_NAMES}

    @property
    def rolling_headroom(self) -> float:
        return self.rolling.headroom

    @property
    def weekly_headroom(self) -> float:
        return self.weekly.headroom

    @property
    def monthly_headroom(self) -> float:
        return self.monthly.headroom

    @property
    def effective_headroom(self) -> float:
        """The minimum headroom across the three windows (epic R2)."""
        return min(window.headroom for window in self.windows.values())

    @property
    def exhausted(self) -> bool:
        """True when any window blocks or no effective headroom remains."""
        if any(window.blocking for window in self.windows.values()):
            return True
        return self.effective_headroom <= 0.0

    @property
    def seconds_to_reset(self) -> float:
        """Seconds until the account's monthly allowance resets.

        The monthly window is the account's allowance period (reset dates are
        per account and not calendar-aligned). Returns ``0`` when the reset
        time is missing or unparseable in **any** window — a partial report is
        never allowed to produce a guessed reset (child AC2b).
        """
        now = _now()
        if any(window.resets_at is None for window in self.windows.values()):
            return 0.0
        return self.monthly.seconds_to_reset_at(now)


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------


def _coerce_percent(value: Any) -> float:
    """Coerce a ``percent`` field to a float in ``[0, 100]``.

    A missing / malformed value is treated as fully consumed (conservative).
    """
    try:
        percent = float(value)
    except (TypeError, ValueError):
        return 100.0
    return max(0.0, min(100.0, percent))


def _parse_resets_at(value: Any) -> datetime | None:
    """Parse an ISO-8601 ``resetsAt`` into an aware UTC datetime.

    Returns ``None`` for a missing or unparseable value so callers derive
    ``seconds_to_reset <= 0`` rather than a guessed reset.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def parse_usage_payload(body: Any) -> dict[str, QuotaWindow]:
    """Derive the three :class:`QuotaWindow` values from a ``/v1/usage`` body.

    Raises :class:`QuotaPollError` when the body has no ``usage`` object — a
    malformed response is a poll failure the caller fails open on.
    """
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        raise QuotaPollError("missing 'usage' object in /v1/usage response")
    windows: dict[str, QuotaWindow] = {}
    for name in WINDOW_NAMES:
        raw = usage.get(name)
        if not isinstance(raw, dict):
            windows[name] = QuotaWindow(
                percent=100.0, resets_at=None, status="missing", present=False
            )
            continue
        windows[name] = QuotaWindow(
            percent=_coerce_percent(raw.get("percent")),
            resets_at=_parse_resets_at(raw.get("resetsAt")),
            status=str(raw.get("status") or "ok"),
        )
    return windows


def snapshot_from_usage(
    account: str, body: Any, *, fetched_at: datetime | None = None
) -> QuotaSnapshot:
    """Build a :class:`QuotaSnapshot` from a raw ``/v1/usage`` body."""
    windows = parse_usage_payload(body)
    return QuotaSnapshot(
        account=account,
        rolling=windows["rolling"],
        weekly=windows["weekly"],
        monthly=windows["monthly"],
        last_success_at=fetched_at or _now(),
    )


# ---------------------------------------------------------------------------
# Config helpers (flat-first, then ``server.*`` — same pattern as provider.py)
# ---------------------------------------------------------------------------


def _config_value(config: dict | None, key: str, default: Any) -> Any:
    if not isinstance(config, dict):
        return default
    value = config.get(key)
    if value is None:
        server = config.get("server")
        if isinstance(server, dict):
            value = server.get(key)
    return default if value is None else value


def reset_quota_exhaustion_warning() -> None:
    """Forget the one-shot invalid-threshold WARN (tests / config reload)."""
    global _QUOTA_EXHAUSTION_PERCENT_WARNED
    _QUOTA_EXHAUSTION_PERCENT_WARNED = False


def quota_exhaustion_percent(config: dict | None) -> float:
    """``server.quota_exhaustion_percent`` (default 99.5), always fail-safe.

    Returns the monthly-consumption percentage at or above which the quota
    hard stop skips an account (LP-0MVF7KO8C005NB59). The absent key yields
    :data:`DEFAULT_QUOTA_EXHAUSTION_PERCENT` so the guard is on by default.

    Invalid values never disable the guard:

    - a numeric value outside ``[1, 100]`` is clamped into that range (so
      ``0``/``-5`` become ``1`` and ``101`` becomes ``100``);
    - a non-numeric or non-finite value (``"abc"``, ``NaN``, ``inf``) falls
      back to the default ``99.5``;
    - either case logs exactly one WARN per process (``once``) so a
      misconfigured key is visible without flooding the log on every
      dispatch decision.
    """
    global _QUOTA_EXHAUSTION_PERCENT_WARNED
    raw = _config_value(config, "quota_exhaustion_percent", None)
    if raw is None:
        return DEFAULT_QUOTA_EXHAUSTION_PERCENT
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = None
    if value is not None and math.isfinite(value) and (
        value < MIN_QUOTA_EXHAUSTION_PERCENT
        or value > MAX_QUOTA_EXHAUSTION_PERCENT
    ):
        clamped = min(
            MAX_QUOTA_EXHAUSTION_PERCENT,
            max(MIN_QUOTA_EXHAUSTION_PERCENT, value),
        )
        _warn_invalid_quota_exhaustion_percent(raw, clamped)
        return clamped
    if value is None or not math.isfinite(value):
        _warn_invalid_quota_exhaustion_percent(
            raw, DEFAULT_QUOTA_EXHAUSTION_PERCENT
        )
        return DEFAULT_QUOTA_EXHAUSTION_PERCENT
    return value


def _warn_invalid_quota_exhaustion_percent(raw: Any, used: float) -> None:
    """WARN once about an invalid threshold, naming the value actually used."""
    global _QUOTA_EXHAUSTION_PERCENT_WARNED
    if _QUOTA_EXHAUSTION_PERCENT_WARNED:
        return
    _QUOTA_EXHAUSTION_PERCENT_WARNED = True
    logger.warning(
        "quota_exhaustion_percent value=%r is invalid or out of range; "
        "using %.1f (valid range [%.0f, %.0f], default %.1f)",
        raw,
        used,
        MIN_QUOTA_EXHAUSTION_PERCENT,
        MAX_QUOTA_EXHAUSTION_PERCENT,
        DEFAULT_QUOTA_EXHAUSTION_PERCENT,
    )


def quota_state_max_age_seconds(config: dict | None) -> float:
    """``server.quota_state_max_age_seconds`` (default 3600 s, never negative).

    The staleness horizon for :func:`quota_behaviour`: a snapshot whose
    ``last_success_at`` is older than this many seconds suspends surge for
    that account (fail-safe, not fail-open).

    Invalid values leak in the **fail-safe** direction only: a malformed or
    non-finite value falls back to the default, and a negative value is
    clamped to ``0`` ("any age is stale"), which suspends rather than
    trusting. ``0`` therefore means "always suspend once polled".
    """
    raw = _config_value(config, "quota_state_max_age_seconds", None)
    if raw is None:
        return DEFAULT_QUOTA_STATE_MAX_AGE_SECONDS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_QUOTA_STATE_MAX_AGE_SECONDS
    if not math.isfinite(value):
        return DEFAULT_QUOTA_STATE_MAX_AGE_SECONDS
    return max(0.0, value)


def poll_interval_seconds(config: dict | None) -> float:
    """``server.quota_poll_interval_seconds`` (default 900 s, minimum 1 s)."""
    try:
        value = float(
            _config_value(
                config, "quota_poll_interval_seconds", DEFAULT_POLL_INTERVAL_SECONDS
            )
        )
    except (TypeError, ValueError):
        return DEFAULT_POLL_INTERVAL_SECONDS
    return max(1.0, value)


def surge_accounts(config: dict | None) -> list[str] | None:
    """``server.quota_surge_accounts`` allow-list (``None`` = all accounts).

    Accepts an entry-name or ``api_key_env`` allow-list; a comma-separated
    string is accepted for convenience.
    """
    value = _config_value(config, "quota_surge_accounts", None)
    if value is None:
        return None
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple, set)):
        return None
    resolved = [str(item).strip() for item in value if str(item).strip()]
    return resolved or None


def is_quota_provider(provider_cfg: Any, config: dict | None = None) -> bool:
    """Whether *provider_cfg* is a quota-managed ``opencode-*`` entry.

    The default scope is every entry whose name or brand starts with
    ``opencode``; an explicit ``server.quota_surge_accounts`` allow-list
    restricts it to the named entry names / ``api_key_env`` values.
    """
    if not isinstance(provider_cfg, dict):
        return False
    name = str(provider_cfg.get("name") or "")
    brand = str(provider_cfg.get("provider") or "")
    if not (
        name.startswith(_OPENCODE_PROVIDER_PREFIX)
        or brand.startswith(_OPENCODE_PROVIDER_PREFIX)
    ):
        return False
    allowed = surge_accounts(config)
    if allowed is None:
        return True
    api_key_env = str(provider_cfg.get("api_key_env") or "")
    return name in allowed or api_key_env in allowed


def account_key(provider_cfg: dict) -> str:
    """Canonical account key for the quota cache.

    Quota is per API key, **not** per model or endpoint (epic finding 3), and
    the ``/v1/usage`` endpoint takes no model selector — so two entries that
    share an ``api_key_env`` must share one snapshot and one poll
    (child AC1c). Unlike :func:`proxy.provider._usage_limit_account_key`, the
    model is deliberately excluded. Entries without an ``api_key_env`` fall
    back to the entry name so unrelated entries never collide.
    """
    api_key_env = provider_cfg.get("api_key_env")
    if api_key_env:
        return str(api_key_env)
    return str(provider_cfg.get("name") or provider_cfg.get("provider") or "unknown")


def collect_quota_providers(config: dict | None) -> list[dict]:
    """Flatten every provider entry declared in *config*.

    Scans both the top-level ``providers`` list and each ``models.<name>``
    entry's ``providers`` list, so the lifecycle poller covers every model
    that routes through an ``opencode-*`` account.
    """
    providers: list[dict] = []
    if not isinstance(config, dict):
        return providers
    top = config.get("providers")
    if isinstance(top, list):
        providers.extend(p for p in top if isinstance(p, dict))
    models = config.get("models")
    if isinstance(models, dict):
        for model_cfg in models.values():
            if not isinstance(model_cfg, dict):
                continue
            entries = model_cfg.get("providers")
            if isinstance(entries, list):
                providers.extend(p for p in entries if isinstance(p, dict))
    return providers


def _providers_from(model_config: Any) -> list[dict]:
    """Provider entries for a model config or a whole config dict."""
    if not isinstance(model_config, dict):
        return []
    entries = model_config.get("providers")
    if isinstance(entries, list):
        return [p for p in entries if isinstance(p, dict)]
    return collect_quota_providers(model_config)


# ---------------------------------------------------------------------------
# Cache state and reads
# ---------------------------------------------------------------------------

#: account key -> last known snapshot.
_snapshots: dict[str, QuotaSnapshot] = {}
#: account key -> wall-clock time of the last poll *start* (interval gating).
_last_poll_at: dict[str, datetime] = {}
#: account key -> in-flight poll task (single-flight coalescing).
_inflight: dict[str, asyncio.Task] = {}
#: account key -> wall-clock time of the last failure WARN (log rate limit).
_last_warn_at: dict[str, datetime] = {}


def get_quota_snapshot(
    provider_cfg: dict, config: dict | None = None
) -> QuotaSnapshot | None:
    """Return the cached snapshot for *provider_cfg*, or ``None``.

    A pure cache read: it never blocks, never polls and never raises, so a
    routing caller can consult it in the request's critical path. Returns
    ``None`` before the first successful poll (or when the entry has no
    account key).
    """
    if not isinstance(provider_cfg, dict):
        return None
    return _snapshots.get(account_key(provider_cfg))


def cached_snapshots() -> dict[str, QuotaSnapshot]:
    """Return a copy of the full account-key → snapshot cache."""
    return dict(_snapshots)


# ---------------------------------------------------------------------------
# Central behaviour predicate (kill switch / dry-run / stale suspension)
# ---------------------------------------------------------------------------

#: account key -> currently-suspended flag, for the once-per-suspension WARN.
_stale_suspension_active: set[str] = set()


@dataclass(frozen=True)
class QuotaBehaviour:
    """One immutable verdict: how may quota state influence routing right now.

    The single predicate consulted by every quota decision site (ordering,
    bypass and skip — LP-0MVF7KPB9009U7CG), so a test can prove "all quota
    behaviour off" by construction. ``mode`` is one of the :data:`MODE_*`
    constants:

    - ``enabled`` — quota state may change routing (ordering, bypass, skip);
    - ``dry_run`` — decisions are still *computed* (and logged with
      ``reason=quota_surge_dry_run``) but never applied: routing is
      byte-identical to the pre-feature behaviour;
    - ``suspended`` — at least one account's snapshot is stale (older than
      ``server.quota_state_max_age_seconds``): quota state is **not**
      consulted for routing at all, reverting to configured
      ``available_times`` and config order (fail-safe, not fail-open);
    - ``disabled`` — ``server.quota_surge_enabled: false``: every quota-derived
      decision is skipped and routing is byte-identical to today.

    Suspension is deliberately **global** (any stale snapshot suspends surge
    for the routing decision): a stale reading is guesswork, and the fail-safe
    is that the surge makes *no* ordering/bypass/skip decisions from it — so a
    stale candidate can never be ranked below fresh ones or dropped (child
    AC9b). ``stale_accounts`` names every account that caused the suspension,
    and ``last_poll_age_seconds`` is the oldest such age (for log
    observability).
    """

    mode: str
    stale_accounts: tuple[str, ...] = ()
    last_poll_age_seconds: float = 0.0
    max_age_seconds: float = DEFAULT_QUOTA_STATE_MAX_AGE_SECONDS

    @property
    def enabled(self) -> bool:
        """The feature itself is on (not the kill switch)."""
        return self.mode != MODE_DISABLED

    @property
    def dry_run(self) -> bool:
        return self.mode == MODE_DRY_RUN

    @property
    def suspended(self) -> bool:
        return self.mode == MODE_SUSPENDED

    @property
    def disabled(self) -> bool:
        return self.mode == MODE_DISABLED

    @property
    def applies(self) -> bool:
        """True when quota state may change routing (strictly enabled)."""
        return self.mode == MODE_ENABLED

    @property
    def computes(self) -> bool:
        """True when quota decisions are (still) worth computing for logging."""
        return self.mode in (MODE_ENABLED, MODE_DRY_RUN)

    def log_reason(self) -> str:
        """The stable ``reason=`` code call sites append to their log lines."""
        if self.dry_run:
            return "quota_surge_dry_run"
        if self.suspended:
            return "quota_surge_suspended"
        return "quota_surge"


#: account key -> currently-suspended flag, for the once-per-suspension WARN.
_stale_suspension_active: set[str] = set()


def _warn_stale_suspension(
    account: str,
    snapshot: QuotaSnapshot,
    age: float,
    max_age: float,
) -> None:
    """WARN once per account per suspension transition (fail-safe notice)."""
    if account in _stale_suspension_active:
        return
    _stale_suspension_active.add(account)
    logger.warning(
        "quota surge suspended account=%s last_poll_age=%.0fs "
        "max_age=%.0fs reason=quota_surge_suspended fail_safe=true "
        "effective_headroom=%.1f%% resets_in=%ds",
        account,
        age,
        max_age,
        snapshot.effective_headroom,
        int(snapshot.seconds_to_reset),
    )


def _resume_suspension(account: str) -> None:
    """Clear the suspension flag and note the recovery once per resume."""
    if account not in _stale_suspension_active:
        return
    _stale_suspension_active.discard(account)
    logger.info(
        "quota surge resumed account=%s reason=quota_surge_resumed",
        account,
    )


def reset_quota_behaviour_state() -> None:
    """Forget per-account suspension WARN state (tests, config reload)."""
    _stale_suspension_active.clear()


def quota_behaviour(
    config: dict | None = None,
    *,
    now: datetime | None = None,
) -> QuotaBehaviour:
    """Return the mode quota state may influence routing for *config*.

    The **single predicate** consulted by every quota decision site — headroom
    ordering, the final-week ``available_times`` bypass and the quota-driven
    skip (LP-0MVF7KPB9009U7CG) — so "all quota behaviour off" is scrutably
    true whenever it reports ``disabled`` or ``suspended``.

    Mode resolution, in order:

    1. ``server.quota_surge_enabled: false`` → :data:`MODE_DISABLED` (the kill
       switch; nothing is computed or applied);
    2. any cached snapshot older than ``server.quota_state_max_age_seconds``
       (default 3600 s) → :data:`MODE_SUSPENDED`, with exactly one WARN per
       account per suspension transition. Suspension is global: **any** stale
       snapshot suspends the whole decision, so no routing decision is ever
       made from guesswork (child AC9 / AC9b);
    3. ``server.quota_surge_dry_run: true`` → :data:`MODE_DRY_RUN`;
    4. otherwise :data:`MODE_ENABLED`.

    A **missing** snapshot (never polled, account not matched, endpoint
    absent) never suspends: with nothing to be stale about, the mode is
    ``dry_run``/``enabled`` and the routing call sites fail closed on the
    absent snapshot by themselves (child AC9b). ``now`` defaults to the
    module clock (:func:`set_clock`), so tests freeze it together with the
    poller.
    """
    if not quota_pacing.surge_enabled(config):
        return QuotaBehaviour(MODE_DISABLED)

    current = now if now is not None else _now()
    max_age = quota_state_max_age_seconds(config)
    stale: list[tuple[str, float, QuotaSnapshot]] = []
    for account, snapshot in _snapshots.items():
        age = (current - snapshot.last_success_at).total_seconds()
        if age > max_age:
            stale.append((account, age, snapshot))

    if stale:
        for account, age, snapshot in stale:
            _warn_stale_suspension(account, snapshot, age, max_age)
        recovered = set(_stale_suspension_active) - {a for a, _, _ in stale}
        for account in recovered:
            _resume_suspension(account)
        oldest = max(age for _, age, _ in stale)
        return QuotaBehaviour(
            MODE_SUSPENDED,
            stale_accounts=tuple(a for a, _, _ in stale),
            last_poll_age_seconds=oldest,
            max_age_seconds=max_age,
        )

    for account in list(_stale_suspension_active):
        _resume_suspension(account)

    if quota_pacing.surge_dry_run(config):
        return QuotaBehaviour(MODE_DRY_RUN)
    return QuotaBehaviour(MODE_ENABLED)


def reset_quota_state() -> None:
    """Clear all cached state, in-flight tasks and injected hooks (tests)."""
    for task in list(_inflight.values()):
        if not task.done():
            task.cancel()
    _snapshots.clear()
    _last_poll_at.clear()
    _inflight.clear()
    _last_warn_at.clear()
    reset_quota_exhaustion_warning()
    reset_quota_behaviour_state()
    set_quota_transport(None)
    set_clock(None)


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------


async def _default_quota_usage_transport(provider_cfg: dict) -> dict:
    """Fetch ``GET {endpoint}/v1/usage`` for a provider entry.

    Bearer-authenticated; ``cache-control: no-store`` is honoured both by never
    caching the response locally and by asking the upstream not to. Raises on
    a missing endpoint/key, a non-2xx status or any transport error.
    """
    endpoint = str(provider_cfg.get("endpoint") or "").rstrip("/")
    if not endpoint:
        raise QuotaPollError("provider entry has no endpoint")
    api_key_env = provider_cfg.get("api_key_env")
    api_key = os.environ.get(api_key_env) if api_key_env else None
    if not api_key:
        api_key = provider_cfg.get("api_key")
    if not api_key:
        raise QuotaPollError("no API key available for provider entry")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Cache-Control": "no-store",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=DEFAULT_USAGE_TIMEOUT_SECONDS) as client:
        response = await client.get(f"{endpoint}/v1/usage", headers=headers)
    response.raise_for_status()
    return response.json()


async def _fetch_usage(provider_cfg: dict) -> dict:
    """Fetch a usage body through the injected transport, else the default."""
    transport = _quota_usage_transport
    if transport is not None:
        return await transport(provider_cfg)
    return await _default_quota_usage_transport(provider_cfg)


def _warn_poll_failure(
    account: str, exc: Exception, previous: QuotaSnapshot | None
) -> None:
    """Warn about a failed poll, rate-limited per account."""
    now = _now()
    last = _last_warn_at.get(account)
    if last is not None and (now - last).total_seconds() < WARN_MIN_INTERVAL_SECONDS:
        return
    _last_warn_at[account] = now
    logger.warning(
        "quota poll failed account=%s error=%s keeping_last_known=%s",
        account,
        type(exc).__name__,
        "true" if previous is not None else "false",
    )


def _handle_poll_failure(
    account: str, exc: Exception
) -> QuotaSnapshot | None:
    """Fail-open handler: keep previous values, record status, WARN.

    Returns the (unchanged-values) snapshot with ``last_status`` set to the
    error reason, or ``None`` when no snapshot has ever succeeded.
    """
    previous = _snapshots.get(account)
    _warn_poll_failure(account, exc, previous)
    if previous is None:
        return None
    failed = replace(previous, last_status=f"poll_error:{type(exc).__name__}")
    _snapshots[account] = failed
    return failed


async def _poll_account(account: str, provider_cfg: dict) -> QuotaSnapshot | None:
    """Poll one account once. Never raises (except cancellation)."""
    try:
        body = await _fetch_usage(provider_cfg)
        snapshot = snapshot_from_usage(account, body)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return _handle_poll_failure(account, exc)

    _snapshots[account] = snapshot
    _last_warn_at.pop(account, None)
    logger.info(
        "quota poll ok account=%s rolling=%.1f%% weekly=%.1f%% monthly=%.1f%% "
        "seconds_to_reset=%ds effective_headroom=%.1f",
        account,
        snapshot.rolling.percent,
        snapshot.weekly.percent,
        snapshot.monthly.percent,
        int(snapshot.seconds_to_reset),
        snapshot.effective_headroom,
    )
    return snapshot


async def _refresh_account(
    account: str,
    provider_cfg: dict,
    config: dict | None,
    *,
    force: bool,
) -> QuotaSnapshot | None:
    """Coalesce + interval-gate a single account's poll."""
    task = _inflight.get(account)
    if task is not None:
        if not task.done():
            # Single-flight: a concurrent refresh joins the in-flight request.
            return await task
        _inflight.pop(account, None)

    if not force:
        last = _last_poll_at.get(account)
        if last is not None and (_now() - last).total_seconds() < poll_interval_seconds(
            config
        ):
            cached = _snapshots.get(account)
            if cached is not None:
                return cached

    _last_poll_at[account] = _now()
    task = asyncio.ensure_future(_poll_account(account, provider_cfg))
    _inflight[account] = task
    try:
        return await task
    finally:
        if _inflight.get(account) is task:
            _inflight.pop(account, None)


async def refresh_quota_accounts(
    model_config: dict,
    config: dict | None = None,
    *,
    force: bool = False,
) -> dict[str, QuotaSnapshot]:
    """Refresh quota state for every eligible account in *model_config*.

    Iterates the ``opencode-*`` provider entries, de-duplicates them by
    account key, and polls each distinct account at most once per
    ``server.quota_poll_interval_seconds`` (single-flight coalescing means
    concurrent callers share one in-flight request). Never raises; failures
    fail open. ``force=True`` bypasses the interval gate (used by tests and
    explicit refreshes).
    """
    accounts: dict[str, dict] = {}
    for provider_cfg in _providers_from(model_config):
        if not is_quota_provider(provider_cfg, config):
            continue
        accounts.setdefault(account_key(provider_cfg), provider_cfg)

    results: dict[str, QuotaSnapshot] = {}
    for account, provider_cfg in accounts.items():
        snapshot = await _refresh_account(
            account, provider_cfg, config, force=force
        )
        if snapshot is not None:
            results[account] = snapshot
    return results


# ---------------------------------------------------------------------------
# Background task (wired into startup by proxy.lifecycle / proxy.server)
# ---------------------------------------------------------------------------

_quota_poll_task: asyncio.Task | None = None


async def quota_poll_loop(config: dict | None) -> None:
    """Refresh every quota account, then sleep for the poll interval.

    A poll-cycle failure is logged and swallowed so a dead endpoint can never
    break startup or the loop (child scope: "a failed poll must never break
    startup").
    """
    while True:
        try:
            providers = collect_quota_providers(config)
            await refresh_quota_accounts({"providers": providers}, config)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("quota poll cycle failed", exc_info=True)
        await asyncio.sleep(poll_interval_seconds(config))


def start_quota_poller(config: dict | None) -> asyncio.Task | None:
    """Start the background quota poller if not already running.

    Best-effort: returns ``None`` when no event loop is running so a sync
    startup path cannot fail. The task itself never raises out of the loop.
    """
    global _quota_poll_task
    if _quota_poll_task is not None and not _quota_poll_task.done():
        return _quota_poll_task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    _quota_poll_task = loop.create_task(quota_poll_loop(config))
    logger.info(
        "Quota poller task started (interval=%ss)",
        poll_interval_seconds(config),
    )
    return _quota_poll_task


async def stop_quota_poller() -> None:
    """Cancel the background quota poller and await its termination."""
    global _quota_poll_task
    task = _quota_poll_task
    _quota_poll_task = None
    if task is None:
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
