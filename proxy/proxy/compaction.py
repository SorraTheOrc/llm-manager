"""
Proxy-side proactive session compaction — core summarization strategy.

Feature: LP-0MTGBOOJY004OWPI (child of LP-0MTCWE8NG003P0SD).

Implements the *summarize-first* (Strategy C) compaction primitive: when a
session's estimated token count exceeds the per-mode trigger (configurable
``compaction_trigger_ratio`` × the effective per-slot context, default
0.70), the middle turns are folded into a summary injected below the
retained first prompt, and whole recent turns are kept within the per-mode
target budget (fast ≤ 38K, cheap ≤ 30K).

Retention invariant (AC1): the system prompt and the very first user prompt
appear verbatim in every compacted output — same dict objects, never
re-serialized. The summary is injected at the top of the prompt,
immediately below the retained first prompt (AC5).

Non-compactable sessions (summarizer unavailable) resolve to an explicit
``remote_with_guidance`` action: the dispatcher must route remote with
guidance, never silently (AC4).

The ``estimate_tokens`` callable is injectable so callers can reuse the
production routing estimator (``_estimate_prompt_tokens_for_routing``);
unit tests inject deterministic estimators. Output is fully deterministic
for a given input (AC7 — composes with slot save/restore).

The backstop (``truncate_backstop``, LP-0MTGBOYJX006KVN8) is the logged
safety net: when the summary path alone leaves the session over budget it
drops the oldest whole recent turns; ``plan_session_compaction(backstop=True)``
chains it automatically. When even the backstop cannot reach the budget it
reports ``backstop_exhausted`` so the dispatcher can escalate to remote
with guidance. Structured churn/compaction logging is the sibling logging
child's scope; this module reports ``compacted_over_budget`` (or
``backstop_*`` reasons) so the dispatcher can escalate.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-mode compaction budgets (operator-approved, LP-0MTCWE8NG003P0SD):
#   fast  : compacted prompt must be ≤ 38 000 tokens
#   cheap : compacted prompt must be ≤ 30 000 tokens
# ---------------------------------------------------------------------------
FAST_COMPACTION_TARGET_TOKENS = 38000
CHEAP_COMPACTION_TARGET_TOKENS = 30000

_SYSTEM_ROLE = "system"
_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"

# Marker wrapping the injected summary. Kept stable with the experiment
# harness format (proxy/scripts/run_compaction_experiment.py) so operators
# and downstream tooling recognise compaction artifacts.
_SUMMARY_MARKER = (
    "The conversation history before this point was compacted into "
    "the following summary:\n\n<summary>\n"
)
_SUMMARY_MARKER_END = "\n</summary>"

# Callable that turns the middle messages into a concise summary string.
Summarizer = Callable[[list[dict[str, Any]]], str]
# Callable that estimates the token count of a full message list.
TokenEstimator = Callable[[list[dict[str, Any]]], int]


def compaction_target_tokens(mode: str) -> int:
    """Return the compaction target budget (max compacted tokens) for *mode*.

    Unknown modes resolve to the cheap budget to stay conservative.
    """
    if mode == "fast":
        return FAST_COMPACTION_TARGET_TOKENS
    return CHEAP_COMPACTION_TARGET_TOKENS


def compaction_trigger_tokens(mode: str, config: dict) -> int:
    """Resolve the compaction trigger (tokens) for *mode*.

    ``trigger = compaction_trigger_ratio × effective_per_slot_threshold``
    where the per-slot threshold uses the same clamp as the routing
    machinery (``ctx_size // slots - 4096`` headroom, LP-0MSAZXXDY005AWA1).
    The product is rounded half-up so the operator-approved constants
    resolve exactly:

    - fast  schedule (3 slots × 262144): round(0.70 × 83285) = 58300
    - cheap schedule (2 slots × 131072): round(0.70 × 61440) = 43008 (≈43K)

    Returns 0 when compaction is disabled (ratio ≤ 0 or per-slot clamp
    not computable). Lazy-imports the provider helpers to avoid a circular
    import (provider imports this module for routing integration).
    """
    from proxy.provider import (
        _get_active_local_ctx_size,
        _get_active_local_slots,
        compaction_config,
        effective_per_slot_threshold,
    )

    ratio = compaction_config(config)["trigger_ratio"]
    if ratio <= 0:
        return 0
    ctx_size = _get_active_local_ctx_size(config)
    slots = _get_active_local_slots(config)
    per_slot = effective_per_slot_threshold(ctx_size, slots)
    if per_slot <= 0:
        return 0
    # Decimal arithmetic keeps the operator-approved constants exact
    # (float 0.7 noise would truncate 58299.5 → 58299).
    return int(
        (Decimal(str(per_slot)) * Decimal(str(ratio))).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )


def should_compact_session(estimated_tokens: int, mode: str, config: dict) -> bool:
    """True when a session at *estimated_tokens* should be compacted.

    Fires strictly above the trigger (``est_tokens > trigger``), matching
    the operator-approved thresholds (fast 58,300 / cheap ≈43K). Disabled
    (trigger 0) never fires.
    """
    trigger = compaction_trigger_tokens(mode, config)
    if trigger <= 0:
        return False
    return int(estimated_tokens) > trigger


def estimate_session_tokens(
    messages: list[dict[str, Any]],
    estimate_tokens: TokenEstimator | None = None,
) -> int:
    """Estimate the token cost of a message list.

    Uses the injected estimator when provided (production wiring passes the
    routing estimator through); otherwise falls back to a lightweight
    char-based heuristic (1 token ≈ 4 chars + per-message overhead) that is
    deterministic and suitable for degraded operation.
    """
    if estimate_tokens is not None:
        return int(estimate_tokens(messages) or 0)
    return sum(1 + len(str(m.get("content", ""))) // 4 for m in messages)


def pair_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group a message list into whole turns.

    A new turn begins at each user message; every following message
    (assistant, tool, etc.) attaches to the current turn until the next
    user message. A leading assistant message (no preceding user) opens its
    own turn. Lossless: no message is dropped.
    """
    turns: list[list[dict[str, Any]]] = []
    for msg in messages:
        if msg.get("role") == _USER_ROLE or not turns:
            turns.append([msg])
        else:
            turns[-1].append(msg)
    return turns


def _summary_message(summary_text: str) -> dict[str, str]:
    """Wrap a summary in the compaction marker as a user-role message."""
    return {
        "role": _USER_ROLE,
        "content": f"{_SUMMARY_MARKER}{summary_text}{_SUMMARY_MARKER_END}",
    }


def truncate_backstop(
    compacted_messages: list[dict[str, Any]],
    target_tokens: int,
    estimate_tokens: TokenEstimator | None = None,
) -> dict[str, Any]:
    """Drop the oldest whole recent turns until the estimate fits the target.

    Safety-net backstop (LP-0MTGBOYJX006KVN8) for when summarization alone
    leaves the session over budget. Operates on the compacted list produced
    by ``plan_session_compaction``: everything through the summary marker
    (system prompt + first user prompt + summary) is protected and never
    dropped (AC2); the recent turns after the marker are dropped in whole
    turn units, oldest first — never splitting a turn, never touching the
    system/first-prompt region.

    Only acts when the list is over budget (AC1): at/below the target the
    input is returned untouched.

    Args:
        compacted_messages: The compacted message list (summary marker
            required to define the droppable region).
        target_tokens: The per-mode budget to restore.
        estimate_tokens: Estimator over a full message list (defaults to
            the module heuristic).

    Returns:
        A dict with:
        - action: "noop" | "dropped" | "exhausted"
        - messages: the resulting list (same object when noop)
        - dropped_turns / dropped_messages: drop accounting
        - estimated_before / estimated_after: token estimates
        - remaining_budget: target - estimated_after (< 0 when exhausted)
    """
    est = estimate_tokens or estimate_session_tokens
    estimated_before = est(compacted_messages)
    result: dict[str, Any] = {
        "action": "noop",
        "messages": compacted_messages,
        "dropped_turns": 0,
        "dropped_messages": 0,
        "estimated_before": estimated_before,
        "estimated_after": estimated_before,
        "remaining_budget": target_tokens - estimated_before,
    }
    if estimated_before <= target_tokens:
        return result

    summary_idx = next(
        (
            i
            for i, m in enumerate(compacted_messages)
            if isinstance(m.get("content"), str)
            and m["content"].startswith(_SUMMARY_MARKER)
        ),
        None,
    )
    if summary_idx is None:
        # No compaction marker → no defined droppable region; never touch
        # the input (defensive; wired only onto plan output).
        result["action"] = "exhausted"
        return result

    protected = compacted_messages[: summary_idx + 1]
    region = list(compacted_messages[summary_idx + 1 :])
    turns = pair_turns(region)
    dropped_turns = 0
    estimated = estimated_before
    while estimated > target_tokens and turns:
        turn = turns.pop(0)  # oldest whole turn
        region = region[len(turn) :]
        turns = pair_turns(region)
        dropped_turns += 1
        estimated = est(protected + region)

    action = "dropped" if dropped_turns and estimated <= target_tokens else "exhausted"
    result.update(
        action=action,
        messages=protected + region,
        dropped_turns=dropped_turns,
        dropped_messages=len(compacted_messages) - len(protected + region),
        estimated_after=estimated,
        remaining_budget=target_tokens - estimated,
    )
    if action != "noop":
        logger.warning(
            "compaction_backstop action=%s dropped_turns=%d dropped_messages=%d "
            "estimated_before=%d estimated_after=%d target=%d remaining_budget=%d",
            action,
            dropped_turns,
            result["dropped_messages"],
            estimated_before,
            estimated,
            target_tokens,
            result["remaining_budget"],
        )
    return result


# Log levels for the canonical compaction event (LP-0MTGBP8DX003R5ZO).
# Tidy compactions are INFO; anything that signals loss or escalation
# (backstop drop, budget exhaustion, remote fallback) is WARNING so it is
# never silent in operational monitoring.
_WARNING_REASONS = frozenset(
    {
        "backstop_dropped",
        "backstop_exhausted",
        "compacted_over_budget",
        "remote_with_guidance",
    }
)


def log_compaction_event(
    plan_result: dict[str, Any],
    *,
    session_id: str,
    dry_run: bool = False,
    logger_obj: logging.Logger | None = None,
    estimate_tokens: TokenEstimator | None = None,
) -> dict[str, Any] | None:
    """Emit ONE structured log entry for a compaction event (AC1/AC2).

    Every compaction event — summary path, backstop, dry-run advisory,
    remote fallback — is logged with the full field set (no silent drops):

    - session: Session ID (truncated per codebase convention)
    - mode: "fast" | "cheap"
    - action / reason: the plan decision
    - pre_tokens / post_tokens: token estimates around the event
    - turns_summarized / turns_dropped: turn accounting
    - summary_tokens: summary length in the caller's token units
    - dry_run: advisory mode flag (true/false)

    Non-events (``action="noop"``) are NOT compaction events and emit
    nothing — even in dry-run, a below-trigger session produces no
    advisory noise.

    Args:
        plan_result: A ``plan_session_compaction`` result dict.
        session_id: The session ID associated with the event.
        dry_run: Advisory (dry-run) mode flag.
        logger_obj: Logger to emit on (defaults to this module's logger).
        estimate_tokens: Estimator used to price the summary (defaults to
            the module heuristic).

    Returns:
        The emitted field dict (None when nothing was logged). Callers
        (e.g. the dry-run child) can collect the same structure for
        churn statistics without re-parsing the log.
    """
    action = plan_result.get("action")
    est = estimate_tokens or estimate_session_tokens
    if action in (None, "noop"):
        return None

    summary_text = plan_result.get("summary_text")
    summary_tokens = 0
    if summary_text:
        summary_tokens = int(est([{"role": _USER_ROLE, "content": summary_text}]) or 0)

    fields: dict[str, Any] = {
        "session": str(session_id)[:8],
        "mode": plan_result.get("mode", "fast"),
        "action": action,
        "reason": plan_result.get("reason"),
        "pre_tokens": int(plan_result.get("estimated_before", 0) or 0),
        "post_tokens": int(plan_result.get("estimated_after", 0) or 0),
        "turns_summarized": int(plan_result.get("turns_summarized", 0) or 0),
        "turns_dropped": int(plan_result.get("backstop_dropped_turns", 0) or 0),
        "summary_tokens": int(summary_tokens),
        "dry_run": bool(dry_run),
    }
    line = "compaction_event " + " ".join(f"{k}={v}" for k, v in fields.items())
    emit = logger_obj if logger_obj is not None else logger
    if fields["action"] == "remote_with_guidance" or (
        fields["reason"] in _WARNING_REASONS
    ):
        emit.warning(line)
    else:
        emit.info(line)
    return fields


# ---------------------------------------------------------------------------
# Warn-only dry-run mode (LP-0MTGBPICV003JMXI)
# ---------------------------------------------------------------------------


def is_compaction_dry_run(config: dict) -> bool:
    """True when compaction runs in warn-only advisory mode.

    Reads ``server.compaction_dry_run`` (or flat ``compaction_dry_run``,
    or the spec-style nested ``server.compaction.dry_run``). When enabled
    the proxy must log what WOULD happen (would-summarize / would-drop)
    without changing dispatch (AC1).
    """
    server = config.get("server", {}) if isinstance(config, dict) else {}
    raw = server.get("compaction_dry_run")
    if raw is None:
        raw = config.get("compaction_dry_run")
    if raw is None:
        raw = server.get("compaction", {}).get("dry_run")
    return bool(raw)


class CompactionChurnCollector:
    """Collects compaction churn per session per time window (AC2).

    In-memory, thread-safe telemetry for the operational target
    (< 1 compaction/session/hour). Timestamps come from an injectable
    clock so tests are deterministic; production uses ``time.time``.
    """

    def __init__(
        self, now_fn: Callable[[], float] | None = None
    ) -> None:
        self._lock = threading.Lock()
        self._now = now_fn or time.time
        self._events: dict[str, list[float]] = {}

    def record(self, session_id: str) -> None:
        """Record one compaction event for *session_id*."""
        ts = self._now()
        with self._lock:
            self._events.setdefault(str(session_id), []).append(ts)

    def churn_counts(self, window_seconds: float = 3600.0) -> dict[str, int]:
        """Per-session event counts within the rolling window."""
        cutoff = self._now() - window_seconds
        with self._lock:
            return {
                sid: sum(1 for ts in stamps if ts > cutoff)
                for sid, stamps in self._events.items()
            }

    def churn_report(
        self, window_seconds: float = 3600.0, target_rate: float = 1.0
    ) -> dict[str, dict[str, Any]]:
        """Per-session churn stats: count, rate/hour, target breach."""
        counts = self.churn_counts(window_seconds)
        hours = max(window_seconds / 3600.0, 1e-9)
        # Target is churn < 1 compaction/session/hour: exactly 1.0/hour
        # does not breach it.
        return {
            sid: {
                "count": n,
                "rate_per_hour": round(n / hours, 3),
                "exceeds_target": (n / hours) > target_rate,
            }
            for sid, n in counts.items()
        }

    def log_churn_report(
        self,
        window_seconds: float = 3600.0,
        target_rate: float = 1.0,
        logger_obj: logging.Logger | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Emit the churn report as structured lines; return the report."""
        report = self.churn_report(window_seconds, target_rate)
        emit = logger_obj if logger_obj is not None else logger
        for sid, stats in report.items():
            emit.warning(
                "compaction_churn session=%s count=%d rate_per_hour=%.3f "
                "exceeds_target=%s",
                str(sid)[:8],
                stats["count"],
                stats["rate_per_hour"],
                stats["exceeds_target"],
            )
        return report


def run_dry_run_plan(
    messages: list[dict[str, Any]],
    config: dict,
    mode: str = "fast",
    summarizer: Summarizer | None = None,
    estimate_tokens: TokenEstimator | None = None,
    session_id: str = "",
    logger_obj: logging.Logger | None = None,
) -> dict[str, Any]:
    """Compute and log the advisory compaction plan WITHOUT dispatch change.

    Dry-run mode (AC1): plans what WOULD happen — summarization and, when
    needed, the truncate backstop (``backstop=True``) so would-drop is
    surfaced — emits the advisory event (``dry_run=true``), and returns the
    plan. The session history is never mutated and never swapped: the
    dispatcher ignores the plan's ``messages`` entirely.

    Returns the plan result dict (see ``plan_session_compaction``).
    """
    plan = plan_session_compaction(
        messages,
        config,
        mode,
        summarizer=summarizer,
        estimate_tokens=estimate_tokens,
        backstop=True,
    )
    log_compaction_event(
        plan,
        session_id=session_id,
        dry_run=True,
        logger_obj=logger_obj,
        estimate_tokens=estimate_tokens,
    )
    return plan


def decide_session_compaction(
    messages: list[dict[str, Any]],
    config: dict,
    mode: str = "fast",
    summarizer: Summarizer | None = None,
    estimate_tokens: TokenEstimator | None = None,
    session_id: str = "",
    dry_run: bool | None = None,
    churn_collector: CompactionChurnCollector | None = None,
    logger_obj: logging.Logger | None = None,
) -> dict[str, Any]:
    """Full prompt-assembly-time dispatch decision (integration child).

    End-to-end flow (parent LP-0MTCWE8NG003P0SD test matrix):

    1. Trigger check via ``plan_session_compaction`` (fires when the
       session estimate exceeds the per-mode trigger; 0 disables).
    2. *Dry-run mode* (``compaction_dry_run`` config, default): log the
       advisory event (would-summarize / would-drop) + record churn,
       apply NOTHING — ``messages`` stays the exact same list object
       (AC3/AC8: zero dispatch change in warn-only phase).
    3. *Live mode* (opt-in): summarize + backstop the session history,
       emit the structured event + churn, and hand the dispatcher the
       compacted ``messages``. ``remote_with_guidance`` means the session
       cannot be compacted — the dispatcher must route remote WITH
       guidance, never near-full-slot local, never silently.

    Args:
        messages: The session's message history (to be dispatched).
        config: Proxy configuration (flat or nested ``server`` dict).
        mode: "fast" (≤38K target) or "cheap" (≤30K target).
        summarizer: Callable(middle_messages) -> summary text; None marks
            the summarizer unavailable (non-compactable).
        estimate_tokens: Estimator over a full message list (production
            passes the routing estimator).
        session_id: Session ID for this decision.
        dry_run: Override the config flag (None = read config).
        churn_collector: Optional collector for <1/session/hour telemetry.
        logger_obj: Logger for the structured event (defaults to module
            logger).

    Returns:
        The plan dict plus two dispatch fields:

        - ``dry_run``: whether advisory mode was in effect.
        - ``applied``: True only when ``messages`` has been compacted and
          the dispatcher MUST use the compacted history (live mode,
          action == "compact"). Never True in dry-run or for
          ``remote_with_guidance``/``noop``.
    """
    dry_run = is_compaction_dry_run(config) if dry_run is None else dry_run

    if dry_run:
        # Warn-only advisory: log what WOULD happen, never apply.
        plan = run_dry_run_plan(
            messages, config, mode,
            summarizer=summarizer, estimate_tokens=estimate_tokens,
            session_id=session_id, logger_obj=logger_obj,
        )
        plan["dry_run"] = True
        plan["applied"] = False
        plan["messages"] = messages
        if churn_collector is not None and plan["action"] != "noop":
            churn_collector.record(session_id)
        return plan

    # Live enforcement path (opt-in after the AC8 experiment gate).
    plan = plan_session_compaction(
        messages, config, mode,
        summarizer=summarizer, estimate_tokens=estimate_tokens,
        backstop=True,
    )
    log_compaction_event(
        plan, session_id=session_id, dry_run=False,
        logger_obj=logger_obj, estimate_tokens=estimate_tokens,
    )
    if churn_collector is not None and plan["action"] != "noop":
        churn_collector.record(session_id)
    plan["dry_run"] = False
    plan["applied"] = plan["action"] == "compact"
    return plan


def plan_session_compaction(
    messages: list[dict[str, Any]],
    config: dict,
    mode: str = "fast",
    summarizer: Summarizer | None = None,
    estimate_tokens: TokenEstimator | None = None,
    backstop: bool = False,
) -> dict[str, Any]:
    """Plan and produce the compacted message list for a session.

    Strategy C (summarize-first):

    1. When the session estimate is at/below the per-mode trigger, or
       compaction is disabled (trigger 0), the session is untouched
       (``action="noop"``).
    2. Otherwise the system prompt(s) + very first user prompt are retained
       verbatim (retention invariant, AC1); the middle turns are folded into
       a summary injected immediately below the first prompt (AC5); whole
       recent turns are kept while the total estimate stays within the
       per-mode target.
    3. If no summarizer is available the session cannot be compacted — the
       result is ``action="remote_with_guidance"`` so the dispatcher routes
       remote WITH guidance, never silently (AC4).

    Args:
        messages: The session's message list (OpenAI-style role/content).
        config: Proxy configuration (flat or nested ``server`` dict).
        mode: "fast" (≤38K target) or "cheap" (≤30K target).
        summarizer: Callable(middle_messages) -> summary text. None marks
            the summarizer as unavailable (non-compactable).
        estimate_tokens: Estimator over a full message list; defaults to
            the module heuristic (production passes the routing estimator).
        backstop: When True, chain ``truncate_backstop`` when the summary
            path alone still leaves the session over budget: the oldest
            whole recent turns are dropped and the reason becomes
            ``backstop_dropped`` / ``backstop_exhausted``.

    Returns:
        A dict with the compaction decision:

        - action: "noop" | "compact" | "remote_with_guidance"
        - messages: original list (noop / compactable-missing) or the
          compacted list
        - summary_text: the summarizer output when compacted, else None
        - turns_summarized / recent_turns_kept: turn accounting
        - estimated_before / estimated_after: token estimates (same
          estimator)
        - trigger_tokens / target_tokens: resolved thresholds
        - reason: machine-readable explanation (see constants below)
    """
    trigger = compaction_trigger_tokens(mode, config)
    target = compaction_target_tokens(mode)
    est = estimate_tokens or estimate_session_tokens

    result: dict[str, Any] = {
        "mode": mode,
        "action": "noop",
        "messages": messages,
        "summary_text": None,
        "turns_summarized": 0,
        "recent_turns_kept": 0,
        "estimated_before": est(messages),
        "estimated_after": est(messages),
        "trigger_tokens": trigger,
        "target_tokens": target,
        "reason": None,
    }

    if trigger <= 0:
        result["reason"] = "compaction_disabled"
        return result
    if result["estimated_before"] <= trigger:
        result["reason"] = "below_trigger"
        return result

    # Retention set: all system prompts + the very first user prompt,
    # verbatim (AC1).
    system_msgs = [m for m in messages if m.get("role") == _SYSTEM_ROLE]
    first_user_idx = next(
        (i for i, m in enumerate(messages) if m.get("role") == _USER_ROLE), None
    )
    if first_user_idx is None:
        result["reason"] = "no_user_message"
        return result
    first_user_msg = messages[first_user_idx]

    turns = pair_turns(messages[first_user_idx + 1 :])
    if not turns:
        # Nothing to fold — a summary of an empty middle adds no value.
        result["reason"] = "no_turns_to_compact"
        return result

    if summarizer is None:
        # AC4: non-compactable — dispatcher must route remote, with
        # guidance. The session is left untouched.
        result["action"] = "remote_with_guidance"
        result["reason"] = "summarizer_unavailable"
        return result

    retained = list(system_msgs) + [first_user_msg]
    placeholder = _summary_message("")

    # Select the recent window: keep whole turns (newest first) while the
    # candidate estimate stays within the target budget. Re-derive each
    # candidate from the retained set so any (non-additive) estimator is
    # respected.
    accepted: list[list[dict[str, Any]]] = []
    for turn in reversed(turns):
        candidate = list(retained) + [placeholder]
        for kept in accepted:
            candidate.extend(kept)
        candidate.extend(turn)
        if est(candidate) <= target:
            accepted.append(turn)
        else:
            break
    recent = accepted[::-1]

    middle_turns = turns[: len(turns) - len(recent)]
    middle_messages = [m for turn in middle_turns for m in turn]
    summary_text = summarizer(middle_messages)

    compacted = (
        list(retained)
        + [_summary_message(summary_text)]
        + [m for turn in recent for m in turn]
    )
    estimated_after = est(compacted)
    over_budget = estimated_after > target

    result.update(
        action="compact",
        messages=compacted,
        summary_text=summary_text,
        turns_summarized=len(middle_turns),
        recent_turns_kept=len(recent),
        estimated_after=estimated_after,
        reason="compacted_over_budget" if over_budget else "compacted_within_target",
    )
    if backstop and result["reason"] == "compacted_over_budget":
        back = truncate_backstop(result["messages"], target, est)
        result["messages"] = back["messages"]
        result["estimated_after"] = back["estimated_after"]
        result["backstop_dropped_turns"] = back["dropped_turns"]
        result["backstop_dropped_messages"] = back["dropped_messages"]
        result["remaining_budget"] = back["remaining_budget"]
        if back["action"] == "dropped":
            result["reason"] = "backstop_dropped"
        else:
            result["reason"] = "backstop_exhausted"
    return result
