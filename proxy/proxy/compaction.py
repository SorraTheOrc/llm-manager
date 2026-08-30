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
