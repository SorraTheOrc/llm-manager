"""Retry-outcome correlation for post-acceptance aborts (LP-0MWM0CPQN009SHJR).

A post-acceptance abort (``local_eviction_after_acceptance``) tells an operator
that an admitted local dispatch was released before its first byte. It does
**not** say what the caller then experienced: a 15 s abort followed by an
immediate successful retry is cheap to the user, whereas the same abort leaving
the request unanswered is expensive. This module is the emission half of the
join that answers it — the analysis half lives in
``proxy/scripts/analyse_prefill_deadline_residual.py`` (``retry_join``).

The correlation key
-------------------

An abort and the caller's next dispatch are joined on **client/session id plus
prompt identity**:

- ``session`` — the same value the eviction line already carries;
- ``prompt_fp`` — the SHA-256 of the request's ``messages`` list (canonical
  JSON), the same digest the session single-flight coordinator already uses
  (:func:`proxy.router_helpers._request_dedup_hash`). See
  :data:`PROMPT_TOLERANCE_NOTE` for the "same prompt" tolerance.

``turn`` (per-request correlation id, :mod:`proxy.turn_id`) is carried as extra
provenance but is **not** the join key: a retry is a new turn (or, inside the
provider chain, an in-turn re-dispatch), so the analyser orders by timestamp and
groups by turn only to pick a single turn's terminal outcome.

This is a **measurement-only** surface (AC4): every function here is wrapped so
it can never raise into a request path, and it neither reads nor mutates an
admission decision, an abort policy or a threshold. The diff is logging plus
analysis.

Log grammar
-----------

::

    local_dispatch_outcome session=S prompt_fp=F outcome=O endpoint=E \
        [dispatch_token=T] [<caller evidence>] [turn=X]

``outcome`` is one of :data:`OUTCOMES`:

- ``local_accepted`` — a local dispatch was admitted for this request;
- ``local_refused`` — a local dispatch was refused (gate, lease, capacity);
- ``remote`` — the request was served by a remote provider.

A single client request can emit several of these lines as the provider chain
moves (e.g. ``local_refused`` then ``remote``); the analyser resolves the
terminal outcome per turn with :data:`OUTCOME_PRECEDENCE`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

#: A local dispatch was admitted for the caller's request.
OUTCOME_LOCAL_ACCEPTED = "local_accepted"

#: Local was refused (gate / lease / capacity) and the caller did not get a
#: local dispatch from this attempt.
OUTCOME_LOCAL_REFUSED = "local_refused"

#: The caller's request was served by a remote provider.
OUTCOME_REMOTE = "remote"

#: The closed outcome vocabulary the analyser understands. ``none`` (no retry
#: observed in the window) is an analysis result, never an emitted outcome.
OUTCOMES: tuple[str, ...] = (
    OUTCOME_LOCAL_ACCEPTED,
    OUTCOME_LOCAL_REFUSED,
    OUTCOME_REMOTE,
)

#: When one turn emits several outcome lines (a refusal that then escalates
#: remotely), the terminal outcome is the first name present, in this order.
OUTCOME_PRECEDENCE: tuple[str, ...] = (
    OUTCOME_REMOTE,
    OUTCOME_LOCAL_ACCEPTED,
    OUTCOME_LOCAL_REFUSED,
)

#: The "same prompt" tolerance, stated verbatim for the docs and the analyser.
PROMPT_TOLERANCE_NOTE = (
    "Two dispatches carry the 'same prompt' when their ``prompt_fp`` values are "
    "equal. The fingerprint is the SHA-256 of the request's ``messages`` list "
    "serialised as canonical JSON, so sampling/generation parameters are ignored "
    "and only a changed message list (new turn content) changes the value. The "
    "join cannot cross a client restart or an anonymous session without a stable "
    "``X-Session-Id``, because the session id is the join key's other half."
)

#: The eviction/outcome fields the analysis reads (all appended by this change).
RETRY_JOIN_SOURCE_FIELDS: tuple[str, ...] = (
    "session",
    "prompt_fp",
    "outcome",
    "turn",
    "elapsed_s",
)

#: Coverage caveat recorded beside every retry-join block.
RETRY_JOIN_CAVEAT = (
    "The join is log-only and cross-turn: an abort is matched to the next "
    "dispatch of the same session (by timestamp), not to a request the client "
    "identified. It therefore cannot correlate across a client restart that "
    "changes the session id, nor across sessions. A retry that lands outside the "
    "analysed window is reported as never-completed (``none``), not as escaped."
)

logger = logging.getLogger("llama-proxy")


def prompt_fingerprint(body_json: dict | None) -> str | None:
    """Return the stable prompt identity for a request body (or ``None``).

    Delegates to :func:`proxy.router_helpers._request_dedup_hash` so the retry
    join and the single-flight dedup coordinator agree on what "the same
    prompt" means. Never raises: a body without a usable ``messages`` list
    returns ``None`` and the line renders ``prompt_fp=unknown``.
    """
    try:
        from proxy.router_helpers import _request_dedup_hash

        if not isinstance(body_json, dict):
            return None
        return _request_dedup_hash(body_json)
    except Exception:
        return None


def stamp_prompt_fingerprint(request, body_json: dict | None) -> str | None:
    """Compute and stamp the prompt fingerprint on *request* (best-effort).

    Called at the earliest point the client's **original** body is available
    (before session-delta rewriting), so every later reader — the dispatch
    record and every ``local_dispatch_outcome`` line — uses one value.
    """
    value = prompt_fingerprint(body_json)
    try:
        if request is not None and value is not None:
            request._prompt_fp = value
    except Exception:
        pass
    return value


def request_prompt_fingerprint(request, body_json: dict | None = None) -> str | None:
    """Prefer the fingerprint stamped on *request*, else recompute it."""
    try:
        stamped = getattr(request, "_prompt_fp", None) if request is not None else None
    except Exception:
        stamped = None
    if stamped:
        return stamped
    return prompt_fingerprint(body_json)


def _render_value(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def record_dispatch_outcome(
    srv,
    session_id: str | None,
    prompt_fp: str | None,
    outcome: str,
    *,
    endpoint: str | None = None,
    dispatch_token: int | None = None,
    extra: Mapping[str, object] | None = None,
) -> bool:
    """Emit one ``local_dispatch_outcome`` line (never raises).

    Args:
        srv: Server object (its ``logger`` when present); may be ``None``.
        session_id: The client/session id (the join key's other half).
        prompt_fp: The prompt fingerprint from
            :func:`request_prompt_fingerprint` (``None`` renders ``unknown``).
        outcome: One of :data:`OUTCOMES`.
        endpoint: The dispatch endpoint scope, when known.
        dispatch_token: The per-dispatch identity token, when known.
        extra: Optional trailing ``key=value`` evidence.

    Returns:
        ``True`` always (the call is never gated); a logging failure is
        swallowed so observability cannot break a request path.
    """
    try:
        parts = [
            f"local_dispatch_outcome session={session_id or 'unknown'}",
            f"prompt_fp={prompt_fp or 'unknown'}",
            f"outcome={outcome}",
        ]
        if endpoint is not None:
            parts.append(f"endpoint={endpoint or 'unknown'}")
        if dispatch_token is not None:
            parts.append(f"dispatch_token={_render_value(dispatch_token)}")
        if extra:
            for key, value in extra.items():
                parts.append(f"{key}={_render_value(value)}")
        line = " ".join(parts)
        try:
            from proxy.turn_id import turn_field

            line += turn_field()
        except Exception:
            pass
        target = getattr(srv, "logger", None) or logger
        try:
            target.info(line)
        except Exception:
            pass
    except Exception:
        pass
    return True
