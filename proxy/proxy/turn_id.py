"""Per-turn correlation id for structured proxy log lines (LP-0MVMBM467005BT4A).

One client request ("turn") leaves a trail of structured log lines that were
previously tied together only by ``session=`` plus timestamp ordering:
``routing_skip_local`` -> ``Fallback triggered`` xN -> ``Stream started`` ->
``Stream finished``. The usage analysis compensated with heuristics (FIFO
stream pairing, 60 s fallback windows, nearest-attribution) that mis-assign
reasons when two turns of the same session interleave.

This module stamps a short (8 hex character) identifier at the earliest point a
request is known and carries it on every structured line the turn emits, so a
request's whole lifecycle can be reconstructed exactly instead of by
heuristics.

Design
------
* The id is generated once per client request and **never** re-generated
  inside the request, so it is stable across fallback hops, chain-hold
  retries, and local/remote module boundaries (AC2).
* It is held in a :class:`contextvars.ContextVar`, which is copied — not
  shared — by every ``asyncio`` task created during the request. Uvicorn
  creates one task per request, so two turns can never observe each other's
  id: the id is unique per turn of a session (AC2).
* It is mirrored onto ``request.state.turn_id`` so code holding the request
  object can read it without relying on the context.
* Absence is explicitly supported: helpers return ``""``/``None`` when no turn
  is bound (direct helper calls in unit tests, offline tools), so the emitted
  lines stay byte-compatible with pre-change logs and every legacy consumer
  keeps parsing them (backwards compatibility, AC4).

Usage
-----
Bind at a request entry point (idempotent — a nested call keeps the id)::

    from proxy.turn_id import bind_turn_id, turn_field

    turn = bind_turn_id(request)

Append to any structured line, without needing the request object::

    logger.info("routing_skip_local ... session=%s%s", session, turn_field())
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

#: Log field name carrying the per-turn correlation id.
TURN_FIELD = "turn"

#: Length of a generated id (hex characters). Short enough to keep log lines
#: readable, long enough that a per-session collision is negligible (32 bits,
#: i.e. ~1 in 77 for 1000 turns of one session).
TURN_ID_LENGTH = 8

_TURN_CTX: ContextVar[str | None] = ContextVar("llama_proxy_turn_id", default=None)


def new_turn_id() -> str:
    """Return a fresh, unused per-turn correlation id (8 hex characters)."""
    return uuid.uuid4().hex[:TURN_ID_LENGTH]


def bind_turn_id(request=None) -> str:
    """Return the turn id for the current request, generating it if needed.

    Idempotent: the first call in a request context creates the id, every
    later call (including from ``proxy_to_local``/``proxy_to_remote`` inside a
    fallback chain, and from a chain-hold retry cycle) returns the same value,
    which is what makes the id stable across the whole turn (AC2).

    When *request* is given the id is also mirrored onto
    ``request.state.turn_id`` so request-aware code can read it back with
    :func:`get_turn_id`. Mirrors and ContextVar writes are fail-open: a
    request object without a writable ``state`` (a mock, or a non-ASGI
    request) still gets a returned id and never raises into the request path.
    """
    turn = _TURN_CTX.get()
    if turn is None:
        turn = new_turn_id()
        _TURN_CTX.set(turn)
    if request is not None:
        try:
            request.state.turn_id = turn
        except Exception:  # pragma: no cover - defensive (non-ASGI request)
            pass
    return turn


def get_turn_id(request=None) -> str | None:
    """Return the current turn id, or ``None`` when no turn is bound.

    The request's ``state`` (when it carries one) wins over the context so a
    line logged from a task that lost the context — e.g. a reaper task holding
    the request object — still attributes correctly.
    """
    if request is not None:
        try:
            turn = getattr(request.state, "turn_id", None)
        except Exception:  # pragma: no cover - defensive (non-ASGI request)
            turn = None
        if turn:
            return str(turn)
    return _TURN_CTX.get()


def turn_field(request=None) -> str:
    """Return ``" turn=<id>"`` for appending to a log line, or ``""``.

    Returning an empty string (rather than ``turn=none``) when no turn is
    bound keeps lines emitted outside a request byte-compatible with the
    pre-change format, so rotated-log consumers keep working unchanged.
    """
    turn = get_turn_id(request)
    if not turn:
        return ""
    return f" {TURN_FIELD}={turn}"


def reset_turn_id() -> None:
    """Clear the bound turn id in the current context.

    Used by tests (and by any code that deliberately ends a turn early). In
    production the context is per-request, so nothing has to clear it.
    """
    _TURN_CTX.set(None)


__all__ = [
    "TURN_FIELD",
    "TURN_ID_LENGTH",
    "bind_turn_id",
    "get_turn_id",
    "new_turn_id",
    "reset_turn_id",
    "turn_field",
]
