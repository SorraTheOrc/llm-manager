"""b8782 ``prompt_progress`` contract: extraction and per-dispatch capture.

F1 harness (LP-0MVR7CR34000AG12) for the capture half of Option D
(LP-0MVQIT9QY007QU27): llama-server b8782 (build ``e97492369``) emits
``prompt_progress`` objects inside the streaming SSE response when the
upstream request carries ``return_progress: true``. This module pins that
shape to the real b8782 output so the capture implementation
(LP-0MVR7DAFF006HBRX: ``router.proxy_to_local`` injection + reframed-loop
parse) and its consumers (early abort, usage analysis) share one contract.

Verified against the b8782 sources (``/home/rgardler/llama.cpp``, HEAD
``e97492369``):

* ``result_prompt_progress::to_json()`` (``tools/server/server-task.cpp``)
  emits exactly ``{"total", "cache", "processed", "time_ms"}``.
* ``server_context::send_partial_response(slot, {}, true)``
  (``tools/server/server-context.cpp``) sets
  ``total = task->n_tokens()``, ``cache = slot.n_prompt_tokens_cache``,
  ``processed = slot.prompt.tokens.size()`` and
  ``time_ms = (ggml_time_us() - t_start_process_prompt) / 1000``.
* The initial 0% event is sent right after
  ``slot.n_prompt_tokens_cache = n_past`` (``server-context.cpp``), so
  ``cache`` on the *first* event is the exact reused prefix and
  ``processed``/``time_ms`` are 0.
* On the OAI-compatible chat path the object is added to the last delta
  chunk (``to_json_oaicompat_chat``) and written by ``format_oai_sse`` as a
  single ``data: <json>\\n\\n`` event, i.e. ``prompt_progress`` is a
  top-level key next to ``choices``/``timings`` — never nested in
  ``choices``.
* C++ types are ``int32_t`` for ``total``/``cache``/``processed`` and
  ``int64_t`` for ``time_ms``, so JSON carries integers.

Fail-open by contract: parsing never raises. A stream without
``prompt_progress`` events (``return_progress`` dropped upstream), a
malformed event, or a wrongly typed field degrades to "no measurement"
(``None``) or an explicit ``None`` for that single field — the capture layer
must never change dispatch behaviour (AC4).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

#: The four numeric fields b8782 emits, in ``to_json()`` order.
PROMPT_PROGRESS_FIELDS: tuple[str, ...] = ("total", "cache", "processed", "time_ms")

_SSE_DATA_PREFIX = "data:"


def _payload_text(event: bytes | bytearray | memoryview | str) -> str | None:
    """Return the JSON payload text carried by a single reframed SSE event.

    ``SSEEventReframer`` emits raw events with their ``data:`` prefix and
    terminating blank line, so the payload is the first ``data:`` line. A
    bare JSON payload (no ``data:`` line) is accepted too, which keeps the
    extractor usable on already-unwrapped chunk bodies.
    """
    if isinstance(event, (bytes, bytearray, memoryview)):
        try:
            text = bytes(event).decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            return None
    elif isinstance(event, str):
        text = event
    else:
        return None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(_SSE_DATA_PREFIX):
            return stripped[len(_SSE_DATA_PREFIX) :].strip() or None
    return text.strip() or None


def _as_token_count(value: object) -> int | None:
    """Coerce a token-count field to ``int``, or ``None`` when not integral."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        return int(value)
    return value


def _as_time_ms(value: object) -> float | None:
    """Coerce ``time_ms`` to a finite ``float``, or ``None`` when invalid."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return float(value)


@dataclass(frozen=True)
class PromptProgress:
    """One b8782 ``prompt_progress`` observation.

    Every field is ``None`` when absent (or not a valid number) in the
    upstream event: missing fields carry explicit "absent" semantics rather
    than a default that could be mistaken for a measurement.

    ``measured_cached_ratio`` and ``measured_new_tokens`` are the derived
    signals the early-abort decision (F3/F4) consumes; both are ``None``
    when the underlying fields are unavailable.
    """

    total: int | None
    cache: int | None
    processed: int | None
    time_ms: float | None

    @property
    def measured_cached_ratio(self) -> float | None:
        """``cache / total`` — the measured KV-cache reuse ratio."""
        if self.total is None or self.cache is None or self.total <= 0:
            return None
        return self.cache / self.total

    @property
    def measured_new_tokens(self) -> int | None:
        """``total - cache`` — tokens that actually had to be computed."""
        if self.total is None or self.cache is None:
            return None
        return self.total - self.cache


def parse_prompt_progress_event(
    event: bytes | bytearray | memoryview | str,
) -> PromptProgress | None:
    """Extract ``prompt_progress`` from one reframed SSE event.

    Args:
        event: A single event as emitted by
            :class:`proxy.utils.SSEEventReframer` (``data: <json>`` plus its
            blank-line terminator, ``\\n\\n`` or ``\\r\\n\\r\\n``), or a bare
            JSON chunk payload.

    Returns:
        A :class:`PromptProgress` when the event carries a
        ``prompt_progress`` object; ``None`` when it carries none (a normal
        content chunk) or the event cannot be understood. Individual fields
        that are absent or not numbers are ``None`` on the returned object.

    Never raises: malformed bytes, non-UTF-8 data, invalid JSON and
    non-object payloads all return ``None`` so the capture layer stays
    fail-open (AC4).
    """
    payload = _payload_text(event)
    if payload is None:
        return None
    try:
        chunk = json.loads(payload)
    except (TypeError, ValueError):
        return None
    if not isinstance(chunk, dict):
        return None
    progress = chunk.get("prompt_progress")
    if not isinstance(progress, dict):
        return None
    return PromptProgress(
        total=_as_token_count(progress.get("total")),
        cache=_as_token_count(progress.get("cache")),
        processed=_as_token_count(progress.get("processed")),
        time_ms=_as_time_ms(progress.get("time_ms")),
    )


class PromptProgressCapture:
    """Per-dispatch capture of the latest b8782 ``prompt_progress`` event.

    Feed every reframed event of a dispatch's SSE stream to :meth:`observe`;
    it returns the observed :class:`PromptProgress` (or ``None`` when the
    event carries no progress) and retains the latest measurement. Attach
    point the capture implementation (F2) uses in the proxy's reframed loop.

    Fail-open by construction: :func:`parse_prompt_progress_event` never
    raises, so :meth:`observe` cannot raise or change the bytes delivered to
    the client. A stream with no progress events leaves the capture in its
    initial "no measurement" state.
    """

    def __init__(self) -> None:
        self._latest: PromptProgress | None = None
        self._events_seen = 0

    @property
    def latest(self) -> PromptProgress | None:
        """The most recently observed progress, or ``None`` if none seen."""
        return self._latest

    @property
    def observed(self) -> bool:
        """Whether at least one ``prompt_progress`` event was seen."""
        return self._latest is not None

    @property
    def events_seen(self) -> int:
        """How many ``prompt_progress`` events have been observed."""
        return self._events_seen

    def observe(
        self, event: bytes | bytearray | memoryview | str
    ) -> PromptProgress | None:
        """Observe one reframed event and record it when it carries progress."""
        progress = parse_prompt_progress_event(event)
        if progress is not None:
            self._latest = progress
            self._events_seen += 1
        return progress
