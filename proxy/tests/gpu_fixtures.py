"""Shared ``/slots`` payload builders for GPU-telemetry tests.

LP-0MVT1NVAJ007KUGO (parent LP-0MVNWUEY8008XM6W). These are **builders, not
golden blobs**: every helper returns a freshly built payload, so a test case is
one keyword argument away and a builder change cannot silently invalidate a
dozen recorded blobs.

Two payload shapes exist in the wild and both must be covered:

* **lean** — what the deployed llama.cpp build answers today (verified
  2026-10-29 with ``GET :8080/slots?model=mxbai-embed``)::

      [{"id": 0, "n_ctx": 256, "speculative": false, "is_processing": false}, ...]

  No ``id_task`` and no ``next_token`` are published, even while processing.

* **rich** — older/other builds carry ``id_task`` (``-1`` when the slot has no
  task) and ``next_token``, which is what the sampler sibling
  (LP-0MVT1O0VG00905EL) prefers when counting in-flight tasks, falling back to
  ``is_processing`` when ``id_task`` is absent.

Failure modes are modelled as **clients**, not payloads, because that is what
the consumer sees: ``_fetch_slots_payload()`` returns
``(status_code, parsed)`` for a served response and *raises* for a transport
failure (timeout, connection error). A fixture that returned a payload for a
timeout would let a "never publish zero busy slots on failure" regression pass.

Consumers (do not re-invent these):

* ``proxy.proxy.observability._fetch_slots_payload`` — ``(status, data)``, raises
  on transport failure.
* ``proxy.proxy.observability._query_slots`` — ``(available, total)`` plus
  ``status_out["reason"]`` in ``http_<code>`` / ``timeout`` /
  ``connection_error`` / ``invalid_payload`` / ``unknown``.
* ``proxy.proxy.observability._query_slots_detail`` — ``[{"slot_id",
  "is_processing", "n_decoded", "context_tokens"}]``; ``[]`` on failure *and* on
  an empty backend, which is why a sampler must use ``_fetch_slots_payload``
  and check the status rather than trusting ``[]``.
"""

from __future__ import annotations

import asyncio
import json

#: Keys the deployed (lean) build publishes for every slot.
LEAN_SLOT_KEYS = ("id", "n_ctx", "speculative", "is_processing")

#: Keys only richer builds publish, used for the in-flight task count.
RICH_SLOT_KEYS = (*LEAN_SLOT_KEYS, "id_task", "next_token")


# ---------------------------------------------------------------------------
# /slots payload builders
# ---------------------------------------------------------------------------


def slot(
    slot_id: int,
    *,
    processing: bool = False,
    rich: bool = False,
    id_task: int = -1,
    next_token: dict | None = None,
    n_ctx: int = 256,
) -> dict:
    """Build one ``/slots`` entry.

    ``rich=False`` reproduces the deployed build exactly (no ``id_task``, no
    ``next_token``); ``rich=True`` adds both. ``id_task`` defaults to ``-1``
    ("no task assigned"), which is what a rich build reports for an idle slot.
    """
    entry = {
        "id": slot_id,
        "n_ctx": n_ctx,
        "speculative": False,
        "is_processing": processing,
    }
    if rich:
        entry["id_task"] = id_task
        entry["next_token"] = (
            next_token
            if next_token is not None
            else ({"has_next_token": True, "n_remain": 12, "n_decoded": 5} if processing
                  else {"has_next_token": False, "n_remain": -1, "n_decoded": 0})
        )
    return entry


def slots_payload(*, total: int = 4, busy_ids: tuple[int, ...] = (), rich: bool = False) -> list[dict]:
    """Build a ``/slots`` body where *busy_ids* are processing.

    The healthy case for ``total=4, busy_ids=(0, 2)`` is two busy and two idle
    slots; a rich build additionally assigns ``id_task`` to each busy slot.
    """
    unknown = [i for i in busy_ids if not 0 <= i < total]
    if unknown:
        raise ValueError(f"busy_ids {unknown} outside range(total={total})")
    entries = []
    for slot_id in range(total):
        processing = slot_id in busy_ids
        entries.append(
            slot(
                slot_id,
                processing=processing,
                rich=rich,
                # Rich builds report a real task id (>= 0) only for a busy slot.
                id_task=(1000 + slot_id) if processing else -1,
            )
        )
    return entries


def mixed_slots_payload(*, total: int = 4, busy_ids: tuple[int, ...] = (0, 2)) -> list[dict]:
    """Healthy lean payload with mixed ``is_processing`` — the common case."""
    return slots_payload(total=total, busy_ids=busy_ids)


def empty_slots_payload() -> list[dict]:
    """A backend that answered truthfully and has no slots.

    Distinguishable from a failure only through the HTTP status: the consumer
    must see ``(200, [])``, never a raised exception, or "no slots" becomes
    indistinguishable from "we could not look".
    """
    return []


def rich_slots_payload() -> list[dict]:
    """Richer payload carrying ``id_task``/``next_token`` (two busy of four)."""
    return slots_payload(total=4, busy_ids=(1, 3), rich=True)


def expected_in_flight_tasks(payload: list[dict]) -> int:
    """In-flight task count a consumer must derive from *payload*.

    ``len([s for s in payload if s.get("id_task", -1) >= 0])`` when the build
    publishes ``id_task``, else the ``is_processing`` count (the documented
    fallback for the deployed lean build). Kept next to the builders so a test
    asserts the *contract* rather than re-deriving it inline.
    """
    if any("id_task" in entry for entry in payload):
        return sum(1 for entry in payload if entry.get("id_task", -1) >= 0)
    return sum(1 for entry in payload if entry.get("is_processing") is True)


def expected_busy_slots(payload: list[dict]) -> int:
    """Busy-slot count (``is_processing`` true) a consumer must derive."""
    return sum(1 for entry in payload if entry.get("is_processing") is True)


# ---------------------------------------------------------------------------
# Fake HTTP clients — one per fetch outcome
# ---------------------------------------------------------------------------


class FakeSlotsResponse:
    """Minimal stand-in for an ``httpx.Response`` from ``/slots``."""

    def __init__(self, status_code: int, data) -> None:
        self.status_code = status_code
        self._data = data
        self.text = json.dumps(data)

    async def json(self):
        return self._data


class FakeSlotsClient:
    """Serve *data* with *status_code*; optionally delay past the timeout."""

    def __init__(self, data, status_code: int = 200, delay: float = 0.0) -> None:
        self.data = data
        self.status_code = status_code
        self.delay = delay
        self.calls: list[str] = []

    async def get(self, url):
        self.calls.append(url)
        if self.delay:
            await asyncio.sleep(self.delay)
        return FakeSlotsResponse(self.status_code, self.data)


class FailingSlotsClient:
    """Raise *exc* from ``get()``, modelling a transport failure."""

    def __init__(self, exc: BaseException | None = None) -> None:
        self.exc = exc if exc is not None else ConnectionError("connection refused")
        self.calls: list[str] = []

    async def get(self, url):
        self.calls.append(url)
        raise self.exc


#: Failure modes a sampler must survive without publishing a zero busy count.
FAILURE_MODES = ("http_500", "timeout", "connection_error")


def failing_slots_client(mode: str, *, timeout: float = 0.05) -> object:
    """Return the client that reproduces failure *mode*.

    * ``http_500`` — llama-server answers 500 while the model reloads.
    * ``timeout`` — the request exceeds its budget (``_do_slots_fetch`` wraps
      the call in ``asyncio.wait_for``, so a slow client surfaces as
      ``TimeoutError``; pass the consumer's timeout budget as *timeout*).
    * ``connection_error`` — the server is not listening at all.
    """
    if mode == "http_500":
        return FakeSlotsClient(None, status_code=500)
    if mode == "timeout":
        # Delay far beyond any sane budget; the caller's timeout raises.
        return FakeSlotsClient(None, status_code=200, delay=timeout * 20)
    if mode == "connection_error":
        return FailingSlotsClient(ConnectionError("connection refused"))
    raise ValueError(f"unknown failure mode {mode!r}; expected one of {FAILURE_MODES}")


def successful_slots_client(payload: list[dict] | None = None) -> FakeSlotsClient:
    """Serve a healthy payload with HTTP 200."""
    return FakeSlotsClient(payload if payload is not None else mixed_slots_payload())
