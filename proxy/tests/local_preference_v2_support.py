"""Shared harness for the vocabulary-v2 / bounded-deferral contract specs.

LP-0MVUH9OMY004MQ3U (child of LP-0MVU1GSDY003EJ0O: parent AC4, AC9, AC11, AC21).

Test-first scaffolding. The three spec modules

- :mod:`tests.test_local_preference_vocabulary_v2` (retired values, vocabulary
  shape, docs contract),
- :mod:`tests.test_local_deferral_budget` (the bounded refusal budget), and
- :mod:`tests.test_remote_ok_never_waits` (the redefined ``remote-ok``)

all drive the **same** fake-transport harness the refusal matrix already owns
(:mod:`tests.test_local_only_refusal_matrix`). Nothing here is a second test
framework: the exit fixtures (``_EXIT_ROWS`` / ``_ExitRow``), their patches
(``_row_patches``), the model/config builders and the fake upstreams are
imported from that module and reused verbatim. This module only adds the three
capabilities the new contract needs and the existing rows never did:

1. **A request builder that can carry a session id** — the deferral budget is
   keyed per session, and retries arrive as *separate* HTTP requests, so a row
   needs to present the same ``X-Session-ID`` across several issues.
2. **A deterministic clock / TTL knob** for the refusal counter, so "the entry
   exceeded its TTL" is asserted without sleeping.
3. **Zero-side-effect assertions** (no upstream dispatch, no queue entry, no
   lease) so "rejected before any routing state is written" is one call site
   instead of three copied ones.

Pre-agreed testability surface
------------------------------

The implementing children own the internals, but these names are the contract
this suite drives (documented here so a later child does not have to reverse
engineer it from the specs):

``proxy.local_deferral`` (new module, LP-0MVUH9PL0005PW90)
    ``TTL_SECONDS`` — entry lifetime in seconds.
    ``MAX_ENTRIES`` — the bounded store cap.
    ``_now()`` — the module clock hook (patched by :func:`frozen_clock`).
    ``record_refusal(session_id) -> int`` — count a retryable refusal.
    ``refusal_count(session_id) -> int`` — current consecutive count.
    ``reset(session_id)`` — drop one session's counter.
    ``reset_all()`` — drop every counter (test hygiene).

``proxy.admission_reserve`` (LP-0MVUH9PAZ002UAPA / LP-0MVUH9PUU005HVS4)
    ``VALID_PREFERENCES == ("local-preferred", "remote-ok")``,
    ``DEFAULT_PREFERENCE == "remote-ok"``, ``LEGACY_PREFERENCE_ALIASES`` without
    ``local-only``, and ``LOCAL_FALLBACK_REASON_TOKENS`` containing
    ``"deferral_refusals_exhausted"``.

Everything asserted below is an *observation* (status, headers, body fields,
upstream dispatches, queue depth, log lines). The specs deliberately do not
pin the mechanism that produces them.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import proxy.provider as provider
import proxy.server as proxy_server
import pytest

from proxy import contention_queue, prefill_eta

# ``proxy/tests`` is on ``sys.path`` (see ``proxy/conftest.py``) and pytest
# imports test modules by their basename, so this reuses the module object the
# matrix tests already use instead of importing the file a second time under a
# different name.
try:  # pragma: no cover - the fallback only runs outside a pytest session
    import test_local_only_refusal_matrix as matrix
except ImportError:  # pragma: no cover
    from tests import test_local_only_refusal_matrix as matrix

# ---------------------------------------------------------------------------
# Vocabulary under test (parent AC14/AC21)
# ---------------------------------------------------------------------------

LOCAL_ONLY = "local-only"
PREFER_LOCAL = "prefer-local"
LOCAL_PREFERRED = "local-preferred"
REMOTE_OK = "remote-ok"

#: The intended two-value live set, in canonical order.
LIVE_PREFERENCES: tuple[str, ...] = (LOCAL_PREFERRED, REMOTE_OK)
#: The intended default (an absent header resolves to it).
DEFAULT_PREFERENCE = REMOTE_OK

#: Stable rejection code for a withdrawn wire value (parent AC1/AC14).
RETIRED_CODE = "local_preference_retired"

#: The escalation reason token / body-field names / config key the bounded
#: deferral contract adds (parent AC6/AC7/AC10).
ESCALATION_TOKEN = "deferral_refusals_exhausted"
BODY_REFUSALS = "deferral_refusals"
BODY_REFUSALS_REMAINING = "deferral_refusals_remaining"
DEFERRAL_CONFIG_KEY = "local_deferral_max_refusals"
DEFAULT_DEFERRAL_MAX_REFUSALS = 3

HEADER_PREFERENCE = "X-Local-Preference"
HEADER_SESSION_ID = "X-Session-ID"

#: ``X-Local-Fallback-Reason`` value a remote-served response carries when the
#: request never consulted the local route at all (``stamp_route_headers``).
NO_LOCAL_PROVIDER = "no_local_provider"

#: The two dispatch entry points a preference is parsed on. ``local_first`` is
#: the local+remote cycle (``proxy_with_fallback``); ``remote_only`` is the
#: remote-only cycle (``proxy_with_remote_fallback``).
ENTRY_POINTS = ("local_first", "remote_only")

PROXY_DIR = Path(__file__).resolve().parents[1]
ROUTING_DOC = PROXY_DIR / "docs" / "routing.md"
CONFIG_YAML = PROXY_DIR / "config.yaml"


@dataclass(frozen=True)
class RetiredValue:
    """One withdrawn ``X-Local-Preference`` wire value.

    ``successors`` are the live values the rejection ``detail`` must name, so a
    client can migrate from the response body alone (parent AC1/AC14).
    """

    value: str
    successors: tuple[str, ...]
    #: The child that lands the retirement (the owner of the red spec).
    owner: str


#: Every withdrawn value, with the successors its ``detail`` must name.
RETIRED_VALUES: tuple[RetiredValue, ...] = (
    RetiredValue(
        value=LOCAL_ONLY,
        successors=(LOCAL_PREFERRED,),
        owner="LP-0MVUH9PAZ002UAPA",
    ),
    RetiredValue(
        value=PREFER_LOCAL,
        successors=(LOCAL_PREFERRED, REMOTE_OK),
        owner="LP-0MVUH9PUU005HVS4",
    ),
)

RETIRED_VALUE_IDS = [row.value for row in RETIRED_VALUES]


# ---------------------------------------------------------------------------
# Imports of behaviour that does not exist yet
# ---------------------------------------------------------------------------


def require_local_deferral():
    """Return ``proxy.local_deferral`` or fail with an actionable message.

    ``pytest.fail`` (rather than a bare import at module scope) keeps a
    not-yet-implemented module an *assertion* failure with a named owner
    instead of a collection error, which is what a red-first spec must be.
    """
    try:
        return import_module("proxy.local_deferral")
    except ImportError as exc:  # pragma: no cover - red phase only
        pytest.fail(
            "proxy.local_deferral is not implemented yet "
            "(LP-0MVUH9PL0005PW90 owns the bounded per-session refusal "
            f"counter): {exc}"
        )


def require_attribute(module, name: str):
    """Return ``module.<name>`` or fail naming the missing pre-agreed surface."""
    value = getattr(module, name, None)
    if value is None:  # pragma: no cover - red phase only
        pytest.fail(
            f"{module.__name__}.{name} is missing; the contract specs drive "
            "this pre-agreed name (see local_preference_v2_support's "
            "docstring)"
        )
    return value


# ---------------------------------------------------------------------------
# Request builder (header + session)
# ---------------------------------------------------------------------------


def build_request(
    *,
    preference: str | None = None,
    session_id: str | None = None,
    body: bytes = b'{"model":"test"}',
    extra_headers: dict | None = None,
):
    """Build the request stub every spec issues.

    *preference* ``None`` sends no ``X-Local-Preference`` header at all (the
    absent-header baseline); *session_id* ``None`` sends none either, which is
    the "no stable counting key" case the deferral budget must leave alone.
    """
    headers: dict[str, str] = dict(extra_headers or {})
    if preference is not None:
        headers[HEADER_PREFERENCE] = preference
    if session_id is not None:
        headers[HEADER_SESSION_ID] = session_id
    return matrix._DummyRequest(body=body, headers=headers)


# ---------------------------------------------------------------------------
# Config / model builders
# ---------------------------------------------------------------------------


def config_for(
    row,
    *,
    deferral_max_refusals: int | None = None,
    server_overrides: dict | None = None,
) -> dict:
    """The matrix row's server config, plus this contract's knobs.

    *deferral_max_refusals* sets ``local_deferral_max_refusals``; leaving it
    ``None`` means "do not set the key", which is how the specs pin the
    documented default without restating it.
    """
    config = deepcopy(matrix._config(row))
    server = config.setdefault("server", {})
    if deferral_max_refusals is not None:
        server[DEFERRAL_CONFIG_KEY] = deferral_max_refusals
    server.update(server_overrides or {})
    return config


def remote_less_model_config(row) -> dict:
    """The row's chain with every non-local entry removed (a ``private`` model)."""
    model_config = deepcopy(matrix._model_config(row))
    model_config["providers"] = [
        entry for entry in model_config["providers"] if entry.get("type") == "local"
    ]
    return model_config


def remote_only_model_config(row) -> dict:
    """The row's chain with every local entry removed (the remote-only cycle)."""
    model_config = deepcopy(matrix._model_config(row))
    model_config["providers"] = [
        entry for entry in model_config["providers"] if entry.get("type") != "local"
    ]
    return model_config


def private_model_config() -> dict:
    """The ``private`` model shape: a single ``type: local`` provider.

    Mirrors ``proxy/config.yaml``'s ``private`` model (the remote-less chain the
    contract must never strip into an empty one): every entry is local, so
    there is no tier to escalate to.
    """
    return {
        "providers": [
            {
                "name": matrix.LOCAL_PROVIDER_NAME,
                "type": "local",
                "llama_model": "Qwen3",
            }
        ],
        "aliases": ["private*"],
    }


# ---------------------------------------------------------------------------
# Issue: one end-to-end request through a dispatch entry point
# ---------------------------------------------------------------------------


async def issue(
    row,
    *,
    preference: str | None = None,
    session_id: str | None = None,
    model_config: dict | None = None,
    config: dict | None = None,
    entry_point: str = "local_first",
    request=None,
):
    """Issue *row*'s request; return ``(response, dispatches, request)``.

    *dispatches* is the ordered record of upstream attempts: one entry per
    dispatch, so ``[]`` proves a rejection wrote no dispatch/lease state and
    ``["remote"]`` proves exactly one remote call. The local stub honours the
    row's own ``local_response`` (post-dispatch rows), exactly like the matrix
    harness it is adapted from.
    """
    calls: list[str] = []

    async def _local(_req, _path):
        calls.append("local")
        response_factory = row.local_response or matrix._ok_response
        return response_factory()

    async def _remote(_req, _path, _pc):
        calls.append("remote")
        return matrix._ok_response()

    if entry_point not in ENTRY_POINTS:  # pragma: no cover - guard
        raise AssertionError(f"unknown entry point {entry_point!r}")
    if request is None:
        request = build_request(preference=preference, session_id=session_id)
    if model_config is None:
        model_config = matrix._model_config(row)
    if config is None:
        config = config_for(row)

    with ExitStack() as stack:
        stack.enter_context(patch("proxy.router.proxy_to_local", _local))
        stack.enter_context(patch("proxy.server.proxy_to_remote", _remote))
        stack.enter_context(
            patch(
                "proxy.provider._get_local_concurrency_info",
                matrix._MutableConcurrency(*row.concurrency),
            )
        )
        stack.enter_context(patch("proxy.provider.request_priority", return_value="low"))
        stack.enter_context(
            patch(
                "proxy.router_helpers.evaluate_and_apply_compaction",
                matrix._no_compaction,
            )
        )
        for ctx in matrix._row_patches(row):
            stack.enter_context(ctx)
        if entry_point == "local_first":
            result = await provider.proxy_with_fallback(
                request, "v1/chat/completions", model_config, config
            )
        else:
            result = await provider.proxy_with_remote_fallback(
                request, "v1/chat/completions", model_config, config
            )
    return result, calls, request


async def issue_sequence(row, *, count: int, **kwargs) -> list:
    """Issue the same row *count* times, reusing the request's session key.

    Returns the ``(response, dispatches, request)`` tuples in order, so a spec
    can assert a *consecutive* behaviour ("refusals 1–3, then the fourth")
    rather than a single response.
    """
    results = []
    for _ in range(count):
        results.append(await issue(row, **kwargs))
    return results


# ---------------------------------------------------------------------------
# Observed-state helpers
# ---------------------------------------------------------------------------


def body_of(response) -> dict:
    """Decode a response body as JSON (never assume the shape)."""
    return json.loads(response.body)


def route_of(response) -> str | None:
    return response.headers.get("X-Local-Route")


def fallback_reason_of(response) -> str | None:
    return response.headers.get("X-Local-Fallback-Reason")


def queue_wait_of(request) -> float:
    """The measured contention-queue wait this request spent (0.0 when none)."""
    return float(getattr(request, "_contention_queue_wait_seconds", 0.0) or 0.0)


def assert_no_upstream_calls(dispatches: list[str], *, context: str) -> None:
    """A rejection precedes routing: nothing is dispatched upstream."""
    assert dispatches == [], (
        f"{context}: a rejected request must not reach any provider "
        f"(dispatched={dispatches})"
    )


def assert_no_queue_or_lease_writes(dispatches: list[str], *, context: str) -> None:
    """No queue entry, no dispatch/lease record was written for the request.

    A lease is only ever taken by a *local dispatch* (the matrix harness takes
    it at the local stub), so "no dispatch + empty queue + unchanged dispatch
    records" is the queue/lease half of "nothing was written".
    """
    assert_no_upstream_calls(dispatches, context=context)
    assert contention_queue.queue_depth() == 0, (
        f"{context}: no queue entry may be created (depth="
        f"{contention_queue.queue_depth()})"
    )
    assert proxy_server.local_dispatch_records == {}, (
        f"{context}: no dispatch/lease record may be written "
        f"(records={proxy_server.local_dispatch_records!r})"
    )
    assert contention_queue._local_grant_times == {}, (
        f"{context}: no lease grant may be registered "
        f"(grants={contention_queue._local_grant_times!r})"
    )


def assert_canonical_fallback_reason(reason: str | None, *, context: str) -> None:
    """A stamped reason must come from the one canonical token table."""
    from proxy import admission_reserve

    assert reason, f"{context}: a remote fall-through must stamp a reason token"
    assert reason in admission_reserve.LOCAL_FALLBACK_REASON_TOKENS, (
        f"{context}: reason={reason!r} is not in "
        "admission_reserve.LOCAL_FALLBACK_REASON_TOKENS — the header, the "
        "body and the log field must draw from one table"
    )


# ---------------------------------------------------------------------------
# Deterministic clock / TTL knob
# ---------------------------------------------------------------------------


class _Clock:
    """A monotonic clock the specs advance explicitly."""

    def __init__(self, start: float = 1_000_000.0):
        self.now = float(start)

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


@contextmanager
def frozen_clock(start: float = 1_000_000.0) -> Iterator[_Clock]:
    """Freeze the refusal counter's clock so TTL expiry is deterministic.

    Patches the pre-agreed ``proxy.local_deferral._now()`` hook: a TTL that is
    asserted by sleeping would make this suite slow and flaky, so the counter
    reads its clock through that one function.
    """
    module = require_local_deferral()
    clock = _Clock(start)
    with patch.object(module, "_now", lambda: clock.now):
        yield clock


# ---------------------------------------------------------------------------
# Log capture
# ---------------------------------------------------------------------------


def log_messages(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def escalation_lines(caplog) -> list[str]:
    """Every log line that reports the escalation token."""
    return [line for line in log_messages(caplog) if ESCALATION_TOKEN in line]


def set_debug_logging(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="llama-proxy")


# ---------------------------------------------------------------------------
# Docs helpers
# ---------------------------------------------------------------------------


def collapsed(path: Path) -> str:
    """The file's text with all whitespace runs collapsed to single spaces.

    Prose assertions then survive re-wrapping (the docs are hard-wrapped) while
    still pinning the sentence a client author reads.
    """
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# State isolation
# ---------------------------------------------------------------------------


def reset_state() -> None:
    """Reset every module-level state the rows can touch.

    Mirrors the matrix module's autouse fixture, plus the refusal counter this
    contract adds (tolerating its absence while the specs are red).
    """
    contention_queue.reset()
    prefill_eta.reset()
    provider._provider_unavailable_until.pop(matrix.LOCAL_PROVIDER_NAME, None)
    proxy_server.local_dispatch_records.clear()
    try:
        local_deferral = import_module("proxy.local_deferral")
    except ImportError:  # pragma: no cover - red phase only
        return
    reset_all = getattr(local_deferral, "reset_all", None)
    if callable(reset_all):
        reset_all()
