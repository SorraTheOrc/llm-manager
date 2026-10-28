"""Preference and attribution on routing log lines.

LP-0MVRMC1AD0013HND (child of LP-0MVNU2KTK009D4AG, parent AC4/AC5). Test-first
for the logging child LP-0MVRMC275000R2W6.

Two properties are pinned here:

1. **Preference** (parent AC4): every log line that decides "not local" names
   the parsed ``X-Local-Preference`` as ``preference=<parsed value>``. The exit
   lines are enumerated once in :data:`_NOT_LOCAL_LINE_ROWS` —
   ``routing_skip_local``, ``routing_check``,
   ``contention_queue_fallback_after_queue``,
   ``local_admission_guard_direct_fallback``, the chain-resolution
   ``Skipping provider=<local> … in cooldown`` line and the quota hard-stop skip.
2. **Attribution** (parent AC5): ``priority_routing`` on **both** dispatch
   cycles carries the request's session id, so a preference can be tied to a
   specific request without correlation heuristics.

Two guards keep the change additive:

- :func:`_parse_kv` parses ``key=value`` pairs from each line, and a live test
  asserts every *pre-existing* key is still emitted. Log-grep consumers
  (``.pi/skills/proxy-log-debug``, ``.pi/skills/proxy-usage-analysis``) key off
  those names, so losing one would silently break them.
- A request with no header logs ``preference=prefer-local`` — never an empty or
  absent value.

One more property lives here because it is the same observability contract:
**the refusal logs itself** (``local_only_refused reason=<token>
preference=local-only``), so a violated intent is visible in ``proxy.log``
without correlating headers.

Red phase (test-first): this module was written and demonstrated red before
LP-0MVRMC275000R2W6 landed (the live assertions run unpinned through the
test-skill runner, ``20 failed, 22 passed``). The ``xfail(strict=True)`` markers
that pinned the missing behaviour were removed with the behaviour they pinned;
the red evidence is recorded on the work item.

Structural note — why ``remote-ok`` has no "not local" line row: ``remote-ok``
is enforced by :func:`proxy.priority.apply_local_preference`, which strips every
local entry from the chain *before* dispatch. A ``remote-ok`` request therefore
never reaches a local-route exit and never emits a "not local" line; its
preference is still visible on ``priority_routing``. A dedicated test asserts
exactly that (and that no local-route line is emitted), which is the honest
three-value coverage the parent AC asks for: all three parsed values are
provably loggable, but only the two that keep the local route in the chain can
appear on its exit lines.
"""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import proxy.provider as provider
import pytest
from fastapi import Response

from proxy import contention_queue, prefill_eta

#: Prompt size priced by the rows that need a known measurement.
PROMPT_TOKENS = 20_000
#: Calibrated budget refusal: a cache-cold 100k prompt at ~3.9 ms/token is
#: ~390 s of prefill against a 60 s wait budget.
ADMISSION_PROMPT_TOKENS = 100_000
#: Wait budget every row runs with (seconds).
WAIT_BUDGET_SECONDS = 60
#: Local and remote entries used by every row.
LOCAL_PROVIDER_NAME = "local-llama"
REMOTE_PROVIDER_NAME = "remote-fallback"
#: Model name logged by ``priority_routing`` — asserted so the line is
#: attributable to a model, not just to a field set.
MODEL_NAME = "hybrid-log-model"
#: Session id sent by the attribution tests; the resolved value must round-trip.
SESSION_ID = "sess-log-attribution"

LOCAL_ONLY = "local-only"
PREFER_LOCAL = "prefer-local"
REMOTE_OK = "remote-ok"
#: What the canonical parser yields when the header is absent or unknown.
DEFAULT_PREFERENCE = PREFER_LOCAL

#: ``key=value`` pairs in a log line. Values are non-space runs, matching the
#: format every routing line already emits (``session=unknown``,
#: ``cooldown_remaining=90s``, ``provider=local-llama:`` …).
_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


def _parse_kv(line: str) -> dict[str, str]:
    """Parse the ``key=value`` pairs of a log line into a dict.

    This is the same shape log-grep tooling relies on, so a test written with
    it fails for the same reason an operator's grep would.
    """
    return {match.group(1): match.group(2) for match in _KV_RE.finditer(line)}


@dataclass(frozen=True)
class _NotLocalLineRow:
    """One routing log line that decides "not local".

    ``trigger`` names the fixture that takes the exit (the same vocabulary as
    the refusal matrix, ``test_local_only_refusal_matrix.py``), ``prefix`` is
    the start of the line, ``contains`` disambiguates lines sharing a prefix,
    and ``existing_keys`` is the pre-existing ``key=value`` contract that an
    additive change must never drop.
    """

    trigger: str
    prefix: str
    existing_keys: tuple[str, ...]
    contains: tuple[str, ...] = ()
    description: str = ""


_NOT_LOCAL_LINE_ROWS: tuple[_NotLocalLineRow, ...] = (
    _NotLocalLineRow(
        trigger="thrash_cooldown",
        prefix="routing_skip_local",
        existing_keys=(
            "provider",
            "model",
            "session",
            "cooldown_remaining",
            "reason",
        ),
        description=(
            "session in post-deadline prefill-thrash cooldown "
            "(router_skip_local reason=prefill_thrash_cooldown)"
        ),
    ),
    _NotLocalLineRow(
        trigger="clean",
        prefix="routing_check",
        existing_keys=(
            "provider",
            "model",
            "estimated_tokens",
            "cold_threshold",
            "warm_threshold",
            "new_tokens",
            "cached_ratio",
            "cache_state",
            "messages",
            "session",
            "chain_hold_cycle",
        ),
        description="smart-routing estimate computed for the local entry",
    ),
    _NotLocalLineRow(
        trigger="queue_depth_cap",
        prefix="contention_queue_fallback_after_queue",
        existing_keys=("provider", "session", "queued_duration"),
        description="contention-queue depth cap reached (never queued)",
    ),
    _NotLocalLineRow(
        trigger="admission_guard_free_slot",
        prefix="local_admission_guard_direct_fallback",
        existing_keys=("provider", "model", "session", "reason", "free_slot"),
        description="prefill-gate denial on the free-slot (direct dispatch) path",
    ),
    _NotLocalLineRow(
        trigger="chain_cooldown",
        prefix="Skipping provider=",
        contains=("in cooldown",),
        existing_keys=("provider", "reason"),
        description="chain resolution skipped the local entry while it cooled down",
    ),
    _NotLocalLineRow(
        trigger="quota_hard_stop",
        prefix="Skipping provider=",
        contains=("reason=quota_hard_stop",),
        existing_keys=("provider", "reason"),
        description="chain resolution skipped the local entry on a quota hard stop",
    ),
    _NotLocalLineRow(
        trigger="lease_fallback",
        prefix="contention_queue_fallback_after_queue",
        contains=("reason=local_lease_active",),
        existing_keys=("provider", "session", "queued_duration", "reason"),
        description=(
            "lease denial fell back to the next provider after the queue budget"
        ),
    ),
)

#: Which ``priority_routing`` line each dispatch cycle emits.
_PRIORITY_ROUTING_CYCLES = (
    pytest.param("local_remote", id="local+remote-cycle"),
    pytest.param("remote_only", id="remote-only-cycle"),
)


def _preference_params() -> list:
    """``X-Local-Preference`` cases: each parsed value plus the absent header.

    ``remote-ok`` is excluded by design — see the module docstring: it strips
    the local route before dispatch, so a local-route exit line cannot fire for
    it. Its value coverage lives on ``priority_routing`` instead.
    """
    cases = (
        (LOCAL_ONLY, "local-only"),
        (PREFER_LOCAL, "prefer-local"),
        (None, "absent-header"),
    )
    return [pytest.param(value, id=cid) for value, cid in cases]


class _DummyRequest:
    """Minimal request stub (matches the other routing test helpers)."""

    def __init__(self, body: bytes = b'{"model":"test"}', headers=None):
        self._body = body
        self.headers = headers or {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


class _MutableConcurrency:
    """Mutable ``_get_local_concurrency_info`` stand-in."""

    def __init__(self, active: int = 0, max_: int = 1):
        self.active = active
        self.max = max_

    def __call__(self, config, endpoint=None) -> tuple[int, int]:
        return (self.active, self.max)


def _ok_response() -> Response:
    return Response(
        content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
        status_code=200,
        media_type="application/json",
    )


async def _no_compaction(_server, _result, _body_json, _server_config) -> dict:
    """Server-side compaction is a no-op: these rows pin *logging*, not rescue."""
    return {
        "evaluated": False,
        "action": "noop",
        "applied": False,
        "estimated_before": 0,
        "estimated_after": 0,
        "reason": None,
        "messages": None,
    }


@contextmanager
def _estimate_patches(tokens: int, cached_ratio: float = 0.0):
    """Price every prompt exactly *tokens*, cache-cold by default."""
    with (
        patch(
            "proxy.provider._estimate_effective_prompt_tokens_for_routing",
            new=AsyncMock(return_value=tokens),
        ),
        patch("proxy.provider._get_tokenizer_for_model", lambda *_: (None, 1.0)),
        patch("proxy.provider._get_cached_ratio", lambda *_: cached_ratio),
    ):
        yield


def _seed_calibration(tokens: int, ms_per_token: float = 3.9) -> None:
    for _ in range(6):
        prefill_eta.record_dispatch_sample("Qwen3", tokens, 0.0, tokens * ms_per_token)


@contextmanager
def _local_entry_in_cooldown(seconds: float = 60.0):
    """Put the local entry into a *real* cooldown."""
    previous = provider._provider_unavailable_until.get(LOCAL_PROVIDER_NAME)
    provider._provider_unavailable_until[LOCAL_PROVIDER_NAME] = time.time() + seconds
    try:
        yield
    finally:
        if previous is None:
            provider._provider_unavailable_until.pop(LOCAL_PROVIDER_NAME, None)
        else:
            provider._provider_unavailable_until[LOCAL_PROVIDER_NAME] = previous


def _quota_hard_stop_for_local(original):
    """Quota hard stop on the *local* entry only, so a remote fallback exists.

    Chain resolution walks every provider, so a blanket patch would exhaust the
    remote entry too and the row would pin the exhaustion path instead of the
    local skip. *original* is the real predicate, captured before the patch.
    """

    def _check(provider_cfg: dict, config: dict):
        if provider_cfg.get("type") == "local":
            return True, "quota_hard_stop (monthly quota reached)"
        return original(provider_cfg, config)

    return _check


def _occupy_the_only_queue_slot() -> None:
    """Take the queue's single depth slot, as a concurrently waiting request would.

    The contention-queue module documents its waiter tuple as
    ``(enqueued_at, event, session_id, retained, eta_seconds)``; seeding one
    waiter reaches the depth cap deterministically. ``contention_queue.reset()``
    in the autouse fixture clears it again.
    """
    contention_queue._waiters.append((0.0, None, "queue-holder", False, None))


def _lease_active_response() -> Response:
    """The synthetic 503 a held local dispatch lease returns (router)."""
    return Response(
        content=json.dumps({
            "error": {
                "code": "no_slots_available",
                "type": "server_busy",
                "message": "Local dispatch lease is held by another session",
            },
            "total_slots": 1,
            "available_slots": 0,
            "reason": "local_lease_active",
        }).encode("utf-8"),
        status_code=503,
        media_type="application/json",
    )


def _model_config(cycle: str = "local_remote") -> dict:
    """The model chain for *cycle* (local + remote, or remote only)."""
    providers = [
        {
            "name": REMOTE_PROVIDER_NAME,
            "type": "remote",
            "endpoint": "https://api.example.com/v1",
        }
    ]
    if cycle == "local_remote":
        providers.insert(
            0,
            {"name": LOCAL_PROVIDER_NAME, "type": "local", "llama_model": "Qwen3"},
        )
    return {"name": MODEL_NAME, "providers": providers, "aliases": ["hybrid*"]}


def _config(row: _NotLocalLineRow) -> dict:
    """Server config for *row*.

    Thresholds are deliberately disabled (``local_model_ctx_size: 0`` and no
    economic ceiling) so each row is skipped for the reason it names and not for
    an ambient context threshold.
    """
    server = {
        "session_slot_pool_size": 1,
        "contention_queue_policy": "queue",
        "contention_queue_max_wait_seconds": WAIT_BUDGET_SECONDS,
        "contention_queue_max_depth": 4,
        "local_model_ctx_size": 0,
        # Isolate the budget bound from the deadline bound.
        "local_prefill_deadline_seconds": 0,
    }
    if row.trigger == "queue_depth_cap":
        server["contention_queue_max_depth"] = 1
    return {"provider_cooldown_seconds": 60, "server": server}


def _row_concurrency(row: _NotLocalLineRow) -> tuple[int, int]:
    """``(active, max)`` local slots the row runs with."""
    return (1, 1) if row.trigger == "queue_depth_cap" else (0, 1)


def _row_patches(row: _NotLocalLineRow) -> list:
    """The row-specific patches that take *row*'s exit (on top of the shared ones)."""
    trigger = row.trigger
    patches: list = []
    if trigger == "clean":
        patches.append(_estimate_patches(PROMPT_TOKENS))
    elif trigger == "thrash_cooldown":
        patches.append(
            patch(
                "proxy.router_helpers._prefill_thrash_cooldown_remaining",
                lambda *a, **k: 90.0,
            )
        )
    elif trigger == "queue_depth_cap":
        _occupy_the_only_queue_slot()
    elif trigger == "admission_guard_free_slot":
        _seed_calibration(ADMISSION_PROMPT_TOKENS)
        patches.append(_estimate_patches(ADMISSION_PROMPT_TOKENS))
    elif trigger == "chain_cooldown":
        patches.append(_local_entry_in_cooldown())
    elif trigger == "quota_hard_stop":
        patches.append(
            patch(
                "proxy.provider._quota_exhausted",
                _quota_hard_stop_for_local(provider._quota_exhausted),
            )
        )
    elif trigger == "lease_fallback":
        # The lease is denied *after* the local dispatch, so the queue decision
        # is what falls back: pin the budget-expiry verdict deterministically
        # instead of waiting out a real queue budget.
        patches.append(
            patch(
                "proxy.provider._maybe_queue_for_local_slot",
                new=AsyncMock(return_value=("fallback", None, 300.0)),
            )
        )
    else:  # pragma: no cover - a new row must name its trigger
        raise AssertionError(f"unknown not-local line trigger {trigger!r}")
    return patches


async def _issue(
    row: _NotLocalLineRow,
    preference: str | None,
    model_config: dict | None = None,
) -> Response:
    """Issue *row*'s request with *preference*; ``caplog`` captures the lines."""
    async def _local(_req, _path):
        if row.trigger == "lease_fallback":
            return _lease_active_response()
        return _ok_response()

    async def _remote(_req, _path, _pc):
        return _ok_response()

    headers = {} if preference is None else {"X-Local-Preference": preference}
    with ExitStack() as stack:
        stack.enter_context(patch("proxy.router.proxy_to_local", _local))
        stack.enter_context(patch("proxy.server.proxy_to_remote", _remote))
        stack.enter_context(
            patch(
                "proxy.provider._get_local_concurrency_info",
                _MutableConcurrency(*_row_concurrency(row)),
            )
        )
        stack.enter_context(patch("proxy.provider.request_priority", return_value="low"))
        stack.enter_context(
            patch(
                "proxy.router_helpers.evaluate_and_apply_compaction",
                _no_compaction,
            )
        )
        for ctx in _row_patches(row):
            stack.enter_context(ctx)
        return await provider.proxy_with_fallback(
            _DummyRequest(headers=headers),
            "v1/chat/completions",
            model_config if model_config is not None else _model_config(),
            _config(row),
        )


@pytest.fixture(autouse=True)
def _capture_provider_info(caplog):
    """Capture ``llama-proxy.provider`` INFO records for every test here.

    The rows log at INFO, so the level must be raised once per test; the test
    body then reads the very same ``caplog`` instance.
    """
    caplog.set_level(logging.INFO, logger="llama-proxy.provider")


@pytest.fixture(autouse=True)
def _clean_state():
    contention_queue.reset()
    prefill_eta.reset()
    provider._provider_unavailable_until.pop(LOCAL_PROVIDER_NAME, None)
    yield
    contention_queue.reset()
    prefill_eta.reset()
    provider._provider_unavailable_until.pop(LOCAL_PROVIDER_NAME, None)


def _find_line(messages: list[str], row: _NotLocalLineRow) -> str | None:
    """Return the first logged line matching *row*, or ``None``."""
    for message in messages:
        if message.startswith(row.prefix) and all(
            needle in message for needle in row.contains
        ):
            return message
    return None


def _messages(caplog) -> list[str]:
    """The captured log messages, in emission order."""
    return [record.getMessage() for record in caplog.records]


def _assert_line_present(messages: list[str], row: _NotLocalLineRow) -> str:
    line = _find_line(messages, row)
    assert line is not None, (
        f"{row.description}: expected a log line starting {row.prefix!r} "
        f"{('containing ' + repr(row.contains)) if row.contains else ''} — the "
        "not-local decision must stay attributable; logs were:\n"
        + "\n".join(messages)
    )
    return line


# ---------------------------------------------------------------------------
# AC3 (additive contract): every pre-existing key survives on every line
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row", _NOT_LOCAL_LINE_ROWS, ids=[r.trigger for r in _NOT_LOCAL_LINE_ROWS]
)
@pytest.mark.parametrize("preference", _preference_params())
async def test_not_local_line_keeps_every_pre_existing_key(row, preference, caplog):
    """An additive change must not drop a field log-grep tooling depends on.

    This is deliberately live (not red-phase): the pre-existing keys are the
    ``key=value`` contract that ``proxy-log-debug`` / ``proxy-usage-analysis``
    parse today, and they must survive the ``preference=`` addition untouched.
    """
    await _issue(row, preference)
    line = _assert_line_present(_messages(caplog), row)
    kv = _parse_kv(line)

    missing = [key for key in row.existing_keys if key not in kv]
    assert not missing, (
        f"{row.description}: the line lost pre-existing key(s) {missing} — "
        f"log-grep consumers break. Line: {line!r}"
    )


# ---------------------------------------------------------------------------
# AC1/AC4: every "not local" line names the parsed preference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row", _NOT_LOCAL_LINE_ROWS, ids=[r.trigger for r in _NOT_LOCAL_LINE_ROWS]
)
@pytest.mark.parametrize("preference", _preference_params())
async def test_not_local_line_names_the_parsed_preference(row, preference, caplog):
    """Each exit line carries ``preference=<parsed value>``; absent ⇒ default.

    ``preference=`` must be the value the canonical parser produced (never a
    re-parse, never blank): ``local-only`` for an explicit pin, and
    ``prefer-local`` for the two cases that keep today's behaviour (explicit
    ``prefer-local`` and no header at all).
    """
    await _issue(row, preference)
    line = _assert_line_present(_messages(caplog), row)
    expected = DEFAULT_PREFERENCE if preference is None else preference
    kv = _parse_kv(line)

    assert kv.get("preference"), (
        f"{row.description}: the not-local decision line must carry "
        f"preference=<parsed value> (parent AC4). Line: {line!r}"
    )
    assert kv["preference"] == expected, (
        f"{row.description}: parsed preference is {expected!r}; the line must "
        f"say preference={expected} — got preference={kv['preference']!r}. "
        f"Line: {line!r}"
    )


# ---------------------------------------------------------------------------
# AC2: priority_routing carries session attribution on both dispatch cycles
# ---------------------------------------------------------------------------


async def _issue_priority_routing(
    cycle: str, preference: str | None, caplog
) -> str:
    """Issue one request for *cycle*; return its single ``priority_routing`` line."""
    async def _local(_req, _path):
        return _ok_response()

    async def _remote(_req, _path, _pc):
        return _ok_response()

    headers = {"x-session-id": SESSION_ID}
    if preference is not None:
        headers["X-Local-Preference"] = preference
    model_config = _model_config(cycle)
    config = {
        "provider_cooldown_seconds": 60,
        "server": {"session_slot_pool_size": 1, "local_model_ctx_size": 0},
    }
    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(0, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        patch("proxy.router_helpers.evaluate_and_apply_compaction", _no_compaction),
    ):
        if cycle == "local_remote":
            await provider.proxy_with_fallback(
                _DummyRequest(headers=headers),
                "v1/chat/completions",
                model_config,
                config,
            )
        else:
            await provider.proxy_with_remote_fallback(
                _DummyRequest(headers=headers),
                "v1/chat/completions",
                model_config,
                config,
            )
    lines = [
        message
        for message in _messages(caplog)
        if message.startswith("priority_routing ")
    ]
    assert len(lines) == 1, (
        f"the {cycle} cycle must log exactly one priority_routing line per "
        f"request (no added log volume, LP-0MVRMC275000R2W6 AC); got {lines!r}"
    )
    return lines[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("cycle", _PRIORITY_ROUTING_CYCLES)
async def test_priority_routing_keeps_its_existing_fields(cycle, caplog):
    """AC2 additive half: model/priority/providers/preference are unchanged.

    ``priority_routing`` is the line an operator greps to see *which* chain was
    selected; it must keep exactly those fields on both cycles while gaining
    the session attribution.
    """
    line = await _issue_priority_routing(cycle, PREFER_LOCAL, caplog)
    kv = _parse_kv(line)

    assert kv.get("model") == MODEL_NAME, f"model field lost/changed: {line!r}"
    assert kv.get("priority") == "low", f"priority field lost/changed: {line!r}"
    assert kv.get("providers"), f"providers field lost: {line!r}"
    assert kv.get("preference") == PREFER_LOCAL, (
        f"preference field lost/changed: {line!r}"
    )
    assert kv["preference"] != "", "preference must never be an empty value"


@pytest.mark.asyncio
@pytest.mark.parametrize("cycle", _PRIORITY_ROUTING_CYCLES)
async def test_priority_routing_carries_session_attribution(cycle, caplog):
    """AC5: the routing decision names the request's session.

    Without the session id an operator can see ``preference=local-only`` on a
    skip line but cannot tie it to the client request that made the choice —
    exactly the correlation heuristic the parent item exists to remove.
    """
    line = await _issue_priority_routing(cycle, PREFER_LOCAL, caplog)
    kv = _parse_kv(line)

    attribution = kv.get("session") or kv.get("request_id")
    assert attribution, (
        "priority_routing must carry session=/request_id= so a preference is "
        f"attributable to a request (parent AC5). Line: {line!r}"
    )
    assert attribution != "unknown", (
        "the request carried x-session-id, so the attribution must name it — "
        f"not the fallback placeholder. Line: {line!r}"
    )
    assert attribution == SESSION_ID, (
        f"the logged session must be the request's own ({SESSION_ID!r}); got "
        f"{attribution!r}. Line: {line!r}"
    )


# ---------------------------------------------------------------------------
# AC4: the default preference is always explicit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_absent_header_logs_the_default_preference(caplog):
    """AC4: no header ⇒ ``preference=prefer-local``, never empty or absent."""
    line = await _issue_priority_routing("local_remote", None, caplog)
    kv = _parse_kv(line)

    assert kv.get("preference") == DEFAULT_PREFERENCE, (
        f"a request with no X-Local-Preference must log the parsed default "
        f"{DEFAULT_PREFERENCE!r}; got {kv.get('preference')!r}. Line: {line!r}"
    )


# ---------------------------------------------------------------------------
# Three-value coverage: remote-ok is visible on priority_routing only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remote_ok_preference_is_logged_and_takes_no_local_exit(caplog):
    """``remote-ok`` is loggable, but it never decides "not local".

    ``apply_local_preference()`` strips local from the chain before dispatch, so
    there is no local-route exit to log. The preference must still be visible —
    on ``priority_routing`` — and no "not local" line may claim otherwise.
    """
    line = await _issue_priority_routing("local_remote", REMOTE_OK, caplog)
    kv = _parse_kv(line)
    assert kv.get("preference") == REMOTE_OK, (
        f"the third parsed value must be loggable too; got {kv.get('preference')!r}. "
        f"Line: {line!r}"
    )

    messages = _messages(caplog)
    emitted = [row.trigger for row in _NOT_LOCAL_LINE_ROWS if _find_line(messages, row)]
    assert emitted == [], (
        "a remote-ok request never enters the local route, so no not-local "
        f"decision line can exist; got {emitted}. Lines:\n" + "\n".join(messages)
    )


# ---------------------------------------------------------------------------
# AC4: the refusal names itself in the log
# ---------------------------------------------------------------------------


def _row(trigger: str) -> _NotLocalLineRow:
    """The table row named *trigger* (rows stay the single source of exits)."""
    return next(row for row in _NOT_LOCAL_LINE_ROWS if row.trigger == trigger)


@pytest.mark.asyncio
async def test_local_only_refusal_logs_one_line_with_reason_and_preference(caplog):
    """A refusal is visible without correlating headers.

    The skip reason the chain recorded (here: the local entry's cooldown) must
    appear on the refusal line together with ``preference=local-only``, so an
    operator grepping ``proxy.log`` sees the violation and its cause instead of
    joining the request's headers to a routing line.
    """
    result = await _issue(_row("chain_cooldown"), LOCAL_ONLY)
    assert result.status_code == 429, "the local-only client is refused, not served"

    lines = [m for m in _messages(caplog) if m.startswith("local_only_refused ")]
    assert len(lines) == 1, (
        f"the refusal must be logged exactly once per refused request; got {lines!r}"
    )
    kv = _parse_kv(lines[0])
    assert kv.get("reason") == "local_provider_cooldown", (
        f"the refusal line must name the recorded skip reason; got {kv!r}"
    )
    assert kv.get("preference") == LOCAL_ONLY, (
        f"the refusal line must name the intent that caused it; got {kv!r}"
    )


@pytest.mark.asyncio
async def test_local_only_refusal_without_a_local_route_names_that_reason(caplog):
    """No local entry at all ⇒ the refusal says so, still naming the intent."""
    remote_only = {
        **_model_config(),
        "providers": [
            p for p in _model_config()["providers"] if p["type"] != "local"
        ],
    }
    result = await _issue(
        _row("chain_cooldown"), LOCAL_ONLY, model_config=remote_only
    )
    assert result.status_code == 429

    lines = [m for m in _messages(caplog) if m.startswith("local_only_refused ")]
    assert len(lines) == 1, f"one line per refusal; got {lines!r}"
    kv = _parse_kv(lines[0])
    assert kv.get("reason") == "local_only_no_local_route", (
        f"a chain with no local entry must say so; got {kv!r}"
    )
    assert kv.get("preference") == LOCAL_ONLY


@pytest.mark.asyncio
async def test_prefer_local_fall_through_logs_no_refusal_line(caplog):
    """The additive half: the default client's fall-through claims no refusal."""
    result = await _issue(_row("chain_cooldown"), PREFER_LOCAL)
    assert result.status_code == 200, "the default client keeps the remote answer"
    assert [
        m for m in _messages(caplog) if m.startswith("local_only_refused ")
    ] == [], "a prefer-local request is served remotely, never refused"
