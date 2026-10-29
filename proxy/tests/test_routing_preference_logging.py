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
   ``Skipping provider=<local> … in cooldown`` line, the quota hard-stop skip
   and the per-entry ``routing_skip_provider_context`` ceiling skip
   (LP-0MVTF2WLQ008BNYK).
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
preference=local-preferred verdict=<deferrable|too_large>``), so a violated
intent and its capacity verdict are visible in ``proxy.log`` without correlating
headers. ``preference`` is the *parsed* value, so the legacy ``local-only``
header is logged under its canonical spelling (LP-0MVSMINPR006SWT6).

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
#: ``entry_context_cap`` row: the local entry's own ceiling, below the priced
#: prompt so the request leaves the local route (a pin-local request is promoted,
#: the default client takes the warm-context skip).
ENTRY_CAP_LOCAL_TOKENS = PROMPT_TOKENS // 2
#: ``entry_context_cap`` row: the remote entry's ceiling, also below the priced
#: prompt, so the per-entry skip is taken on the remote entry too.
ENTRY_CAP_REMOTE_TOKENS = PROMPT_TOKENS // 4
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
LOCAL_PREFERRED = "local-preferred"
PREFER_LOCAL = "prefer-local"
REMOTE_OK = "remote-ok"
#: What the canonical parser yields when the header is absent or unknown.
DEFAULT_PREFERENCE = PREFER_LOCAL
#: The value each wire value is logged as: the parser normalises the legacy
#: ``local-only`` alias to the canonical ``local-preferred``
#: (LP-0MVSHWYVE005MS16 AC1), so the exit lines name that canonical value.
_PARSED_PREFERENCE = {
    LOCAL_ONLY: LOCAL_PREFERRED,
    LOCAL_PREFERRED: LOCAL_PREFERRED,
    PREFER_LOCAL: PREFER_LOCAL,
    None: DEFAULT_PREFERENCE,
}

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
        trigger="entry_context_cap",
        prefix="routing_skip_provider_context",
        existing_keys=(
            "provider",
            "type",
            "status",
            "estimated_tokens",
            "limit",
            "reason",
            "session",
        ),
        description=(
            "per-entry max_context_tokens ceiling exceeded: the prompt leaves "
            "the entry before the budget gate and dispatch"
        ),
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
    """``X-Local-Preference`` cases: each accepted wire value plus no header.

    ``remote-ok`` is excluded by design — see the module docstring: it strips
    the local route before dispatch, so a local-route exit line cannot fire for
    it. Its value coverage lives on ``priority_routing`` instead. The legacy
    ``local-only`` case is kept deliberately: it is the deployed wire value and
    must take exactly the same exits as the canonical ``local-preferred``.
    """
    cases = (
        (LOCAL_ONLY, "local-only-legacy-alias"),
        (LOCAL_PREFERRED, "local-preferred"),
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


def _row_model_config(row: _NotLocalLineRow) -> dict:
    """The model chain *row* runs against.

    Only the ``entry_context_cap`` row caps its entries: the ceiling is the
    trigger under test there and must not leak into any other row (the other
    exits are conditional on the entry being dispatchable).
    """
    model_config = _model_config()
    if row.trigger != "entry_context_cap":
        return model_config
    capped = []
    for entry in model_config["providers"]:
        limit = (
            ENTRY_CAP_LOCAL_TOKENS
            if entry.get("type") == "local"
            else ENTRY_CAP_REMOTE_TOKENS
        )
        capped.append({**entry, "max_context_tokens": limit})
    return {**model_config, "providers": capped}


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
    if row.trigger == "entry_context_cap":
        # The smart-routing gate short-circuits entirely at a cold threshold of
        # 0, so the local entry's own ceiling would never bind. Enabling the
        # cold term (kept above the priced prompt) makes Check 1 — the capped
        # warm threshold — the binding limit for the default client.
        server["local_large_context_cold_cache_threshold"] = PROMPT_TOKENS * 2
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
    elif trigger == "entry_context_cap":
        # The ceiling is compared against the routing estimate, so the prompt is
        # priced deterministically above both capped entries.
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
            model_config if model_config is not None else _row_model_config(row),
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
    re-parse, never blank): ``local-preferred`` for either pin-local spelling
    (the legacy ``local-only`` alias is normalised to it), and ``prefer-local``
    for the two cases that keep today's behaviour (explicit ``prefer-local``
    and no header at all).
    """
    await _issue(row, preference)
    line = _assert_line_present(_messages(caplog), row)
    expected = _PARSED_PREFERENCE[preference]
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
    appear on the refusal line together with the *parsed* preference
    (``preference=local-preferred`` for the legacy ``local-only`` header), so an
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
    assert kv.get("preference") == LOCAL_PREFERRED, (
        "the refusal line names the parsed intent — the legacy ``local-only`` "
        f"alias canonicalises to ``local-preferred``; got {kv!r}"
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
    assert kv.get("preference") == LOCAL_PREFERRED


@pytest.mark.asyncio
async def test_prefer_local_fall_through_logs_no_refusal_line(caplog):
    """The additive half: the default client's fall-through claims no refusal."""
    result = await _issue(_row("chain_cooldown"), PREFER_LOCAL)
    assert result.status_code == 200, "the default client keeps the remote answer"
    assert [
        m for m in _messages(caplog) if m.startswith("local_only_refused ")
    ] == [], "a prefer-local request is served remotely, never refused"


# ---------------------------------------------------------------------------
# The single per-request routing-outcome line (LP-0MVSM06Z8003UQTM)
#
# Every request emits exactly **one** ``local_preference_resolved`` line naming
# the resolved preference, the outcome and the recorded reason token. The line
# is emitted from ``_stamp_local_route_headers()`` — the top-level return choke
# point both entries reach once, *after* ``_run_chain_cycles()`` — so neither a
# per-skip branch nor a chain-hold retry cycle can duplicate it.
#
# The assertions here are deliberately per-outcome rather than one aggregate
# check: the point of the line is that an operator can tell a deferral
# (``outcome=deferred reason=fallback_after_queue``) from chain exhaustion
# (``outcome=exhausted``) without correlating the per-skip lines, and that a
# legacy ``local-only`` request is reported under its normalised value.
# ---------------------------------------------------------------------------

RESOLVED_LINE_PREFIX = "local_preference_resolved "


def _resolved_lines(caplog) -> list[str]:
    """Every ``local_preference_resolved`` line captured in *caplog*."""
    return [
        message
        for message in _messages(caplog)
        if message.startswith(RESOLVED_LINE_PREFIX)
    ]


def _one_resolved_line(caplog) -> dict[str, str]:
    """The request's single outcome line, parsed; asserts ``count == 1``."""
    lines = _resolved_lines(caplog)
    assert len(lines) == 1, (
        "exactly one local_preference_resolved line per request is required "
        f"(AC1/AC5); got {len(lines)}: {lines!r}"
    )
    return _parse_kv(lines[0])


def _assert_outcome(
    caplog,
    *,
    outcome: str,
    preference: str,
    reason: str,
    session: str = "unknown",
) -> dict[str, str]:
    """Assert the one line's outcome/preference/reason/session/model fields."""
    fields = _one_resolved_line(caplog)
    assert fields.get("outcome") == outcome, (
        f"expected outcome={outcome}, got {fields.get('outcome')!r}: {fields!r}"
    )
    assert fields.get("preference") == preference, (
        f"expected preference={preference}, got {fields.get('preference')!r}: "
        f"{fields!r}"
    )
    assert fields.get("reason") == reason, (
        f"expected reason={reason}, got {fields.get('reason')!r}: {fields!r}"
    )
    assert fields.get("session") == session, (
        f"expected session={session}, got {fields.get('session')!r}: {fields!r}"
    )
    assert fields.get("model") == MODEL_NAME, (
        f"the line must name the model it routed, got {fields.get('model')!r}"
    )
    return fields


@pytest.mark.asyncio
async def test_local_success_logs_the_local_outcome_once(caplog):
    """AC1/AC2: a locally served request reports ``outcome=local``."""
    result = await _issue(_row("clean"), None)

    assert result.status_code == 200
    _assert_outcome(
        caplog,
        outcome=provider.LOCAL_ROUTING_OUTCOME_LOCAL,
        preference=DEFAULT_PREFERENCE,
        reason="none",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("preference", [LOCAL_PREFERRED, LOCAL_ONLY])
async def test_pin_refusal_logs_the_pinned_outcome_once(preference, caplog):
    """AC1/AC2: a refused pin-local request reports ``outcome=pinned``.

    The reason token is the recorded skip reason, so the line says *why* the
    request was pinned instead of forcing an operator to correlate the
    ``local_only_refused`` line with the ``Skipping provider=…`` line.
    """
    result = await _issue(_row("chain_cooldown"), preference)

    assert result.status_code == 429
    _assert_outcome(
        caplog,
        outcome=provider.LOCAL_ROUTING_OUTCOME_PINNED,
        preference=LOCAL_PREFERRED,
        reason="local_provider_cooldown",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger", "expected_reason"),
    [
        pytest.param("queue_depth_cap", "fallback_after_queue", id="depth-cap"),
        pytest.param("lease_fallback", "local_lease_active", id="lease-active"),
    ],
)
async def test_deferral_logs_the_deferred_outcome_once(
    trigger, expected_reason, caplog
):
    """AC2: a spent deferral is distinguishable from chain exhaustion.

    ``outcome=deferred`` with the queue/lease reason token is the whole point of
    the line: the incident's dead end reported "All providers exhausted" for a
    request that was really waiting on local capacity, and an operator reading
    only that could not tell the two apart.
    """
    result = await _issue(_row(trigger), LOCAL_PREFERRED)

    assert result.status_code == 429
    fields = _assert_outcome(
        caplog,
        outcome=provider.LOCAL_ROUTING_OUTCOME_DEFERRED,
        preference=LOCAL_PREFERRED,
        reason=expected_reason,
    )
    assert fields.get("outcome") != provider.LOCAL_ROUTING_OUTCOME_EXHAUSTED, (
        "a deferral must never be reported as chain exhaustion"
    )


#: The matrix's ``entry_context_cap`` shape: a local entry whose own ceiling is
#: below the priced prompt, so the pre-loop verdict strips local.
PROMOTED_LOCAL_CAP_TOKENS = PROMPT_TOKENS // 2


def _promoted_model_config() -> dict:
    return {
        "name": MODEL_NAME,
        "providers": [
            {
                "name": LOCAL_PROVIDER_NAME,
                "type": "local",
                "llama_model": "Qwen3",
                "max_context_tokens": PROMOTED_LOCAL_CAP_TOKENS,
            },
            {
                "name": REMOTE_PROVIDER_NAME,
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
        ],
        "aliases": ["hybrid*"],
    }


def _promoted_config() -> dict:
    return {
        "provider_cooldown_seconds": 60,
        "server": {
            "session_slot_pool_size": 1,
            "contention_queue_policy": "queue",
            "contention_queue_max_wait_seconds": WAIT_BUDGET_SECONDS,
            "contention_queue_max_depth": 4,
            "local_model_ctx_size": 0,
            "local_prefill_deadline_seconds": 0,
            "local_large_context_cold_cache_threshold": 1_000,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("preference", [LOCAL_PREFERRED, LOCAL_ONLY])
async def test_too_large_promotion_logs_the_promoted_outcome_once(
    preference, caplog
):
    """AC1/AC2: a request stripped from the chain reports ``outcome=promoted``.

    The request is *served* by a remote provider, but the outcome that explains
    it is ``promoted``: reporting ``remote`` would hide the one fact an operator
    needs (local was dropped because the prompt outgrew it).
    """
    async def _local(_req, _path):  # pragma: no cover - must never run
        raise AssertionError("a promoted request never dispatches locally")

    async def _remote(_req, _path, _pc):
        return _ok_response()

    with (
        patch("proxy.router.proxy_to_local", _local),
        patch("proxy.server.proxy_to_remote", _remote),
        patch("proxy.provider._get_local_concurrency_info", _MutableConcurrency(0, 1)),
        patch("proxy.provider.request_priority", return_value="low"),
        patch("proxy.router_helpers.evaluate_and_apply_compaction", _no_compaction),
        _estimate_patches(PROMPT_TOKENS),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(headers={"X-Local-Preference": preference}),
            "v1/chat/completions",
            _promoted_model_config(),
            _promoted_config(),
        )

    assert result.status_code == 200
    assert result.headers.get("X-Local-Route") == "remote"
    _assert_outcome(
        caplog,
        outcome=provider.LOCAL_ROUTING_OUTCOME_PROMOTED,
        preference=LOCAL_PREFERRED,
        reason="none",
    )


@pytest.mark.asyncio
async def test_prefer_local_remote_fall_through_logs_the_remote_outcome(caplog):
    """AC2: ``prefer-local`` reports its preference verbatim plus the real serve.

    ``prefer-local`` is not a pin, so the request is served remotely as today —
    the line must say exactly that instead of pretending local was used.
    """
    result = await _issue(_row("chain_cooldown"), PREFER_LOCAL)

    assert result.status_code == 200
    _assert_outcome(
        caplog,
        outcome=provider.LOCAL_ROUTING_OUTCOME_REMOTE,
        preference=PREFER_LOCAL,
        reason="local_provider_cooldown",
    )


@pytest.mark.asyncio
async def test_legacy_local_only_is_reported_under_its_normalised_value(caplog):
    """AC3: ``local-only`` never appears as the resolved preference.

    The alias is normalised by the canonical parser (and cached on the request),
    so the line reports ``preference=local-preferred``. The deprecation warning
    from LP-0MVSHWYVE005MS16 is separate and unchanged.
    """
    await _issue(_row("chain_cooldown"), LOCAL_ONLY)

    lines = _resolved_lines(caplog)
    assert len(lines) == 1, f"one line per request; got {lines!r}"
    assert "local-only" not in lines[0].replace("local-preferred", ""), (
        f"the legacy alias must never be presented as the resolved value: "
        f"{lines[0]!r}"
    )
    fields = _parse_kv(lines[0])
    assert fields.get("preference") == LOCAL_PREFERRED, (
        f"expected the normalised value, got {fields.get('preference')!r}"
    )


def _always_failing_remote():
    """A remote stub that always fails, so the chain exhausts."""
    async def _remote(_req, _path, _pc):
        return Response(status_code=502, content=b"Bad gateway")

    return _remote


@pytest.mark.asyncio
async def test_chain_exhaustion_logs_the_exhausted_outcome_once(caplog):
    """AC1/AC2: exhaustion reports ``outcome=exhausted`` with no reason token.

    This is the shape the incident mislabelled: exhaustion is a genuine
    terminal state, and the line now says so explicitly, so it can never be
    confused with a deferral (which carries ``outcome=deferred`` and a reason).
    """
    model_config = {
        "name": MODEL_NAME,
        "providers": [
            {
                "name": REMOTE_PROVIDER_NAME,
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
        ],
    }
    with patch(
        "proxy.provider._get_proxy_to_remote",
        return_value=_always_failing_remote(),
    ):
        result = await provider.proxy_with_remote_fallback(
            _DummyRequest(),
            "v1/chat/completions",
            model_config,
            {"provider_cooldown_seconds": 0},
        )

    assert result.status_code == 503, f"expected exhaustion, got {result.status_code}"
    _assert_outcome(
        caplog,
        outcome=provider.LOCAL_ROUTING_OUTCOME_EXHAUSTED,
        preference=DEFAULT_PREFERENCE,
        reason="none",
    )


@pytest.mark.asyncio
async def test_multi_cycle_chain_hold_still_logs_exactly_one_line(caplog):
    """AC5: a chain-hold retry (several cycles) is still one request, one line.

    The emission lives outside ``_run_chain_cycles()``: cycle 0 exhausts, the
    request is held and cycle 1 succeeds, yet only one
    ``local_preference_resolved`` line exists. Logging inside the cycle would
    produce one line per attempt — the duplication this item removes.
    """
    model_config = {
        "name": MODEL_NAME,
        "providers": [
            {"name": LOCAL_PROVIDER_NAME, "type": "local", "llama_model": "Qwen3"},
            {
                "name": REMOTE_PROVIDER_NAME,
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
        ],
    }
    config = {
        "provider_cooldown_seconds": 0,
        "chain_hold_seconds": 0.01,
        "chain_hold_max_cycles": 3,
    }
    remote_calls: list[int] = []

    async def _local(_req, _path):
        return Response(status_code=400, content=b"bad request")

    async def _remote(_req, _path, _pc):
        remote_calls.append(1)
        if len(remote_calls) == 1:
            return Response(status_code=502, content=b"Bad gateway")
        return _ok_response()

    with (
        patch("proxy.provider._get_proxy_to_local", return_value=_local),
        patch("proxy.provider._get_proxy_to_remote", return_value=_remote),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(), "v1/chat/completions", model_config, config
        )

    assert result.status_code == 200
    assert len(remote_calls) == 2, "the request must have taken two cycles"
    assert len(_resolved_lines(caplog)) == 1, (
        "a chain-hold retry must not duplicate the per-request line: "
        f"{_resolved_lines(caplog)!r}"
    )
    fields = _one_resolved_line(caplog)
    assert fields.get("outcome") == provider.LOCAL_ROUTING_OUTCOME_REMOTE, (
        f"the retried request was served remotely, got {fields.get('outcome')!r}"
    )
    assert fields.get("preference") == DEFAULT_PREFERENCE, (
        f"expected preference={DEFAULT_PREFERENCE}, got {fields.get('preference')!r}"
    )


@pytest.mark.asyncio
async def test_line_is_not_emitted_per_skip_branch(caplog):
    """AC6: two skipped local entries still produce exactly one outcome line.

    The existing per-skip lines (``Skipping provider=…``) stay exactly as they
    are — one per skipped entry — while the outcome line stays one per request.
    This is the structural difference between "log every skip" and "report the
    request's resolution".
    """
    model_config = {
        "name": MODEL_NAME,
        "providers": [
            {"name": "local-a", "type": "local", "llama_model": "Qwen3"},
            {"name": "local-b", "type": "local", "llama_model": "Qwen3"},
            {
                "name": REMOTE_PROVIDER_NAME,
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
            },
        ],
    }

    async def _local(_req, _path):  # pragma: no cover - both locals are skipped
        raise AssertionError("both local entries are in cooldown")

    async def _remote(_req, _path, _pc):
        return _ok_response()

    previous = {
        name: provider._provider_unavailable_until.get(name)
        for name in ("local-a", "local-b")
    }
    provider._provider_unavailable_until["local-a"] = time.time() + 60
    provider._provider_unavailable_until["local-b"] = time.time() + 60
    try:
        with (
            patch("proxy.router.proxy_to_local", _local),
            patch("proxy.server.proxy_to_remote", _remote),
            patch(
                "proxy.provider._get_local_concurrency_info",
                _MutableConcurrency(0, 1),
            ),
            patch("proxy.provider.request_priority", return_value="low"),
            patch("proxy.router_helpers.evaluate_and_apply_compaction", _no_compaction),
        ):
            result = await provider.proxy_with_fallback(
                _DummyRequest(),
                "v1/chat/completions",
                model_config,
                {"provider_cooldown_seconds": 60, "server": {"local_model_ctx_size": 0}},
            )
    finally:
        for name, value in previous.items():
            if value is None:
                provider._provider_unavailable_until.pop(name, None)
            else:
                provider._provider_unavailable_until[name] = value

    assert result.status_code == 200
    skips = [m for m in _messages(caplog) if m.startswith("Skipping provider=")]
    assert len(skips) >= 2, (
        f"the existing per-skip lines stay as they are (one per skipped local "
        f"entry, per resolution pass); got {skips!r}"
    )
    assert len(_resolved_lines(caplog)) == 1, (
        "the outcome line is per request, never per skip: "
        f"{_resolved_lines(caplog)!r}"
    )


# ---------------------------------------------------------------------------
# AC4: emission cannot change behaviour
# ---------------------------------------------------------------------------


#: Response headers whose value is a live clock reading, so two identical
#: requests issued moments apart legitimately differ (``Retry-After`` /
#: ``retry_at``). They are excluded from the AC4 response comparison: the
#: property under test is that *the emission* changes nothing, not that two
#: separate requests advertise the same instant.
_VOLATILE_RESPONSE_HEADERS = frozenset({"date", "retry-after", "retry_at"})
#: Body fields derived from the same live clock.
_VOLATILE_BODY_FIELDS = ("retry_at", "retry_after")


def _response_signature(result: Response) -> tuple:
    """Everything a client can observe about *result*, minus live-clock values."""
    headers = tuple(
        sorted(
            (key.lower(), value)
            for key, value in result.headers.items()
            if key.lower() not in _VOLATILE_RESPONSE_HEADERS
        )
    )
    try:
        body: object = json.loads(result.body)
    except Exception:
        body = result.body
    if isinstance(body, dict):
        for key in _VOLATILE_BODY_FIELDS:
            body.pop(key, None)
    return (
        result.status_code,
        headers,
        tuple(result.headers.getlist("x-local-route")),
        body,
    )


@contextmanager
def _emission_suppressed():
    """Suppress the outcome line entirely (the pre-change behaviour)."""
    with patch.object(
        provider, "_log_local_preference_resolved", lambda *a, **k: None
    ):
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger", "preference"),
    [
        pytest.param("clean", None, id="local-served"),
        pytest.param("chain_cooldown", LOCAL_PREFERRED, id="pinned-refusal"),
        pytest.param("chain_cooldown", PREFER_LOCAL, id="remote-served"),
    ],
)
async def test_response_is_identical_with_the_line_suppressed(
    trigger, preference, caplog
):
    """AC4: status, headers and body do not depend on the emission at all."""
    with _emission_suppressed():
        baseline = await _issue(_row(trigger), preference)
    caplog.clear()
    emitted = await _issue(_row(trigger), preference)

    assert _response_signature(emitted) == _response_signature(baseline), (
        "the observability line must not alter the response (status, headers, "
        "body)"
    )
    assert len(_resolved_lines(caplog)) == 1


@pytest.mark.asyncio
async def test_a_raising_logger_still_returns_the_response_unchanged(caplog):
    """AC4: an exception inside the emission is swallowed, response intact.

    Observability must never be able to break a request, so the emission is
    best-effort. The logger is made to raise *only* for the outcome line so the
    routing code's own logging is untouched.
    """
    with _emission_suppressed():
        baseline = await _issue(_row("chain_cooldown"), LOCAL_PREFERRED)
    caplog.clear()

    real_info = provider.logger.info

    def _raising_info(message, *args, **kwargs):
        if isinstance(message, str) and message.startswith(RESOLVED_LINE_PREFIX):
            raise RuntimeError("logger exploded")
        return real_info(message, *args, **kwargs)

    with patch.object(provider.logger, "info", _raising_info):
        result = await _issue(_row("chain_cooldown"), LOCAL_PREFERRED)

    assert _response_signature(result) == _response_signature(baseline), (
        "a raising logger must not change the response"
    )


@pytest.mark.asyncio
async def test_exhaustion_of_a_non_pinned_client_is_not_reported_as_pinned(caplog):
    """AC2: a refusal-shaped exhaustion is not a pin the client never asked for.

    A ``prefer-local`` (or header-less) client whose admission refusal leaves no
    provider to serve it receives the admission-refusal payload — which carries
    ``X-Local-Route: refused`` — but it was never pinned: the chain exhausted.
    The line must say ``outcome=exhausted``, so an operator never reads a pin
    that did not happen, and the reason is still the token the client is told.
    """
    local_only_chain = {
        "name": MODEL_NAME,
        "providers": [
            {"name": LOCAL_PROVIDER_NAME, "type": "local", "llama_model": "Qwen3"},
        ],
        "aliases": ["private*"],
    }

    result = await _issue(
        _row("admission_guard_free_slot"), None, model_config=local_only_chain
    )

    assert result.status_code == 429
    assert result.headers.get("X-Local-Route") == "refused"
    assert json.loads(result.body).get("code") == provider.EXHAUSTION_CODE_LOCAL_ADMISSION
    _assert_outcome(
        caplog,
        outcome=provider.LOCAL_ROUTING_OUTCOME_EXHAUSTED,
        preference=DEFAULT_PREFERENCE,
        reason="eta_exceeds_budget_direct",
    )


# ---------------------------------------------------------------------------
# The per-entry context ceiling names the preference on BOTH dispatch cycles
#
# ``_record_provider_context_skip()`` (LP-0MVTF2WLQ008BNYK) logs
# ``routing_skip_provider_context`` from two call sites: the remote-only cycle
# (``_proxy_with_remote_fallback_cycle``) and the local+remote cycle
# (``_proxy_with_fallback_cycle``). The ``entry_context_cap`` row in
# ``_NOT_LOCAL_LINE_ROWS`` covers the second — where a pin-local request is
# promoted away from its capped local entry and the default client takes the
# warm-context skip, so all three parsed preferences reach the line. This block
# covers the first: a lone remote entry above its ceiling.
#
# A pin-local request provably never reaches that loop: with no local entry in
# the chain there is nothing to pin to, so the choke point refuses it before the
# per-entry gate. Pinning that (rather than asserting a line that cannot exist)
# is what keeps the three-value coverage honest.
# ---------------------------------------------------------------------------


def _capped_remote_only_chain() -> dict:
    """One remote entry whose ceiling is below the priced prompt."""
    return {
        "name": MODEL_NAME,
        "providers": [
            {
                "name": REMOTE_PROVIDER_NAME,
                "type": "remote",
                "endpoint": "https://api.example.com/v1",
                "max_context_tokens": ENTRY_CAP_REMOTE_TOKENS,
            },
        ],
        "aliases": ["hybrid*"],
    }


def _never_dispatch_remote():
    """A remote stub that fails loudly: the capped entry must never dispatch."""

    async def _remote(_req, _path, _pc):  # pragma: no cover - must never run
        raise AssertionError("an entry above its own ceiling must not dispatch")

    return _remote


async def _issue_remote_only(preference: str | None) -> Response:
    """Issue one request on the remote-only cycle with *preference*."""
    headers = {} if preference is None else {"X-Local-Preference": preference}
    with (
        patch(
            "proxy.provider._get_proxy_to_remote",
            return_value=_never_dispatch_remote(),
        ),
        patch("proxy.provider.request_priority", return_value="low"),
        patch("proxy.router_helpers.evaluate_and_apply_compaction", _no_compaction),
        _estimate_patches(PROMPT_TOKENS),
    ):
        return await provider.proxy_with_remote_fallback(
            _DummyRequest(headers=headers),
            "v1/chat/completions",
            _capped_remote_only_chain(),
            _config(_row("entry_context_cap")),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("preference", "expected"),
    [
        pytest.param(None, DEFAULT_PREFERENCE, id="absent-header"),
        pytest.param(PREFER_LOCAL, PREFER_LOCAL, id="prefer-local"),
        pytest.param(REMOTE_OK, REMOTE_OK, id="remote-ok"),
    ],
)
async def test_remote_only_cycle_context_skip_names_the_preference(
    preference, expected, caplog
):
    """The remote-only call site carries ``preference=`` too.

    Same additive contract as the row test: every pre-existing key survives and
    the parsed intent is appended, so an operator grepping ``proxy.log`` for
    ``routing_skip_provider_context`` can attribute the skip to the request's
    ``X-Local-Preference`` on either cycle.
    """
    result = await _issue_remote_only(preference)

    assert result.status_code == 503, "the only entry was skipped, so the chain exhausted"
    lines = [
        message
        for message in _messages(caplog)
        if message.startswith("routing_skip_provider_context ")
    ]
    assert len(lines) == 1, (
        "one line per skipped entry on the remote-only cycle; got "
        f"{lines!r}\n" + "\n".join(_messages(caplog))
    )
    kv = _parse_kv(lines[0])
    assert kv.get("type") == "remote", f"the line names the skipped entry: {lines[0]!r}"
    for key in _row("entry_context_cap").existing_keys:
        assert key in kv, f"pre-existing key {key!r} lost: {lines[0]!r}"
    assert kv.get("preference") == expected, (
        f"the remote-only skip must name preference={expected}; got "
        f"{kv.get('preference')!r}. Line: {lines[0]!r}"
    )


@pytest.mark.asyncio
async def test_remote_only_cycle_refuses_a_pin_local_client_before_the_ceiling(caplog):
    """Pinned intent wins over the per-entry gate: refusal, not a context skip.

    A pin-local client on a chain with no local entry has nothing to pin to, so
    the choke point refuses it (429, ``X-Local-Route: refused``) before the
    per-entry ceiling is consulted. No ``routing_skip_provider_context`` line is
    emitted, exactly as for the ``remote-ok`` strip: the line reports entries the
    request actually walked past.
    """
    result = await _issue_remote_only(LOCAL_PREFERRED)

    assert result.status_code == 429, "a pin-local request is refused, not served"
    assert result.headers.get("X-Local-Route") == "refused"
    assert [
        message
        for message in _messages(caplog)
        if message.startswith("routing_skip_provider_context ")
    ] == [], "the refused request never reached the per-entry gate"
    refusal_lines = [
        message
        for message in _messages(caplog)
        if message.startswith("local_only_refused ")
    ]
    assert len(refusal_lines) == 1, (
        f"the refusal names itself once; got {refusal_lines!r}"
    )
    assert _parse_kv(refusal_lines[0]).get("preference") == LOCAL_PREFERRED
