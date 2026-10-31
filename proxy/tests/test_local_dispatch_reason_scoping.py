
# <!-- REFACTOR-LP-0MVWOK6LH006BTN4
# smell: modernization
# severity: medium
# description: Use format specifiers instead of percent format
# -->
"""Reason-scoped deferral of the measured-cache-deficit local 503.

Parent: LP-0MVWEZCB0006WPBQ (AC1, AC2, AC3, AC5).
Red-test sibling of LP-0MVWFKESK0005G61
(``[impl] Reason-scoped deferral of the deficit-marked local 503``).

The defect
----------
A local dispatch aborted by the measured cache-deficit guard
(``measured_cache_abort … reason=measured_cache_deficit``) returns the same
retryable 503 the prefill-deadline path returns. Outside the cold-start grace
window that 503 takes the generic *provider* path, so the proxy calls
``mark_provider_unavailable("local-llama", provider_cooldown_seconds=60)`` —
cooling the **whole local entry for every session**, not just the
session/prompt that lost the cache. With concurrent sessions the entry
cooldown is re-armed before it expires, starving ``prefer-local`` work while
the GPU idles (measured 2026-10-31: 217 arms / 5.5 h, local share 2.8 %).

These tests pin the post-fix contract **before** the implementation lands:

* **AC1 — no entry quarantine from a deficit abort.** A deficit-marked local
  503 is DEFERRED: no ``mark_provider_unavailable`` for the entry, no
  failure-counter bump, and a *different* session's subsequent
  ``prefer-local`` request still reaches local (no
  ``local_provider_cooldown`` chain-resolution skip).
* **AC2 — the aborting prompt is still refused.** The aborting session's own
  prompt is still refused by the per-session / per-prompt thrash window
  (``prefill_thrash_cooldown``), while a different prompt from the same
  session is admitted. The arming itself is unchanged.
* **AC3 — genuine local faults keep their quarantine.** An unmarked local
  503 outside the cold window, a local 502, a connection error after the
  cold-start grace window, and slot exhaustion still apply their documented
  cooldowns.
* **AC5 — the abort stays observable.** The marker value is the canonical
  deficit token (accepted by
  ``admission_reserve.canonical_local_fallback_reason``), and the aborting
  request's fallback reason is that token, never the generic ``HTTP 503``.

Contract pinned here (shared with the implementation child)
-----------------------------------------------------------
The router's deficit ``_build_backend_error_response`` carries the marker as
the response header ``X-Local-Dispatch-Reason: measured_cache_deficit`` (the
token is ``measured_cache_abort.MEASURED_CACHE_DEFICIT_REASON``).
``provider._handle_http_error_with_cooldown`` therefore takes the
cold-start-style deferral branch for a **local 503 carrying that marker** —
recording ``status=local_dispatch_deferral``, applying no entry cooldown and
no failure-counter bump, and returning the response's ``Retry-After`` so the
retry hint still reaches the caller. Every other 5xx / connection error keeps
its behaviour (AC3): the exemption is reason-scoped, never "any 503 from
local". The in-process half of the marker (so header loss cannot silently
re-enable the quarantine) is pinned by
``test_measured_cache_abort_wiring.py`` where the real router abort produces
the response.

Harness
-------
Modelled on ``test_prefill_thrash_cooldown.py``'s provider-fallback harness
(``_DummyRequest`` with an ``x-session-id`` header, monkeypatched
``provider._get_local_concurrency_info`` so a slot is free, real
``_handle_http_error_with_cooldown``) and ``test_cold_start_immunity.py`` for
the cold-start boundary. No network, no GPU, real routing code.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time

import httpx
import proxy.provider as provider
import pytest
from fastapi import Response
from fastapi.responses import JSONResponse, StreamingResponse
from proxy.measured_cache_abort import MEASURED_CACHE_DEFICIT_REASON

from proxy import admission_reserve, cold_start
from proxy import router_helpers as helpers

#: The header the router's deficit 503 carries (see module docstring).
DEFICIT_MARKER_HEADER = "X-Local-Dispatch-Reason"

#: The canonical deficit token reused as the marker value / fallback reason.
DEFICIT_REASON = MEASURED_CACHE_DEFICIT_REASON

#: The implementation child that makes the red rows pass; it must delete the
#: ``xfail(strict=True)`` markers below when the reason-scoped deferral lands.
IMPL_ITEM = "LP-0MVWFKESK0005G61"

#: Why the red rows are marked ``xfail(strict=True)`` today (test-first):
#: they must FAIL on the pre-fix tree (the entry is still quarantined and the
#: deficit token is not yet canonical), while the changed-scope gate has to
#: stay green for this test-only item to land. ``strict=True`` turns a row that
#: starts passing into a hard failure, so the implementation child must delete
#: the marker rather than leave the contract half-pinned.
RED_REASON = (
    "red until %s lands: the deficit-marked local 503 is still quarantined on "
    "the provider entry and the deficit token is not yet canonical" % IMPL_ITEM
)

#: The local provider ENTRY name these tests quarantine (or refuse to).
ENTRY = "local-llama"

SESSION_A = "sess-deficit-owner-a"
SESSION_B = "sess-neighbour-b"

#: The short slot-exhaustion cooldown the genuine-fault guard pins.
SLOT_UNAVAILABLE_RETRY_AFTER = 5

#: A 60 s entry cooldown is the value that re-arms before it expires under
#: concurrency (the parent's measured baseline). Tests assert the deficiency
#: path never applies it; they never tune it.
PROVIDER_COOLDOWN_SECONDS = 60


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_shared_state():
    """Isolate every global these tests touch, around each test."""
    cold_start.reset()
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._usage_reset_window.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    helpers._reset_prefill_thrash_cooldowns()
    yield
    cold_start.reset()
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._usage_reset_window.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    helpers._reset_prefill_thrash_cooldowns()


def _server_config(**overrides) -> dict:
    """Server config with the cooldown knobs the fallback chain reads."""
    server = {
        "provider_cooldown_seconds": PROVIDER_COOLDOWN_SECONDS,
        "slot_unavailable_retry_after": SLOT_UNAVAILABLE_RETRY_AFTER,
        "local_prefill_thrash_cooldown_seconds": 600,
        "local_cold_start_grace_seconds": 120,
        "local_cold_start_retry_after_seconds": 15,
    }
    server.update(overrides)
    return {"server": server, "provider_cooldown_seconds": server["provider_cooldown_seconds"]}


def _mixed_model_config() -> dict:
    """Local first, remote fallback — the production shape of the defect."""
    return {
        "providers": [
            {"name": ENTRY, "type": "local", "llama_model": "Qwen3"},
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
        ],
        "aliases": ["hybrid*"],
    }


class _SessionRequest:
    """Minimal FastAPI Request stand-in with a session header."""

    def __init__(self, session_id: str | None, body: dict | None = None):
        self._body = json.dumps(
            body
            or {
                "model": "Qwen3",
                "messages": [{"role": "user", "content": "short prompt"}],
            }
        ).encode("utf-8")
        self.headers = {"x-session-id": session_id} if session_id else {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


class _FreeSlotConcurrency:
    """A free local slot so the concurrency gate never deflects the request."""

    def __call__(self, *_args, **_kwargs):
        return (0, 1)


def _deficit_503(retry_after: int = 11) -> Response:
    """The marked retryable 503 the router returns for a deficit abort."""
    return Response(
        content=json.dumps(
            {
                "error": {
                    "type": "backend_error",
                    "code": "backend_error",
                    "message": "Backend unavailable, please retry later",
                },
                "status": 503,
                "retry_after": retry_after,
            }
        ),
        status_code=503,
        media_type="application/json",
        headers={
            "Retry-After": str(retry_after),
            DEFICIT_MARKER_HEADER: DEFICIT_REASON,
            "Cache-Control": "no-store",
        },
    )


def _unmarked_503(*, status_code: int = 503, retry_after: int = 11) -> Response:
    """A plain local 5xx carrying no deficit marker (a genuine fault)."""
    return Response(
        content=json.dumps({"error": {"message": "boom"}}),
        status_code=status_code,
        media_type="application/json",
        headers={"Retry-After": str(retry_after), "Cache-Control": "no-store"},
    )


def _slot_exhaustion_503() -> JSONResponse:
    """The router's slot-exhaustion payload (no deficit marker)."""
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "type": "server_busy",
                "code": "no_slots_available",
                "message": "Model server busy: 0/1 slots available. Please retry later.",
            },
            "total_slots": 1,
            "available_slots": 0,
            "retry_after": SLOT_UNAVAILABLE_RETRY_AFTER,
        },
        headers={"Retry-After": str(SLOT_UNAVAILABLE_RETRY_AFTER), "Cache-Control": "no-store"},
    )


def _ok_response() -> Response:
    return Response(
        content=json.dumps({"choices": [{"message": {"content": "ok"}}]}),
        status_code=200,
        media_type="application/json",
    )


def _streaming_ok() -> StreamingResponse:
    return StreamingResponse(
        iter([b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n', b"data: [DONE]\n\n"]),
        media_type="text/event-stream",
        status_code=200,
    )


def _messages(caplog) -> str:
    return " ".join(record.getMessage() for record in caplog.records)


@pytest.fixture
def fallback_chain(monkeypatch):
    """Drive the real fallback cycle with an in-memory local dispatcher.

    ``local`` is a callback ``(session_id, body) -> Response`` so each test
    decides what the local attempt returns. The remote always succeeds (with
    a streaming response by default) so a deferral or a skip is observable as
    "the request reached the remote".

    The redirect patchers are scoped to the test via an ``ExitStack`` teardown:
    a leaked ``proxy.router.proxy_to_local`` patch would silently reroute every
    later test in the same pytest session.
    """
    calls: list[tuple[str, str, dict]] = []  # (provider, session, body)
    stack = contextlib.ExitStack()

    def _install(local_fn, *, remote_fn=None):
        async def _local(req, path):
            body = json.loads(await req.body())
            session = (getattr(req, "headers", {}) or {}).get("x-session-id")
            calls.append(("local", session, body))
            return local_fn(session, body)

        async def _remote(req, path, provider_cfg):
            body = json.loads(await req.body())
            session = (getattr(req, "headers", {}) or {}).get("x-session-id")
            calls.append(("remote", session, body))
            return (remote_fn or (lambda *_: _streaming_ok()))(session, body)

        from unittest.mock import patch

        monkeypatch.setattr(
            provider, "_get_local_concurrency_info", _FreeSlotConcurrency()
        )
        stack.enter_context(patch("proxy.router.proxy_to_local", _local))
        stack.enter_context(patch("proxy.server.proxy_to_remote", _remote))

    yield _install, calls
    stack.close()


async def _run(request, cfg=None):
    return await provider.proxy_with_fallback(
        request, "v1/chat/completions", _mixed_model_config(), cfg or _server_config()
    )


# ---------------------------------------------------------------------------
# AC1 — the deficit-marked 503 is deferred (no entry quarantine)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=RED_REASON)
def test_marked_deficit_503_defers_without_quarantine():
    """AC1: `_handle_http_error_with_cooldown` defers the marked local 503.

    No entry cooldown, no failure-counter bump, and the retry hint (the
    response's ``Retry-After``) is returned so the caller can surface it.
    """
    attempts: list[dict] = []
    cooldown = provider._handle_http_error_with_cooldown(
        _deficit_503(retry_after=11),
        ENTRY,
        "local",
        PROVIDER_COOLDOWN_SECONDS,
        attempts,
        "measured cache deficit abort",
        config=_server_config(),
    )

    assert not provider._is_provider_unavailable(ENTRY), (
        "a measured-cache-deficit abort must not quarantine the local entry"
    )
    assert ENTRY not in provider._provider_unavailable_until
    assert provider._provider_failure_count.get(ENTRY, 0) == 0
    assert cooldown == 11.0, (
        f"the deferral must return the response's Retry-After hint, got {cooldown!r}"
    )
    assert attempts and attempts[0]["status"] == "local_dispatch_deferral", attempts


@pytest.mark.xfail(strict=True, reason=RED_REASON)
@pytest.mark.asyncio
async def test_neighbour_session_is_not_skipped_after_a_deficit_abort(
    monkeypatch, fallback_chain, caplog
):
    """AC1: a neighbour's deficit abort must not cool local for other sessions.

    Session A's local attempt returns the marked 503 and detours remote.
    Session B's subsequent ``prefer-local`` request must still reach local —
    the whole point of the fix.
    """
    install, calls = fallback_chain

    def _local(session, body):
        if session == SESSION_A:
            return _deficit_503()
        return _ok_response()

    install(_local)
    caplog.set_level(logging.INFO, logger="llama-proxy.provider")

    # Session A — deficit abort, detours to the remote provider.
    first = await _run(_SessionRequest(SESSION_A))
    assert first.status_code == 200
    assert first.headers.get("X-Provider") == "remote-fallback"
    assert not provider._is_provider_unavailable(ENTRY), (
        "the deficit abort must leave the local entry eligible"
    )
    assert provider._provider_failure_count.get(ENTRY, 0) == 0

    # Session B — a *different* session must still be served locally.
    second = await _run(_SessionRequest(SESSION_B))
    assert second.status_code == 200
    assert second.headers.get("X-Provider") == ENTRY, (
        "a neighbour's deficit abort must not remove local from another "
        "session's route"
    )

    local_sessions = [session for provider_name, session, _ in calls if provider_name == "local"]
    assert local_sessions == [SESSION_A, SESSION_B], local_sessions

    # AC1: the path never mints the entry-cooldown skip token for anyone.
    messages = _messages(caplog)
    assert "reason=local_provider_cooldown" not in messages, messages


# ---------------------------------------------------------------------------
# AC2 — the aborting prompt is still refused (thrash window unchanged)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_own_prompt_still_refused_by_fingerprint_scoped_window(
    monkeypatch, fallback_chain, caplog
):
    """AC2: the same session + same prompt is still refused for the window.

    The abort handler arms ``_record_prefill_thrash_cooldown(session, …,
    fingerprint=…)``; this test re-establishes the same window and asserts the
    gate refuses the *identical* prompt with ``reason=prefill_thrash_cooldown``
    while a different prompt from the same session is admitted.
    """
    install, calls = fallback_chain
    install(lambda session, body: _ok_response())

    # The abort's own arming is unchanged: per-session + per-prompt.
    monkeypatch.setattr(
        helpers, "_release_prefill_thrash_cooldown_on_free_gpu", lambda *a, **k: False
    )
    failed_body = {
        "model": "Qwen3",
        "messages": [{"role": "user", "content": "the doomed 56k prompt"}],
    }
    other_body = {
        "model": "Qwen3",
        "messages": [{"role": "user", "content": "an unrelated smaller prompt"}],
    }
    failed_fingerprint = helpers._prefill_thrash_fingerprint(failed_body)
    other_fingerprint = helpers._prefill_thrash_fingerprint(other_body)
    assert failed_fingerprint and other_fingerprint
    assert failed_fingerprint != other_fingerprint

    helpers._record_prefill_thrash_cooldown(
        SESSION_A, _server_config(), tokens=56_000, fingerprint=failed_fingerprint
    )
    assert (
        helpers._prefill_thrash_cooldown_remaining(
            SESSION_A, _server_config(), fingerprint=failed_fingerprint
        )
        > 0
    )

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")

    # The same prompt is still refused for this session.
    same = await _run(_SessionRequest(SESSION_A, failed_body))
    assert same.status_code == 200
    assert same.headers.get("X-Provider") == "remote-fallback"
    assert [provider_name for provider_name, _, _ in calls] == ["remote"], calls
    messages = _messages(caplog)
    assert messages.count("routing_skip_local") == 1, messages
    assert "reason=prefill_thrash_cooldown" in messages, messages

    # A different prompt from the SAME session is admitted (fingerprint scope
    # is preserved, so the window never widens into a session-wide lock).
    calls.clear()
    other = await _run(_SessionRequest(SESSION_A, other_body))
    assert other.status_code == 200
    assert other.headers.get("X-Provider") == ENTRY
    assert [provider_name for provider_name, _, _ in calls] == ["local"], calls


# ---------------------------------------------------------------------------
# AC3 — genuine local faults keep their documented quarantine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    ["unmarked_503", "local_502"],
    ids=["unmarked-503-outside-cold-window", "local-502"],
)
def test_unmarked_local_5xx_still_quarantines(case):
    """AC3: only the deficit-marked local *503* is exempt, never any 503.

    An unmarked local 503 (outside the cold window) and a local 502 still
    apply the full provider cooldown and bump the failure path exactly as
    before.
    """
    attempts: list[dict] = []
    response = _unmarked_503() if case == "unmarked_503" else _unmarked_503(status_code=502)

    cooldown = provider._handle_http_error_with_cooldown(
        response,
        ENTRY,
        "local",
        PROVIDER_COOLDOWN_SECONDS,
        attempts,
        "genuine local fault",
        config=_server_config(),
    )

    assert cooldown == PROVIDER_COOLDOWN_SECONDS
    assert provider._is_provider_unavailable(ENTRY), (
        f"{case} must still quarantine the local entry"
    )
    assert attempts and attempts[0]["status"] == "http_error"


def test_connection_error_after_grace_window_still_quarantines():
    """AC3: the connection-error path is unchanged after the cold window.

    A connection error inside the cold-start window defers (existing
    immunity), but the deficit exemption must not touch that boundary: after
    the window expires the documented connection-error quarantine applies.
    """
    config = _server_config(local_cold_start_grace_seconds=10)
    cold_start.note_model_loaded()
    # Move the arm time into the real past so the handler's own is_cold()
    # (real monotonic clock) sees an expired window.
    cold_start._state.loaded_at = time.monotonic() - 15

    attempts: list[dict] = []
    result = provider._handle_connection_error_in_fallback(
        httpx.ConnectError("Connection refused"),
        ENTRY,
        "local",
        PROVIDER_COOLDOWN_SECONDS,
        attempts,
        config=config,
    )

    assert result is True
    assert provider._is_provider_unavailable(ENTRY), (
        "a post-grace connection error must still quarantine"
    )
    assert attempts and attempts[0]["status"] == "connection_error"


@pytest.mark.asyncio
async def test_slot_exhaustion_still_applies_its_short_cooldown(
    monkeypatch, fallback_chain
):
    """AC3: slot exhaustion is a genuine local fault, not a deficit abort.

    The deficit deferral is reason-scoped: a local 503 whose body is the
    ``no_slots_available`` payload must still arm the documented *short* slot
    cooldown (``slot_unavailable_retry_after``), never the deferral branch and
    never the full provider cooldown.
    """
    install, _calls = fallback_chain
    install(lambda session, body: _slot_exhaustion_503())

    response = await _run(_SessionRequest(SESSION_B))
    assert response.status_code == 200
    assert response.headers.get("X-Provider") == "remote-fallback"

    expiry = provider._provider_unavailable_until.get(ENTRY)
    assert expiry is not None, "slot exhaustion must still quarantine the entry"
    remaining = expiry - time.time()
    assert remaining <= SLOT_UNAVAILABLE_RETRY_AFTER + 1.5, (
        f"expected the short slot cooldown, got {remaining:.1f}s"
    )
    assert remaining >= SLOT_UNAVAILABLE_RETRY_AFTER - 2.0, (
        f"expected ~{SLOT_UNAVAILABLE_RETRY_AFTER}s cooldown, got {remaining:.1f}s"
    )


# ---------------------------------------------------------------------------
# AC5 — the marker value and fallback reason use the canonical deficit token
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=RED_REASON)
def test_deficit_token_is_canonical():
    """AC5: the marker/fallback token is accepted by the canonical table.

    ``admission_reserve.canonical_local_fallback_reason`` fails loudly on an
    unknown token, so the aborting request's fallback reason can only carry
    ``measured_cache_deficit`` once the token is registered.
    """
    assert (
        admission_reserve.canonical_local_fallback_reason(DEFICIT_REASON)
        == DEFICIT_REASON
    )


@pytest.mark.xfail(strict=True, reason=RED_REASON)
@pytest.mark.asyncio
async def test_aborting_request_names_the_deficit_token_not_generic_503(
    monkeypatch, fallback_chain, caplog
):
    """AC5: the detour is attributable as a deficit abort, not ``HTTP 503``.

    When the marked 503 detours to a streaming remote provider the
    ``Fallback triggered`` line must name the canonical deficit token, so a
    log reader can tell the reason-scoped deferral apart from a genuine local
    fault.
    """
    install, _calls = fallback_chain
    install(lambda session, body: _deficit_503())

    caplog.set_level(logging.INFO, logger="llama-proxy.provider")
    response = await _run(_SessionRequest(SESSION_A))
    assert response.status_code == 200
    assert response.headers.get("X-Provider") == "remote-fallback"

    fallback_lines = [
        record.getMessage()
        for record in caplog.records
        if "Fallback triggered" in record.getMessage()
    ]
    assert fallback_lines, "the fall-through must be logged once"
    assert any(
        f"reason={DEFICIT_REASON}" in line for line in fallback_lines
    ), fallback_lines
    assert not any("reason=HTTP 503" in line for line in fallback_lines), (
        "a deficit abort must never be reported as a generic HTTP 503"
    )


def test_deficit_marker_header_uses_the_canonical_token():
    """AC5: the marker value on the wire is the canonical deficit token."""
    assert DEFICIT_MARKER_HEADER == "X-Local-Dispatch-Reason"
    assert DEFICIT_REASON == "measured_cache_deficit"
