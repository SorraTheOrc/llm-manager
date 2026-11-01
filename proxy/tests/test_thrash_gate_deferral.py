"""The post-deadline thrash-gate refusal is a deferral, not an entry quarantine.

LP-0MVXSSYTN001TBW5 (discovered from LP-0MVWFKJIP006UXXW; parent
LP-0MVWEZCB0006WPBQ).

The defect
----------
The measured cache-deficit deferral (LP-0MVWFKESK0005G61) removed one way a
*prompt-level* verdict quarantined the whole local entry. One hop later the same
request reaches the router's post-deadline thrash gate
(``proxy_to_local`` → ``_local_dispatch_gate``), which refuses with
``routing_skip_local reason=post_deadline_terminal`` and returns a retryable
``503`` **without** a ``local_dispatch_reason`` marker. The fallback loop
therefore classified it as ``reason=HTTP 503`` and called
``mark_provider_unavailable("local-qwen3", 60)`` — one neighbour's aborted
prompt became a 60 s cross-session outage of the local route.

Measured (2026-10-31 23:56 → 2026-11-01 12:12, build ``5d8318e3``): 1 193 of
1 369 (87.1 %) ``local-qwen3 in cooldown`` skips belong to bursts whose nearest
antecedent is that unmarked gate 503, and 23 of 41 cooldown bursts start
0.1–8 s after a ``post_deadline_terminal`` refusal.

What these tests pin
--------------------
* **AC1** — both thrash-gate refusals (``post_deadline_terminal`` and
  ``prefill_thrash_cooldown``) carry the local-dispatch reason marker
  (``X-Local-Dispatch-Reason`` header **and** the in-process attribute), and a
  *different* session's subsequent ``prefer-local`` request is **not** refused
  with ``<local> in cooldown``.
* **AC2** — the gate refusal path applies no ``mark_provider_unavailable`` and
  bumps no failure counter; the attempt record reads
  ``status=local_dispatch_deferral`` with the gate reason.
* **AC3** — the refusing request still detours to the next provider within the
  same request, and the response keeps its retryable ``503`` shape and
  ``Retry-After``.
* **AC4** — the exemption is reason-scoped: a local 503 carrying a marker that
  is not a deferral keeps the pre-change quarantine.
* **AC5** — the ``Fallback triggered`` line names the gate reason (never
  ``reason=HTTP 503``), the served response carries it as
  ``X-Local-Fallback-Reason``, and the starvation detector records no arm.
* **AC6** — ``proxy/docs/routing.md`` documents the refusals as
  non-quarantining deferrals.

Harness: the router-level refusal is produced by the **real** ``proxy_to_local``
(same minimal harness as ``test_prefill_thrash_cooldown.py``); the produced
response object is then handed to the fallback chain (``proxy_with_fallback``)
exactly as ``proxy.proxy_to_local`` would return it, so the end-to-end claim is
asserted against one real refusal rather than a hand-built imitation. No
network, no GPU.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import proxy.provider as provider
import pytest
from fastapi import Response
from fastapi.responses import StreamingResponse

from proxy import admission_reserve, cold_start, local_route_starvation
from proxy import router_helpers as helpers

#: The local entry name used by the incident's log lines
#: (``Skipping provider=local-qwen3: local-qwen3 in cooldown``).
ENTRY = "local-qwen3"
REMOTE = "remote-fallback"

SESSION_A = "sess-gate-owner-a"
SESSION_B = "sess-neighbour-b"

#: The two gate refusals this item is about, read from the owning module so a
#: rename cannot leave the tests asserting a stale spelling.
GATE_REASONS = (
    helpers.LOCAL_GATE_REASON_POST_DEADLINE_TERMINAL,
    helpers.LOCAL_GATE_REASON_PREFILL_THRASH_COOLDOWN,
)

#: A representative entry cooldown (the value the gate 503 armed in the wild).
PROVIDER_COOLDOWN_SECONDS = 60

ROUTING_DOC = Path(__file__).resolve().parents[1] / "docs" / "routing.md"
DOC_GATE_HEADING = "**Reason-scoped sibling: the post-deadline thrash-gate refusals"
DOC_COLD_START_HEADING = "## Local cold-start immunity"
DOC_NEXT_HEADING = "## Usage-limit reset quarantine"


# ---------------------------------------------------------------------------
# Fixtures / shared state
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
    local_route_starvation.reset_local_route_starvation_tracker()
    yield
    cold_start.reset()
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._usage_reset_window.clear()
    provider._sibling_failure_count.clear()
    provider._sibling_failure_streak_start.clear()
    helpers._reset_prefill_thrash_cooldowns()
    local_route_starvation.reset_local_route_starvation_tracker()


def _server_config(**overrides) -> dict:
    server = {
        "provider_cooldown_seconds": PROVIDER_COOLDOWN_SECONDS,
        "local_prefill_thrash_cooldown_seconds": 120,
        "local_cold_start_grace_seconds": 0,
        "llama_server_port": 8080,
        "session_slot_pool_size": 3,
        "max_concurrent_queries": 16,
        "local_prefill_deadline_seconds": 90,
        "local_dispatch_first_byte_timeout_seconds": 840,
        "local_dispatch_max_prefill_seconds": 900,
        "local_dispatch_no_progress_timeout_seconds": 180,
        "local_dispatch_lease_timeout_seconds": 30,
        "local_dispatch_lease_chunk_refresh_buffer_seconds": 30,
        "local_max_concurrent_prefills": 0,
    }
    server.update(overrides)
    return {"server": server, "provider_cooldown_seconds": server["provider_cooldown_seconds"]}


def _mixed_model_config() -> dict:
    """Local first, remote fallback — the production shape of the defect."""
    return {
        "providers": [
            {"name": ENTRY, "type": "local", "llama_model": "Qwen3"},
            {
                "name": REMOTE,
                "type": "remote",
                "endpoint": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
        ],
        "aliases": ["hybrid*"],
    }


class _RouterRequest:
    """Minimal FastAPI Request stand-in (x-session-id header + JSON body)."""

    def __init__(self, session_id: str, body: dict):
        self._body = json.dumps(body).encode("utf-8")
        self.headers = {"x-session-id": session_id}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


def _prompt_body(prompt: str = "the doomed 56k prompt") -> dict:
    return {
        "model": "Qwen3",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }


# ---------------------------------------------------------------------------
# The real router refusal (AC1 marker, AC3 response shape)
# ---------------------------------------------------------------------------


def _install_router_harness(monkeypatch, server_cfg: dict):
    """Minimal ``proxy_to_local`` harness (mirrors test_prefill_thrash_cooldown).

    Only the pieces the dispatch path needs *before* the thrash gate are
    wired; the gate refuses before any local resource is acquired, which is
    exactly what the tests then assert (no dispatch record, no slot check).
    """
    from proxy import server as srv

    monkeypatch.setattr(srv, "config", server_cfg)
    proc = MagicMock()
    proc.poll.return_value = None
    monkeypatch.setattr(srv, "llama_process", proc)
    monkeypatch.setattr(srv, "backend_ready", True)
    monkeypatch.setattr(srv, "current_model", "Qwen3")
    monkeypatch.setattr(srv, "active_queries", 0)
    monkeypatch.setattr(srv, "active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_active_queries", 0)
    monkeypatch.setattr(srv, "local_active_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_generating_queries", 0)
    monkeypatch.setattr(srv, "local_generating_queries_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_generating_sessions", set())
    monkeypatch.setattr(srv, "local_dispatch_records", {})
    monkeypatch.setattr(srv, "local_dispatch_records_lock", asyncio.Lock())
    monkeypatch.setattr(srv, "local_prefill_in_flight", {})
    monkeypatch.setattr(srv, "local_prefill_in_flight_lock", asyncio.Lock())

    import proxy.router as router_mod

    monkeypatch.setattr(router_mod, "_is_self_healing_active", lambda: False)
    monkeypatch.setattr(
        router_mod,
        "_handle_session",
        AsyncMock(
            return_value={
                "session_id": SESSION_A,
                "session_id_header": SESSION_A,
                "session_explicit": True,
                "session_created": False,
                "is_delta_request": False,
                "session_fallback_reason": None,
                "delta_messages": None,
                "original_message_count": 1,
                "body_json": _prompt_body(),
                "body_override": None,
            }
        ),
    )
    monkeypatch.setattr(router_mod, "_build_slot_context", lambda *_: (None, None, 3.0))
    monkeypatch.setattr(router_mod, "_resolve_slot_model_name", lambda model, *_: model)
    monkeypatch.setattr(router_mod, "_check_slot_availability", AsyncMock(return_value=None))
    monkeypatch.setattr(
        router_mod, "_update_session_and_slot", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        router_mod,
        "_call_with_empty_retry",
        AsyncMock(
            return_value=SimpleNamespace(
                status_code=200,
                headers={"content-type": "application/json"},
                content=b'{"choices":[]}',
            )
        ),
    )
    return router_mod


def _arm_gate(reason: str, request, server_cfg: dict) -> None:
    """Arm the gate refusal named by *reason* for *request*."""
    if reason == helpers.LOCAL_GATE_REASON_POST_DEADLINE_TERMINAL:
        # The request-scoped terminal marker the deadline/deficit abort sets.
        helpers._mark_local_dispatch_terminal(request)
        return
    assert reason == helpers.LOCAL_GATE_REASON_PREFILL_THRASH_COOLDOWN
    helpers._record_prefill_thrash_cooldown(
        SESSION_A,
        server_cfg,
        tokens=40_000,
        fingerprint=helpers._prefill_thrash_fingerprint(_prompt_body()),
    )


async def _real_gate_refusal(monkeypatch, reason: str) -> Response:
    """Drive the real ``proxy_to_local`` gate refusal and return its response."""
    from proxy.router import proxy_to_local

    server_cfg = _server_config()
    _install_router_harness(monkeypatch, server_cfg)
    request = _RouterRequest(SESSION_A, _prompt_body())
    _arm_gate(reason, request, server_cfg)

    response = await proxy_to_local(request, "v1/chat/completions")
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", GATE_REASONS)
async def test_gate_refusal_carries_the_dispatch_reason_marker(
    monkeypatch, reason, caplog
):
    """AC1: the refusal names its reason on the wire **and** in-process.

    Without the marker the fallback loop classifies the 503 as the generic
    ``HTTP 503`` and quarantines the whole local entry — the defect.
    """
    from proxy import server as srv

    caplog.set_level(logging.INFO, logger="llama-proxy")
    response = await _real_gate_refusal(monkeypatch, reason)

    assert response.status_code == 503
    # AC3: the refusal keeps the retryable backend_error shape + Retry-After.
    body = json.loads(bytes(response.body))
    assert body["error"]["code"] == "backend_error"
    assert int(response.headers["Retry-After"]) >= 1
    assert body["retry_after"] == int(response.headers["Retry-After"])

    # AC1: both halves of the marker.
    assert response.headers[admission_reserve.HEADER_LOCAL_DISPATCH_REASON] == reason
    assert getattr(response, admission_reserve.LOCAL_DISPATCH_REASON_ATTR) == reason
    assert admission_reserve.local_dispatch_reason(response) == reason

    # The refusal happened before any local resource was acquired.
    assert SESSION_A not in srv.local_dispatch_records
    assert getattr(srv, "local_active_queries", 0) == 0

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert f"reason={reason}" in messages, messages
    assert "entry=router_dispatch" in messages, messages


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", GATE_REASONS)
async def test_gate_marker_is_classified_as_a_deferral(monkeypatch, reason):
    """AC1/AC2: the marker maps to the gate reason as a deferral, not a fault."""
    response = await _real_gate_refusal(monkeypatch, reason)

    assert provider._local_dispatch_deferral_reason(response, "local") == reason
    # Local-only: a remote 503 is never deferred by a local marker.
    assert provider._local_dispatch_deferral_reason(response, "remote") is None
    # A non-503 with the same marker is never deferred.
    promoted = Response(content=b"{}", status_code=502)
    setattr(promoted, admission_reserve.LOCAL_DISPATCH_REASON_ATTR, reason)
    assert provider._local_dispatch_deferral_reason(promoted, "local") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", GATE_REASONS)
async def test_gate_deferral_applies_no_cooldown_and_records_the_reason(
    monkeypatch, reason
):
    """AC2: no `mark_provider_unavailable`, no failure bump, deferral attempt.

    The attempt record's status and reason are the contract the audit reads:
    `local_dispatch_deferral` naming the gate reason.
    """
    response = await _real_gate_refusal(monkeypatch, reason)
    retry_after = int(response.headers["Retry-After"])

    attempts: list[dict] = []
    cooldown = provider._handle_http_error_with_cooldown(
        response,
        ENTRY,
        "local",
        PROVIDER_COOLDOWN_SECONDS,
        attempts,
        "thrash-gate refusal",
        config=_server_config(),
    )

    assert not provider._is_provider_unavailable(ENTRY), (
        f"a {reason} gate refusal must not quarantine the local entry"
    )
    assert ENTRY not in provider._provider_unavailable_until
    assert provider._provider_failure_count.get(ENTRY, 0) == 0
    assert cooldown == float(retry_after), (
        "the deferral must return the refusal's own Retry-After, got "
        f"{cooldown!r}"
    )
    assert attempts and attempts[0]["status"] == "local_dispatch_deferral", attempts
    assert attempts[0]["reason"] == reason, attempts
    assert "cooldown_seconds" not in attempts[0], (
        "a deferral applies no cooldown, so it must not record one"
    )


@pytest.mark.asyncio
async def test_unlisted_marker_still_quarantines(monkeypatch):
    """AC4: the exemption is reason-scoped, never 'any marked local 503'."""
    response = await _real_gate_refusal(
        monkeypatch, helpers.LOCAL_GATE_REASON_POST_DEADLINE_TERMINAL
    )
    assert admission_reserve.local_dispatch_reason(response) is not None
    setattr(
        response,
        admission_reserve.LOCAL_DISPATCH_REASON_ATTR,
        admission_reserve.LOCAL_FALLBACK_DEFERRAL_EXHAUSTED,
    )
    response.headers[admission_reserve.HEADER_LOCAL_DISPATCH_REASON] = (
        admission_reserve.LOCAL_FALLBACK_DEFERRAL_EXHAUSTED
    )

    attempts: list[dict] = []
    cooldown = provider._handle_http_error_with_cooldown(
        response,
        ENTRY,
        "local",
        PROVIDER_COOLDOWN_SECONDS,
        attempts,
        "a marker that is not a deferral",
        config=_server_config(),
    )

    assert cooldown == PROVIDER_COOLDOWN_SECONDS
    assert provider._is_provider_unavailable(ENTRY), (
        "only the documented deferral markers may skip the entry quarantine"
    )
    assert attempts and attempts[0]["status"] == "http_error", attempts


# ---------------------------------------------------------------------------
# End-to-end: a neighbour session keeps its local route (AC1) + AC3/AC5
# ---------------------------------------------------------------------------


class _FreeSlotConcurrency:
    """A free local slot so the provider-side gate never deflects the request."""

    def __call__(self, *_args, **_kwargs):
        return (0, 1)


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


@pytest.fixture
def fallback_chain(monkeypatch):
    """Drive the real fallback cycle with in-memory local/remote dispatchers."""
    calls: list[tuple[str, str | None]] = []
    stack = contextlib.ExitStack()

    def _install(local_fn):
        async def _local(req, path):
            session = (getattr(req, "headers", {}) or {}).get("x-session-id")
            calls.append(("local", session))
            return local_fn(session)

        async def _remote(req, path, provider_cfg):
            session = (getattr(req, "headers", {}) or {}).get("x-session-id")
            calls.append(("remote", session))
            return _streaming_ok()

        monkeypatch.setattr(
            provider, "_get_local_concurrency_info", _FreeSlotConcurrency()
        )
        stack.enter_context(patch("proxy.router.proxy_to_local", _local))
        stack.enter_context(patch("proxy.server.proxy_to_remote", _remote))

    yield _install, calls
    stack.close()


async def _run_chain(request) -> Response:
    return await provider.proxy_with_fallback(
        request, "v1/chat/completions", _mixed_model_config(), _server_config()
    )


def _chain_request(session_id: str) -> _RouterRequest:
    body = _prompt_body()
    body["stream"] = False
    return _RouterRequest(session_id, body)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", GATE_REASONS)
async def test_neighbour_session_is_not_refused_with_local_in_cooldown(
    monkeypatch, fallback_chain, reason, caplog
):
    """AC1/AC3: the refusing request detours; a neighbour still gets local.

    Session A's local dispatch returns the **real** gate refusal (driven above),
    so this is the measured shape: one request detours to the remote provider,
    and session B's subsequent ``prefer-local`` request is served locally — no
    ``<local> in cooldown`` skip for anyone.
    """
    install, calls = fallback_chain
    refusal = await _real_gate_refusal(monkeypatch, reason)
    # The provider path re-reads the marker from the response it receives, so
    # the response object must survive the hop unchanged (header + attribute).
    install(lambda session: refusal if session == SESSION_A else _ok_response())
    caplog.set_level(logging.INFO, logger="llama-proxy.provider")

    first = await _run_chain(_chain_request(SESSION_A))
    assert first.status_code == 200
    assert first.headers.get("X-Provider") == REMOTE, dict(first.headers)
    assert not provider._is_provider_unavailable(ENTRY), (
        f"the {reason} gate refusal must leave the local entry eligible"
    )
    assert provider._provider_failure_count.get(ENTRY, 0) == 0

    second = await _run_chain(_chain_request(SESSION_B))
    assert second.status_code == 200
    assert second.headers.get("X-Provider") == ENTRY, (
        "a neighbour's gate refusal must not remove local from this session's route"
    )
    assert calls.count(("remote", SESSION_B)) == 0, calls

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert f"{ENTRY} in cooldown" not in messages, messages
    assert "reason=local_provider_cooldown" not in messages, messages


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", GATE_REASONS)
async def test_fallback_line_names_the_gate_reason(
    monkeypatch, fallback_chain, reason, caplog
):
    """AC5: the detour is attributable as the gate reason, not ``HTTP 503``."""
    install, _calls = fallback_chain
    refusal = await _real_gate_refusal(monkeypatch, reason)
    install(lambda session: refusal)
    caplog.set_level(logging.INFO, logger="llama-proxy.provider")

    response = await _run_chain(_chain_request(SESSION_A))
    assert response.status_code == 200
    assert response.headers.get("X-Provider") == REMOTE
    assert response.headers.get(admission_reserve.HEADER_FALLBACK_REASON) == reason, (
        f"expected X-Local-Fallback-Reason={reason!r}; got {dict(response.headers)}"
    )

    fallback_lines = [
        record.getMessage()
        for record in caplog.records
        if "Fallback triggered" in record.getMessage()
    ]
    assert fallback_lines, "the fall-through must be logged once"
    assert any(f"reason={reason}" in line for line in fallback_lines), fallback_lines
    assert not any("reason=HTTP 503" in line for line in fallback_lines), fallback_lines

    # AC5: these refusals never feed the local-route starvation detector.
    snapshot = local_route_starvation.starvation_snapshot(config=_server_config())
    assert snapshot["arms"] == 0, snapshot
    assert reason not in snapshot["reasons"], snapshot


# ---------------------------------------------------------------------------
# Drift guards: one vocabulary across gate, provider, docs
# ---------------------------------------------------------------------------


def test_gate_tokens_are_canonical_and_match_the_gate_constants():
    """The marker, the skip token and the gate's own reasons cannot drift."""
    canonical = admission_reserve.LOCAL_FALLBACK_REASON_TOKENS
    deferrable = provider.DEFERRABLE_LOCAL_DISPATCH_REASONS

    for reason in GATE_REASONS:
        assert reason in canonical, f"{reason!r} must be a canonical reason token"
        assert deferrable.get(reason) == reason, (
            "the deferrable marker must classify as its own canonical token; "
            f"got {deferrable.get(reason)!r}"
        )
        assert admission_reserve.canonical_local_fallback_reason(reason) == reason

    # The gate's constants are what provider.py spells, so a rename on either
    # side fails here instead of silently splitting the vocabulary.
    assert GATE_REASONS == (
        "post_deadline_terminal",
        "prefill_thrash_cooldown",
    )


def _gate_doc_section() -> str:
    text = ROUTING_DOC.read_text(encoding="utf-8")
    start = text.index(DOC_GATE_HEADING)
    rest = text[start:]
    return rest.split(f"\n{DOC_NEXT_HEADING}", 1)[0]


def test_docs_document_the_gate_refusals_as_deferrals():
    """AC6: the routing docs list both refusals as non-quarantining deferrals."""
    section = _gate_doc_section()
    collapsed = " ".join(section.split())

    for reason in GATE_REASONS:
        assert f"`{reason}`" in section, (
            f"the thrash-gate table must document {reason!r}"
        )
    assert "local_dispatch_deferral" in section
    assert "deferral" in collapsed
    assert "defers" in collapsed
    assert "never on local's health" in collapsed.replace("never on local’s health", "never on local's health")

    # The exit table carries the same claim for the token a pinned client sees.
    text = ROUTING_DOC.read_text(encoding="utf-8")
    for reason in GATE_REASONS:
        rows = [
            line for line in text.splitlines()
            if line.startswith("| Router skip") and f"`{reason}`" in line
        ]
        assert rows, f"the local-route exit table must carry a {reason!r} row"
        assert any("local_dispatch_deferral" in row for row in rows), rows
