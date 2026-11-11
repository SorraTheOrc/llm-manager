"""
Test-first: local HTTP 400 observability + deterministic-repeat escalation.

Targets LP-0MTXEBQ4E001BMI2 — "Local HTTP 400 on compacted history forces
silent remote fallthrough".

Problem: a local 400 is currently invisible. The local-4xx branch in
``_proxy_with_fallback_cycle`` records ``http_error_no_cooldown`` and falls
back to the next provider WITHOUT any log line or metric, so a request-shape
rejection that is deterministic for a session (repeats turn after turn on
the same history) silently routes remote every turn — exactly what happened
after compaction in session 01a091ac (63282 -> 35567 tokens, then
``reason=HTTP 400`` from local-qwen3 on every turn).

ACs covered:

- AC1 — Root cause identifiable: the upstream 400 body snippet must be
  logged per occurrence (currently dropped for local providers).
- AC2 — No silent fallthrough: a local 400 repeating for the SAME session
  escalates to a WARNING + ``local_http_400_deterministic`` metric.
- AC3 — Compaction -> local path: a compacted-history request (summary
  marker + reasoning_content, cache_prompt=True) that is within target
  routes to local and succeeds without a remote fallback.
- AC4 — No regression: transient one-off local 400s (different sessions)
  still fall back cleanly without WARNING escalation and without poisoning
  the local provider cooldown.
"""

import json
import logging
import re
from unittest.mock import patch

import proxy.metrics as metrics
import proxy.provider as provider
import pytest
from fastapi.responses import Response

from proxy import fingerprint_backoff as fb


class _DummyRequest:
    """Minimal request stub with optional session headers."""

    def __init__(self, body: bytes = b'{"model":"test"}', headers: dict | None = None):
        self._body = body
        self.headers = headers or {}
        self.method = "POST"
        self.url = type("U", (), {"path": "/v1/chat/completions"})()

    async def body(self):
        return self._body


@pytest.fixture(autouse=True)
def reset_state():
    """Reset cooldown/failure/streak state between tests to avoid leakage."""
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._usage_reset_window.clear()
    provider._local_http_400_streaks.clear()
    fb.reset()
    yield


@pytest.fixture
def mixed_model_config():
    """Model config with a local provider first and a remote fallback."""
    return {
        "providers": [
            {
                "name": "local-qwen3",
                "type": "local",
                "llama_model": "Qwen3",
            },
            {
                "name": "remote-fallback",
                "type": "remote",
                "endpoint": "https://api.anthropic.com/v1",
                "api_key_env": "ANTHROPIC_API_KEY",
            },
        ],
        "aliases": ["plan*"],
    }


def _metric_value(endpoint, status, reason):
    """Return the current counter value for the given labels (0 if never incremented)."""
    try:
        return metrics.proxy_http_errors_total.labels(
            endpoint=endpoint, status=status, reason=reason
        )._value.get()
    except Exception:
        return 0


def _make_mocks(local_responses):
    """Build async mocks for proxy_to_local / proxy_to_remote.

    ``local_responses`` is a list of Responses returned on successive local
    calls (consumed in order). The remote mock always succeeds.
    """
    responses = list(local_responses)
    call_log = []

    async def _mock_proxy_to_local(_req, _path):
        call_log.append("local")
        if responses:
            return responses.pop(0)
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok-local"}}]}),
            status_code=200,
            media_type="application/json",
        )

    async def _mock_proxy_to_remote(_req, _path, _provider_cfg):
        call_log.append("remote")
        return Response(
            content=json.dumps({"choices": [{"message": {"content": "ok-remote"}}]}),
            status_code=200,
            media_type="application/json",
        )

    return _mock_proxy_to_local, _mock_proxy_to_remote, call_log


def _local_400_body(message: str = "invalid_request") -> Response:
    return Response(
        content=json.dumps({"error": {"message": message, "type": "invalid_request_error"}}),
        status_code=400,
        media_type="application/json",
    )


# =====================================================================
# AC1 — upstream body snippet observable per local 400
# =====================================================================


@pytest.mark.asyncio
async def test_local_400_logs_upstream_body_snippet_per_occurrence(mixed_model_config, caplog):
    """A local 400 must log the upstream body snippet (currently silent)."""
    _mock_local, _mock_remote, _ = _make_mocks(
        [_local_400_body(message="missing field `tool_call_id`")]
    )
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
        caplog.at_level(logging.INFO, logger="llama-proxy.provider"),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    assert result.headers.get("X-Provider") == "remote-fallback"
    log_text = caplog.text
    assert "Local HTTP 400 from provider=local-qwen3" in log_text, log_text[:2000]
    assert "tool_call_id" in log_text, (
        f"Expected the local 400 body snippet in logs, got:\n{log_text[:2000]}"
    )
    # AC4: no cooldown poisoning on a transient local 400.
    assert "local-qwen3" not in provider._provider_unavailable_until


@pytest.mark.asyncio
async def test_local_400_increments_http_errors_metric(mixed_model_config):
    """A local 400 must increment proxy_http_errors_total{reason=local_http_400}."""
    _mock_local, _mock_remote, _ = _make_mocks([_local_400_body()])
    request = _DummyRequest()
    cfg = {"provider_cooldown_seconds": 60}

    before = _metric_value("v1/chat/completions", "400", "local_http_400")
    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200
    after = _metric_value("v1/chat/completions", "400", "local_http_400")
    assert after == before + 1, (
        f"Expected local_http_400 counter +1 (before={before}, after={after})"
    )


# =====================================================================
# AC2 — deterministic local 400 (same session) escalates visibly
# =====================================================================

SESSION_HEADERS = {"x-session-id": "sess-compacted-400"}


@pytest.mark.asyncio
async def test_repeated_local_400_same_session_escalates_to_warning(mixed_model_config, caplog):
    """A local 400 repeating for the same session must emit a WARNING."""
    _mock_local, _mock_remote, _ = _make_mocks(
        [_local_400_body("reject-a"), _local_400_body("reject-b")]
    )
    cfg = {"provider_cooldown_seconds": 60}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
        caplog.at_level(logging.WARNING, logger="llama-proxy.provider"),
    ):
        result_first = await provider.proxy_with_fallback(
            _DummyRequest(headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )
        result_second = await provider.proxy_with_fallback(
            _DummyRequest(headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )

    assert result_first.status_code == 200
    assert result_second.status_code == 200
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "Expected at least one WARNING record for the repeat"
    text = "\n".join(r.getMessage() for r in warnings)
    assert "Deterministic local HTTP 400" in text, text[:2000]
    assert "session=sess-compacted-400" in text, text[:2000]
    assert "reject-b" in text, (
        "Expected the repeated 400 body snippet in the WARNING, got:\n" + text[:2000]
    )
    # AC4: escalation must NOT poison the local cooldown.
    assert "local-qwen3" not in provider._provider_unavailable_until


@pytest.mark.asyncio
async def test_repeated_local_400_same_session_increments_deterministic_metric(mixed_model_config):
    """The deterministic escalation must increment the dedicated metric."""
    _mock_local, _mock_remote, _ = _make_mocks(
        [_local_400_body(), _local_400_body()]
    )
    cfg = {"provider_cooldown_seconds": 60}
    before = _metric_value("v1/chat/completions", "400", "local_http_400_deterministic")

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        for _ in range(2):
            result = await provider.proxy_with_fallback(
                _DummyRequest(headers=dict(SESSION_HEADERS)),
                "v1/chat/completions", mixed_model_config, cfg,
            )
            assert result.status_code == 200

    after = _metric_value("v1/chat/completions", "400", "local_http_400_deterministic")
    assert after == before + 1, (
        f"Expected deterministic counter +1 on the repeat "
        f"(before={before}, after={after})"
    )


@pytest.mark.asyncio
async def test_success_after_local_400_resets_streak(mixed_model_config):
    """A successful local dispatch between 400s must reset the streak."""
    _mock_local, _mock_remote, _ = _make_mocks(
        [
            _local_400_body(),
            Response(
                content=json.dumps({"choices": [{"message": {"content": "ok-local"}}]}),
                status_code=200,
                media_type="application/json",
            ),
            _local_400_body(),
        ]
    )
    cfg = {"provider_cooldown_seconds": 60}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        # 1st: local 400 -> remote fallback
        r1 = await provider.proxy_with_fallback(
            _DummyRequest(headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )
        assert r1.headers.get("X-Provider") == "remote-fallback"
        # 2nd: local success (streak reset)
        r2 = await provider.proxy_with_fallback(
            _DummyRequest(headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )
        assert r2.headers.get("X-Provider") == "local-qwen3"
        # 3rd: local 400 again — must NOT escalate (streak was reset)
        r3 = await provider.proxy_with_fallback(
            _DummyRequest(headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )
        assert r3.headers.get("X-Provider") == "remote-fallback"

    assert provider._local_http_400_streaks.get(
        ("sess-compacted-400", "local-qwen3"), {}
    ).get("count") == 1, "Streak must reset on the intervening success"


@pytest.mark.asyncio
async def test_streaming_success_after_local_400_resets_streak(mixed_model_config):
    """A streaming local success between 400s must reset the streak too.

    Production local traffic streams (``stream: true``); the reset must apply
    on the streaming success path, not only the non-streaming one.
    """
    from starlette.responses import StreamingResponse

    async def _gen():
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield b'data: [DONE]\n\n'

    _mock_local, _mock_remote, _ = _make_mocks(
        [
            _local_400_body(),
            StreamingResponse(content=_gen(), status_code=200, media_type="text/event-stream"),
            _local_400_body(),
        ]
    )
    cfg = {"provider_cooldown_seconds": 60}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        r1 = await provider.proxy_with_fallback(
            _DummyRequest(headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )
        assert r1.headers.get("X-Provider") == "remote-fallback"
        r2 = await provider.proxy_with_fallback(
            _DummyRequest(headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )
        assert r2.headers.get("X-Provider") == "local-qwen3"
        r3 = await provider.proxy_with_fallback(
            _DummyRequest(headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )
        assert r3.headers.get("X-Provider") == "remote-fallback"

    assert provider._local_http_400_streaks.get(
        ("sess-compacted-400", "local-qwen3"), {}
    ).get("count") == 1, "Streaming local success must reset the streak"


# =====================================================================
# AC4 — transient local 400s (different sessions) stay quiet
# =====================================================================


@pytest.mark.asyncio
async def test_transient_local_400_different_sessions_no_escalation(mixed_model_config, caplog):
    """One-off local 400s on different sessions must NOT escalate to WARNING."""
    _mock_local, _mock_remote, _ = _make_mocks(
        [_local_400_body("noise-a"), _local_400_body("noise-b")]
    )
    cfg = {"provider_cooldown_seconds": 60}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
        caplog.at_level(logging.WARNING, logger="llama-proxy.provider"),
    ):
        for idx in range(2):
            result = await provider.proxy_with_fallback(
                _DummyRequest(headers={"x-session-id": f"transient-{idx}"}),
                "v1/chat/completions", mixed_model_config, cfg,
            )
            assert result.status_code == 200
            assert "local-qwen3" not in provider._provider_unavailable_until

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
        "Transient local 400s on different sessions must not escalate:\n"
        + caplog.text[:2000]
    )


# =====================================================================
# AC3 — compacted-history request within target routes local
# =====================================================================

_COMPACTED_BODY = json.dumps(
    {
        "model": "plan",
        "messages": [
            {
                "role": "user",
                "content": (
                    "The conversation history before this point was compacted "
                    "into the following summary:\n\n<summary>\nFolded summary.\n"
                    "</summary>"
                ),
            },
            {"role": "user", "content": "Say hi"},
            {
                "role": "assistant",
                "content": "Hi there! How can I help you today?",
                "reasoning_content": "Hmm, the user said \"Say hi\". This is a greeting.",
            },
            {"role": "user", "content": "Continue"},
        ],
        "stream": False,
        "cache_prompt": True,
        "max_tokens": 8192,
    }
).encode()


@pytest.mark.asyncio
async def test_compacted_history_within_target_routes_local_without_fallback(
    mixed_model_config,
):
    """A compacted-history request (summary marker, reasoning_content,
    cache_prompt=True) that is within target must dispatch to local and
    succeed — no silent remote fallthrough on the post-compaction turn.

    Mirrors the AC3 log contract: ``Stream started: provider=local`` follows
    ``session_compaction applied`` with est_after under target.
    """
    _mock_local, _mock_remote, call_log = _make_mocks([])
    request = _DummyRequest(body=_COMPACTED_BODY, headers=dict(SESSION_HEADERS))
    cfg = {
        "provider_cooldown_seconds": 60,
        "local_large_context_cold_cache_threshold": 40000,
    }

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        result = await provider.proxy_with_fallback(
            request, "v1/chat/completions", mixed_model_config, cfg
        )

    assert result.status_code == 200, result.body.decode()
    assert result.headers.get("X-Provider") == "local-qwen3"
    assert call_log == ["local"], (
        f"Compacted-within-target must dispatch local only, got: {call_log}"
    )


# =====================================================================
# LP-0MW95BCVV004LO0K — per-prompt-fingerprint backoff wiring
#
# Parent AC2: repeat requests with the same prompt fingerprint are backed off
# on an exponential schedule (1 -> 5 -> 15 min cap), reset by a successful
# local stream or by a changed history. The module's own contract lives in
# ``test_provider_fingerprint_backoff.py``; these tests pin the *wiring* — the
# local-400 path must feed that state from the real request body, and the local
# success path must clear it.
# =====================================================================


def _chat_body(*contents: str, model: str = "test") -> bytes:
    """A chat-completions body whose history is ``contents``."""
    return json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": content} for content in contents],
        }
    ).encode()


def _fingerprint_of(body: bytes) -> str:
    return fb.compute_prompt_fingerprint(json.loads(body)["messages"])


@pytest.mark.asyncio
async def test_local_400_records_a_fingerprint_backoff_entry(mixed_model_config):
    """A local 400 must open the 1-minute rung for the request's history."""
    body = _chat_body("poisoned history")
    _mock_local, _mock_remote, _ = _make_mocks([_local_400_body()])
    cfg = {"provider_cooldown_seconds": 60}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )

    assert result.status_code == 200
    entry = fb.get_backoff_entry(_fingerprint_of(body))
    assert entry is not None, "a local 400 must record a per-fingerprint backoff entry"
    assert entry["session_id"] == "sess-compacted-400"
    assert entry["resend_count"] == 1
    assert entry["retry_after"] == 60.0, "the first rung is 1 minute"
    assert fb.check_backoff_expired(_fingerprint_of(body)) is False


@pytest.mark.asyncio
async def test_repeated_local_400_escalates_the_shared_fingerprint_ladder(
    mixed_model_config,
):
    """Two sessions resending the same history share one 1 -> 5 -> 15 min ladder."""
    body = _chat_body("identical", "bad history")
    _mock_local, _mock_remote, _ = _make_mocks([_local_400_body(), _local_400_body()])
    cfg = {"provider_cooldown_seconds": 60}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        for session in ("sess-a", "sess-b"):
            result = await provider.proxy_with_fallback(
                _DummyRequest(body=body, headers={"x-session-id": session}),
                "v1/chat/completions", mixed_model_config, cfg,
            )
            assert result.status_code == 200

    fingerprint = _fingerprint_of(body)
    entry = fb.get_backoff_entry(fingerprint)
    assert entry["resend_count"] == 2, "the same history must continue the shared ladder"
    assert entry["retry_after"] == 300.0
    assert entry["session_id"] == "sess-b", "the entry records the last session that sent it"


@pytest.mark.asyncio
async def test_changed_history_clears_the_previous_fingerprint_backoff(
    mixed_model_config,
):
    """A compaction rewrite must drop the old history's verdict (parent AC2)."""
    old_body = _chat_body("old compacted history")
    new_body = _chat_body("rewritten after compaction")
    _mock_local, _mock_remote, _ = _make_mocks([_local_400_body(), _local_400_body()])
    cfg = {"provider_cooldown_seconds": 60}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        for body in (old_body, new_body):
            result = await provider.proxy_with_fallback(
                _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
                "v1/chat/completions", mixed_model_config, cfg,
            )
            assert result.status_code == 200

    assert fb.get_backoff_entry(_fingerprint_of(old_body)) is None, (
        "the session moved to a different history, so the old verdict must be dropped"
    )
    assert fb.get_backoff_entry(_fingerprint_of(new_body))["resend_count"] == 1, (
        "the rewritten history must start at the bottom of the ladder"
    )


@pytest.mark.asyncio
async def test_local_success_clears_the_fingerprint_backoff(mixed_model_config):
    """A successful local stream resets the history's ladder (parent AC2)."""
    body = _chat_body("recovering history")
    local_ok = Response(
        content=json.dumps({"choices": [{"message": {"content": "ok-local"}}]}),
        status_code=200,
        media_type="application/json",
    )
    _mock_local, _mock_remote, _ = _make_mocks([_local_400_body(), local_ok])
    cfg = {"provider_cooldown_seconds": 60}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        first = await provider.proxy_with_fallback(
            _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )
        assert first.headers.get("X-Provider") == "remote-fallback"
        assert fb.get_backoff_entry(_fingerprint_of(body)) is not None

        second = await provider.proxy_with_fallback(
            _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )
        assert second.headers.get("X-Provider") == "local-qwen3"

    assert fb.get_backoff_entry(_fingerprint_of(body)) is None, (
        "a local success must clear the backoff for the history it served"
    )


@pytest.mark.asyncio
async def test_streaming_local_success_clears_the_fingerprint_backoff(
    mixed_model_config,
):
    """Production local traffic streams; the streaming success must reset too."""
    from starlette.responses import StreamingResponse

    async def _gen():
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield b'data: [DONE]\n\n'

    body = _chat_body("recovering stream")
    _mock_local, _mock_remote, _ = _make_mocks(
        [
            _local_400_body(),
            StreamingResponse(content=_gen(), status_code=200, media_type="text/event-stream"),
        ]
    )
    cfg = {"provider_cooldown_seconds": 60}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        await provider.proxy_with_fallback(
            _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )
        assert fb.get_backoff_entry(_fingerprint_of(body)) is not None

        streamed = await provider.proxy_with_fallback(
            _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )
        assert streamed.headers.get("X-Provider") == "local-qwen3"

    assert fb.get_backoff_entry(_fingerprint_of(body)) is None


@pytest.mark.asyncio
async def test_streak_entry_carries_the_fingerprint_information(mixed_model_config):
    """IAC7: the existing streak tracking is extended, not replaced."""
    body = _chat_body("poisoned history")
    _mock_local, _mock_remote, _ = _make_mocks([_local_400_body()])
    cfg = {"provider_cooldown_seconds": 60}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        await provider.proxy_with_fallback(
            _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )

    streak = provider._local_http_400_streaks[("sess-compacted-400", "local-qwen3")]
    assert streak["count"] == 1, "the pre-existing streak semantics are unchanged"
    assert streak["fingerprint"] == _fingerprint_of(body)
    assert streak["backoff_resend_count"] == 1
    assert streak["backoff_retry_after"] == 60.0
    assert streak["backoff_until"] > streak["start"]


@pytest.mark.asyncio
async def test_local_400_without_messages_records_no_fingerprint_verdict(
    mixed_model_config,
):
    """A body with no conversation has nothing to fingerprint — no verdict."""
    _mock_local, _mock_remote, _ = _make_mocks([_local_400_body()])
    cfg = {"provider_cooldown_seconds": 60}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )

    assert result.status_code == 200
    assert provider._local_http_400_streaks[("sess-compacted-400", "local-qwen3")]["count"] == 1
    assert fb.get_backoff_entry(fb.compute_prompt_fingerprint([])) is None


def test_fingerprint_backoff_state_reports_the_verdict_for_a_request_body():
    """The read-side helper the typed-error layer consumes (I2 contract)."""
    body = json.loads(_chat_body("bad"))
    fingerprint = _fingerprint_of(_chat_body("bad"))
    assert provider.fingerprint_backoff_state("sess", body) is None

    fb.record_backoff(fingerprint, "sess", now=100.0)

    state = provider.fingerprint_backoff_state("sess", body, now=130.0)
    assert state["fingerprint"] == fingerprint
    assert state["expired"] is False
    assert state["remaining_seconds"] == pytest.approx(30.0)
    assert state["entry"]["resend_count"] == 1

    later = provider.fingerprint_backoff_state("sess", body, now=160.0)
    assert later["expired"] is True
    assert later["remaining_seconds"] == 0.0
    assert later["entry"]["resend_count"] == 1, "an expired entry survives for the next rung"

    # No conversation history -> nothing to report.
    assert provider.fingerprint_backoff_state("sess", {"model": "test"}) is None


# =====================================================================
# LP-0MW95BDJQ002DSVI — typed deterministic client error, before fallback
#
# Parent AC1: a deterministic local 400 returns immediately with a
# machine-readable ``deterministic_client_error`` (reason=identical_history)
# and a short ``Retry-After``; no chain-hold cycle is consumed and no remote
# dispatch happens. The full contract (chain-hold skip, single-400 fallback,
# unchanged observability) is T2's scope; these tests pin the *wiring* — the
# local-4xx branch must answer an identical resend instead of falling through.
# =====================================================================


def _reject_body(message: str = "reject") -> str:
    return json.dumps({"error": {"message": message, "type": "invalid_request_error"}})


@pytest.mark.asyncio
async def test_identical_history_resend_returns_typed_deterministic_client_error(
    mixed_model_config,
):
    """The second identical local 400 is answered, not routed remote again."""
    from proxy import fingerprint_backoff as fb

    body = _chat_body("poisoned", "history")
    _mock_local, _mock_remote, call_log = _make_mocks(
        [_local_400_body("reject"), _local_400_body("reject")]
    )
    cfg = {"provider_cooldown_seconds": 60}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        first = await provider.proxy_with_fallback(
            _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )
        assert first.status_code == 200, "a single 400 must still fall back"
        assert first.headers.get("X-Provider") == "remote-fallback"

        second = await provider.proxy_with_fallback(
            _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )

    assert second.status_code == 400
    payload = json.loads(second.body)
    assert payload["error"] == "deterministic_client_error"
    assert payload["reason"] == "identical_history"
    assert isinstance(payload["retry_after"], int)
    assert payload["retry_after"] > 0
    assert payload["body_snippet"] == _reject_body("reject")
    assert second.headers["Retry-After"] == str(payload["retry_after"]), (
        "the header and the body must advertise the same wait"
    )
    assert (
        second.headers[provider.DETERMINISTIC_CLIENT_ERROR_HEADER]
        == "identical_history"
    )
    assert call_log == ["local", "remote", "local"], (
        "the typed 400 must terminate the request — no remote dispatch: "
        f"{call_log}"
    )
    # The ladder still advanced: the typed error reports the *current* verdict.
    entry = fb.get_backoff_entry(fb.compute_prompt_fingerprint(json.loads(body)["messages"]))
    assert entry["resend_count"] == 2
    assert payload["retry_after"] == pytest.approx(300, abs=1), (
        "the second rung of the ladder is advertised"
    )


@pytest.mark.asyncio
async def test_changed_history_resend_does_not_return_typed_error(
    mixed_model_config,
):
    """A session repeating 400s with a *changed* history is not 'identical'."""
    _mock_local, _mock_remote, call_log = _make_mocks(
        [_local_400_body("a"), _local_400_body("b")]
    )
    cfg = {"provider_cooldown_seconds": 60}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        for body in (
            _chat_body("first history"),
            _chat_body("rewritten", "history"),
        ):
            result = await provider.proxy_with_fallback(
                _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
                "v1/chat/completions", mixed_model_config, cfg,
            )
            assert result.status_code == 200, (
                "a changed history is not a deterministic repeat — it must fall back"
            )
            assert result.headers.get("X-Provider") == "remote-fallback"

    assert call_log == ["local", "remote", "local", "remote"]


@pytest.mark.asyncio
async def test_typed_400_increments_the_deterministic_client_error_metric(
    mixed_model_config,
):
    """The typed error is observable as its own metric reason (parent AC1)."""
    body = _chat_body("poisoned", "history")
    _mock_local, _mock_remote, _ = _make_mocks([_local_400_body(), _local_400_body()])
    cfg = {"provider_cooldown_seconds": 60}
    before = _metric_value("v1/chat/completions", "400", "deterministic_client_error")

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        for _ in range(2):
            await provider.proxy_with_fallback(
                _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
                "v1/chat/completions", mixed_model_config, cfg,
            )

    after = _metric_value("v1/chat/completions", "400", "deterministic_client_error")
    assert after == before + 1, (
        f"expected one typed error (before={before}, after={after})"
    )


# =====================================================================
# LP-0MW959GMC0042J7O (T2) — typed deterministic client error contract
#
# Parent AC1: a deterministic local 400 returns HTTP 400 with a
# machine-readable ``deterministic_client_error`` body and a ``Retry-After``
# header, consumes no chain-hold cycle, and leaves the pre-existing local-400
# observability untouched. I2 (LP-0MW95BDJQ002DSVI) owns the wiring; these
# tests pin the *contract* the wedged client (and the taxonomy in I4) depends
# on.
# =====================================================================


def _typed_error_payload(response) -> dict:
    """Decode a typed-400 response body, asserting it is JSON first."""
    assert response.headers["content-type"].startswith("application/json"), (
        "the typed error must be a JSON document, not SSE: "
        f"{response.headers.get('content-type')!r}"
    )
    return json.loads(response.body)


@pytest.mark.asyncio
async def test_typed_400_body_and_retry_after_header_match_the_contract(
    mixed_model_config,
):
    """TAC1 + TAC2: exact machine-readable body shape and a Retry-After header."""
    body = _chat_body("compacted", "poisoned history")
    _mock_local, _mock_remote, _ = _make_mocks(
        [_local_400_body("reject-a"), _local_400_body("reject-b")]
    )
    cfg = {"provider_cooldown_seconds": 60}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        first = await provider.proxy_with_fallback(
            _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )
        assert first.status_code == 200
        typed = await provider.proxy_with_fallback(
            _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )

    assert typed.status_code == 400, "TAC1: the deterministic repeat is a 400"
    payload = _typed_error_payload(typed)
    assert set(payload) == {"error", "reason", "retry_after", "body_snippet"}, (
        f"TAC1: the body is a fixed contract, got keys {sorted(payload)}"
    )
    assert payload["error"] == "deterministic_client_error"
    assert payload["reason"] == "identical_history"
    assert isinstance(payload["retry_after"], int), (
        "TAC1: retry_after is an integer number of seconds"
    )
    assert 1 <= payload["retry_after"] <= fb.BACKOFF_MAX_SECONDS, (
        f"TAC1: retry_after must stay inside the ladder, got {payload['retry_after']}"
    )
    assert payload["body_snippet"] == _reject_body("reject-b"), (
        "TAC1: the snippet carries the upstream rejection that caused it"
    )
    assert len(payload["body_snippet"]) <= 512
    # TAC2: the header and the body must not disagree.
    assert typed.headers["Retry-After"] == str(payload["retry_after"])


@pytest.mark.asyncio
async def test_typed_400_consumes_no_chain_hold_cycle(
    mixed_model_config, caplog
):
    """TAC3: the typed error is returned before fallback routing / chain hold."""
    body = _chat_body("compacted", "poisoned history")
    _mock_local, _mock_remote, call_log = _make_mocks(
        [_local_400_body(), _local_400_body()]
    )
    # Chain hold is deliberately ENABLED: if the typed error ever fell through
    # to the exhaustion path, this config would hold and replay the cycle.
    cfg = {
        "provider_cooldown_seconds": 60,
        "chain_hold_seconds": 5,
        "chain_hold_max_cycles": 2,
    }
    hold_events: list[tuple] = []

    def _record_hold(*args, **kwargs):
        hold_events.append(args)

    second_request = {}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
        patch("proxy.provider._log_chain_hold_start", _record_hold),
        caplog.at_level(logging.INFO, logger="llama-proxy.provider"),
    ):
        await provider.proxy_with_fallback(
            _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )
        second_request["request"] = _DummyRequest(body=body, headers=dict(SESSION_HEADERS))
        typed = await provider.proxy_with_fallback(
            second_request["request"],
            "v1/chat/completions", mixed_model_config, cfg,
        )

    assert typed.status_code == 400
    assert hold_events == [], "TAC3: no chain-hold cycle may be started"
    request = second_request["request"]
    assert provider._get_chain_hold_cycle(request) == 0, (
        "TAC3: the typed error must not advance the chain-hold cycle"
    )
    assert provider._get_chain_hold_consumed(request) == 0.0, (
        "TAC3: no hold budget may be consumed"
    )
    assert "chain_hold_cycle=1" not in caplog.text
    assert call_log == ["local", "remote", "local"], (
        f"TAC3: no dispatch may follow the typed 400, got {call_log}"
    )


@pytest.mark.asyncio
async def test_single_local_400_falls_back_without_typed_error(
    mixed_model_config,
):
    """TAC4: a single (non-deterministic) local 400 still falls back normally."""
    body = _chat_body("only", "one rejection")
    _mock_local, _mock_remote, _ = _make_mocks([_local_400_body()])
    cfg = {"provider_cooldown_seconds": 60}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        result = await provider.proxy_with_fallback(
            _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )

    assert result.status_code == 200
    assert result.headers.get("X-Provider") == "remote-fallback"
    assert provider.DETERMINISTIC_CLIENT_ERROR_HEADER not in result.headers
    assert b"deterministic_client_error" not in result.body


@pytest.mark.asyncio
async def test_typed_400_keeps_existing_local_400_observability(
    mixed_model_config, caplog
):
    """TAC5: the LP-0MTXEBQ4E001BMI2 INFO log + metrics still fire."""
    body = _chat_body("compacted", "poisoned history")
    _mock_local, _mock_remote, _ = _make_mocks(
        [_local_400_body("first-reject"), _local_400_body("second-reject")]
    )
    cfg = {"provider_cooldown_seconds": 60}
    info_before = _metric_value("v1/chat/completions", "400", "local_http_400")
    deterministic_before = _metric_value(
        "v1/chat/completions", "400", "local_http_400_deterministic"
    )

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
        caplog.at_level(logging.INFO, logger="llama-proxy.provider"),
    ):
        for _ in range(2):
            await provider.proxy_with_fallback(
                _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
                "v1/chat/completions", mixed_model_config, cfg,
            )

    assert _metric_value(
        "v1/chat/completions", "400", "local_http_400"
    ) == info_before + 2, "TAC5: each occurrence still increments local_http_400"
    assert _metric_value(
        "v1/chat/completions", "400", "local_http_400_deterministic"
    ) == deterministic_before + 1, "TAC5: the repeat still escalates its metric"
    assert "Local HTTP 400 from provider=local-qwen3" in caplog.text
    assert "first-reject" in caplog.text and "second-reject" in caplog.text, (
        "TAC5: the per-occurrence body snippet log must be unchanged"
    )
    assert "Deterministic local HTTP 400" in caplog.text, (
        "TAC5: the WARNING escalation must be unchanged"
    )


@pytest.mark.asyncio
async def test_typed_400_is_a_json_error_even_for_a_streaming_request(
    mixed_model_config,
):
    """A wedged streaming session gets the typed JSON error, not an SSE hang."""
    body = json.dumps(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "streaming poisoned history"}],
            "stream": True,
        }
    ).encode()
    _mock_local, _mock_remote, call_log = _make_mocks(
        [_local_400_body("reject"), _local_400_body("reject")]
    )
    cfg = {"provider_cooldown_seconds": 60}

    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        first = await provider.proxy_with_fallback(
            _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )
        assert first.status_code == 200
        typed = await provider.proxy_with_fallback(
            _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
            "v1/chat/completions", mixed_model_config, cfg,
        )

    assert typed.status_code == 400
    payload = _typed_error_payload(typed)
    assert payload["error"] == "deterministic_client_error"
    assert call_log == ["local", "remote", "local"]


# =====================================================================
# LP-0MW95BE7X00574OL (I3) — terminal breaker: ``session_wedged``
#
# Parent AC3: after ``deterministic_client_error_escalation_threshold``
# (default 5) consecutive backoff-expired identical resends the client is told
# the session is wedged — ``{error: "session_wedged", cause:
# "repeated_deterministic_400", hint: "compaction may help"}`` — instead of
# being handed yet another typed 400 it has already ignored. I3 owns the
# *wiring* into the typed-400 path; the mechanism's own contract is pinned in
# ``test_provider_fingerprint_backoff.py`` (T3, LP-0MW959H72003PF67).
# =====================================================================


def _json_error_payload(response) -> dict:
    """Decode a JSON error response body, asserting it is JSON first."""
    assert response.headers["content-type"].startswith("application/json"), (
        "a terminal error must be a JSON document, not SSE: "
        f"{response.headers.get('content-type')!r}"
    )
    return json.loads(response.body)


async def _send_identical_resends(
    body: bytes,
    count: int,
    model_config: dict,
    cfg: dict | None = None,
    *,
    session: str = "sess-wedged",
    local_responses=None,
):
    """Dispatch ``count`` byte-identical resends, collecting every response.

    The local mock rejects with a 400 unless ``local_responses`` overrides the
    sequence (used to interleave a successful local stream); the remote mock
    always succeeds, so a request that falls back is visible as a 200.
    """
    responses = (
        list(local_responses)
        if local_responses is not None
        else [_local_400_body() for _ in range(count)]
    )
    _mock_local, _mock_remote, call_log = _make_mocks(responses)
    results = []
    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        for _ in range(count):
            results.append(
                await provider.proxy_with_fallback(
                    _DummyRequest(body=body, headers={"x-session-id": session}),
                    "v1/chat/completions", model_config, cfg or {"provider_cooldown_seconds": 60},
                )
            )
    return results, call_log


@pytest.mark.asyncio
async def test_fifth_identical_resend_returns_terminal_session_wedged(
    mixed_model_config,
):
    """IAC1 + IAC4: the 5th identical resend is terminal, 400, Retry-After >= 60."""
    body = _chat_body("wedged", "history")
    wedged_before = _metric_value("v1/chat/completions", "400", "session_wedged")
    typed_before = _metric_value("v1/chat/completions", "400", "deterministic_client_error")

    results, call_log = await _send_identical_resends(body, 5, mixed_model_config)

    assert [r.status_code for r in results] == [200, 400, 400, 400, 400], (
        "the first 400 falls back; resends 2-4 are typed 400s; the 5th is terminal"
    )
    for earlier in results[1:4]:
        assert _json_error_payload(earlier)["error"] == "deterministic_client_error", (
            "only the resend that reaches the escalation threshold is terminal"
        )

    terminal = results[4]
    payload = _json_error_payload(terminal)
    assert payload["error"] == "session_wedged"
    assert payload["cause"] == "repeated_deterministic_400"
    assert payload["hint"] == "compaction may help"
    assert isinstance(payload["retry_after"], int)
    assert payload["retry_after"] >= 60, (
        f"IAC4: a wedged session must not be told to retry soon, got {payload['retry_after']}"
    )
    assert terminal.headers["Retry-After"] == str(payload["retry_after"]), (
        "IAC4: the header and the body must advertise the same wait"
    )
    assert call_log == ["local", "remote"] + ["local"] * 4, (
        f"the terminal error must terminate the request — no remote dispatch: {call_log}"
    )
    assert _metric_value(
        "v1/chat/completions", "400", "session_wedged"
    ) == wedged_before + 1, "the terminal error has its own metric reason"
    assert _metric_value(
        "v1/chat/completions", "400", "deterministic_client_error"
    ) == typed_before + 3, (
        "the three typed 400s before the threshold keep their own metric reason"
    )


@pytest.mark.asyncio
async def test_session_wedged_is_distinguishable_from_the_typed_error(
    mixed_model_config,
):
    """IAC5: herdr can tell the terminal error from the typed one by headers."""
    body = _chat_body("distinguishable", "history")

    results, _ = await _send_identical_resends(body, 5, mixed_model_config)

    typed, terminal = results[3], results[4]
    assert (
        typed.headers[provider.DETERMINISTIC_CLIENT_ERROR_HEADER]
        == "identical_history"
    ), "the typed 400 keeps its documented header value"
    assert provider.SESSION_WEDGED_HEADER not in typed.headers, (
        "the typed error must not be mistaken for a terminal one"
    )
    assert terminal.headers[provider.SESSION_WEDGED_HEADER] == "repeated_deterministic_400"
    assert (
        terminal.headers[provider.DETERMINISTIC_CLIENT_ERROR_HEADER]
        == "session_wedged"
    ), "the shared header carries a distinct value on the terminal error"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {fb.ESCALATION_THRESHOLD_CONFIG_KEY: 3},
        {"server": {fb.ESCALATION_THRESHOLD_CONFIG_KEY: 3}},
    ],
    ids=["top-level", "server-section"],
)
async def test_session_wedged_threshold_is_configurable(mixed_model_config, overrides):
    """IAC2: ``deterministic_client_error_escalation_threshold`` moves the trip point."""
    body = _chat_body("configured", "threshold")

    results, _ = await _send_identical_resends(
        body, 3, mixed_model_config, {"provider_cooldown_seconds": 60, **overrides}
    )

    assert _json_error_payload(results[1])["error"] == "deterministic_client_error", (
        "below the configured threshold the typed error is still the answer"
    )
    assert _json_error_payload(results[2])["error"] == "session_wedged"


@pytest.mark.asyncio
async def test_terminal_breaker_never_fires_on_a_lone_local_400(
    mixed_model_config,
):
    """A degenerate threshold (1) must still not convict a single 400."""
    body = _chat_body("lone", "rejection")

    results, _ = await _send_identical_resends(
        body,
        1,
        mixed_model_config,
        {
            "provider_cooldown_seconds": 60,
            fb.ESCALATION_THRESHOLD_CONFIG_KEY: 1,
        },
    )

    assert results[0].status_code == 200
    assert results[0].headers.get("X-Provider") == "remote-fallback"
    assert provider.SESSION_WEDGED_HEADER not in results[0].headers


@pytest.mark.asyncio
async def test_local_success_resets_the_terminal_breaker(mixed_model_config):
    """IAC3: a successful stream resets the counter (no terminal error after)."""
    body = _chat_body("recovering", "history")
    local_ok = Response(
        content=json.dumps({"choices": [{"message": {"content": "ok-local"}}]}),
        status_code=200,
        media_type="application/json",
    )
    # Four rejections (one short of the threshold), a successful local stream,
    # then four more rejections: without the reset the second run would trip.
    responses = [_local_400_body()] * 4 + [local_ok] + [_local_400_body()] * 4

    results, _ = await _send_identical_resends(
        body, 9, mixed_model_config, local_responses=responses
    )

    assert results[4].status_code == 200
    assert results[4].headers.get("X-Provider") == "local-qwen3", (
        "the interleaved success must actually be served locally"
    )
    for result in results:
        if result.status_code == 400:
            assert _json_error_payload(result)["error"] == "deterministic_client_error", (
                "IAC3: a success between rejections resets the counter"
            )
    assert _json_error_payload(results[8])["error"] == "deterministic_client_error"


@pytest.mark.asyncio
async def test_changed_history_does_not_trip_the_terminal_breaker(
    mixed_model_config,
):
    """IAC3: a different fingerprint restarts the ladder, so 5 rejections total
    across two histories never escalate."""
    first_body = _chat_body("first", "history")
    second_body = _chat_body("second", "history")
    responses = [_local_400_body()] * 8

    _mock_local, _mock_remote, _ = _make_mocks(responses)
    results = []
    with (
        patch("proxy.router.proxy_to_local", _mock_local),
        patch("proxy.server.proxy_to_remote", _mock_remote),
    ):
        for body in [first_body] * 4 + [second_body] * 4:
            results.append(
                await provider.proxy_with_fallback(
                    _DummyRequest(body=body, headers=dict(SESSION_HEADERS)),
                    "v1/chat/completions",
                    mixed_model_config,
                    {"provider_cooldown_seconds": 60},
                )
            )

    errors = [
        _json_error_payload(r)["error"]
        for r in results
        if r.status_code == 400
    ]
    assert "session_wedged" not in errors, (
        f"IAC3: a changed history restarts the count, got {errors}"
    )
    assert errors.count("deterministic_client_error") == 6, (
        "each history contributes three typed 400s (its first 400 falls back)"
    )


# =====================================================================
# LP-0MW959HRZ00231FN (T4) — the usage-report taxonomy contract
#
# Parent AC5: the proxy-usage-analysis taxonomy reports
# ``deterministic_client_error`` and ``session_wedged`` as their own
# categories. The report reads *log lines*, not Prometheus counters, so the
# WARNING each response emits is that contract: its exact prefix is what
# ``.pi/skills/proxy-usage-analysis/scripts/log_parser.py`` classifies (pinned
# from the parser side by
# ``tests/test_error_taxonomy_deterministic_client_error.py``).
#
# Pinned here so a reworded log line fails a test instead of silently
# dropping the category from the daily report. The counter reasons themselves
# are already pinned by
# ``test_typed_400_increments_the_deterministic_client_error_metric`` and
# ``test_fifth_identical_resend_returns_terminal_session_wedged``.
# =====================================================================

TYPED_LOG_PREFIX = "Deterministic client error: "
WEDGED_LOG_PREFIX = "Session wedged: "


def _provider_messages_starting_with(caplog, prefix: str) -> list[str]:
    """Every ``llama-proxy.provider`` message that starts with ``prefix``."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "llama-proxy.provider"
        and record.getMessage().startswith(prefix)
    ]


def _assert_taxonomy_fields(message: str, resend_count: int) -> None:
    """The attribution fields the taxonomy parses off the line."""
    assert "provider=local-qwen3" in message, message
    assert "model=v1/chat/completions" in message, message
    assert f"session={SESSION_HEADERS['x-session-id']}" in message, message
    assert f"resend_count={resend_count}" in message, message
    assert re.search(r"\bretry_after=\d+s\b", message), message


@pytest.mark.asyncio
async def test_typed_400_logs_the_taxonomy_line(mixed_model_config, caplog):
    """TAC1: the typed error emits the line the taxonomy classifies."""
    body = _chat_body("taxonomy", "typed")
    with caplog.at_level(logging.WARNING, logger="llama-proxy.provider"):
        results, _ = await _send_identical_resends(
            body,
            2,
            mixed_model_config,
            {"provider_cooldown_seconds": 60},
            session=SESSION_HEADERS["x-session-id"],
        )

    assert _json_error_payload(results[1])["error"] == "deterministic_client_error"
    lines = _provider_messages_starting_with(caplog, TYPED_LOG_PREFIX)
    assert len(lines) == 1, (
        "one typed 400 must log exactly one taxonomy line, got "
        f"{len(lines)}: {lines}"
    )
    _assert_taxonomy_fields(lines[0], resend_count=2)


@pytest.mark.asyncio
async def test_terminal_breaker_logs_the_taxonomy_line(mixed_model_config, caplog):
    """TAC2: the terminal error emits the line the taxonomy classifies."""
    body = _chat_body("taxonomy", "wedged")
    with caplog.at_level(logging.WARNING, logger="llama-proxy.provider"):
        results, _ = await _send_identical_resends(
            body,
            5,
            mixed_model_config,
            {"provider_cooldown_seconds": 60},
            session=SESSION_HEADERS["x-session-id"],
        )

    assert _json_error_payload(results[4])["error"] == "session_wedged"
    lines = _provider_messages_starting_with(caplog, WEDGED_LOG_PREFIX)
    assert len(lines) == 1, (
        "the terminal error must log exactly one taxonomy line, got "
        f"{len(lines)}: {lines}"
    )
    _assert_taxonomy_fields(lines[0], resend_count=5)
    # The terminal error replaces the typed one on that resend: the ladder's
    # three earlier typed 400s are the only typed lines.
    typed_lines = _provider_messages_starting_with(caplog, TYPED_LOG_PREFIX)
    assert len(typed_lines) == 3, (
        "resends 2-4 are typed errors; the 5th is terminal, got "
        f"{len(typed_lines)} typed lines"
    )
