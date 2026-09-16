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
from unittest.mock import patch

import proxy.metrics as metrics
import proxy.provider as provider
import pytest
from fastapi.responses import Response


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
    provider._local_http_400_streaks.clear()
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
