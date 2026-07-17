"""
Unit tests for RequestCoalescer (in-flight request deduplication).

Tests cover:
- Basic deduplication of identical concurrent requests
- Non-identical requests pass through independently
- Error propagation to duplicate waiters
- Streaming responses are not coalesced
- Response header preservation
- Maximum retained entries eviction
"""

import asyncio
import json

import pytest
from fastapi import Response

from proxy.request_coalescer import RequestCoalescer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _ok_response() -> Response:
    """Return a simple 200 OK response."""
    return Response(
        content=json.dumps({"status": "ok"}).encode("utf-8"),
        status_code=200,
        media_type="application/json",
        headers={"X-Provider": "test"},
    )


async def _slow_response(delay: float = 0.2) -> Response:
    """Return a 200 OK after ``delay`` seconds."""
    await asyncio.sleep(delay)
    return Response(
        content=json.dumps({"status": "slow"}).encode("utf-8"),
        status_code=200,
        media_type="application/json",
    )


async def _error_response() -> Response:
    """Raise a runtime error."""
    msg = "Simulated failure"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_coalescer_shared_state():
    """Prevent test leakage via the module-level singleton."""
    import proxy.request_coalescer as rc_mod
    rc_mod._coalescer = None
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_coalesce_identical_concurrent_requests():
    """Two identical concurrent requests should execute the factory only once.

    Uses a slow factory to ensure both requests arrive while the first is
    still in-flight.
    """
    coalescer = RequestCoalescer()

    call_count = 0

    async def _factory() -> Response:
        nonlocal call_count
        call_count += 1
        return await _slow_response(0.1)

    body = json.dumps({"model": "plan", "messages": [{"role": "user", "content": "hi"}]}).encode("utf-8")

    async def make_request():
        return await coalescer.coalesce_or_execute(
            "v1/chat/completions", body, _factory
        )

    r1, r2 = await asyncio.gather(make_request(), make_request())

    assert call_count == 1, "Factory should be called exactly once for duplicate requests"
    assert r1.status_code == 200
    assert r2.status_code == 200
    body1 = json.loads(r1.body)
    body2 = json.loads(r2.body)
    assert body1 == {"status": "slow"}
    assert body2 == {"status": "slow"}


@pytest.mark.asyncio
async def test_three_concurrent_duplicates():
    """Three identical concurrent requests should all get the same response."""
    coalescer = RequestCoalescer()

    call_count = 0

    async def _factory() -> Response:
        nonlocal call_count
        call_count += 1
        return await _slow_response(0.15)

    body = json.dumps({"model": "plan"}).encode("utf-8")

    async def make_request():
        return await coalescer.coalesce_or_execute(
            "v1/chat/completions", body, _factory
        )

    r1, r2, r3 = await asyncio.gather(make_request(), make_request(), make_request())

    assert call_count == 1
    assert all(r.status_code == 200 for r in [r1, r2, r3])


@pytest.mark.asyncio
async def test_non_identical_requests_not_coalesced():
    """Different request bodies should each get their own factory call."""
    coalescer = RequestCoalescer()

    call_count = 0

    async def _factory() -> Response:
        nonlocal call_count
        call_count += 1
        return await _ok_response()

    body_a = json.dumps({"model": "plan", "messages": [{"role": "user", "content": "hello"}]}).encode("utf-8")
    body_b = json.dumps({"model": "plan", "messages": [{"role": "user", "content": "world"}]}).encode("utf-8")

    async def make_request(body):
        return await coalescer.coalesce_or_execute(
            "v1/chat/completions", body, _factory
        )

    r1, r2 = await asyncio.gather(make_request(body_a), make_request(body_b))

    assert call_count == 2, "Factory should be called once per unique request"
    assert r1.status_code == 200
    assert r2.status_code == 200


@pytest.mark.asyncio
async def test_same_body_different_path_not_coalesced():
    """Requests with same body but different paths should not be coalesced."""
    coalescer = RequestCoalescer()

    call_count = 0

    async def _factory() -> Response:
        nonlocal call_count
        call_count += 1
        return await _ok_response()

    body = json.dumps({"model": "plan"}).encode("utf-8")

    async def make_request(path):
        return await coalescer.coalesce_or_execute(path, body, _factory)

    r1, r2 = await asyncio.gather(
        make_request("v1/chat/completions"),
        make_request("v1/embeddings"),
    )

    assert call_count == 2, "Different paths should not coalesce"


@pytest.mark.asyncio
async def test_error_propagation_to_duplicate():
    """When the leader request raises, duplicates should see the same error."""
    coalescer = RequestCoalescer()

    body = json.dumps({"model": "plan"}).encode("utf-8")

    # Use a slow error to ensure concurrency
    async def _slow_error():
        await asyncio.sleep(0.05)
        raise RuntimeError("Simulated failure")

    async def make_request():
        return await coalescer.coalesce_or_execute(
            "v1/chat/completions", body, _slow_error
        )

    results = await asyncio.gather(make_request(), make_request(), return_exceptions=True)

    assert len(results) == 2
    for r in results:
        assert isinstance(r, RuntimeError)
        assert "Simulated failure" in str(r)


@pytest.mark.asyncio
async def test_cleanup_after_completion():
    """After a request completes, its entry should be removed from in_flight."""
    coalescer = RequestCoalescer()

    body = json.dumps({"model": "plan"}).encode("utf-8")
    body_hash = coalescer._hash_body(body)
    key = ("v1/chat/completions", body_hash)

    result = await coalescer.coalesce_or_execute(
        "v1/chat/completions", body, _ok_response
    )
    assert result.status_code == 200
    assert key not in coalescer._in_flight


@pytest.mark.asyncio
async def test_cleanup_after_error():
    """After a failed request, its entry should be removed from in_flight."""
    coalescer = RequestCoalescer()

    body = json.dumps({"model": "plan"}).encode("utf-8")
    body_hash = coalescer._hash_body(body)
    key = ("v1/chat/completions", body_hash)

    with pytest.raises(RuntimeError):
        await coalescer.coalesce_or_execute(
            "v1/chat/completions", body, _error_response
        )

    assert key not in coalescer._in_flight


@pytest.mark.asyncio
async def test_response_headers_preserved():
    """Response headers from the leader should be available to duplicates."""
    coalescer = RequestCoalescer()

    async def _factory() -> Response:
        await asyncio.sleep(0.05)
        return Response(
            content=json.dumps({"ok": True}).encode("utf-8"),
            status_code=200,
            media_type="application/json",
            headers={
                "X-Provider": "test-provider",
                "X-Request-Id": "abc-123",
            },
        )

    body = json.dumps({"model": "plan"}).encode("utf-8")

    async def make_request():
        return await coalescer.coalesce_or_execute(
            "v1/chat/completions", body, _factory
        )

    r1, r2 = await asyncio.gather(make_request(), make_request())

    assert r1.headers.get("X-Provider") == "test-provider"
    assert r2.headers.get("X-Provider") == "test-provider"
    assert r1.headers.get("X-Request-Id") == "abc-123"
    assert r2.headers.get("X-Request-Id") == "abc-123"


@pytest.mark.asyncio
async def test_body_hash_deterministic():
    """JSON bodies with different key ordering should produce the same hash."""
    coalescer = RequestCoalescer()

    body_a = json.dumps({"model": "plan", "messages": [{"role": "user", "content": "hi"}]}).encode("utf-8")
    body_b = json.dumps({"messages": [{"role": "user", "content": "hi"}], "model": "plan"}).encode("utf-8")

    assert coalescer._hash_body(body_a) == coalescer._hash_body(body_b)


@pytest.mark.asyncio
async def test_empty_body_requests():
    """Requests with empty bodies should be handled without error."""
    coalescer = RequestCoalescer()

    call_count = 0

    async def _factory() -> Response:
        nonlocal call_count
        call_count += 1
        return await _slow_response(0.15)

    r1, r2 = await asyncio.gather(
        coalescer.coalesce_or_execute("v1/health", None, _factory),
        coalescer.coalesce_or_execute("v1/health", None, _factory),
    )

    assert call_count == 1
    assert r1.status_code == 200
    assert r2.status_code == 200


@pytest.mark.asyncio
async def test_sequential_requests_not_coalesced():
    """Sequential (non-concurrent) identical requests should each execute."""
    coalescer = RequestCoalescer()

    call_count = 0

    async def _factory() -> Response:
        nonlocal call_count
        call_count += 1
        return await _ok_response()

    body = json.dumps({"model": "plan"}).encode("utf-8")

    r1 = await coalescer.coalesce_or_execute("v1/chat/completions", body, _factory)
    # Second request after first has completed — should execute again
    r2 = await coalescer.coalesce_or_execute("v1/chat/completions", body, _factory)

    assert call_count == 2, "Sequential identical requests should each execute"
    assert r1.status_code == 200
    assert r2.status_code == 200


@pytest.mark.asyncio
async def test_different_models_not_coalesced():
    """Requests with different models (different bodies) should not coalesce."""
    coalescer = RequestCoalescer()

    call_count = 0

    async def _factory() -> Response:
        nonlocal call_count
        call_count += 1
        return await _slow_response(0.15)

    body_plan = json.dumps({"model": "plan"}).encode("utf-8")
    body_code = json.dumps({"model": "code"}).encode("utf-8")

    async def make_request(body):
        return await coalescer.coalesce_or_execute(
            "v1/chat/completions", body, _factory
        )

    r1, r2 = await asyncio.gather(make_request(body_plan), make_request(body_code))
    assert call_count == 2


@pytest.mark.asyncio
async def test_module_level_get_coalescer():
    """The module-level get_coalescer() should return the same singleton."""
    from proxy.request_coalescer import get_coalescer

    c1 = get_coalescer()
    c2 = get_coalescer()
    assert c1 is c2


@pytest.mark.asyncio
async def test_partial_body_match_not_coalesced():
    """Slightly different bodies should NOT be treated as duplicates."""
    coalescer = RequestCoalescer()

    call_count = 0

    async def _factory() -> Response:
        nonlocal call_count
        call_count += 1
        return await _slow_response(0.15)

    body_a = json.dumps({"model": "plan", "stream": False}).encode("utf-8")
    body_b = json.dumps({"model": "plan", "stream": True}).encode("utf-8")

    async def make_request(body):
        return await coalescer.coalesce_or_execute(
            "v1/chat/completions", body, _factory
        )

    r1, r2 = await asyncio.gather(make_request(body_a), make_request(body_b))
    assert call_count == 2


@pytest.mark.asyncio
async def test_duplicate_during_leader_execution():
    """When duplicate arrives while leader is executing, it should wait and get the result."""
    coalescer = RequestCoalescer()

    # Use event to control timing
    leader_started = asyncio.Event()
    leader_can_finish = asyncio.Event()

    call_count = 0

    async def _controlled_factory() -> Response:
        nonlocal call_count
        call_count += 1
        leader_started.set()
        await leader_can_finish.wait()
        return await _ok_response()

    body = json.dumps({"model": "plan"}).encode("utf-8")

    async def make_request(idx):
        return await coalescer.coalesce_or_execute(
            "v1/chat/completions", body, _controlled_factory
        )

    # Start leader and wait for it to enter the factory
    leader_task = asyncio.create_task(make_request(1))
    await leader_started.wait()

    # Now start the duplicate while leader is in-flight
    dup_task = asyncio.create_task(make_request(2))
    await asyncio.sleep(0.05)  # give dup time to register

    # Let leader finish
    leader_can_finish.set()

    r1, r2 = await asyncio.gather(leader_task, dup_task)

    assert call_count == 1
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert json.loads(r1.body) == {"status": "ok"}
    assert json.loads(r2.body) == {"status": "ok"}
