import asyncio
import time
import threading
import pytest

from proxy import server


class MockResp:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data

    def json(self):
        return self._json


class MockClient:
    def __init__(self, responses):
        # responses: list of dicts with keys 'status' and 'json'
        self._responses = list(responses)
        self._i = 0

    async def get(self, url, timeout=None):
        # Simulate network call latency minimal
        await asyncio.sleep(0)
        if self._i >= len(self._responses):
            # default to 500
            return MockResp(500, None)
        r = self._responses[self._i]
        self._i += 1
        # If r is an exception instruct to raise
        if isinstance(r, Exception):
            raise r
        return MockResp(r.get("status", 200), r.get("json"))


@pytest.mark.asyncio
async def test_poll_slots_list_responses():
    # Sequence: processing 10, processing 20, finished
    responses = [
        {"status": 200, "json": [{"is_processing": True, "next_token": {"n_decoded": 10}}]},
        {"status": 200, "json": [{"is_processing": True, "next_token": {"n_decoded": 20}}]},
        {"status": 200, "json": [{"is_processing": False}]},
    ]
    server._http_client = MockClient(responses)
    # Run poller for exactly 3 polls
    await server.poll_slots_for_model("qwen3", llama_port=1234, interval=0.01, max_polls=3)
    assert "qwen3" in server.slot_polling_state
    state = server.slot_polling_state["qwen3"]
    assert state["is_processing"] is False
    assert state["n_decoded"] is None


@pytest.mark.asyncio
async def test_poll_slots_dict_response():
    responses = [
        {"status": 200, "json": {"is_processing": True, "next_token": {"n_decoded": 100}}},
        {"status": 200, "json": {"is_processing": False, "n_decoded": 100}},
    ]
    server._http_client = MockClient(responses)
    await server.poll_slots_for_model("mymodel", llama_port=4321, interval=0.01, max_polls=2)
    assert "mymodel" in server.slot_polling_state
    state = server.slot_polling_state["mymodel"]
    # After second poll (is_processing False, n_decoded present at top-level) we expect n_decoded==100
    assert state["is_processing"] is False
    assert state["n_decoded"] == 100


@pytest.mark.asyncio
async def test_poll_slots_error_and_500():
    # Simulate exception on first call, then 500 response
    class DummyError(Exception):
        pass

    responses = [DummyError("connect failed"), {"status": 500, "json": None}]
    server._http_client = MockClient(responses)
    await server.poll_slots_for_model("errmodel", llama_port=9, interval=0.01, max_polls=2)
    assert "errmodel" in server.slot_polling_state
    state = server.slot_polling_state["errmodel"]
    assert state["is_processing"] is False
    assert state["n_decoded"] is None


@pytest.mark.asyncio
async def test_polling_interval_respected():
    # Use a tiny interval but measure elapsed time to ensure sleeps occur
    responses = [
        {"status": 200, "json": [{"is_processing": True, "next_token": {"n_decoded": 1}}]},
        {"status": 200, "json": [{"is_processing": True, "next_token": {"n_decoded": 2}}]},
        {"status": 200, "json": [{"is_processing": False}]},
    ]
    server._http_client = MockClient(responses)
    start = time.time()
    await server.poll_slots_for_model("timer", llama_port=1, interval=0.02, max_polls=3)
    elapsed = time.time() - start
    # Expect at least two intervals worth of sleeping (rough heuristic)
    assert elapsed >= 0.02 * 2 - 0.01


# ---------------------------------------------------------------------------
# poll_slots_for_model – edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_slots_empty_model_returns_none():
    """Passing an empty/None model should return None immediately."""
    result = await server.poll_slots_for_model("", llama_port=1234, interval=0.01, max_polls=1)
    assert result is None
    result = await server.poll_slots_for_model(None, llama_port=1234, interval=0.01, max_polls=1)
    assert result is None


@pytest.mark.asyncio
async def test_poll_slots_empty_list_response():
    """Empty list response should set is_processing=False, n_decoded=None."""
    responses = [
        {"status": 200, "json": []},
    ]
    server._http_client = MockClient(responses)
    await server.poll_slots_for_model("empty-list", llama_port=1234, interval=0.01, max_polls=1)
    assert "empty-list" in server.slot_polling_state
    state = server.slot_polling_state["empty-list"]
    assert state["is_processing"] is False
    assert state["n_decoded"] is None


@pytest.mark.asyncio
async def test_poll_slots_next_token_not_dict_falls_back_to_top_level():
    """When next_token is present but not a dict, fall back to top-level n_decoded."""
    responses = [
        {"status": 200, "json": [{"is_processing": True, "next_token": "not-a-dict", "n_decoded": 42}]},
    ]
    server._http_client = MockClient(responses)
    await server.poll_slots_for_model("fallback-token", llama_port=1234, interval=0.01, max_polls=1)
    state = server.slot_polling_state["fallback-token"]
    assert state["is_processing"] is True
    assert state["n_decoded"] == 42


@pytest.mark.asyncio
async def test_poll_slots_dict_without_next_token():
    """Dict response without next_token uses top-level n_decoded directly."""
    responses = [
        {"status": 200, "json": {"is_processing": True, "n_decoded": 77}},
    ]
    server._http_client = MockClient(responses)
    await server.poll_slots_for_model("no-next-token", llama_port=1234, interval=0.01, max_polls=1)
    state = server.slot_polling_state["no-next-token"]
    assert state["is_processing"] is True
    assert state["n_decoded"] == 77


@pytest.mark.asyncio
async def test_poll_slots_multi_slot_list():
    """Multi-slot list response should aggregate across all slots.

    When data[0] has is_processing=False but data[1] has is_processing=True,
    the aggregated is_processing should be True (any slot processing).
    n_decoded should reflect the max across all slots.
    """
    responses = [
        {
            "status": 200,
            "json": [
                {"is_processing": False, "next_token": {"n_decoded": 5}},
                {"is_processing": True, "next_token": {"n_decoded": 10}},
            ],
        },
    ]
    server._http_client = MockClient(responses)
    await server.poll_slots_for_model("multi-slot", llama_port=1234, interval=0.01, max_polls=1)
    assert "multi-slot" in server.slot_polling_state
    state = server.slot_polling_state["multi-slot"]
    # Any slot processing => is_processing True
    assert state["is_processing"] is True, (
        f"Expected is_processing=True (slot 1 is processing), got {state['is_processing']}"
    )
    # n_decoded should be max across slots (10, not 5)
    assert state["n_decoded"] == 10, (
        f"Expected n_decoded=10 (max across slots), got {state['n_decoded']}"
    )


@pytest.mark.asyncio
async def test_poll_slots_single_slot_still_works():
    """Single-slot list response should still work as before (no regression)."""
    responses = [
        {"status": 200, "json": [{"is_processing": True, "next_token": {"n_decoded": 42}}]},
    ]
    server._http_client = MockClient(responses)
    await server.poll_slots_for_model("single-slot", llama_port=1234, interval=0.01, max_polls=1)
    assert "single-slot" in server.slot_polling_state
    state = server.slot_polling_state["single-slot"]
    assert state["is_processing"] is True
    assert state["n_decoded"] == 42


@pytest.mark.asyncio
async def test_poll_slots_non_json_response():
    """Non-JSON response body should be gracefully handled (no crash)."""
    class NonJsonResp:
        status_code = 200
        def json(self):
            raise ValueError("not json")

    class NonJsonClient:
        async def get(self, url, timeout=None):
            await asyncio.sleep(0)
            return NonJsonResp()

    server._http_client = NonJsonClient()
    await server.poll_slots_for_model("non-json", llama_port=1234, interval=0.01, max_polls=1)
    state = server.slot_polling_state["non-json"]
    assert state["is_processing"] is False
    assert state["n_decoded"] is None


# ---------------------------------------------------------------------------
# start_slot_polling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_slot_polling_empty_model():
    """Empty model should return None without creating a task."""
    result = server.start_slot_polling("", llama_port=1234)
    assert result is None
    result = server.start_slot_polling(None, llama_port=1234)
    assert result is None


@pytest.mark.asyncio
async def test_start_slot_polling_creates_task():
    """Calling start_slot_polling should create an asyncio task in _slot_polling_tasks."""
    # Clear any leftover state
    if "test-task-model" in server._slot_polling_tasks:
        task = server._slot_polling_tasks["test-task-model"]
        if not task.done():
            task.cancel()
        del server._slot_polling_tasks["test-task-model"]

    server.start_slot_polling("test-task-model", llama_port=1, interval=0.01)
    assert "test-task-model" in server._slot_polling_tasks
    task = server._slot_polling_tasks["test-task-model"]
    assert task.done() is False  # task is still running

    # Cleanup: cancel the task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        if "test-task-model" in server._slot_polling_tasks:
            del server._slot_polling_tasks["test-task-model"]
        if "test-task-model" in server.slot_polling_state:
            del server.slot_polling_state["test-task-model"]


@pytest.mark.asyncio
async def test_start_slot_polling_prevents_duplicate():
    """Calling start_slot_polling twice for the same model should not duplicate."""
    if "dup-model" in server._slot_polling_tasks:
        task = server._slot_polling_tasks["dup-model"]
        if not task.done():
            task.cancel()
        del server._slot_polling_tasks["dup-model"]

    server.start_slot_polling("dup-model", llama_port=1, interval=0.01)
    task1 = server._slot_polling_tasks["dup-model"]

    # Call again – should not replace the task
    server.start_slot_polling("dup-model", llama_port=2, interval=0.02)
    task2 = server._slot_polling_tasks["dup-model"]
    assert task2 is task1  # same task object

    # Cleanup
    task1.cancel()
    try:
        await task1
    except asyncio.CancelledError:
        pass
    finally:
        if "dup-model" in server._slot_polling_tasks:
            del server._slot_polling_tasks["dup-model"]
        if "dup-model" in server.slot_polling_state:
            del server.slot_polling_state["dup-model"]


@pytest.mark.asyncio
async def test_start_slot_polling_replaces_completed_task():
    """If a previous polling task is already done, a new one should be created."""
    if "replaced-model" in server._slot_polling_tasks:
        task = server._slot_polling_tasks["replaced-model"]
        if not task.done():
            task.cancel()
        del server._slot_polling_tasks["replaced-model"]

    # Run a short-lived poller that completes quickly
    server._http_client = MockClient([
        {"status": 200, "json": [{"is_processing": False}]},
    ])
    server.start_slot_polling("replaced-model", llama_port=1, interval=0.01)
    first_task = server._slot_polling_tasks["replaced-model"]
    await asyncio.sleep(0.05)
    # First task should be done after max_polls=None + one poll... but the task runs
    # indefinitely so we can't wait for it to complete. Let's cancel it first to
    # simulate a completed task scenario.
    first_task.cancel()
    try:
        await first_task
    except asyncio.CancelledError:
        pass

    # Now start a new one – it should accept
    server.start_slot_polling("replaced-model", llama_port=2, interval=0.01)
    second_task = server._slot_polling_tasks.get("replaced-model")
    assert second_task is not None
    assert second_task is not first_task  # new task created

    # Cleanup
    if second_task and not second_task.done():
        second_task.cancel()
        try:
            await second_task
        except asyncio.CancelledError:
            pass
    if "replaced-model" in server._slot_polling_tasks:
        del server._slot_polling_tasks["replaced-model"]
    if "replaced-model" in server.slot_polling_state:
        del server.slot_polling_state["replaced-model"]
