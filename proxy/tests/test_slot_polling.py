import asyncio
import time
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
