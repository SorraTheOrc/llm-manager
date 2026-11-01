from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_ensure_model_loaded_emits_switching_then_ready(monkeypatch):
    from proxy import server

    events = []

    async def fake_broadcast(event_type, data):
        events.append((event_type, data))

    class DummyLock:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    # monkeypatch restores module globals after the test. ensure_model_loaded
    # assigns server.llama_process = <mocked start_llama_server return value>
    # and server.current_model; leaking them breaks later tests that call
    # srv.llama_process.poll() (test_version_footer LP-0MS9MBVJK003D3RR).
    monkeypatch.setattr(server, "llama_process", None)

    with patch.object(server, "config", {"server": {"llama_router_mode": False}}):
        with patch.object(server, "model_switch_lock", DummyLock()):
            with patch.object(server, "broadcast_status", side_effect=fake_broadcast):
                with patch.object(server, "stop_llama_server") as mock_stop:
                    with patch.object(server, "start_llama_server", return_value=object()):
                        with patch.object(server, "wait_for_llama_server", new=AsyncMock(return_value=True)):
                            with patch.object(server, "get_local_model_name", return_value="llama-7b"):
                                server.current_model = "old-model"
                                ok = await server.ensure_model_loaded("new-model")

    assert ok is True
    mock_stop.assert_called_once()
    assert events[0][0] == "switching"
    assert events[0][1]["target_model"] == "llama-7b"
    assert events[0][1]["previous_model"] == "old-model"
    assert events[-1][0] == "ready"
    assert events[-1][1]["current_model"] == "llama-7b"
    assert server.current_model == "llama-7b"


@pytest.mark.asyncio
async def test_ensure_model_loaded_does_not_update_current_model_on_failure(monkeypatch):
    from proxy import server

    events = []

    async def fake_broadcast(event_type, data):
        events.append((event_type, data))

    class DummyLock:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(server, "llama_process", None)

    with patch.object(server, "config", {"server": {"llama_router_mode": False}}):
        with patch.object(server, "model_switch_lock", DummyLock()):
            with patch.object(server, "broadcast_status", side_effect=fake_broadcast):
                with patch.object(server, "stop_llama_server") as mock_stop:
                    with patch.object(server, "start_llama_server", return_value=object()):
                        with patch.object(server, "wait_for_llama_server", new=AsyncMock(return_value=False)):
                            with patch.object(server, "get_local_model_name", return_value="llama-7b"):
                                server.current_model = "old-model"
                                ok = await server.ensure_model_loaded("new-model")

    assert ok is False
    mock_stop.assert_called()
    assert events[0][0] == "switching"
    assert events[-1][0] == "error"
    assert server.current_model == "old-model"


# ---------------------------------------------------------------------------
# LP-0MVT2QFAP0082BKR — a model switch invalidates the captured measurements
#
# A ``prompt_progress`` measurement (``proxy.measured_prefill``) describes the
# prefix held by one slot of one model. A switch tears that slot pool down, so
# the measurement must not survive into the next dispatch on the same model
# name (A -> B -> A would otherwise read a prefix size that no longer exists as
# a numeric reuse ratio). The store's ``(model, session)`` key is the primary
# guarantee; these pins cover the explicit clear where the switch is
# observable, and the full reset when a dead llama-server is restarted.
# ---------------------------------------------------------------------------


def _measured(total: int):
    from proxy.measured_prefill import MeasuredPrefill

    return MeasuredPrefill(total=total, cache=0, processed=0, time_ms=1.0)


class _DummyLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DeadProcess:
    """A llama-server handle whose process has exited."""

    def poll(self):
        return 1


async def _run_switch(
    monkeypatch,
    *,
    requested: str,
    llama_name: str,
    current_model,
    llama_process=None,
):
    from proxy import server

    monkeypatch.setattr(server, "llama_process", llama_process)
    monkeypatch.setattr(server, "current_model", current_model)

    async def fake_broadcast(event_type, data):
        return None

    with patch.object(server, "config", {"server": {"llama_router_mode": False}}):
        with patch.object(server, "model_switch_lock", _DummyLock()):
            with patch.object(server, "broadcast_status", side_effect=fake_broadcast):
                with patch.object(server, "stop_llama_server"):
                    with patch.object(
                        server, "start_llama_server", return_value=object()
                    ):
                        with patch.object(
                            server,
                            "wait_for_llama_server",
                            new=AsyncMock(return_value=True),
                        ):
                            with patch.object(
                                server,
                                "get_local_model_name",
                                side_effect=lambda name: llama_name,
                            ):
                                return await server.ensure_model_loaded(requested)


@pytest.mark.asyncio
async def test_model_switch_forgets_the_previous_models_measurements(monkeypatch):
    """A switch keeps only the target model's names and forgets the rest.

    ``code`` is the client-facing name the previous model (``Qwen3``) was
    dispatched under in single-model mode, so both names belong to the
    departing model and both must go. The target model's own entry is kept —
    its slots are the ones being loaded.
    """
    import proxy.measured_prefill as measured_prefill

    measured_prefill.reset_measured_prefill()
    try:
        measured_prefill.set_measured_prefill("code", "sess-a", _measured(40_000))
        measured_prefill.set_measured_prefill("Qwen3", "sess-a", _measured(40_000))
        measured_prefill.set_measured_prefill(
            "Qwen3-Coder-Next", "sess-b", _measured(30_000)
        )

        ok = await _run_switch(
            monkeypatch,
            requested="coder",
            llama_name="Qwen3-Coder-Next",
            current_model="Qwen3",
        )

        assert ok is True
        assert measured_prefill.get_measured_prefill("code", "sess-a") is None
        assert measured_prefill.get_measured_prefill("Qwen3", "sess-a") is None
        assert (
            measured_prefill.get_measured_prefill("Qwen3-Coder-Next", "sess-b")
            is not None
        )
    finally:
        measured_prefill.reset_measured_prefill()


@pytest.mark.asyncio
async def test_dead_backend_restart_forgets_every_measurement(monkeypatch):
    """A crashed llama-server invalidates *every* slot, target model included.

    The restart reuses the same model name, so a name-scoped clear would keep
    the entry and the next admission would report a prefix the restarted pool
    does not hold.
    """
    import proxy.measured_prefill as measured_prefill

    measured_prefill.reset_measured_prefill()
    try:
        measured_prefill.set_measured_prefill("code", "sess-a", _measured(40_000))
        measured_prefill.set_measured_prefill("Qwen3", "sess-a", _measured(40_000))

        ok = await _run_switch(
            monkeypatch,
            requested="code",
            llama_name="Qwen3",
            current_model="Qwen3",
            llama_process=_DeadProcess(),
        )

        assert ok is True
        assert measured_prefill.get_measured_prefill("code", "sess-a") is None
        assert measured_prefill.get_measured_prefill("Qwen3", "sess-a") is None
    finally:
        measured_prefill.reset_measured_prefill()
