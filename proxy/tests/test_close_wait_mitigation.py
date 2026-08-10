"""Tests for CLOSE-WAIT mitigation (LP-0MSNM9UCC002CHYU).

Verifies:
1. uvicorn.run() is configured with timeout_keep_alive and
   timeout_graceful_shutdown from server config (so abandoned keep-alive
   sockets are closed promptly instead of piling up in CLOSE-WAIT).
2. Defaults are applied when config keys are absent.
3. The disconnect-reaping middleware cancels abandoned in-flight requests.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ===================================================================
# uvicorn timeout configuration
# ===================================================================


class TestUvicornTimeoutConfig:
    """uvicorn.run must receive keep-alive/graceful-shutdown timeouts."""

    def test_timeouts_passed_to_uvicorn_run(self, tmp_path, monkeypatch):
        """server.timeout_keep_alive / timeout_graceful_shutdown flow to uvicorn."""
        import proxy.server as server_module

        cfg = {
            "server": {
                "host": "127.0.0.1",
                "port": 0,
                "timeout_keep_alive": 7,
                "timeout_graceful_shutdown": 45,
            }
        }
        monkeypatch.setattr(server_module, "load_config", lambda: cfg)

        captured = {}
        def fake_run(app, **kwargs):
            captured.update(kwargs)

        with patch("uvicorn.run", side_effect=fake_run):
            with patch.object(__import__("argparse"), "ArgumentParser") as mock_parser:
                mock_parser.return_value.parse_known_args.return_value = (
                    MagicMock(verbose=False), []
                )
                server_module.main()

        assert captured.get("timeout_keep_alive") == 7
        assert captured.get("timeout_graceful_shutdown") == 45

    def test_timeout_defaults_when_config_missing(self, tmp_path, monkeypatch):
        """Defaults are used when config lacks the timeout keys."""
        import proxy.server as server_module

        cfg = {"server": {"host": "127.0.0.1", "port": 0}}
        monkeypatch.setattr(server_module, "load_config", lambda: cfg)

        captured = {}
        def fake_run(app, **kwargs):
            captured.update(kwargs)

        with patch("uvicorn.run", side_effect=fake_run):
            with patch.object(__import__("argparse"), "ArgumentParser") as mock_parser:
                mock_parser.return_value.parse_known_args.return_value = (
                    MagicMock(verbose=False), []
                )
                server_module.main()

        assert captured.get("timeout_keep_alive") == 5
        assert captured.get("timeout_graceful_shutdown") == 30


# ===================================================================
# Disconnect-reaping middleware
# ===================================================================


class TestDisconnectReaperMiddleware:
    """The reaper middleware tracks in-flight requests without body reads."""

    @pytest.mark.asyncio
    async def test_dispatches_and_unregisters(self):
        """Middleware dispatches and removes the task from the registry."""
        from proxy.disconnect_reaper import (
            DisconnectReaperMiddleware,
            reaper_registry,
        )

        reaper_registry.clear()
        mock_request = MagicMock()
        mock_response = MagicMock(status_code=200)
        mock_call_next = AsyncMock(return_value=mock_response)

        mw = DisconnectReaperMiddleware(app=None)
        response = await mw.dispatch(mock_request, mock_call_next)

        assert response is mock_response
        mock_call_next.assert_awaited_once()
        # Task unregistered after completion.
        assert asyncio.current_task() not in reaper_registry
        reaper_registry.clear()

    @pytest.mark.asyncio
    async def test_registers_task_while_running(self):
        """The in-flight task is registered while the handler runs."""
        from proxy.disconnect_reaper import (
            DisconnectReaperMiddleware,
            reaper_registry,
        )

        reaper_registry.clear()
        mock_request = MagicMock()
        mock_response = MagicMock(status_code=200)
        seen = {}

        async def call_next(req):
            seen["registered"] = asyncio.current_task() in reaper_registry
            return mock_response

        mw = DisconnectReaperMiddleware(app=None)
        await mw.dispatch(mock_request, call_next)

        assert seen["registered"] is True, "Task must be registered mid-request"
        assert asyncio.current_task() not in reaper_registry
        reaper_registry.clear()

    @pytest.mark.asyncio
    async def test_unregisters_on_exception(self):
        """The task is removed from the registry even when the handler raises."""
        from proxy.disconnect_reaper import (
            DisconnectReaperMiddleware,
            reaper_registry,
        )

        reaper_registry.clear()
        mock_request = MagicMock()

        async def call_next(req):
            raise RuntimeError("boom")

        mw = DisconnectReaperMiddleware(app=None)
        with pytest.raises(RuntimeError):
            await mw.dispatch(mock_request, call_next)

        assert asyncio.current_task() not in reaper_registry
        reaper_registry.clear()

    @pytest.mark.asyncio
    async def test_reaper_registry_reaps_stale_requests(self):
        """The reaper cancels tasks whose client disconnected while running."""
        from proxy.disconnect_reaper import (
            DisconnectReaper,
            reaper_registry,
        )

        reaper_registry.clear()

        # Task A: client stays connected → survives.
        req_a = MagicMock()
        req_a.is_disconnected = AsyncMock(return_value=False)
        # Task B: client disconnects mid-flight → should be cancelled.
        req_b = MagicMock()
        req_b.is_disconnected = AsyncMock(return_value=True)

        async def long_running():
            await asyncio.sleep(60)

        task_a = asyncio.create_task(long_running())
        task_b = asyncio.create_task(long_running())
        reaper_registry[task_a] = req_a
        reaper_registry[task_b] = req_b

        try:
            reaper = DisconnectReaper()
            await reaper.reap_once()

            # reap_once cancels task_b (disconnected). task_a stays alive.
            with pytest.raises(asyncio.CancelledError):
                await task_b

            assert not task_a.cancelled(), "Connected client task must survive"
            assert task_b.cancelled(), "Disconnected client task must be cancelled"
        finally:
            for t in (task_a, task_b):
                if not t.done():
                    t.cancel()
            reaper_registry.clear()

    @pytest.mark.asyncio
    async def test_reaper_skips_completed_tasks(self):
        """Completed tasks are dropped from the registry without cancellation."""
        from proxy.disconnect_reaper import (
            DisconnectReaper,
            reaper_registry,
        )

        reaper_registry.clear()

        mock_request = MagicMock()
        mock_request.is_disconnected = AsyncMock(return_value=True)

        async def quick():
            return 42

        task = asyncio.create_task(quick())
        await task  # completes immediately
        reaper_registry[task] = mock_request

        try:
            reaper = DisconnectReaper()
            await reaper.reap_once()
            # Task already done → left alone, and removed from registry.
            assert not task.cancelled()
            assert task not in reaper_registry
        finally:
            reaper_registry.clear()
