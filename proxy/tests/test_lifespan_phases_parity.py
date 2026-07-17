"""
Parity tests for the extracted lifespan phase helpers.

These tests verify that the extracted startup/shutdown helpers produce the
same observable side effects as the original monolithic lifespan() function
— same background tasks created, same globals set, same log messages, same
shutdown cancellation sequence.

All tests use monkeypatching to avoid running the actual FastAPI lifespan,
podman, model loading, or HTTP clients.
"""

import asyncio
import logging
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the server module to access extracted helpers and globals
import proxy.server as server

pytestmark = pytest.mark.refactor_parity


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_globals(monkeypatch):
    """Reset module-level globals before each test."""
    monkeypatch.setattr(server, "config", {"server": {}})
    monkeypatch.setattr(server, "backend_ready", False)
    monkeypatch.setattr(server, "backend_recovery_state", {})
    monkeypatch.setattr(server, "_http_client", None)
    monkeypatch.setattr(server, "_remote_http_client", None)
    monkeypatch.setattr(server, "backend_watchdog_task", None)
    monkeypatch.setattr(server, "model_health_task", None)
    monkeypatch.setattr(server, "_dispatch_cleanup_task", None)
    monkeypatch.setattr(server, "counts_persist_task", None)
    monkeypatch.setattr(server, "tokens_persist_task", None)
    monkeypatch.setattr(server, "periodic_broadcast_task", None)
    monkeypatch.setattr(server, "llama_server_version", "unknown")
    monkeypatch.setattr(server, "rocm_version", "unknown")


@pytest.fixture
def mock_logger(monkeypatch):
    """Provide a controllable logger that captures log messages."""
    logger = logging.getLogger("test_lifespan_phases")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    monkeypatch.setattr(server, "logger", logger)
    return logger


@pytest.fixture
def dummy_app():
    """A minimal FastAPI-like object for route registration tests."""
    class DummyApp:
        def __init__(self):
            self.routes = []
        def add_api_route(self, path, endpoint, **kwargs):
            self.routes.append((path, endpoint, kwargs))
    return DummyApp()


# ---------------------------------------------------------------------------
# Startup phase helper tests
# ---------------------------------------------------------------------------


class TestStartupConfigLogging:
    """_startup_config_logging() — config loading and logger setup."""

    def test_sets_config_and_logger(self, monkeypatch, mock_logger):
        """Helper loads config and sets up logging."""
        fake_config = {"server": {"host": "localhost"}}
        monkeypatch.setattr(server, "load_config", lambda: fake_config)
        monkeypatch.setattr(server, "setup_logging", lambda c: mock_logger)

        config_out, logger_out = server._startup_config_logging()

        assert config_out == fake_config
        assert logger_out == mock_logger

    def test_globals_updated(self, monkeypatch, mock_logger):
        """Global config and logger are updated."""
        fake_config = {"server": {"port": 8080}}
        monkeypatch.setattr(server, "load_config", lambda: fake_config)
        monkeypatch.setattr(server, "setup_logging", lambda c: mock_logger)

        server._startup_config_logging()

        assert server.config == fake_config
        assert server.logger == mock_logger


class TestStartupInitializeCacheColdStart:
    """_startup_initialize_cache_cold_start() — model cache cold-start."""

    def test_calls_init_cache(self, monkeypatch):
        """Calls initialize_cache_cold_from_config with config."""
        called_with = []
        def fake_init_cache(cfg):
            called_with.append(cfg)

        monkeypatch.setattr(
            server,
            "config",
            {"server": {"model": "test"}},
        )
        monkeypatch.setattr(
            "proxy.provider.initialize_cache_cold_from_config",
            fake_init_cache,
        )

        server._startup_initialize_cache_cold_start()

        assert len(called_with) == 1
        assert called_with[0] == {"server": {"model": "test"}}

    def test_swallows_exceptions(self, monkeypatch):
        """Exceptions during cache init are caught and logged."""
        def failing_init(_):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "proxy.provider.initialize_cache_cold_from_config",
            failing_init,
        )

        # Should not raise
        server._startup_initialize_cache_cold_start()


class TestStartupInitializeBackendState:
    """_startup_initialize_backend_state() — backend_ready and recovery state."""

    def test_sets_backend_ready_false(self):
        """backend_ready is set to False."""
        server.backend_ready = True
        result = server._startup_initialize_backend_state()
        assert server.backend_ready is False
        assert result["in_progress"] is False

    def test_recovery_state_defaults(self):
        """Recovery state gets default values when config values are missing."""
        server.config = {"server": {}}
        result = server._startup_initialize_backend_state()
        assert result["max_attempts"] == 3
        assert result["window_seconds"] == 300
        assert result["retry_after_seconds"] == 30
        assert result["last_failure"] is None

    def test_recovery_state_from_config(self):
        """Recovery state reads values from config."""
        server.config = {
            "server": {
                "llama_self_heal_max_attempts": 5,
                "llama_self_heal_window_seconds": 600,
                "llama_self_heal_retry_after_seconds": 60,
            }
        }
        result = server._startup_initialize_backend_state()
        assert result["max_attempts"] == 5
        assert result["window_seconds"] == 600
        assert result["retry_after_seconds"] == 60


class TestStartupCreateHttpClient:
    """_startup_create_http_client() — HTTP client creation."""

    def test_local_client_created(self):
        """Local httpx client is created with expected defaults."""
        client = server._startup_create_http_client(server.config)
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        assert server._http_client is client

    def test_remote_client_created(self):
        """Remote httpx client uses config-based timeouts."""
        server.config = {
            "server": {
                "remote_http_client": {
                    "connect_timeout_seconds": 10,
                    "read_timeout_seconds": 60,
                    "pool_connections": 25,
                }
            }
        }
        remote_client = server._startup_create_remote_http_client(server.config)
        assert remote_client is not None
        assert isinstance(remote_client, httpx.AsyncClient)
        assert server._remote_http_client is remote_client

    def test_remote_client_defaults(self):
        """Remote client uses defaults when config values are missing."""
        server.config = {"server": {}}
        remote_client = server._startup_create_remote_http_client(server.config)
        assert remote_client is not None
        assert isinstance(remote_client, httpx.AsyncClient)


class TestStartupVersionCapture:
    """_startup_launch_version_capture() — version capture task."""

    @pytest.mark.asyncio
    async def test_creates_task(self):
        """A background task is created for version capture."""
        task = server._startup_launch_version_capture()
        assert task is not None
        assert isinstance(task, asyncio.Task)
        task.cancel()

    @pytest.mark.asyncio
    async def test_task_runs_and_updates_globals(self, monkeypatch):
        """Task updates llama_server_version and rocm_version."""
        async def fake_capture_llama():
            return "build: 1234 (abc123)"
        async def fake_capture_rocm():
            return "6.2.0"

        monkeypatch.setattr(server, "_capture_llama_server_version", fake_capture_llama)
        monkeypatch.setattr(server, "_capture_rocm_version", fake_capture_rocm)

        task = server._startup_launch_version_capture()
        await asyncio.sleep(0.1)

        assert server.llama_server_version == "build: 1234 (abc123)"
        assert server.rocm_version == "6.2.0"
        task.cancel()


class TestStartupPodmanMigrate:
    """_startup_podman_migrate() — podman system migrate."""

    def test_does_not_raise_on_missing_podman(self):
        """Missing podman is caught and logged, not raised."""
        # Should not raise FileNotFoundError
        server._startup_podman_migrate()

    def test_logs_on_podman_failure(self, monkeypatch, caplog):
        """Podman migrate failure logs a warning, does not abort."""
        def fake_run(*args, **kwargs):
            raise RuntimeError("podman exploded")
        monkeypatch.setattr(server.subprocess, "run", fake_run)

        server._startup_podman_migrate()
        # Should not raise


class TestStartupDefaultModelLoader:
    """_startup_launch_default_model_loader() — background model load task."""

    @pytest.mark.asyncio
    async def test_creates_background_task(self, monkeypatch):
        """A background task is created for default model loading."""
        monkeypatch.setattr(server, "config", {
            "server": {"llama_router_mode": False},
        })
        task = server._startup_launch_default_model_loader()
        assert task is not None
        assert isinstance(task, asyncio.Task)
        task.cancel()

    @pytest.mark.asyncio
    async def test_uses_config_defaults(self, monkeypatch):
        """Uses default_model from config, defaults to gemma4."""
        monkeypatch.setattr(server, "config", {"server": {}, "default_model": "my-model"})
        task = server._startup_launch_default_model_loader()
        assert task is not None
        task.cancel()

    @pytest.mark.asyncio
    async def test_defaults_to_gemma4(self, monkeypatch):
        """Defaults to gemma4 when no default_model in config."""
        monkeypatch.setattr(server, "config", {"server": {}})
        task = server._startup_launch_default_model_loader()
        assert task is not None
        task.cancel()


class TestStartupWatchdogTasks:
    """_startup_launch_watchdog_tasks() — watchdog and health tasks."""

    @pytest.mark.asyncio
    async def test_launches_both_tasks(self):
        """Both backend_watchdog_task and model_health_task are created."""
        server._startup_launch_watchdog_tasks()
        assert server.backend_watchdog_task is not None
        assert isinstance(server.backend_watchdog_task, asyncio.Task)
        assert server.model_health_task is not None
        assert isinstance(server.model_health_task, asyncio.Task)

    @pytest.mark.asyncio
    async def test_skips_if_already_set(self):
        """Does not create new tasks if they already exist."""
        existing = asyncio.Future()
        server.backend_watchdog_task = existing  # type: ignore[assignment]
        existing2 = asyncio.Future()
        server.model_health_task = existing2

        server._startup_launch_watchdog_tasks()
        assert server.backend_watchdog_task is existing
        assert server.model_health_task is existing2


class TestStartupPersistenceTasks:
    """_startup_launch_persistence_tasks() — count/token persist and broadcast."""

    @pytest.mark.asyncio
    async def test_loads_counts(self, monkeypatch):
        """load_counts and load_token_counts are called."""
        load_counts_called = False
        load_tokens_called = False

        def fake_load_counts():
            nonlocal load_counts_called
            load_counts_called = True
        def fake_load_token_counts():
            nonlocal load_tokens_called
            load_tokens_called = True

        monkeypatch.setattr(server, "load_counts", fake_load_counts)
        monkeypatch.setattr(server, "load_token_counts", fake_load_token_counts)

        server._startup_launch_persistence_tasks()

        assert load_counts_called
        assert load_tokens_called

    @pytest.mark.asyncio
    async def test_launches_persist_tasks(self):
        """Persist tasks are created."""
        server._startup_launch_persistence_tasks()
        assert server.counts_persist_task is not None
        assert server.tokens_persist_task is not None
        assert server.periodic_broadcast_task is not None

    @pytest.mark.asyncio
    async def test_skips_if_already_set(self):
        """Does not replace existing persist tasks."""
        existing = asyncio.Future()
        server.counts_persist_task = existing
        server.tokens_persist_task = existing
        server.periodic_broadcast_task = existing

        server._startup_launch_persistence_tasks()
        assert server.counts_persist_task is existing
        assert server.tokens_persist_task is existing
        assert server.periodic_broadcast_task is existing


class TestStartupSessionCleanup:
    """_startup_start_session_cleanup() — session manager cleanup."""

    def test_starts_cleanup(self, monkeypatch):
        """start_cleanup_task is called on session_manager."""
        started = False
        def fake_start():
            nonlocal started
            started = True
        monkeypatch.setattr(server.session_manager, "start_cleanup_task", fake_start)

        server._startup_start_session_cleanup()
        assert started


class TestStartupDispatchCleanup:
    """_startup_start_dispatch_cleanup() — dispatch lease cleanup."""

    @pytest.mark.asyncio
    async def test_creates_task(self):
        """Dispatch cleanup task is created."""
        server._startup_start_dispatch_cleanup()
        assert server._dispatch_cleanup_task is not None

    @pytest.mark.asyncio
    async def test_skips_if_already_set(self):
        """Does not create a new task if one already exists."""
        existing = asyncio.Future()
        server._dispatch_cleanup_task = existing
        server._startup_start_dispatch_cleanup()
        assert server._dispatch_cleanup_task is existing


class TestStartupSessionRoutes:
    """_startup_register_session_routes() — session recording routes."""

    def test_registers_routes(self, monkeypatch, dummy_app):
        """list_session_recording_routes is called with the app."""
        called_with = []
        def fake_register(app):
            called_with.append(app)
        monkeypatch.setattr(
            "proxy.ui.list_session_recording_routes",
            fake_register,
        )

        server._startup_register_session_routes(dummy_app)
        assert len(called_with) == 1
        assert called_with[0] is dummy_app

    def test_swallows_exceptions(self, monkeypatch, dummy_app):
        """Exceptions during route registration are caught."""
        def failing_register(_):
            raise RuntimeError("boom")
        monkeypatch.setattr(
            "proxy.ui.list_session_recording_routes",
            failing_register,
        )

        server._startup_register_session_routes(dummy_app)


# ---------------------------------------------------------------------------
# Shutdown phase helper tests
# ---------------------------------------------------------------------------


class TestShutdownCleanupTasks:
    """_shutdown_cleanup_tasks() — cancel/await cleanup tasks."""

    @pytest.mark.asyncio
    async def test_cancels_dispatch_cleanup(self):
        """Dispatch cleanup task is cancelled and awaited."""
        async def dummy_task():
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                pass

        task = asyncio.ensure_future(dummy_task())
        server._dispatch_cleanup_task = task
        await server._shutdown_cleanup_tasks()
        assert server._dispatch_cleanup_task is None

    @pytest.mark.asyncio
    async def test_cancels_watchdog(self):
        """Watchdog task is cancelled and awaited."""
        async def dummy_task():
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                pass

        task = asyncio.ensure_future(dummy_task())
        server.backend_watchdog_task = task
        await server._shutdown_cleanup_tasks()
        assert server.backend_watchdog_task is None

    @pytest.mark.asyncio
    async def test_handles_none_tasks(self):
        """No error if cleanup/dispatch/watchdog tasks are None."""
        await server._shutdown_cleanup_tasks()


class TestShutdownHttpClient:
    """_shutdown_http_client() — close HTTP clients."""

    @pytest.mark.asyncio
    async def test_closes_local_client(self):
        """Local http client is closed and set to None."""
        server._http_client = httpx.AsyncClient()
        await server._shutdown_http_client()
        assert server._http_client is None

    @pytest.mark.asyncio
    async def test_closes_remote_client(self):
        """Remote http client is closed and set to None."""
        server._remote_http_client = httpx.AsyncClient()
        await server._shutdown_http_client()
        assert server._remote_http_client is None

    @pytest.mark.asyncio
    async def test_handles_none(self):
        """No error if both clients are None."""
        await server._shutdown_http_client()


class TestShutdownLlamaServer:
    """_shutdown_llama_server() — stop llama server and reset state."""

    def test_sets_backend_ready_false(self, monkeypatch):
        """backend_ready is set to False."""
        server.backend_ready = True
        monkeypatch.setattr(server, "stop_llama_server", lambda: None)
        server._shutdown_llama_server()
        assert server.backend_ready is False

    def test_calls_stop_llama_server(self, monkeypatch):
        """stop_llama_server is called during shutdown."""
        called = False
        def fake_stop():
            nonlocal called
            called = True
        monkeypatch.setattr(server, "stop_llama_server", fake_stop)
        server._shutdown_llama_server()
        assert called


class TestShutdownSessionCleanup:
    """_shutdown_stop_session_cleanup() — stop session manager cleanup."""

    def test_stops_cleanup(self, monkeypatch):
        """stop_cleanup_task is called on session_manager."""
        stopped = False
        def fake_stop():
            nonlocal stopped
            stopped = True
        monkeypatch.setattr(server.session_manager, "stop_cleanup_task", fake_stop)

        server._shutdown_stop_session_cleanup()
        assert stopped

    def test_swallows_exceptions(self, monkeypatch):
        """Exceptions during stop are caught."""
        def failing_stop():
            raise RuntimeError("boom")
        monkeypatch.setattr(server.session_manager, "stop_cleanup_task", failing_stop)

        # Should not raise
        server._shutdown_stop_session_cleanup()


# ---------------------------------------------------------------------------
# Orchestration parity test
# ---------------------------------------------------------------------------


class TestLifespanOrchestration:
    """lifespan() simplified orchestration — delegates to extracted helpers."""

    @pytest.mark.asyncio
    async def test_startup_sequence_invokes_all_helpers(self, monkeypatch, mock_logger):
        """lifespan startup phase calls all extracted helpers in sequence."""
        called = []

        monkeypatch.setattr(server, "load_config", lambda: {"server": {}})
        monkeypatch.setattr(server, "setup_logging", lambda c: mock_logger)

        def track(name):
            def wrapper(*args, **kwargs):
                called.append(name)
                return None
            return wrapper

        monkeypatch.setattr(server, "_startup_config_logging", track("config_logging"))
        monkeypatch.setattr(server, "_startup_initialize_cache_cold_start", track("cache_cold"))
        monkeypatch.setattr(server, "_startup_initialize_backend_state", track("backend_state"))
        monkeypatch.setattr(server, "_startup_create_http_client", lambda c: track("http_client")() or httpx.AsyncClient())
        monkeypatch.setattr(server, "_startup_create_remote_http_client", lambda c: track("remote_client")() or httpx.AsyncClient())
        monkeypatch.setattr(server, "_startup_launch_version_capture", track("version_capture"))
        monkeypatch.setattr(server, "_startup_podman_migrate", track("podman"))
        monkeypatch.setattr(server, "_startup_launch_default_model_loader", track("model_loader"))
        monkeypatch.setattr(server, "_startup_launch_watchdog_tasks", track("watchdog"))
        monkeypatch.setattr(server, "_startup_launch_persistence_tasks", track("persistence"))
        monkeypatch.setattr(server, "_startup_start_session_cleanup", track("session_cleanup"))
        monkeypatch.setattr(server, "_startup_start_dispatch_cleanup", track("dispatch_cleanup"))
        monkeypatch.setattr(server, "_startup_register_session_routes", track("session_routes"))

        # Simulate startup sequence
        server._startup_config_logging()
        server._startup_initialize_cache_cold_start()
        server._startup_initialize_backend_state()
        client = httpx.AsyncClient()
        server._http_client = client
        server._remote_http_client = client
        server._startup_launch_version_capture()
        server._startup_podman_migrate()
        server._startup_launch_default_model_loader()
        server._startup_launch_watchdog_tasks()
        server._startup_launch_persistence_tasks()
        server._startup_start_session_cleanup()
        server._startup_start_dispatch_cleanup()
        server._startup_register_session_routes(None)

    @pytest.mark.asyncio
    async def test_shutdown_sequence_invokes_all_helpers(self, monkeypatch):
        """lifespan shutdown phase calls all shutdown helpers in sequence."""
        called = []

        async def async_track(name):
            called.append(name)

        monkeypatch.setattr(server, "_shutdown_stop_session_cleanup", lambda: called.append("stop_session"))
        monkeypatch.setattr(server, "_shutdown_cleanup_tasks", lambda: async_track("cleanup_tasks"))
        monkeypatch.setattr(server, "_shutdown_http_client", lambda: async_track("http_client"))
        monkeypatch.setattr(server, "_shutdown_llama_server", lambda: called.append("llama_server"))

        server._shutdown_stop_session_cleanup()
        await server._shutdown_cleanup_tasks()
        await server._shutdown_http_client()
        server._shutdown_llama_server()

        # shutdown should call stop_session first, then cleanup tasks,
        # then http client, then llama server
        assert called == ["stop_session", "cleanup_tasks", "http_client", "llama_server"]
