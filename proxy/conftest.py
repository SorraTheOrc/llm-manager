import sys
from pathlib import Path
from typing import Any

import pytest

# Insert the ``proxy/`` directory (this conftest's own directory) on
# ``sys.path`` so ``from tests.<module> import ...`` and ``import proxy.<x>``
# resolve regardless of the pytest invocation cwd. Using ``os.getcwd()``
# broke combined runs whose rootdir is the repository root (e.g. the test
# skill's changed-scope selection mixing proxy tests and .pi skill tests in
# one invocation).
proxy_dir = str(Path(__file__).resolve().parent)
if proxy_dir not in sys.path:
    sys.path.insert(0, proxy_dir)
# ``proxy/tests`` uses a bare ``from config_test_utils import ...`` (it has no
# ``__init__.py``, so a top-level ``tests`` name could collide with the
# .pi skill test packages when both are collected in one invocation). Put the
# tests directory directly on the path so the helper imports without a
# package prefix.
_proxy_tests_dir = str(Path(proxy_dir) / "tests")
if _proxy_tests_dir not in sys.path:
    sys.path.insert(0, _proxy_tests_dir)


@pytest.fixture(autouse=True)
def _reset_slot_counts_cache():
    """Reset the last-known slot counts cache before every test.

    ``_query_slots()`` maintains a module-level cache of the last successful
    (available, total) counts (graceful degradation, LP-0MSVP7XJ6008QPKX).
    Without a reset, an earlier test that populated the cache would make a
    later test observing a stubbed (0,0) /slots failure see degraded counts
    instead of 0/0.
    """
    import proxy.observability as obs

    obs._last_slot_counts_cache = None
    yield
    obs._last_slot_counts_cache = None


@pytest.fixture(autouse=True)
def _reset_slots_poll_cache():
    """Reset the coalesced /slots poll cache before every test.

    ``_fetch_slots_payload()`` caches /slots responses behind a short TTL
    (LP-0MUCEFD5B003TURT). Tests stub the HTTP client and expect each call to
    be observed, so the cache must not leak between tests.
    """
    import proxy.observability as obs

    obs.reset_slots_poll_cache()
    yield
    obs.reset_slots_poll_cache()


@pytest.fixture(autouse=True)
def _reset_slot_details_cache():
    """Reset the last-known per-slot detail cache before every test.

    ``_last_slot_details_cache`` is populated by the broadcast/SSE paths and
    consumed as a fallback by the ``/llama/local/status`` handler when the
    fresh per-slot query fails (LP-0MUFSVXID0039ZAQ). Without a reset, an
    earlier test that populated the cache would make a later test observing a
    stubbed failure see cached detail instead of ``[]``.
    """
    import proxy.observability as obs

    obs._last_slot_details_cache = []
    yield
    obs._last_slot_details_cache = []


@pytest.fixture(autouse=True)
def _reset_server_global_state(monkeypatch):
    """Reset global server counters that leak across tests.

    Module-level ``proxy.server`` state (generating/prefill/dispatch) is
    mutated by router_helpers during real proxy_to_local calls. Tests that
    touch those paths without resetting would pollute subsequent tests —
    e.g. a test leaving ``local_generating_queries==1`` causes the next
    test's ``_try_acquire_local_dispatch`` to 503 via the generating cap.
    Hardened to catch every known leased-state fixture without requiring
    each test file to individually reset it.
    """
    import proxy.server as server

    # Only reset; never touch mocks established by per-file fixtures.
    server.active_queries = 0
    server.local_active_queries = 0
    server.local_generating_queries = 0
    server.local_dispatch_records = {}
    server.local_generating_sessions = set()
    server.local_prefill_in_flight = {}
    # Don't rebind locks mid-test — they may be acquired by in-flight tasks.
    # Just ensure the containers pointed at by existing locks are empty.
    yield
    server.local_dispatch_records = {}
    server.local_generating_sessions = set()
    server.local_prefill_in_flight = {}
    server.active_queries = 0
    server.local_active_queries = 0
    server.local_generating_queries = 0


@pytest.fixture(autouse=True)
def _isolate_startup_ramp_config():
    """Isolate the process-global startup-ramp config between tests.

    ``proxy.startup_ramp._startup_ramp_config`` is a module-level variable
    mutated by ``set_startup_ramp_config()``.  A test that calls
    ``set_startup_ramp_config(None)`` installs the *enabled* defaults
    (``{"enabled": True, ...}``), not a disabled state — so a teardown that
    does ``set_startup_ramp_config(None)`` leaves the global enabled and
    causes 503-gate failures in all subsequent tests that expect the process
    default of ``None``.

    This autouse fixture saves the value before each test and restores it
    afterwards, preventing any test from inheriting a leaked enabled ramp
    configuration (LP-0MUD0D0DU005R2VQ).
    """
    import proxy.startup_ramp as ramp_module

    saved = ramp_module._startup_ramp_config
    yield
    ramp_module._startup_ramp_config = saved


@pytest.fixture(autouse=True)
def _reset_stall_circuit_breaker():
    """Reset the process-global stall circuit breaker between tests.

    ``StallCircuitBreaker._stall_timestamps`` is a module-level sliding window
    (default 300 s window / threshold 3). Stalls recorded by one test can
    therefore mark a provider unavailable for a later, unrelated test in the
    same pytest process — observed as
    ``test_remote_failover_budget.py::test_failover_budget_param_overrides_config``
    losing its second upstream attempt because an earlier test had already
    recorded three ``opencode-go`` stalls (LP-0MVGHHSUS003GPQ1). Ordering is
    timing dependent, so without this reset the suite is flaky by wall clock.

    Only the in-memory window is cleared (production behaviour is unchanged);
    the cooldown map is already reset by
    ``_isolate_provider_availability_state``.
    """
    import proxy.stall_circuit_breaker as breaker_module

    breaker_module.stall_circuit_breaker._stall_timestamps.clear()
    yield
    breaker_module.stall_circuit_breaker._stall_timestamps.clear()


@pytest.fixture(autouse=True)
def _isolate_provider_availability_state(tmp_path, monkeypatch):
    """Isolate persisted provider availability state between tests.

    Provider cooldowns and the usage-limit account quarantine are persisted
    to ``proxy/provider-state.json`` and restored at startup
    (LP-0MUI6KB67005X44B). ``mark_provider_unavailable()`` and the
    usage-limit quarantine sites now write on every cold-path mutation, so
    without a redirect a test would write the real runtime file and leak
    availability state across test modules.

    Point the loader/saver at a per-test temporary file (via
    ``LLAMA_PROXY_PROVIDER_STATE_FILE``) and clear the in-memory maps so every
    test starts from a clean, empty availability state. Tests can still read
    and write a state file — they just never touch the real one.
    """
    import proxy.provider as provider

    monkeypatch.setenv(
        provider._PROVIDER_STATE_FILE_ENV,
        str(tmp_path / "provider-state.json"),
    )
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._usage_reset_guessed.clear()
    provider._usage_limit_last_probe.clear()
    yield
    provider._provider_unavailable_until.clear()
    provider._provider_failure_count.clear()
    provider._usage_reset_at.clear()
    provider._usage_reset_guessed.clear()
    provider._usage_limit_last_probe.clear()


def _find_live_e2e_summary_data() -> tuple[dict[str, Any] | None, str | None]:
    """Locate live E2E summary payload and best available text rendering."""
    module_names = (
        "tests.test_plan_fallback_live_e2e",
        "test_plan_fallback_live_e2e",
        "proxy.tests.test_plan_fallback_live_e2e",
    )
    for name in module_names:
        mod = sys.modules.get(name)
        if mod is None:
            continue

        payload = getattr(mod, "_LATEST_SUMMARY_PAYLOAD", None)
        if payload is None:
            continue

        renderer = getattr(mod, "_render_summary_text", None)
        if callable(renderer):
            try:
                return payload, renderer(payload)
            except Exception:
                pass

        # Fallback plain-text rendering if module renderer is unavailable.
        text = [
            f"Base URL: {payload.get('base_url', 'n/a')}",
            f"Total requests: {payload.get('total_requests', 0)}",
        ]
        sessions = payload.get("sessions", {}) if isinstance(payload, dict) else {}
        for session_id, entries in sessions.items():
            text.append(f"Session: {session_id} ({len(entries) if isinstance(entries, list) else 0} request(s))")
        return payload, "\n".join(text)

    return None, None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Always print a human-readable live E2E summary in terminal output."""
    payload, summary_text = _find_live_e2e_summary_data()
    if payload is None or summary_text is None:
        return

    terminalreporter.section("plan live e2e summary", sep="=")
    terminalreporter.write_line(summary_text)



@pytest.fixture(autouse=True)
def _isolate_mode_switch_lock(tmp_path, monkeypatch):
    """Legacy no-op fixture retained as a placeholder.

    The mode-switch lock and the whole fast/cheap mode axis were removed
    (LP-0MVDT94F0003VHUA AC4); there is no lock state left to isolate. The
    fixture is kept (empty) so any test that names it explicitly does not
    error.
    """
    yield
