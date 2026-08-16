import os
import sys
from typing import Any

import pytest

proxy_dir = os.path.join(os.getcwd())
if proxy_dir not in sys.path:
    sys.path.insert(0, proxy_dir)


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

