import os
import sys
from typing import Any, Dict, Optional, Tuple

proxy_dir = os.path.join(os.getcwd())
if proxy_dir not in sys.path:
    sys.path.insert(0, proxy_dir)


def _find_live_e2e_summary_data() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
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

