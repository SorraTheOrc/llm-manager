import json
import os
import sys

proxy_dir = os.path.join(os.getcwd())
if proxy_dir not in sys.path:
    sys.path.insert(0, proxy_dir)


def _find_live_e2e_summary_payload():
    """Locate the live E2E summary payload when that module has run."""
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
        if payload is not None:
            return payload
    return None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Always print the live E2E summary block in console output.

    This makes the summary visible even when pytest output capture is enabled
    (i.e., without requiring `-s`).
    """
    payload = _find_live_e2e_summary_payload()
    if payload is None:
        return

    terminalreporter.section("plan live e2e summary", sep="=")
    terminalreporter.write_line(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

