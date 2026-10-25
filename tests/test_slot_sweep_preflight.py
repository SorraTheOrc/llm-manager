#!/usr/bin/env python3
"""Tests for scripts/slot_sweep_preflight.py (LP-0MVK0969J004MTG3).

The quiet-window preflight gate is the fail-closed precondition for the
N=2/N=3/N=4 slot sweep (LP-0MVGPEP3O000RKKD): a sweep may only start when no
other session holds a local lease. These tests assert the predicate clause by
clause (removing any single clause must fail at least one test), the exit-code
contract (0 quiet / 1 not-quiet / 2 endpoint unusable), the ``--json`` output
shape, and — via the real harness CLI — the corrected sweep command forms
(``--sweep --resume 2`` must list 2, 3 and no N=1; ``--slots 4`` must list 4)
and that neither the preflight nor the dry runs write to ``proxy/config.yaml``.
"""

import hashlib
import http.server
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "slot_sweep_preflight.py"
_CONFIG = REPO_ROOT / "proxy" / "config.yaml"

_module_path = SCRIPT
_spec = importlib.util.spec_from_file_location("slot_sweep_preflight", _module_path)
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)

evaluate_quiet = mod.evaluate_quiet
fetch_status = mod.fetch_status
build_result = mod.build_result
CHECK_NAMES = mod.CHECK_NAMES


# ── Status payload fixtures ──────────────────────────────────────────────────


def quiet_status(**overrides):
    """A /llama/local/status payload that the quiet predicate must accept."""
    status = {
        "llama_server_running": True,
        "local_owner_session_id": None,
        "local_idle_seconds": 600.0,
        "available_slots": 2,
        "total_slots": 2,
        "slots_stale": False,
        "slots": [
            {
                "slot_id": 0,
                "lease_state": "free",
                "owner_session_id": None,
                "is_processing": False,
                "phase": None,
            },
            {
                "slot_id": 1,
                "lease_state": "free",
                "owner_session_id": None,
                "is_processing": False,
                "phase": None,
            },
        ],
    }
    status.update(overrides)
    return status


# ── Predicate unit tests ─────────────────────────────────────────────────────


def test_quiet_pass_all_checks_ok():
    result = evaluate_quiet(quiet_status())
    assert result["quiet"] is True
    assert result["reasons"] == []
    assert all(check["ok"] for check in result["checks"].values())
    # Every defined clause is evaluated (removing one from the implementation
    # must fail this set-equality test).
    assert set(result["checks"]) == set(CHECK_NAMES)


@pytest.mark.parametrize(
    ("name", "status", "failing_check"),
    [
        ("llama server not running", {"llama_server_running": False}, "llama_server_running"),
        ("local dispatch lease held", {"local_owner_session_id": "herdr-1792774550-1330446"}, "local_owner_session_id"),
        (
            "leased-but-idle slot",
            {
                "slots": [
                    {
                        "slot_id": 1,
                        "lease_state": "idle_leased",
                        "owner_session_id": "herdr-x",
                        "is_processing": False,
                        "phase": None,
                    }
                ]
            },
            "slots_free",
        ),
        ("slot is_processing", {"slots": [dict(quiet_status()["slots"][0], is_processing=True)]}, "slots_idle"),
        ("available != total", {"available_slots": 1}, "slot_capacity"),
        ("zero total slots", {"total_slots": 0}, "slot_capacity"),
        ("idle below 120 s minimum", {"local_idle_seconds": 30.0}, "local_idle_seconds"),
        ("idle age not published", {"local_idle_seconds": None}, "local_idle_seconds"),
        ("no per-slot detail", {"slots": []}, "slots_present"),
        ("stale slot data", {"slots_stale": True}, "slots_fresh"),
    ],
)
def test_each_failing_clause_is_reported(name, status, failing_check):
    result = evaluate_quiet(quiet_status(**status))
    assert result["quiet"] is False, name
    assert result["checks"][failing_check]["ok"] is False, name
    assert any(reason.startswith(f"{failing_check}:") for reason in result["reasons"]), name
    # Every *other* clause stays green: a single deviation must trip exactly
    # the one clause it targets, so a carpenter can read which clause failed.
    for check_name, check in result["checks"].items():
        if check_name != failing_check:
            assert check["ok"] is True, f"{name}: unexpected extra failure in {check_name}"


def test_min_idle_seconds_is_a_parameter():
    assert evaluate_quiet(quiet_status(local_idle_seconds=30.0), min_idle_seconds=10.0)["quiet"] is True
    assert evaluate_quiet(quiet_status(local_idle_seconds=30.0), min_idle_seconds=120.0)["quiet"] is False


def test_owned_idle_slot_is_not_free_even_when_available_count_says_full():
    # The live proxy sample that motivated the per-slot check: slot 1
    # idle_leased but available_slots == total_slots == 2.
    status = quiet_status(
        slots=[
            quiet_status()["slots"][0],
            {
                "slot_id": 1,
                "lease_state": "idle_leased",
                "owner_session_id": "herdr-1792774550-1330446",
                "is_processing": False,
                "phase": None,
            },
        ],
        available_slots=2,
        total_slots=2,
    )
    result = evaluate_quiet(status)
    assert result["quiet"] is False
    assert result["checks"]["slots_free"]["ok"] is False
    assert any("idle_leased" in reason for reason in result["reasons"])


def test_build_result_unreachable():
    result = build_result(None, "ConnectionRefusedError: refused (unreachable: http://x)")
    assert result["quiet"] is False
    assert result["error"] is not None
    assert result["reasons"] == [f"status_unavailable: {result['error']}"]


def test_build_result_quiet_wraps_evaluation():
    result = build_result(quiet_status(), None, status_url="http://localhost:8000/llama/local/status")
    assert result["quiet"] is True
    assert result["error"] is None
    assert result["status_url"] == "http://localhost:8000/llama/local/status"
    assert "checked_at" in result


# ── CLI-level tests against a stub status server ─────────────────────────────


class _StatusHandler(http.server.BaseHTTPRequestHandler):
    """Stub /llama/local/status endpoint with swappable payload + status code."""

    payload = quiet_status()
    status_code = 200

    def do_GET(self):
        body = json.dumps(self.payload).encode("utf-8")
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def status_server():
    handler = type("Handler", (_StatusHandler,), {})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/llama/local/status"
    try:
        yield handler, url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _run_preflight(argv, cwd=None):
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + argv,
        capture_output=True,
        text=True,
        cwd=str(cwd or REPO_ROOT),
    )


def test_cli_quiet_json_exit_zero(status_server):
    _, url = status_server
    proc = _run_preflight(["--url", url, "--json"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["quiet"] is True
    assert payload["reasons"] == []
    assert payload["error"] is None
    assert payload["min_idle_seconds"] == 120.0
    assert payload["inputs"]["total_slots"] == 2


def test_cli_not_quiet_exit_one(status_server):
    handler, url = status_server
    handler.payload = quiet_status(
        slots=[
            quiet_status()["slots"][0],
            {
                "slot_id": 1,
                "lease_state": "idle_leased",
                "owner_session_id": "herdr-x",
                "is_processing": False,
                "phase": None,
            },
        ]
    )
    proc = _run_preflight(["--url", url, "--json"])
    assert proc.returncode == 1, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["quiet"] is False
    assert payload["error"] is None
    assert any("idle_leased" in reason for reason in payload["reasons"])


def test_cli_non_200_exit_two(status_server):
    handler, url = status_server
    handler.status_code = 503
    handler.payload = {"error": "unavailable"}
    proc = _run_preflight(["--url", url, "--json"])
    assert proc.returncode == 2, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["quiet"] is False
    assert "HTTP 503" in payload["error"]


def test_cli_unreachable_exit_two():
    # Port 9 (discard) is never listening on this host; connection is refused
    # immediately rather than hanging on a timeout.
    url = "http://127.0.0.1:9/llama/local/status"
    proc = _run_preflight(["--url", url, "--json", "--timeout", "2"])
    assert proc.returncode == 2, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["quiet"] is False
    assert payload["error"] is not None


def test_cli_human_output_mentions_verdict(status_server):
    _, url = status_server
    proc = _run_preflight(["--url", url])
    assert proc.returncode == 0
    assert "QUIET" in proc.stdout


def test_cli_min_idle_flag_is_honoured(status_server):
    handler, url = status_server
    handler.payload = quiet_status(local_idle_seconds=30.0)
    tight = _run_preflight(["--url", url, "--json"])
    assert tight.returncode == 1
    relaxed = _run_preflight(["--url", url, "--json", "--min-idle-seconds", "10"])
    assert relaxed.returncode == 0, relaxed.stdout + relaxed.stderr
    assert json.loads(relaxed.stdout)["quiet"] is True


# ── Harness dry-run validation (LP-0MVK0969J004MTG3 AC4/AC5) ─────────────────


def _run_harness_dry_run(argv):
    return subprocess.run(
        [sys.executable, "-m", "proxy.benchmarks.slot_benchmark"] + argv,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def _slot_counts_line(stdout):
    for line in stdout.splitlines():
        if line.strip().startswith("Slot counts:"):
            return [part.strip() for part in line.split(":", 1)[1].split(",") if part.strip()]
    return None


def test_sweep_resume_2_dry_run_lists_2_3_without_1():
    proc = _run_harness_dry_run(
        ["--sweep", "--resume", "2", "--concurrency", "3", "--prompt-tokens", "2000", "--dry-run"]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    counts = _slot_counts_line(proc.stdout)
    assert counts == ["2", "3"], f"expected [2, 3], got {counts!r} (stdout below)\n{proc.stdout}"


def test_slots_4_dry_run_lists_4():
    proc = _run_harness_dry_run(
        ["--slots", "4", "--concurrency", "3", "--prompt-tokens", "2000", "--dry-run"]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _slot_counts_line(proc.stdout) == ["4"]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preflight_and_dry_runs_do_not_write_config(status_server):
    """AC5: sha256 of proxy/config.yaml identical before and after."""
    assert _CONFIG.exists(), "proxy/config.yaml must exist for the sweep harness"
    before = _sha256(_CONFIG)
    _, url = status_server

    preflight = _run_preflight(["--url", url, "--json"])
    assert preflight.returncode == 0, preflight.stdout + preflight.stderr
    dry_sweep = _run_harness_dry_run(
        ["--sweep", "--resume", "2", "--concurrency", "3", "--prompt-tokens", "2000", "--dry-run"]
    )
    assert dry_sweep.returncode == 0, dry_sweep.stdout + dry_sweep.stderr
    dry_4 = _run_harness_dry_run(
        ["--slots", "4", "--concurrency", "3", "--prompt-tokens", "2000", "--dry-run"]
    )
    assert dry_4.returncode == 0, dry_4.stdout + dry_4.stderr

    assert _sha256(_CONFIG) == before, "preflight + dry runs must not write proxy/config.yaml"
