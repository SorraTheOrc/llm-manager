#!/usr/bin/env python3
"""Tests for scripts/slot_geometry_arm.py (LP-0MVNCMDW50043RSP AC4).

The arm helper is the apply/verify half of the slot-geometry A/B protocol
(docs/dev/slot-geometry-ab.md). It must:

  * encode both arms' geometry in one place (A = 2 x 131,072 current,
    B = 1 x 262,144);
  * patch only the geometry keys in ``proxy/config.yaml`` and never claim a
    change it did not make (dry-run by default, ``--write`` to apply);
  * keep the routing band non-collapsed (cold < effective warm) in both arms,
    because a collapsed band makes the cached-ratio routing check unreachable
    (LP-0MSI2M5BT004BCDP);
  * verify post-restart reality from llama-server (``/slots`` then ``/props``)
    and refuse to report success on any mismatch.

The HTTP paths are exercised against a throwaway local HTTP server, so no
running proxy or GPU is required.
"""

from __future__ import annotations

import http.server
import json
import os
import socket
import subprocess
import threading

import pytest

SCRIPT = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "scripts",
        "slot_geometry_arm.py",
    )
)

CONFIG = """\
server:
  # Slot pool: also exported as LLAMA_PARALLEL by the lifecycle.
  session_slot_pool_size: 2
  local_model_ctx_size: 262144
  local_large_context_cold_cache_threshold: 38000
  local_large_context_warm_cache_threshold: 120000
  local_large_context_economic_ceiling_ratio: 0.4578
"""


def _run(*args, expect=0):
    proc = subprocess.run(
        ["python3", SCRIPT, *args], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == expect, f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    return proc


def _json(*args, expect=0):
    return json.loads(_run(*args, "--json", expect=expect).stdout)


def _config(tmp_path, text=CONFIG):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# A throwaway llama-server stand-in (only the endpoints the probe reads).
# ---------------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    slots_payload = None
    props_payload = None
    slots_status = 200

    def do_GET(self):
        if self.path.startswith("/slots"):
            if self.slots_status != 200 or self.slots_payload is None:
                self.send_response(self.slots_status)
                self.end_headers()
                self.wfile.write(b'{"error":{}}')
                return
            body = json.dumps(self.slots_payload).encode()
        elif self.path.startswith("/props"):
            body = json.dumps(self.props_payload).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep test output clean
        pass


@pytest.fixture
def llama_stub():
    def _start(slots=None, props=None, slots_status=200):
        handler = type(
            "H",
            (_Handler,),
            {
                "slots_payload": slots,
                "props_payload": props,
                "slots_status": slots_status,
            },
        )
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, f"http://127.0.0.1:{server.server_address[1]}"

    started = []
    yield lambda **kwargs: started.append(_start(**kwargs)) or started[-1]
    for server, _ in started:
        server.shutdown()
        server.server_close()


class _Slots:
    """Two slots x 131,072 tokens — the live Arm A geometry."""

    slots = [{"id": 0, "n_ctx": 131072}, {"id": 1, "n_ctx": 131072}]
    props = {
        "total_slots": 2,
        "default_generation_settings": {"n_ctx": 131072},
    }


# ===================================================================
# Arm definitions
# ===================================================================


def test_show_reports_both_arm_definitions(tmp_path):
    data = _json("show", "--config", str(_config(tmp_path)))
    assert data["arms"]["A"]["slots"] == 2
    assert data["arms"]["A"]["per_slot_ctx"] == 131072
    assert data["arms"]["B"]["slots"] == 1
    assert data["arms"]["B"]["per_slot_ctx"] == 262144


def test_arm_b_raises_the_warm_ceiling_to_the_single_slot_clamp(tmp_path):
    data = _json("show", "--config", str(_config(tmp_path)))
    arm_b = data["arms"]["B"]
    # physical clamp = 262144 // 1 - 4096
    assert arm_b["per_slot_route_clamp"] == 258048
    assert arm_b["config"]["local_large_context_warm_cache_threshold"] == "258048"
    # the economic ceiling must be raised with it or it shadows the cap
    ratio = float(arm_b["config"]["local_large_context_economic_ceiling_ratio"])
    assert round(ratio * 262144) >= 258048
    assert arm_b["effective_warm_threshold"] == 258048


def test_both_arms_keep_the_routing_band_non_collapsed(tmp_path):
    data = _json("show", "--config", str(_config(tmp_path)))
    for arm in ("A", "B"):
        assert data["arms"][arm]["band_width"] > 0, arm
        assert data["arms"][arm]["cold_threshold"] < data["arms"][arm][
            "effective_warm_threshold"
        ]


def test_show_reports_the_live_config_geometry(tmp_path):
    data = _json("show", "--config", str(_config(tmp_path)))
    assert data["live"]["slots"] == 2
    assert data["live"]["per_slot_ctx"] == 131072
    assert data["live"]["matching_arm"] == "A"


# ===================================================================
# apply
# ===================================================================


def test_apply_is_a_dry_run_by_default(tmp_path):
    config = _config(tmp_path)
    before = config.read_text()
    data = _json("apply", "--arm", "B", "--config", str(config))
    assert data["written"] is False
    assert config.read_text() == before
    changed = {c["key"]: c for c in data["changes"] if c["from"] != c["to"]}
    assert changed["session_slot_pool_size"]["to"] == "1"
    assert changed["local_large_context_warm_cache_threshold"]["to"] == "258048"


def test_apply_write_patches_only_the_geometry_keys(tmp_path):
    config = _config(tmp_path)
    _run("apply", "--arm", "B", "--config", str(config), "--write")
    text = config.read_text()
    assert "session_slot_pool_size: 1\n" in text
    assert "local_large_context_warm_cache_threshold: 258048\n" in text
    # untouched keys and comments survive
    assert "local_large_context_cold_cache_threshold: 38000\n" in text
    assert "local_model_ctx_size: 262144\n" in text
    assert "# Slot pool: also exported as LLAMA_PARALLEL by the lifecycle." in text
    assert "local_large_context_warm_cache_threshold: 120000" not in text


def test_apply_reports_the_restart_command_and_server_flags(tmp_path):
    data = _json("apply", "--arm", "B", "--config", str(_config(tmp_path)))
    assert data["llama_server_flags"] == ["--parallel", "1", "--ctx-size", "262144"]
    assert "--restart" in data["restart_command"]


def test_apply_refuses_a_config_without_the_geometry_keys(tmp_path):
    config = _config(tmp_path, "server:\n  llama_server_port: 8080\n")
    proc = _run("apply", "--arm", "B", "--config", str(config), expect=1)
    assert "session_slot_pool_size" in proc.stderr


def test_apply_rejects_an_unknown_arm(tmp_path):
    proc = _run("apply", "--arm", "C", "--config", str(_config(tmp_path)), expect=2)
    assert "invalid choice" in proc.stderr.lower()


# ===================================================================
# verify
# ===================================================================


def test_verify_accepts_matching_arm_a(llama_stub, tmp_path):
    server, url = llama_stub(slots=_Slots.slots, props=_Slots.props)
    data = _json("verify", "--arm", "A", "--llama-url", url, "--model", "Qwen3")
    assert data["verified"] is True
    assert data["measured"] == {"slots": 2, "per_slot_ctx": 131072}
    assert data["source"] == "slots"


def test_verify_rejects_the_wrong_slot_count(llama_stub, tmp_path):
    """Arm A's live geometry is NOT arm B — verify must refuse success."""
    server, url = llama_stub(slots=_Slots.slots, props=_Slots.props)
    data = _json(
        "verify", "--arm", "B", "--llama-url", url, "--model", "Qwen3", expect=1
    )
    assert data["verified"] is False
    assert any("slot" in problem for problem in data["problems"])


def test_verify_rejects_the_wrong_per_slot_context(llama_stub, tmp_path):
    slots = [{"id": 0, "n_ctx": 262144}]
    server, url = llama_stub(slots=slots, props={"total_slots": 1})
    data = _json(
        "verify", "--arm", "A", "--llama-url", url, "--model", "Qwen3", expect=1
    )
    assert data["verified"] is False
    assert any("n_ctx" in problem for problem in data["problems"])


def test_verify_rejects_inconsistent_per_slot_context(llama_stub, tmp_path):
    slots = [{"id": 0, "n_ctx": 131072}, {"id": 1, "n_ctx": 65536}]
    server, url = llama_stub(slots=slots, props={"total_slots": 2})
    data = _json(
        "verify", "--arm", "A", "--llama-url", url, "--model", "Qwen3", expect=1
    )
    assert any("differ" in problem for problem in data["problems"])


def test_verify_falls_back_to_props_when_slots_is_unavailable(llama_stub):
    server, url = llama_stub(
        slots=None,
        props={"total_slots": 1, "default_generation_settings": {"n_ctx": 262144}},
        slots_status=404,
    )
    data = _json("verify", "--arm", "B", "--llama-url", url, "--model", "Qwen3")
    assert data["verified"] is True
    assert data["source"] == "props"
    assert data["measured"] == {"slots": 1, "per_slot_ctx": 262144}


def test_verify_fails_when_the_server_is_unreachable():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = _run(
        "verify",
        "--arm",
        "A",
        "--llama-url",
        f"http://127.0.0.1:{port}",
        "--model",
        "Qwen3",
        expect=1,
    )
    assert "could not read" in proc.stderr.lower() or "unreachable" in proc.stderr.lower()


def test_verify_also_checks_the_routing_band_when_a_config_is_given(
    llama_stub, tmp_path
):
    """A collapsed band (cold >= warm) invalidates the arm even if slots match."""
    server, url = llama_stub(slots=_Slots.slots, props=_Slots.props)
    config = _config(
        tmp_path,
        CONFIG.replace(
            "local_large_context_warm_cache_threshold: 120000",
            "local_large_context_warm_cache_threshold: 38000",
        ).replace(
            "local_large_context_cold_cache_threshold: 38000",
            "local_large_context_cold_cache_threshold: 120000",
        ),
    )
    data = _json(
        "verify",
        "--arm",
        "A",
        "--llama-url",
        url,
        "--model",
        "Qwen3",
        "--config",
        str(config),
        expect=1,
    )
    assert any("band" in problem for problem in data["problems"])
