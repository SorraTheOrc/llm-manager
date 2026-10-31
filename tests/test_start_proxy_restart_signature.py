#!/usr/bin/env python3
"""Behavioural coverage for the ``start-proxy.sh --restart`` preserve-vs-restart decision.

What this module pins (LP-0MVNQ6CQY000GDY4): the **shell** branch of
``proxy/scripts/start-proxy.sh`` that compares the ``config_sha256`` recorded in
``$REPO_ROOT/.llama_server_signature.json`` with a freshly computed
``merged_config_sha256($REPO_ROOT/config.yaml, $REPO_ROOT/config.yaml)`` and
either preserves the co-located llama-server (warm KV / prompt cache survives a
proxy-only restart) or kills and restarts it:

* a matching signature preserves llama-server — the "Preserving running
  llama-server" line is emitted and **no** llama-server kill is issued
  (neither the graceful ``pkill -f`` nor the ``pkill -9 -f`` phase, nor the
  ``fuser -k 8080/tcp`` port fallback);
* a local-backend config change after the signature was written restarts
  llama-server;
* missing, empty or corrupt signature files — and a hash helper that cannot be
  imported — degrade to a conservative restart (never an unearned preserve);
* ``--keep-llama-server`` beats a mismatching signature (flag wins over hash).

Why a shell-level test exists at all: the Python suites pin the hash helper
(``proxy/tests/test_config_merge.py``) and the signature writer
(``proxy/tests/test_restart_lifecycle.py``), not the shell decision that acts on
them. A regression there — comparing the wrong path, ignoring the flag, parsing
the signature upside down — would silently kill a warm llama-server on every
restart or preserve a stale one.

Sandbox guarantees (AC5): every case runs a **copy** of the script inside
``tmp_path/proxy/`` (so ``REPO_ROOT`` — derived from the script location — is
sandbox-local), with its own minimal ``config.yaml``, an isolated ``$HOME``, an
ephemeral port, stubbed ``pkill``/``fuser`` (recording their argv into a log)
and stubbed ``ss``/``netstat`` port probes, plus a fake ``python3`` that
intercepts the final ``exec … -m uvicorn`` and exits instead of starting a
server. The live proxy, llama-server, TTS server and ports 8000/8080/8081 are
never touched and no blanket kill is ever issued. Unlike the bare
``tests/*.sh`` harnesses, this module is collected by the default pytest run.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# ``PROXY_ROOT`` mirrors the ``REPO_ROOT`` the script itself derives from its own
# location (``<repo>/proxy``); ``PROXY_PKG_DIR`` is the directory the script puts
# on ``sys.path`` before ``from config_merge import merged_config_sha256``.
PROXY_ROOT = REPO_ROOT / "proxy"
PROXY_PKG_DIR = PROXY_ROOT / "proxy"
SOURCE_SCRIPT = PROXY_ROOT / "scripts" / "start-proxy.sh"
REAL_PY = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable


def _load_merged_config_sha256():
    """Load the helper the script itself imports, without touching ``sys.path``.

    ``start-proxy.sh`` inserts ``$REPO_ROOT/proxy`` on ``sys.path`` and imports
    ``config_merge`` from there. Adding that directory here would be unsafe in a
    shared pytest session: it makes every module in ``proxy/proxy/`` importable
    under its bare name, so ``proxy/proxy/tokenizers.py`` would shadow the real
    ``tokenizers`` package for later tests. Loading the file by path gives the
    identical helper (the fixture hash must be byte-identical to the one
    recomputed at restart time) with no import side effects.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_start_proxy_restart_signature_config_merge",
        PROXY_PKG_DIR / "config_merge.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.merged_config_sha256


merged_config_sha256 = _load_merged_config_sha256()

# Minimal config: one local provider and a local-backend key the AC2 edit can
# move. No ``api_key_env`` anywhere, so key resolution is a no-op and no
# provider credential is needed.
CONFIG_YAML = (
    "models:\n"
    "  test-model:\n"
    "    providers:\n"
    "      - name: test-local\n"
    "        type: local\n"
    "        llama_model: Test\n"
    "server:\n"
    "  session_slot_pool_size: 1\n"
    "  llama_server_port: 8080\n"
)

# The llama-server port the script's ``fuser -k`` fallback targets.
LLAMA_PORT = 8080

FAKE_PY = """#!/usr/bin/env bash
# Intercept only the final `exec <python> -m uvicorn ...`; delegate the script's
# own helper calls (merged-config hashing, key resolution) to the real
# interpreter.
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "uvicorn" ]; then
    printf '%s\\n' "$*" >> "$FAKE_UVICORN_ARGV"
    echo "FAKE_UVICORN $*"
    exit 0
fi
exec "@REAL_PY@" "$@"
"""

# Kill helpers record their argv so a preserve case can prove that no
# llama-server kill was issued in ANY of the script's kill phases (graceful
# pkill, force pkill -9, fuser port fallback). Neither stub ever signals a
# process.
FAKE_KILLER = """#!/usr/bin/env bash
printf '%s %s\\n' "@NAME@" "$*" >> "$FAKE_KILL_LOG"
exit 0
"""


def _free_port() -> int:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


@dataclass
class _Sandbox:
    """An isolated copy of the real script plus everything it reads."""

    root: Path
    proxy_dir: Path
    env: dict[str, str]
    kill_log: Path
    argv_log: Path

    @property
    def signature(self) -> Path:
        """``$REPO_ROOT/.llama_server_signature.json`` for the sandbox."""
        return self.proxy_dir / ".llama_server_signature.json"

    @property
    def config(self) -> Path:
        """``$REPO_ROOT/config.yaml`` for the sandbox."""
        return self.proxy_dir / "config.yaml"


def _sandbox(tmp_path: Path, *, with_config_merge: bool = True) -> _Sandbox:
    """Build an isolated proxy tree and return it with its run environment.

    ``with_config_merge=False`` omits the ``config_merge`` helper from the
    sandbox, which is how the AC3 "unreadable hash" case makes the script's
    inline hash computation fail.
    """
    proxy_dir = tmp_path / "proxy"
    (proxy_dir / "scripts").mkdir(parents=True)
    (proxy_dir / "proxy").mkdir()
    shutil.copy2(SOURCE_SCRIPT, proxy_dir / "scripts" / "start-proxy.sh")
    helpers = ["api_key_resolver.py"] + (["config_merge.py"] if with_config_merge else [])
    for helper in helpers:
        shutil.copy2(PROXY_PKG_DIR / helper, proxy_dir / "proxy" / helper)

    config = proxy_dir / "config.yaml"
    config.write_text(CONFIG_YAML, encoding="utf-8")

    home = tmp_path / "home"
    (home / ".pi" / "agent").mkdir(parents=True)
    (home / ".pi" / "agent" / "auth.json").write_text("{}\n", encoding="utf-8")

    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "python3").write_text(FAKE_PY.replace("@REAL_PY@", REAL_PY), encoding="utf-8")
    (tools / "python3").chmod(0o755)
    # Port probes report "free" so the port-release waits return immediately and
    # never look at the live proxy ports.
    for name in ("ss", "netstat"):
        (tools / name).write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        (tools / name).chmod(0o755)
    for name in ("pkill", "fuser"):
        (tools / name).write_text(FAKE_KILLER.replace("@NAME@", name), encoding="utf-8")
        (tools / name).chmod(0o755)

    kill_log = tmp_path / "kill.log"
    argv_log = tmp_path / "uvicorn-argv.log"
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in (
            "START_PROXY_FOREGROUND",
            "LLAMA_PROXY_USE_ENV_KEYS",
            "LLAMA_PROXY_RESTART_REASON",
            "PROXY_PORT",
            "PORT",
            "PYTHONPATH",
        )
    }
    env.update(
        {
            "HOME": str(home),
            "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
            "PYTHONPATH": f"{proxy_dir}{os.pathsep}{proxy_dir / 'proxy'}",
            "FAKE_UVICORN_ARGV": str(argv_log),
            "FAKE_KILL_LOG": str(kill_log),
        }
    )
    return _Sandbox(
        root=tmp_path,
        proxy_dir=proxy_dir,
        env=env,
        kill_log=kill_log,
        argv_log=argv_log,
    )


def _run(sandbox: _Sandbox, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the sandboxed script with its cwd OUTSIDE ``$REPO_ROOT``.

    Running from ``tmp_path`` (not the sandbox ``proxy/`` dir) makes the AC1
    decoy assertion meaningful: a future refactor to a ``$PWD``-relative
    signature path would read the decoy and fail loudly instead of silently
    testing the live repository.
    """
    return subprocess.run(
        ["bash", str(sandbox.proxy_dir / "scripts" / "start-proxy.sh"), *args],
        cwd=sandbox.root,
        env=sandbox.env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _output(proc: subprocess.CompletedProcess[str]) -> str:
    return proc.stdout + proc.stderr


def _config_sha(sandbox: _Sandbox) -> str:
    """The hash the script records: base merged with itself (no mode axis)."""
    return merged_config_sha256(sandbox.config, sandbox.config)


def _signature_payload(config_sha256: str) -> dict[str, object]:
    """The signature schema ``lifecycle._write_llama_server_signature`` writes."""
    return {
        "mode": "unified",
        "model": "router",
        "parallel": 1,
        "router_mode": True,
        "config_sha256": config_sha256,
    }


def _write_signature(sandbox: _Sandbox, config_sha256: str) -> None:
    """Record the sandbox's ``$REPO_ROOT/.llama_server_signature.json``."""
    sandbox.signature.write_text(
        json.dumps(_signature_payload(config_sha256)), encoding="utf-8"
    )


def _uvicorn_argv(sandbox: _Sandbox) -> list[str]:
    """The final ``exec … -m uvicorn`` argv, once the fake interpreter ran."""
    deadline = time.monotonic() + 10
    while True:
        calls = (
            [line for line in sandbox.argv_log.read_text(encoding="utf-8").splitlines() if line]
            if sandbox.argv_log.exists()
            else []
        )
        if calls or time.monotonic() > deadline:
            return calls
        time.sleep(0.05)


def _llama_server_kills(sandbox: _Sandbox) -> list[str]:
    """Kill-log lines that would have killed llama-server.

    Covers both ``pkill`` phases (graceful and ``-9``) and the ``fuser -k``
    port fallback on the llama-server port.
    """
    if not sandbox.kill_log.exists():
        return []
    lines = [line for line in sandbox.kill_log.read_text(encoding="utf-8").splitlines() if line]
    return [line for line in lines if "llama-server" in line or f"{LLAMA_PORT}/tcp" in line]


def _killed_llama_server_via_pkill(sandbox: _Sandbox) -> bool:
    return any(
        line.startswith("pkill") and "llama-server" in line
        for line in _llama_server_kills(sandbox)
    )


def test_restart_preserves_llama_server_when_signature_matches(tmp_path):
    """AC1: an unchanged local-backend config keeps the warm llama-server."""
    sandbox = _sandbox(tmp_path)
    _write_signature(sandbox, _config_sha(sandbox))
    # Decoy: a signature in the invocation cwd carrying a mismatching hash. The
    # script must read $REPO_ROOT/.llama_server_signature.json, never $PWD.
    (sandbox.root / ".llama_server_signature.json").write_text(
        json.dumps(_signature_payload("0" * 64)),
        encoding="utf-8",
    )

    proc = _run(sandbox, "--foreground", "--restart", "--port", str(_free_port()))

    assert proc.returncode == 0, _output(proc)
    assert len(_uvicorn_argv(sandbox)) == 1, "the restart phase never reached the launch"
    assert "Preserving running llama-server" in _output(proc), _output(proc)
    assert _llama_server_kills(sandbox) == [], (
        "llama-server was killed despite a matching signature: "
        f"{_llama_server_kills(sandbox)}"
    )
    # The other services are still recycled — a proxy-only restart is not a
    # no-op restart.
    assert "uvicorn proxy" in sandbox.kill_log.read_text(encoding="utf-8")


def test_restart_restarts_llama_server_when_config_changed(tmp_path):
    """AC2: a local-backend change after the signature was written restarts it."""
    sandbox = _sandbox(tmp_path)
    recorded_sha = _config_sha(sandbox)
    _write_signature(sandbox, recorded_sha)

    sandbox.config.write_text(
        CONFIG_YAML.replace("llama_server_port: 8080", "llama_server_port: 9090"),
        encoding="utf-8",
    )
    assert _config_sha(sandbox) != recorded_sha, "the config edit did not move the merged hash"

    proc = _run(sandbox, "--foreground", "--restart", "--port", str(_free_port()))

    assert proc.returncode == 0, _output(proc)
    assert len(_uvicorn_argv(sandbox)) == 1, "the restart phase never reached the launch"
    assert "Preserving running llama-server" not in _output(proc), _output(proc)
    assert _killed_llama_server_via_pkill(sandbox), (
        f"no llama-server kill observed, kill log: {_llama_server_kills(sandbox)}"
    )


def test_keep_llama_server_flag_beats_signature_mismatch(tmp_path):
    """AC4: ``--keep-llama-server`` forces preservation on a stale signature."""
    sandbox = _sandbox(tmp_path)
    # Same stale-signature setup as AC2 — the flag must be the only difference.
    _write_signature(sandbox, _config_sha(sandbox))
    sandbox.config.write_text(
        CONFIG_YAML.replace("llama_server_port: 8080", "llama_server_port: 9090"),
        encoding="utf-8",
    )

    proc = _run(
        sandbox,
        "--foreground",
        "--restart",
        "--keep-llama-server",
        "--port",
        str(_free_port()),
    )

    assert proc.returncode == 0, _output(proc)
    assert len(_uvicorn_argv(sandbox)) == 1, "the restart phase never reached the launch"
    assert "Preserving running llama-server" in _output(proc), _output(proc)
    assert _llama_server_kills(sandbox) == [], (
        "--keep-llama-server was overridden by the signature comparison: "
        f"{_llama_server_kills(sandbox)}"
    )


def _degrade_signature(sandbox: _Sandbox, case: str) -> None:
    """Prepare one fail-safe case on an otherwise matching setup."""
    if case == "missing":
        return  # no signature file at all
    if case == "empty":
        sandbox.signature.write_text("", encoding="utf-8")
        return
    if case == "corrupt":
        sandbox.signature.write_text('{"mode": "unified", "config_sha256": ', encoding="utf-8")
        return
    # ``unreadable-hash``: the signature matches, but the sandbox lacks the
    # config_merge helper, so the script's inline hash computation fails and
    # _restart_sha stays empty — an unearned preserve must not happen.
    _write_signature(sandbox, _config_sha(sandbox))


@pytest.mark.parametrize("case", ["missing", "empty", "corrupt", "unreadable-hash"])
def test_fail_safe_signatures_restart_llama_server(tmp_path, case):
    """AC3: an unusable signature or hash always restarts (never preserves).

    Four variants, one script run each — every run pays the restart phase's own
    settle delays, so the case list is deliberately kept to one per distinct
    degradation path: no signature file, an empty file, unparseable JSON, and a
    hash helper that cannot be imported.
    """
    sandbox = _sandbox(tmp_path, with_config_merge=case != "unreadable-hash")
    _degrade_signature(sandbox, case)

    proc = _run(sandbox, "--foreground", "--restart", "--port", str(_free_port()))

    assert proc.returncode == 0, _output(proc)
    assert len(_uvicorn_argv(sandbox)) == 1, "the restart phase never reached the launch"
    assert "Preserving running llama-server" not in _output(proc), (
        f"{case}: an unusable signature earned a preserve: {_output(proc)}"
    )
    assert _killed_llama_server_via_pkill(sandbox), (
        f"{case}: no llama-server kill observed, kill log: {_llama_server_kills(sandbox)}"
    )
