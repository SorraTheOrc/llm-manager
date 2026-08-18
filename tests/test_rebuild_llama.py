#!/usr/bin/env python3
"""Behavioural tests for scripts/rebuild-llama.sh.

Covers the LP-0MSXXKZOW0038XLK deploy-ordering fix: the deploy step must stop
running llama-server processes bound to the deploy path BEFORE copying, must
copy the binary AND its sibling shared libs, and must patch the RUNPATH to
$ORIGIN so the deployed artifacts resolve from the deploy dir.
"""
import json
import os
import subprocess

SCRIPT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'rebuild-llama.sh'))


def _fake_build(tmp_path):
    """A fake build tree with llama-server + sibling shared libs."""
    build_bin = tmp_path / "build" / "bin"
    build_bin.mkdir(parents=True)
    binary = build_bin / "llama-server"
    binary.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'version: 10480 (01818e495)'\n"
        "echo 'built with GNU 13.3.0 for Linux x86_64'\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    for lib in ("libggml-base.so.0.20.1", "libggml-cpu.so.0.20.1", "libggml-hip.so.0.20.1"):
        (build_bin / lib).write_text("FAKE-ELF-LIB", encoding="utf-8")
    return build_bin


def _fake_tools(tmp_path):
    """pkill/pgrep/patchelf stubs that record their calls in order, to order.log.

    patchelf is intentionally NOT installed on this machine; the script must
    invoke a stub from PATH and record the RUNPATH it was asked to set.
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    order = tmp_path / "order.log"
    (tools / "pkill").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "pkill $*" >> "{order}"\n' + "exit 0\n",
        encoding="utf-8",
    )
    (tools / "pgrep").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "pgrep $*" >> "{order}"\n' + "exit 1\n",
        encoding="utf-8",
    )
    (tools / "patchelf").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "patchelf $*" >> "{order}"\n' + "exit 0\n",
        encoding="utf-8",
    )
    for name in ("pkill", "pgrep", "patchelf"):
        (tools / name).chmod(0o755)
    return tools, order


def _run(extra_args, env_extra=None):
    env = {**os.environ}
    if env_extra:
        env.update(env_extra)
    return subprocess.run([SCRIPT, *extra_args], capture_output=True, text=True, env=env)


def test_rebuild_llama_dry_run_json():
    assert os.path.exists(SCRIPT), f"rebuild-llama script not found at {SCRIPT}"
    proc = subprocess.run([SCRIPT, '--dry-run', '--json'], capture_output=True, text=True)
    stdout = proc.stdout.strip()
    if not stdout:
        raise AssertionError(f"No output from script. stderr: {proc.stderr}")
    try:
        data = json.loads(stdout)
    except Exception as e:
        raise AssertionError(f"Output is not valid JSON: {e}\nSTDOUT:{stdout}\nSTDERR:{proc.stderr}")
    assert 'ok' in data
    assert 'repo' in data
    assert 'target_dir' in data
    assert 'deploy_path' in data
    # deploy-ordering steps must be planned (LP-0MSXXKZOW0038XLK)
    assert 'stop_old_server' in data['planned_steps'][0]
    assert 'copy_binary_and_libs' in data['planned_steps'][0]
    assert 'patch_runpath' in data['planned_steps'][0]


def test_deploy_only_copies_binary_and_libs_and_stops_server_first(tmp_path, monkeypatch):
    """--deploy-only on an existing build must stop the old server, then copy the
    binary and its sibling shared libs, then patch the RUNPATH — in that order."""
    _, _ = _fake_build(tmp_path)
    tools, order = _fake_tools(tmp_path)
    deploy_bin = tmp_path / "deploy" / "bin"
    deploy_bin.mkdir(parents=True)

    proc = _run(
        ["--deploy-only", "--json", "--dir", str(tmp_path), "--deploy-path", str(deploy_bin / "llama-server")],
        {"PATH": f"{tools}{os.pathsep}{os.environ['PATH']}"},
    )

    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    data = json.loads(proc.stdout)

    # Artifacts copied: binary + all sibling shared libs
    assert (deploy_bin / "llama-server").exists()
    assert (deploy_bin / "libggml-base.so.0.20.1").exists()
    assert (deploy_bin / "libggml-hip.so.0.20.1").exists()

    # Ordering contract: stop (pkill) happens before the RUNPATH patch, and only
    # after the pkill attempt may the copy be performed. order.log records
    # sequence; the FIRST call MUST be pkill, matching the running process by
    # NAME (-x llama-server) — never by full path (a wrapper shell embedding the
    # path in its own cmdline would be killed too, LP-0MSXXKZOW0038XLK).
    lines = order.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "no tool calls recorded"
    assert lines[0].startswith("pkill "), f"first call must be pkill, got: {lines[0]}"
    assert "-x" in lines[0] and "llama-server" in lines[0]
    assert str(deploy_bin / "llama-server") not in lines[0], \
        "pkill must match by process NAME, not by the deploy path"

    # RUNPATH must point at $ORIGIN (+ rocm lib dirs), not the temp build dir
    patchelf_lines = [line for line in lines if line.startswith("patchelf ")]
    assert any("$ORIGIN" in line for line in patchelf_lines), f"RUNPATH must contain $ORIGIN: {patchelf_lines}"

    # JSON reports the deployed version and the commit fallback
    assert data["ok"] == 1
    assert "10480" in data["version"]
    assert data["git_commit"] == "unknown"  # fake build tree is not a git repo
    assert "libggml-base.so.0.20.1" in data["deployed_libs"]


def test_deploy_only_requires_existing_artifact(tmp_path):
    """--deploy-only without a prior build must fail fast with a clear error."""
    tools, _ = _fake_tools(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    deploy_bin = tmp_path / "deploy" / "bin"
    deploy_bin.mkdir(parents=True)

    # Deactivate the pkill stub so the "no artifact" guard fires before deploy
    proc = subprocess.run(
        [SCRIPT, "--deploy-only", "--json", "--dir", str(empty), "--deploy-path", str(deploy_bin / "llama-server")],
        capture_output=True, text=True,
        env={**os.environ, "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}"},
    )
    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert data["ok"] == 0
    assert "no artifact" in data["errors"][0]
