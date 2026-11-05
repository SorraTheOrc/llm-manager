#!/usr/bin/env python3
"""Behavioural tests for scripts/rebuild-llama.sh.

Covers the LP-0MSXXKZOW0038XLK deploy-ordering fix: the deploy step must stop
running llama-server processes bound to the deploy path BEFORE copying, must
copy the binary AND its sibling shared libs, and must patch the RUNPATH to
$ORIGIN so the deployed artifacts resolve from the deploy dir.

Covers the LP-0MUK0KYMV003TIWV fix: `--deploy-only` re-deploys an already-built
artifact and must not require build-only tooling (`cmake`), so the script no
longer aborts with "cmake missing" on hosts/CI images without a compiler.

Covers LP-0MW3K7Y3C003WEXW: the build configures `-DGGML_CPU_REPACK=OFF` (the
`ggml_cpu_extra_work_size` crash path) plus an explicit `-DGGML_NATIVE=ON`, the
CPU-backend census gate runs BEFORE the live deploy dir is touched, and every
deploy writes a provenance record (commit, cmake flags, artifact sha256).
"""
import json
import os
import shutil
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


def _path_without(tmp_path, excluded):
    """Mirror the current PATH into a shadow dir, omitting executables named
    ``excluded``.

    Reproduces environments (e.g. CI images) where a build-only tool such as
    ``cmake`` is absent, so a test can assert that ``--deploy-only`` never
    consults it (LP-0MUK0KYMV003TIWV).
    """
    shadow = tmp_path / "shadow-bin"
    shadow.mkdir()
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry or not os.path.isdir(entry):
            continue
        for name in os.listdir(entry):
            if name == excluded:
                continue
            src = os.path.join(entry, name)
            dest = shadow / name
            if dest.exists() or not os.access(src, os.X_OK):
                continue
            try:
                dest.symlink_to(src)
            except OSError:
                pass
    return shadow


def _run(extra_args, env_extra=None):
    env = {**os.environ}
    if env_extra:
        env.update(env_extra)
    return subprocess.run([SCRIPT, *extra_args], capture_output=True, text=True, env=env)


def _census_stub(tmp_path, *, verbose=False):
    """A stand-in for scripts/lib/ggml_cpu_census.py (GGML_CENSUS_SCRIPT).

    The real census needs a real ELF artifact; the deploy fixtures here are
    plain files, so the gate is exercised through its documented override.
    """
    stub = tmp_path / "census_stub.py"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"verbose = {verbose!r}\n"
        "library = sys.argv[1]\n"
        "ok = not verbose\n"
        "report = {\n"
        "    'library': library,\n"
        "    'generic_kernels': 3,\n"
        "    'arch_kernels': ['ggml_vec_dot_q4_K_q8_K'],\n"
        "    'hot_kernels': {'ggml_vec_dot_q4_K_q8_K': {'present': True,\n"
        "        'disassembled': True, 'vector_insns': 0 if verbose else 61,\n"
        "        'generic_tail_call': verbose}},\n"
        "    'arch_specific': ok,\n"
        "    'repack_registered': verbose,\n"
        "    'repack_linked': True,\n"
        "    'repack_markers': ['ggml_backend_cpu_repack'] if verbose else [],\n"
        "    'expect_repack': 'off',\n"
        "    'verdict': 'ok' if ok else 'fail',\n"
        "    'errors': [] if ok else [\n"
        "        'no arch-specific quant kernel codegen detected',\n"
        "        'CPU repack extra-buffer path is REGISTERED "
        "(GGML_CPU_REPACK=ON): this is the ggml_cpu_extra_work_size crash path'],\n"
        "    'warnings': [],\n"
        "}\n"
        "print(json.dumps(report))\n"
        "if not ok:\n"
        "    for err in report['errors']:\n"
        "        print('ERROR:', err, file=sys.stderr)\n"
        "sys.exit(0 if ok else 1)\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


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
    _fake_build(tmp_path)  # side effect: create the fake build tree; return value unused
    tools, order = _fake_tools(tmp_path)
    deploy_bin = tmp_path / "deploy" / "bin"
    deploy_bin.mkdir(parents=True)

    proc = _run(
        ["--deploy-only", "--skip-census", "--json", "--dir", str(tmp_path),
         "--deploy-path", str(deploy_bin / "llama-server")],
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


def test_deploy_only_does_not_require_cmake(tmp_path):
    """--deploy-only deploys an existing artifact and must not require cmake.

    cmake is only needed to build; requiring it in deploy-only mode made the
    script abort ("Errors: cmake missing") on hosts/CI images without cmake,
    even though the deploy path never builds (LP-0MUK0KYMV003TIWV).
    """
    _fake_build(tmp_path)
    tools, _ = _fake_tools(tmp_path)
    deploy_bin = tmp_path / "deploy" / "bin"
    deploy_bin.mkdir(parents=True)

    shadow = _path_without(tmp_path, "cmake")
    # Guard against a vacuous pass on machines where cmake is genuinely absent
    # from the inherited PATH: the shadow must actively hide it when present.
    assert shutil.which("cmake", path=str(shadow)) is None

    proc = subprocess.run(
        [SCRIPT, "--deploy-only", "--skip-census", "--json", "--dir", str(tmp_path),
         "--deploy-path", str(deploy_bin / "llama-server")],
        capture_output=True, text=True,
        env={**os.environ, "PATH": f"{tools}{os.pathsep}{shadow}"},
    )

    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    data = json.loads(proc.stdout)
    assert data["ok"] == 1
    assert (deploy_bin / "llama-server").exists()
    # The fake build tree is not a git repo, so the commit falls back cleanly.
    assert data["git_commit"] == "unknown"


def test_deploy_only_requires_existing_artifact(tmp_path):
    """--deploy-only without a prior build must fail fast with a clear error."""
    tools, _ = _fake_tools(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    deploy_bin = tmp_path / "deploy" / "bin"
    deploy_bin.mkdir(parents=True)

    # Deactivate the pkill stub so the "no artifact" guard fires before deploy
    proc = subprocess.run(
        [SCRIPT, "--deploy-only", "--skip-census", "--json", "--dir", str(empty),
         "--deploy-path", str(deploy_bin / "llama-server")],
        capture_output=True, text=True,
        env={**os.environ, "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}"},
    )
    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert data["ok"] == 0
    assert "no artifact" in data["errors"][0]


# ---------------------------------------------------------------------------
# LP-0MW3K7Y3C003WEXW: CPU-backend build facts and the census gate
# ---------------------------------------------------------------------------


def test_dry_run_plans_the_repack_free_native_cpu_build():
    """The plan must show the two build facts the crash fix depends on."""
    proc = subprocess.run([SCRIPT, '--dry-run', '--json'], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)

    flags = data["cmake_flags"]
    assert "-DGGML_CPU_REPACK=OFF" in flags, flags
    assert "-DGGML_NATIVE=ON" in flags, flags
    assert data["cpu_repack"] == "OFF"
    assert data["expect_cpu_repack"] == "off"
    # The census is planned BEFORE the live deploy dir is touched.
    planned = data["planned_steps"][0]
    assert planned.index("verify_cpu_backend") < planned.index("copy_binary_and_libs")


def test_with_cpu_repack_restores_upstream_default_and_expectation():
    proc = subprocess.run(
        [SCRIPT, '--dry-run', '--json', '--with-cpu-repack'], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)

    assert "-DGGML_CPU_REPACK=ON" in data["cmake_flags"]
    assert data["expect_cpu_repack"] == "on"


def test_unknown_expect_repack_value_is_rejected():
    proc = subprocess.run(
        [SCRIPT, '--dry-run', '--json', '--expect-cpu-repack', 'maybe'],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "expect-cpu-repack" in proc.stderr


def test_deploy_is_refused_before_touching_the_live_dir_when_census_fails(tmp_path):
    """A CPU backend that reproduces the crash path must never be deployed.

    The gate runs before the old server is stopped and before anything is
    copied, so a failing artifact leaves the running deployment untouched.
    """
    _fake_build(tmp_path)
    tools, order = _fake_tools(tmp_path)
    stub = _census_stub(tmp_path, verbose=True)
    deploy_bin = tmp_path / "deploy" / "bin"
    deploy_bin.mkdir(parents=True)

    proc = _run(
        ["--deploy-only", "--json", "--dir", str(tmp_path),
         "--deploy-path", str(deploy_bin / "llama-server")],
        {"PATH": f"{tools}{os.pathsep}{os.environ['PATH']}", "GGML_CENSUS_SCRIPT": str(stub)},
    )

    assert proc.returncode == 2, f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["ok"] == 0
    assert "cpu_backend_census_failed" in data["errors"][0]

    # Nothing was copied and the live server was never stopped.
    assert not (deploy_bin / "llama-server").exists()
    recorded = order.read_text(encoding="utf-8") if order.exists() else ""
    assert "pkill" not in recorded, f"the live server must not be stopped: {recorded!r}"

    # The reason is reported, not just a bare failure.
    assert "ggml_cpu_extra_work_size" in proc.stderr


def test_deploy_writes_a_provenance_record_with_checksums(tmp_path):
    """AC3: the build record carries the commit, flags and artifact sha256."""
    import hashlib

    build_bin = _fake_build(tmp_path)
    tools, _ = _fake_tools(tmp_path)
    stub = _census_stub(tmp_path)
    deploy_bin = tmp_path / "deploy" / "bin"
    deploy_bin.mkdir(parents=True)
    record = tmp_path / "record" / "build-record.json"

    proc = _run(
        ["--deploy-only", "--json", "--dir", str(tmp_path), "--record-file", str(record),
         "--deploy-path", str(deploy_bin / "llama-server")],
        {"PATH": f"{tools}{os.pathsep}{os.environ['PATH']}", "GGML_CENSUS_SCRIPT": str(stub)},
    )

    assert proc.returncode == 0, f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["ok"] == 1
    assert data["build_record"] == str(record)
    assert data["cpu_census_deployed"]["verdict"] == "ok"
    assert data["cpu_census_built"]["arch_specific"] is True

    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["work_item"] == "LP-0MW3K7Y3C003WEXW"
    assert "-DGGML_CPU_REPACK=OFF" in payload["cmake_flags"]
    assert payload["cpu_repack"] == "OFF"

    deployed = {entry["path"]: entry for entry in payload["artifacts"]}
    deployed_lib = str(deploy_bin / "libggml-cpu.so.0.20.1")
    assert deployed_lib in deployed
    expected = hashlib.sha256((build_bin / "libggml-cpu.so.0.20.1").read_bytes()).hexdigest()
    assert deployed[deployed_lib]["sha256"] == expected


def test_skip_census_deploys_non_elf_fixtures(tmp_path):
    """--skip-census is the documented emergency bypass for the artifact gate."""
    _fake_build(tmp_path)
    tools, _ = _fake_tools(tmp_path)
    deploy_bin = tmp_path / "deploy" / "bin"
    deploy_bin.mkdir(parents=True)

    proc = _run(
        ["--deploy-only", "--skip-census", "--json", "--dir", str(tmp_path),
         "--deploy-path", str(deploy_bin / "llama-server")],
        {"PATH": f"{tools}{os.pathsep}{os.environ['PATH']}"},
    )

    assert proc.returncode == 0, f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["census_skipped"] is True
    assert data["cpu_census_deployed"] is None
    assert (deploy_bin / "llama-server").exists()


def test_build_only_stages_the_artifact_without_touching_the_live_deploy_dir():
    """--build-only is the staging path for a coordinated restart (AC4).

    It must plan no stop/copy/patch steps, so an operator can inspect the
    artifact and its census before deciding when to promote it.
    """
    proc = subprocess.run(
        [SCRIPT, '--dry-run', '--json', '--build-only'], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)

    planned = data["planned_steps"][0]
    assert "verify_cpu_backend" in planned
    for forbidden in ("stop_old_server", "copy_binary_and_libs", "patch_runpath", "restart_service"):
        assert forbidden not in planned, f"--build-only must not plan {forbidden}: {planned}"


def test_build_only_and_deploy_only_are_mutually_exclusive():
    proc = subprocess.run(
        [SCRIPT, '--dry-run', '--json', '--build-only', '--deploy-only'],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "mutually exclusive" in proc.stderr
