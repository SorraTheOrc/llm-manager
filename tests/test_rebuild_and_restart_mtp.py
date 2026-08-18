#!/usr/bin/env python3
"""Behavioural tests for scripts/rebuild-and-restart-mtp.sh.

Covers the LP-0MSXXKZOW0038XLK fix: a rebuild/deploy failure must NOT abort the
proxy-stack restart (the old binary keeps serving), and when the build artifact
exists, the deploy is re-attempted --deploy-only AFTER the restart.
"""
import json
import os
import subprocess

WRAPPER = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'rebuild-and-restart-mtp.sh'))


def _fake_tree(tmp_path):
    """A fake repo tree with a stub start-proxy.sh, plus crontab/curl stubs."""
    repo = tmp_path / "repo"
    (repo / "proxy" / "scripts").mkdir(parents=True)
    start_marker = tmp_path / "start-proxy.marker"
    (repo / "proxy" / "scripts" / "start-proxy.sh").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "start-proxy $*" >> "{start_marker}"\n',
        encoding="utf-8",
    )
    (repo / "proxy" / "scripts" / "start-proxy.sh").chmod(0o755)

    tools = tmp_path / "tools"
    tools.mkdir()
    cron_log = tmp_path / "cron.log"
    (tools / "crontab").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "crontab $*" >> "{cron_log}"\n'
        f'if [[ "${{1:-}}" == "-" ]]; then cat >> "{cron_log}"; fi\n',
        encoding="utf-8",
    )
    (tools / "crontab").chmod(0o755)
    (tools / "curl").write_text(
        "#!/usr/bin/env bash\n"
        'echo \'{"ready": true}\'\n',
        encoding="utf-8",
    )
    (tools / "curl").chmod(0o755)
    return repo, tools, start_marker, cron_log


def _fake_rebuild(tmp_path, rc):
    """A rebuild stub that fails with rc on the full run but succeeds on a
    --deploy-only re-attempt (mirrors a build-completed/deploy-failed run)."""
    rebuild_marker = tmp_path / "rebuild.marker"
    script = tmp_path / "fake-rebuild.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> "{rebuild_marker}"\n'
        'if [[ " $* " == *" --deploy-only "* ]]; then exit 0; fi\n'
        f'exit {rc}\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script, rebuild_marker


def _run(tmp_path, repo, tools, rebuild_script, new_bin, artifact, extra_env=None):
    log = tmp_path / "rebuild.log"
    env = {
        **os.environ,
        "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
        "MTP_REPO_ROOT": str(repo),
        "MTP_LOG": str(log),
        "MTP_REBUILD_CMD": f"bash {rebuild_script}",
        "MTP_NEW_BIN": str(new_bin),
        "MTP_ARTIFACT": str(artifact),
        "MTP_SKIP_DATE_GATE": "1",
        "MTP_HEALTH_URL": "http://unused/health",
        "MTP_POST_REDEPLOY_SLEEP": "0",
    }
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(["bash", WRAPPER], capture_output=True, text=True, env=env)
    return proc, log


def _fake_new_bin(tmp_path, mtp=True):
    p = tmp_path / "new-llama-server"
    p.write_text(
        "#!/usr/bin/env bash\n"
        + ("echo '--spec-type draft-mtp'" if mtp else "echo 'no-mtp-flag'") + "\n"
        + "echo 'version: 10480 (01818e495)'\n",
        encoding="utf-8",
    )
    p.chmod(0o755)
    return p


def test_syntax_checks():
    for script in ("scripts/rebuild-llama.sh", "scripts/rebuild-and-restart-mtp.sh"):
        path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', script))
        proc = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
        assert proc.returncode == 0, f"bash -n failed for {script}: {proc.stderr}"


def test_rebuild_failure_does_not_abort_restart(tmp_path):
    """A failing rebuild (rc=2) with no artifact must still restart the stack
    (old binary keeps serving) and must exit 1 with a warning, not abort."""
    repo, tools, start_marker, _ = _fake_tree(tmp_path)
    rebuild_script, rebuild_marker = _fake_rebuild(tmp_path, rc=2)
    new_bin = _fake_new_bin(tmp_path)

    artifact = tmp_path / "nonexistent" / "llama-server"
    proc, log = _run(tmp_path, repo, tools, rebuild_script, new_bin, artifact)

    assert start_marker.exists(), "stack was NOT restarted despite rebuild failure"
    assert (start_marker.read_text().strip().splitlines()[0]
            .startswith("start-proxy --restart"))
    assert "NOT aborting" in log.read_text(encoding="utf-8")
    assert proc.returncode == 1  # stack healthy, deploy failed
    # rebuild invoked exactly once (no deploy-only re-attempt without artifact)
    assert len(rebuild_marker.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_rebuild_failure_retries_deploy_after_restart(tmp_path):
    """A failing rebuild WITH an existing artifact must restart, re-attempt the
    deploy (--deploy-only), restart again, and succeed (exit 0)."""
    repo, tools, start_marker, _ = _fake_tree(tmp_path)
    rebuild_script, rebuild_marker = _fake_rebuild(tmp_path, rc=2)
    new_bin = _fake_new_bin(tmp_path)

    artifact = tmp_path / "build" / "bin" / "llama-server"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("#!/usr/bin/env bash\necho version\n", encoding="utf-8")
    artifact.chmod(0o755)

    proc, log = _run(tmp_path, repo, tools, rebuild_script, new_bin, artifact)

    log_text = log.read_text(encoding="utf-8")
    assert "re-attempt deploy" in log_text
    assert "MTP Rebuild Complete" in log_text
    assert proc.returncode == 0
    # rebuild called twice: first full, then --deploy-only
    calls = rebuild_marker.read_text(encoding="utf-8").strip().splitlines()
    assert len(calls) == 2
    assert "--deploy-only" in calls[1]
    # stack restarted twice (initial + post-re-deploy)
    assert len(start_marker.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_rebuild_success_restarts_and_completes(tmp_path):
    """A clean rebuild+deploy restarts once and reports completion."""
    repo, tools, start_marker, _ = _fake_tree(tmp_path)
    rebuild_script, rebuild_marker = _fake_rebuild(tmp_path, rc=0)
    new_bin = _fake_new_bin(tmp_path)

    artifact = tmp_path / "build" / "bin" / "llama-server"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("#!/usr/bin/env bash\necho version\n", encoding="utf-8")
    artifact.chmod(0o755)

    proc, log = _run(tmp_path, repo, tools, rebuild_script, new_bin, artifact)

    log_text = log.read_text(encoding="utf-8")
    assert "MTP Rebuild Complete" in log_text
    assert "VERIFIED: binary supports --spec-type draft-mtp" in log_text
    assert proc.returncode == 0
    assert len(rebuild_marker.read_text(encoding="utf-8").strip().splitlines()) == 1
    assert len(start_marker.read_text(encoding="utf-8").strip().splitlines()) == 1