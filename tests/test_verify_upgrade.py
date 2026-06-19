#!/usr/bin/env python3
import json
import subprocess
import os

SCRIPT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'verify-upgrade.sh'))

def test_verify_upgrade_dry_run_json():
    assert os.path.exists(SCRIPT), f"verify-upgrade script not found at {SCRIPT}"
    proc = subprocess.run([SCRIPT, '--dry-run', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout = proc.stdout.strip()
    if not stdout:
        raise AssertionError(f"No output from script. stderr: {proc.stderr}")
    try:
        data = json.loads(stdout)
    except Exception as e:
        raise AssertionError(f"Output is not valid JSON: {e}\nSTDOUT:{stdout}\nSTDERR:{proc.stderr}")
    assert 'ok' in data
    assert 'base_url' in data
    assert 'model' in data
    assert 'planned_steps' in data
