#!/usr/bin/env python3
import json
import os
import subprocess

SCRIPT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'rebuild-container.sh'))

def test_rebuild_container_dry_run_json():
    assert os.path.exists(SCRIPT), f"rebuild-container script not found at {SCRIPT}"
    proc = subprocess.run([SCRIPT, '--dry-run', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout = proc.stdout.strip()
    if not stdout:
        raise AssertionError(f"No output from script. stderr: {proc.stderr}")
    try:
        data = json.loads(stdout)
    except Exception as e:
        raise AssertionError(f"Output is not valid JSON: {e}\nSTDOUT:{stdout}\nSTDERR:{proc.stderr}")
    assert 'ok' in data
    assert 'image' in data
    assert 'tag' in data
    assert 'update_containerfile' in data
