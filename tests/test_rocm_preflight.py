#!/usr/bin/env python3
import json
import subprocess
import os

SCRIPT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'rocm-preflight.sh'))


def test_rocm_preflight_json():
    assert os.path.exists(SCRIPT), f"rocm-preflight script not found at {SCRIPT}"
    proc = subprocess.run([SCRIPT, '--dry-run', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout = proc.stdout.strip()
    if not stdout:
        raise AssertionError(f"No output from script. stderr: {proc.stderr}")
    try:
        data = json.loads(stdout)
    except Exception as e:
        raise AssertionError(f"Output is not valid JSON: {e}\nSTDOUT:{stdout}\nSTDERR:{proc.stderr}")
    # Basic schema checks
    assert 'ok' in data
    assert 'repo_present' in data
    assert 'gpu_detected' in data
    assert 'uname' in data
