"""Documentation-drift and consistency tests for the idle-admission config.

Work items LP-0MVGHXD59008UOUD / LP-0MVJXGSEG0045IAF. The override's behaviour
is unit-tested in ``test_local_idle_admission.py``; this module covers the
*shipped-config* axis, which is what the ≥24 h measurement window depends on:

- ``proxy/config.yaml`` carries the two documented keys
  (``local_idle_admit_enabled`` / ``local_idle_admit_seconds``) with the types
  the reader expects;
- the shipped threshold is positive, so an *enabled* override can never be
  silently disabled by a bad threshold (``_idle_admit_config`` treats a
  non-positive value as "off", which would waste a measurement window: zero
  ``routing_idle_admit`` events and no explanation in the logs);
- ``provider._idle_admit_config`` reads the shipped config exactly as written —
  the flag it reports enabled must be the flag in the file, never an accident
  of defaulting;
- the knob names still appear in ``docs/dev/idle-window-measurements.md``, so
  the runbook and the config cannot drift apart.

These assertions hold whether the override is enabled (during the measurement
window) or disabled again afterwards, so they are not a temporary
experiment guard.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from proxy.provider import _idle_admit_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "proxy" / "config.yaml"
DOC_PATH = REPO_ROOT / "docs" / "dev" / "idle-window-measurements.md"


def _server_config() -> dict:
    with open(CONFIG_PATH) as handle:
        config = yaml.safe_load(handle) or {}
    return config["server"]


def test_shipped_config_declares_the_idle_admission_knobs():
    server = _server_config()
    assert isinstance(server["local_idle_admit_enabled"], bool)
    assert isinstance(server["local_idle_admit_seconds"], (int, float))


def test_shipped_threshold_cannot_silently_disable_an_enabled_override():
    """A non-positive threshold disables the override — never in the shipped file."""
    server = _server_config()
    threshold = float(server["local_idle_admit_seconds"])
    assert threshold > 0
    # And the reader agrees the override is usable with the shipped values.
    _enabled, read_threshold = _idle_admit_config(server)
    assert read_threshold == threshold


def test_reader_reports_the_flag_exactly_as_shipped():
    """The effective state is the file's state (no accidental defaulting)."""
    server = _server_config()
    enabled, _threshold = _idle_admit_config(server)
    assert enabled is server["local_idle_admit_enabled"]


def test_absent_flag_still_defaults_off():
    """Guard the default itself: only an explicit `true` enables the override."""
    enabled, threshold = _idle_admit_config({"local_idle_admit_seconds": 60})
    assert enabled is False
    assert threshold == 60.0


def test_runbook_documents_both_knobs():
    text = DOC_PATH.read_text()
    assert "local_idle_admit_enabled" in text
    assert "local_idle_admit_seconds" in text
    # The zero-downtime enablement path the operator uses for the window.
    assert "/admin/reload-config" in text
