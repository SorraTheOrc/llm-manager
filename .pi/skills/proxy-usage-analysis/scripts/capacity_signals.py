"""Path-based access to the repo-level shared capacity-signal derivation.

The nightly analyzer lives under ``.pi/skills/proxy-usage-analysis/scripts/``
and is run with only that directory on ``sys.path`` (``analyze_proxy_usage.py``
inserts it, and the skill's ``conftest.py`` does the same for tests), so the
repo-level module it shares with ``scripts/capacity_signal_verification.py`` is
not importable by name. This shim loads it **by file path** — the documented
precedent in ``tests/test_admission_rate_bucket_report.py`` — and re-exports its
public surface, so the skill's modules write ``capacity_signals.stale_rate(...)``
and the derivation itself lives in exactly one place
(``scripts/lib/capacity_signals.py``, LP-0MVU8O4EL005IYP9 AC1.1).

The 05:00 cron runs from the repo root, so the path always resolves there. It
**fails closed**: a missing or unloadable shared module raises ``ImportError``
naming the expected path, never a silent fallback to a local copy — a
duplicated derivation is precisely the divergence this shim exists to prevent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

#: Repository root, derived from this file's location
#: (``<repo>/.pi/skills/proxy-usage-analysis/scripts/capacity_signals.py``).
REPO_ROOT = Path(__file__).resolve().parents[4]
#: The single shared implementation of the status-poll derivations.
SHARED_PATH = REPO_ROOT / "scripts" / "lib" / "capacity_signals.py"
#: Registry name — deliberately not ``capacity_signals`` so it can never shadow
#: (or be shadowed by) this shim module.
_MODULE_NAME = "_repo_capacity_signals"


def _load_shared():
    """Load ``scripts/lib/capacity_signals.py`` by path, or raise ``ImportError``."""
    loaded = sys.modules.get(_MODULE_NAME)
    if loaded is not None:
        return loaded
    if not SHARED_PATH.is_file():
        raise ImportError(
            "shared capacity-signal derivation not found at "
            f"{SHARED_PATH}; the nightly analyzer must run from a checkout of "
            "the repository that contains scripts/lib/capacity_signals.py"
        )
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, SHARED_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"cannot load the shared capacity-signal derivation at {SHARED_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_MODULE_NAME, None)
        raise
    return module


_shared = _load_shared()

DEFAULT_STALE_RATE_TARGET = _shared.DEFAULT_STALE_RATE_TARGET
STATUS_POLL_RE = _shared.STATUS_POLL_RE
STATUS_FIELD_RE = _shared.STATUS_FIELD_RE
FIELD_SLOTS_STALE = _shared.FIELD_SLOTS_STALE
LATENCY_BANDS = _shared.LATENCY_BANDS
is_status_poll = _shared.is_status_poll
status_fields = _shared.status_fields
parse_latency_ms = _shared.parse_latency_ms
parse_int = _shared.parse_int
latency_band = _shared.latency_band
has_live_local_work = _shared.has_live_local_work
in_window = _shared.in_window
stale_flag = _shared.stale_flag
tally_stale = _shared.tally_stale
stale_rate = _shared.stale_rate
stale_rate_check = _shared.stale_rate_check
