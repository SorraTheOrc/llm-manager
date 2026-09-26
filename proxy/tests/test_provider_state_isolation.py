"""Isolation regression tests for persisted provider availability state.

Parent: LP-0MUI6KB67005X44B. Provider cooldowns and the usage-limit account
quarantine are persisted to ``proxy/provider-state.json`` and restored at
startup. Because ``mark_provider_unavailable()`` and the usage-limit quarantine
sites write on every cold-path mutation, tests must never touch the real
runtime file (LP-0MUIGX73I001862F).

The shared autouse fixture in ``proxy/conftest.py`` redirects the state path to
a per-test temporary file. These tests pin that contract:

- the configured path is never the real checkout file;
- a mutation writes only the isolated file;
- a fresh test starts from empty maps (no cross-run leakage);
- a stale real on-disk state file in the checkout cannot affect routing in
  tests.
"""

import json
import time
from pathlib import Path

import proxy.provider as provider


def _provider_cfg() -> dict:
    return {
        "name": "acme-primary",
        "type": "remote",
        "provider": "acme",
        "endpoint": "https://acme.example/v1",
        "api_key_env": "ACME_API_KEY",
        "model": "m1",
    }


def _real_state_file() -> Path:
    """The real state-file path (beside ``proxy/.mode``), ignoring the override."""
    return Path(provider.__file__).parent.parent / "provider-state.json"


def test_state_path_is_redirected_to_a_temp_file(tmp_path):
    configured = provider.default_provider_state_file()

    assert configured != _real_state_file()
    assert tmp_path in configured.parents


def test_mark_provider_unavailable_never_writes_the_real_state_file():
    real = _real_state_file()
    existed_before = real.exists()
    mtime_before = real.stat().st_mtime if existed_before else None

    provider.mark_provider_unavailable("acme-primary", 60)

    # The real runtime file is neither created nor modified...
    if not existed_before:
        assert not real.exists()
    else:
        assert real.stat().st_mtime == mtime_before
    # ...but the isolated state file captured the cooldown.
    assert provider.default_provider_state_file().exists()


def test_fresh_state_starts_empty_no_cross_run_leakage():
    # The conftest fixture clears the maps and points the path at a file that
    # does not exist yet, so a fresh test sees empty availability state.
    assert provider._provider_unavailable_until == {}
    assert provider._usage_reset_at == {}

    assert provider.load_provider_state() == (0, 0)

    assert provider._provider_unavailable_until == {}
    assert provider._usage_reset_at == {}


def test_real_on_disk_file_does_not_affect_routing():
    real = _real_state_file()
    cfg = _provider_cfg()
    account_key = provider._usage_limit_account_key(cfg)
    now = time.time()
    # Preserve any genuine runtime state so this test cannot destroy it.
    original = real.read_bytes() if real.exists() else None
    real.write_text(
        json.dumps(
            {
                "version": 1,
                "provider_unavailable_until": {"acme-primary": now + 3600},
                "usage_reset_at": {account_key: now + 3600},
            }
        ),
        encoding="utf-8",
    )
    try:
        # The isolation fixture points the loader at a separate temp file, so
        # the stale real file is invisible: state is empty and the provider is
        # not skipped.
        assert provider.default_provider_state_file() != real
        assert provider.load_provider_state() == (0, 0)
        assert provider.resolve_provider({"providers": [cfg]}) == cfg
    finally:
        if original is None:
            real.unlink(missing_ok=True)
        else:
            real.write_bytes(original)
