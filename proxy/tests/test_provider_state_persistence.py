"""Tests for provider-availability persistence (LP-0MUIGX0QU0042NZZ).

Parent: LP-0MUI6KB67005X44B — persist the usage-limit account quarantine
(``_usage_reset_at``) and provider/entry cooldowns
(``_provider_unavailable_until``) across proxy restarts.

This slice covers the persistence helpers only:

- empty-map and both-map round-trips;
- expired entries dropped at load (and excluded from the written payload);
- missing / unreadable / malformed / non-dict state file handled safely;
- malformed entries skipped while valid siblings survive;
- absolute epoch expiries round-trip unchanged;
- state-file path default (beside ``proxy/.mode``) and env override;
- atomic write (temp file + replace, no partial file left behind, failure
  cleans up the temp file).

The mutation hooks and the startup restore call are separate child slices
(LP-0MUIGX4HF / LP-0MUIGX4H8); here the helpers are exercised directly.
"""

import json
import os
import time
from pathlib import Path

import proxy.provider as provider
import pytest


@pytest.fixture(autouse=True)
def _isolate_provider_state(tmp_path, monkeypatch):
    """Isolate the maps and the state-file path for every test in this module.

    The env override is pointed at a per-test temporary file so no test can
    ever touch the real ``proxy/provider-state.json``.
    """
    monkeypatch.setenv(
        provider._PROVIDER_STATE_FILE_ENV,
        str(tmp_path / "provider-state.json"),
    )
    provider._provider_unavailable_until.clear()
    provider._usage_reset_at.clear()
    yield
    provider._provider_unavailable_until.clear()
    provider._usage_reset_at.clear()


def _write_state(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestSaveLoadRoundTrip:
    def test_empty_maps_round_trip(self, tmp_path):
        path = tmp_path / "empty.json"

        provider.save_provider_state(path)
        assert path.exists()

        restored = provider.load_provider_state(path)

        assert restored == (0, 0)
        assert provider._provider_unavailable_until == {}
        assert provider._usage_reset_at == {}

    def test_written_document_is_versioned_with_both_maps(self, tmp_path):
        path = tmp_path / "state.json"
        provider._provider_unavailable_until["brand"] = time.time() + 60
        provider._usage_reset_at["key@domain"] = time.time() + 600

        provider.save_provider_state(path)

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["version"] == provider._PROVIDER_STATE_VERSION
        assert set(raw["provider_unavailable_until"]) == {"brand"}
        assert set(raw["usage_reset_at"]) == {"key@domain"}

    def test_both_maps_round_trip_with_absolute_expiries(self, tmp_path):
        path = tmp_path / "state.json"
        cooldown_expiry = time.time() + 123.5
        quarantine_expiry = time.time() + 987654.25
        provider._provider_unavailable_until["opencode-go"] = cooldown_expiry
        provider._usage_reset_at["OPENCODE_2_API_KEY@opencode.ai"] = (
            quarantine_expiry
        )

        provider.save_provider_state(path)
        # Wipe the in-memory maps to simulate a process restart.
        provider._provider_unavailable_until.clear()
        provider._usage_reset_at.clear()

        restored = provider.load_provider_state(path)

        assert restored == (1, 1)
        # Absolute epoch expiries round-trip unchanged (neither extended nor
        # reset by the restart boundary).
        assert provider._provider_unavailable_until == {
            "opencode-go": cooldown_expiry
        }
        assert provider._usage_reset_at == {
            "OPENCODE_2_API_KEY@opencode.ai": quarantine_expiry
        }

    def test_load_replaces_rather_than_merges_in_memory_state(self, tmp_path):
        path = tmp_path / "state.json"
        provider._provider_unavailable_until["from_file"] = time.time() + 60
        provider.save_provider_state(path)
        provider._provider_unavailable_until.clear()

        # A stale in-memory entry must not survive a reload.
        provider._provider_unavailable_until["stale"] = time.time() + 60
        provider.load_provider_state(path)

        assert set(provider._provider_unavailable_until) == {"from_file"}

    def test_round_trip_uses_env_override_path_by_default(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "via-env.json"
        monkeypatch.setenv(provider._PROVIDER_STATE_FILE_ENV, str(path))
        provider._usage_reset_at["acct"] = time.time() + 60

        provider.save_provider_state()
        assert path.exists()

        provider._usage_reset_at.clear()
        assert provider.load_provider_state() == (0, 1)
        assert set(provider._usage_reset_at) == {"acct"}


class TestExpiryHandling:
    def test_expired_entries_dropped_at_load(self, tmp_path):
        path = tmp_path / "state.json"
        now = time.time()
        _write_state(
            path,
            {
                "version": 1,
                "provider_unavailable_until": {
                    "expired": now - 5,
                    "live": now + 300,
                },
                "usage_reset_at": {
                    "expired@domain": now - 1,
                    "live@domain": now + 3000,
                },
            },
        )

        restored = provider.load_provider_state(path)

        assert restored == (1, 1)
        assert provider._provider_unavailable_until == {"live": now + 300}
        assert provider._usage_reset_at == {"live@domain": now + 3000}

    def test_save_excludes_expired_entries(self, tmp_path):
        path = tmp_path / "state.json"
        now = time.time()
        provider._provider_unavailable_until["expired"] = now - 10
        provider._provider_unavailable_until["live"] = now + 10
        provider._usage_reset_at["expired@domain"] = now - 10

        provider.save_provider_state(path)

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert set(raw["provider_unavailable_until"]) == {"live"}
        assert raw["usage_reset_at"] == {}

    def test_entry_expiring_exactly_now_is_dropped(self, tmp_path):
        path = tmp_path / "state.json"
        _write_state(
            path,
            {
                "version": 1,
                "provider_unavailable_until": {},
                "usage_reset_at": {},
            },
        )
        provider._usage_reset_at["boundary"] = time.time()
        provider.save_provider_state(path)

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["usage_reset_at"] == {}


class TestCorruptionTolerance:
    def test_missing_file_is_safe(self, tmp_path, caplog):
        path = tmp_path / "does-not-exist.json"

        with caplog.at_level("WARNING"):
            restored = provider.load_provider_state(path)

        assert restored == (0, 0)
        assert provider._provider_unavailable_until == {}
        assert provider._usage_reset_at == {}
        assert any(
            "no state file" in record.message for record in caplog.records
        )

    def test_corrupt_json_is_safe(self, tmp_path, caplog):
        path = tmp_path / "corrupt.json"
        path.write_text("{not valid json", encoding="utf-8")

        with caplog.at_level("WARNING"):
            restored = provider.load_provider_state(path)

        assert restored == (0, 0)
        assert provider._provider_unavailable_until == {}
        assert provider._usage_reset_at == {}
        assert any(
            "unreadable state file" in record.message
            for record in caplog.records
        )

    def test_non_dict_top_level_payload_is_safe(self, tmp_path, caplog):
        path = tmp_path / "list.json"
        _write_state(path, [1, 2, 3])

        with caplog.at_level("WARNING"):
            restored = provider.load_provider_state(path)

        assert restored == (0, 0)
        assert provider._provider_unavailable_until == {}
        assert provider._usage_reset_at == {}
        assert any(
            "malformed state file" in record.message
            for record in caplog.records
        )

    def test_non_dict_map_payload_is_safe(self, tmp_path):
        path = tmp_path / "badmap.json"
        _write_state(
            path,
            {
                "version": 1,
                "provider_unavailable_until": "not-a-dict",
                "usage_reset_at": ["also", "not", "a", "dict"],
            },
        )

        assert provider.load_provider_state(path) == (0, 0)
        assert provider._provider_unavailable_until == {}
        assert provider._usage_reset_at == {}

    def test_malformed_entries_are_skipped_valid_ones_survive(self, tmp_path):
        path = tmp_path / "mixed.json"
        now = time.time()
        _write_state(
            path,
            {
                "version": 1,
                "provider_unavailable_until": {
                    "good": now + 100,
                    "bad_string": "soon",
                    "bad_bool": True,
                    "bad_null": None,
                    "bad_list": [1, 2],
                },
                "usage_reset_at": {
                    "good@domain": now + 200,
                    "bad_object": {"expiry": now + 200},
                },
            },
        )

        restored = provider.load_provider_state(path)

        assert restored == (1, 1)
        assert provider._provider_unavailable_until == {"good": now + 100}
        assert provider._usage_reset_at == {"good@domain": now + 200}

    def test_corrupt_file_clears_any_existing_in_memory_state(self, tmp_path):
        provider._provider_unavailable_until["leftover"] = time.time() + 60
        provider._usage_reset_at["leftover@domain"] = time.time() + 60
        path = tmp_path / "corrupt.json"
        path.write_text("<<<", encoding="utf-8")

        provider.load_provider_state(path)

        assert provider._provider_unavailable_until == {}
        assert provider._usage_reset_at == {}


class TestStateFilePath:
    def test_default_path_is_beside_proxy_mode(self, monkeypatch):
        monkeypatch.delenv(provider._PROVIDER_STATE_FILE_ENV, raising=False)

        path = provider.default_provider_state_file()

        assert path.name == "provider-state.json"
        # Parent is the outer ``proxy`` project directory (where ``.mode``
        # and ``grandfathering-state.json`` live).
        assert path.parent.name == "proxy"
        assert path.parent == Path(provider.__file__).parent.parent

    def test_env_override_wins(self, monkeypatch, tmp_path):
        override = tmp_path / "custom" / "state.json"
        monkeypatch.setenv(provider._PROVIDER_STATE_FILE_ENV, str(override))

        assert provider.default_provider_state_file() == override


class TestAtomicWrite:
    def test_successful_save_leaves_no_temp_files(self, tmp_path):
        path = tmp_path / "state.json"
        provider._provider_unavailable_until["x"] = time.time() + 60

        provider.save_provider_state(path)

        assert path.exists()
        assert list(tmp_path.glob("state.json.*.tmp")) == []
        # A complete, parseable document — never a partial file.
        json.loads(path.read_text(encoding="utf-8"))

    def test_failed_replace_cleans_up_temp_and_raises(self, tmp_path, monkeypatch):
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps({"version": 1, "marker": "original"}), encoding="utf-8"
        )

        def _boom(_src, _dst):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(os, "replace", _boom)

        with pytest.raises(OSError, match="simulated replace failure"):
            provider.save_provider_state(path)

        # The original file is untouched and no partial temp file is left.
        assert json.loads(path.read_text(encoding="utf-8"))["marker"] == "original"
        assert list(tmp_path.glob("state.json.*.tmp")) == []
