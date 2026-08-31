"""Tests for the slot-persistence analysis harness (F1, LP-0MTCMEJX2008W85X).

These tests validate parsing behaviour against synthetic fixture lines that
mirror real proxy.log / llama-server.log shapes (extracted from the live
2026-08-26 incident logs). Assertions are on observable behaviour: parsed
counts, per-file breakdowns, incident-number reproduction, and corpus
regenerability — never on private implementation details.
"""

from __future__ import annotations

import gzip as gz
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import slot_persistence_harness as h

# ---------------------------------------------------------------------------
# Fixture lines (real shapes from /var/log/llama-proxy)
# ---------------------------------------------------------------------------

SLOT_SAVE_SUCCESS = (
    "2026-08-26 16:01:33,000 - INFO - slot_save success session=herdr-17 slot=2"
)
SLOT_SAVE_FAILURE = (
    "2026-08-26 21:02:06,576 - WARNING - slot_save failed slot=0 "
    "error=PoolTimeout/PoolTimeout elapsed=9.3s timeout=9.3s "
    'busy={"active_queries": 1, "local_active_queries": 1, '
    '"active_sessions": 1, "slot_busy": true}'
)
SLOT_RESTORE_SUCCESS = (
    "2026-08-26 16:28:02,887 - INFO - slot_restore success session=herdr-17 slot=2"
)
SLOT_RESTORE_FAILURE = (
    "2026-08-26 16:30:00,000 - WARNING - slot_restore failed slot=1 "
    "error=PoolTimeout/PoolTimeout elapsed=20.0s timeout=20.0s"
)
SKIP_TOO_LARGE = (
    "2026-08-26 22:02:00,000 - INFO - routing_skip_local provider=local-qwen3-next "
    "model=Qwen3 estimated_tokens=83494 cold_threshold=38000 warm_threshold=83285 "
    "new_tokens=40 cached_ratio=1.00 reason=context_too_large \u2192 skipping local, "
    "routing to next remote provider session=herdr-1787941523-285389-9160"
)
SKIP_BYPASS = (
    "2026-08-26 22:03:00,000 - INFO - routing_skip_local provider=local-qwen3-next "
    "model=Qwen3 estimated_tokens=424128 cold_threshold=38000 warm_threshold=83285 "
    "new_tokens=424128 cached_ratio=0.00 reason=large_context_bypass \u2192 skipping "
    "local, routing to next remote provider session=herdr-1787941523-285389-9160"
)
PERSIST_SKIP_BUSY = (
    "2026-08-26 22:04:00,000 - INFO - slot persistence skipped session=sess-1 "
    "slot=0 reason=slot_busy active_sessions=1"
)
PERSIST_COOLDOWN = (
    "2026-08-26 22:05:00,000 - WARNING - slot persistence disabled slot=2 "
    "consecutive_failures=3 cooldown_remaining=123.4s"
)
LEASE_RENEWED = (
    "2026-08-29 00:00:45,402 - INFO - lease_renewed "
    "session=herdr-1787957376-1596736-30595 timeout=30s"
)
LEASE_RELEASED = (
    "2026-08-29 00:02:02,910 - WARNING - lease_released "
    "session=herdr-1787957376-1596736-30595 reason=orphan_cleanup "
    "stream_abandoned=True"
)
STATUS_STALE = (
    "2026-08-26 20:33:06,038 - INFO - status_request active_query=false "
    "available_slots=0 client_ip=192.168.0.191 client_port=56110 "
    "current_model=None latency_ms=1000 llama_server_running=false "
    "local_active_query=false local_owner_lease_remaining_seconds=None "
    "local_owner_session_id=None model_switch_in_progress=false "
    "slots_stale=true total_slots=3"
)
STATUS_FRESH = (
    "2026-08-26 20:33:06,038 - INFO - status_request active_query=false "
    "available_slots=2 client_ip=192.168.0.191 client_port=56110 "
    "current_model=None latency_ms=1000 llama_server_running=true "
    "local_active_query=false local_owner_lease_remaining_seconds=None "
    "local_owner_session_id=None model_switch_in_progress=false "
    "slots_stale=false total_slots=3"
)
LLAMA_CHECKPOINT_CREATE = (
    "[59455] slot update_slots: id  2 | task 1 | created context checkpoint "
    "1 of 32 (pos_min = 906, pos_max = 906, n_tokens = 907, size = 62.813 MiB)"
)
LLAMA_CHECKPOINT_RESTORE = (
    "[59455] slot update_slots: id  1 | task 2547 | restored context checkpoint "
    "(pos_min = 22801, pos_max = 22801, n_tokens = 22802, n_past = 22802, "
    "size = 62.813 MiB)"
)
LLAMA_SLOTS_200 = "[51873] srv  log_server_r: done request: GET /slots 127.0.0.1 200"
LLAMA_SLOTS_500 = "[51873] srv  log_server_r: done request: GET /slots 127.0.0.1 500"
LLAMA_SLOTS_400 = "[51873] srv  log_server_r: done request: GET /slots 127.0.0.1 400"
LLAMA_PREFILL = (
    "[51873] prompt eval time =  50782.80 ms /  3255 tokens "
    "(   15.60 ms per token,    64.10 tokens per second)"
)
LLAMA_PROMPT_DONE = (
    "[59455] slot update_slots: id  2 | task 1 | prompt processing done, "
    "n_tokens = 1423, batch.n_tokens = 4"
)
LLAMA_PROMPT_SAVE = "[51873] srv   prompt_save:  - saving prompt with length 26234, total state size = 335.517 MiB"
LLAMA_PROMPT_LOAD = "[51873] srv   prompt_load:  - loading prompt with length 26234, total state size = 335.517 MiB"


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    p = tmp_path / name
    opener = gz.open if name.endswith(".gz") else open
    with opener(p, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return p


def _write_cache(tmp_path: Path, filename: str, size: int) -> None:
    cache = tmp_path / "slot-cache"
    cache.mkdir(exist_ok=True)
    p = cache / filename
    p.write_bytes(b"x" * size)
    os.utime(p, (datetime(2026, 8, 26, 22, 0).timestamp(),)*2)


DAY = datetime(2026, 8, 26)
DAY_END = datetime(2026, 8, 26, 23, 59, 59)


class TestProxyRegexes:
    def test_save_success(self):
        m = h._PROXY_SLOT_SAVE_SUCCESS_RE.match(SLOT_SAVE_SUCCESS)
        assert m is not None
        assert m.group("session") == "herdr-17"
        assert int(m.group("slot")) == 2

    def test_save_failure_with_timeout_and_busy(self):
        m = h._PROXY_SLOT_SAVE_FAILURE_RE.match(SLOT_SAVE_FAILURE)
        assert m is not None
        assert m.group("error") == "PoolTimeout/PoolTimeout"
        assert int(m.group("slot")) == 0

    def test_restore_success(self):
        m = h._PROXY_SLOT_RESTORE_SUCCESS_RE.match(SLOT_RESTORE_SUCCESS)
        assert m is not None
        assert m.group("session") == "herdr-17"

    def test_restore_failure(self):
        m = h._PROXY_SLOT_RESTORE_FAILURE_RE.match(SLOT_RESTORE_FAILURE)
        assert m is not None
        assert m.group("error") == "PoolTimeout/PoolTimeout"

    def test_skip_reason_extraction(self):
        info = h._extract_skip_reason(
            SKIP_TOO_LARGE.split("routing_skip_local ", 1)[1]
        )
        assert info["reason"] == "context_too_large"
        assert info["estimated_tokens"] == 83494
        assert info["cold_threshold"] == 38000
        assert info["session"] == "herdr-1787941523-285389-9160"

    def test_persistence_skip_busy_reason(self):
        m = h._SKIP_RE.match(PERSIST_SKIP_BUSY)
        assert m is not None
        assert m.group("event") == "slot persistence skipped"
        info = h._extract_skip_reason(m.group("rest"))
        assert info["reason"] == "slot_busy"
        assert info["session"] == "sess-1"
        assert info["active_sessions"] == 1

    def test_persistence_cooldown_event(self):
        m = h._SKIP_RE.match(PERSIST_COOLDOWN)
        assert m is not None
        assert m.group("event") == "slot persistence disabled"
        info = h._extract_skip_reason(m.group("rest"))
        assert info["consecutive_failures"] == 3

    def test_lease_renewed_and_released(self):
        m = h._LEASE_RE.match(LEASE_RENEWED)
        assert m is not None
        assert m.group("kind") == "renewed"
        assert m.group("session") == "herdr-1787957376-1596736-30595"
        m2 = h._LEASE_RE.match(LEASE_RELEASED)
        assert m2 is not None
        assert m2.group("kind") == "released"

    def test_status_request_parsing(self):
        m = h._STATUS_REQUEST_RE.match(STATUS_STALE)
        assert m is not None
        fields = h._parse_status_request(m.group("rest"))
        assert fields["slots_stale"] is True
        assert fields["total_slots"] == 3

        m2 = h._STATUS_REQUEST_RE.match(STATUS_FRESH)
        fields2 = h._parse_status_request(m2.group("rest"))
        assert fields2["slots_stale"] is False
        assert fields2["llama_server_running"] is True
        assert fields2["available_slots"] == 2


class TestLlamaRegexes:
    def test_checkpoint_create(self):
        m = h._LLAMA_CHECKPOINT_RE.match(LLAMA_CHECKPOINT_CREATE)
        assert m is not None
        assert int(m.group("n_tokens")) == 907
        assert float(m.group("size")) == 62.813
        assert int(m.group("slot")) == 2

    def test_checkpoint_restore(self):
        m = h._LLAMA_CHECKPOINT_RESTORE_RE.match(LLAMA_CHECKPOINT_RESTORE)
        assert m is not None
        assert int(m.group("n_past")) == 22802
        assert int(m.group("slot")) == 1

    def test_slots_access_status(self):
        assert h._LLAMA_SLOTS_ACCESS_RE.search(LLAMA_SLOTS_200).group("status") == "200"
        assert h._LLAMA_SLOTS_ACCESS_RE.search(LLAMA_SLOTS_500).group("status") == "500"
        assert h._LLAMA_SLOTS_ACCESS_RE.search(LLAMA_SLOTS_400).group("status") == "400"

    def test_prefill(self):
        m = h._LLAMA_PREFILL_RE.match(LLAMA_PREFILL)
        assert m is not None
        assert int(m.group("tokens")) == 3255

    def test_prompt_processing_done(self):
        m = h._LLAMA_PROMPT_DONE_RE.match(LLAMA_PROMPT_DONE)
        assert m is not None
        assert int(m.group("tokens")) == 1423
        assert int(m.group("slot")) == 2

    def test_prompt_save_load(self):
        assert h._LLAMA_SAVE_RE.search(LLAMA_PROMPT_SAVE) is not None
        assert h._LLAMA_LOAD_RE.search(LLAMA_PROMPT_LOAD) is not None


class TestAnalyzeCorpus:
    def test_full_corpus_breakdown(self, tmp_path):
        """Mixed proxy+llama fixture produces a structured corpus with per-file
        breakdown and correct baseline rollups."""
        _write(tmp_path, "proxy.log", [
            SLOT_SAVE_SUCCESS,
            SLOT_SAVE_FAILURE,
            SLOT_RESTORE_SUCCESS,
            SLOT_RESTORE_FAILURE,
            SKIP_TOO_LARGE,
            SKIP_BYPASS,
            STATUS_STALE,
            STATUS_FRESH,
        ])
        _write(tmp_path, "llama-server.log", [
            LLAMA_CHECKPOINT_CREATE,
            LLAMA_CHECKPOINT_RESTORE,
            LLAMA_SLOTS_200,
            LLAMA_SLOTS_500,
            LLAMA_SLOTS_400,
            LLAMA_PREFILL,
            LLAMA_PROMPT_SAVE,
            LLAMA_PROMPT_LOAD,
        ])
        _write_cache(tmp_path, "slot_audit-test.bin", 1024)

        corpus = h.analyze(tmp_path, tmp_path / "slot-cache", DAY, DAY_END)

        b = corpus["baseline_metrics"]
        assert b["slot_save_success"] == 1
        assert b["slot_save_failure"] == 1
        assert b["slot_restore_success"] == 1
        assert b["slot_restore_failure"] == 1
        assert b["total_slot_saves"] == 1
        assert b["total_slot_restores"] == 1
        # inc details: restore rate is restores/saves
        assert b["restore_rate_pct"] == 100.0
        assert b["skip_reasons"] == {
            "context_too_large": 1,
            "large_context_bypass": 1,
        }
        assert b["slots_stale_count"] == 1
        assert b["slots_status_polls"] == 2

        # llama-side
        assert b["llama_checkpoints_created"] == 1
        assert b["llama_checkpoints_restored"] == 1
        assert b["llama_checkpoint_restore_rate_pct"] == 100.0
        assert b["llama_slots_status_counts"] == {200: 1, 400: 1, 500: 1}
        assert b["llama_slots_500_pct"] == pytest.approx(33.33)
        assert b["prefill_token_total"] == 3255
        assert b["prefill_prompt_eval_lines"] == 1
        assert b["llama_prompt_save_lines"] == 1
        assert b["llama_prompt_load_lines"] == 1
        assert b["cache_files"] == 1
        assert b["cache_total_bytes"] == 1024

        # per-file llama breakdown keyed by filename
        llama_files = corpus["llama_files_seen"]
        assert "llama-server.log" in llama_files
        assert llama_files["llama-server.log"]["created_checkpoints"] == 1
        assert llama_files["llama-server.log"]["slots_500"] == 1

        # events carry status/session fields
        saves = [e for e in corpus["slot_save_events"]]
        assert saves[0]["session"] == "herdr-17"
        assert saves[0]["status"] == "success"
        assert saves[1]["status"] == "failure"
        assert saves[1]["error"] == "PoolTimeout/PoolTimeout"

    def test_save_failure_timeout_fields(self, tmp_path):
        """Failure events parse elapsed/timeout/busy into structured fields."""
        _write(tmp_path, "proxy.log", [SLOT_SAVE_FAILURE])
        corpus = h.analyze(tmp_path, None, DAY, DAY_END)
        ev = corpus["slot_save_events"][0]
        assert ev["status"] == "failure"
        assert ev["elapsed_seconds"] == 9.3
        assert ev["timeout_seconds"] == 9.3
        assert ev["busy_info"]["slot_busy"] is True
        assert ev["busy_info"]["active_queries"] == 1

    def test_gzip_rotated_logs_are_read(self, tmp_path):
        """Gzip-compressed rotated logs contribute to the corpus."""
        _write(tmp_path, "proxy.log.2026-08-26_16.gz", [SLOT_SAVE_SUCCESS])
        _write(tmp_path, "llama-server.log-2026-08-27.gz", [LLAMA_CHECKPOINT_CREATE])
        corpus = h.analyze(tmp_path, None, DAY, DAY_END)
        assert corpus["baseline_metrics"]["total_slot_saves"] == 1
        assert corpus["baseline_metrics"]["llama_checkpoints_created"] == 1
        assert corpus["meta"]["proxy_files"] == 1
        assert corpus["meta"]["llama_files"] == 1

    def test_time_window_filter(self, tmp_path):
        """"--start/--end filters proxy events but llama events (no timestamps)
        are always included."""
        outside = SLOT_SAVE_SUCCESS.replace("16:01:33", "00:01:33")
        _write(tmp_path, "proxy.log", [outside, SLOT_SAVE_SUCCESS])
        _write(tmp_path, "llama-server.log", [LLAMA_CHECKPOINT_CREATE])
        corpus = h.analyze(tmp_path, None, datetime(2026, 8, 26, 10), datetime(2026, 8, 26, 20))
        assert corpus["baseline_metrics"]["total_slot_saves"] == 1
        assert corpus["baseline_metrics"]["llama_checkpoints_created"] == 1

    def test_meta_file_counts_match_iterators(self, tmp_path):
        """meta.proxy_files / meta.llama_files reflect exactly the files the
        iterators read (both rotation naming schemes)."""
        _write(tmp_path, "proxy.log", [SLOT_SAVE_SUCCESS])
        _write(tmp_path, "proxy.log.2026-08-26_16.gz", [SLOT_SAVE_SUCCESS])
        _write(tmp_path, "proxy.log-2026-08-27_00.gz", [SLOT_SAVE_SUCCESS])
        _write(tmp_path, "llama-server.log", [LLAMA_CHECKPOINT_CREATE])
        _write(tmp_path, "llama-server.10.log", [LLAMA_CHECKPOINT_CREATE])
        _write(tmp_path, "llama-server.log-2026-08-27.gz", [LLAMA_SLOTS_200])
        corpus = h.analyze(tmp_path, None, None, None)
        assert corpus["meta"]["proxy_files"] == len(h._iter_proxy_files(tmp_path)) == 3
        assert corpus["meta"]["llama_files"] == len(h._iter_llama_files(tmp_path)) == 3
        assert corpus["baseline_metrics"]["total_slot_saves"] == 3
        assert corpus["baseline_metrics"]["llama_checkpoints_created"] == 2

    def test_llama_file_glob_filters_day(self, tmp_path):
        """--llama-file restricts llama parsing to the day's rotated file
        (llama logs have no timestamps; day attribution is file-level)."""
        _write(tmp_path, "llama-server.log-2026-08-27.gz", [LLAMA_CHECKPOINT_CREATE])
        _write(tmp_path, "llama-server.log-2026-08-28.gz", [LLAMA_CHECKPOINT_RESTORE])
        corpus = h.analyze(tmp_path, None, None, None, llama_file_glob="*2026-08-27*")
        assert corpus["baseline_metrics"]["llama_checkpoints_created"] == 1
        assert corpus["baseline_metrics"]["llama_checkpoints_restored"] == 0
        assert corpus["meta"]["llama_files"] == 1
        # the per-file breakdown carries the day-exact stats
        assert corpus["llama_files_seen"]["llama-server.log-2026-08-27.gz"]["created_checkpoints"] == 1

    def test_schema_is_documented(self):
        """The schema documents the corpus keys in the docstring contract."""
        schema_str = json.dumps(h.SCHEMA)
        assert "slot_save_events" in h.SCHEMA
        assert "slot_restore_events" in h.SCHEMA
        assert "skip_events" in h.SCHEMA
        assert "llama_files_seen" in h.SCHEMA
        assert "baseline_metrics" in schema_str

    def test_lease_churn_in_corpus(self, tmp_path):
        """Lease renewed/released events land in the corpus with reason fields."""
        _write(tmp_path, "proxy.log", [LEASE_RENEWED, LEASE_RELEASED])
        corpus = h.analyze(tmp_path, None, None, None)
        assert corpus["baseline_metrics"]["lease_events_total"] == 2
        events = corpus["lease_events"]
        assert events[0]["event"] == "lease_renewed"
        assert events[0]["timeout_seconds"] == 30
        assert events[1]["event"] == "lease_released"
        assert events[1]["reason"] == "orphan_cleanup"
        assert events[1]["stream_abandoned"] is True

    def test_routing_check_events_collected(self, tmp_path):
        """routing_check lines (proxy-side token estimates) land in the
        corpus with per-request estimates for prefill-work rollups."""
        _write(tmp_path, "proxy.log", [
            "2026-08-26 22:00:00,000 - INFO - routing_check "
            "provider=local-qwen3-next model=Qwen3 estimated_tokens=92404 "
            "cold_threshold=38000 warm_threshold=83285 new_tokens=92404 "
            "cached_ratio=0.00 messages=165 session=sess-1",
            "2026-08-26 22:00:01,000 - INFO - routing_check "
            "provider=local-qwen3-next model=Qwen3 estimated_tokens=14041 "
            "cold_threshold=38000 warm_threshold=83285 new_tokens=14041 "
            "cached_ratio=0.00 messages=2 session=sess-2",
        ])
        corpus = h.analyze(tmp_path, None, DAY, DAY_END)
        assert corpus["baseline_metrics"]["routing_check_count"] == 2
        assert corpus["baseline_metrics"]["routing_check_tokens_total"] == 92404 + 14041
        assert corpus["routing_check_events"][0]["details"]["estimated_tokens"] == 92404

    def test_cooldown_event_type_in_corpus(self, tmp_path):
        """Circuit-breaker cooldown events get event_type=persistence_cooldown."""
        _write(tmp_path, "proxy.log", [PERSIST_COOLDOWN, PERSIST_SKIP_BUSY])
        corpus = h.analyze(tmp_path, None, None, None)
        types = {e["event_type"] for e in corpus["skip_events"]}
        assert types == {"persistence_cooldown", "persistence_skip"}
        assert corpus["baseline_metrics"]["skip_events_total"] == 2

    def test_missing_log_dir_returns_error(self, tmp_path, capsys):
        rc = h.main(["--log-dir", str(tmp_path / "nonexistent")])
        assert rc == 1
        assert "not found" in capsys.readouterr().err


class TestIncidentReproduction:
    """Reproduce the 2026-08-26 incident claims within documented tolerance.

    Incident claims (from LP-0MTAQNB7J0094X71):
      - 2,954 checkpoints saved vs 145 restored (llama-server native checkpoints)
      - 6,459 of ~69.6K /slots polls returned 500 (9.3%), plus 527 HTTP 400
      - restore rate ~5%

    The harness reads the full-day log snapshot (llama-server.log-2026-08-27.gz
    covers 2026-08-26 01:00→23:59). The incident analysis was taken mid-day
    (22:12), which explains the small count deltas; the *ratios* must match.
    """

    def test_incident_ratio_reproduction(self):
        """Out-of-process run over live logs reproduces the claimed ratios.

        Guarded: skips (rather than fails) when live logs are unavailable so
        CI without /var/log/llama-proxy does not break.
        """
        log_dir = Path("/var/log/llama-proxy")
        if not log_dir.exists() or not list(log_dir.glob("llama-server.log-*")):
            pytest.skip("live incident logs not available")

        import subprocess
        out = subprocess.run(
            [sys.executable, str(Path(h.__file__)), "--log-dir", str(log_dir),
             "--start", "2026-08-26", "--end", "2026-08-27",
             "--llama-file", "*2026-08-27*", "--compact"],
            capture_output=True, text=True,
        )
        assert out.returncode == 0, out.stderr
        corpus = json.loads(out.stdout)
        b = corpus["baseline_metrics"]

        # restore rate ~5%: incident 145/2954 = 4.9%
        assert 3.0 <= b["llama_checkpoint_restore_rate_pct"] <= 8.0, b
        # /slots 500 rate ~9.3%
        assert 5.0 <= b["llama_slots_500_pct"] <= 15.0, b
        # 400s present in the access log; JSON keys are strings post-parse
        assert b["llama_slots_status_counts"].get("400", 0) > 0
        # 42.7M prefill claim reproduces via prompt processing done (46.1M
        # full-day vs 42.7M mid-day measurement at 22:12, 2026-08-26)
        assert 3.5e7 <= b["prompt_done_tokens_total"] <= 5.5e7, b

    def test_incident_day_file_counts(self):
        """The Aug 26 rotated llama log carries the specific incident numbers."""
        log_dir = Path("/var/log/llama-proxy")
        target = log_dir / "llama-server.log-2026-08-27.gz"
        if not target.exists():
            pytest.skip("incident-day llama log not available")

        import gzip as gz
        created = restored = 0
        slots = {"200": 0, "400": 0, "500": 0}
        with gz.open(target, "rt", errors="replace") as fh:
            for line in fh:
                if h._LLAMA_CHECKPOINT_RE.match(line):
                    created += 1
                elif h._LLAMA_CHECKPOINT_RESTORE_RE.match(line):
                    restored += 1
                m = h._LLAMA_SLOTS_ACCESS_RE.search(line)
                if m:
                    slots[m.group("status")] += 1
        # 2,954 created / 145 restored; 6,459/69.6K 500s + 527 400s
        assert created >= 2500, created
        assert restored >= 100, restored
        # restore ratio ~5%
        assert 3.0 <= 100.0 * restored / created <= 8.0
        # 400 count matches the incident's 527 exactly (access-log evidence)
        assert slots["400"] == 527, slots
        # 500 ratio ~9.3%
        total = slots["200"] + slots["400"] + slots["500"]
        assert 5.0 <= 100.0 * slots["500"] / total <= 15.0
