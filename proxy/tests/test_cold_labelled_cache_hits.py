"""Cold-labelled sub-second prompt-cache evidence (LP-0MWRQVLFY003BK5L).

Pins the contract the investigation publishes, not the implementation:

- a dispatch is *cold-labelled* when routing priced ``cached_ratio=0.0`` (or the
  classifier said ``cold``), and only the sub-second / >min-token subset is the
  set under audit;
- each such dispatch is paired with the ``prefill_progress_measured`` line
  emitted just before it for the **same** session, consumed once, never reused
  and never paired across a stale gap or across sessions;
- the summary reports the llama-server reuse (``cache / total``) and the implied
  prefill rate side by side, so a physically impossible rate is visible;
- the independent llama-server log cross-check derives reuse from
  ``task.n_tokens`` and ``n_tokens = X, memory_seq_rm [X, end)`` lines and the
  bulk rate from ``prompt eval time`` lines;
- the bounded replay classifies a fresh-session request priced ``cold`` but
  measured warm, and extracts its ``prompt eval`` line.
"""

from __future__ import annotations

import importlib
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

COLD_DISPATCH = (
    "2026-11-21 03:16:34,276 - INFO - dispatch_first_byte_ms=127.8 "
    "dispatch_to_first_byte_ms=127.5 cache_state=cold cached_ratio=0.000 "
    "queue_wait_ms=0.3 session=sess-a model=Qwen3 priority=low "
    "estimated_tokens=7185 turn=d500773f"
)
WARM_DISPATCH = (
    "2026-11-21 03:16:31,514 - INFO - dispatch_first_byte_ms=721.7 "
    "dispatch_to_first_byte_ms=721.3 cache_state=warm cached_ratio=0.986 "
    "queue_wait_ms=0.4 session=sess-a model=Qwen3 priority=low "
    "estimated_tokens=7185 turn=301d8908"
)
MEASURED = (
    "2026-11-21 03:16:34,275 - INFO - prefill_progress_measured session=sess-a "
    "model=Qwen3 total=10011 cache=10007 processed=10007 time_ms=1.0 "
    "measured_cached_ratio=0.9996 measured_new_tokens=4 priced_new_tokens=100"
)


@pytest.fixture(name="analysis")
def _analysis():
    return importlib.import_module("analyse_cold_labelled_cache_hits")


@pytest.fixture(name="replay")
def _replay():
    return importlib.import_module("replay_cold_labelled_cache_hit")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_dispatch_extracts_every_field(analysis):
    dispatch = analysis.parse_dispatch(COLD_DISPATCH)
    assert dispatch is not None
    assert dispatch.cache_state == "cold"
    assert dispatch.cached_ratio == 0.0
    assert dispatch.session == "sess-a"
    assert dispatch.estimated_tokens == 7185
    assert dispatch.to_first_byte_ms == pytest.approx(127.5)
    assert dispatch.turn == "d500773f"


def test_parse_dispatch_ignores_non_dispatch_lines(analysis):
    assert analysis.parse_dispatch(MEASURED) is None
    assert analysis.parse_dispatch("2026-11-21 03:16:34,276 - INFO - unrelated") is None


def test_parse_measurement_extracts_reuse(analysis):
    measurement = analysis.parse_measurement(MEASURED)
    assert measurement is not None
    assert measurement.total == 10011
    assert measurement.cache == 10007
    assert measurement.measured_cached_ratio == pytest.approx(0.9996)
    assert measurement.measured_new_tokens == 4
    assert analysis.parse_measurement(COLD_DISPATCH) is None


def test_cold_labelled_accepts_ratio_zero_or_cold_state(analysis):
    cold = analysis.parse_dispatch(COLD_DISPATCH)
    assert cold is not None and cold.cold_labelled
    warm = analysis.parse_dispatch(WARM_DISPATCH)
    assert warm is not None and not warm.cold_labelled


# ---------------------------------------------------------------------------
# Joining
# ---------------------------------------------------------------------------


def _measured(session="sess-a", ts=datetime(2026, 11, 21, 3, 16, 34, 275000), **kw):
    from importlib import import_module

    analysis = import_module("analyse_cold_labelled_cache_hits")
    return analysis.Measurement(
        ts=ts,
        session=session,
        total=kw.get("total", 10011),
        cache=kw.get("cache", 10007),
        measured_cached_ratio=kw.get("mcr", 0.9996),
        measured_new_tokens=kw.get("mnt", 4),
        priced_new_tokens=kw.get("pnt", 100),
    )


def _dispatch(session="sess-a", ts=datetime(2026, 11, 21, 3, 16, 34, 276000), **kw):
    from importlib import import_module

    analysis = import_module("analyse_cold_labelled_cache_hits")
    return analysis.Dispatch(
        ts=ts,
        first_byte_ms=kw.get("fb", 127.8),
        to_first_byte_ms=kw.get("tfb", 127.5),
        cache_state=kw.get("state", "cold"),
        cached_ratio=kw.get("ratio", 0.0),
        session=session,
        model="Qwen3",
        priority="low",
        estimated_tokens=kw.get("est", 7185),
        turn=kw.get("turn", "d500773f"),
    )


def test_join_pairs_measurement_to_following_dispatch(analysis):
    paired = analysis.join_measurements([_measured(), _dispatch()])
    assert len(paired) == 1
    assert paired[0].measurement is not None
    assert paired[0].gap_seconds == pytest.approx(0.001)


def test_join_consumes_measurement_once(analysis):
    """A single measurement cannot be attached to two dispatches."""
    base = datetime(2026, 11, 21, 3, 16, 34, 275000)
    paired = analysis.join_measurements(
        [
            _measured(ts=base),
            _dispatch(ts=base + timedelta(milliseconds=1)),
            _dispatch(ts=base + timedelta(milliseconds=2)),
        ]
    )
    assert [pair.measurement is not None for pair in paired] == [True, False]


def test_join_never_pairs_across_sessions(analysis):
    paired = analysis.join_measurements([_measured(session="sess-a"), _dispatch(session="sess-b")])
    assert paired[0].measurement is None


def test_join_rejects_stale_measurement(analysis):
    base = datetime(2026, 11, 21, 3, 0, 0)
    paired = analysis.join_measurements([_measured(ts=base), _dispatch(ts=base + timedelta(seconds=30))])
    assert paired[0].measurement is None
    assert paired[0].gap_seconds == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_measures_reuse_for_cold_subsecond_dispatches(analysis):
    base = datetime(2026, 11, 21, 3, 16, 34, 275000)
    paired = analysis.join_measurements(
        [
            _measured(ts=base),
            _dispatch(ts=base + timedelta(milliseconds=1), est=7185, tfb=127.5),
        ]
    )
    summary = analysis.summarise(paired)
    assert summary["dispatches"] == 1
    assert summary["cold_labelled_sub_second"] == 1
    assert summary["with_measurement"] == 1
    assert summary["reused_tokens"] == 10007
    assert summary["measured_total_tokens"] == 10011
    assert summary["measured_reuse_pct"] == pytest.approx(99.96)
    # 7185 tokens in 127.5 ms is ~56k tok/s — far above any inline prefill rate.
    assert summary["implied_prefill_tok_s"]["median"] > 50_000


def test_summary_excludes_warm_slow_and_small_dispatches(analysis):
    base = datetime(2026, 11, 21, 3, 16, 34, 275000)
    events = [
        _measured(ts=base),
        _dispatch(ts=base + timedelta(milliseconds=1), est=7185, tfb=127.5),
        # warm-labelled: excluded
        _dispatch(ts=base + timedelta(milliseconds=2), state="warm", ratio=0.98),
        # cold but too slow: excluded
        _dispatch(ts=base + timedelta(milliseconds=3), tfb=5000.0),
        # cold but too small: excluded
        _dispatch(ts=base + timedelta(milliseconds=4), est=100),
    ]
    summary = analysis.summarise(analysis.join_measurements(events))
    assert summary["cold_labelled_sub_second"] == 1


def test_summary_window_bounds_are_chronological(analysis):
    """Window bounds must be min/max timestamps, not input order."""
    later = datetime(2026, 11, 21, 23, 0, 0)
    earlier = datetime(2026, 11, 21, 1, 0, 0)
    paired = [
        analysis.PairedDispatch(_dispatch(ts=later), None, None),
        analysis.PairedDispatch(_dispatch(ts=earlier), None, None),
    ]
    summary = analysis.summarise(paired)
    assert summary["window_first"].startswith("2026-11-21 01:00:00")
    assert summary["window_last"].startswith("2026-11-21 23:00:00")


def test_cli_joins_across_files_in_timestamp_order(analysis, tmp_path, capsys):
    """Rotation file names are not chronological; the CLI must sort by time."""
    # The dispatch's file ('a') sorts before the measurement's file ('z'), so
    # without the timestamp sort the dispatch is read first (no measurement)
    # and the pair is missed.
    (tmp_path / "proxy.log-z").write_text(MEASURED.replace("03:16:34,275", "03:16:40,000"))
    (tmp_path / "proxy.log-a").write_text(COLD_DISPATCH.replace("03:16:34,276", "03:16:40,001"))
    exit_code = analysis.main(["--log-dir", str(tmp_path), "--date", "2026-11-21", "--json"])
    assert exit_code == 0
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["summary"]["cold_labelled_sub_second"] == 1
    assert payload["summary"]["with_measurement"] == 1


# ---------------------------------------------------------------------------
# llama-server cross-check
# ---------------------------------------------------------------------------


def test_parse_prompt_eval_parses_line(analysis):
    entry = analysis.parse_prompt_eval(
        "[44409] prompt eval time =    81.46 ms /     4 tokens (   20.36 ms per token, "
        "49.10 tokens per second)"
    )
    assert entry is not None
    assert entry["tokens"] == 4
    assert entry["ms"] == pytest.approx(81.46)
    assert entry["tok_s"] == pytest.approx(49.1, abs=0.1)
    assert analysis.parse_prompt_eval("not an eval line") is None


def test_parse_llama_reuse_from_slot_log(analysis):
    lines = [
        "[44409] slot update_slots: id 0 | task 1 | new prompt, n_ctx_slot = 131072, "
        "n_keep = 0, task.n_tokens = 10011",
        "[44409] slot update_slots: id 0 | task 1 | n_tokens = 10007, memory_seq_rm [10007, end)",
        "[44409] slot update_slots: id 0 | task 2 | new prompt, n_ctx_slot = 131072, "
        "n_keep = 0, task.n_tokens = 10011",
        "[44409] slot update_slots: id 0 | task 2 | n_tokens = 10007, memory_seq_rm [10007, end)",
        "[44409] prompt eval time =    81.46 ms /     4 tokens (   20.36 ms per token, "
        "49.10 tokens per second)",
        "[44409] prompt eval time =  6174.08 ms /  4237 tokens (    1.46 ms per token, "
        "686.26 tokens per second)",
    ]
    result = analysis.parse_llama_reuse(lines)
    assert result["tasks"] == 2
    assert result["tasks_with_reuse_line"] == 2
    assert result["reused_tokens"] == 20014
    assert result["prompt_tokens"] == 20022
    assert result["reuse_pct"] == pytest.approx(99.96, abs=0.01)
    assert result["dominant_prompt_tokens"] == 10011
    assert result["dominant_prompt_count"] == 2
    assert result["prompt_eval_lines"] == 2
    assert result["prompt_eval_tokens"] == 4241
    assert result["prompt_eval_aggregate_tok_s"] == pytest.approx(678.2, abs=1.0)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def test_build_prompt_is_deterministic_and_scales(replay):
    small = replay.build_prompt(1000)
    larger = replay.build_prompt(4000)
    assert small == replay.build_prompt(1000)
    assert len(larger) > len(small)


def test_extract_replay_lines_matches_by_session(replay):
    lines = [MEASURED, COLD_DISPATCH]
    captured = replay._extract_replay_lines(lines, ["sess-a"])
    assert captured["sess-a"]["dispatch"]["cache_state"] == "cold"
    assert captured["sess-a"]["measurement"]["measured_cached_ratio"] == pytest.approx(0.9996)


def test_extract_prompt_eval_parses_lines(replay):
    evals = replay._extract_prompt_eval(
        ["[44409] prompt eval time =    81.46 ms /     4 tokens (   20.36 ms per token, "
         "49.10 tokens per second)"]
    )
    assert len(evals) == 1
    assert evals[0]["tokens"] == 4
    assert evals[0]["tok_s"] == pytest.approx(49.1, abs=0.1)


def test_run_replay_demonstrates_warm_prefix_priced_cold(replay, tmp_path, monkeypatch):
    """A fresh second session is priced cold but served from the shared cache."""

    proxy_log = tmp_path / "proxy.log"
    llama_log = tmp_path / "llama-server.log"
    proxy_log.write_text("")
    llama_log.write_text("")

    def fake_send_once(proxy_url, model, prompt, session, max_tokens=8, timeout=180.0):
        first = "replay-cold-a" in session
        with proxy_log.open("a", encoding="utf-8") as handle:
            if first:
                handle.write(
                    f"2026-11-21 03:16:34,275 - INFO - prefill_progress_measured "
                    f"session={session} model=Qwen3 total=4237 cache=0 processed=0 "
                    f"time_ms=24.0 measured_cached_ratio=0.0000 measured_new_tokens=4237 "
                    f"priced_new_tokens=3626\n"
                )
                handle.write(
                    f"2026-11-21 03:16:34,276 - INFO - dispatch_first_byte_ms=6360.8 "
                    f"dispatch_to_first_byte_ms=6360.6 cache_state=cold cached_ratio=0.000 "
                    f"queue_wait_ms=0.3 session={session} model=Qwen3 priority=medium "
                    f"estimated_tokens=3626 turn=aaaaaaaa\n"
                )
            else:
                handle.write(
                    f"2026-11-21 03:16:36,209 - INFO - prefill_progress_measured "
                    f"session={session} model=Qwen3 total=4237 cache=4233 processed=4233 "
                    f"time_ms=1.0 measured_cached_ratio=0.9991 measured_new_tokens=4 "
                    f"priced_new_tokens=3626\n"
                )
                handle.write(
                    f"2026-11-21 03:16:36,210 - INFO - dispatch_first_byte_ms=154.9 "
                    f"dispatch_to_first_byte_ms=154.7 cache_state=cold cached_ratio=0.000 "
                    f"queue_wait_ms=0.1 session={session} model=Qwen3 priority=medium "
                    f"estimated_tokens=3626 turn=bbbbbbbb\n"
                )
        with llama_log.open("a", encoding="utf-8") as handle:
            tokens = 4237 if first else 4
            handle.write(
                f"[44409] prompt eval time = 6174.08 ms / {tokens} tokens "
                f"( 1.46 ms per token, 686.26 tokens per second)\n"
            )
        return {
            "session": session,
            "client_first_byte_s": 6.4 if first else 0.18,
            "client_total_s": 6.7 if first else 0.48,
        }

    monkeypatch.setattr(replay, "send_once", fake_send_once)
    monkeypatch.setattr(replay.time, "sleep", lambda *_: None)

    result = replay.run_replay(
        proxy_url="http://test", model="private", log_dir=tmp_path, retries=0
    )

    assert result["verdict"]["second_session_priced_cold"] is True
    assert result["verdict"]["second_session_served_warm"] is True
    assert result["verdict"]["demonstrated"] is True
    assert result["verdict"]["second_session_measured_cached_ratio"] == pytest.approx(0.9991)
    assert result["verdict"]["first_byte_speedup"] == pytest.approx(6.4 / 0.18, abs=0.1)
    assert len(result["llama_prompt_eval_lines"]) == 2
