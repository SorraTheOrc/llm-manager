#!/usr/bin/env python3
"""Tests for scripts/prompt_cache_analysis.py (LP-0MUPPAC7R0053J8L).

The script is the AC1 reproducible measurement of llama-server prompt-cache
behaviour (full-prefill events, prefill tokens, evictions, checkpoint
overhead). These tests run it against synthetic llama-server logs so no
production logs are needed, and assert the observable metric contract plus
the derived sizing recommendation.
"""

import gzip
import json
import os
import subprocess

SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "scripts", "prompt_cache_analysis.py")
)

# A synthetic corpus covering every event class the parser handles.
#   task 0: 20000 tokens, fully prefilled (large full prefill)
#   task 1: 21000-token prompt, only 2000 prefilled (cache hit)
#   task 2: 500 tokens, fully prefilled (small, new session — expected)
#   task 3: 30000 tokens, forced full re-process (large lost prefix)
SYNTH = """\
[100] srv        update:  - cache state: 4 prompts, 7000.000 MiB (limits: 8192.000 MiB, 262144 tokens, 393209 est)
[100] slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 262144, n_keep = 0, task.n_tokens = 20000
[100] slot update_slots: id  0 | task 0 | prompt processing progress, n_tokens = 8192, batch.n_tokens = 8192, progress = 0.409600
[100] slot update_slots: id  0 | task 0 | prompt processing progress, n_tokens = 16384, batch.n_tokens = 8192, progress = 0.819200
[100] slot update_slots: id  0 | task 0 | prompt processing done, n_tokens = 20000, batch.n_tokens = 3616
[100] slot update_slots: id  0 | task 0 | created context checkpoint 1 of 32 (pos_min = 8000, pos_max = 8000, n_tokens = 8001, size = 62.813 MiB)
[100] prompt eval time =   12345.00 ms / 20000 tokens (    0.62 ms per token,  1620.00 tokens per second)
[100] slot update_slots: id  0 | task 1 | new prompt, n_ctx_slot = 262144, n_keep = 0, task.n_tokens = 21000
[100] slot update_slots: id  0 | task 1 | prompt processing progress, n_tokens = 20000, batch.n_tokens = 1000, progress = 0.952381
[100] slot update_slots: id  0 | task 1 | prompt processing done, n_tokens = 21000, batch.n_tokens = 1000
[100] prompt eval time =     800.00 ms / 2000 tokens (    0.40 ms per token,  2500.00 tokens per second)
[100] srv        update:  - cache size limit reached, removing oldest entry (size = 1500.000 MiB)
[100] slot update_slots: id  0 | task 2 | new prompt, n_ctx_slot = 262144, n_keep = 0, task.n_tokens = 500
[100] slot update_slots: id  0 | task 2 | prompt processing done, n_tokens = 500, batch.n_tokens = 500
[100] prompt eval time =     100.00 ms / 500 tokens (    0.20 ms per token,  5000.00 tokens per second)
[100] slot update_slots: id  0 | task 3 | new prompt, n_ctx_slot = 262144, n_keep = 0, task.n_tokens = 30000
[100] slot update_slots: id  0 | task 3 | forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
[100] slot update_slots: id  0 | task 3 | prompt processing progress, n_tokens = 24576, batch.n_tokens = 24576, progress = 0.819200
[100] slot update_slots: id  0 | task 3 | prompt processing done, n_tokens = 30000, batch.n_tokens = 5424
[100] prompt eval time =   20000.00 ms / 30000 tokens (    0.67 ms per token,  1500.00 tokens per second)
[100] srv        update:  - cache state: 2 prompts, 7900.000 MiB (limits: 8192.000 MiB, 262144 tokens, 393209 est)
"""


def _write_logs(tmp_path, text=SYNTH, name="llama-server.log"):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / name).write_text(text)
    return log_dir


def _run(log_dir, *extra):
    return subprocess.run(
        ["python3", SCRIPT, "--log-dir", str(log_dir), "--json", *extra],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _summary(log_dir, *extra):
    proc = _run(log_dir, *extra)
    assert proc.returncode == 0, f"script failed: {proc.stderr}"
    return json.loads(proc.stdout)


def test_request_and_token_totals(tmp_path):
    """AC1: requests, requested/reused/prefilled tokens are measured."""
    s = _summary(_write_logs(tmp_path))["summary"]
    assert s["requests"] == 4
    assert s["prompt_tokens_requested"] == 71500
    assert s["tokens_prefilled"] == 52500
    assert s["tokens_reused"] == 19000


def test_full_prefill_split_and_forced(tmp_path):
    """AC1: full prefills split small (new session) vs large (lost prefix)."""
    s = _summary(_write_logs(tmp_path))["summary"]
    assert s["full_prefill_requests"] == 3
    assert s["small_full_prefill_requests"] == 1
    assert s["small_full_prefill_tokens"] == 500
    assert s["large_full_prefill_requests"] == 2
    assert s["large_full_prefill_tokens"] == 50000
    assert s["forced_full_prefills"] == 1


def test_prefill_wall_time(tmp_path):
    """AC1: prompt-eval wall time is summed in seconds."""
    s = _summary(_write_logs(tmp_path))["summary"]
    assert 33.2 <= s["prefill_wall_seconds"] <= 33.3


def test_evictions_and_cache_saturation(tmp_path):
    """AC1: eviction count/size and cache-saturation samples are measured."""
    s = _summary(_write_logs(tmp_path))["summary"]
    assert s["evictions"] == 1
    assert s["eviction_mean_mib"] == 1500.0
    assert s["eviction_max_mib"] == 1500.0
    assert s["cache_limit_mib"] == 8192.0
    assert s["cache_peak_mib"] == 7900.0
    # 7000 < 0.9*8192 (7372.8) but 7900 >= 7372.8 -> one saturated sample.
    assert s["cache_saturated_samples"] == 1


def test_checkpoint_overhead(tmp_path):
    """AC3: checkpoint size/count and per-prompt overhead are measured."""
    s = _summary(_write_logs(tmp_path))["summary"]
    assert s["checkpoints_per_slot_max"] == 32
    assert s["checkpoint_size_mib"] == 62.813
    assert s["checkpoint_overhead_per_prompt_mib"] == 2010.0


def test_recommendation_is_derived_from_measurement(tmp_path):
    """AC2/AC4: recommended cache-ram derives from measured size + headroom."""
    data = _summary(
        _write_logs(tmp_path),
        "--recommend",
        "--host-total-mib", "100000",
        "--host-available-mib", "90000",
    )
    rec = data["recommendation"]
    assert rec["per_prompt_source"] == "eviction_size"
    assert rec["per_prompt_state_mib"] == 1500.0
    assert rec["target_prompts"] == 8  # floor, since observed max is 4
    # 1500 * 8 * 1.25 = 15000 MiB -> round up to 15 GiB
    assert rec["recommended_cache_ram_mib"] == 15360
    assert rec["headroom_safe"] is True
    assert "0.4" in rec["oom_fallback"]


def test_recommendation_flags_unsafe_headroom(tmp_path):
    """AC4: an oversized cache against a small host is flagged unsafe."""
    rec = _summary(
        _write_logs(tmp_path), "--recommend",
        "--host-total-mib", "16384", "--host-available-mib", "16000",
    )["recommendation"]
    assert rec["headroom_safe"] is False


def test_capacity_projection(tmp_path):
    """AC5: a larger assumed cap projects avoided evictions/prefills."""
    proj = _summary(_write_logs(tmp_path), "--assume-cache-ram-mib", "16384")["projection"]
    assert proj["holdable_prompts"] == 10
    assert proj["capacity_adequate"] is True
    assert proj["projected_evictions_avoided"] == 1
    assert proj["projected_large_full_prefills_avoided"] == 1
    assert proj["projected_prefill_tokens_saved"] == 25000


def test_reads_gzip_rotated_log(tmp_path):
    """Rotation compatibility: .gz daily logs are part of the corpus."""
    log_dir = _write_logs(tmp_path, text="")
    with gzip.open(log_dir / "llama-server.log-2026-01-01.gz", "wt") as fh:
        fh.write(SYNTH)
    s = _summary(log_dir)["summary"]
    assert s["requests"] == 4


def test_json_meta_is_self_describing(tmp_path):
    """AC1: the JSON artifact records the corpus it was measured over."""
    data = _summary(_write_logs(tmp_path))
    meta = data["meta"]
    assert meta["large_tokens_threshold"] == 10000
    assert meta["files"] == ["llama-server.log"]
    assert "generated_at" in meta


def test_missing_log_dir_exits_nonzero(tmp_path):
    proc = _run(tmp_path / "does-not-exist")
    assert proc.returncode == 1
    assert "not found" in proc.stderr


def test_human_summary_renders(tmp_path):
    proc = subprocess.run(
        ["python3", SCRIPT, "--log-dir", str(_write_logs(tmp_path)), "--recommend"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "full-prefill requests" in proc.stdout
    assert "recommended --cache-ram" in proc.stdout
