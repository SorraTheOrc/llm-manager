"""Tests for the hard local-routing cap evaluation (LP-0MTAQNAIH001RN1S).

Validates ``proxy/benchmarks/evaluate_hard_routing_cap.py`` against a
small synthetic llama-server log: prefill classification under current
policy and under a candidate hard cap, plus the cross-slot decode-collapse
correlation.
"""

import textwrap


def _import_module():
    import sys
    from pathlib import Path
    try:
        import proxy.benchmarks.evaluate_hard_routing_cap as m
        return m
    except ImportError:
        pass
    try:
        from benchmarks import evaluate_hard_routing_cap as m
        return m
    except ImportError:
        pass
    this_dir = Path(__file__).resolve().parent
    root_dir = this_dir.parent.parent  # repo root
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    try:
        import proxy.benchmarks.evaluate_hard_routing_cap as m
        return m
    except ImportError:
        return None


MOD = _import_module()


def _synthetic_log(tmp_path, lines: str):
    path = tmp_path / "llama-server.log"
    path.write_text(textwrap.dedent(lines))
    return str(path)


SAMPLE_LOG = """
[1] slot update_slots: id  0 | task 1 | new prompt, n_ctx_slot = 87552, n_keep = 0, task.n_tokens = 5000
[2] slot update_slots: id  0 | task 1 | prompt processing done, n_tokens = 5000, batch.n_tokens = 6
[3] slot update_slots: id  1 | task 2 | new prompt, n_ctx_slot = 87552, n_keep = 0, task.n_tokens = 75000
[4] slot update_slots: id  1 | task 2 | prompt processing done, n_tokens = 75000, batch.n_tokens = 8
[5] slot update_slots: id  0 | task 3 | prompt processing done, n_tokens = 60000, batch.n_tokens = 7
[6] slot update_slots: id  0 | task 3 | new prompt, n_ctx_slot = 87552, n_keep = 0, task.n_tokens = 90000
[7] slot update_slots: id  2 | task 4 | prompt processing done, n_tokens = 90000, batch.n_tokens = 9
[8]        eval time =   1000.00 ms /    10 tokens (  100.00 ms per token,    10.00 tokens per second)
[9]        eval time =  50000.00 ms /    10 tokens ( 5000.00 ms per token,     0.20 tokens per second)
[10]       eval time =   1000.00 ms /    10 tokens (  100.00 ms per token,    2.00 tokens per second)
"""


def test_parse_log_files(tmp_path):
    if MOD is None:
        import pytest
        pytest.skip("evaluate_hard_routing_cap module not importable")
    path = _synthetic_log(tmp_path, SAMPLE_LOG)
    events = MOD.parse_log_files([path])
    prefills = [e for e in events if e["type"] == "prefill_done"]
    evals = [e for e in events if e["type"] == "eval"]
    assert len(prefills) == 4
    assert len(evals) == 3
    sizes = sorted(p["n_tokens"] for p in prefills)
    assert sizes == [5000, 60000, 75000, 90000]
    # slot 0 is the current slot of the prefill lines; evals inherit the
    # most recent slot line (slot 2 from the last prefill due to interleave)
    assert prefills[0]["slot"] == 0
    assert evals[0]["tps"] == 10.0
    assert evals[1]["tps"] == 0.2  # collapsed decode


def test_summarize_prefills_current_policy_fast(tmp_path):
    if MOD is None:
        import pytest
        pytest.skip("evaluate_hard_routing_cap module not importable")
    path = _synthetic_log(tmp_path, SAMPLE_LOG)
    events = MOD.parse_log_files([path])
    cfg = MOD.MODES["fast"]
    s = MOD.summarize_prefills(events, cfg)
    assert s["n"] == 4
    assert s["total_tokens"] == 5000 + 60000 + 75000 + 90000
    # 5K under cold (38000); 60K and 75K in-band; 90K above warm (83285)
    assert s["under_cold"]["count"] == 1
    assert s["in_band"]["count"] == 2
    assert s["context_too_large"]["count"] == 1


def test_summarize_prefills_with_hard_cap(tmp_path):
    if MOD is None:
        import pytest
        pytest.skip("evaluate_hard_routing_cap module not importable")
    path = _synthetic_log(tmp_path, SAMPLE_LOG)
    events = MOD.parse_log_files([path])
    cfg = MOD.MODES["fast"]
    # hard cap 70K: 60K stays in_band, 75K and 90K become above_cap
    s = MOD.summarize_prefills(events, cfg, hard_cap=70000)
    assert s["above_cap"]["count"] == 2
    assert s["in_band"]["count"] == 1
    assert s["under_cold"]["count"] == 1
    assert "context_too_large" not in s  # no event between 70K and 83.3K

    # hard cap 55K: 60K also gated
    s2 = MOD.summarize_prefills(events, cfg, hard_cap=55000)
    assert s2["above_cap"]["count"] == 3


def test_decode_collapse_correlation(tmp_path):
    if MOD is None:
        import pytest
        pytest.skip("evaluate_hard_routing_cap module not importable")
    path = _synthetic_log(tmp_path, SAMPLE_LOG)
    events = MOD.parse_log_files([path])
    d = MOD.decode_collapse_analysis(events, window=100)
    assert d["n"] == 3
    assert d["collapsed_lt5"] == 2  # 0.2 and 2.0 t/s evals
    # at least one collapse has a >=50K cross-slot prefill nearby
    # (evals inherit slot 2; the 90K prefill is slot 2 = same slot, but the
    # 75K prefill of slot 1 is a cross-slot consumer within the window)
    assert d["collapsed_with_50k_cross_prefill"] >= 1


def test_warm_threshold_resolution():
    if MOD is None:
        import pytest
        pytest.skip("evaluate_hard_routing_cap module not importable")
    assert MOD.warm_threshold(MOD.MODES["fast"]) == 83285
    assert MOD.warm_threshold(MOD.MODES["cheap"]) == 100000