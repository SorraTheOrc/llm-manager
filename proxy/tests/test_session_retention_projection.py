"""Tests for F5 session-retention projection (LP-0MSC95W7M009VMEM)."""

import json

import pytest


def _import_module():
    import sys
    from pathlib import Path
    try:
        import proxy.benchmarks.project_session_retention as p
        return p
    except ImportError:
        pass
    try:
        from benchmarks import project_session_retention as p
        return p
    except ImportError:
        pass
    this_dir = Path(__file__).resolve().parent
    root_dir = this_dir.parent.parent  # repo root
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    try:
        import proxy.benchmarks.project_session_retention as p
        return p
    except ImportError:
        return None


SAMPLE_ROWS = [
    {
        "session_id": "s-ctx-small",
        "max_context_size": "30000",
        "local_requests": "10",
        "remote_requests": "0",
        "fallback_reason": "",
        "dispatch_denied": "0",
    },
    {
        "session_id": "s-ctx-huge",
        "max_context_size": "150000",
        "local_requests": "0",
        "remote_requests": "10",
        "fallback_reason": "warm_cache_bypass",
        "dispatch_denied": "0",
    },
    {
        "session_id": "s-conc-limited",
        "max_context_size": "20000",
        "local_requests": "0",
        "remote_requests": "8",
        "fallback_reason": "local_concurrency_limit",
        "dispatch_denied": "0",
    },
    {
        "session_id": "s-above-cap",
        "max_context_size": "50000",
        "local_requests": "3",
        "remote_requests": "2",
        "fallback_reason": "local_lease_active",
        "dispatch_denied": "1",
    },
]


@pytest.fixture
def mod():
    m = _import_module()
    if m is None:
        pytest.skip("project_session_retention not importable")
    return m


def test_routing_clamp_formula(mod):
    # 262144 / 4 slots - 4096 headroom = 61440
    assert mod.routing_clamp(262144, 4) == 61440
    # 131072 / 3 - 4096 = 43690 - 4096 = 39594
    assert mod.routing_clamp(131072, 3) == 39594
    # 262144 / 2 - 4096 = 131072 - 4096 = 126976
    assert mod.routing_clamp(262144, 2) == 126976


def test_classify_full_local(mod):
    p = mod.classify_session(SAMPLE_ROWS[0], clamp=39594, slots=3)
    assert p["projection"] == "full_local"
    assert p["ctx_ok"] is True
    assert p["concurrency_blocked"] is False


def test_classify_context_bypass(mod):
    p = mod.classify_session(SAMPLE_ROWS[1], clamp=39594, slots=3)
    assert p["projection"] == "context_bypass"
    assert p["ctx_ok"] is False


def test_classify_concurrency_blocked(mod):
    p = mod.classify_session(SAMPLE_ROWS[2], clamp=39594, slots=3)
    assert p["projection"] == "concurrency_blocked"
    assert p["ctx_ok"] is True
    assert p["concurrency_blocked"] is True


def test_slot_availability_gates_expected_local(mod):
    # Context-eligible session, 3 slots, peak 21 concurrent -> expected 3/21
    p = mod.classify_session(SAMPLE_ROWS[0], clamp=39594, slots=3, peak_concurrency=21)
    assert p["slot_avail"] == pytest.approx(3 / 21)
    assert p["expected_local"] == pytest.approx(3 / 21)
    # With enough slots (>= peak), availability is 1.0
    p2 = mod.classify_session(SAMPLE_ROWS[0], clamp=39594, slots=21, peak_concurrency=21)
    assert p2["slot_avail"] == 1.0
    assert p2["expected_local"] == 1.0


def test_concurrency_blocked_has_zero_expected(mod):
    p = mod.classify_session(SAMPLE_ROWS[2], clamp=39594, slots=3, peak_concurrency=21)
    assert p["expected_local"] == 0.0


def test_main_writes_table(mod, tmp_path, capsys):
    csv_path = tmp_path / "sessions.csv"
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(SAMPLE_ROWS[0].keys()))
        w.writeheader()
        w.writerows(SAMPLE_ROWS)

    rc = mod.main(["--csv", str(csv_path)])
    assert rc == 0
    out = capsys.readouterr().out
    # All four configs printed
    for name in ("baseline 3x43.7K", "4x65.5K", "3x87.4K", "2x131K"):
        assert name in out
    # s-ctx-huge (150K) never full-local; s-ctx-small always is
    assert "full_local:" in out


def test_main_json_output(mod, tmp_path, capsys):
    csv_path = tmp_path / "sessions.csv"
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(SAMPLE_ROWS[0].keys()))
        w.writeheader()
        w.writerows(SAMPLE_ROWS)

    rc = mod.main(["--csv", str(csv_path), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    # JSON block: find the line containing '"configs":' and back up to the '{'
    lines = out.split("\n")
    cfg_idx = next(i for i, l in enumerate(lines) if '"configs":' in l)
    start = cfg_idx
    while start > 0 and lines[start - 1].strip() != "{":
        start -= 1
    if lines[start].strip() != "{":
        start = cfg_idx - 1
    data = json.loads("\n".join(lines[start:]))
    assert len(data["configs"]) == 4
    # 2x131K clamp 126976 admits s-ctx-small and s-above-cap (50K)
    two = [c for c in data["configs"] if c["config"] == "2x131K"][0]
    assert two["full_local"] >= 1
