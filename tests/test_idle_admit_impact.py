#!/usr/bin/env python3
"""Tests for scripts/idle_admit_impact.py (LP-0MVGHXD59008UOUD).

The script is the reproducible before/after measurement for the idle-aware
local admission override (feature: LP-0MVGD9Z8Z0055Y5J). These tests drive the
CLI against synthetic proxy logs — no production log and no running proxy is
needed — and assert the observable metric contract the acceptance criteria are
written against:

  * AC1 — window aggregates (``local_idle_admit``, local serve share,
    ``large_context_bypass``, remote fallbacks, denials, prefill slot-seconds)
    are counted per window, and chain-hold re-emissions are not double-counted;
  * AC2 — idle-admitted requests are paired to their local dispatch by session
    inside the correlation window, bucketed by prompt size, and compared with
    non-idle dispatches in the same bucket with sample counts;
  * AC3 — denials are compared per hour and mid-stream timeouts are split into
    *attributed* (session-matched), *coincident* (no session id, inside the
    window) and *unrelated* rather than silently blamed on the override;
  * AC4 — the interactive priority tier's first-byte p50/p90 is compared
    before/after with a noise band;
  * AC5 — the recommendation is deterministic: ``measure`` without samples,
    ``enable_by_default`` when the fallback rate falls materially without harm,
    and ``tune`` with a concrete follow-up otherwise.
"""

import json
import os
import subprocess
from datetime import datetime, timedelta

SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "scripts", "idle_admit_impact.py")
)

BEFORE_START = "2026-10-19 00:00"
BEFORE_END = "2026-10-20 00:00"
AFTER_START = "2026-10-20 00:00"
AFTER_END = "2026-10-21 00:00"


# ---------------------------------------------------------------------------
# Synthetic log helpers
# ---------------------------------------------------------------------------


def _stamp(base: str, seconds: int) -> str:
    """Log timestamp *seconds* after *base* (``YYYY-MM-DD HH:MM:SS,mmm``)."""
    moment = datetime.strptime(base, "%Y-%m-%d %H:%M") + timedelta(seconds=seconds)
    return moment.strftime("%Y-%m-%d %H:%M:%S,000")


def _skip_line(ts, session, reason="large_context_bypass", tokens=90000, cycle=0):
    return (
        f"{ts} - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 "
        f"estimated_tokens={tokens} cold_threshold=38000 warm_threshold=120000 "
        f"new_tokens={tokens} cached_ratio=0.00 cache_state=cold reason={reason} "
        f"→ skipping local, routing to next remote provider session={session} "
        f"chain_hold_cycle={cycle}\n"
    )


def _idle_admit(ts, session, tokens=90000, idle=412.3):
    return (
        f"{ts} - INFO - routing_idle_admit provider=local model=Qwen3 "
        f"reason=economic_bypass_idle idle_seconds={idle} threshold_seconds=60.0 "
        f"estimated_tokens={tokens} new_tokens={tokens} cached_ratio=0.05 "
        f"cache_state=cold session={session}\n"
    )


def _dispatch(ts, session, total_ms=61000.0, service_ms=60900.0, qwait=100.0,
              cache_state="cold", priority=None, tokens=None):
    fields = (
        f"dispatch_first_byte_ms={total_ms} dispatch_to_first_byte_ms={service_ms} "
        f"cache_state={cache_state} cached_ratio=0.050 queue_wait_ms={qwait} "
        f"session={session} model=Qwen3"
    )
    if priority is not None:
        fields += f" priority={priority}"
    if tokens is not None:
        fields += f" estimated_tokens={tokens}"
    return f"{ts} - INFO - {fields}\n"


def _routing_check(ts, session, tokens=90000, cache_state="warm"):
    return (
        f"{ts} - INFO - routing_check provider=local-qwen3 model=Qwen3 "
        f"estimated_tokens={tokens} cold_threshold=38000 warm_threshold=120000 "
        f"new_tokens=100 cached_ratio=0.99 cache_state={cache_state} messages=10 "
        f"session={session} chain_hold_cycle=0\n"
    )


def _fallback(ts, reason="large_context_bypass"):
    return (
        f"{ts} - INFO - Fallback triggered for model=v1/chat/completions, "
        f"from=local-qwen3, to=munitoai-free, reason={reason}\n"
    )


def _denied(ts, session="sess-denied"):
    return (
        f"{ts} - INFO - local_dispatch_denied session={session} owner=sess-owner "
        f"active=1 cold_start=False\n"
    )


def _stream_finished(ts, session, provider):
    return (
        f"{ts} - INFO - Stream finished: reason=tool_calls session={session} "
        f"provider={provider} model=Qwen3 request=[{{'type': 'text', 'text': 'x'}}]\n"
    )


def _write_log(tmp_path, lines, name="proxy.log"):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / name).write_text("".join(lines))
    return log_dir


def _run(log_dir, *extra, windows=True):
    args = ["python3", SCRIPT, "--log-dir", str(log_dir), "--json"]
    if windows:
        args += [
            "--before-start", BEFORE_START,
            "--before-end", BEFORE_END,
            "--after-start", AFTER_START,
            "--after-end", AFTER_END,
        ]
    args += list(extra)
    return subprocess.run(args, capture_output=True, text=True, timeout=120)


def _result(log_dir, *extra, windows=True):
    proc = _run(log_dir, *extra, windows=windows)
    assert proc.returncode in (0, 2), proc.stderr
    return json.loads(proc.stdout), proc.returncode


def _baseline_lines():
    """Baseline window: no idle admits, some bypasses/fallbacks/denials."""
    return [
        _routing_check("2026-10-19 00:10:00,000", "base-1", tokens=20000),
        _dispatch("2026-10-19 00:10:01,000", "base-1", 40000.0, 39900.0,
                  cache_state="warm"),
        _skip_line("2026-10-19 01:00:00,000", "base-2", cycle=0),
        _skip_line("2026-10-19 01:00:05,000", "base-2", cycle=2),
        _fallback("2026-10-19 01:00:06,000"),
        _fallback("2026-10-19 02:00:00,000"),
        _fallback("2026-10-19 03:00:00,000"),
        _fallback("2026-10-19 04:00:00,000"),
        _denied("2026-10-19 05:00:00,000"),
        _denied("2026-10-19 06:00:00,000"),
        _stream_finished("2026-10-19 07:00:00,000", "base-1", "local"),
        _stream_finished("2026-10-19 08:00:00,000", "base-2", "remote"),
    ]


def _baseline_with_interactive():
    """Baseline plus one interactive-tier dispatch so AC4 has a reference."""
    return _baseline_lines() + [
        _dispatch(
            "2026-10-19 12:00:00,000",
            "interactive-old",
            5000.0,
            4990.0,
            cache_state="warm",
            priority="high",
            tokens=20000,
        )
    ]


def _after_lines(*, idle_admits=6, fallbacks=6, denials=1, priority="high",
                 idle_service_ms=60000.0, interactive_service_ms=5000.0,
                 attributed_timeout=False, actions_after=None):
    """Post-deploy window with idle-admitted dispatches and non-idle traffic.

    Each idle-admitted request is a ``routing_idle_admit`` line followed two
    seconds later by its local dispatch line, in the same session — the shape
    the proxy emits when the override fires.
    """
    lines = []
    # Local activity so the counterfactual estimator has an anchor.
    lines.append(_dispatch("2026-10-20 00:00:00,000", "anchor", 1000.0, 900.0,
                           priority=priority, tokens=4000))
    for index in range(idle_admits):
        session = f"idle-{index}"
        tokens = 90000 + index * 1000
        lines.append(_idle_admit(_stamp("2026-10-20 01:00", index * 600), session, tokens=tokens))
        lines.append(
            _dispatch(
                _stamp("2026-10-20 01:00", index * 600 + 2),
                session,
                idle_service_ms + 100.0,
                idle_service_ms,
                priority=priority,
                tokens=tokens,
            )
        )
    # Non-idle local dispatches at the interactive tier — the AC4 baseline.
    for index in range(4):
        lines.append(
            _dispatch(
                _stamp("2026-10-20 10:00", index * 300),
                f"interactive-{index}",
                interactive_service_ms,
                interactive_service_ms - 10.0,
                cache_state="warm",
                priority=priority,
                tokens=20000,
            )
        )
    for index in range(fallbacks):
        lines.append(_fallback(_stamp("2026-10-20 15:00", index * 300)))
    for index in range(denials):
        lines.append(_denied(_stamp("2026-10-20 20:00", index * 300)))
    lines.append(_stream_finished("2026-10-20 04:00:00,000", "idle-0", "local"))
    lines.append(_stream_finished("2026-10-20 05:00:00,000", "remote-1", "remote"))
    if attributed_timeout:
        # 30 s after idle-0's admission (inside the 60 s correlation window).
        lines.append(
            "2026-10-20 01:00:30,000 - WARNING - dispatch_first_byte_timeout "
            "session=idle-0 ceiling=840s\n"
        )
    if actions_after:
        lines.extend(actions_after)
    return lines


# ---------------------------------------------------------------------------
# AC1 — window aggregates
# ---------------------------------------------------------------------------


def test_ac1_window_aggregates(tmp_path):
    data, _ = _result(_write_log(tmp_path, _baseline_lines() + _after_lines()))
    before = data["windows"]["before"]["ac1"]
    after = data["windows"]["after"]["ac1"]

    assert before["local_idle_admit"] == 0
    assert after["local_idle_admit"] == 6
    # Hold-cycle re-emission (cycle=2) is not counted as extra demand.
    assert before["large_context_bypass"] == 1
    assert before["remote_fallback_events"] == 4
    assert after["remote_fallback_events"] == 6
    assert before["local_dispatch_denied"] == 2
    assert after["local_dispatch_denied"] == 1
    # Serve share = local streams / (local + remote) streams.
    assert before["local_streams"] == 1
    assert before["remote_streams"] == 1
    assert before["local_serve_share_pct"] == 50.0
    assert after["local_serve_share_pct"] == 50.0
    # Prefill slot-seconds accumulate the slot-held service time.
    assert before["local_prefill_slot_seconds"] == 39.9
    assert after["local_prefill_slot_seconds"] > 0
    assert data["windows"]["before"]["hours"] == 24.0


def test_ac1_fallback_reasons_are_attributed(tmp_path):
    lines = _baseline_lines() + [
        _fallback("2026-10-19 09:00:00,000", reason="rate_limited"),
        _fallback("2026-10-19 09:10:00,000", reason="rate_limited"),
    ]
    data, _ = _result(_write_log(tmp_path, lines))
    reasons = data["windows"]["before"]["ac1"]["fallback_by_reason"]
    assert reasons["rate_limited"] == 2
    assert reasons["large_context_bypass"] == 4


def test_ac1_session_bearing_fallback_lines_keep_their_reason(tmp_path):
    """LP-0MVMBMSWR0036RES appends ``session=``/``attempt=`` (plus budget
    fields for ``rate_limited*``) after ``reason=``; the reason must still be
    counted, and the session must be picked up by the timeout-attribution
    evidence."""
    lines = _baseline_lines() + [
        "2026-10-19 09:00:00,000 - INFO - Fallback triggered for model=v1/chat/completions, "
        "from=local-qwen3, to=munitoai-free, reason=rate_limited "
        "session=herdr-1792767061-1282111-17770 attempt=1 limit=60 remaining=0 waited=0.000\n",
        "2026-10-19 09:10:00,000 - INFO - Fallback triggered for model=v1/chat/completions, "
        "from=munitoai-free, to=opencode-go, reason=HTTP 502 "
        "session=unknown attempt=2\n",
    ]
    data, _ = _result(_write_log(tmp_path, lines))
    reasons = data["windows"]["before"]["ac1"]["fallback_by_reason"]
    assert reasons["rate_limited"] == 1
    assert reasons["HTTP 502"] == 1
    assert reasons["large_context_bypass"] == 4


# ---------------------------------------------------------------------------
# AC2 — idle vs non-idle dispatches at the same context bucket
# ---------------------------------------------------------------------------


def test_ac2_pairs_idle_admits_with_their_dispatch(tmp_path):
    data, _ = _result(_write_log(tmp_path, _baseline_lines() + _after_lines()))
    ac2 = data["windows"]["after"]["ac2"]
    assert ac2["idle_admitted_sample_counts"] == {
        "idle_admit_events": 6,
        "paired_local_dispatches": 6,
        "unpaired_admits": 0,
    }
    assert ac2["idle_admitted_service_ms"]["n"] == 6
    assert ac2["idle_admitted_service_ms"]["p50_ms"] == 60000.0
    # Non-idle = anchor + 4 interactive dispatches (the 6 idle ones are claimed).
    assert ac2["non_idle_service_ms"]["n"] == 5


def test_ac2_bucket_comparison_uses_admit_prompt_size(tmp_path):
    data, _ = _result(_write_log(tmp_path, _baseline_lines() + _after_lines()))
    buckets = data["windows"]["after"]["ac2"]["by_bucket"]
    # 90k-96k prompts land in the 65536-131072 bucket.
    assert buckets["65536-131072"]["idle"]["n"] == 6
    assert buckets["65536-131072"]["non_idle"]["n"] == 0
    # The four interactive dispatches (20k tokens each) share their bucket.
    assert buckets["16384-32768"]["non_idle"]["n"] == 4
    assert buckets["16384-32768"]["idle"]["n"] == 0
    assert buckets["0-16384"]["non_idle"]["n"] == 1  # the 4k anchor
    assert data["windows"]["after"]["ac2"]["bucket_edges"] == [16384, 32768, 65536, 131072]


def test_ac2_unmatched_admit_is_reported_not_dropped(tmp_path):
    lines = _after_lines(idle_admits=1) + [
        _idle_admit("2026-10-20 06:00:00,000", "never-dispatched", tokens=95000),
    ]
    data, _ = _result(_write_log(tmp_path, lines))
    counts = data["windows"]["after"]["ac2"]["idle_admitted_sample_counts"]
    assert counts["idle_admit_events"] == 2
    assert counts["paired_local_dispatches"] == 1
    assert counts["unpaired_admits"] == 1


def test_ac2_dispatch_outside_correlation_window_is_not_paired(tmp_path):
    lines = [
        _idle_admit("2026-10-20 01:00:00,000", "late", tokens=95000),
        # 61 s later — outside the 60 s correlation window.
        _dispatch("2026-10-20 01:01:01,000", "late", 61000.0, 60900.0),
    ]
    data, _ = _result(_write_log(tmp_path, lines))
    counts = data["windows"]["after"]["ac2"]["idle_admitted_sample_counts"]
    assert counts["paired_local_dispatches"] == 0
    assert counts["unpaired_admits"] == 1
    # The dispatch still counts as a non-idle local dispatch.
    assert data["windows"]["after"]["ac2"]["non_idle_service_ms"]["n"] == 1


def test_ac2_legacy_dispatch_falls_back_to_routing_check_bucket(tmp_path):
    """Logs predating the dispatch-line instrumentation still bucket correctly."""
    lines = [
        _routing_check("2026-10-20 01:00:00,000", "legacy", tokens=20000),
        _dispatch("2026-10-20 01:00:02,000", "legacy", 40000.0, 39900.0),
    ]
    data, _ = _result(_write_log(tmp_path, lines))
    buckets = data["windows"]["after"]["ac2"]["by_bucket"]
    assert buckets["16384-32768"]["non_idle"]["n"] == 1


# ---------------------------------------------------------------------------
# AC3 — denials and timeout attribution
# ---------------------------------------------------------------------------


def test_ac3_attributed_coincident_and_unrelated_timeouts(tmp_path):
    extra = [
        # No session id, 30 s after idle-2's admission (01:20) → coincident.
        _fallback("2026-10-20 01:20:30,000", reason="empty_response"),
        # No session id, far from every admission → unrelated.
        _fallback("2026-10-20 09:00:00,000", reason="ReadTimeout"),
    ]
    lines = _baseline_lines() + _after_lines(attributed_timeout=True, actions_after=extra)
    data, _ = _result(_write_log(tmp_path, lines))
    ac3 = data["windows"]["after"]["ac3"]

    attributed = ac3["timeouts_attributed_to_idle_admit"]
    assert [entry["session"] for entry in attributed] == ["idle-0"]
    assert attributed[0]["reason"] == "dispatch_first_byte_timeout"
    assert attributed[0]["seconds_after_idle_admit"] == 30.0
    assert [entry["reason"] for entry in ac3["timeouts_coincident_with_idle_admit"]] == [
        "empty_response"
    ]
    assert ac3["timeouts_unrelated"] == 1
    assert ac3["correlation_window_seconds"] == 60.0
    assert "session id" in ac3["attribution_limit"]


def test_ac3_attributed_timeout_fails_the_check(tmp_path):
    lines = _baseline_lines() + _after_lines(attributed_timeout=True)
    data, _ = _result(_write_log(tmp_path, lines))
    checks = {check["target"]: check for check in data["verdict"]["checks"]}
    ac3_check = next(
        check
        for target, check in checks.items()
        if target.startswith("AC3: zero timeouts attributable")
    )
    assert ac3_check["pass"] is False


def test_ac3_denial_increase_fails_the_check(tmp_path):
    lines = _baseline_lines() + _after_lines(denials=40)
    data, _ = _result(_write_log(tmp_path, lines))
    check = next(
        check
        for check in data["verdict"]["checks"]
        if check["target"].startswith("AC3: local_dispatch_denied")
    )
    assert check["pass"] is False
    assert check["observed_pct_change"] > 5.0


# ---------------------------------------------------------------------------
# AC4 — interactive traffic
# ---------------------------------------------------------------------------


def test_ac4_interactive_latency_within_noise(tmp_path):
    data, _ = _result(_write_log(tmp_path, _baseline_with_interactive() + _after_lines()))
    ac4 = data["windows"]["after"]["ac4"]
    assert ac4["interactive_priority"] == "high"
    # Idle-admitted dispatches are the treatment, so the AC4 control is the
    # non-idle tier: 4 interactive dispatches + the anchor.
    assert ac4["interactive"]["n"] == 5
    assert ac4["interactive"]["p50_ms"] == 5000.0
    assert ac4["idle_admitted_by_priority"] == {"high": 6}
    assert "non-idle" in ac4["semantics"]
    check = next(
        check
        for check in data["verdict"]["checks"]
        if check["target"].startswith("AC4:")
    )
    assert check["pass"] is True


def test_ac4_regression_beyond_noise_fails_and_names_the_cause(tmp_path):
    after = _after_lines(interactive_service_ms=40000.0)
    data, _ = _result(_write_log(tmp_path, _baseline_with_interactive() + after))
    check = next(
        check
        for check in data["verdict"]["checks"]
        if check["target"].startswith("AC4:")
    )
    assert check["pass"] is False
    note = " ".join(data["verdict"]["notes"])
    assert "Interactive latency regressed" in note
    assert "idle-0" in note


def test_ac4_unmeasurable_when_dispatch_lines_lack_priority(tmp_path):
    """Pre-instrumentation logs carry no priority field — AC4 cannot pass."""
    lines = [
        _idle_admit("2026-10-20 01:00:00,000", "old", tokens=95000),
        _dispatch("2026-10-20 01:00:02,000", "old", 61000.0, 60900.0),  # no priority
    ]
    data, _ = _result(_write_log(tmp_path, lines))
    check = next(
        check
        for check in data["verdict"]["checks"]
        if check["target"].startswith("AC4:")
    )
    assert check["pass"] is False
    assert any("priority=" in note for note in data["verdict"]["notes"])


# ---------------------------------------------------------------------------
# AC5 — recommendation
# ---------------------------------------------------------------------------


def test_ac5_recommends_measure_when_the_override_never_fired(tmp_path):
    data, code = _result(_write_log(tmp_path, _baseline_lines()))
    recommendation = data["verdict"]["recommendation"]
    assert recommendation["decision"] == "measure"
    assert "server.local_idle_admit_enabled" in recommendation["follow_up"]
    assert code == 2


def test_ac5_recommends_enable_by_default_when_fallbacks_fall_without_harm(tmp_path):
    lines = _baseline_with_interactive() + _after_lines(fallbacks=2, denials=1)
    data, code = _result(_write_log(tmp_path, lines))
    recommendation = data["verdict"]["recommendation"]
    assert recommendation["decision"] == "enable_by_default"
    assert recommendation["fallback_reduction_pct"] > 0
    assert recommendation["reasons"]
    assert data["verdict"]["verdict"] == "PASS"
    assert code == 0


def test_ac5_recommends_tuning_when_interactive_latency_regresses(tmp_path):
    after = _after_lines(fallbacks=2, interactive_service_ms=40000.0)
    data, _ = _result(_write_log(tmp_path, _baseline_with_interactive() + after))
    recommendation = data["verdict"]["recommendation"]
    assert recommendation["decision"] == "tune"
    assert "priority" in recommendation["follow_up"]
    assert any("high-priority first byte moved" in reason
               for reason in recommendation["reasons"])


def test_ac5_tuning_follow_up_covers_threshold_when_no_fallback_reduction(tmp_path):
    lines = _baseline_with_interactive() + _after_lines(fallbacks=12)
    data, _ = _result(_write_log(tmp_path, lines))
    recommendation = data["verdict"]["recommendation"]
    assert recommendation["decision"] == "tune"
    assert "local_idle_admit_seconds" in recommendation["follow_up"]


def test_ac5_requires_enough_samples(tmp_path):
    lines = _baseline_with_interactive() + _after_lines(idle_admits=2, fallbacks=2)
    data, _ = _result(_write_log(tmp_path, lines))
    recommendation = data["verdict"]["recommendation"]
    assert recommendation["decision"] == "measure"
    assert "idle-admitted dispatch" in recommendation["follow_up"]


# ---------------------------------------------------------------------------
# Counterfactual estimator
# ---------------------------------------------------------------------------


def test_counterfactual_marks_only_long_idle_gaps_eligible(tmp_path):
    lines = [
        _dispatch("2026-10-20 01:00:00,000", "anchor", 1000.0, 900.0),
        # 200 s of silence → eligible at the 60 s threshold.
        _skip_line("2026-10-20 01:03:20,000", "gap-a"),
        # Immediately after the previous dispatch activity → not eligible.
        _dispatch("2026-10-20 01:04:00,000", "anchor", 1000.0, 900.0),
        _skip_line("2026-10-20 01:04:05,000", "gap-b"),
    ]
    data, _ = _result(_write_log(tmp_path, lines), "--idle-threshold", "60")
    cf = data["windows"]["after"]["counterfactual"]
    assert cf["approximation"] is True
    assert cf["measured_bypasses"] == 2
    assert cf["eligible_bypasses"] == 1
    assert cf["eligible_pct_of_bypasses"] == 50.0
    assert cf["idle_gap_p50_seconds"] is not None


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def test_cli_window_closed_requires_24h(tmp_path):
    lines = _baseline_lines() + _after_lines()
    log_dir = _write_log(tmp_path, lines)
    proc = subprocess.run(
        [
            "python3", SCRIPT, "--log-dir", str(log_dir), "--json", "--window-closed",
            "--before-start", "2026-10-19 00:00", "--before-end", "2026-10-19 06:00",
            "--after-start", "2026-10-20 00:00", "--after-end", "2026-10-20 06:00",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2, proc.stderr
    data = json.loads(proc.stdout)
    check = next(
        check
        for check in data["verdict"]["checks"]
        if check["target"].startswith("AC1: post-deploy window is >= 24 h")
    )
    assert check["pass"] is False
    assert check["observed_hours"] == 6.0


def test_cli_rejects_overlapping_windows(tmp_path):
    log_dir = _write_log(tmp_path, _baseline_lines())
    proc = _run(
        log_dir,
        "--before-start", "2026-10-19 00:00",
        "--before-end", "2026-10-20 12:00",
        "--after-start", "2026-10-20 00:00",
        "--after-end", "2026-10-21 00:00",
    )
    assert proc.returncode == 1
    assert "baseline window must end at or before" in proc.stderr


def test_cli_json_has_no_internal_keys(tmp_path):
    data, _ = _result(_write_log(tmp_path, _baseline_lines() + _after_lines()))
    assert set(data) == {"meta", "windows", "verdict"}
    for window in data["windows"].values():
        assert not [key for key in window if key.startswith("_")]
    assert data["meta"]["tool"] == "scripts/idle_admit_impact.py"
    assert data["meta"]["work_item"] == "LP-0MVGHXD59008UOUD"
    assert data["meta"]["feature_work_item"] == "LP-0MVGD9Z8Z0055Y5J"


def test_cli_markdown_renders_the_report_table(tmp_path):
    lines = _baseline_lines() + _after_lines()
    log_dir = _write_log(tmp_path, lines)
    proc = subprocess.run(
        [
            "python3", SCRIPT, "--log-dir", str(log_dir), "--markdown",
            "--before-start", BEFORE_START, "--before-end", BEFORE_END,
            "--after-start", AFTER_START, "--after-end", AFTER_END,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2, proc.stderr
    assert "| Metric | Baseline | Post-deploy |" in proc.stdout
    assert "`local_idle_admit`" in proc.stdout
    assert "**Verdict:** `WARN`" in proc.stdout


def test_cli_survives_a_deployment_after_the_window_start(tmp_path):
    """``--deployed-at`` later than the window start fails the AC1 precheck."""
    log_dir = _write_log(tmp_path, _baseline_lines() + _after_lines())
    proc = _run(log_dir, "--deployed-at", "2026-10-20 12:00")
    assert proc.returncode == 2, proc.stderr
    data = json.loads(proc.stdout)
    check = next(
        check
        for check in data["verdict"]["checks"]
        if check["target"].startswith("AC1: post-deploy window starts")
    )
    assert check["pass"] is False
