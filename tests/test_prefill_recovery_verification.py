#!/usr/bin/env python3
"""Tests for scripts/prefill_recovery_verification.py (LP-0MVKUKGER000TW2W).

The script is the reproducible measurement behind the live post-fix
verification of prefill-deadline recovery: a dispatch cancelled by the
prefill deadline must release the session's lease / in-flight marker within
a bounded interval (AC2), the per-hour report must show no thrash hours and
zero ``reason=local_lease_active`` fallbacks (AC1), and local serving must
resume afterwards (AC3).

These tests run the pure analysis functions against synthetic proxy log
lines (no production log, no network), and assert the observable metric
contract the acceptance criteria are written against:

  * AC2 — ``prefill_deadline_exceeded`` is paired with the session's
    ``lease_released ... reason=prefill_deadline`` and the wall-clock delta
    is bounded (default <= 300 s); a deadline with no release is flagged;
  * AC1 — hours with >= ``min_attempts`` local dispatch attempts and zero
    local streams served are thrash hours; ``local_lease_active`` fallbacks
    are summed and must be zero;
  * AC3 — local stream finishes in the window after a deadline are counted
    and resume verdict derived.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "scripts")
    ),
)
import prefill_recovery_verification as prv


def _lines() -> list[str]:
    """Synthetic log covering the event shapes of both the incident and the fix.

    session A: deadline fires at 10:00:00, lease released at 10:00:02
               (2 s delta), cooldown started, a same-session retry is skipped
               (routing_skip_local reason=prefill_thrash_cooldown) and a
               different session is served locally at 10:03:00.
    session B: deadline at 10:05:00 with NO following release -> unreleased.
    session C: a release BEFORE any deadline (orphan) -> not paired.
    session D: interleaved with A: deadline 10:01:00, release 10:06:00
               (300 s delta — exactly the bound) and another release with no
               deadline (orphan, ignored).
    """
    return [
        "2026-10-23 10:00:00,123 - WARNING - prefill_deadline_exceeded session=A deadline=90s → cancelling local dispatch and detouring to the next provider",
        "2026-10-23 10:00:01,100 - INFO - [local] POST http://127.0.0.1:35939/v1/chat/completions session_id=A body={...}",
        "2026-10-23 10:00:02,000 - INFO - lease_released session=A reason=prefill_deadline endpoint=default",
        "2026-10-23 10:00:03,000 - INFO - prefill_thrash_cooldown_started session=A cooldown_s=600",
        "2026-10-23 10:00:30,000 - INFO - routing_skip_local provider=local model=Qwen3 reason=prefill_thrash_cooldown → skipping local, falling back to next provider session=A cooldown_remaining=590s",
        "2026-10-23 10:01:00,000 - WARNING - prefill_deadline_exceeded session=D deadline=90s → cancelling local dispatch and detouring to the next provider",
        "2026-10-23 10:03:00,000 - INFO - Stream finished: reason=stop tokens=100/20/120 session=C provider=local model=Qwen3",
        "2026-10-23 10:05:00,000 - WARNING - prefill_deadline_exceeded session=B deadline=90s → cancelling local dispatch and detouring to the next provider",
        "2026-10-23 10:06:00,000 - INFO - lease_released session=D reason=prefill_deadline endpoint=default",
        "2026-10-23 10:07:00,000 - INFO - lease_released session=C reason=prefill_deadline endpoint=default",
        "2026-10-23 10:08:00,000 - INFO - Stream finished: reason=stop tokens=20/10/30 session=E provider=opencode-go model=deepseek-v4-flash",
        "2026-10-23 10:09:00,000 - INFO - Stream finished: reason=stop tokens=50/5/55 session=F provider=local model=Qwen3",
    ]


class DeadlineReleasePairingTest(unittest.TestCase):
    def test_pairs_deadline_with_following_release_per_session(self):
        releases = prv.parse_deadline_release_events(_lines())
        by_session = {r.session: r for r in releases}
        self.assertIn("A", by_session)
        self.assertEqual(by_session["A"].release_seconds, 2.0)
        # D's release (10:06:00) pairs with ITS deadline (10:01:00).
        self.assertEqual(by_session["D"].release_seconds, 300.0)

    def test_interleaved_sessions_do_not_cross_pair(self):
        releases = prv.parse_deadline_release_events(_lines())
        # C's release at 10:07 has no deadline before it -> orphan, skipped.
        self.assertNotIn("C", {r.session for r in releases})

    def test_deadline_without_release_is_unreleased(self):
        result = prv.parse_deadline_release_events(_lines())
        self.assertIn("B", result.unreleased_sessions)

    def test_recovery_bound_verdict(self):
        # Only A (2 s) and D (300 s) participate; B's unreleased deadline is
        # excluded here and covered by its own verdict test below.
        lines = [line for line in _lines() if "session=B" not in line]
        releases = prv.parse_deadline_release_events(lines)
        v = prv.ac2_recovery_verdict(releases, bound_seconds=300)
        self.assertTrue(v.within_bound)
        self.assertEqual(v.max_release_seconds, 300.0)
        # A tighter bound must fail (D's 300 s delta exceeds 299 s).
        tight = prv.ac2_recovery_verdict(releases, bound_seconds=299)
        self.assertFalse(tight.within_bound)

    def test_unreleased_deadline_fails_the_verdict(self):
        releases = prv.parse_deadline_release_events(_lines())
        v = prv.ac2_recovery_verdict(releases, bound_seconds=300)
        self.assertIn("B", v.unreleased_sessions)
        self.assertFalse(v.within_bound)


class Ac1ClassificationTest(unittest.TestCase):
    RECORDS = [
        {"hour": "2026-10-23 02:00:00", "local_dispatch_attempts": 173,
         "local_served": 0, "lease_active_fallbacks": 14},
        {"hour": "2026-10-23 03:00:00", "local_dispatch_attempts": 5,
         "local_served": 0, "lease_active_fallbacks": 0},
        {"hour": "2026-10-23 16:00:00", "local_dispatch_attempts": 147,
         "local_served": 70, "lease_active_fallbacks": 19},
    ]

    def test_thrash_hour_detection(self):
        m = prv.measure_ac1(self.RECORDS, min_attempts=10)
        self.assertEqual(m.thrash_hours, ["2026-10-23 02:00:00"])
        self.assertEqual(m.lease_active_fallbacks, 33)
        self.assertFalse(m.clean)  # 33 fallbacks, one thrash hour

    def test_clean_window(self):
        records = [
            {"hour": "2026-10-23 20:00:00", "local_dispatch_attempts": 12,
             "local_served": 8, "lease_active_fallbacks": 0},
        ]
        m = prv.measure_ac1(records, min_attempts=10)
        self.assertTrue(m.clean)
        self.assertEqual(m.thrash_hours, [])

    def test_helpers_standalone(self):
        # The same thrash-hour logic used standalone (defensive guarding
        # against a malformed records list).
        records = [
            {"hour": "2026-10-23 02:00:00", "local_dispatch_attempts": 173,
             "local_served": 0, "lease_active_fallbacks": 14},
        ]
        self.assertTrue(prv.is_thrash_hour(records[0], min_attempts=10))
        self.assertFalse(
            prv.is_thrash_hour(records[0], min_attempts=200)
        )


class Ac3ResumeTest(unittest.TestCase):
    def test_local_finishes_after_deadline(self):
        releases = prv.parse_deadline_release_events(_lines())
        last = max(r.release_ts for r in releases)
        counts = prv.count_local_finishes_after(
            _lines(), after=last, window_seconds=3600
        )
        # Only the 10:09 local finish strictly follows the last release
        # (10:06); the 10:03 finish predates it; the remote finish is
        # never counted.
        self.assertEqual(counts, 1)

    def test_resume_verdict(self):
        releases = prv.parse_deadline_release_events(_lines())
        last = max(r.release_ts for r in releases)
        counts = prv.count_local_finishes_after(
            _lines(), after=last, window_seconds=3600
        )
        self.assertTrue(prv.local_serving_resumed(counts))


class SessionRecoveryTest(unittest.TestCase):
    """Stall-test timeline parsing (dispatch -> deadline -> release)."""

    LINES = [
        "2026-10-23 20:19:17,590 - INFO - [local] POST http://127.0.0.1:8000/v1/chat/completions body={...} session_id=stall-A session={'x-session-id': 'stall-A'} slot=0",
        "2026-10-23 20:20:47,590 - WARNING - prefill_deadline_exceeded session=stall-A deadline=90s → cancelling local dispatch and detouring to the next provider",
        "2026-10-23 20:20:47,590 - INFO - lease_released session=stall-A reason=prefill_deadline endpoint=default",
        "2026-10-23 20:21:00,000 - INFO - Stream finished: reason=stop tokens=10/2/12 session=stall-A provider=local model=Qwen3",
    ]

    def test_full_timeline(self):
        rec = prv.parse_session_recovery(self.LINES, "stall-A")
        self.assertIsNotNone(rec.dispatch_ts)
        self.assertIsNotNone(rec.deadline_ts)
        self.assertEqual(rec.release_reason, "prefill_deadline")
        self.assertEqual(rec.delta_seconds, 0.0)
        self.assertIsNotNone(rec.local_finish_ts)
        self.assertTrue(prv.stall_recovery_verdict(rec, bound_seconds=300))

    def test_watchdog_release_without_deadline_measures_from_dispatch(self):
        # Non-streaming dispatch: headers return early, so the deadline never
        # fires and the no-progress watchdog frees the lease instead. The
        # delta must be measured from the dispatch marker.
        lines = [
            "2026-10-23 20:19:17,590 - INFO - [local] POST http://x/v1/chat/completions body={...} session_id=stall-B session={'x-session-id': 'stall-B'} slot=0",
            "2026-10-23 20:22:26,978 - WARNING - lease_released session=stall-B reason=no_progress_watchdog endpoint=default",
        ]
        rec = prv.parse_session_recovery(lines, "stall-B")
        self.assertIsNone(rec.deadline_ts)
        self.assertEqual(rec.release_reason, "no_progress_watchdog")
        self.assertAlmostEqual(rec.delta_seconds, 189.0, places=2)
        self.assertTrue(prv.stall_recovery_verdict(rec, bound_seconds=300))
        self.assertFalse(prv.stall_recovery_verdict(rec, bound_seconds=180))

    def test_missing_release_is_not_recovered(self):
        lines = [line for line in self.LINES if "lease_released" not in line]
        rec = prv.parse_session_recovery(lines, "stall-A")
        self.assertIsNone(rec.release_ts)
        self.assertIsNone(rec.delta_seconds)
        self.assertFalse(prv.stall_recovery_verdict(rec))

    def test_other_sessions_are_ignored(self):
        rec = prv.parse_session_recovery(self.LINES, "other-session")
        self.assertIsNone(rec.dispatch_ts)
        self.assertIsNone(rec.release_ts)


class PromptGeneratorTest(unittest.TestCase):
    def test_deterministic(self):
        a = prv.build_novel_prompt(token_target=40000, seed=42)
        b = prv.build_novel_prompt(token_target=40000, seed=42)
        self.assertEqual(a, b)

    def test_size_scales_with_target(self):
        small = prv.build_novel_prompt(token_target=10000, seed=1)
        large = prv.build_novel_prompt(token_target=60000, seed=1)
        self.assertGreater(large.upper().count(" "), small.upper().count(" "))
        # Generation targets ~1.3 tokens/word, so 10000 tokens needs ~7700
        # words; assert a generous floor to catch gross under-generation.
        self.assertGreaterEqual(small.upper().count(" ") + 1, 6000)

    def test_different_seeds_differ(self):
        a = prv.build_novel_prompt(token_target=40000, seed=1)
        b = prv.build_novel_prompt(token_target=40000, seed=2)
        self.assertNotEqual(a, b, "two seeds must produce distinct prompts")


if __name__ == "__main__":
    unittest.main()
