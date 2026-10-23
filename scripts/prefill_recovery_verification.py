#!/usr/bin/env python3
"""Reproducible live verification of post-deadline prefill recovery.

Measurement for LP-0MVKUKGER000TW2W (part of LP-0MVKN9IRP003XJH6 — the
2026-10-23 wedged-prefill lease loop). After the fix chain C1-C5, a local
dispatch cancelled by the prefill deadline must release the session's lease
and in-flight marker within a bounded interval (bounded by
``local_prefill_deadline_seconds`` + overhead, verified against a
``--recovery-bound-seconds`` default of 300 = the AC's <= 5 minutes), the
post-fix hourly report must show no thrash hour (>= ``--min-attempts`` local
dispatch attempts with zero local streams served) and zero
``reason=local_lease_active`` fallbacks, and local serving must resume.

What is measured and how (metric provenance):

  * AC1 — thrash hours / ``local_lease_active`` fallbacks come from the
    per-hour report figures of the proxy-usage-analysis pipeline (the report
    is the reference the AC is written against). Pass them in with
    ``--hour-records <json>`` (the pipeline's ``hourly_activity`` list, e.g.
    from ``analyze_proxy_usage.py --json``), or let ``--measure`` shell out
    to the pipeline with the same window and read them itself.
  * AC2 — ``prefill_deadline_exceeded`` events are paired, per session, with
    the first following ``lease_released ... reason=prefill_deadline``; each
    pair's wall-clock delta is the lease-hold after cancellation. A deadline
    with no release within the window is flagged ``unreleased`` (the pre-fix
    ~4 h wedge shape would appear here). Parsed directly from the proxy log
    (pure, deterministic, unit-tested with synthetic lines).
  * AC3 — ``Stream finished ... provider=local`` events within a window after
    the last deadline release prove local decode resumed.
  * Coverage — an adjacent-file gap larger than ``--max-log-gap-seconds``
    inside the requested window fails the verdict, so a lost window (the
    2026-10-23 18:29:12 -> 22:04:16 hole) is reported instead of silently
    producing a partial read (LP-0MVLHMAWV008EKTI).

Usage:
  ./scripts/prefill_recovery_verification.py --measure [--since ..] [--until ..] [--json]
  ./scripts/prefill_recovery_verification.py --stall-test [--json]

Exit codes:
  0 - success
  1 - no proxy logs found / unexpected error
  2 - the measurement verdicts do not hold (ACs not met)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from lib.proxy_logs import (  # noqa: E402
    discover_proxy_log_files,
    measure_log_coverage,
    open_proxy_log_text,
)

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")
_DEADLINE_RE = re.compile(r"prefill_deadline_exceeded session=(\S+)")
_RELEASE_RE = re.compile(r"lease_released session=(\S+) reason=prefill_deadline")
_RELEASE_ANY_RE = re.compile(r"lease_released session=(\S+) reason=(\w+)")
_DISPATCH_RE = re.compile(r"\[local\] POST .*session_id=(\S+)")
_FINISH_LOCAL_RE = re.compile(r"Stream finished:.*provider=local")
_FINISH_RE = re.compile(r"Stream finished:.*session=(\S+) provider=(\S+)")

_TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")
_WORDS = (
    "quartz fjord melody acorn velvet dusk lantern prism harbour cobalt "
    "meadow arctic beacon marble willow horizon amber granite cedar glacier "
    "porcelain sable azure tundra orchard summit velvet coral ember willow "
    "harbour phoenix granite cosine nebula quartz alpine beryl cobalt dune "
    "foliage granite horizon iris jasper kelp lagoon meadow nickel opal "
    "palm quartz river summit tundra urn velvet wave xenon yield zone"
).split()


def _parse_ts(text: str) -> dt.datetime | None:
    m = _TS_RE.match(text)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")


def _parse_bound(text: str) -> dt.datetime:
    for fmt in _TS_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"invalid timestamp {text!r} (expected 'YYYY-MM-DD [HH:MM[:SS]]')"
    )


# --------------------------------------------------------------------------
# AC2 — deadline -> lease-release pairing
# --------------------------------------------------------------------------
@dataclass
class DeadlineRelease:
    session: str
    deadline_ts: dt.datetime
    release_ts: dt.datetime
    release_seconds: float


@dataclass
class DeadlineReleaseResult:
    releases: list[DeadlineRelease]
    unreleased_sessions: list[str] = field(default_factory=list)

    def __iter__(self):
        return iter(self.releases)


def parse_deadline_release_events(
    lines, since: dt.datetime | None = None, until: dt.datetime | None = None
) -> DeadlineReleaseResult:
    """Pair ``prefill_deadline_exceeded`` with the next ``lease_released ...
    reason=prefill_deadline`` for the same session (in log order).

    Per-session FIFO: a release consumes the oldest pending deadline of its
    session. A deadline still pending at window end (or at EOF) is
    ``unreleased`` — the pre-fix wedge kept the lease for ~4 h; the fix
    releases it inline, so any deadline paired within the window at all is
    evidence of post-fix behaviour, and the recorded delta is the measured
    hold.
    """
    pending: dict[str, dt.datetime] = {}
    releases: list[DeadlineRelease] = []
    done: set[str] = set()

    for line in lines:
        ts = _parse_ts(line)
        if ts is None:
            continue
        if since is not None and ts < since:
            continue
        if until is not None and ts >= until:
            continue
        m = _DEADLINE_RE.search(line)
        if m:
            pending.setdefault(m.group(1), ts)
            continue
        m = _RELEASE_RE.search(line)
        if m:
            session = m.group(1)
            if session in pending:
                deadline_ts = pending.pop(session)
                releases.append(
                    DeadlineRelease(
                        session=session,
                        deadline_ts=deadline_ts,
                        release_ts=ts,
                        release_seconds=(ts - deadline_ts).total_seconds(),
                    )
                )
                done.add(session)
            # An orphan release with no pending deadline is intentionally
            # ignored (not evidence of a stall).
    unreleased = sorted(set(pending) - done)
    return DeadlineReleaseResult(releases=releases, unreleased_sessions=unreleased)


@dataclass
class RecoveryVerdict:
    within_bound: bool
    max_release_seconds: float | None
    releases: list[DeadlineRelease]
    unreleased_sessions: list[str]


def ac2_recovery_verdict(
    result: DeadlineReleaseResult, bound_seconds: int | float = 300
) -> RecoveryVerdict:
    """AC2: every deadline release must fall within ``bound_seconds``.

    An unreleased (or over-bound) deadline fails the verdict. The bound
    default 300 s is the AC's <= 5 minutes.
    """
    max_seconds = (
        max(r.release_seconds for r in result.releases) if result.releases else None
    )
    within = (
        not result.unreleased_sessions
        and (max_seconds is None or max_seconds <= bound_seconds)
    )
    return RecoveryVerdict(
        within_bound=within,
        max_release_seconds=max_seconds,
        releases=result.releases,
        unreleased_sessions=result.unreleased_sessions,
    )


@dataclass
class SessionRecovery:
    """One session's forced-stall timeline (stall-test mode).

    ``delta_seconds`` is the release delay measured from the prefill deadline
    when one fired, else from the dispatch marker — bounded by
    ``--recovery-bound-seconds`` (300) for AC2. ``release_reason`` records
    which mechanism freed the gate (``prefill_deadline`` or
    ``no_progress_watchdog``).
    """

    session: str
    dispatch_ts: dt.datetime | None = None
    deadline_ts: dt.datetime | None = None
    release_ts: dt.datetime | None = None
    release_reason: str | None = None
    local_finish_ts: dt.datetime | None = None

    @property
    def delta_seconds(self) -> float | None:
        if self.release_ts is None:
            return None
        origin = self.deadline_ts or self.dispatch_ts
        if origin is None:
            return None
        return (self.release_ts - origin).total_seconds()


def parse_session_recovery(lines, session_id: str) -> SessionRecovery:
    """Extract the stall-test timeline for one session from proxy-log lines.

    Dispatch (``[local] POST ... session_id=<sid>``), any prefill deadline,
    any lease release (the AC2 gate-freeing events come from both the
    deadline path and, before/after headers, the no-progress watchdog —
    whichever fires is the observed recovery) and any local stream finish
    (AC3 evidence for this session).
    """
    rec = SessionRecovery(session=session_id)
    for line in lines:
        ts = _parse_ts(line)
        if ts is None:
            continue
        m = _DISPATCH_RE.search(line)
        if m and m.group(1) == session_id and rec.dispatch_ts is None:
            rec.dispatch_ts = ts
            continue
        m = _DEADLINE_RE.search(line)
        if m and m.group(1) == session_id and rec.deadline_ts is None:
            rec.deadline_ts = ts
            continue
        m = _RELEASE_ANY_RE.search(line)
        if m and m.group(1) == session_id and rec.release_ts is None:
            rec.release_ts = ts
            rec.release_reason = m.group(2)
            continue
        m = _FINISH_RE.search(line)
        if m and m.group(1) == session_id and m.group(2) == "local":
            rec.local_finish_ts = ts
    return rec


def stall_recovery_verdict(rec: SessionRecovery, bound_seconds: int | float = 300) -> bool:
    """AC2: the stall released the gate within ``bound_seconds``."""
    if rec.release_ts is None or rec.delta_seconds is None:
        return False
    return rec.delta_seconds <= bound_seconds


# --------------------------------------------------------------------------
# AC1 — thrash hours / lease-active fallbacks (report figures)
# --------------------------------------------------------------------------

def is_thrash_hour(record: dict, min_attempts: int = 10) -> bool:
    """True when the hour attempted local >= ``min_attempts`` times and never
    served a local stream — the 2026-10-23 wedge signature."""
    return (
        int(record.get("local_dispatch_attempts") or 0) >= min_attempts
        and int(record.get("local_served") or 0) == 0
    )


@dataclass
class Ac1Summary:
    thrash_hours: list[str]
    lease_active_fallbacks: int
    clean: bool  # no thrash hour AND zero lease-active fallbacks


def measure_ac1(hour_records: list[dict], min_attempts: int = 10) -> Ac1Summary:
    """AC1 from per-hour report figures (the analysis pipeline's
    ``hourly_activity`` records)."""
    thrash = [
        r["hour"] for r in hour_records if is_thrash_hour(r, min_attempts)
    ]
    fallbacks = sum(int(r.get("lease_active_fallbacks") or 0) for r in hour_records)
    return Ac1Summary(
        thrash_hours=thrash,
        lease_active_fallbacks=fallbacks,
        clean=not thrash and fallbacks == 0,
    )


# --------------------------------------------------------------------------
# AC3 — local decode resumes after recovery
# --------------------------------------------------------------------------

def count_local_finishes_after(
    lines, after: dt.datetime, window_seconds: int | float = 3600
) -> int:
    count = 0
    for line in lines:
        ts = _parse_ts(line)
        if ts is None or ts < after:
            continue
        if (ts - after).total_seconds() > window_seconds:
            break  # lines are ordered; the tail is beyond the window
        if _FINISH_LOCAL_RE.search(line):
            count += 1
    return count


def local_serving_resumed(local_finishes_after: int) -> bool:
    return local_finishes_after > 0


# --------------------------------------------------------------------------
# Stall-test helpers
# --------------------------------------------------------------------------

def build_novel_prompt(token_target: int, seed: int) -> str:
    """Deterministic, novel prompt of ~``token_target`` tokens.

    Seeded random words (English prose ≈ 1.3 tokens/word) with a unique
    per-seed header line so no two tests share a prompt-prefix (defeats
    prompt-cache prefix reuse, forcing a full prefill).
    """
    rng = random.Random(seed)
    # English prose runs ~1.3 tokens/word; over-generate a little so the
    # prompt reliably reaches the target token count.
    words_needed = max(1, int(token_target / 1.3))
    words = [rng.choice(_WORDS) for _ in range(words_needed)]
    text = " ".join(words)
    # Unique prefix: forces a cold cache regardless of any shared vocabulary.
    header = f"VERIFICATION PROMPT SEED {seed} {rng.randrange(10**12):012d}. "
    return header + text


def _probe_local_serving(
    base_url: str, model: str, session_id: str, timeout: float = 120
) -> dict:
    """AC3: a small request on a fresh session must be served locally.

    Returns {status, served_local, tokens, snippet}: served_local is True when
    the response came from the local provider (X-Local-Route: local) or a
    local stream finished for the session."""
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Reply with the single word: pong"}
        ],
        "max_tokens": 16,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "X-Session-ID": session_id,
        "X-Local-Preference": "prefer-local",
    }
    result = {"status": None, "served_local": False, "tokens": 0, "snippet": ""}
    try:
        resp = _post(base_url, "/v1/chat/completions", payload, headers, timeout=timeout)
        result["status"] = str(resp.status)
        route = resp.headers.get("X-Local-Route") or ""
        body = resp.read(4000).decode(errors="replace")
        result["snippet"] = body[:200]
        result["served_local"] = (
            route == "local" or "local" in route.lower()
            or "provider=local" in body
        )
        try:
            data = json.loads(body)
            usage = data.get("usage") or {}
            result["tokens"] = int(usage.get("completion_tokens") or 0)
        except Exception:
            pass
    except Exception as e:
        result["status"] = f"error({type(e).__name__})"
    return result


def _post(base_url: str, path: str, body: dict, headers: dict, timeout: float):
    req = urllib.request.Request(
        base_url + path, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    return urllib.request.urlopen(req, timeout=timeout)


def run_stall_test(
    base_url: str,
    model: str,
    prompt_tokens: int,
    session_id: str,
    log_dir: Path,
    stream: bool = False,
    recovery_bound: int = 300,
    json_out: bool = False,
) -> int:
    """Drive one deliberate stalled prefill and measure gate recovery.

    Timeline comes from both the HTTP client view and the proxy log:
      dispatch_ts — first ``[local] POST`` for the session (log)
      deadline_ts — ``prefill_deadline_exceeded`` (streaming path; the
                    deadline is the 90 s first-byte backstop)
      release_ts  — first ``lease_released`` for the session (deadline or
                    no-progress watchdog; either is the gate recovery)
      gate poll   — ``/llama/local/status`` free again
    AC2 passes when release occurred within ``recovery_bound`` (default
    300 s) of (deadline or dispatch). AC3 probes that a small fresh-session
    request is then served locally, returning the response route/tokens.
    """
    seed = int(time.time()) % (2**31)
    prompt = build_novel_prompt(prompt_tokens, seed)
    print(f"prompt: ~{len(prompt.split()):,} words / ~{len(prompt):,} chars, "
          f"target={prompt_tokens} tokens, seed={seed}")

    common_headers = {
        "Content-Type": "application/json",
        "X-Session-ID": session_id,
        "X-Local-Preference": "local-only",
    }
    t0 = time.time()
    status_code: str | None = None
    body_snippet: str = ""
    try:
        resp = _post(
            base_url,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8,
                "stream": stream,
            },
            common_headers,
            timeout=250,
        )
        status_code = str(resp.status)
        payload = resp.read(4000).decode(errors="replace")
        body_snippet = payload[:400]
    except urllib.error.HTTPError as e:
        status_code = str(e.code)
        try:
            body_snippet = e.read(2000).decode(errors="replace")[:200]
        except Exception:
            pass
    except Exception as e:  # timeout / connection errors are expected outcomes
        status_code = f"error({type(e).__name__})"
        body_snippet = str(e)[:200]
    t_response = time.time() - t0
    print(f"dispatch response: {status_code} after {t_response:.1f}s "
          f"(body: {body_snippet!r})")

    # Rebuild the per-line view of the log within the test's span.
    window_start = dt.datetime.now() - dt.timedelta(minutes=10)
    lines: list[str] = []
    for path in discover_proxy_log_files(log_dir):
        with open_proxy_log_text(path) as fh:
            for line in fh:
                ts = _parse_ts(line)
                if ts is None or ts < window_start:
                    continue
                lines.append(line)
    lines.sort(key=lambda line: _parse_ts(line) or dt.datetime.min)

    rec = parse_session_recovery(lines, session_id)
    ac2_met = stall_recovery_verdict(rec, bound_seconds=recovery_bound)

    # Gate recovery: poll until free (or the bound expires).
    recovered_s = None
    bound = recovery_bound + int(t_response)
    deadline = time.time() + bound
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                base_url + "/llama/local/status", timeout=5
            ) as r:
                st = json.loads(r.read())
            in_flight = st.get("local_prefills_in_flight") or 0
            available = st.get("available_slots") or 0
            if in_flight == 0 and available > 0:
                recovered_s = round(time.time() - t0, 1)
                break
        except Exception:
            pass
        time.sleep(2)
    if recovered_s is not None:
        print(f"gate recovered: in_flight=0 available_slots>0 at +{recovered_s}s")
    else:
        print(f"gate NOT observed free within {bound}s of dispatch")

    # AC3: local serving resumes on a fresh session.
    probe = _probe_local_serving(base_url, model, session_id + "-served")
    print(f"ac3 probe: status={probe['status']} served_local={probe['served_local']} "
          f"tokens={probe['tokens']} (X-Local-Route via snippet: {probe['snippet'][:80]!r})")

    outcome = {
        "session": session_id,
        "model": model,
        "stream": stream,
        "prompt_tokens_target": prompt_tokens,
        "prompt_words": len(prompt.split()),
        "response_status": status_code,
        "response_body_snippet": body_snippet,
        "dispatch_to_response_seconds": round(t_response, 2),
        "gate_recovered_seconds": recovered_s,
        "ac2": {
            "dispatch": rec.dispatch_ts.isoformat(sep=" ") if rec.dispatch_ts else None,
            "deadline": rec.deadline_ts.isoformat(sep=" ") if rec.deadline_ts else None,
            "release": rec.release_ts.isoformat(sep=" ") if rec.release_ts else None,
            "release_reason": rec.release_reason,
            "release_delta_seconds": round(rec.delta_seconds, 2)
            if rec.delta_seconds is not None else None,
            "recovery_bound_seconds": recovery_bound,
            "within_bound": ac2_met,
        },
        "ac3": probe,
        "all_verdicts_met": ac2_met and probe["served_local"],
    }
    if json_out:
        print(json.dumps(outcome, indent=2))
    else:
        print(f"ac2: deadline={rec.deadline_ts} release={rec.release_ts} "
              f"reason={rec.release_reason} delta={rec.delta_seconds}s "
              f"within_bound={ac2_met}")
        print(f"ac3: served_local={probe['served_local']} tokens={probe['tokens']}")
    return 0 if outcome["all_verdicts_met"] else 2


# --------------------------------------------------------------------------
# Measure mode
# --------------------------------------------------------------------------

def _pipeline_hour_records(
    analysis_cmd: list[str],
    since: dt.datetime | None,
    until: dt.datetime | None,
    log_dir: Path,
) -> list[dict]:
    """Run the proxy-usage-analysis pipeline and return its hourly_activity
    records (the AC1 reference figures)."""
    cmd = [
        *analysis_cmd,
        "--log-dir", str(log_dir),
        "--json", "--quiet",
    ]
    if until is not None:
        cmd += ["--end", until.strftime("%Y-%m-%d %H:%M:%S")]
    if since is not None:
        cmd += ["--start", since.strftime("%Y-%m-%d %H:%M:%S")]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(
            f"analysis pipeline failed ({proc.returncode}): "
            f"{proc.stderr[-2000:]}"
        )
    data = json.loads(proc.stdout)
    return data.get("hourly_activity") or []


def run_measure(
    log_dir: Path,
    since: dt.datetime | None,
    until: dt.datetime | None,
    hour_records: list[dict] | None,
    analysis_cmd: list[str],
    min_attempts: int,
    recovery_bound: int,
    resume_window: int,
    max_log_gap_seconds: float = 3600.0,
) -> dict:
    lines: list[str] = []
    for path in discover_proxy_log_files(log_dir):
        with open_proxy_log_text(path) as fh:
            lines.extend(fh)
    if not lines:
        raise RuntimeError(f"no parseable proxy logs found in {log_dir}")
    # Rotated files overlap and the two naming schemes sort ambiguously;
    # global timestamp order keeps the per-session FIFO pairing correct and
    # the AC3 window break exact.
    lines.sort(key=lambda line: _parse_ts(line) or dt.datetime.min)

    result = parse_deadline_release_events(lines, since, until)
    ac2 = ac2_recovery_verdict(result, bound_seconds=recovery_bound)
    if hour_records is None:
        hour_records = _pipeline_hour_records(
            analysis_cmd, since, until, log_dir
        )
    ac1 = measure_ac1(hour_records, min_attempts=min_attempts)

    last_release = (
        max(r.release_ts for r in result.releases) if result.releases else None
    )
    finishes_after = (
        count_local_finishes_after(lines, after=last_release, window_seconds=resume_window)
        if last_release is not None
        else 0
    )
    ac3 = local_serving_resumed(finishes_after)

    # Coverage (LP-0MVLHMAWV008EKTI): a lost window must fail the verdict
    # instead of silently producing a partial read. Adjacent-file gaps over
    # the threshold are the incident shape (a file boundary spanning hours).
    coverage = measure_log_coverage(
        log_dir, since=since, until=until, max_gap_seconds=max_log_gap_seconds
    )

    report = {
        "window": {
            "since": since.isoformat(sep=" ") if since else None,
            "until": until.isoformat(sep=" ") if until else None,
            "log_dir": str(log_dir),
        },
        "coverage": {
            "max_gap_seconds": coverage.max_gap_seconds,
            "files": [
                {
                    "file": span.path.name,
                    "first": span.first.isoformat(sep=" "),
                    "last": span.last.isoformat(sep=" "),
                    "lines": span.line_count,
                }
                for span in coverage.spans
            ],
            "gaps": [
                {
                    "before_file": gap.before.name,
                    "before_last": gap.before_last.isoformat(sep=" "),
                    "after_file": gap.after.name,
                    "after_first": gap.after_first.isoformat(sep=" "),
                    "gap_seconds": round(gap.gap_seconds, 2),
                }
                for gap in coverage.gaps
            ],
            "window_start_gap_seconds": (
                round(coverage.window_start_gap_seconds, 2)
                if coverage.window_start_gap_seconds is not None
                else None
            ),
            "window_end_gap_seconds": (
                round(coverage.window_end_gap_seconds, 2)
                if coverage.window_end_gap_seconds is not None
                else None
            ),
            "verdict_met": coverage.covered,
        },
        "ac1": {
            "min_attempts": min_attempts,
            "thrash_hours": ac1.thrash_hours,
            "lease_active_fallbacks": ac1.lease_active_fallbacks,
            "hours_checked": len(hour_records),
            "verdict_met": ac1.clean,
        },
        "ac2": {
            "recovery_bound_seconds": recovery_bound,
            "pairs": [
                {
                    "session": r.session,
                    "deadline": r.deadline_ts.isoformat(sep=" "),
                    "release": r.release_ts.isoformat(sep=" "),
                    "release_seconds": round(r.release_seconds, 2),
                }
                for r in result.releases
            ],
            "unreleased_sessions": result.unreleased_sessions,
            "max_release_seconds": round(ac2.max_release_seconds, 2)
            if ac2.max_release_seconds is not None
            else None,
            "verdict_met": ac2.within_bound,
        },
        "ac3": {
            "resume_window_seconds": resume_window,
            "local_finishes_after_last_release": finishes_after,
            "verdict_met": ac3,
        },
        "all_verdicts_met": (
            ac1.clean and ac2.within_bound and ac3 and coverage.covered
        ),
    }
    return report


def _render(report: dict) -> str:
    a1, a2, a3 = report["ac1"], report["ac2"], report["ac3"]
    cov = report["coverage"]
    out = []
    out.append("Prefill-deadline recovery verification (LP-0MVKUKGER000TW2W)")
    out.append("=" * 66)
    out.append(f"window  : {report['window']['since'] or '(start)'} .. "
               f"{report['window']['until'] or '(end)'}")
    out.append("")
    out.append("Log coverage (LP-0MVLHMAWV008EKTI):")
    out.append(f"  files in window                    : {len(cov['files'])}")
    if not cov["files"]:
        out.append("  !! no parseable logs in the window -- measurement is impossible")
    for gap in cov["gaps"]:
        out.append(
            f"  !! gap {gap['before_file']} (ends {gap['before_last']}) -> "
            f"{gap['after_file']} (starts {gap['after_first']}): "
            f"{gap['gap_seconds']}s > {cov['max_gap_seconds']}s"
        )
    out.append(
        f"  verdict                          : "
        f"{'MET' if cov['verdict_met'] else 'NOT MET -- window has gaps'}"
    )
    out.append("")
    out.append("AC1 (report figures):")
    out.append(f"  hours checked                    : {a1['hours_checked']}")
    out.append(f"  thrash hours (>= {a1['min_attempts']} attempts, 0 served): "
               f"{a1['thrash_hours'] or '(none)'}")
    out.append(f"  reason=local_lease_active fallbacks: {a1['lease_active_fallbacks']}")
    out.append(f"  verdict                          : {'MET' if a1['verdict_met'] else 'NOT MET'}")
    out.append("")
    out.append(f"AC2 (deadline -> lease release, bound {a2['recovery_bound_seconds']}s):")
    for pair in a2["pairs"]:
        out.append(
            f"  {pair['session']}: deadline {pair['deadline']} -> "
            f"release {pair['release']} ({pair['release_seconds']}s)"
        )
    out.append(f"  unreleased deadlines              : {a2['unreleased_sessions'] or '(none)'}")
    out.append(f"  max release seconds               : {a2['max_release_seconds']}")
    out.append(f"  verdict                           : {'MET' if a2['verdict_met'] else 'NOT MET'}")
    out.append("")
    out.append(f"AC3 (local decode resumes, window {a3['resume_window_seconds']}s):")
    out.append(f"  local finishes after last release : {a3['local_finishes_after_last_release']}")
    out.append(f"  verdict                           : {'MET' if a3['verdict_met'] else 'NOT MET'}")
    out.append("")
    out.append(f"ALL VERDICTS                        : "
               f"{'MET' if report['all_verdicts_met'] else 'NOT MET'}")
    return "\n".join(out)


def _default_analysis_cmd() -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    candidate = (
        repo_root
        / ".pi"
        / "skills"
        / "proxy-usage-analysis"
        / "scripts"
        / "analyze_proxy_usage.py"
    )
    return [sys.executable, str(candidate)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="/var/log/llama-proxy")
    parser.add_argument("--since", default=None)
    parser.add_argument("--until", default=None)
    parser.add_argument("--min-attempts", type=int, default=10)
    parser.add_argument("--recovery-bound-seconds", type=int, default=300)
    parser.add_argument("--resume-window-seconds", type=int, default=3600)
    parser.add_argument(
        "--max-log-gap-seconds", type=float, default=3600.0,
        help="adjacent-file coverage gap considered a lost window (default 3600)",
    )
    parser.add_argument("--hour-records", default=None,
                        help="JSON file with the pipeline's hourly_activity "
                             "records (skips running the pipeline)")
    parser.add_argument("--no-pipeline", action="store_true",
                        help="skip the pipeline: AC1 verdict only (no hour "
                             "records available)")
    parser.add_argument("--analysis-cmd", action="append", default=None,
                        help="analysis pipeline command (default: the "
                             "proxy-usage-analysis CLI)")
    parser.add_argument("--measure", action="store_true",
                        help="measure a log window (AC1-AC3)")
    parser.add_argument("--stall-test", action="store_true",
                        help="drive a deliberate stalled prefill live and "
                             "measure gate recovery (AC2)")
    parser.add_argument("--stream", action="store_true",
                        help="stall test with stream=true (deadline path; "
                             "default non-stream exercises the watchdog path)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3")
    parser.add_argument("--prompt-tokens", type=int, default=55000)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"error: log directory not found: {log_dir}", file=sys.stderr)
        return 1

    since = _parse_bound(args.since) if args.since else None
    until = _parse_bound(args.until) if args.until else None

    if args.stall_test:
        session_id = args.session_id or (
            f"prefill-recovery-{int(time.time())}"
        )
        try:
            return run_stall_test(
                base_url=args.base_url,
                model=args.model,
                prompt_tokens=args.prompt_tokens,
                session_id=session_id,
                log_dir=log_dir,
                stream=args.stream,
                recovery_bound=args.recovery_bound_seconds,
                json_out=args.json,
            )
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    # default mode: measure
    if args.no_pipeline:
        hour_records: list[dict] | None = []
    elif args.hour_records:
        with open(args.hour_records) as fh:
            hour_records = json.load(fh)
    else:
        hour_records = None  # resolved inside run_measure via pipeline

    if args.no_pipeline or args.hour_records:
        try:
            report = run_measure(
                log_dir=log_dir,
                since=since,
                until=until,
                hour_records=hour_records,
                analysis_cmd=[],
                min_attempts=args.min_attempts,
                recovery_bound=args.recovery_bound_seconds,
                resume_window=args.resume_window_seconds,
                max_log_gap_seconds=args.max_log_gap_seconds,
            )
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    else:
        try:
            report = run_measure(
                log_dir=log_dir,
                since=since,
                until=until,
                hour_records=None,
                analysis_cmd=(
                    args.analysis_cmd or _default_analysis_cmd()
                ),
                min_attempts=args.min_attempts,
                recovery_bound=args.recovery_bound_seconds,
                resume_window=args.resume_window_seconds,
                max_log_gap_seconds=args.max_log_gap_seconds,
            )
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_render(report))
    return 0 if report["all_verdicts_met"] else 2


if __name__ == "__main__":
    sys.exit(main())
