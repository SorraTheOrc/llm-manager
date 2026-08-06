#!/usr/bin/env python3
"""Stall-gap distribution analysis (LP-0MSF5IAXE005BG33 / LP-0MSF1PUM90099ZSW F5).

Computes, for the post-restart window in proxy.log:

- per-stall observed gap: the proxy's idle-timeout detection fires exactly
  ``upstream_idle_timeout_seconds`` after the last upstream chunk, so the
  observed gap == the configured timeout by construction. What varies is
  (a) whether the upstream was slow-but-alive (false stall) vs truly dead,
  inferred from the Tier-1 retry outcome, and (b) the stream composition
  (reasoning-only vs content-committed vs tool-calls) from the session
  recordings, which determines re-route eligibility (F2/F3).
- distribution of stream durations (Stream started → stall detected) and
  retry outcomes, split by provider.
- quantified impact: how many client-visible ``stall_after_content`` errors
  would be avoided by (a) the 240s raise alone and (b) the re-route behavior.

Usage:
    python3 proxy/docs/stall-gap-analysis-2026-08-05/extract_stall_gaps.py \
        [path-to-proxy.log] [path-to-session-recordings-dir]
"""

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

LOG_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "/var/log/llama-proxy/proxy.log")
RECORDINGS_DIR = Path(
    sys.argv[2] if len(sys.argv) > 2 else "/home/rgardler/projects/llm/proxy/session-recordings"
)

STALL_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - WARNING - "
    r"Upstream stall detected: idle timeout session=(?P<session>\S+) "
    r"provider=(?P<provider>\S+) model=(?P<model>\S+) timeout=(?P<timeout>[\d.]+)s"
)
AFTER_CONTENT_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - WARNING - "
    r"Upstream stall after content delivered: terminating stream without retry "
    r"session=(?P<session>\S+) provider=(?P<provider>\S+) model=(?P<model>\S+)"
)
RETRY_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - "
    r"Upstream stall: retrying session=(?P<session>\S+) provider=(?P<provider>\S+) "
    r"model=(?P<model>\S+) attempt=(?P<attempt>\d+) backoff=(?P<backoff>[\d.]+)s"
)
FINISHED_ERROR_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - "
    r"Stream finished: reason=error session=(?P<session>\S+) provider=(?P<provider>\S+) "
    r"model=(?P<model>\S+)"
)
STREAM_STARTED_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - "
    r"Stream started: provider=(?P<provider>\S+) model=(?P<model>\S+) session=(?P<session>\S+)"
)
RETRY_EXHAUSTED_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - WARNING - "
    r"Upstream stalled repeatedly"
)

TS_FMT = "%Y-%m-%d %H:%M:%S,%f"


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, TS_FMT)


def pct(vals, p):
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = min(len(vals) - 1, int(round((p / 100.0) * (len(vals) - 1))))
    return vals[idx]


def summarize(name: str, vals):
    if not vals:
        print(f"  {name}: n=0")
        return
    print(
        f"  {name}: n={len(vals)} min={min(vals):.1f} p50={median(vals):.1f} "
        f"p90={pct(vals, 90):.1f} p95={pct(vals, 95):.1f} max={max(vals):.1f}"
    )


def analyze_recording(session_id: str, before_ts=None):
    """Return dict with stream composition from the session recording's raw SSE.

    Picks the response recording whose timestamp is closest to (and before)
    *before_ts* (the stall detection time), so the composition reflects the
    stalled stream rather than a later one.

    Returns None if no recording is found for the session.
    """
    sdir = RECORDINGS_DIR / session_id
    if not sdir.is_dir():
        return None
    responses = sorted(sdir.glob("*response.json"))
    if not responses:
        return None
    if before_ts is not None:
        # Pick the recording closest to the stall (prefer the one just before).
        best = None
        best_delta = None
        for p in responses:
            try:
                ts = datetime.fromisoformat(p.stem.split("T")[0] + "T" + p.stem.split("T")[1].split("+")[0].split("-")[0] + "+00:00")
            except Exception:
                # Fall back to filename ordering
                ts = None
            if ts is None:
                continue
            delta = (before_ts.replace(tzinfo=None) - ts.replace(tzinfo=None)).total_seconds()
            if delta >= 0 and (best_delta is None or delta < best_delta):
                best = p
                best_delta = delta
        if best is not None:
            responses = [best]
    try:
        j = json.loads(responses[-1].read_text())
    except Exception:
        return None
    payload = j.get("payload")
    if not isinstance(payload, str):
        return None
    n_chunks = payload.count("data: {")
    # Non-null, non-empty content: match "content":"text" (not null/true/false).
    has_content = bool(re.search(r'"content"\s*:\s*"[^"]+', payload))
    has_reasoning = "reasoning_content" in payload
    has_tool_calls = "tool_calls" in payload
    has_error = bool(re.search(r'"finish_reason"\s*:\s*"error"', payload))
    has_stop = '"finish_reason":"stop"' in payload or '"finish_reason": "stop"' in payload
    return {
        "n_chunks": n_chunks,
        "has_content": has_content,
        "has_reasoning": has_reasoning,
        "has_tool_calls": has_tool_calls,
        "has_error": has_error,
        "has_stop": has_stop,
    }


def main():
    print("# Stall-gap distribution analysis (LP-0MSF5IAXE005BG33)")
    print(f"\nLog: {LOG_PATH}")
    print(f"Recordings dir: {RECORDINGS_DIR}\n")

    stalls = []  # dicts: ts, session, provider, model, timeout, after_content, retries, recovered
    retry_events = defaultdict(list)
    finished_errors = []
    stream_starts = defaultdict(list)

    for line in LOG_PATH.read_text(errors="replace").splitlines():
        m = STALL_RE.match(line)
        if m:
            stalls.append({
                "ts": parse_ts(m.group("ts")),
                "session": m.group("session"),
                "provider": m.group("provider"),
                "model": m.group("model"),
                "timeout": float(m.group("timeout")),
                "after_content": False,
                "retries": 0,
                "recovered": False,
            })
            continue
        m = AFTER_CONTENT_RE.match(line)
        if m:
            # The after-content termination is the terminal client-visible error.
            # The stall detection line precedes it; match by session+provider.
            for st in reversed(stalls):
                if st["session"] == m.group("session") and not st["after_content"]:
                    st["after_content"] = True
                    break
            continue
        m = RETRY_RE.match(line)
        if m:
            retry_events[m.group("session")].append(
                (parse_ts(m.group("ts")), int(m.group("attempt")), float(m.group("backoff")))
            )
            for st in reversed(stalls):
                if st["session"] == m.group("session") and not st["retries"]:
                    st["retries"] = 1
                    break
            continue
        m = FINISHED_ERROR_RE.match(line)
        if m:
            finished_errors.append({
                "ts": parse_ts(m.group("ts")),
                "session": m.group("session"),
                "provider": m.group("provider"),
                "model": m.group("model"),
            })
            continue
        m = STREAM_STARTED_RE.match(line)
        if m:
            stream_starts[m.group("session")].append(parse_ts(m.group("ts")))

    # Mark session_continued: the session had a later stream start after the
    # stall (upstream reachable again / the session kept going). This is
    # evidence the upstream was slow-but-alive rather than permanently dead.
    error_sessions = {e["session"] for e in finished_errors}
    for st in stalls:
        later_starts = [s for s in stream_starts.get(st["session"], []) if s > st["ts"]]
        st["session_continued"] = bool(later_starts)
        st["client_visible_error"] = st["session"] in error_sessions
        st["recovered"] = st["session_continued"]

    print("=" * 72)
    print("## 1. Stall events (post-restart window)")
    print("=" * 72)
    print(f"\nTotal stall detections: {len(stalls)}")
    print(f"  after-content terminations: {sum(1 for s in stalls if s['after_content'])}")
    print(f"  with Tier-1 retry: {sum(1 for s in stalls if s['retries'])}")
    print(f"  session continued after stall (upstream slow-but-alive): {sum(1 for s in stalls if s['session_continued'])}")
    print(f"Client-visible Stream finished reason=error: {len(finished_errors)}")

    print("\n### Per-stall observed gap")
    print("The proxy's idle detection fires exactly `upstream_idle_timeout_seconds`")
    print("after the last upstream chunk (asyncio.wait_for), so the observed gap is")
    print("**by construction** equal to the configured timeout:")
    timeouts = Counter(s["timeout"] for s in stalls)
    for t, c in sorted(timeouts.items()):
        print(f"  timeout={t:.0f}s: {c} stalls")

    print("\n### Stream duration (Stream started → stall detected)")
    durations = []
    for st in stalls:
        starts = [s for s in stream_starts.get(st["session"], []) if s <= st["ts"]]
        if starts:
            start = max(starts)
            durations.append((st["ts"] - start).total_seconds())
    summarize("duration (s)", durations)

    print("\n### Retry backoff timeline (per session)")
    for session, events in sorted(retry_events.items()):
        ev = ", ".join(f"a={a}@{b}s" for _, a, b in sorted(events))
        print(f"  {session[:20]}: {ev}")

    # Provider split
    print("\n### By provider")
    by_provider = defaultdict(list)
    for st in stalls:
        by_provider[st["provider"]].append(st)
    for prov, items in sorted(by_provider.items()):
        print(f"  {prov}: {len(items)} stalls, "
              f"{sum(1 for s in items if s['after_content'])} after-content, "
              f"{sum(1 for s in items if s['session_continued'])} session-continued")

    # =====================================================================
    print("\n" + "=" * 72)
    print("## 2. Stream composition from session recordings (re-route eligibility)")
    print("=" * 72)
    comp_counts = Counter()
    reroute_eligible = 0
    reroute_eligible_after_content = 0
    analyzed = 0
    for st in stalls:
        rec = analyze_recording(st["session"], before_ts=st["ts"])
        if rec is None:
            comp_counts["no_recording"] += 1
            continue
        analyzed += 1
        if rec["has_content"]:
            comp_counts["content_committed"] += 1
        elif rec["has_tool_calls"]:
            comp_counts["tool_calls_only"] += 1
        elif rec["has_reasoning"]:
            comp_counts["reasoning_only"] += 1
            reroute_eligible += 1
            if st["after_content"]:
                reroute_eligible_after_content += 1
        else:
            comp_counts["empty"] += 1
    print(f"\nRecordings analyzed: {analyzed}")
    for k, v in comp_counts.most_common():
        print(f"  {k}: {v}")
    print(f"\nRe-route eligible (reasoning-only, zero content, zero tool_calls): {reroute_eligible}")
    print(f"  of which client-visible after-content terminations: {reroute_eligible_after_content}")

    # =====================================================================
    print("\n" + "=" * 72)
    print("## 3. Impact quantification (AC4)")
    print("=" * 72)
    total_client_errors = len(finished_errors)
    print(f"\nClient-visible stall_after_content errors: {total_client_errors}")
    print("(a) 240s raise alone: avoids stalls where the upstream resumed between")
    print("    120s and 240s. Direct evidence is unavailable (the proxy terminates")
    print(f"    at the timeout); proxy evidence: {sum(1 for s in stalls if s['session_continued'])} of")
    print(f"    {len(stalls)} stalls were in sessions that continued afterward (slow-but-alive")
    print("    upstream), supporting a longer timeout.")
    print("(b) re-route behavior (F2/F3): avoids reasoning-only stalls — the client")
    print("    receives the next provider's completion instead of an error.")
    print(f"    Estimated re-route-eligible after-content errors: {reroute_eligible_after_content}")

    # Recommendation
    print("\n" + "=" * 72)
    print("## 4. Recommendation (AC3)")
    print("=" * 72)
    if durations:
        p95 = pct(durations, 95)
        print(f"\nStream-duration p95 = {p95:.1f}s (n={len(durations)})")
        if p95 > 300:
            print("RECOMMENDATION: keep 240s (p95 stream duration exceeds 240s; raising")
            print("further would slow true-failure detection). Refine via F5 follow-ups.")
        else:
            print("RECOMMENDATION: 240s is reasonable; no further raise warranted based on")
            print("this window. The re-route behavior (F2/F3) is the primary mitigation.")

    # JSON output
    out = {
        "n_stalls": len(stalls),
        "n_after_content": sum(1 for s in stalls if s["after_content"]),
        "n_retried": sum(1 for s in stalls if s["retries"]),
        "n_session_continued": sum(1 for s in stalls if s["session_continued"]),
        "n_client_errors": len(finished_errors),
        "reroute_eligible_after_content": reroute_eligible_after_content,
        "composition": dict(comp_counts),
        "durations": durations,
    }
    out_path = Path(__file__).parent / "stall-gaps.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nJSON artifact: {out_path}")


if __name__ == "__main__":
    main()
