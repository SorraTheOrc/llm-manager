#!/usr/bin/env python3
"""Capture the cold-prefill thrash-window fixture (LP-0MVN4LAV9007PM2X).

The fixture recorded by this tool is the executable baseline for the parent
item (LP-0MVN3KMGI009BZ0D, AC4): it records every local admission attempt in a
measured thrash window, joined with the admission verdict the proxy logged and
the outcome the dispatch actually had, so later tests can replay it through
:func:`proxy.prefill_eta.evaluate_admission` and assert post-change behaviour
without re-parsing gigabytes of production logs.

Usage (regenerating the checked-in fixture)::

    python3 proxy/scripts/capture_prefill_thrash_window.py \\
        --start "2026-10-24 18:38" --end "2026-10-24 23:47" \\
        --log /var/log/llama-proxy/proxy.log.2026-10-24_17 \\
        --log /var/log/llama-proxy/proxy.log-2026-10-25_00 \\
        --out proxy/tests/fixtures/prefill_thrash_2026_10_24.jsonl

Rotated ``.gz`` logs are accepted directly (they are decompressed on the fly),
so the same command works after log rotation:

    python3 proxy/scripts/capture_prefill_thrash_window.py --start ... --end ... \\
        --log /var/log/llama-proxy/proxy.log-2026-10-24_17.gz --out ...

Join rules (deterministic, no heuristics beyond these):

* **Attempt population** — one record per local admission attempt:
  ``routing_check`` (the queue/contended path), ``routing_idle_admit`` (the
  economic/idle bypass) and ``routing_skip_local`` (refused by the route, e.g.
  ``large_context_bypass``). Each carries ``estimated_tokens``,
  ``cached_ratio``, ``new_tokens`` and ``cache_state`` as logged.
* **Admission verdict** — the most recent ``local_admission_guard`` line for
  the same session at or before the attempt (``verdict``/``reason``). When the
  guard never logged for that session the verdict is ``unknown`` — the guard
  was inert (uncalibrated bucket, policy off, or no budget) and logging is
  suppressed for inert decisions by design.
* **Outcome** — classified from the session's subsequent lines, in order:
  ``prefill_deadline_exceeded`` when ``prefill_deadline_exceeded`` /
  ``lease_released reason=prefill_deadline`` follows the attempt before the
  session's next attempt; ``first_byte_ok`` when ``dispatch_first_byte_ms``
  follows; ``detoured`` when ``routing_hop`` follows; otherwise ``other``.
* **Prefill tokens computed before cancellation** — ``null``. Per-slot prefill
  progress was unobservable throughout the window
  (``prefill_progress_unobservable``, ``progress=None``/``0`` on every
  ``lease_extended_during_prefill`` and ``prefill_wait_exceeds_threshold``
  line), so any number here would be invented. The replay harness therefore
  models the discarded volume as ``min(new_tokens, deadline_s * band_rate)``
  from the pinned calibration profile, and the fixture records the inputs.

No prompt bodies, model text or credentials are copied: only sizes, ratios,
verdicts and outcomes.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

#: Pinned per-band prompt-eval rates (tokens/second) measured on this host,
#: 2026-10-24/25 (268 llama-server samples). The replay harness prices with
#: these so the fixture is independent of live calibration drift.
BAND_RATES: tuple[tuple[int, float], ...] = (
    (10_000, 281.0),
    (20_000, 281.0),
    (30_000, 249.0),
    (40_000, 181.0),
    (50_000, 152.0),
    (10**9, 114.0),
)

_LINE_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ - (\w+) - (.*)$")
_SESSION_RE = re.compile(r"session=(\S+)")
_EST_RE = re.compile(r"estimated_tokens=(\d+)")
_NEW_RE = re.compile(r"new_tokens=(\d+)")
_RATIO_RE = re.compile(r"cached_ratio=([\d.]+)")
_CACHE_RE = re.compile(r"cache_state=(\w+)")
_REASON_RE = re.compile(r"reason=(\S+)")
_DECISION_RE = re.compile(r"decision=(\w+)")
_GUARD_REASON_RE = re.compile(r"reason=(eta_exceeds_budget\w*|within_budget|"
                             r"prefill_eta_exceeds_deadline|policy_off|"
                             r"insufficient_samples|no_budget)")

ATTEMPT_KINDS = ("routing_check", "routing_idle_admit", "routing_skip_local")


def band_rate(prompt_tokens: int) -> float:
    """Pinned prompt-eval rate (tokens/s) for a prompt of *prompt_tokens*."""
    for upper, rate in BAND_RATES:
        if prompt_tokens <= upper:
            return rate
    return BAND_RATES[-1][1]


def _iter_lines(paths: list[str], start: str, end: str):
    """Yield ``(ts, message)`` from *paths* restricted to ``start..end``."""
    for raw in paths:
        path = Path(raw)
        opener = gzip.open if path.suffix == ".gz" else open
        try:
            handle = opener(path, "rt", errors="replace")
        except OSError as exc:  # pragma: no cover - operator-facing
            raise SystemExit(f"cannot read log {path}: {exc}") from exc
        with handle:
            for line in handle:
                match = _LINE_RE.match(line.rstrip("\n"))
                if not match:
                    continue
                ts = match.group(1)
                if start <= ts <= end:
                    yield ts, match.group(3)


def _int(pattern: re.Pattern[str], text: str, default: int = 0) -> int:
    match = pattern.search(text)
    if not match:
        return default
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return default


def _float(pattern: re.Pattern[str], text: str, default: float = 0.0) -> float:
    match = pattern.search(text)
    if not match:
        return default
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return default


def _session(text: str) -> str:
    match = _SESSION_RE.search(text)
    return match.group(1) if match else ""


def capture(paths: list[str], start: str, end: str) -> tuple[dict, list[dict]]:
    """Build the fixture header and per-attempt records."""
    attempts: list[dict] = []
    pending: list[dict] = []  # attempts awaiting an outcome, per session
    last_guard: dict[str, dict] = {}

    def settle(session: str, outcome: str) -> None:
        # Settle the most recent *unsettled* attempt for this session: a
        # session repeats the same prompt many times in a thrash window, and
        # each repetition has its own outcome.
        for record in reversed(pending):
            if record["session"] == session and record["outcome"] == "other":
                record["outcome"] = outcome
                return

    for ts, message in _iter_lines(paths, start, end):
        kind = message.split(":", 1)[0].split(" ", 1)[0]
        session = _session(message)
        if kind == "local_admission_guard":
            guard_reason = _GUARD_REASON_RE.search(message)
            last_guard[session] = {
                "verdict": _DECISION_RE.search(message).group(1)
                if _DECISION_RE.search(message)
                else "unknown",
                "reason": guard_reason.group(1) if guard_reason else "unknown",
                "ts": ts,
            }
            continue
        if kind in ATTEMPT_KINDS:
            if not _EST_RE.search(message):
                continue
            if kind == "routing_check":
                route = "queue"
            elif kind == "routing_idle_admit":
                route = "idle_bypass"
            else:
                route = "large_context_bypass"
            estimated = _int(_EST_RE, message)
            guard = last_guard.get(session)
            attempts.append(
                {
                    "ts": ts,
                    "session": session,
                    "model": "Qwen3",
                    "route": route,
                    "estimated_tokens": estimated,
                    "cached_ratio": _float(_RATIO_RE, message),
                    "new_tokens": _int(_NEW_RE, message, default=estimated),
                    "cache_state": (_CACHE_RE.search(message).group(1)
                                    if _CACHE_RE.search(message) else "unknown"),
                    "guard_verdict": guard["verdict"] if guard else "unknown",
                    "guard_reason": guard["reason"] if guard else "unknown",
                    "outcome": "other",
                    "prefill_tokens_computed": None,
                }
            )
            pending.append(attempts[-1])
            continue
        if kind == "prefill_deadline_exceeded" or (
            kind == "lease_released" and "reason=prefill_deadline" in message
        ):
            settle(session, "prefill_deadline_exceeded")
        elif kind == "dispatch_first_byte_ms":
            settle(session, "first_byte_ok")
        elif kind == "routing_hop":
            settle(session, "detoured")

    deadline_attempts = [
        a for a in attempts if a["outcome"] == "prefill_deadline_exceeded"
    ]
    # A *task* is one distinct (session, prompt size) prefill: repeating the
    # same prompt in the same session re-prefills it from scratch, so distinct
    # tasks are the unit the waste model costs.
    task_new_tokens: dict[tuple[str, int], int] = {}
    for attempt in deadline_attempts:
        task_new_tokens.setdefault(
            (attempt["session"], attempt["estimated_tokens"]),
            attempt["new_tokens"],
        )
    wasted_model = sum(
        min(new, _deadline_model(new)) for new in task_new_tokens.values()
    )
    header = {
        "_meta": {
            "description": (
                "Cold-prefill thrash window capture for LP-0MVN3KMGI009BZ0D "
                "(parent AC4 baseline, child LP-0MVN4LAV9007PM2X AC1/AC2)."
            ),
            "window": {"start": start, "end": end},
            "source_logs": list(paths),
            "capture_command": (
                "python3 proxy/scripts/capture_prefill_thrash_window.py "
                f'--start "{start}" --end "{end}" '
                + " ".join(f"--log {p}" for p in paths)
                + " --out proxy/tests/fixtures/prefill_thrash_2026_10_24.jsonl"
            ),
            "band_rates_tokens_per_second": {
                str(upper if upper < 10**9 else "inf"): rate
                for upper, rate in BAND_RATES
            },
            "waste_model": (
                "min(new_tokens, local_prefill_deadline_seconds * band_rate) "
                "per distinct (session, estimated_tokens) cancelled task — the "
                "volume a 90 s deadline can compute before it cancels. Per-slot "
                "prefill progress was unobservable in this window "
                "(prefill_progress_unobservable, progress=None/0), so "
                "prefill_tokens_computed is null per record rather than guessed."
            ),
            "local_prefill_deadline_seconds": 90,
            "totals": {
                "attempts": len(attempts),
                "deadline_cancelled_attempts": len(deadline_attempts),
                "deadline_cancelled_tasks": len(task_new_tokens),
                "wasted_prefill_tokens_model": wasted_model,
                "idle_bypass_attempts": sum(
                    1 for a in attempts if a["route"] == "idle_bypass"
                ),
                "idle_bypass_over_25k_new": sum(
                    1
                    for a in attempts
                    if a["route"] == "idle_bypass" and a["new_tokens"] > 25_000
                ),
            },
            "privacy": (
                "sizes, ratios, verdicts and outcomes only — no prompt bodies, "
                "no model text, no credentials"
            ),
        }
    }
    return header, attempts


def _deadline_model(new_tokens: int) -> float:
    """Tokens a 90 s deadline can compute at the pinned band rate."""
    return 90.0 * band_rate(new_tokens)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="window start timestamp")
    parser.add_argument("--end", required=True, help="window end timestamp")
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        dest="logs",
        help="proxy log file (plain or .gz); repeat for each rotation file",
    )
    parser.add_argument("--out", required=True, help="output JSONL fixture path")
    args = parser.parse_args(argv)

    header, attempts = capture(args.logs, args.start, args.end)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for record in attempts:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    totals = header["_meta"]["totals"]
    print(
        f"wrote {len(attempts)} attempts "
        f"({totals['deadline_cancelled_attempts']} deadline-cancelled, "
        f"{totals['deadline_cancelled_tasks']} tasks, "
        f"~{totals['wasted_prefill_tokens_model']} wasted tokens modelled) "
        f"to {out} ({out.stat().st_size} bytes)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
