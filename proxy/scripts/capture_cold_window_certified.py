#!/usr/bin/env python3
"""Capture the certified window's cold-prefill evidence into a replayable fixture.

LP-0MWGAEBWJ00334W2 (parent LP-0MW957TAE000QD9N, cold-regime replay proof).

The certified window (2026-11-06 14:15:50 -> 2026-11-08 00:00) is the reference
for the deadline token capacity (LP-0MW957RB3000ER79). Its shipped cold-sample
source — the first-byte branch of ``router.proxy_to_local`` — cannot calibrate
the large-prompt buckets: a dispatch the prefill deadline cancels never emits a
first byte, so it can never enter the cold window, and the certified window's
*entire* completing cold population is 23 232-26 514 tokens (below the shipped
``COLD_MIN_PROCESSED_TOKENS = 32_000`` floor). Bucket 2 (65k+) therefore holds
zero samples and the capacity bound is inert (bucket 2 can never supply the
sample that would let it refuse its own prompts).

This script turns that window into a checked-in fixture the regression suite can
replay without reading production logs. Three record kinds:

``kind="abandoned"``
    One record per **abandoned** prefill (a llama-server task that never reached
    ``prompt processing done``) joined to the proxy eviction that cancelled it:
    ``total`` (the task's ``task.n_tokens``), ``cache`` (the reused prefix
    ``n_past``), ``processed`` (the position the prefill reached, from the
    ``slot release: ... stop processing: n_tokens`` line) and ``time_ms`` (the
    proxy's own measured ``elapsed_s`` for that dispatch, in milliseconds).
    These are exactly the four fields a ``prompt_progress`` event carries, i.e.
    what the peek in ``router.proxy_to_local`` observes in production. They are
    the new cold-sample source this item adds (option 1 of the work item: feed
    the window from aborted cold prefills).

``kind="completed"``
    One record per genuinely-cold **completed** prefill (measured reuse
    ``< 0.10``, position ``>= 32 000``) — the shipped first-byte sample source.
    ``time_ms`` is llama-server's own ``prompt eval time`` for the task and
    ``processed`` is the task's position at completion (``total``).

``kind="replay"``
    One record per *predicted-warm* deadline eviction (admission predicted a
    reused prefix ``>= 0.98`` yet the dispatch was still prefilling at the
    deadline). ``total`` is the proxy's own prompt estimate, ``priced_new_tokens``
    the admission-priced work. These rows are the AC2 replay population: with a
    calibrated cold window and an **unmeasured** residency basis (the shipped
    correction) the whole prompt is priced cold and must be refused by the
    deadline token capacity, with ``REASON_DEADLINE_CAPACITY``.

Matching method (documented because it is the one inferential step)
------------------------------------------------------------------

llama-server log lines carry no timestamps, so an abandoned task cannot be
joined to a proxy eviction by time. Each abandoned task is therefore matched to
the ``local_eviction_after_acceptance`` event whose prompt size is closest
(within :data:`EVIDENCE_SIZE_TOLERANCE` of the proxy estimate); ties prefer the
smaller delta. The delta is recorded per row (``evidence.delta_tokens``) so the
evidence quality stays auditable. A task with no eviction inside tolerance is
dropped rather than guessed at — an unmatched task's elapsed is unknown, and an
unknown elapsed can never price a rate.

Usage
-----

    python3 proxy/scripts/capture_cold_window_certified.py \
        --log-dir /var/log/llama-proxy \
        --out proxy/tests/fixtures/cold_window_certified_2026_11_06_08.jsonl

The output is deterministic for an unchanged log set (records are sorted).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

#: The certified window the parent item (and LP-0MW957TAE000QD9N) cites.
WINDOW_START = "2026-11-06 14:15:50"
WINDOW_END = "2026-11-08 00:00:00"

#: llama-server evidence match tolerance (fraction of the proxy estimate).
EVIDENCE_SIZE_TOLERANCE = 0.02

#: Sample predicates the shipped cold window enforces (mirrored here so the
#: fixture only carries rows the production code could ever record).
COLD_MAX_CACHED_RATIO = 0.10
COLD_MIN_PROCESSED_TOKENS = 32_000

#: A predicted reuse at/above this ratio is "priced as fully cached" — the
#: parent item's predicted-warm definition.
PREDICTED_WARM_REUSE = 0.98

#: Eviction reasons that abandon a prefill while it is still running: the
#: deadline detour plus the no-progress/hard ceiling watchdog. The measured
#: cache-deficit early abort is deliberately **not** a sample source (it prices
#: the deficit itself before the prefill is abandoned).
ABANDON_REASONS = ("prefill_deadline", "no_progress_watchdog")

EVICTION_RE = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ - \w+ - "
    r"local_eviction_after_acceptance session=(?P<session>\S+) "
    r"reason=(?P<reason>\S+) estimated_tokens=(?P<estimate>\d+) "
    r"new_tokens=(?P<new_tokens>\d+) eta_prefill_s=(?P<eta>[\d.]+) "
    r"elapsed_s=(?P<elapsed>[\d.]+)"
)

NEW_PROMPT_RE = re.compile(
    r"^\[(?P<pid>\d+)\] slot update_slots: id\s+(?P<slot>\d+) \| "
    r"task (?P<task>\d+) \| new prompt, n_ctx_slot = \d+, n_keep = \d+, "
    r"task\.n_tokens = (?P<total>\d+)"
)
INITIAL_N_RE = re.compile(
    r"^\[(?P<pid>\d+)\] slot update_slots: id\s+(?P<slot>\d+) \| "
    r"task (?P<task>\d+) \| n_tokens = (?P<n>\d+), memory_seq_rm"
)
PROGRESS_RE = re.compile(
    r"^\[(?P<pid>\d+)\] slot update_slots: id\s+(?P<slot>\d+) \| "
    r"task (?P<task>\d+) \| prompt processing progress, n_tokens = (?P<n>\d+)"
)
DONE_RE = re.compile(
    r"^\[(?P<pid>\d+)\] slot update_slots: id\s+(?P<slot>\d+) \| "
    r"task (?P<task>\d+) \| prompt processing done, n_tokens = (?P<n>\d+)"
)
RELEASE_RE = re.compile(
    r"^\[(?P<pid>\d+)\] slot\s+release: id\s+(?P<slot>\d+) \| "
    r"task (?P<task>\d+) \| stop processing: n_tokens = (?P<n>\d+)"
)
PROMPT_EVAL_RE = re.compile(
    r"^\[(?P<pid>\d+)\] prompt eval time =\s+(?P<ms>[\d.]+) ms /\s+"
    r"(?P<tokens>\d+) tokens"
)


def _open_text(path: Path):
    """Open a plain or gzipped log file for text reading."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open("rt", errors="replace")


def _log_files(log_dir: Path, prefix: str) -> list[Path]:
    return sorted(
        path
        for path in log_dir.iterdir()
        if path.name.startswith(prefix) and path.is_file()
    )


def load_evictions(log_dir: Path) -> list[dict[str, Any]]:
    """All ``local_eviction_after_acceptance`` events inside the window.

    Rotated files overlap at their boundaries, so records are deduplicated by
    ``(timestamp, session, estimate, new_tokens)``.
    """
    events: dict[tuple, dict[str, Any]] = {}
    for path in _log_files(log_dir, "proxy.log"):
        try:
            with _open_text(path) as handle:
                for line in handle:
                    match = EVICTION_RE.match(line)
                    if match is None:
                        continue
                    ts = match.group("ts")
                    if not (WINDOW_START <= ts <= WINDOW_END):
                        continue
                    key = (
                        ts,
                        match.group("session"),
                        match.group("estimate"),
                        match.group("new_tokens"),
                    )
                    events[key] = {
                        "ts": ts,
                        "session": match.group("session"),
                        "reason": match.group("reason"),
                        "estimated_tokens": int(match.group("estimate")),
                        "new_tokens": int(match.group("new_tokens")),
                        "eta_prefill_s": float(match.group("eta")),
                        "elapsed_s": float(match.group("elapsed")),
                    }
        except OSError as exc:  # pragma: no cover - unreadable rotation
            print(f"warning: skipping {path}: {exc}")
    return sorted(events.values(), key=lambda event: event["ts"])


def _llama_log_files(log_dir: Path) -> list[Path]:
    """The llama-server logs covering the window's calendar days.

    llama-server lines carry no timestamps, so the dated rotation is the only
    time restriction available.
    """
    first_day = WINDOW_START[:10]
    last_day = WINDOW_END[:10]
    selected: list[Path] = []
    for path in _log_files(log_dir, "llama-server.log"):
        date_match = re.search(r"llama-server\.log-(\d{4}-\d\d-\d\d)", path.name)
        if date_match is None:
            continue
        if first_day <= date_match.group(1) <= last_day:
            selected.append(path)
    return selected


def load_llama_tasks(log_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """Per-task prompt size, reused prefix, final position and completion flag.

    A ``(pid, task)`` pair is unique per dispatch; the same task can appear in
    several rotated files, so the records are merged by that key.
    """
    tasks: dict[tuple[int, int], dict[str, Any]] = {}
    for path in _llama_log_files(log_dir):
        try:
            with _open_text(path) as handle:
                for line in handle:
                    match = NEW_PROMPT_RE.match(line)
                    if match is not None:
                        _task_of(tasks, match)["total"] = int(match["total"])
                        continue
                    match = INITIAL_N_RE.match(line)
                    if match is not None:
                        _task_of(tasks, match)["cache"] = int(match["n"])
                        continue
                    match = PROGRESS_RE.match(line)
                    if match is not None:
                        _task_of(tasks, match)["last_progress"] = int(match["n"])
                        continue
                    match = DONE_RE.match(line)
                    if match is not None:
                        _task_of(tasks, match)["done"] = True
                        continue
                    match = RELEASE_RE.match(line)
                    if match is not None:
                        _task_of(tasks, match)["release"] = int(match["n"])
                        continue
                    match = PROMPT_EVAL_RE.match(line)
                    if match is not None:
                        _attach_eval(
                            tasks,
                            int(match["pid"]),
                            float(match["ms"]),
                            int(match["tokens"]),
                        )
        except OSError as exc:  # pragma: no cover - unreadable rotation
            print(f"warning: skipping {path}: {exc}")
    return tasks


def _task_of(
    tasks: dict[tuple[int, int], dict[str, Any]], match: re.Match[str]
) -> dict[str, Any]:
    """The task record for a parsed llama-server line (created on first sight)."""
    return tasks.setdefault((int(match["pid"]), int(match["task"])), _empty_task())


def _empty_task() -> dict[str, Any]:
    return {
        "total": None,
        "cache": None,
        "last_progress": None,
        "done": False,
        "release": None,
        "eval_ms": None,
        "eval_tokens": None,
    }


def _attach_eval(
    tasks: dict[tuple[int, int], dict[str, Any]],
    pid: int,
    ms: float,
    tokens: int,
) -> None:
    """Attach a ``prompt eval time`` line to the task that produced it.

    llama-server writes the line after the task's last progress line and before
    its release line, so the most recently started not-yet-scored task of that
    pid is the producer. The pairing is *validated* (the reported token count
    cannot exceed the task's fresh work) and dropped otherwise, so a mis-ordered
    rotation can never attach a timing to the wrong prompt.
    """
    candidates = [
        record
        for (task_pid, _task), record in tasks.items()
        if task_pid == pid and record["done"] and record["eval_ms"] is None
    ]
    for record in reversed(candidates):
        total, cache = record["total"], record["cache"]
        if total is None or cache is None:
            continue
        if 0 < tokens <= max(1, total - cache):
            record["eval_ms"] = ms
            record["eval_tokens"] = tokens
            return


def _closest_eviction(
    evictions: Iterable[dict[str, Any]], total: int
) -> dict[str, Any] | None:
    """The eviction whose proxy prompt estimate is closest to *total* (2 %)."""
    best = None
    best_delta = None
    for event in evictions:
        delta = abs(event["estimated_tokens"] - total)
        if best is None or delta < best_delta:
            best, best_delta = event, delta
    if best is None or best_delta is None:
        return None
    tolerance = EVIDENCE_SIZE_TOLERANCE * max(1, best["estimated_tokens"])
    if best_delta > tolerance:
        return None
    return best


def build_fixture(
    evictions: list[dict[str, Any]],
    tasks: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the abandoned / completed / replay records."""
    abandonable = [e for e in evictions if e["reason"] in ABANDON_REASONS]

    abandoned: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    for (pid, task_id), task in tasks.items():
        total = task["total"]
        cache = task["cache"]
        if total is None or cache is None or total <= 0:
            continue
        position = task["release"] if task["release"] is not None else task["last_progress"]
        if position is None:
            continue
        ratio = cache / total
        if task["done"]:
            if (
                ratio < COLD_MAX_CACHED_RATIO
                and total >= COLD_MIN_PROCESSED_TOKENS
                and task["eval_ms"] is not None
            ):
                completed.append(
                    {
                        "kind": "completed",
                        "total": total,
                        "cache": cache,
                        "processed": total,
                        "time_ms": round(float(task["eval_ms"]), 1),
                        "measured_cached_ratio": round(ratio, 4),
                        "evidence": {
                            "source": "llama_server.prompt_eval_time",
                            "pid": pid,
                            "task": task_id,
                            "eval_tokens": task["eval_tokens"],
                        },
                    }
                )
            continue
        event = _closest_eviction(abandonable, total)
        if event is None:
            continue
        abandoned.append(
            {
                "kind": "abandoned",
                "total": total,
                "cache": cache,
                "processed": position,
                "time_ms": round(event["elapsed_s"] * 1000.0, 1),
                "measured_cached_ratio": round(ratio, 4),
                "evidence": {
                    "source": "llama_server.slot_release",
                    "pid": pid,
                    "task": task_id,
                    "delta_tokens": abs(event["estimated_tokens"] - total),
                    "eviction_ts": event["ts"],
                    "eviction_session": event["session"],
                    "eviction_reason": event["reason"],
                },
            }
        )

    replay: list[dict[str, Any]] = []
    for event in evictions:
        total = event["estimated_tokens"]
        if total <= 0:
            continue
        predicted_reuse = 1.0 - event["new_tokens"] / total
        if predicted_reuse < PREDICTED_WARM_REUSE:
            continue
        replay.append(
            {
                "kind": "replay",
                "total": total,
                "priced_new_tokens": event["new_tokens"],
                "predicted_reuse_ratio": round(predicted_reuse, 4),
                "eta_prefill_s": event["eta_prefill_s"],
                "elapsed_s": event["elapsed_s"],
                "evidence": {
                    "eviction_ts": event["ts"],
                    "eviction_session": event["session"],
                    "eviction_reason": event["reason"],
                },
            }
        )

    abandoned.sort(key=lambda r: (r["total"], r["processed"], r["time_ms"]))
    completed.sort(key=lambda r: (r["total"], r["time_ms"]))
    replay.sort(key=lambda r: (r["total"], r["priced_new_tokens"]))

    header = {
        "_meta": {
            "captured_by": "proxy/scripts/capture_cold_window_certified.py",
            "work_item": "LP-0MWGAEBWJ00334W2",
            "parent": "LP-0MW957TAE000QD9N",
            "window": {"start": WINDOW_START, "end": WINDOW_END},
            "source": "/var/log/llama-proxy/{proxy,llama-server}.log*",
            "cold_predicate": {
                "max_cached_ratio": COLD_MAX_CACHED_RATIO,
                "min_processed_tokens": COLD_MIN_PROCESSED_TOKENS,
            },
            "matching_note": (
                "llama-server lines carry no timestamps, so an abandoned task is "
                "matched to the local_eviction_after_acceptance event with the "
                f"closest prompt size (<= {EVIDENCE_SIZE_TOLERANCE:.0%}); the "
                "per-row delta_tokens records the match quality. A task with no "
                "event inside tolerance is dropped rather than guessed at."
            ),
            "field_note": (
                "total/cache/processed/time_ms are the four prompt_progress "
                "fields the proxy observes; processed is the position the "
                "prefill reached (it includes the reused prefix) and time_ms is "
                "the dispatch's own measured elapsed."
            ),
            "counts": {
                "evictions": len(evictions),
                "abandonable_evictions": len(abandonable),
                "abandoned": len(abandoned),
                "completed": len(completed),
                "replay": len(replay),
            },
        }
    }
    return {
        "header": header,
        "records": abandoned + completed + replay,
    }


def write_fixture(fixture: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(fixture["header"], sort_keys=True) + "\n")
        for record in fixture["records"]:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="/var/log/llama-proxy", type=Path)
    parser.add_argument(
        "--out",
        default=Path(
            "proxy/tests/fixtures/cold_window_certified_2026_11_06_08.jsonl"
        ),
        type=Path,
    )
    args = parser.parse_args(argv)

    evictions = load_evictions(args.log_dir)
    tasks = load_llama_tasks(args.log_dir)
    fixture = build_fixture(evictions, tasks)
    write_fixture(fixture, args.out)
    print(
        json.dumps(
            {"out": str(args.out), **fixture["header"]["_meta"]["counts"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
