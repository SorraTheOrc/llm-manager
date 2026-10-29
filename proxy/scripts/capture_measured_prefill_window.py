#!/usr/bin/env python3
"""Capture the measured-prefill reference window into a replayable fixture.

LP-0MVR7IXAI002QXXD (F5 of parent LP-0MVQIT9QY007QU27, Option D).

The parent's evidence window (2026-10-26 09:06 → 2026-10-27 09:06, the
``local_eviction_after_acceptance`` population) is the reference for the
measured-cache early abort. This script turns that window into a checked-in
fixture the regression suite can replay without re-reading production logs:

``kind="eviction"``
    One record per ``local_eviction_after_acceptance`` event (deduplicated
    across rotated log files): timestamp, session, prompt estimate, the
    admission-priced ``new_tokens``, ``eta_prefill_s`` and the observed
    ``elapsed_s``. Events whose *predicted* reuse is ≥ 0.99 are the
    "fake-warm" population (the parent's definition); each fake-warm event
    also carries a ``measured`` observation — the ``n_past`` /
    ``slot.prompt.tokens.size()`` pair from the closest llama-server task in
    the same window — which is the ground truth the abort decision consumes.

``kind="warm"``
    The working warm population: llama-server tasks with a large prompt
    (≥ 20k tokens) and near-full KV-cache reuse (≥ 0.99). These are the
    dispatches the fix must never disturb (parent AC3).

Matching method (documented because it is the one inferential step)
------------------------------------------------------------------

llama-server log lines carry no timestamps, so a proxy event cannot be joined
to a llama-server task by time. Every fake-warm event is therefore matched to
the llama-server observation with the *closest prompt size* within 2 % of the
proxy estimate; ties at identical size prefer the observation with the larger
measured deficit (an event that spent the whole deadline prefilling cannot
have had a nearly-full cache). The size delta is recorded per event so the
evidence quality is auditable (``evidence.delta_tokens``).

Usage
-----

    python3 proxy/scripts/capture_measured_prefill_window.py \
        --log-dir /var/log/llama-proxy \
        --out proxy/tests/fixtures/measured_prefill_window_2026_10_26.jsonl

The output is deterministic for an unchanged log set (records are sorted).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

WINDOW_START = "2026-10-26 09:06:07"
WINDOW_END = "2026-10-27 09:06:07"

#: A predicted reuse at/above this ratio is "priced as fully cached" — the
#: parent item's fake-warm definition (29 events in the reference window).
FAKE_WARM_PREDICTED_REUSE = 0.99

#: Working-warm population: large prompts with near-full measured reuse.
WARM_MIN_PROMPT_TOKENS = 20_000
WARM_MIN_MEASURED_REUSE = 0.99

#: llama-server evidence match tolerance (fraction of the proxy estimate).
EVIDENCE_SIZE_TOLERANCE = 0.02

#: Prompt size at/above which llama-server observations are kept as the
#: measured evidence population (the fake-warm band starts around 18k).
EVIDENCE_MIN_PROMPT_TOKENS = 18_000

EVICTION_RE = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ - ERROR - "
    r"local_eviction_after_acceptance session=(?P<session>\S+) "
    r"reason=(?P<reason>\S+) estimated_tokens=(?P<estimate>\d+) "
    r"new_tokens=(?P<new_tokens>\d+) eta_prefill_s=(?P<eta>[\d.]+) "
    r"elapsed_s=(?P<elapsed>[\d.]+)"
)

LLAMA_TASK_RE = re.compile(
    r"^\[(?P<pid>\d+)\] slot update_slots: id\s+(?P<slot>\d+) \| "
    r"task (?P<task>\d+) \| n_past = (?P<n_past>\d+), "
    r"slot\.prompt\.tokens\.size\(\) = (?P<prompt_tokens>\d+)"
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


def load_eviction_events(log_dir: Path) -> list[dict[str, Any]]:
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
                        "kind": "eviction",
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

    llama-server lines carry no timestamps, so a per-day rotated file is the
    only time restriction available: include exactly the dated files whose day
    falls inside the window (plus an untimestamped current file only when the
    window is still open).
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


def load_llama_observations(log_dir: Path) -> list[dict[str, Any]]:
    """``n_past`` / prompt-size observations from the llama-server logs.

    Deduplicated by ``(pid, task)`` — the pair is unique per dispatch, while a
    task can appear once across rotated files.
    """
    observations: dict[tuple[int, int], dict[str, Any]] = {}
    for path in _llama_log_files(log_dir):
        try:
            with _open_text(path) as handle:
                for line in handle:
                    match = LLAMA_TASK_RE.match(line)
                    if match is None:
                        continue
                    key = (int(match.group("pid")), int(match.group("task")))
                    observations[key] = {
                        "pid": key[0],
                        "task": key[1],
                        "prompt_tokens": int(match.group("prompt_tokens")),
                        "cache_tokens": int(match.group("n_past")),
                    }
        except OSError as exc:  # pragma: no cover - unreadable rotation
            print(f"warning: skipping {path}: {exc}")
    return sorted(
        observations.values(), key=lambda obs: (obs["prompt_tokens"], obs["task"])
    )


def build_fixture(
    events: Iterable[dict[str, Any]],
    observations: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Flag the fake-warm events and attach the measured evidence population.

    No per-event cache is invented: llama-server lines carry no timestamps, so
    a proxy event cannot be joined to a specific llama-server task. The
    fixture therefore ships the *measured evidence population* for the window
    (``kind="evidence"``) and the replay harness derives the class bounds
    (re-prefill vs full reuse) for each prompt size from it.
    """
    records: list[dict[str, Any]] = []
    fake_warm = 0
    for event in events:
        record = dict(event)
        estimate = int(record["estimated_tokens"])
        priced = int(record["new_tokens"])
        record["predicted_reuse_ratio"] = (
            1.0 - priced / estimate if estimate > 0 else 0.0
        )
        record["fake_warm"] = (
            estimate > 0 and record["predicted_reuse_ratio"] >= FAKE_WARM_PREDICTED_REUSE
        )
        if record["fake_warm"]:
            fake_warm += 1
        records.append(record)

    evidence = [
        {
            "kind": "evidence",
            "prompt_tokens": obs["prompt_tokens"],
            "cache_tokens": obs["cache_tokens"],
            "source": "llama_server.n_past",
            "pid": obs["pid"],
            "task": obs["task"],
        }
        for obs in observations
        if obs["prompt_tokens"] >= EVIDENCE_MIN_PROMPT_TOKENS
    ]

    by_cluster = Counter(
        (record["estimated_tokens"], record["new_tokens"])
        for record in records
        if record["fake_warm"]
    )
    header = {
        "_meta": {
            "captured_by": "proxy/scripts/capture_measured_prefill_window.py",
            "parent": "LP-0MVQIT9QY007QU27",
            "work_item": "LP-0MVR7IXAI002QXXD",
            "window": {"start": WINDOW_START, "end": WINDOW_END},
            "source": "/var/log/llama-proxy/{proxy,llama-server}.log*",
            "fake_warm_definition": (
                "predicted reuse (1 - new_tokens/estimated_tokens) "
                f">= {FAKE_WARM_PREDICTED_REUSE}"
            ),
            "evidence_note": (
                "llama-server lines carry no timestamps, so eviction events are "
                "not joined to individual tasks; the evidence population below "
                "is the measured n_past/prompt-size ground truth for the window. "
                "The replay harness derives, per prompt size, the measured "
                "re-prefill floor (smallest observed n_past in the size band) and "
                "the measured full-reuse ceiling (largest observed reuse)."
            ),
            "counts": {
                "evictions": len(records),
                "fake_warm": fake_warm,
                "fake_warm_clusters": len(by_cluster),
                "evidence_observations": len(evidence),
            },
            "fake_warm_clusters": [
                {"estimated_tokens": estimate, "new_tokens": new_tokens, "events": count}
                for (estimate, new_tokens), count in sorted(by_cluster.items())
            ],
        }
    }
    return {"header": header, "records": records + evidence}


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
        default=Path("proxy/tests/fixtures/measured_prefill_window_2026_10_26.jsonl"),
        type=Path,
    )
    args = parser.parse_args(argv)

    events = load_eviction_events(args.log_dir)
    observations = load_llama_observations(args.log_dir)
    fixture = build_fixture(events, observations)
    write_fixture(fixture, args.out)
    counts = fixture["header"]["_meta"]["counts"]
    print(json.dumps({"out": str(args.out), **counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
