#!/usr/bin/env python3
"""Session-compaction quality experiment harness.

Execute the three-arm experiment described in
``docs/session-compaction-experiment-design.md`` (LP-0MSG9PUHU0059TTZ):
replay tasks drawn from logged sessions through arms A/B/C, score blind
on the rubric, and report go/no-go against the pre-registered quality bar.

Three arms:
  A — uncompacted prompt → remote ``deepseek-v4-flash`` (baseline)
  B — compacted prompt   → local Qwen3                (proposed)
  C — uncompacted prompt → local Qwen3                (ceiling)

Compaction strategies (from F3/F4):
  - fast profile: Strategy B — hard-cap auto-truncate to ≤ 38K when > 58.3K
  - cheap profile: Strategy C — summarize then truncate to ≤ 30K when > 43K

CLI::

    # Dry-run (analysis only, no API calls):
    python3 proxy/scripts/run_compaction_experiment.py \\
        --log-dir /var/log/llama-proxy \\
        --mode fast cheap \\
        --dry-run

    # Execute (requires API access):
    python3 proxy/scripts/run_compaction_experiment.py \\
        --log-dir /var/log/llama-proxy \\
        --mode fast cheap \\
        --output-dir experiment-results \\
        --deepseek-key "$DEEPSEEK_API_KEY" \\
        --local-endpoint http://192.168.0.199:8000

    # Re-score a previous run (from JSONL):
    python3 proxy/scripts/run_compaction_experiment.py \\
        --replay experiment-results/run-20260830.jsonl \\
        --score-only

Deliverables written on successful run:
  - ``<output-dir>/run-<timestamp>.jsonl``  — raw per-task results
  - ``<output-dir>/metrics.csv``           — aggregated metrics per arm/mode
  - ``<output-dir>/report.md``             — human-readable go/no-go report
  - ``<output-dir>/run-<timestamp>.json``  — full machine-readable report

This script is **eval-only** — no behaviour change, no dispatch.
"""  # noqa: E501, EXE001

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants (mirrors design doc §4 / F3/F4 thresholds)
# ---------------------------------------------------------------------------

# Lazy-initialised tiktoken encoder for token estimation
_TIKTOKEN_ENCODER = None

# Compaction triggers (0.70 × per-slot clamp)
FAST_TRIGGER = 58300   # 0.70 × 83285
CHEAP_TRIGGER = 43000  # 0.70 × 61440

# Compaction targets
FAST_TARGET = 38000    # local_large_context_cold_cache_threshold
CHEAP_TARGET = 30000

# Routing caps (Tier 2, LP-0MTBOX45O005LD1S)
FAST_CAP = 70000       # fast-mode hard cap
CHEAP_CAP = 61440      # cheap-mode hard cap

# Per-slot clamps (for mode classification from warm_threshold in logs)
FAST_WARM_THRESHOLD = 83285
CHEAP_WARM_THRESHOLD = 100000
# Tier 2 hard caps used as warm_threshold in the current deployment
# (LP-0MTBOX45O005LD1S) — fast 70000, cheap 61440.
FAST_CAP_WARM = FAST_CAP          # 70000
CHEAP_CAP_WARM = CHEAP_CAP        # 61440

# Pre-registered quality bar (§5)
QUALITY_BAR_RUBRIC_RATIO = 0.95      # B ≥ 0.95 × A
QUALITY_BAR_COMPLETION_DELTA = 3     # B ≥ A − 3pp
QUALITY_BAR_ALPHA = 0.05             # significance level

# Efficiency gate (§6)
EFFICIENCY_PREFILL_REDUCTION_MIN = 0.25  # ≥ 25% wasted-prefill reduction
EFFICIENCY_TTFT_WORSEN_PCT = 0.20        # TTFT P95 not > 20% worse

# Task suite minimums (§6)
TASKS_TRIGGER_CAP_BAND = 30   # per arm per mode
TASKS_EXTREME_BAND = 10       # per arm per mode

# Compaction strategy parameters
FAST_STRATEGY = "truncate"     # Strategy B: hard-cap auto-truncate
CHEAP_STRATEGY = "summarize"   # Strategy C: summarize then truncate

TS_FMT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Log parsing (reuses patterns from analyze_context_distribution.py)
# ---------------------------------------------------------------------------

ROUTING_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?routing_check .*?"
    r"estimated_tokens=(\d+) .*?warm_threshold=(\d+) .*?session=([A-Za-z0-9_.\-]+)"
)

PRESSURE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?context_pressure "
    r"session=([A-Za-z0-9_.\-]+) estimated_tokens=(\d+) per_slot_ctx=(\d+) "
    r"ratio=([\d.]+)"
)

SKIP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?routing_skip_local .*?"
    r"reason=(\w+) .*?session=([A-Za-z0-9_.\-]+)"
)

# Pattern to extract the request body from [local]/[remote] POST lines.
# The body is JSON embedded in the log line; we extract and parse it.
POST_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*?"
    r"\[(local|remote)\] POST .*?body=(\{.*?\}) .*?session_id=([A-Za-z0-9_.\-]+)"
)

# Pattern to extract Stream finished lines (for response tokens).
STREAM_FINISHED_RE = re.compile(
    r"Stream finished:.*?session=([A-Za-z0-9_.\-]+).*?tokens=(\d+)/(\d+)/(\d+)"
)

# Pattern to extract model from Stream started.
STREAM_STARTED_RE = re.compile(
    r"Stream started:.*?model=([A-Za-z0-9_\-]+).*?session=([A-Za-z0-9_.\-]+)"
)


class Mode(Enum):
    """Operating mode derived from warm_threshold."""
    FAST = "fast"
    CHEAP = "cheap"
    OTHER = "other"

    @classmethod
    def from_warm_threshold(cls, threshold: int) -> Mode:
        # Older deployments logged the per-slot clamp (83285/100000); the
        # current deployment logs the Tier 2 hard cap (70000/61440). Either
        # value classifies the mode.
        if threshold in (FAST_WARM_THRESHOLD, FAST_CAP_WARM):
            return cls.FAST
        if threshold in (CHEAP_WARM_THRESHOLD, CHEAP_CAP_WARM):
            return cls.CHEAP
        return cls.OTHER


@dataclass
class EstimateSample:
    """One routing-time context estimate for a session."""
    ts: datetime
    session: str
    estimated_tokens: int
    warm_threshold: int

    @property
    def mode(self) -> Mode:
        return Mode.from_warm_threshold(self.warm_threshold)


@dataclass
class RequestRecord:
    """A single request from the proxy log."""
    ts: datetime
    provider: str  # "local" or "remote"
    model: str
    session_id: str
    messages: list[dict]  # extracted from request body
    estimated_tokens: int | None = None
    response_tokens: int | None = None


@dataclass
class Task:
    """One replay task: a final turn from a session."""
    task_id: str
    session_id: str
    mode: Mode
    category: str  # "code", "qa", "agent", "reasoning"
    # Original (uncompacted) prompt history
    original_messages: list[dict]
    # Estimated tokens at the point of trigger breach
    estimated_tokens: int
    # The operator's final turn (the "target" to be answered)
    target_prompt: str
    # Context band classification
    band: str  # "trigger-cap" or "extreme"
    # Transcript source: "recording", "log", or "synthetic"
    transcript_src: str = "synthetic"

    @property
    def is_fast(self) -> bool:
        return self.mode == Mode.FAST

    @property
    def is_cheap(self) -> bool:
        return self.mode == Mode.CHEAP

    @property
    def trigger(self) -> int:
        return FAST_TRIGGER if self.is_fast else CHEAP_TRIGGER

    @property
    def target_tokens(self) -> int:
        return FAST_TARGET if self.is_fast else CHEAP_TARGET


# ---------------------------------------------------------------------------
# Log file iteration
# ---------------------------------------------------------------------------


def discover_log_files(log_dir: Path) -> list[Path]:
    """All proxy log files in ``log_dir``, sorted by name."""
    if not log_dir.is_dir():
        return []
    return sorted(
        p for p in log_dir.iterdir()
        if p.is_file() and (p.name == "proxy.log" or p.name.startswith("proxy.log."))
    )


def discover_recording_dirs(recordings_dir: Path) -> list[Path]:
    """All session-recording directories under ``recordings_dir``."""
    if not recordings_dir.is_dir():
        return []
    return sorted(p for p in recordings_dir.iterdir() if p.is_dir())


def read_recording_session(rec_dir: Path) -> tuple[list[dict] | None, str | None]:
    """Read the full message history from a session-recording directory.

    The recording dir contains timestamped ``*-request.json`` files with
    direction ``client_to_proxy`` whose ``payload.messages`` carry the FULL
    message list at that point in time (the proxy logs truncate request
    bodies to 500 chars; the recordings do not). The last request's message
    list is the session transcript up to that turn; the last user message is
    the target request.

    Returns:
        (messages, target_prompt): the full history (including the target
        user message as the final element) and the final user prompt text.
        ``None``s when the recording has no usable client_to_proxy request.
    """
    requests = sorted(rec_dir.glob("*-request.json"))
    if not requests:
        return None, None

    last_messages: list[dict] | None = None
    for path in requests:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("direction") != "client_to_proxy":
            continue
        payload = data.get("payload")
        if not isinstance(payload, dict):
            continue
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            last_messages = messages

    if last_messages is None:
        return None, None

    target = None
    for msg in reversed(last_messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in reversed(content):
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text.strip():
                        target = text
                        break
        elif isinstance(content, str) and content.strip():
            target = content
        if target:
            break

    return last_messages, target


def iter_log_lines(path: Path):
    """Yield text lines from a proxy log (transparent .gz handling)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(line) < 24 or not line[:4].isdigit():
                continue
            yield line


# ---------------------------------------------------------------------------
# Task extraction from logs
# ---------------------------------------------------------------------------


def parse_routing_sample(line: str) -> EstimateSample | None:
    """Parse a ``routing_check`` line."""
    m = ROUTING_RE.match(line)
    if m is None:
        return None
    ts_s, est, warm, session = m.groups()
    return EstimateSample(
        ts=datetime.strptime(ts_s, TS_FMT),
        session=session,
        estimated_tokens=int(est),
        warm_threshold=int(warm),
    )


def extract_tasks_from_logs(
    log_dir: Path,
    modes: list[Mode] | None = None,
    days: list[str] | None = None,
    min_tasks: int = TASKS_TRIGGER_CAP_BAND,
    min_extreme: int = TASKS_EXTREME_BAND,
    recordings_dir: Path | None = None,
    match_window_s: float = 7200.0,
) -> list[Task]:
    """Extract candidate tasks from proxy logs (and optionally recordings).

    Strategy:
    1. Find all sessions with estimated_tokens exceeding per-mode trigger.
    2. For each session, extract the final turn (target request).
    3. Classify into trigger-cap band or extreme band.
    4. Stratify by mode and category.

    Transcripts come from:
    - ``recordings_dir`` (session recordings with FULL message payloads; the
      proxy log body is truncated at 500 chars), time-matched to the breach
      sample when given, OR
    - the proxy log POST preview (truncated; used as fallback), OR
    - a synthetic transcript mirroring the estimated token size when no
      transcript is recoverable (per design §2, synthetic mirroring is
      permitted where transcripts are unavailable).

    Returns a list of :class:`Task` objects suitable for replay.
    """
    if modes is None:
        modes = [Mode.FAST, Mode.CHEAP]
    triggers = {
        Mode.FAST: FAST_TRIGGER,
        Mode.CHEAP: CHEAP_TRIGGER,
    }

    # Phase 1: collect routing samples to find breach sessions
    breach_sessions: dict[str, dict[Mode, list[EstimateSample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_sessions: dict[str, list[RequestRecord]] = defaultdict(list)

    for path in discover_log_files(log_dir):
        # First pass: collect routing samples
        for line in iter_log_lines(path):
            sample = parse_routing_sample(line)
            if sample is None:
                continue
            if sample.mode not in modes:
                continue
            trigger = triggers[sample.mode]
            if sample.estimated_tokens > trigger:
                breach_sessions[sample.session][sample.mode].append(sample)

        # Second pass: extract request records (truncated previews)
        for line in iter_log_lines(path):
            m = POST_LINE_RE.match(line)
            if m:
                provider, body_str, session_id = m.groups()
                try:
                    body = json.loads(body_str)
                except (json.JSONDecodeError, ValueError):
                    continue
                if "messages" not in body:
                    continue
                messages = body.get("messages", [])
                if not messages:
                    continue
                try:
                    ts = datetime.strptime(line[:19], TS_FMT)
                except ValueError:
                    ts = datetime.now()
                all_sessions[session_id].append(
                    RequestRecord(
                        ts=ts, provider=provider,
                        model=body.get("model", "unknown"),
                        session_id=session_id, messages=messages,
                    )
                )

    # Optional transcript index from session recordings. Each recording is
    # consumable by at most ONE task (greedy nearest-unused match) so a
    # single long session isn't reused across many tasks.
    transcript_index: dict[str, tuple[list[dict], str, datetime, int]] = {}
    if recordings_dir is not None:
        from datetime import timezone as _tz

        for rec_dir in discover_recording_dirs(recordings_dir):
            messages, target = read_recording_session(rec_dir)
            if not messages or not target:
                continue
            files = sorted(rec_dir.glob("*-request.json"))
            try:
                latest = datetime.fromisoformat(
                    files[-1].name.split("-request.json")[0]
                )
                if latest.tzinfo is None:
                    latest = latest.replace(tzinfo=_tz.utc)
            except (Exception, IndexError):
                latest = datetime.now(_tz.utc)
            # Recording timestamps are UTC-aware; routing samples are naive
            # local. Normalise both to a naive UTC-ish base for matching.
            raw_n = sum(
                _estimate_message_tokens(m) for m in messages
            )
            transcript_index[rec_dir.name] = (
                messages, target, latest.replace(tzinfo=None), raw_n,
            )

    # Phase 2: build tasks from breach sessions
    tasks: list[Task] = []
    processed: set[str] = set()
    used_recordings: set[str] = set()

    for session_id, mode_samples in breach_sessions.items():
        records = all_sessions.get(session_id, [])

        for mode, samples in mode_samples.items():
            best_sample = max(samples, key=lambda s: s.estimated_tokens)
            est = best_sample.estimated_tokens
            cap = FAST_CAP if mode == Mode.FAST else CHEAP_CAP
            band = "extreme" if est > cap else "trigger-cap"

            # --- Transcript acquisition ----------------------------------
            messages: list[dict] | None = None
            target_prompt: str | None = None
            transcript_src = "synthetic"

            # 1) Time-matched recording (nearest unused, each used once)
            if transcript_index:
                best_match: tuple[float, str] | None = None
                for rec_name, (_msgs, _tgt, rec_ts, raw_n) in transcript_index.items():
                    if rec_name in used_recordings:
                        continue
                    if raw_n <= 0:
                        continue
                    # Sanity: the transcript must plausibly hold the
                    # estimated token count (scale within a sane band).
                    scale = est / raw_n
                    if not (0.25 <= scale <= 12.0):
                        continue
                    delta = abs((rec_ts - best_sample.ts).total_seconds())
                    if delta < match_window_s and (
                        best_match is None or delta < best_match[0]
                    ):
                        best_match = (delta, rec_name)
                if best_match is not None:
                    messages, target_prompt, _, _ = transcript_index[best_match[1]]
                    used_recordings.add(best_match[1])
                    transcript_src = "recording"

            # 2) Log request records (truncated preview)
            if messages is None and records:
                final_prompt = _extract_final_prompt(records)
                if final_prompt:
                    messages = _build_history(records, final_prompt)
                    target_prompt = final_prompt
                    transcript_src = "log"

            # 3) Synthetic mirror of the estimated size
            if messages is None:
                messages, target_prompt = _synthetic_transcript(est)
                transcript_src = "synthetic"

            if not messages or not target_prompt:
                continue

            task_id = f"{session_id[:12]}-{mode.value}-{est}"
            if task_id in processed:
                continue
            processed.add(task_id)

            tasks.append(
                Task(
                    task_id=task_id,
                    session_id=session_id,
                    mode=mode,
                    category=_categorize_task(target_prompt, messages),
                    original_messages=messages,
                    estimated_tokens=est,
                    target_prompt=target_prompt,
                    band=band,
                    transcript_src=transcript_src,
                )
            )

    # Phase 3: stratified sampling
    return _stratify_tasks(tasks, modes, min_tasks, min_extreme)


def _synthetic_transcript(estimated_tokens: int) -> tuple[list[dict], str]:
    """Build a synthetic transcript mirroring the estimated context size.

    Used only when no real transcript is recoverable (design §2 permits
    synthetic mirroring where transcripts are unavailable, requiring the
    synth task to reproduce the transcript's length/detail structure).
    """
    system = {
        "role": "system",
        "content": (
            "You are an expert coding assistant. Help the user with code, "
            "documentation, and reasoning tasks. Use available tools when "
            "appropriate and explain your reasoning."
        ),
    }
    first_user = {
        "role": "user",
        "content": "Set up a project and complete the following task with full detail.",
    }
    assistant = {
        "role": "assistant",
        "content": "Understood. I will work through the task step by step and keep the context updated as we go. Let me start by reviewing the requirements and preparing the workspace.",
    }
    # Fill the estimated budget with topical detail turns (code/qa mix)
    target = int(estimated_tokens * 0.9)  # leave headroom
    base_tokens = sum(_estimate_message_tokens(m) for m in (system, first_user, assistant))
    messages = [system, first_user, assistant]
    topics = [
        "refactor the caching layer to use a segmented LRU",
        "add a retry-with-backoff wrapper around the streaming client",
        "document the fallback routing decision order",
        "analyze the token pressure distribution for the oversized sessions",
        "trace the wasted-prefill path for the decode-collapse incident",
        "implement the compaction trigger at prompt-assembly time",
        "verify the hard-routing cap against the slot schedule",
    ]
    i = 0
    while sum(_estimate_message_tokens(m) for m in messages) < target and i < 200:
        topic = topics[i % len(topics)]
        messages.append({"role": "user", "content": f"Continue: {topic}."})
        messages.append({
            "role": "assistant",
            "content": (
                f"I examined {topic}. Key findings: the subsystem has several "
                "interacting states and the fix must preserve the existing "
                "routing invariants while adding the new behaviour. I will "
                "proceed with a minimal, testable change and verify it against "
                "the suite before finishing."
            ),
        })
        i += 1
    messages.append({
        "role": "user",
        "content": (
            "Now complete the task: summarize your approach, implement the "
            "change, run the tests, and report the final result."
        ),
    })
    return messages, messages[-1]["content"]


def _extract_final_prompt(records: list[RequestRecord]) -> str | None:
    """Extract the final user prompt from a session's request records."""
    # The final record is the one with the latest timestamp
    if not records:
        return None
    final = max(records, key=lambda r: r.ts)
    messages = final.messages
    if not messages:
        return None

    # Walk backwards to find the last user message
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                for item in reversed(content):
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        if text.strip():
                            return text
    return None


def _build_history(records: list[RequestRecord], final_prompt: str) -> list[dict]:
    """Build the message history up to (but not including) the final prompt."""
    if not records:
        return []

    # Get all messages from the final record, excluding the final prompt
    final = max(records, key=lambda r: r.ts)
    messages = final.messages

    # Find the index of the final prompt
    result = []
    for msg in messages:
        if msg.get("role") != "user":
            result.append(msg)
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and content == final_prompt:
            break
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    if item.get("text") == final_prompt:
                        break
        result.append(msg)

    return result


def _categorize_task(prompt: str, history: list[dict]) -> str:
    """Categorize a task into one of the four categories.

    Categories: code, qa, agent, reasoning
    """
    prompt_lower = prompt.lower()

    # Check for code-related signals
    code_signals = [
        "implement", "refactor", "function", "class", "def ", "import ",
        "code", "file", "edit", "patch", "diff", "bug", "fix ",
        "algorithm", "data structure", "api",
    ]
    if any(s in prompt_lower for s in code_signals):
        return "code"

    # Check for agent/tool signals
    agent_signals = [
        "skill:", "/skill:", "tool", "command", "run ", "execute",
        "workitem", "work-item", "task", "plan ", "intake ",
        "implement ", "audit ", "approve ",
    ]
    if any(s in prompt_lower for s in agent_signals):
        return "agent"

    # Check for reasoning/analysis signals
    reasoning_signals = [
        "analyze", "evaluate", "assess", "determine", "compare",
        "reason", "chain of thought", "step by step", "explain",
        "hypothesis", "conclusion", "inference",
    ]
    if any(s in prompt_lower for s in reasoning_signals):
        return "reasoning"

    # Default to qa
    return "qa"


def _stratify_tasks(
    tasks: list[Task],
    modes: list[Mode],
    min_tasks: int,
    min_extreme: int,
) -> list[Task]:
    """Stratified sampling: ensure minimums per band per mode.

    Returns at least min_tasks per mode in the trigger-cap band and
    min_extreme per mode in the extreme band, if available.
    """
    # Group by mode and band
    by_mode_band: dict[tuple[Mode, str], list[Task]] = defaultdict(list)
    for task in tasks:
        by_mode_band[(task.mode, task.band)].append(task)

    selected: list[Task] = []

    for mode in modes:
        # Trigger-cap band
        band_tasks = by_mode_band.get((mode, "trigger-cap"), [])
        selected.extend(band_tasks[:max(min_tasks, len(band_tasks))])

        # Extreme band
        extreme_tasks = by_mode_band.get((mode, "extreme"), [])
        selected.extend(extreme_tasks[:max(min_extreme, len(extreme_tasks))])

    return selected


# ---------------------------------------------------------------------------
# Compaction logic
# ---------------------------------------------------------------------------


def compact_prompt_messages(
    messages: list[dict],
    strategy: str,
    target_tokens: int,
    estimated_tokens: int,
    trigger: int,
    token_scale: float = 1.0,
) -> list[dict] | None:
    """Apply compaction to a message list.

    Args:
        messages: The full message history.
        strategy: "truncate" (Strategy B, fast) or "summarize" (Strategy C, cheap).
        target_tokens: Target token count after compaction.
        estimated_tokens: Current estimated token count.
        trigger: The trigger that fired.
        token_scale: Multiplier mapping tiktoken estimates to the proxy's
            Qwen3-native estimates (the proxy multiplies tiktoken counts by
            ``token_estimate_multiplier``; the log ``estimated_tokens`` is the
            ground truth).

    Returns:
        Compacted message list, or None if non-compactable.
    """
    if estimated_tokens <= trigger:
        return None  # No compaction needed

    if strategy == FAST_STRATEGY:
        return _compact_truncate(messages, target_tokens, token_scale)
    elif strategy == CHEAP_STRATEGY:
        return _compact_summarize(messages, target_tokens, token_scale)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _compact_truncate(
    messages: list[dict], target: int, token_scale: float = 1.0
) -> list[dict] | None:
    """Strategy B: hard-cap auto-truncate.

    Drop oldest whole turns (never split a turn, never drop system prompt)
    until estimated tokens ≤ target.
    """
    if not messages:
        return None

    # Always retain the first messages (system prompt + first user message)
    # Find the first user message index
    first_user_idx = 0
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            first_user_idx = i
            break

    # Build compacted list: keep system + first user + recent turns
    # Estimate token count per message (scaled to proxy/Qwen3 units)
    retained = messages[:first_user_idx + 1]  # system + first user
    recent = []
    for msg in reversed(messages[first_user_idx + 1:]):
        # Approximate tokens (scaled)
        msg_tokens = _estimate_message_tokens(msg) * token_scale
        if sum(_estimate_message_tokens(m) for m in retained + [msg] + recent) * token_scale <= target:
            recent.append(msg)
        else:
            break

    if not recent:
        # Must keep at least one recent turn
        recent = [messages[-1]]

    recent.reverse()
    return retained + recent


def _compact_summarize(
    messages: list[dict], target: int, token_scale: float = 1.0
) -> list[dict] | None:
    """Strategy C: summarize then truncate.

    1. Retain system prompt + very first user prompt verbatim
    2. Summarize the middle turns (injected as a summary block)
    3. Keep recent turns within budget
    4. Backstop: if still over budget after summarization, drop oldest
       non-first turns (logged)
    """
    if not messages:
        return None

    # Find system prompt and first user prompt
    system_msgs = [m for m in messages if m.get("role") == "system"]
    first_user = None
    other_users = []
    assistant_msgs = []

    first_user_found = False
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system_msgs.append(msg)
        elif role == "user":
            if not first_user_found:
                first_user = msg
                first_user_found = True
            else:
                other_users.append(msg)
        elif role == "assistant":
            assistant_msgs.append(msg)

    # Pair user-assistant turns (excluding first user)
    paired_turns = []
    i = 0
    while i < len(other_users):
        turn = {"user": other_users[i]}
        if i + 1 < len(assistant_msgs):
            turn["assistant"] = assistant_msgs[i]
        paired_turns.append(turn)
        i += 1

    # Estimate total tokens (scaled to proxy/Qwen3 units)
    total_tokens = (
        sum(_estimate_message_tokens(m) for m in system_msgs)
        + _estimate_message_tokens(first_user)
        + sum(_estimate_turn_tokens(t) for t in paired_turns)
    ) * token_scale

    # If already under budget, no compaction needed
    if total_tokens <= target:
        return None

    # Summarize the middle: we create a synthetic summary
    # In practice, this would call a summarizer; here we create a
    # structural placeholder that the harness can replace with real output
    summary_text = _generate_summary_placeholder(paired_turns)

    # Build compacted messages: system + first user + summary + recent turns
    recent_turns = paired_turns[-5:]  # Keep last 5 turns
    compacted = list(system_msgs) + [first_user]

    # Inject summary
    compacted.append({
        "role": "user",
        "content": (
            f"The conversation history before this point was compacted into "
            f"the following summary:\n\n"
            f"<summary>\n{summary_text}\n</summary>"
        ),
    })

    # Add recent turns
    for turn in recent_turns:
        compacted.append(turn["user"])
        if "assistant" in turn:
            compacted.append(turn["assistant"])

    # Verify target (scaled)
    compacted_tokens = sum(
        _estimate_message_tokens(m) for m in compacted
    ) * token_scale
    if compacted_tokens <= target:
        return compacted

    # Backstop: drop older recent turns
    compacted = list(system_msgs) + [first_user]
    compacted.append({
        "role": "user",
        "content": (
            f"The conversation history before this point was compacted into "
            f"the following summary:\n\n"
            f"<summary>\n{summary_text}\n</summary>"
        ),
    })
    recent_turns = paired_turns[-3:]  # Keep last 3 turns
    for turn in recent_turns:
        compacted.append(turn["user"])
        if "assistant" in turn:
            compacted.append(turn["assistant"])

    return compacted


def _generate_summary_placeholder(turns: list[dict]) -> str:
    """Generate a placeholder summary text for the experiment.

    In the real experiment, this would be the output of a summarizer call.
    For the harness, we create a representative placeholder based on the
    turn content.
    """
    if not turns:
        return "No prior history."

    summaries = []
    for turn in turns:
        user_content = turn.get("user", {}).get("content", "")
        if isinstance(user_content, list):
            for item in user_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    user_content = item.get("text", "")
                    break
        assistant_content = turn.get("assistant", {}).get("content", "")
        if isinstance(assistant_content, list):
            for item in assistant_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    assistant_content = item.get("text", "")
                    break

        # Create a brief summary
        summary = f"User asked about: {str(user_content)[:100]}... "
        summary += f"Response covered: {str(assistant_content)[:100]}..."
        summaries.append(summary)

    return "Summary of prior conversation turns:\n" + "\n\n".join(summaries)


def _estimate_message_tokens(msg: dict) -> int:
    """Estimate token count using tiktoken (mirrors the proxy's approach).

    Uses ``cl100k_base`` (the encoding the proxy uses for routing) with a
    configurable multiplier to match the proxy's per-model estimation.
    Falls back to character-based estimation when tiktoken is unavailable.
    """
    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_ENCODER is None:
        try:
            import tiktoken as _tk
            _TIKTOKEN_ENCODER = _tk.get_encoding("cl100k_base")
        except ImportError:
            pass  # Fall back to char/4

    content = msg.get("content", "")
    if isinstance(content, str):
        if _TIKTOKEN_ENCODER is not None:
            try:
                return max(1, len(
                    _TIKTOKEN_ENCODER.encode(content, disallowed_special=())
                ))
            except Exception:
                return max(1, len(content) // 4)
        return max(1, len(content) // 4)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if _TIKTOKEN_ENCODER is not None:
                    try:
                        total += max(1, len(
                            _TIKTOKEN_ENCODER.encode(text, disallowed_special=())
                        ))
                    except Exception:
                        total += max(1, len(text) // 4)
                else:
                    total += max(1, len(text) // 4)
        return max(1, total)
    return 1


def _estimate_turn_tokens(turn: dict) -> int:
    """Estimate tokens for a user-assistant turn pair."""
    total = 0
    for key in ("user", "assistant"):
        if key in turn:
            total += _estimate_message_tokens(turn[key])
    return max(1, total)


# ---------------------------------------------------------------------------
# Arm execution
# ---------------------------------------------------------------------------


class ArmConfig:
    """Configuration for an experiment arm."""
    def __init__(
        self,
        name: str,
        model: str,
        endpoint: str,
        api_key_env: str | None = None,
        compacted: bool = False,
    ):
        self.name = name
        self.model = model
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.compacted = compacted


# ARM CONFIGURATIONS (design doc §3)
ARMS = {
    "A": ArmConfig(
        name="A — baseline",
        model="deepseek-v4-flash",
        endpoint="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        compacted=False,
    ),
    "B": ArmConfig(
        name="B — compacted-local",
        model="Qwen3",
        endpoint="http://192.168.0.199:8000",
        api_key_env=None,
        compacted=True,
    ),
    "C": ArmConfig(
        name="C — ceiling",
        model="Qwen3",
        endpoint="http://192.168.0.199:8000",
        api_key_env=None,
        compacted=False,
    ),
}


@dataclass
class ArmResult:
    """Result from one arm execution for one task."""
    task_id: str
    arm: str  # "A", "B", or "C"
    status: str  # "success", "error", "timeout", "gate"
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    response_content: str | None = None
    ttft_ms: float | None = None
    total_ms: float | None = None
    error: str | None = None
    compaction_before: int | None = None
    compaction_after: int | None = None
    compaction_strategy: str | None = None


@dataclass
class TaskResult:
    """Results for one task across all three arms."""
    task_id: str
    session_id: str
    mode: str
    category: str
    band: str
    estimated_tokens: int
    arms: dict[str, ArmResult] = field(default_factory=dict)
    transcript_src: str = "synthetic"

    def get_arm(self, arm: str) -> ArmResult | None:
        return self.arms.get(arm)

    def has_complete_arms(self) -> bool:
        return all(arm in self.arms for arm in ("A", "B", "C"))

    @property
    def completion_rates(self) -> dict[str, float]:
        """Completion rate per arm (1.0 for success, 0.0 for error)."""
        rates = {}
        for arm_id in ("A", "B", "C"):
            result = self.arms.get(arm_id)
            if result is None:
                rates[arm_id] = None
            else:
                rates[arm_id] = 1.0 if result.status == "success" else 0.0
        return rates

    @property
    def rubric_scores(self) -> dict[str, float | None]:
        """Rubric scores per arm (filled in during scoring phase)."""
        return {
            arm_id: getattr(result, "rubric_score", None)
            for arm_id, result in self.arms.items()
        }


async def execute_task_arms(
    task: Task,
    arms: dict[str, ArmConfig],
    dry_run: bool = False,
    max_concurrent: int = 5,
    timeout_s: int = 120,
) -> TaskResult:
    """Execute all three arms for a task.

    Args:
        task: The task to replay.
        arms: Arm configurations.
        dry_run: If True, simulate API calls with mock responses.
        max_concurrent: Max concurrent API calls.
        timeout_s: Per-call timeout in seconds.

    Returns:
        TaskResult with results from all arms.
    """
    result = TaskResult(
        task_id=task.task_id,
        session_id=task.session_id,
        mode=task.mode.value,
        category=task.category,
        band=task.band,
        estimated_tokens=task.estimated_tokens,
        transcript_src=task.transcript_src,
    )

    strategy = FAST_STRATEGY if task.is_fast else CHEAP_STRATEGY

    # Execute arms concurrently (limited)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_arm(arm_id: str) -> ArmResult:
        async with semaphore:
            arm_cfg = arms[arm_id]
            return await _execute_single_arm(
                task, arm_id, arm_cfg, strategy, dry_run, timeout_s
            )

    # Run all arms concurrently
    tasks_list = [run_arm(arm_id) for arm_id in ("A", "B", "C")]
    arm_results = await asyncio.gather(*tasks_list, return_exceptions=True)

    for arm_id, arm_result in zip(("A", "B", "C"), arm_results):
        if isinstance(arm_result, Exception):
            result.arms[arm_id] = ArmResult(
                task_id=task.task_id,
                arm=arm_id,
                status="error",
                error=str(arm_result),
            )
        else:
            result.arms[arm_id] = arm_result

    return result


async def _execute_single_arm(
    task: Task,
    arm_id: str,
    arm_cfg: ArmConfig,
    strategy: str,
    dry_run: bool,
    timeout_s: int,
) -> ArmResult:
    """Execute a single arm for a task."""
    start_time = time.monotonic()

    # Build the prompt
    if arm_cfg.compacted:
        # Apply compaction. Token scale maps tiktoken estimates to the
        # proxy's Qwen3-native estimates (the log estimated_tokens is ground
        # truth), so compaction decisions and prefill accounting use the
        # same units as routing.
        raw_est = _estimate_message_tokens_list(task.original_messages)
        token_scale = task.estimated_tokens / raw_est if raw_est else 1.0
        compaction_before = task.estimated_tokens
        compacted_messages = compact_prompt_messages(
            task.original_messages,
            strategy,
            task.target_tokens,
            task.estimated_tokens,
            task.trigger,
            token_scale=token_scale,
        )
        if compacted_messages is None:
            # Premise holds (estimated_tokens > trigger), so None means the
            # compaction step determined the content already fits the target
            # (e.g. summarization shrank it below budget). That is a no-op,
            # not an error: proceed with the original messages and record
            # compaction as skipped.
            compacted_messages = task.original_messages
            compaction_success = False
        else:
            compaction_success = True
        compaction_after = round(
            _estimate_message_tokens_list(compacted_messages) * token_scale
        )
        messages = compacted_messages
    else:
        messages = task.original_messages
        raw_est = _estimate_message_tokens_list(messages)
        token_scale = task.estimated_tokens / raw_est if raw_est else 1.0
        compaction_before = None
        compaction_after = None
        compaction_success = False

    if dry_run:
        # Simulate a response
        await asyncio.sleep(0.01)  # Minimal delay
        ttft = 100 + hash(task.task_id) % 500
        response_len = 200 + hash(task.task_id + arm_id) % 500
        return ArmResult(
            task_id=task.task_id,
            arm=arm_id,
            status="success",
            prompt_tokens=round(_estimate_message_tokens_list(messages) * token_scale),
            response_tokens=response_len,
            response_content=f"[DRY RUN] Simulated response for arm {arm_id}",
            ttft_ms=ttft,
            total_ms=time.monotonic() - start_time,
            compaction_before=compaction_before,
            compaction_after=compaction_after,
            compaction_strategy=strategy if compaction_success else None,
        )

    # Actual API call
    try:
        import httpx

        body = {
            "model": arm_cfg.model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.1,
        }

        headers = {"Content-Type": "application/json"}
        api_key = None
        if arm_cfg.api_key_env:
            api_key = _get_api_key(arm_cfg.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            # Measure TTFT (time to first token)
            ttft_start = time.monotonic()
            first_token_time = None
            response_prefix = ""

            async with client.stream("POST", arm_cfg.endpoint + "/v1/chat/completions", json=body, headers=headers) as resp:
                if resp.status_code >= 400:
                    body_text = ""
                    async for chunk in resp.aiter_text():
                        body_text += chunk
                    raise RuntimeError(f"HTTP {resp.status_code}: {body_text[:500]}")
                # Check for compaction-gate responses (429 with gate header)
                for name, value in (
                    resp.headers.items()
                    if hasattr(resp.headers, "items")
                    else []
                ):
                    if name.lower() == "x-compaction-gate" and value == "1":
                        return ArmResult(
                            task_id=task.task_id,
                            arm=arm_id,
                            status="gate",
                            ttft_ms=time.monotonic() - ttft_start,
                            total_ms=time.monotonic() - start_time,
                            compaction_before=compaction_before,
                            compaction_after=compaction_after,
                            compaction_strategy=strategy if compaction_success else None,
                        )

                full_text = ""
                token_count = 0
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                        ttft_ms = (first_token_time - ttft_start) * 1000
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_text += content
                            token_count += 1

                elapsed = time.monotonic() - start_time
                return ArmResult(
                    task_id=task.task_id,
                    arm=arm_id,
                    status="success",
                    prompt_tokens=round(
                        _estimate_message_tokens_list(messages) * token_scale
                    ),
                    response_tokens=token_count,
                    response_content=full_text or None,
                    ttft_ms=ttft_ms if first_token_time is not None else None,
                    total_ms=elapsed * 1000,
                    compaction_before=compaction_before,
                    compaction_after=compaction_after,
                    compaction_strategy=strategy if compaction_success else None,
                )

    except asyncio.TimeoutError:
        return ArmResult(
            task_id=task.task_id,
            arm=arm_id,
            status="timeout",
            error=f"Timed out after {timeout_s}s",
            compaction_before=compaction_before,
            compaction_after=compaction_after,
            compaction_strategy=strategy if compaction_success else None,
        )
    except Exception as exc:  # noqa: BLE001 - broad catch for arm errors
        return ArmResult(
            task_id=task.task_id,
            arm=arm_id,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            compaction_before=compaction_before,
            compaction_after=compaction_after,
            compaction_strategy=strategy if compaction_success else None,
        )


def _get_api_key(env_var: str) -> str | None:
    """Read an API key from the environment (or a local key file)."""
    import os

    val = os.environ.get(env_var)
    if val:
        return val
    # Try local key files that the proxy may use
    candidates = [
        Path.home() / ".deepseek" / "api_key",
        Path.home() / ".config" / "llm-proxy" / f"{env_var.lower()}",
    ]
    for path in candidates:
        if path.is_file():
            try:
                return path.read_text().strip()
            except OSError:
                continue
    return None


def _estimate_message_tokens_list(messages: list[dict]) -> int:
    """Token estimate for a message list."""
    return sum(_estimate_message_tokens(m) for m in messages)


# ---------------------------------------------------------------------------
# Scoring (rubric-based, blind LLM-as-judge)
# ---------------------------------------------------------------------------

RUBRIC_DIMENSIONS = [
    "correctness",      # 1-5: is the answer factually/technically correct
    "completeness",     # 1-5: does it fully address the request
    "detail_recall",    # 1-5: does it retain details from the context
    "instruction_adherence",  # 1-5: does it follow the stated instructions
    "formatting",       # 1-5: is the output well-formed for the task
]

JUDGE_PROMPT = """You are a rubric judge evaluating assistant responses to a user request.
Score the response on each of these dimensions from 1 (poor) to 5 (excellent):

- correctness: is the answer factually/technically correct?
- completeness: does it fully address the request?
- detail_recall: does it retain important details from the provided context?
- instruction_adherence: does it follow the user's stated instructions?
- formatting: is the output well-formed and clear?

Also decide: did the response successfully complete the user's request? Answer yes/no.

Respond ONLY with JSON:
{{"correctness": <int>, "completeness": <int>, "detail_recall": <int>,
"instruction_adherence": <int>, "formatting": <int>,
"completed": <true|false>, "rationale": "<brief 1-sentence rationale>"}}
"""


def score_response(
    task: Task, arm_result: ArmResult, judge_model: str = "judge"
) -> dict:
    """Score a single arm response on the rubric.

    In the real experiment this calls an LLM-as-judge (blind, fixed rubric).
    For offline/dry runs, a deterministic proxy score is computed so the
    pipeline is fully exercisable without live judge calls. Set
    ``--judge-endpoint``/``--judge-key`` to use a live judge.

    Args:
        task: The task the response answers.
        arm_result: The arm result to score.
        judge_model: Judge model identifier (informational).

    Returns:
        Dict with per-dimension scores, completed flag, and rationale.
    """
    if arm_result.status != "success" or not arm_result.response_content:
        # Failure/empty responses score at the floor and count as incomplete
        return {
            "correctness": 1,
            "completeness": 1,
            "detail_recall": 1,
            "instruction_adherence": 1,
            "formatting": 1,
            "completed": False,
            "rationale": f"no usable response (status={arm_result.status})",
        }

    response = arm_result.response_content
    prompt = task.target_prompt
    history = task.original_messages

    # --- Heuristic proxy rubric (offline fallback) -------------------------
    # These are deliberately conservative stand-ins so dry runs and tests
    # exercise the full scoring path. Signal-derived features are computed
    # from the actual response/history, giving the offline score real
    # discriminative power for regression tests.
    correctness = _proxy_correctness_score(response, prompt)
    completeness = _proxy_completeness_score(response, prompt)
    detail_recall = _proxy_detail_recall_score(response, history)
    instruction_adherence = _proxy_instruction_adherence(response, prompt)
    formatting = _proxy_formatting_score(response)
    completed = completeness >= 3 and not _detect_failure(response)

    return {
        "correctness": correctness,
        "completeness": completeness,
        "detail_recall": detail_recall,
        "instruction_adherence": instruction_adherence,
        "formatting": formatting,
        "completed": completed,
        "rationale": "offline heuristic scoring (no judge endpoint configured)",
    }


def _proxy_correctness_score(response: str, prompt: str) -> int:
    """Heuristic correctness: penalize explicit failure markers/inconsistency."""
    score = 3  # neutral default
    if _detect_failure(response):
        score -= 1
    if "error" in response.lower() and "cannot" in response.lower():
        score -= 1
    # Length sanity: empty-ish answers rarely correct
    if len(response.strip()) < 50:
        score -= 1
    return max(1, min(5, score))


def _proxy_completeness_score(response: str, prompt: str) -> int:
    """Heuristic completeness: coverage of prompt keywords + length."""
    prompt_words = {
        w.lower().strip(".,!?;:()")
        for w in prompt.split()
        if len(w) > 3
    }
    resp_lower = response.lower()
    covered = sum(1 for w in prompt_words if w in resp_lower)
    ratio = covered / max(1, len(prompt_words))
    score = 2 if ratio < 0.15 else 3 if ratio < 0.4 else 4
    if len(response) < 100:
        score = max(1, score - 1)
    return max(1, min(5, score))


def _proxy_detail_recall_score(response: str, history: list[dict]) -> int:
    """Heuristic detail recall: how much context-specific vocabulary survives."""
    # Collect rare/technical terms from history (proxy for detail)
    detail_terms = _extract_distinctive_terms(history)
    resp_lower = response.lower()
    matched = sum(1 for t in detail_terms if t in resp_lower)
    if not detail_terms:
        return 3
    ratio = matched / len(detail_terms)
    return 2 if ratio < 0.2 else 3 if ratio < 0.5 else 4


def _proxy_instruction_adherence(response: str, prompt: str) -> int:
    """Heuristic adherence: response shape vs instruction markers in prompt."""
    score = 3
    prompt_lower = prompt.lower()
    if "don't" in prompt_lower or "do not" in prompt_lower or "never" in prompt_lower:
        # Check negative instruction followed (e.g. refusal/absence of the verb)
        score = score  # neutral; hard to verify heuristically
    if any(s in prompt_lower for s in ("json", "format", "table")):
        if "```" in response or "{" in response or "|" in response:
            score += 1
        else:
            score -= 1
    return max(1, min(5, score))


def _proxy_formatting_score(response: str) -> int:
    """Heuristic formatting: structure markers, length stability."""
    score = 3
    if "```" in response or response.count("\n") > 4:
        score += 1
    if len(response) > 3000:
        score -= 1  # rambling
    if len(response) < 30:
        score -= 1
    return max(1, min(5, score))


def _detect_failure(response: str) -> bool:
    """Detect explicit failure signals in a response."""
    lower = response.lower()
    return any(
        marker in lower
        for marker in (
            "i cannot", "i can't", "unable to", "error:", "failed",
            "context too large", "context_too_large", "timed out",
        )
    )


def _extract_distinctive_terms(messages: list[dict]) -> list[str]:
    """Extract distinctive technical terms from conversation history."""
    terms: set[str] = set()
    # Match capitalized tokens and identifier-like tokens
    ident_re = re.compile(r"\b[A-Z][A-Za-z0-9_-]{3,}\b|\b[a-z][a-zA-Z0-9_]{4,}\b")
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        for m in ident_re.findall(str(content)):
            terms.add(m.lower())
    # Filter common words
    COMMON = {
        "about", "after", "before", "being", "below", "between", "could",
        "current", "during", "using", "where", "which", "would", "should",
        "there", "these", "those", "their", "topic", "query", "based",
        "often", "given", "think", "thing", "stuff", "might", "maybe",
    }
    return sorted(t for t in terms if t not in COMMON)[:20]


# ---------------------------------------------------------------------------
# Aggregation and report generation
# ---------------------------------------------------------------------------


def aggregate_metrics(results: list[TaskResult]) -> dict:
    """Aggregate per-arm/per-mode metrics from task results."""
    # Structure: metrics[arm][mode][metric] = value
    arms = ("A", "B", "C")
    modes = ("fast", "cheap")
    metrics: dict[str, dict[str, dict]] = {
        arm: {mode: {"n": 0} for mode in modes} for arm in arms
    }

    rubric_fields = list(RUBRIC_DIMENSIONS)
    for arm in arms:
        for mode in modes:
            m = metrics[arm][mode]
            m["rubric_sum"] = {f: 0.0 for f in rubric_fields}
            m["completions"] = 0
            m["failures"] = 0
            m["gates"] = 0
            m["ttft"] = []
            m["total_ms"] = []
            m["response_tokens"] = []
            m["prefill_est"] = []  # estimated prompt tokens per served task

    for task in results:
        mode = task.mode
        for arm in arms:
            ar = task.arms.get(arm)
            m = metrics[arm][mode]
            if ar is None:
                continue
            m["n"] += 1
            if ar.status == "success":
                m["completions"] += 1 if _scored_completed(task, arm) else 0
                if ar.ttft_ms is not None:
                    m["ttft"].append(ar.ttft_ms)
                if ar.total_ms is not None:
                    m["total_ms"].append(ar.total_ms)
                if ar.response_tokens is not None:
                    m["response_tokens"].append(ar.response_tokens)
                if ar.prompt_tokens is not None:
                    m["prefill_est"].append(ar.prompt_tokens)
                scores = getattr(ar, "rubric_scores", None) or {}
                for f in rubric_fields:
                    m["rubric_sum"][f] += float(scores.get(f, 0))
            elif ar.status == "gate":
                m["gates"] += 1
            else:
                m["failures"] += 1

    # Post-process into final metric dicts
    out: dict[str, dict[str, dict]] = {}
    for arm in arms:
        out[arm] = {}
        for mode in modes:
            m = metrics[arm][mode]
            n = m["n"]
            rubric = {}
            for f in rubric_fields:
                rubric[f] = round(m["rubric_sum"][f] / n, 2) if n else None
            out[arm][mode] = {
                "n": n,
                "completion_rate": round(m["completions"] / n, 3) if n else None,
                "rubric_mean": round(
                    statistics.mean(rubric.values()) if rubric else None, 2
                ) if n else None,
                "rubric": rubric,
                "failure_rate": round(m["failures"] / n, 3) if n else None,
                "gate_rate": round(m["gates"] / n, 3) if n else None,
                "ttft_p50_ms": percentile_or_none(m["ttft"], 50),
                "ttft_p95_ms": percentile_or_none(m["ttft"], 95),
                "total_p50_ms": percentile_or_none(m["total_ms"], 50),
                "total_p95_ms": percentile_or_none(m["total_ms"], 95),
                "response_tokens_avg": mean_or_none(m["response_tokens"]),
                "prefill_est_avg": mean_or_none(m["prefill_est"]),
                "prefill_est_total": sum(m["prefill_est"]),
            }
    return out


def _scored_completed(task: TaskResult, arm: str) -> bool:
    """True when the arm result was rubric-scored as completed."""
    ar = task.arms.get(arm)
    if ar is None:
        return False
    scores = getattr(ar, "rubric_scores", None) or {}
    return bool(scores.get("completed", False))


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def percentile_or_none(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sv = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(sv)))
    idx = min(len(sv) - 1, rank - 1)
    return float(sv[idx])


def evaluate_go_no_go(
    metrics: dict, by_band: dict[str, dict] | None = None
) -> dict:
    """Evaluate the pre-registered go/no-go rules (§5, §8).

    Args:
        metrics: Aggregated metrics from :func:`aggregate_metrics`.
        by_band: Optional per-band (trigger-cap/extreme) metrics;
            quality checks use the trigger-cap band when available (the
            population compaction will act on), falling back to all tasks.

    Returns:
        dict with ``go``, ``checks`` (per-rule results), and ``rationale``.
    """
    checks: list[dict] = []

    def _rubric_ratio(arm_a: dict, arm_b: dict) -> float | None:
        a = arm_a.get("rubric_mean")
        b = arm_b.get("rubric_mean")
        if a in (None, 0) or b is None:
            return None
        return b / a

    def _completion_delta(arm_a: dict, arm_b: dict) -> float | None:
        a = arm_a.get("completion_rate")
        b = arm_b.get("completion_rate")
        if a is None or b is None:
            return None
        return (b - a) * 100  # percentage points

    def _failure_delta(arm_a: dict, arm_b: dict) -> float | None:
        a = arm_a.get("failure_rate", 0)
        b = arm_b.get("failure_rate", 0)
        return (b - a) * 100  # percentage points

    source = by_band or {}
    band_key = next(iter(source), None)

    for mode in ("fast", "cheap"):
        if band_key is not None and mode in source.get(band_key, {}):
            arm_a = source[band_key][mode].get("A")
            arm_b = source[band_key][mode].get("B")
        else:
            arm_a = metrics.get("A", {}).get(mode)
            arm_b = metrics.get("B", {}).get(mode)
        if not arm_a or not arm_b:
            checks.append({"mode": mode, "rule": "rubric", "pass": None, "detail": "insufficient data"})
            continue

        # Rule 1: B rubric >= 0.95 × A
        ratio = _rubric_ratio(arm_a, arm_b)
        checks.append({
            "mode": mode, "rule": "rubric",
            "pass": ratio is not None and ratio >= QUALITY_BAR_RUBRIC_RATIO,
            "detail": f"B/A rubric = {ratio:.3f}" if ratio is not None else "n/a",
        })

        # Rule 2: completion B >= A - 3pp
        delta = _completion_delta(arm_a, arm_b)
        checks.append({
            "mode": mode, "rule": "completion",
            "pass": delta is not None and delta >= -QUALITY_BAR_COMPLETION_DELTA,
            "detail": f"completion delta B-A = {delta:.1f}pp" if delta is not None else "n/a",
        })

        # Rule 3: no failure increase beyond noise
        fdelta = _failure_delta(arm_a, arm_b)
        checks.append({
            "mode": mode, "rule": "failures",
            "pass": fdelta is not None and fdelta <= 5.0,
            "detail": f"failure delta B-A = {fdelta:.1f}pp" if fdelta is not None else "n/a",
        })

    # Efficiency gate (secondary): wasted-prefill reduction >= 25%
    for mode in ("fast", "cheap"):
        a = metrics.get("A", {}).get(mode, {})
        b = metrics.get("B", {}).get(mode, {})
        a_prefill = a.get("prefill_est_total", 0)
        b_prefill = b.get("prefill_est_total", 0)
        if a_prefill and b_prefill:
            reduction = 1 - (b_prefill / a_prefill)
            checks.append({
                "mode": mode, "rule": "prefill_reduction",
                "pass": reduction >= EFFICIENCY_PREFILL_REDUCTION_MIN,
                "detail": f"prefill reduction = {reduction:.1%}",
            })
        else:
            checks.append({
                "mode": mode, "rule": "prefill_reduction",
                "pass": None, "detail": "insufficient data",
            })

    # TTFT P95: B not > 20% worse than A
    for mode in ("fast", "cheap"):
        a = metrics.get("A", {}).get(mode, {})
        b = metrics.get("B", {}).get(mode, {})
        a_ttft = a.get("ttft_p95_ms")
        b_ttft = b.get("ttft_p95_ms")
        if a_ttft and b_ttft:
            worse = (b_ttft - a_ttft) / a_ttft
            checks.append({
                "mode": mode, "rule": "ttft",
                "pass": worse <= EFFICIENCY_TTFT_WORSEN_PCT,
                "detail": f"TTFT P95 B vs A = {worse:+.1%}",
            })
        else:
            checks.append({
                "mode": mode, "rule": "ttft",
                "pass": None, "detail": "insufficient data",
            })

    # Weighted decision: primary quality rules gate the go; efficiency rules
    # are secondary (gate may be waived explicitly per §6).
    quality_rules = [c for c in checks if c["rule"] in ("rubric", "completion", "failures")]
    efficiency_rules = [c for c in checks if c["rule"] in ("prefill_reduction", "ttft")]

    quality_fail = [c for c in quality_rules if c.get("pass") is False]
    efficiency_fail = [c for c in efficiency_rules if c.get("pass") is False]
    insufficient = [c for c in quality_rules if c.get("pass") is None]

    go = len(quality_fail) == 0 and len(insufficient) == 0
    if go and efficiency_fail:
        # Efficiency failures do not veto by default; flagged for operator
        # decision (waivable per §6).
        rationales = [c["detail"] for c in efficiency_fail]
        if rationales:
            go_note = (
                "quality gate met; efficiency rules failed "
                f"({'; '.join(rationales)}) — requires explicit waiver"
            )
        else:
            go_note = "quality gate met; efficiency needs operator waiver"
    elif go:
        go_note = "quality gate met; efficiency gate met"
    else:
        reasons = [c["detail"] for c in quality_fail + insufficient]
        go_note = "quality gate NOT met: " + ("; ".join(reasons) if reasons else "unknown")

    return {
        "go": go,
        "go_requires_waiver": bool(go and efficiency_fail),
        "checks": checks,
        "rationale": go_note,
    }


def render_csv(metrics: dict, path: Path) -> None:
    """Write the aggregated metrics CSV."""
    rows = []
    headers = [
        "arm", "mode", "n", "completion_rate", "rubric_mean",
        "failure_rate", "gate_rate", "ttft_p50_ms", "ttft_p95_ms",
        "total_p50_ms", "total_p95_ms", "response_tokens_avg",
        "prefill_est_avg", "prefill_est_total",
    ] + RUBRIC_DIMENSIONS
    for arm in ("A", "B", "C"):
        for mode in ("fast", "cheap"):
            m = metrics.get(arm, {}).get(mode, {})
            row = {
                "arm": arm, "mode": mode,
                "n": m.get("n", 0),
                "completion_rate": m.get("completion_rate", ""),
                "rubric_mean": m.get("rubric_mean", ""),
                "failure_rate": m.get("failure_rate", ""),
                "gate_rate": m.get("gate_rate", ""),
                "ttft_p50_ms": m.get("ttft_p50_ms", ""),
                "ttft_p95_ms": m.get("ttft_p95_ms", ""),
                "total_p50_ms": m.get("total_p50_ms", ""),
                "total_p95_ms": m.get("total_p95_ms", ""),
                "response_tokens_avg": m.get("response_tokens_avg", ""),
                "prefill_est_avg": m.get("prefill_est_avg", ""),
                "prefill_est_total": m.get("prefill_est_total", 0),
            }
            rubric = m.get("rubric", {})
            for dim in RUBRIC_DIMENSIONS:
                row[dim] = rubric.get(dim, "")
            rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def render_report_markdown(
    metrics: dict, decision: dict, results: list[TaskResult], output_dir: Path
) -> str:
    """Render the human-readable report markdown."""
    lines: list[str] = []
    lines.append("# Session-Compaction Quality Experiment — Results")
    lines.append("")
    lines.append(
        "Three-arm replay experiment (design: "
        "`docs/session-compaction-experiment-design.md`). "
        "A = uncompacted→remote deepseek-v4-flash (baseline), "
        "B = compacted→local Qwen3 (proposed), "
        "C = uncompacted→local Qwen3 (ceiling)."
    )
    lines.append("")

    lines.append("## Go / no-go decision")
    lines.append("")
    lines.append(f"**Decision: {'GO' if decision['go'] else 'NO-GO'}**")
    if decision.get("go_requires_waiver"):
        lines.append("*Efficiency rules failed — decision requires explicit operator waiver.*")
    lines.append("")
    lines.append(f"Rationale: {decision['rationale']}")
    lines.append("")
    lines.append("### Pre-registered checks")
    lines.append("")
    lines.append("| Mode | Rule | Pass | Detail |")
    lines.append("|---|---|---|---|")
    for check in decision["checks"]:
        pass_val = (
            "PASS" if check.get("pass") is True
            else "FAIL" if check.get("pass") is False
            else "n/a"
        )
        lines.append(
            f"| {check['mode']} | {check['rule']} | {pass_val} | "
            f"{check.get('detail', '')} |"
        )
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append("| Arm | Mode | n | Completion | Rubric | Failure | TTFT P95 (ms) | Prefill est. total |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm in ("A", "B", "C"):
        for mode in ("fast", "cheap"):
            m = metrics.get(arm, {}).get(mode, {})
            comp = m.get("completion_rate", "")
            comp_s = f"{comp:.1%}" if isinstance(comp, float) else comp
            rubric = m.get("rubric_mean", "")
            rubric_s = f"{rubric:.2f}" if isinstance(rubric, float) else rubric
            fail = m.get("failure_rate", "")
            fail_s = f"{fail:.1%}" if isinstance(fail, float) else fail
            ttft = m.get("ttft_p95_ms", "")
            ttft_s = f"{ttft:.0f}" if isinstance(ttft, float) else ttft
            prefill = m.get("prefill_est_total", 0)
            lines.append(
                f"| {arm} | {mode} | {m.get('n', 0)} | {comp_s} | {rubric_s} "
                f"| {fail_s} | {ttft_s} | {prefill} |"
            )
    lines.append("")

    # Transcript source breakdown (data provenance)
    from collections import Counter
    src_counts = Counter(t.transcript_src for t in results)
    if src_counts:
        lines.append("## Transcript sources")
        lines.append("")
        for src, cnt in sorted(src_counts.items()):
            lines.append(f"- **{src}**: {cnt} tasks")
        lines.append("")

    lines.append("## Tasks")
    lines.append("")
    lines.append("| Task | Mode | Band | Category | Est. tokens | A | B | C |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t in results:
        status_row = []
        for arm in ("A", "B", "C"):
            ar = t.arms.get(arm)
            if ar is None:
                status_row.append("-")
            elif ar.status == "success":
                status_row.append("ok")
            elif ar.status == "gate":
                status_row.append("gate")
            else:
                status_row.append("fail")
        lines.append(
            f"| {t.task_id} | {t.mode} | {t.band} | {t.category} | "
            f"{t.estimated_tokens} | {' | '.join(status_row)} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence / replay
# ---------------------------------------------------------------------------


def write_results_jsonl(results: list[TaskResult], path: Path) -> None:
    """Write raw per-task results as JSONL for later re-scoring."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for t in results:
            fh.write(json.dumps(_task_result_to_dict(t)) + "\n")


def read_results_jsonl(path: Path) -> list[TaskResult]:
    """Read results previously written by :func:`write_results_jsonl`."""
    results: list[TaskResult] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            task = TaskResult(
                task_id=data["task_id"],
                session_id=data.get("session_id", ""),
                mode=data.get("mode", "fast"),
                category=data.get("category", "qa"),
                band=data.get("band", "trigger-cap"),
                estimated_tokens=data.get("estimated_tokens", 0),
                transcript_src=data.get("transcript_src", "synthetic"),
            )
            for arm_id, arm_data in (data.get("arms") or {}).items():
                task.arms[arm_id] = ArmResult(
                    task_id=task.task_id,
                    arm=arm_id,
                    status=arm_data.get("status", "error"),
                    prompt_tokens=arm_data.get("prompt_tokens"),
                    response_tokens=arm_data.get("response_tokens"),
                    response_content=arm_data.get("response_content"),
                    ttft_ms=arm_data.get("ttft_ms"),
                    total_ms=arm_data.get("total_ms"),
                    error=arm_data.get("error"),
                    compaction_before=arm_data.get("compaction_before"),
                    compaction_after=arm_data.get("compaction_after"),
                    compaction_strategy=arm_data.get("compaction_strategy"),
                )
                if "rubric_scores" in arm_data:
                    setattr(task.arms[arm_id], "rubric_scores", arm_data["rubric_scores"])
            results.append(task)
    return results


def _task_result_to_dict(task: TaskResult) -> dict:
    """Serialize a TaskResult to JSON-compatible dict."""
    arms = {}
    for arm_id, ar in task.arms.items():
        arms[arm_id] = {
            "arm": ar.arm,
            "status": ar.status,
            "prompt_tokens": ar.prompt_tokens,
            "response_tokens": ar.response_tokens,
            "response_content": ar.response_content,
            "ttft_ms": ar.ttft_ms,
            "total_ms": ar.total_ms,
            "error": ar.error,
            "compaction_before": ar.compaction_before,
            "compaction_after": ar.compaction_after,
            "compaction_strategy": ar.compaction_strategy,
        }
        scores = getattr(ar, "rubric_scores", None)
        if scores:
            arms[arm_id]["rubric_scores"] = scores
    return {
        "task_id": task.task_id,
        "session_id": task.session_id,
        "mode": task.mode,
        "category": task.category,
        "band": task.band,
        "estimated_tokens": task.estimated_tokens,
        "transcript_src": task.transcript_src,
        "arms": arms,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_compaction_experiment.py",
        description=(
            "Three-arm session-compaction quality experiment harness "
            "(eval-only; no behaviour change)."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default="/var/log/llama-proxy",
        help="dir containing proxy.log* (default: /var/log/llama-proxy)",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=["fast", "cheap"],
        default=["fast", "cheap"],
        help="modes to include (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        default="experiment-results",
        help="output dir for results (default: experiment-results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="simulate API calls (no live endpoints consulted)",
    )
    parser.add_argument(
        "--replay",
        metavar="JSONL",
        help="re-score a previous run from its JSONL instead of executing",
    )
    parser.add_argument(
        "--score-only", action="store_true",
        help="with --replay: score without re-executing (default in replay mode)",
    )
    parser.add_argument(
        "--min-tasks", type=int, default=TASKS_TRIGGER_CAP_BAND,
        help=f"minim tasks per mode in trigger-cap band (default {TASKS_TRIGGER_CAP_BAND})",
    )
    parser.add_argument(
        "--min-extreme", type=int, default=TASKS_EXTREME_BAND,
        help=f"minim tasks per mode in extreme band (default {TASKS_EXTREME_BAND})",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=5,
        help="max concurrent API calls (default 5)",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="per-call timeout seconds (default 120)",
    )
    parser.add_argument(
        "--deepseek-key", default=None,
        help="DeepSeek API key (default: $DEEPSEEK_API_KEY)",
    )
    parser.add_argument(
        "--local-endpoint", default="http://192.168.0.199:8000",
        help="local Qwen3 endpoint (default: http://192.168.0.199:8000)",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="abort on first unexpected error (CI use)",
    )
    parser.add_argument(
        "--recordings-dir",
        default="~/.llm-proxy/session-recordings",
        help="session-recordings dir with full transcripts (default: "
             "~/.llm-proxy/session-recordings)",
    )
    parser.add_argument(
        "--no-recordings", action="store_true",
        help="ignore session recordings; use log previews/synthetic only",
    )
    parser.add_argument(
        "--match-window", type=float, default=7200.0,
        help="time window (seconds) to match breach samples to recordings "
             "(default 7200)",
    )
    return parser.parse_args(argv)


async def _run_async(args: argparse.Namespace) -> list[TaskResult]:
    """Execute (or replay) the experiment and return task results."""
    if args.replay:
        replay_path = Path(args.replay)
        if not replay_path.is_file():
            print(f"error: replay file not found: {replay_path}", file=sys.stderr)
            return []
        results = read_results_jsonl(replay_path)
        # Re-score every arm result lacking rubric scores (task prompts are
        # not persisted in the JSONL, so scoring uses the stored response
        # content and the heuristic/offline rubric path).
        for task in results:
            t = Task(
                task_id=task.task_id,
                session_id=task.session_id,
                mode=Mode(task.mode),
                category=task.category,
                original_messages=[],
                estimated_tokens=task.estimated_tokens,
                target_prompt="(replay; prompt not persisted)",
                band=task.band,
            )
            for arm_id, ar in task.arms.items():
                if getattr(ar, "rubric_scores", None) is not None:
                    continue  # keep previously assigned scores
                scores = score_response(t, ar)
                setattr(ar, "rubric_scores", scores)
        return results

    modes = [Mode(m) for m in args.mode]
    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"error: log dir not found: {log_dir}", file=sys.stderr)
        return []

    print(f"Extracting tasks from {log_dir} ...")
    recordings_dir = (
        None if args.no_recordings else Path(args.recordings_dir).expanduser()
    )
    tasks = extract_tasks_from_logs(
        log_dir,
        modes=modes,
        min_tasks=args.min_tasks,
        min_extreme=args.min_extreme,
        recordings_dir=recordings_dir,
        match_window_s=args.match_window,
    )
    print(f"Extracted {len(tasks)} tasks")
    if not tasks:
        print("error: no tasks extracted (check --log-dir / log content)", file=sys.stderr)
        return []

    # Configure arms (override endpoints/keys from CLI)
    arms = {
        "A": ArmConfig(
            name=ARMS["A"].name,
            model=ARMS["A"].model,
            endpoint=ARMS["A"].endpoint,
            api_key_env=ARMS["A"].api_key_env,
            compacted=False,
        ),
        "B": ArmConfig(
            name=ARMS["B"].name,
            model=ARMS["B"].model,
            endpoint=args.local_endpoint,
            api_key_env=None,
            compacted=True,
        ),
        "C": ArmConfig(
            name=ARMS["C"].name,
            model=ARMS["C"].model,
            endpoint=args.local_endpoint,
            api_key_env=None,
            compacted=False,
        ),
    }
    if args.deepseek_key:
        import os as _os
        _os.environ.setdefault("DEEPSEEK_API_KEY", args.deepseek_key)

    print(f"Executing {len(tasks)} tasks x 3 arms (dry_run={args.dry_run}) ...")
    results = []
    for i, task in enumerate(tasks, 1):
        if args.fail_fast and i > 1 and not results[-1].has_complete_arms():
            print(f"aborting at task {i} (--fail-fast)", file=sys.stderr)
            break
        task_result = await execute_task_arms(
            task, arms, dry_run=args.dry_run,
            max_concurrent=args.max_concurrent, timeout_s=args.timeout,
        )
        results.append(task_result)
        print(f"  [{i}/{len(tasks)}] {task.task_id}: "
              f"{','.join(a.status[:1] for a in task_result.arms.values())}")

    return results


def _finalize(
    results: list[TaskResult], output_dir: Path, args: argparse.Namespace
) -> dict:
    """Score, aggregate, decide and write artifacts."""
    # Score every arm (idempotent; already-scored results keep scores)
    scored_count = 0
    for task in results:
        # Build a light Task for scoring (prompt not persisted in results)
        light_task = Task(
            task_id=task.task_id,
            session_id=task.session_id,
            mode=Mode(task.mode),
            category=task.category,
            original_messages=[],
            estimated_tokens=task.estimated_tokens,
            target_prompt="(replayed task; see task list)",
            band=task.band,
        )
        for arm_id, ar in task.arms.items():
            if getattr(ar, "rubric_scores", None) is not None:
                continue
            scores = score_response(light_task, ar)
            setattr(ar, "rubric_scores", scores)
            scored_count += 1
    print(f"Scored {scored_count} arm results")

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    jsonl_path = output_dir / f"run-{ts}.jsonl"
    write_results_jsonl(results, jsonl_path)

    metrics = aggregate_metrics(results)

    # Per-band metrics for quality gating (use trigger-cap band)
    by_band = {}
    band_names = sorted({t.band for t in results})
    for band in band_names:
        band_results = [t for t in results if t.band == band]
        by_band[band] = aggregate_metrics(band_results)

    decision = evaluate_go_no_go(metrics, by_band)

    csv_path = output_dir / "metrics.csv"
    render_csv(metrics, csv_path)

    md_path = output_dir / "report.md"
    md_text = render_report_markdown(metrics, decision, results, output_dir)
    md_path.write_text(md_text + "\n", encoding="utf-8")

    report = {
        "generated_at": datetime.now().isoformat(),
        "dry_run": bool(args.dry_run),
        "mode": args.mode,
        "min_tasks": args.min_tasks,
        "min_extreme": args.min_extreme,
        "num_tasks": len(results),
        "decision": decision,
        "metrics": metrics,
        "by_band": by_band,
        "results_path": str(jsonl_path),
        "csv_path": str(csv_path),
        "report_path": str(md_path),
    }
    json_path = output_dir / f"run-{ts}.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.replay and args.score_only is False:
        # Replay mode never re-executes; scoring is the whole point
        pass

    results = asyncio.run(_run_async(args))
    if not results:
        return 2

    _finalize(results, Path(args.output_dir), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())