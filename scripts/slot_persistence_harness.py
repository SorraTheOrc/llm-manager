#!/usr/bin/env python3
"""Slot persistence analysis harness — produces a structured JSON corpus.

F1 (LP-0MTCMEJX2008W85X): Reproducible scripts parse llama-server.log,
proxy.log, and the slot-cache inventory into one structured dataset
establishing baseline metrics for the KV slot save/restore evaluation
(parent: LP-0MTAQNB7J0094X71).

Produced corpus entries:
  - slot_save_events   — every slot_save success/failure from proxy.log*
  - slot_restore_events — every slot_restore success/failure from proxy.log*
  - skip_events        — routing_skip_local + persistence skip reasons
                         (context_too_large, slot_busy, cooldown, etc.)
  - slots_status_codes — every status_request line → HTTP status code
                         proxy (200=ok, 400=bad, 500=error) with
                         slots_stale flag, total_slots, active_query state
  - lease_events       — lease acquire/release/expiry from proxy.log*
  - llama_checkpoint_events — context checkpoint create events from
                              llama-server.log* (token counts, sizes)
  - slot_cache_inventory — files in session_slot_save_path with sizes,
                           mtimes, age

Usage:
  ./scripts/slot-persistence-harness.py                              # defaults
  ./scripts/slot-persistence-harness.py --log-dir /var/log/llama-proxy
  ./scripts/slot-persistence-harness.py --cache-dir /home/rgardler/projects/llm/slot-cache
  ./scripts/slot-persistence-harness.py --start 2026-08-26 --end 2026-08-28
  ./scripts/slot-persistence-harness.py --json                       # JSON output (default)
  ./scripts/slot-persistence-harness.py --schema                     # print corpus schema

Exit codes:
  0 - success
  1 - no log files found / unexpected error
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _parse_ts(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, _TS_FMT)


# ---------------------------------------------------------------------------
# Log file iteration (live + rotated, oldest-first)
# ---------------------------------------------------------------------------

def _iter_proxy_files(log_dir: Path) -> list[Path]:
    """Proxy log files (live + rotated), oldest-first.

    Two rotation naming schemes coexist:
      old: proxy.log.2026-08-22_01.gz
      new: proxy.log-2026-08-27_00.gz
    """
    live = log_dir / "proxy.log"
    rotated = sorted(
        list(log_dir.glob("proxy.log.*")) + list(log_dir.glob("proxy.log-*")),
        key=lambda p: p.name,
    )
    return ([live] if live.exists() else []) + rotated


def _iter_llama_files(log_dir: Path) -> list[Path]:
    """llama-server log files (live + rotated), oldest-first.

    Two rotation naming schemes coexist:
      old: llama-server.10.log
      new: llama-server.log-2026-08-27.gz
    """
    live = log_dir / "llama-server.log"
    rotated = sorted(
        list(log_dir.glob("llama-server.*.log")) + list(log_dir.glob("llama-server.log-*")),
        key=lambda p: p.name,
    )
    return ([live] if live.exists() else []) + rotated


def _match_llama_files(log_dir: Path, name_glob: str | None) -> list[Path]:
    """Restrict llama log discovery to files whose name matches a glob.

    Useful for day attribution: llama-server logs carry no timestamps, so
    analysis of a single day's file (e.g. 'llama-server.log-2026-08-27.gz'
    = 2026-08-26) is the only timestamp-free way to get day-exact llama
    metrics.
    """
    files = _iter_llama_files(log_dir)
    if not name_glob:
        return files
    import fnmatch

    return [p for p in files if fnmatch.fnmatch(p.name, name_glob)]


def _iter_proxy_logs(log_dir: Path):
    """Yield (path, line) from proxy.log then rotated proxy.log.* (oldest first).

    Handles both plain and gzip-compressed rotated logs.
    """
    import gzip as gz

    files = _iter_proxy_files(log_dir)
    for path in files:
        try:
            opener = gz.open if path.suffix == ".gz" else open
            with opener(path, "rt", errors="replace") as fh:
                yield from fh
        except OSError as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)


def _iter_llama_logs(log_dir: Path, name_glob: str | None = None):
    """Yield (path, line) from llama-server logs selected by ``name_glob``.

    Handles both plain and gzip-compressed rotated logs. Pass a glob (e.g.
    "llama-server.log-2026-08-27.gz") to restrict to a single day's file.
    """
    import gzip as gz

    files = _match_llama_files(log_dir, name_glob)
    for path in files:
        try:
            opener = gz.open if path.suffix == ".gz" else open
            with opener(path, "rt", errors="replace") as fh:
                for line in fh:
                    yield (path, line)
        except OSError as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# proxy.log: slot_save success/failure
_PROXY_SLOT_SAVE_SUCCESS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - \w+ - "
    r"slot_save success session=(?P<session>\S+) slot=(?P<slot>\d+)"
)
_PROXY_SLOT_SAVE_FAILURE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - \w+ - "
    r"slot_save failed slot=(?P<slot>\d+) error=(?P<error>\S+(/[^\s]+)?) "
    r"(?P<rest>.*)$"
)
_PROXY_SLOT_RESTORE_SUCCESS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - \w+ - "
    r"slot_restore success session=(?P<session>\S+) slot=(?P<slot>\d+)"
)
_PROXY_SLOT_RESTORE_FAILURE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - \w+ - "
    r"slot_restore failed slot=(?P<slot>\d+) error=(?P<error>\S+(/[^\s]+)?) "
    r"(?P<rest>.*)$"
)

# proxy.log: routing_skip_local (routing-level skip),
# "slot persistence skipped" (persistence-level skip, LP-0MS91DHPZ001VWQO),
# and "slot persistence disabled" (circuit-breaker cooldown)
_SKIP_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - \w+ - "
    r"(?P<event>routing_skip_local|slot persistence skipped|slot persistence disabled) "
    r"(?P<rest>.+)$"
)
# proxy.log: routing_check lines (per-request local-routing evaluations with
# proxy-side token estimates; used by correlate_oversized_sessions.py for
# prefill-work rollups)
_ROUTING_CHECK_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - \w+ - "
    r"routing_check (?P<rest>.+)$"
)

# proxy.log: status_request → /slots health status
_STATUS_REQUEST_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - \w+ - "
    r"status_request (?P<rest>.+)$"
)

# proxy.log: lease events
#   lease_renewed session=herdr-1787957376-1596736-30595 timeout=30s
#   lease_released session=herdr-... reason=orphan_cleanup stream_abandoned=True
_LEASE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - \w+ - "
    r"lease_(?P<kind>renewed|released) session=(?P<session>\S+) "
    r"(?P<rest>.*)$"
)

# llama-server: context checkpoint creates
_LLAMA_CHECKPOINT_RE = re.compile(
    r"^\[(?P<pid>\d+)\] slot update_slots: id\s+(?P<slot>\d+) \| "
    r"task (?P<task>\d+) \| "
    r"created context checkpoint (?P<ckpt>\d+) of (?P<total>\d+) "
    r"\(pos_min = (?P<pos_min>\d+), pos_max = (?P<pos_max>\d+), "
    r"n_tokens = (?P<n_tokens>\d+), size = (?P<size>[0-9.]+) [MGT]iB\)"
)
# llama-server: context checkpoint restores
# Format: [59455] slot update_slots: id  1 | task 2547 | restored context
#   checkpoint (pos_min = 22801, pos_max = 22801, n_tokens = 22802,
#   n_past = 22802, size = 62.813 MiB)
_LLAMA_CHECKPOINT_RESTORE_RE = re.compile(
    r"^\[(?P<pid>\d+)\] slot update_slots: id\s+(?P<slot>\d+) \| "
    r"task (?P<task>\d+) \| "
    r"restored context checkpoint "
    r"\(pos_min = (?P<pos_min>\d+), pos_max = (?P<pos_max>\d+), "
    r"n_tokens = (?P<n_tokens>\d+), n_past = (?P<n_past>\d+), "
    r"size = (?P<size>[0-9.]+) [MGT]iB\)"
)
# llama-server: prompt processing done lines → prefill-token totals
# (the metric the 2026-08-26 incident used: sum of n_tokens over
#  "prompt processing done" events).
# Format: [59455] slot update_slots: id  2 | task 1 | prompt processing done, n_tokens = 1423, batch.n_tokens = 4
_LLAMA_PROMPT_DONE_RE = re.compile(
    r"^\[(?P<pid>\d+)\] slot update_slots: id\s+(?P<slot>\d+) \| "
    r"task (?P<task>\d+) \| prompt processing done, n_tokens = (?P<tokens>\d+), "
    r"batch\.n_tokens = (?P<batch_tokens>\d+)"
)
# llama-server: prompt eval lines → prefill-token totals
# Format: [51873] prompt eval time =  50782.80 ms /  3255 tokens (  ... tokens per second)
_LLAMA_PREFILL_RE = re.compile(
    r"^\[(?P<pid>\d+)\] prompt eval time =\s+[0-9.]+ ms /\s+(?P<tokens>\d+) tokens "
)
# llama-server: prompt_save / prompt_load (no timestamps)
_LLAMA_SAVE_RE = re.compile(r"prompt_save:")
_LLAMA_LOAD_RE = re.compile(r"prompt_load:")

# llama-server access log: done request lines with HTTP method/path/status
# Format: [51873] srv  log_server_r: done request: GET /slots 127.0.0.1 200
_LLAMA_ACCESS_RE = re.compile(
    r"^\[(?P<pid>\d+)\] srv\s+log_server_r: done request: "
    r"(?P<method>\w+) (?P<path>\S+) (?P<client>\S+) (?P<status>\d{3})"
)
_LLAMA_SLOTS_ACCESS_RE = re.compile(
    r"done request: GET /slots\s+\S+\s+(?P<status>\d{3})"
)

# proxy.log: orphan / stale slot events
_ORPHAN_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} - \w+ - "
    r"orphan.*session=(?P<session>\S+)"
)


# ---------------------------------------------------------------------------
# Parse skip reason from skip line
# ---------------------------------------------------------------------------

def _extract_skip_reason(rest: str) -> dict:
    """Extract structured skip info from a skip_reason line.

    Examples:
      routing_skip_local provider=local-qwen3-next model=Qwen3 estimated_tokens=83494
      cold_threshold=38000 warm_threshold=83285 new_tokens=40 cached_ratio=1.00
      reason=context_too_large → skipping local, routing to next remote provider
      session=herdr-1787941523-285389-9160
    """
    info = {}
    # Parse key=value pairs
    for m in re.finditer(r'(\w+)=([\S]+)', rest):
        key, val = m.group(1), m.group(2)
        if key == 'session':
            info['session'] = val
        elif key in ('estimated_tokens', 'cold_threshold', 'warm_threshold',
                      'new_tokens', 'active_sessions', 'active_queries',
                      'local_active_queries', 'consecutive_failures'):
            try:
                info[key] = int(val)
            except ValueError:
                info[key] = val
        elif key == 'cached_ratio':
            try:
                info[key] = float(val)
            except ValueError:
                info[key] = val
        else:
            info[key] = val

    # Extract reason from "reason=X → ..." pattern
    reason_match = re.search(r'reason=(\w+)', rest)
    if reason_match:
        info['reason'] = reason_match.group(1)

    # Extract skip type from persistence skip lines
    persist_match = re.search(r'persistence skipped session=\S+.*?reason=(\w+)', rest)
    if persist_match:
        info['reason'] = persist_match.group(1)

    return info


# ---------------------------------------------------------------------------
# Parse status_request line into structured fields
# ---------------------------------------------------------------------------

def _parse_status_request(rest: str) -> dict:
    """Parse a status_request key=value line into a dict."""
    fields = {}
    for m in re.finditer(r'(\w+)=([\S]+)', rest):
        key, val = m.group(1), m.group(2)
        if key in ('available_slots', 'total_slots', 'latency_ms'):
            try:
                fields[key] = int(val)
            except ValueError:
                fields[key] = val
        elif key in ('active_query', 'local_active_query', 'slots_stale'):
            fields[key] = val.lower() == 'true'
        elif key in ('llama_server_running', 'model_switch_in_progress'):
            fields[key] = val.lower() == 'true'
        elif key == 'local_owner_lease_remaining_seconds':
            try:
                fields[key] = float(val)
            except ValueError:
                fields[key] = None
        else:
            fields[key] = val
    return fields


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(
    log_dir: Path,
    cache_dir: Path | None = None,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    llama_file_glob: str | None = None,
) -> dict:
    """Run the full analysis and return the corpus dict.

    Returns:
        dict with keys:
          - meta: analysis metadata (time range, file counts)
          - slot_save_events: list of save events
          - slot_restore_events: list of restore events
          - skip_events: list of skip events (routing + persistence)
          - slots_status_codes: list of status_request observations
          - slots_status_summary: aggregate counts
          - lease_events: list of lease events
          - llama_checkpoint_events: list of checkpoint create events
          - llama_prompt_io: {save_lines, load_lines} (no timestamps)
          - slot_cache_inventory: list of cache files with metadata
          - baseline_metrics: key numbers for incident validation
    """

    slot_save_events: list[dict] = []
    slot_restore_events: list[dict] = []
    skip_events: list[dict] = []
    routing_check_events: list[dict] = []
    slots_status_codes: list[dict] = []
    lease_events: list[dict] = []
    llama_checkpoint_events: list[dict] = []
    llama_checkpoint_restore_events: list[dict] = []
    llama_slots_access: list[dict] = []  # GET /slots done-request lines
    llama_prompt_save_count = 0
    llama_prompt_load_count = 0
    prefill_token_total = 0
    prefill_token_lines = 0
    prompt_done_tokens = 0
    prompt_done_events = 0
    orphan_events: list[dict] = []
    # Per-file llama stats (llama-server logs have no timestamps, so the
    # per-file breakdown is the only way to attribute events to a day)
    llama_files_seen: dict[str, dict] = {}

    files_read = {"proxy": 0, "llama": 0}

    # --- Parse proxy logs ---
    for line in _iter_proxy_logs(log_dir):
        # slot_save success
        m = _PROXY_SLOT_SAVE_SUCCESS_RE.match(line)
        if m:
            ts = _parse_ts(m.group("ts"))
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            slot_save_events.append({
                "ts": ts.strftime(_TS_FMT),
                "action": "save",
                "status": "success",
                "session": m.group("session"),
                "slot": int(m.group("slot")),
            })
            continue

        # slot_save failure
        m = _PROXY_SLOT_SAVE_FAILURE_RE.match(line)
        if m:
            ts = _parse_ts(m.group("ts"))
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            detail = m.group("rest").strip()
            event = {
                "ts": ts.strftime(_TS_FMT),
                "action": "save",
                "status": "failure",
                "slot": int(m.group("slot")),
                "error": m.group("error"),
            }
            # Parse error detail for additional fields
            elapsed_m = re.search(r'elapsed=([\d.]+)s', detail)
            timeout_m = re.search(r'timeout=([\d.]+)s', detail)
            busy_m = re.search(r'busy=\{([^}]*)\}', detail)
            if elapsed_m:
                event["elapsed_seconds"] = float(elapsed_m.group(1))
            if timeout_m:
                event["timeout_seconds"] = float(timeout_m.group(1))
            if busy_m:
                try:
                    event["busy_info"] = json.loads("{" + busy_m.group(1) + "}")
                except json.JSONDecodeError:
                    event["busy_info_raw"] = busy_m.group(1)
            slot_save_events.append(event)
            continue

        # slot_restore success
        m = _PROXY_SLOT_RESTORE_SUCCESS_RE.match(line)
        if m:
            ts = _parse_ts(m.group("ts"))
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            slot_restore_events.append({
                "ts": ts.strftime(_TS_FMT),
                "action": "restore",
                "status": "success",
                "session": m.group("session"),
                "slot": int(m.group("slot")),
            })
            continue

        # slot_restore failure
        m = _PROXY_SLOT_RESTORE_FAILURE_RE.match(line)
        if m:
            ts = _parse_ts(m.group("ts"))
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            slot_restore_events.append({
                "ts": ts.strftime(_TS_FMT),
                "action": "restore",
                "status": "failure",
                "slot": int(m.group("slot")),
                "error": m.group("error"),
            })
            continue

        # skip events (routing_skip_local + persistence skipped/disabled)
        m = _SKIP_RE.match(line)
        if m:
            ts = _parse_ts(m.group("ts"))
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            event = m.group("event")
            event_type = (
                "routing_skip" if "routing_skip" in event
                else "persistence_skip" if "skipped" in event
                else "persistence_cooldown"
            )
            skip_events.append({
                "ts": ts.strftime(_TS_FMT),
                "event_type": event_type,
                "details": _extract_skip_reason(m.group("rest")),
            })
            continue

        # routing_check lines (proxy-side prefill estimates)
        m = _ROUTING_CHECK_RE.match(line)
        if m:
            ts = _parse_ts(m.group("ts"))
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            routing_check_events.append({
                "ts": ts.strftime(_TS_FMT),
                "details": _extract_skip_reason(m.group("rest")),
            })
            continue

        # status_request
        m = _STATUS_REQUEST_RE.match(line)
        if m:
            ts = _parse_ts(m.group("ts"))
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            fields = _parse_status_request(m.group("rest"))
            fields["ts"] = ts.strftime(_TS_FMT)
            slots_status_codes.append(fields)
            continue

        # lease events (renewed + released, lease churn metric)
        m = _LEASE_RE.match(line)
        if m:
            ts = _parse_ts(m.group("ts"))
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            rest = m.group("rest")
            event = {
                "ts": ts.strftime(_TS_FMT),
                "event": f"lease_{m.group('kind')}",
                "session": m.group("session"),
            }
            reason_m = re.search(r'reason=(\S+)', rest)
            if reason_m:
                event["reason"] = reason_m.group(1)
            timeout_m = re.search(r'timeout=(\d+)s', rest)
            if timeout_m:
                event["timeout_seconds"] = int(timeout_m.group(1))
            abandoned_m = re.search(r'stream_abandoned=(\w+)', rest)
            if abandoned_m:
                event["stream_abandoned"] = abandoned_m.group(1) == "True"
            lease_events.append(event)
            continue

        # orphan events
        m = _ORPHAN_RE.match(line)
        if m:
            ts = _parse_ts(m.group("ts"))
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            orphan_events.append({
                "ts": ts.strftime(_TS_FMT),
                "session": m.group("session"),
            })
            continue

        files_read["proxy"] += 1

    # --- Parse llama-server logs ---
    for path, line in _iter_llama_logs(log_dir, llama_file_glob):
        file_name = path.name
        file_stats = llama_files_seen.setdefault(file_name, {
            "created_checkpoints": 0,
            "restored_checkpoints": 0,
            "slots_200": 0,
            "slots_400": 0,
            "slots_500": 0,
            "prompt_save": 0,
            "prompt_load": 0,
            "prefill_tokens": 0,
            "prefill_lines": 0,
            "prompt_done_tokens": 0,
            "prompt_done_events": 0,
        })

        # Context checkpoint restores
        m = _LLAMA_CHECKPOINT_RESTORE_RE.match(line)
        if m:
            file_stats["restored_checkpoints"] += 1
            llama_checkpoint_restore_events.append({
                "file": file_name,
                "pid": int(m.group("pid")),
                "slot": int(m.group("slot")),
                "task": int(m.group("task")),
                "pos_min": int(m.group("pos_min")),
                "pos_max": int(m.group("pos_max")),
                "n_tokens": int(m.group("n_tokens")),
                "n_past": int(m.group("n_past")),
                "size_mib": float(m.group("size").split()[0]),
                "size_unit": m.group("size").split()[-1] if " " in m.group("size") else "MiB",
            })
            continue

        # Context checkpoint creates
        m = _LLAMA_CHECKPOINT_RE.match(line)
        if m:
            file_stats["created_checkpoints"] += 1
            checkpoint = {
                "file": file_name,
                "pid": int(m.group("pid")),
                "slot": int(m.group("slot")),
                "task": int(m.group("task")),
                "checkpoint_num": int(m.group("ckpt")),
                "checkpoint_total": int(m.group("total")),
                "pos_min": int(m.group("pos_min")),
                "pos_max": int(m.group("pos_max")),
                "n_tokens": int(m.group("n_tokens")),
                "size_mib": float(m.group("size").split()[0]),
                "size_unit": m.group("size").split()[-1] if " " in m.group("size") else "MiB",
            }
            llama_checkpoint_events.append(checkpoint)
            continue

        # llama-server access log: GET /slots done requests → HTTP status counts
        m = _LLAMA_SLOTS_ACCESS_RE.search(line)
        if m:
            llama_slots_access.append({
                "file": file_name,
                "status": int(m.group("status")),
                "line": line.strip()[:200],
            })
            key = f"slots_{m.group('status')}"
            if key in file_stats:
                file_stats[key] += 1
            continue

        # prefill-token aggregation (llama-server prompt processing done;
        # the metric used by the 2026-08-26 incident: 42.7M prefill tokens)
        m = _LLAMA_PROMPT_DONE_RE.match(line)
        if m:
            prompt_done_tokens += int(m.group("tokens"))
            prompt_done_events += 1
            file_stats["prompt_done_tokens"] = file_stats.get("prompt_done_tokens", 0) + int(m.group("tokens"))
            file_stats["prompt_done_events"] = file_stats.get("prompt_done_events", 0) + 1
            continue

        # prefill-token aggregation (llama-server prompt eval lines)
        m = _LLAMA_PREFILL_RE.match(line)
        if m:
            tokens = int(m.group("tokens"))
            prefill_token_total += tokens
            prefill_token_lines += 1
            file_stats["prefill_tokens"] += tokens
            file_stats["prefill_lines"] += 1
            continue

        # prompt_save / prompt_load (no timestamps — just count)
        if _LLAMA_SAVE_RE.search(line):
            llama_prompt_save_count += 1
            file_stats["prompt_save"] += 1
        elif _LLAMA_LOAD_RE.search(line):
            llama_prompt_load_count += 1
            file_stats["prompt_load"] += 1

        files_read["llama"] += 1

    # --- Slot cache inventory ---
    cache_inventory = []
    if cache_dir and cache_dir.exists():
        for f in sorted(cache_dir.iterdir()):
            if f.is_file() and f.suffix == ".bin":
                stat = f.stat()
                cache_inventory.append({
                    "filename": f.name,
                    "size_bytes": stat.st_size,
                    "size_human": _human_bytes(stat.st_size),
                    "mtime": dt.datetime.fromtimestamp(stat.st_mtime).strftime(_TS_FMT),
                    "age_hours": round((dt.datetime.now() - dt.datetime.fromtimestamp(stat.st_mtime)).total_seconds() / 3600, 1),
                })

    # --- Compute summary statistics ---
    save_actions = Counter(e["status"] for e in slot_save_events)
    restore_actions = Counter(e["status"] for e in slot_restore_events)
    skip_reasons = Counter(
        e["details"].get("reason", "unknown")
        for e in skip_events
    )

    # /slots status code summary — proxy returns:
    #   status codes aren't in status_request lines directly, but we infer
    #   health: total_slots > 0 and not slots_stale → healthy (200-class)
    #   slots_stale = true → degraded
    #   total_slots == 0 and llama_server_running → stale cache (400-class)
    #   We count total polls and stale flags
    slots_stale_count = sum(1 for s in slots_status_codes if s.get("slots_stale"))
    slots_total_zero_count = sum(
        1 for s in slots_status_codes
        if s.get("total_slots") == 0 and s.get("llama_server_running")
    )

    # Proxy-side estimated-token rollup from routing_skip/persistence events.
    # The incident's ~42.7M prefill tokens/day derives from proxy-side
    # estimated_tokens on routing checks (routing_skip_local lines carry
    # per-request estimates), not from llama-server prompt-eval lines (which
    # are only logged for local dispatches). Report both so the two measures
    # can be compared.
    proxy_estimated_tokens = sum(
        e["details"].get("estimated_tokens", 0) or 0
        for e in skip_events
    )
    proxy_estimated_requests = sum(
        1 for e in skip_events if e["details"].get("estimated_tokens") is not None
    )
    # routing_check rollup (all local-routing evaluations, incl. those that
    # passed the gate and proceeded local)
    routing_check_tokens = sum(
        e["details"].get("estimated_tokens", 0) or 0
        for e in routing_check_events
    )
    routing_check_count = len(routing_check_events)
    slots_healthy_count = sum(
        1 for s in slots_status_codes
        if s.get("total_slots", 0) > 0 and not s.get("slots_stale", False)
    )

    # Baseline metrics for incident validation
    llama_slots_status_counts = Counter(a["status"] for a in llama_slots_access)
    slots_polls_total = sum(llama_slots_status_counts.values())

    baseline = {
        "slot_save_success": save_actions["success"],
        "slot_save_failure": save_actions["failure"],
        "slot_restore_success": restore_actions["success"],
        "slot_restore_failure": restore_actions["failure"],
        "total_slot_saves": save_actions["success"],
        "total_slot_restores": restore_actions["success"],
        "restore_rate_pct": _safe_pct(restore_actions["success"],
                                       save_actions["success"]),
        "skip_events_total": len(skip_events),
        "skip_reasons": dict(skip_reasons),
        "slots_status_polls": len(slots_status_codes),
        "slots_stale_count": slots_stale_count,
        "slots_stale_pct": _safe_pct(slots_stale_count, len(slots_status_codes)),
        "slots_healthy_count": slots_healthy_count,
        "slots_total_zero_during_server_up": slots_total_zero_count,
        # llama-server /slots access-log evidence (the direct source of the
        # incident's 6,459/69.6K HTTP 500 claim)
        "llama_slots_polls": slots_polls_total,
        "llama_slots_status_counts": dict(llama_slots_status_counts),
        "llama_slots_500_pct": _safe_pct(llama_slots_status_counts.get(500, 0),
                                          slots_polls_total),
        "llama_prompt_save_lines": llama_prompt_save_count,
        "llama_prompt_load_lines": llama_prompt_load_count,
        "prefill_token_total": prefill_token_total,
        "prefill_prompt_eval_lines": prefill_token_lines,
        "prompt_done_tokens_total": prompt_done_tokens,
        "prompt_done_events": prompt_done_events,
        "proxy_estimated_tokens_total": proxy_estimated_tokens,
        "proxy_estimated_tokens_requests": proxy_estimated_requests,
        "routing_check_tokens_total": routing_check_tokens,
        "routing_check_count": routing_check_count,
        # llama-server native context checkpoints (the 2026-08-26 incident
        # numbers: 2,954 created vs 145 restored)
        "llama_checkpoints_created": len(llama_checkpoint_events),
        "llama_checkpoints_restored": len(llama_checkpoint_restore_events),
        "llama_checkpoint_restore_rate_pct": _safe_pct(
            len(llama_checkpoint_restore_events), len(llama_checkpoint_events)),
        "orphan_events": len(orphan_events),
        "lease_events_total": len(lease_events),
        "cache_files": len(cache_inventory),
        "cache_total_bytes": sum(c["size_bytes"] for c in cache_inventory),
    }

    return {
        "meta": {
            "log_dir": str(log_dir),
            "cache_dir": str(cache_dir) if cache_dir else None,
            "start": start.strftime(_TS_FMT) if start else None,
            "end": end.strftime(_TS_FMT) if end else None,
            "generated": dt.datetime.now().strftime(_TS_FMT),
            "proxy_lines_read": files_read["proxy"],
            "llama_lines_read": files_read["llama"],
            "proxy_files": len(_iter_proxy_files(log_dir)),
            "llama_files": len(list(_match_llama_files(log_dir, llama_file_glob))),
        },
        "slot_save_events": slot_save_events,
        "slot_restore_events": slot_restore_events,
        "skip_events": skip_events,
        "routing_check_events": routing_check_events,
        "slots_status_codes": slots_status_codes,
        "slots_status_summary": {
            "total_polls": len(slots_status_codes),
            "healthy": slots_healthy_count,
            "stale": slots_stale_count,
            "stale_pct": _safe_pct(slots_stale_count, len(slots_status_codes)),
            "total_slots_zero_during_server_up": slots_total_zero_count,
        },
        "lease_events": lease_events,
        "orphan_events": orphan_events,
        "llama_checkpoint_events": llama_checkpoint_events,
        "llama_checkpoint_restore_events": llama_checkpoint_restore_events,
        "llama_slots_access": llama_slots_access,
        "llama_files_seen": llama_files_seen,
        "llama_prompt_io": {
            "save_lines": llama_prompt_save_count,
            "load_lines": llama_prompt_load_count,
        },
        "prefill_tokens": {
            "total": prefill_token_total,
            "prompt_eval_lines": prefill_token_lines,
        },
        "prompt_processing_done": {
            "tokens_total": prompt_done_tokens,
            "events": prompt_done_events,
        },
        "slot_cache_inventory": cache_inventory,
        "baseline_metrics": baseline,
    }


def _safe_pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ---------------------------------------------------------------------------
# Schema output
# ---------------------------------------------------------------------------

SCHEMA = {
    "slot_save_events": {
        "type": "array",
        "items": {
            "ts": "string (YYYY-MM-DD HH:MM:SS)",
            "action": "string — 'save'",
            "status": "string — 'success' | 'failure'",
            "session": "string — session ID",
            "slot": "integer — slot ID",
            # On failure only:
            "error": "string — error class",
            "elapsed_seconds": "float — time spent before timeout",
            "timeout_seconds": "float — timeout that was hit",
            "busy_info": "object — active_queries, local_active_queries, etc.",
        },
    },
    "slot_restore_events": {
        "type": "array",
        "items": {
            "ts": "string",
            "action": "string — 'restore'",
            "status": "string — 'success' | 'failure'",
            "session": "string",
            "slot": "integer",
            # On failure only:
            "error": "string",
        },
    },
    "skip_events": {
        "type": "array",
        "items": {
            "ts": "string",
            "level": "string — INFO | WARNING",
            "event_type": "string — 'routing_skip' | 'persistence_skip' | 'persistence_cooldown'",
            "details": {
                "reason": "string — context_too_large | slot_busy | cooldown | ...",
                "session": "string",
                "estimated_tokens": "integer",
                "cold_threshold": "integer",
                "warm_threshold": "integer",
                "cached_ratio": "float",
                # etc.
            },
        },
    },
    "routing_check_events": {
        "type": "array",
        "items": {
            "ts": "string",
            "details": {
                "provider": "string",
                "model": "string",
                "estimated_tokens": "integer",
                "cold_threshold": "integer",
                "warm_threshold": "integer",
                "new_tokens": "integer",
                "cached_ratio": "float",
                "session": "string",
            },
        },
    },
    "slots_status_codes": {
        "type": "array",
        "items": {
            "ts": "string",
            "client_ip": "string",
            "client_port": "integer",
            "available_slots": "integer",
            "total_slots": "integer",
            "latency_ms": "integer",
            "llama_server_running": "boolean",
            "local_active_query": "boolean",
            "slots_stale": "boolean",
        },
    },
    "lease_events": {
        "type": "array",
        "items": {
            "ts": "string",
            "event": "string — 'lease_acquire' | 'lease_release' | "
                      "'lease_renewed' | 'lease_released' | 'lease_expiry'",
            "session": "string (if renew/release)",
        },
    },
    "llama_checkpoint_events": {
        "type": "array",
        "items": {
            "pid": "integer",
            "slot": "integer",
            "task": "integer",
            "checkpoint_num": "integer",
            "checkpoint_total": "integer",
            "n_tokens": "integer",
            "size_mib": "float",
        },
    },
    "slot_cache_inventory": {
        "type": "array",
        "items": {
            "filename": "string",
            "size_bytes": "integer",
            "mtime": "string",
            "age_hours": "float",
        },
    },
    "baseline_metrics": {
        "type": "object",
        "keys": "see analyze() → baseline",
    },
    "llama_files_seen": {
        "type": "object",
        "keys": {
            "<llama-log-filename>": {
                "created_checkpoints": "int",
                "restored_checkpoints": "int",
                "slots_200": "int",
                "slots_400": "int",
                "slots_500": "int",
                "prompt_save": "int",
                "prompt_load": "int",
                "prefill_tokens": "int",
                "prefill_lines": "int",
                "prompt_done_tokens": "int",
                "prompt_done_events": "int",
            }
        },
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-dir", default="/var/log/llama-proxy",
                        help="directory containing proxy.log*/llama-server.log*")
    parser.add_argument("--cache-dir", default=None,
                        help="slot-cache directory (default: use session_slot_save_path from config)")
    parser.add_argument("--start", help="start timestamp filter (YYYY-MM-DD[ HH:MM:SS])")
    parser.add_argument("--end", help="end timestamp filter (YYYY-MM-DD[ HH:MM:SS])")
    parser.add_argument("--llama-file", default=None,
                        help="restrict llama-server log parsing to files whose name "
                             "matches this glob (e.g. '*2026-08-27*' for the "
                             "2026-08-26 incident day); llama logs have no timestamps")
    parser.add_argument("--json", action="store_true", default=True,
                        help="emit JSON output (default)")
    parser.add_argument("--summary", action="store_true",
                        help="emit a compact baseline summary (meta + baseline_metrics + "
                             "per-file llama breakdown) instead of the full corpus")
    parser.add_argument("--schema", action="store_true",
                        help="print corpus schema and exit")
    parser.add_argument("--compact", action="store_true",
                        help="emit compact JSON (no indentation)")
    args = parser.parse_args(argv)

    if args.schema:
        print(json.dumps(SCHEMA, indent=2))
        return 0

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"error: log directory not found: {log_dir}", file=sys.stderr)
        return 1

    # Default cache dir to common location if not specified
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path("/home/rgardler/projects/llm/slot-cache")

    start = _parse_optional_ts(args.start)
    end = _parse_optional_ts(args.end)
    if start and end and start > end:
        print("error: --start must be before --end", file=sys.stderr)
        return 1

    try:
        corpus = analyze(log_dir, cache_dir, start, end, args.llama_file)
    except Exception as exc:
        print(f"error: analysis failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    indent = None if args.compact else 2
    if args.summary:
        summary = {
            "meta": corpus["meta"],
            "baseline_metrics": corpus["baseline_metrics"],
            "slots_status_summary": corpus["slots_status_summary"],
            "llama_files_seen": corpus["llama_files_seen"],
            "log_events_total": {
                "slot_save": len(corpus["slot_save_events"]),
                "slot_restore": len(corpus["slot_restore_events"]),
                "skip": len(corpus["skip_events"]),
                "routing_check": len(corpus["routing_check_events"]),
                "slots_status": len(corpus["slots_status_codes"]),
                "lease": len(corpus["lease_events"]),
                "checkpoints_created": len(corpus["llama_checkpoint_events"]),
                "checkpoints_restored": len(corpus["llama_checkpoint_restore_events"]),
            },
        }
        print(json.dumps(summary, indent=indent, default=str))
    else:
        print(json.dumps(corpus, indent=indent, default=str))
    return 0


def _parse_optional_ts(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise SystemExit(f"error: invalid timestamp: {value!r} (use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)")


if __name__ == "__main__":
    sys.exit(main())
