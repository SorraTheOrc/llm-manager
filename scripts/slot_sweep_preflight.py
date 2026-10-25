#!/usr/bin/env python3
"""Fail-closed quiet-window preflight gate for the slot-count sweep.

The slot-count sweep (``proxy/benchmarks/slot_benchmark.py``) restarts
llama-server between runs, so it is only meaningful when no other session is
using the local backend. The 2026-10-20 attempt ran it while five concurrent
sessions (two audits + four herdr panes) held local leases: every benchmark
turn-1 request stalled at 600-760 s against the 1200 s timeout, llama-server
stopped answering health probes, and the run had to be aborted with the config
restored (LP-0MVGPEP3O000RKKD).

This gate makes "is the host quiet?" machine-checkable and fail-closed, using
only signals the proxy already publishes on ``GET /llama/local/status``
(LP-0MVK0969J004MTG3 AC1/AC2):

  * ``llama_server_running`` is ``true``;
  * ``local_owner_session_id`` is null (no active local dispatch lease);
  * per-slot detail is present and every slot reports ``lease_state ==
    "free"`` with ``is_processing == false`` (an ``idle_leased`` slot is *not*
    free, even when ``available_slots`` still counts it);
  * ``available_slots == total_slots`` (and ``total_slots > 0``);
  * ``local_idle_seconds >= --min-idle-seconds`` (default 120 s);
  * the per-slot detail is fresh (``slots_stale`` is not ``true``).

Usage:
  python3 scripts/slot_sweep_preflight.py                     # human verdict
  python3 scripts/slot_sweep_preflight.py --json               # machine readable
  python3 scripts/slot_sweep_preflight.py --json \
      --min-idle-seconds 300 > preflight.json                  # tighter window
  python3 scripts/slot_sweep_preflight.py --url http://localhost:8000/llama/local/status

Exit codes:
  0 - QUIET: safe to start the slot sweep
  1 - NOT QUIET: a predicate clause failed (the failing clauses are printed)
  2 - status endpoint unreachable, non-200, or unparsable (also NOT-quiet-safe)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any

DEFAULT_STATUS_URL = "http://localhost:8000/llama/local/status"
DEFAULT_MIN_IDLE_SECONDS = 120.0
DEFAULT_TIMEOUT_SECONDS = 5.0

EXIT_QUIET = 0
EXIT_NOT_QUIET = 1
EXIT_STATUS_UNAVAILABLE = 2

#: Ordered clause names. Every clause is evaluated and reported, so a failing
#: preflight names *all* the reasons instead of only the first one.
CHECK_NAMES = (
    "llama_server_running",
    "local_owner_session_id",
    "slots_present",
    "slots_free",
    "slots_idle",
    "slot_capacity",
    "local_idle_seconds",
    "slots_fresh",
)


def _utc_now() -> str:
    """ISO-8601 UTC timestamp for the result payload."""
    return datetime.now(UTC).isoformat()


def _is_number(value: Any) -> bool:
    """True for real numbers (``bool`` is a subclass of ``int``, exclude it)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _slot_summary(slot: Any) -> dict[str, Any]:
    """Reduce a status-payload slot to the fields the gate evaluates."""
    if not isinstance(slot, dict):
        return {
            "slot_id": None,
            "lease_state": None,
            "owner_session_id": None,
            "is_processing": None,
            "phase": None,
        }
    return {
        "slot_id": slot.get("slot_id"),
        "lease_state": slot.get("lease_state"),
        "owner_session_id": slot.get("owner_session_id"),
        "is_processing": slot.get("is_processing"),
        "phase": slot.get("phase"),
    }


def evaluate_quiet(
    status: dict[str, Any],
    min_idle_seconds: float = DEFAULT_MIN_IDLE_SECONDS,
) -> dict[str, Any]:
    """Evaluate the quiet predicate against a ``/llama/local/status`` payload.

    Returns a result dict with ``quiet``, ``reasons`` (one per failing clause),
    ``checks`` (per-clause verdict + detail) and ``inputs`` (the values the
    predicate was evaluated against). Every clause is evaluated, so a caller
    sees all the reasons the host is not quiet rather than just the first.
    """
    checks: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            reasons.append(f"{name}: {detail}")

    llama_running = status.get("llama_server_running")
    record("llama_server_running", llama_running is True, f"llama_server_running={llama_running!r}")

    owner = status.get("local_owner_session_id")
    record(
        "local_owner_session_id",
        not owner,
        f"local dispatch lease held by session {owner!r}",
    )

    raw_slots = status.get("slots")
    slots = [_slot_summary(slot) for slot in raw_slots] if isinstance(raw_slots, list) else []
    record(
        "slots_present",
        bool(slots),
        f"{len(slots)} slot(s) in the status payload (per-slot detail required)",
    )

    leased = [slot for slot in slots if slot["lease_state"] != "free"]
    record(
        "slots_free",
        not leased,
        "; ".join(
            f"slot {slot['slot_id']} lease_state={slot['lease_state']!r} "
            f"owner={slot['owner_session_id']!r} phase={slot['phase']!r}"
            for slot in leased
        )
        or "all slots free",
    )

    processing = [slot for slot in slots if slot["is_processing"] is not False]
    record(
        "slots_idle",
        not processing,
        "; ".join(
            f"slot {slot['slot_id']} is_processing={slot['is_processing']!r}" for slot in processing
        )
        or "no slot is processing",
    )

    available = status.get("available_slots")
    total = status.get("total_slots")
    capacity_ok = (
        _is_number(available)
        and _is_number(total)
        and total > 0
        and available == total
    )
    record(
        "slot_capacity",
        capacity_ok,
        f"available_slots={available!r} total_slots={total!r}",
    )

    idle = status.get("local_idle_seconds")
    if _is_number(idle):
        idle_ok = float(idle) >= float(min_idle_seconds)
        idle_detail = (
            f"local_idle_seconds={float(idle):.1f} < required {float(min_idle_seconds):.1f}s"
            if not idle_ok
            else f"local_idle_seconds={float(idle):.1f}s"
        )
    else:
        idle_ok = False
        idle_detail = (
            f"local_idle_seconds={idle!r} (idle age not published; cannot prove a quiet window)"
        )
    record("local_idle_seconds", idle_ok, idle_detail)

    stale = status.get("slots_stale")
    record(
        "slots_fresh",
        stale is not True,
        f"slots_stale={stale!r} (slot data served from the last-known cache)",
    )

    return {
        "quiet": all(check["ok"] for check in checks.values()),
        "reasons": reasons,
        "checks": checks,
        "min_idle_seconds": float(min_idle_seconds),
        "inputs": {
            "llama_server_running": llama_running,
            "local_owner_session_id": owner,
            "local_idle_seconds": idle,
            "available_slots": available,
            "total_slots": total,
            "slots_stale": stale,
            "slots": slots,
        },
    }


def fetch_status(
    url: str = DEFAULT_STATUS_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any] | None, str | None]:
    """GET the local status endpoint. Returns ``(payload, error)``."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = response.status
            body = response.read()
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code} from {url}"
    except Exception as exc:  # URLError, socket.timeout, connection refused, ...
        return None, f"{type(exc).__name__}: {exc} (unreachable: {url})"

    if status_code != 200:
        return None, f"HTTP {status_code} from {url}"
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON from {url}: {exc}"
    if not isinstance(payload, dict):
        return None, f"unexpected payload type {type(payload).__name__} from {url}"
    return payload, None


def build_result(
    status: dict[str, Any] | None,
    error: str | None,
    status_url: str = DEFAULT_STATUS_URL,
    min_idle_seconds: float = DEFAULT_MIN_IDLE_SECONDS,
) -> dict[str, Any]:
    """Compose the result payload for a status fetch and predicate evaluation."""
    if status is None or error is not None:
        detail = error or "no status payload"
        return {
            "quiet": False,
            "reasons": [f"status_unavailable: {detail}"],
            "checks": {"status_reachable": {"ok": False, "detail": detail}},
            "min_idle_seconds": float(min_idle_seconds),
            "inputs": {},
            "status_url": status_url,
            "checked_at": _utc_now(),
            "error": detail,
        }

    result = evaluate_quiet(status, min_idle_seconds)
    result.update(
        {
            "status_url": status_url,
            "checked_at": _utc_now(),
            "error": None,
        }
    )
    return result


def render_human(result: dict[str, Any]) -> str:
    """Render the human-readable verdict (the JSON form is the machine contract)."""
    lines = [
        "Slot-sweep quiet-window preflight",
        f"  status url: {result.get('status_url')}",
        f"  checked at: {result.get('checked_at')}",
        f"  min idle:   {result.get('min_idle_seconds')}s",
    ]
    inputs = result.get("inputs") or {}
    if inputs:
        lines.append(
            "  observed:   slots {available}/{total}, idle {idle}s, owner {owner!r}".format(
                available=inputs.get("available_slots"),
                total=inputs.get("total_slots"),
                idle=inputs.get("local_idle_seconds"),
                owner=inputs.get("local_owner_session_id"),
            )
        )
    if result.get("quiet"):
        lines.append("")
        lines.append("QUIET - no other local lease holder; safe to start the slot sweep.")
        return "\n".join(lines)

    lines.append("")
    lines.append("NOT QUIET - do not start the slot sweep.")
    for reason in result.get("reasons", []):
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed quiet-window preflight gate for the slot-count sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_STATUS_URL,
        help=f"Local status endpoint (default: {DEFAULT_STATUS_URL})",
    )
    parser.add_argument(
        "--min-idle-seconds",
        type=float,
        default=DEFAULT_MIN_IDLE_SECONDS,
        help=(
            "Required local idle age in seconds "
            f"(default: {DEFAULT_MIN_IDLE_SECONDS:.0f}; raise it for a tighter window)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Status request timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS:.0f})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Machine-readable output (quiet, reasons[], checks{}, inputs{})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    status, error = fetch_status(args.url, args.timeout)
    result = build_result(status, error, args.url, args.min_idle_seconds)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(render_human(result))

    if result["quiet"]:
        return EXIT_QUIET
    if result["error"]:
        return EXIT_STATUS_UNAVAILABLE
    return EXIT_NOT_QUIET


if __name__ == "__main__":
    sys.exit(main())
