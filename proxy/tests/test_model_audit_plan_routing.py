"""On-demand audit test for plan model routing behaviour.

This test makes three sequential requests with the ``plan`` model, spaced
15 seconds apart, and inspects proxy logs to determine which provider served
each request and why. It is only run when explicitly requested.

Run:

    RUN_MODEL_AUDIT=1 pytest -q tests/test_model_audit_plan_routing.py -m e2e_live -s

Optional env vars:
- LIVE_PROXY_BASE_URL (default: http://localhost:8000)
- LIVE_PROXY_TIMEOUT_SECONDS (default: 300)
"""

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import pytest
import requests


pytestmark = [pytest.mark.slow, pytest.mark.e2e_live]

if os.getenv("RUN_MODEL_AUDIT", "0") != "1":
    pytest.skip(
        "model audit tests are disabled; set RUN_MODEL_AUDIT=1 to run on demand",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("model-audit")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [model-audit] %(message)s")
    )
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("LIVE_PROXY_BASE_URL", "http://localhost:8000").rstrip("/")
DEFAULT_TIMEOUT = float(os.getenv("LIVE_PROXY_TIMEOUT_SECONDS", "300"))
PROXY_LOG_DIR = Path("/var/log/llama-proxy")


def _now_iso() -> str:
    """Return current local time as ISO string.

    Uses localtime to match the proxy log timestamps, which are in
    the server's local timezone (e.g., BST/UTC+1).
    """
    return datetime.now().isoformat()


def _new_session_id(prefix: str) -> str:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{prefix}-{ts}-{uuid.uuid4()}"


def _log(message: str, *, payload: Any = None) -> None:
    if payload is not None:
        rendered = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        LOGGER.info("%s\n%s", message, rendered)
        print(f"[model-audit] {message}\n{rendered}")
    else:
        LOGGER.info(message)
        print(f"[model-audit] {message}")


# ---------------------------------------------------------------------------
# Proxy helpers
# ---------------------------------------------------------------------------

def _require_local_proxy() -> None:
    _log(f"Checking proxy health at {BASE_URL}/health")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
    except requests.RequestException as exc:
        pytest.skip(f"live proxy not reachable at {BASE_URL}: {exc}")

    if response.status_code != 200:
        pytest.skip(
            f"live proxy health check failed at {BASE_URL}/health "
            f"status={response.status_code} body={response.text}"
        )
    _log("Proxy health check passed", payload={"status_code": response.status_code})


def _chat(
    *,
    prompt: str,
    session_id: Optional[str] = None,
    max_tokens: int = 800,
    temperature: float = 0.0,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[requests.Response, Dict[str, Any], float]:
    """Send a chat completion request using model=plan.

    Returns (response, body_dict, elapsed_seconds).
    """
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "plan",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
    }
    if session_id:
        headers["X-Session-Id"] = session_id

    _log(
        "Sending chat request",
        payload={
            "url": url,
            "session_id": session_id,
            "prompt_truncated": prompt[:120] + ("..." if len(prompt) > 120 else ""),
        },
    )

    started = time.monotonic()
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    elapsed = time.monotonic() - started

    body: Dict[str, Any] = {}
    try:
        body = response.json()
    except Exception:
        body = {"_raw": response.text}

    _log(
        "Received chat response",
        payload={
            "status_code": response.status_code,
            "elapsed_seconds": round(elapsed, 3),
            "x_provider": response.headers.get("X-Provider"),
            "x_session_id": response.headers.get("X-Session-Id"),
            "model_in_body": body.get("model"),
            "response_truncated": str(body)[:500] + ("..." if len(str(body)) > 500 else ""),
        },
    )

    return response, body, elapsed


def _extract_response_text(body: Dict[str, Any]) -> str:
    try:
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices:
            return ""
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            pieces = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
                    pieces.append(item["text"])
            if pieces:
                return "\n".join(pieces)
        reasoning = msg.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
        return ""
    except Exception:
        return ""


def _provider_from_headers(response: requests.Response) -> str:
    return (response.headers.get("X-Provider") or "").strip()


def _model_from_response(body: Dict[str, Any]) -> str:
    return str(body.get("model", "") or "").strip()


# ---------------------------------------------------------------------------
# Proxy log inspection
# ---------------------------------------------------------------------------

def _find_latest_proxy_log() -> Optional[Path]:
    """Find the most recent proxy log file (current + rotated)."""
    if not PROXY_LOG_DIR.exists():
        _log(f"Proxy log directory {PROXY_LOG_DIR} does not exist")
        return None

    candidates = sorted(PROXY_LOG_DIR.glob("proxy.log*"), reverse=True)
    if not candidates:
        _log("No proxy.log files found")
        return None

    # The current proxy.log is the most recent; use it.
    current = PROXY_LOG_DIR / "proxy.log"
    if current.exists():
        return current

    return candidates[0]


def _read_proxy_log_since(timestamp_iso: str) -> List[str]:
    """Read proxy log lines that occur at or after the given ISO timestamp.

    Since the log format is ``YYYY-MM-DD HH:MM:SS,mmm``, we compare
    as a string prefix against each line.
    """
    log_path = _find_latest_proxy_log()
    if log_path is None:
        _log("No proxy log available for inspection")
        return []

    # Parse the timestamp prefix from the ISO string.
    # ISO: 2026-07-13T10:48:58.123456+00:00
    # Log: 2026-07-13 10:48:58,165
    # We convert the ISO to log format for prefix matching.
    try:
        dt = datetime.fromisoformat(timestamp_iso)
        log_prefix = dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        _log(f"Could not parse timestamp: {timestamp_iso}, reading full log")
        log_prefix = ""

    _log(f"Reading proxy log: {log_path} (since lines with prefix '{log_prefix}')")

    relevant_lines: List[str] = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                # If we have a prefix filter, check if line starts with it
                if log_prefix and line.startswith(log_prefix):
                    relevant_lines.append(line)
                elif not log_prefix:
                    relevant_lines.append(line)
    except Exception as exc:
        _log(f"Failed to read proxy log: {exc}")

    _log(f"Read {len(relevant_lines)} log lines (examples):")
    for example_line in relevant_lines[:5]:
        preview = example_line[:200] + ("..." if len(example_line) > 200 else "")
        _log(f"  {preview}")

    return relevant_lines


def _find_routing_lines(
    log_lines: List[str],
    body_preview_substrings: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract routing_check and [local]/[remote] dispatch lines.

    Returns a list of dicts with keys: timestamp, kind, details.
    """
    results: List[Dict[str, Any]] = []

    for line in log_lines:
        # Match routing_check lines
        m = re.match(
            r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+-\s+\w+\s+-\s+(routing_check.*)$",
            line,
        )
        if m:
            ts = m.group(1)
            details = m.group(2)
            results.append({"timestamp": ts, "kind": "routing_check", "details": details})
            continue

        # Match [local] dispatch lines
        m = re.match(
            r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+-\s+\w+\s+-\s+(\[local\].*)$",
            line,
        )
        if m:
            ts = m.group(1)
            details = m.group(2)
            results.append({"timestamp": ts, "kind": "local_dispatch", "details": details})
            continue

        # Match [remote] dispatch lines
        m = re.match(
            r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+-\s+\w+\s+-\s+(\[remote\].*)$",
            line,
        )
        if m:
            ts = m.group(1)
            details = m.group(2)
            results.append({"timestamp": ts, "kind": "remote_dispatch", "details": details})
            continue

        # Match lease_renewed / lease_released lines
        m = re.match(
            r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+-\s+\w+\s+-\s+(lease_renewed.*)$",
            line,
        )
        if m:
            ts = m.group(1)
            details = m.group(2)
            results.append({"timestamp": ts, "kind": "lease_renewed", "details": details})
            continue

        m = re.match(
            r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+-\s+\w+\s+-\s+(lease_released.*)$",
            line,
        )
        if m:
            ts = m.group(1)
            details = m.group(2)
            results.append({"timestamp": ts, "kind": "lease_released", "details": details})
            continue

        # Match local_dispatch_denied lines
        m = re.match(
            r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+-\s+\w+\s+-\s+(local_dispatch_denied.*)$",
            line,
        )
        if m:
            ts = m.group(1)
            details = m.group(2)
            results.append({"timestamp": ts, "kind": "dispatch_denied", "details": details})
            continue

    # If body_preview_substrings is provided, filter to lines whose body
    # preview contains at least one of the substrings.
    if body_preview_substrings:
        filtered: List[Dict[str, Any]] = []
        for entry in results:
            details = entry.get("details", "")
            if any(sub in details for sub in body_preview_substrings):
                filtered.append(entry)
        return filtered

    return results


def _parse_routing_check(details: str) -> Dict[str, Any]:
    """Parse a routing_check log line into structured data.

    Example:
        routing_check provider=local-qwen3 model=Qwen3
            cache_cold=False estimated_tokens=42676
            cold_threshold=30000 warm_threshold=40000 messages=109
    """
    parsed: Dict[str, Any] = {"raw": details}
    # Extract key=value pairs
    for m in re.finditer(r"(\w+)=(\S+)", details):
        key = m.group(1)
        value = m.group(2)
        # Try numeric conversion
        try:
            if "." in value:
                parsed[key] = float(value)
            else:
                parsed[key] = int(value)
        except ValueError:
            parsed[key] = value
    return parsed


def _find_request_log_timestamps(session_id: str, log_lines: List[str]) -> Dict[str, str]:
    """Find the [local] or [remote] log line timestamp for a given session_id.

    Returns dict with keys: dispatch_timestamp.
    """
    result: Dict[str, str] = {}
    for line in log_lines:
        # Look for session_id in the line
        if session_id[:16] in line or session_id in line:
            m = re.match(
                r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+-\s+\w+\s+-\s+",
                line,
            )
            if m:
                result["dispatch_timestamp"] = m.group(1)
    return result


# ---------------------------------------------------------------------------
# Test result accumulator
# ---------------------------------------------------------------------------

_REQUEST_RESULTS: List[Dict[str, Any]] = []

# Track session IDs used in the current test run so their dispatch leases
# can be explicitly released after completion (LP-0MRFOF7XO003T7CT).
_SESSION_IDS_FOR_CLEANUP: List[str] = []


def _release_lease(session_id: str) -> None:
    """Release the dispatch lease for *session_id* via the proxy API.

    Calls ``POST /v1/leases/release`` which is idempotent — calling it
    for a session with no lease returns 200 with ``{"status": "ok"}``.
    """
    url = f"{BASE_URL}/v1/leases/release"
    try:
        resp = requests.post(
            url,
            json={"session_id": session_id},
            timeout=5,
        )
        if resp.status_code == 200:
            _log(f"Lease released for session {session_id[:16]}...")
        else:
            _log(
                f"Lease release returned {resp.status_code} for "
                f"session {session_id[:16]}..."
            )
    except requests.RequestException as exc:
        _log(f"Lease release failed for session {session_id[:16]}...: {exc}")


@pytest.fixture(scope="module", autouse=True)
def _release_all_leases():
    """Fixture that releases ALL tracked leases after the module completes.

    Runs once when the module finishes (success or failure). Releases
    each session ID registered in ``_SESSION_IDS_FOR_CLEANUP`` via the
    proxy's ``POST /v1/leases/release`` endpoint.
    """
    yield  # run the test(s)
    _log(f"Releasing {len(_SESSION_IDS_FOR_CLEANUP)} dispatch lease(s)...")
    for sid in _SESSION_IDS_FOR_CLEANUP:
        _release_lease(sid)
    _SESSION_IDS_FOR_CLEANUP.clear()


def _print_final_summary() -> None:
    """Print a structured summary of all request results."""
    _log("=" * 80)
    _log("MODEL AUDIT RESULTS SUMMARY")
    _log("=" * 80)

    for idx, r in enumerate(_REQUEST_RESULTS, start=1):
        _log(f"\n--- Request {idx}: {r.get('prompt_truncated', '')} ---")
        _log(f"  Status:       {r.get('status_code')}")
        _log(f"  Elapsed:      {r.get('elapsed_seconds')}s")
        _log(f"  Provider:     {r.get('provider')}")
        _log(f"  Model:        {r.get('model')}")
        _log(f"  Session:      {r.get('session_id')}")
        _log(f"  Time (UTC):   {r.get('timestamp_utc')}")
        _log("  Log analysis:")

        routing_events = r.get("routing_events", [])
        if routing_events:
            for event in routing_events:
                _log(f"    [{event.get('timestamp')}] {event.get('kind')}: {event.get('details')}")
        else:
            _log("    (no routing events found in log)")

    # If first two are NOT served by Qwen3, trace back leases
    first_model = _REQUEST_RESULTS[0].get("model", "") if len(_REQUEST_RESULTS) > 0 else ""
    second_model = _REQUEST_RESULTS[1].get("model", "") if len(_REQUEST_RESULTS) > 1 else ""
    first_is_qwen3 = "qwen3" in first_model.lower() or "qwen3" in _REQUEST_RESULTS[0].get("provider", "").lower()
    second_is_qwen3 = "qwen3" in second_model.lower() or "qwen3" in _REQUEST_RESULTS[1].get("provider", "").lower()

    _log("")
    _log("-" * 80)
    if not first_is_qwen3:
        _log("NOTE: Request 1 was NOT served by Qwen3.")
        _log("Lease analysis for request 1:")
        _log(f"  Lease events: {json.dumps(_REQUEST_RESULTS[0].get('lease_events', []), indent=2)}")
        _log(f"  Routing decision: {_REQUEST_RESULTS[0].get('routing_analysis', 'N/A')}")
    else:
        _log("Request 1 WAS served by Qwen3.")

    if not second_is_qwen3:
        _log("NOTE: Request 2 was NOT served by Qwen3.")
        _log("Lease analysis for request 2:")
        _log(f"  Lease events: {json.dumps(_REQUEST_RESULTS[1].get('lease_events', []), indent=2)}")
        _log(f"  Routing decision: {_REQUEST_RESULTS[1].get('routing_analysis', 'N/A')}")
    else:
        _log("Request 2 WAS served by Qwen3.")

    # If either was not Qwen3, do backward lease trace
    if not first_is_qwen3 or not second_is_qwen3:
        _log("")
        _log("=" * 80)
        _log("BACKWARD LEASE TRACE (for non-Qwen3 requests)")
        _log("=" * 80)
        for idx, r in enumerate(_REQUEST_RESULTS, start=1):
            model_lower = r.get("model", "").lower()
            provider_lower = r.get("provider", "").lower()
            is_qwen3 = "qwen3" in model_lower or "qwen3" in provider_lower
            if not is_qwen3:
                lease_events = r.get("lease_events_before", [])
                _log(f"\nRequest {idx} was routed to remote. Lease history before this request:")
                if lease_events:
                    for ev in lease_events:
                        _log(f"  [{ev.get('timestamp')}] {ev.get('kind')}: {ev.get('details')}")
                else:
                    _log("  (no lease events found in log window before this request)")
                _log(f"\nRouting analysis for request {idx}:")
                _log(f"  {r.get('routing_analysis', 'N/A')}")

    _log("")
    _log("=" * 80)
    _log("END OF MODEL AUDIT SUMMARY")
    _log("=" * 80)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_model_audit_plan_routing() -> None:
    """Make 3 sequential plan-model requests, measure timing, inspect logs.

    Test flow:
    1. Require live proxy
    2. Record a start timestamp (ISO) for log filtering
    3. Request 1: "Why Americans call Football Soccer" — 3 paragraphs
       Wait 15s
    4. Request 2: "Why Americans think the World Series can be called that
       when only American teams are in it" — 3 paragraphs
       Wait 15s
    5. Request 3: "The importance of being ernest" — 3 paragraphs
    6. For each: measure wall clock, record provider/model
    7. Read proxy log and find routing_check + dispatch lines
    8. Explain why each model was used
    9. If first two aren't Qwen3, trace back lease ownership
    """
    _require_local_proxy()

    # Record start time BEFORE any requests for log filtering
    start_iso = _now_iso()
    _log(f"Test start time (ISO): {start_iso}")

    prompts = [
        "Write three paragraphs on why Americans call Football Soccer.",
        "Write three paragraphs on why Americans think the World Series can be called that when only American teams are in it.",
        "Write three paragraphs on the importance of being ernest.",
    ]

    session_ids = [
        _new_session_id("audit-req1"),
        _new_session_id("audit-req2"),
        _new_session_id("audit-req3"),
    ]
    # Register all session IDs for lease cleanup after the test
    _SESSION_IDS_FOR_CLEANUP.extend(session_ids)

    # -----------------------------------------------------------------------
    # Phase 1: Send the three requests with 15s gaps BETWEEN STARTS, so
    # requests 1 & 2 overlap, and 2 & 3 overlap, testing concurrency.
    # -----------------------------------------------------------------------

    _log("Launching 3 requests with 15s gaps between starts (overlapping)...")

    results_lock: Dict[int, Any] = {}
    started_times: Dict[int, float] = {}

    def _send_request(index: int) -> None:
        """Send a single request and store its result."""
        started_times[index] = time.monotonic()
        _log(f"Request {index+1} STARTING at T+{time.monotonic() - started_times.get(0, time.monotonic()):.1f}s")
        resp, body, elapsed = _chat(
            prompt=prompts[index],
            session_id=session_ids[index],
            max_tokens=800,
        )
        results_lock[index] = {
            "response": resp,
            "body": body,
            "elapsed": elapsed,
            "completed_at": time.monotonic(),
        }
        provider = _provider_from_headers(resp)
        model = _model_from_response(body)
        text = _extract_response_text(body)
        _log(
            f"Request {index+1} COMPLETE after {elapsed:.1f}s",
            payload={
                "provider": provider,
                "model": model,
                "status_code": resp.status_code,
                "response_preview": text[:200] + ("..." if len(text) > 200 else ""),
            },
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit request 1 immediately
        future_1 = executor.submit(_send_request, 0)
        time.sleep(15)

        # Submit request 2 after 15s (overlaps with request 1)
        future_2 = executor.submit(_send_request, 1)
        time.sleep(15)

        # Submit request 3 after 30s (overlaps with request 2, possibly request 1)
        future_3 = executor.submit(_send_request, 2)

        # Wait for all 3 to complete
        for fut in (future_1, future_2, future_3):
            fut.result()

    # Process results in order
    for i in range(3):
        r = results_lock[i]
        response = r["response"]
        body = r["body"]
        elapsed = r["elapsed"]
        provider = _provider_from_headers(response)
        model = _model_from_response(body)
        text = _extract_response_text(body)

        _REQUEST_RESULTS.append({
            "index": i + 1,
            "session_id": session_ids[i],
            "prompt_truncated": prompts[i][:100] + ("..." if len(prompts[i]) > 100 else ""),
            "status_code": response.status_code,
            "elapsed_seconds": round(elapsed, 3),
            "provider": provider,
            "model": model,
            "response_text": text,
            "timestamp_utc": _now_iso(),
        })

    # -----------------------------------------------------------------------
    # Phase 2: Read proxy logs and analyse routing decisions
    # -----------------------------------------------------------------------

    _log("\n" + "=" * 60)
    _log("PHASE 2: Analysing proxy logs for routing decisions")
    _log("=" * 60)

    log_lines = _read_proxy_log_since(start_iso)

    if not log_lines:
        _log("WARNING: No proxy log lines available for analysis. "
             "Skipping log inspection.")
    else:
        _log(f"Read {len(log_lines)} log lines from proxy log")

    for idx, r in enumerate(_REQUEST_RESULTS):
        session_id = r["session_id"]
        _prompt_truncated = r["prompt_truncated"]

        _log(f"\n--- Analysing routing for request {idx+1} ---")
        _log(f"  Session: {session_id}")

        # Find routing and dispatch events
        routing_events = _find_routing_lines(
            log_lines,
            body_preview_substrings=[session_id[:16], f"session={session_id}"],
        )

        if not routing_events:
            # Broader search: look for ALL routing-related lines by
            # matching session_id or prompt text (since Fallback triggered
            # lines lack session IDs but are temporally correlated).
            _log("  Broader search for routing events...")
            all_log_lines = _read_proxy_log_since("")
            all_routing_events = _find_routing_lines(all_log_lines)
            prompt_text = r.get("prompt_truncated", "")[:60].rstrip(".")
            routing_events = [
                ev for ev in all_routing_events
                if session_id[:16] in ev.get("details", "")
                or session_id in ev.get("details", "")
                or prompt_text in ev.get("details", "")
            ]
            _log(f"  Broader search found {len(routing_events)} matching event(s)")
        else:
            # We already have events; also try to find Fallback triggered
            # lines that are temporally close (within 20s after dispatch).
            all_fb = _find_routing_lines(_read_proxy_log_since(""))
            all_fb = [ev for ev in all_fb if ev["kind"] == "fallback"]
            import datetime as _dt
            def _ts2epoch(ts: str) -> Optional[float]:
                try:
                    return _dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f").timestamp()
                except Exception:
                    return None
            dispatch_epochs = set()
            for ev in routing_events:
                e = _ts2epoch(ev["timestamp"])
                if e is not None:
                    dispatch_epochs.add(e)
            for fb in all_fb:
                fb_e = _ts2epoch(fb["timestamp"])
                if fb_e is None:
                    continue
                for de in dispatch_epochs:
                    if 0 <= (fb_e - de) <= 20.0:
                        routing_events.append(fb)
                        _log(f"  Added fallback: [{fb['timestamp']}] {fb['details'][:120]}")
                        break

        _REQUEST_RESULTS[idx]["routing_events"] = routing_events
        _log(f"  Found {len(routing_events)} routing event(s):")
        for ev in routing_events:
            _log(f"    [{ev.get('timestamp')}] {ev.get('kind')}: {ev.get('details')}")

        # Parse the routing_check events for explanation
        routing_analysis = ""
        for ev in routing_events:
            if ev["kind"] == "routing_check":
                parsed = _parse_routing_check(ev["details"])
                _log(f"  Parsed routing_check: provider={parsed.get('provider')} "
                     f"cache_cold={parsed.get('cache_cold')} "
                     f"estimated_tokens={parsed.get('estimated_tokens')} "
                     f"cold_threshold={parsed.get('cold_threshold')} "
                     f"warm_threshold={parsed.get('warm_threshold')}")
                routing_analysis += (
                    f"Routing check: provider={parsed.get('provider')}, "
                    f"cache_cold={parsed.get('cache_cold')}, "
                    f"estimated_tokens={parsed.get('estimated_tokens')}, "
                    f"cold_threshold={parsed.get('cold_threshold')}, "
                    f"warm_threshold={parsed.get('warm_threshold')}. "
                )
            elif ev["kind"] == "dispatch_denied":
                _log(f"  DISPATCH DENIED: {ev['details']}")
                routing_analysis += f"Local dispatch denied: {ev['details']}. "
            elif ev["kind"] == "local_dispatch":
                _log(f"  LOCAL DISPATCH: {ev['details']}")
                routing_analysis += "Routed to local backend."
            elif ev["kind"] == "remote_dispatch":
                _log(f"  REMOTE DISPATCH: {ev['details']}")
                routing_analysis += "Routed to remote backend."
            elif ev["kind"] == "fallback":
                _log(f"  FALLBACK: {ev['details']}")
                routing_analysis += f"Fallback: {ev['details']}. "
                m_reason = re.search(r"reason=(\S+)", ev.get("details", ""))
                if m_reason:
                    routing_analysis += f"Fallback reason: {m_reason.group(1)}. "
                    _log(f"  Fallback reason: {m_reason.group(1)}")
            elif ev["kind"] == "stream_started":
                _log(f"  STREAM STARTED: {ev['details']}")
                routing_analysis += f"Assigned to provider: {ev['details']}. "

        _REQUEST_RESULTS[idx]["routing_analysis"] = routing_analysis

        # Find lease events around this request
        lease_events = [
            ev for ev in routing_events
            if ev["kind"] in ("lease_renewed", "lease_released", "dispatch_denied")
        ]
        _REQUEST_RESULTS[idx]["lease_events"] = lease_events

    # -----------------------------------------------------------------------
    # Phase 3: If first two aren't Qwen3, trace backward for lease ownership
    # -----------------------------------------------------------------------

    first_model = _REQUEST_RESULTS[0].get("model", "")
    second_model = _REQUEST_RESULTS[1].get("model", "")
    first_provider = _REQUEST_RESULTS[0].get("provider", "")
    second_provider = _REQUEST_RESULTS[1].get("provider", "")

    first_is_qwen3 = "qwen3" in first_model.lower() or "qwen3" in first_provider.lower()
    second_is_qwen3 = "qwen3" in second_model.lower() or "qwen3" in second_provider.lower()

    _log("\n" + "=" * 60)
    _log("PHASE 3: Backward lease trace analysis")
    _log("=" * 60)

    if not first_is_qwen3 or not second_is_qwen3:
        _log("First two requests were NOT served by Qwen3. Tracing back lease ownership...")

        # Find ALL lease events in the log window (not filtered by session)
        all_lease_events = _find_routing_lines(log_lines)
        lease_history = [
            ev for ev in all_lease_events
            if ev["kind"] in ("lease_renewed", "lease_released", "dispatch_denied", "routing_check")
        ]

        _log("\nAll lease/dispatch events in the test log window:")
        for ev in lease_history:
            _log(f"  [{ev.get('timestamp')}] {ev.get('kind')}: {ev.get('details')}")

        # For each non-Qwen3 request, identify which session held the lease
        for idx_str in ("0", "1"):
            idx = int(idx_str)
            r = _REQUEST_RESULTS[idx]
            if not ("qwen3" in r.get("model", "").lower() or "qwen3" in r.get("provider", "").lower()):
                # Trace back: find the most recent dispatch_denied or
                # lease_renewed events before this request to understand
                # which session owned the local slot.
                r_ts = r.get("timestamp_utc", "")
                before_events = [
                    ev for ev in lease_history
                    if ev.get("timestamp", "") < _log_ts_from_iso(r_ts)
                    if ev["kind"] in ("lease_renewed", "lease_released", "dispatch_denied")
                ]
                # Take the last 10 events before this request
                before_events = before_events[-10:]
                _REQUEST_RESULTS[idx]["lease_events_before"] = before_events

                _log(f"\nLease events before request {idx+1} session={r['session_id']}:")
                for ev in before_events:
                    _log(f"  [{ev.get('timestamp')}] {ev.get('kind')}: {ev.get('details')}")

                # Try to find which session owned the lease
                owner_session = None
                for ev in reversed(before_events):
                    if ev["kind"] == "lease_renewed":
                        # Extract session from "lease_renewed session=<suffix> timeout=..."
                        m = re.search(r"session=(\S+)", ev.get("details", ""))
                        if m:
                            owner_session = m.group(1)
                            break
                    elif ev["kind"] == "dispatch_denied":
                        # Extract owner from "local_dispatch_denied session=<s> owner=<o>"
                        m = re.search(r"owner=(\S+)", ev.get("details", ""))
                        if m:
                            owner_session = m.group(1)
                            break

                _REQUEST_RESULTS[idx]["lease_owner_session"] = owner_session
                if owner_session:
                    _log(f"  Lease owned by session: {owner_session}")
                else:
                    _log("  No lease owner identified (possibly cache_cold bypass or no active lease)")
    else:
        _log("Both first two requests were served by Qwen3. No backward lease trace needed.")

    # -----------------------------------------------------------------------
    # Print final summary
    # -----------------------------------------------------------------------

    _print_final_summary()

    # -----------------------------------------------------------------------
    # Acceptance criteria assertions
    # -----------------------------------------------------------------------

    # All requests must be HTTP 200 with content
    for idx, r in enumerate(_REQUEST_RESULTS, start=1):
        assert r["status_code"] == 200, f"Request {idx}: expected 200"
        assert r["response_text"].strip(), f"Request {idx}: expected non-empty content"

    # Record model for every request
    for idx, r in enumerate(_REQUEST_RESULTS, start=1):
        assert r["provider"], f"Request {idx}: missing X-Provider header"
        assert r["model"], f"Request {idx}: missing model field in response body"

    # Log analysis: warn if no routing events found, but don't fail.
    # The routing_check log is only emitted when the local concurrency
    # check passes (provider.py:1762). If local_concurrency_limit is hit,
    # the code skips directly to the next provider without logging
    # routing_check.
    for idx, r in enumerate(_REQUEST_RESULTS, start=1):
        if log_lines and not r.get("routing_events"):
            _log(
                f"WARNING: Request {idx} has no routing events in proxy log. "
                f"Session={r['session_id']}. This is expected when the local "
                f"concurrency limit is hit before the routing_check log is emitted."
            )


def _log_ts_from_iso(iso_str: str) -> str:
    """Convert ISO timestamp to proxy log format for comparison."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso_str[:19].replace("T", " ")
