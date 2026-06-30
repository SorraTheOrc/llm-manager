"""Live end-to-end integration tests for the `plan` model fallback path.

These tests are intentionally skipped by default and only run on demand
against a live proxy instance started by the operator (for example via
`proxy/scripts/start-proxy.sh`).

Run manually:

    RUN_LIVE_PROXY_E2E=1 pytest -q tests/test_plan_fallback_live_e2e.py -m e2e_live -s

Optional env vars:
- LIVE_PROXY_BASE_URL (default: http://localhost:8000)
- LIVE_PROXY_TIMEOUT_SECONDS (default: 180)
"""

import gzip
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import pytest
import requests
from requests import Response
from requests.exceptions import RequestException


pytestmark = [pytest.mark.integration, pytest.mark.e2e_live]

if os.getenv("RUN_LIVE_PROXY_E2E", "0") != "1":
    pytest.skip(
        "live E2E tests are disabled; set RUN_LIVE_PROXY_E2E=1 to run on demand",
        allow_module_level=True,
    )


LOGGER = logging.getLogger("plan-live-e2e")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [plan-live-e2e] %(message)s")
    )
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.DEBUG)


BASE_URL = os.getenv("LIVE_PROXY_BASE_URL", "http://localhost:8000").rstrip("/")
DEFAULT_TIMEOUT = float(os.getenv("LIVE_PROXY_TIMEOUT_SECONDS", "180"))
LOCAL_QWEN3_RETRY_ATTEMPTS = int(os.getenv("LIVE_E2E_LOCAL_RETRY_ATTEMPTS", "4"))
LOCAL_QWEN3_RETRY_DELAY_SECONDS = float(os.getenv("LIVE_E2E_LOCAL_RETRY_DELAY_SECONDS", "1.5"))


_PHASE_STATE: Dict[str, Any] = {
    "phase_1_passed": False,
    "phase_2_passed": False,
    "phase_3_passed": False,
    "phase_4_passed": False,
    "phase_1_session_id": None,
    "phase_3_remote_session_id": None,
    "phase_3_remote_provider": None,
}

# End-of-run trace data used for final grouped summary output.
_REQUEST_TRACES: List[Dict[str, Any]] = []
_LATEST_SUMMARY_PAYLOAD: Optional[Dict[str, Any]] = None


def _log(message: str, *, payload: Optional[Dict[str, Any]] = None) -> None:
    if payload is None:
        LOGGER.info(message)
        print(f"[plan-live-e2e] {message}")
        return

    rendered = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    LOGGER.info("%s\n%s", message, rendered)
    print(f"[plan-live-e2e] {message}\n{rendered}")


def _new_session_id(prefix: str) -> str:
    """Create a unique session id containing a timestamp and UUID."""
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{prefix}-{ts}-{uuid.uuid4()}"


def _require_local_proxy() -> None:
    _log(f"Checking proxy health at {BASE_URL}/health")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
    except RequestException as exc:
        pytest.skip(f"live proxy not reachable at {BASE_URL}: {exc}")

    if response.status_code != 200:
        pytest.skip(
            f"live proxy health check failed at {BASE_URL}/health "
            f"status={response.status_code} body={response.text}"
        )

    _log("Proxy health check passed", payload={"status_code": response.status_code})


def _admin_metrics() -> Dict[str, Any]:
    url = f"{BASE_URL}/admin/metrics"
    _log(f"Fetching admin metrics: {url}")
    response = requests.get(url, timeout=10)
    data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    _log(
        "Admin metrics response",
        payload={
            "status_code": response.status_code,
            "restore_success_total": data.get("restore_success_total"),
            "restore_fallback_total": data.get("restore_fallback_total"),
            "backend_ready": data.get("backend_ready"),
        },
    )
    return data


def _extract_response_text(body: Dict[str, Any]) -> str:
    """Extract human-usable text from OpenAI-style responses.

    Accepts standard assistant.content and, when empty, reasoning_content as a
    pragmatic fallback for models that emit reasoning-only payloads.
    """
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
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        pieces.append(text)
            if pieces:
                return "\n".join(pieces)

        reasoning = msg.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning

        return ""
    except Exception:
        return ""


def _provider_from_response(response: Response) -> str:
    return (response.headers.get("X-Provider") or "").strip()


def _clip_text(value: str, limit: int = 500) -> str:
    if not isinstance(value, str):
        return ""
    if len(value) <= limit:
        return value
    return value[:limit] + f" … [truncated {len(value) - limit} chars]"


def _response_text_for_summary(body: Dict[str, Any]) -> str:
    text = _extract_response_text(body)
    if text.strip():
        return text
    try:
        return json.dumps(body, ensure_ascii=False, default=str)
    except Exception:
        return str(body)


def _record_request_trace(
    *,
    requested_session_id: Optional[str],
    response: Response,
    body: Dict[str, Any],
    prompt: str,
    elapsed: float,
) -> None:
    summary_session_id = _session_id_from_response(
        response,
        requested_session_id or "no-session",
    )
    model_name = str(body.get("model", "") or "")
    provider_name = _provider_from_response(response)
    response_text = _response_text_for_summary(body)

    _REQUEST_TRACES.append(
        {
            "session_id": summary_session_id,
            "requested_session_id": requested_session_id,
            "response_session_id": response.headers.get("X-Session-Id"),
            "status_code": int(response.status_code),
            "duration_seconds": round(float(elapsed), 3),
            "provider": provider_name,
            "model": model_name,
            "request_sent": _clip_text(prompt, 500),
            "response_received": _clip_text(response_text, 500),
        }
    )


def _build_summary_payload() -> Dict[str, Any]:
    if not _REQUEST_TRACES:
        return {
            "base_url": BASE_URL,
            "total_requests": 0,
            "sessions": {},
        }

    sessions: Dict[str, List[Dict[str, Any]]] = {}
    for trace in _REQUEST_TRACES:
        sid = str(trace.get("session_id") or "no-session")
        sessions.setdefault(sid, []).append(trace)

    summary_payload: Dict[str, Any] = {
        "base_url": BASE_URL,
        "total_requests": len(_REQUEST_TRACES),
        "sessions": {},
    }

    for session_id, traces in sessions.items():
        summary_payload["sessions"][session_id] = []
        for index, trace in enumerate(traces, start=1):
            summary_payload["sessions"][session_id].append(
                {
                    "sequence": index,
                    "status_code": trace.get("status_code"),
                    "duration_seconds": trace.get("duration_seconds"),
                    "model_responding": trace.get("model"),
                    "provider": trace.get("provider"),
                    "request_sent": trace.get("request_sent"),
                    "response_received": trace.get("response_received"),
                }
            )
    return summary_payload


def _render_summary_text(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Base URL: {payload.get('base_url', BASE_URL)}")
    lines.append(f"Total requests: {payload.get('total_requests', 0)}")

    sessions = payload.get("sessions", {}) if isinstance(payload, dict) else {}
    if not sessions:
        lines.append("No sessions recorded.")
        return "\n".join(lines)

    for session_id, entries in sessions.items():
        entries_list = entries if isinstance(entries, list) else []
        lines.append("")
        lines.append(f"Session: {session_id} ({len(entries_list)} request(s))")
        for item in entries_list:
            seq = item.get("sequence", "?") if isinstance(item, dict) else "?"
            status = item.get("status_code", "?") if isinstance(item, dict) else "?"
            duration = item.get("duration_seconds", "?") if isinstance(item, dict) else "?"
            provider = item.get("provider", "") if isinstance(item, dict) else ""
            model = item.get("model_responding", "") if isinstance(item, dict) else ""
            request_sent = item.get("request_sent", "") if isinstance(item, dict) else ""
            response_received = item.get("response_received", "") if isinstance(item, dict) else ""

            lines.append(
                f"  [{seq}] status={status} duration={duration}s provider={provider or 'n/a'} model={model or 'n/a'}"
            )
            lines.append(f"      Request: {request_sent}")
            lines.append(f"      Response: {response_received}")

    return "\n".join(lines)


def _print_end_summary() -> None:
    global _LATEST_SUMMARY_PAYLOAD

    _LATEST_SUMMARY_PAYLOAD = _build_summary_payload()
    summary_text = _render_summary_text(_LATEST_SUMMARY_PAYLOAD)

    LOGGER.info("END-OF-TEST SUMMARY (grouped by session)\n%s", summary_text)
    print("[plan-live-e2e] END-OF-TEST SUMMARY (grouped by session)")
    print(summary_text)


@pytest.fixture(scope="module", autouse=True)
def _emit_summary_after_module() -> None:
    """Print grouped request/response summary when all phases finish."""
    yield
    _print_end_summary()


def _session_id_from_response(response: Response, fallback: str) -> str:
    return (response.headers.get("X-Session-Id") or fallback).strip()


def _response_has_restore_signal(response: Response, body: Dict[str, Any]) -> bool:
    header_candidates = [
        "X-Llama-Session-Restored",
        "X-Session-Restored",
        "X-Llama-Cache-Restored",
        "X-KV-Cache-Restored",
        "X-Cache-Restored",
    ]
    for name in header_candidates:
        value = (response.headers.get(name) or "").strip().lower()
        if value in {"1", "true", "yes", "restored", "hit"}:
            return True

    if isinstance(body, dict):
        for field in ("session_restored", "cache_restored", "restore_success", "kv_cache_restored"):
            if body.get(field) is True:
                return True

    return False


def _wait_for_restore_success_increment(before: int, timeout_seconds: float = 12.0) -> Tuple[bool, int]:
    _log(
        "Polling /admin/metrics for restore_success_total increment",
        payload={"before": before, "timeout_seconds": timeout_seconds},
    )
    deadline = time.monotonic() + timeout_seconds
    current = before
    while time.monotonic() < deadline:
        try:
            data = _admin_metrics()
            current = int(data.get("restore_success_total", 0) or 0)
            if current > before:
                _log(
                    "restore_success_total increment detected",
                    payload={"before": before, "after": current},
                )
                return True, current
        except Exception as exc:
            _log("Failed to poll restore_success_total; retrying", payload={"error": str(exc)})
        time.sleep(0.8)

    _log(
        "restore_success_total did not increment before timeout",
        payload={"before": before, "after": current},
    )
    return False, current


def _is_local_qwen3_response(response: Response, body: Dict[str, Any]) -> Tuple[bool, str, str]:
    provider = _provider_from_response(response)
    model_field = str(body.get("model", "")).strip().lower()
    looks_local_qwen3 = (
        provider == "local-qwen3"
        or ("local" in provider.lower() and "qwen3" in provider.lower())
        or ("qwen3" in model_field)
    )
    return looks_local_qwen3, provider, model_field


def _assert_local_qwen3(response: Response, body: Dict[str, Any], *, phase: str) -> None:
    looks_local_qwen3, provider, model_field = _is_local_qwen3_response(response, body)

    _log(
        f"{phase}: validating local qwen3 provider",
        payload={"provider": provider, "model_field": model_field},
    )

    assert provider, f"{phase}: expected X-Provider header to be present"
    assert looks_local_qwen3, (
        f"{phase}: expected local qwen3 response, got provider={provider!r} model={model_field!r}"
    )


def _chat_until_local_qwen3(
    *,
    phase: str,
    prompt: str,
    max_tokens: int,
    attempts: int = LOCAL_QWEN3_RETRY_ATTEMPTS,
    delay_seconds: float = LOCAL_QWEN3_RETRY_DELAY_SECONDS,
) -> Tuple[Response, Dict[str, Any], float, str]:
    """Retry phase request until local qwen3 answers or attempts exhausted."""
    attempts = max(1, int(attempts))
    last_observation: Dict[str, Any] = {}

    for attempt in range(1, attempts + 1):
        candidate_session_id = _new_session_id(f"{phase}-session")
        response, body, elapsed = _chat(
            prompt=prompt,
            session_id=candidate_session_id,
            max_tokens=max_tokens,
        )

        looks_local_qwen3, provider, model_field = _is_local_qwen3_response(response, body)
        has_content = bool(_extract_response_text(body).strip())

        last_observation = {
            "attempt": attempt,
            "status_code": int(response.status_code),
            "provider": provider,
            "model_field": model_field,
            "has_content": has_content,
            "session_id": candidate_session_id,
        }
        _log(f"{phase}: local-qwen3 retry observation", payload=last_observation)

        if response.status_code == 200 and has_content and looks_local_qwen3:
            return response, body, elapsed, candidate_session_id

        if attempt < attempts:
            time.sleep(max(0.0, float(delay_seconds)))

    pytest.fail(
        f"{phase}: expected local qwen3 response after {attempts} attempts; "
        f"last_observation={last_observation}"
    )


def _assert_remote_provider(response: Response, *, phase: str) -> str:
    provider = _provider_from_response(response)
    _log(f"{phase}: validating remote provider", payload={"provider": provider})

    assert provider, f"{phase}: expected X-Provider header for provider provenance"
    assert not provider.lower().startswith("local-"), (
        f"{phase}: expected remote provider, got local provider {provider!r}"
    )
    assert "qwen3" not in provider.lower(), (
        f"{phase}: expected non-qwen3 remote provider, got {provider!r}"
    )
    return provider


def _chat(
    *,
    prompt: str,
    session_id: Optional[str] = None,
    max_tokens: int = 160,
    temperature: float = 0.0,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[Response, Dict[str, Any], float]:
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
        # Use identity encoding to reduce ambiguity when upstream/proxy
        # headers are inconsistent under concurrent fallback paths.
        "Accept-Encoding": "identity",
    }
    if session_id:
        headers["X-Session-Id"] = session_id

    _log(
        "Sending chat request",
        payload={
            "url": url,
            "headers": headers,
            "payload": payload,
            "timeout": timeout,
        },
    )

    started = time.monotonic()
    response = requests.post(url, json=payload, headers=headers, timeout=timeout, stream=True)
    elapsed = time.monotonic() - started

    raw_bytes = response.raw.read(decode_content=False)
    response.close()

    content_encoding = (response.headers.get("Content-Encoding") or "").lower()
    decoded_bytes = raw_bytes
    gzip_decode_error = None
    if "gzip" in content_encoding:
        try:
            decoded_bytes = gzip.decompress(raw_bytes)
        except Exception as exc:
            gzip_decode_error = str(exc)
            decoded_bytes = raw_bytes

    # Populate response.content so downstream asserts/messages can use response.text/body.
    response._content = decoded_bytes  # type: ignore[attr-defined]

    body: Dict[str, Any]
    try:
        body = json.loads(decoded_bytes.decode("utf-8")) if decoded_bytes else {}
    except Exception:
        body = {
            "_raw": decoded_bytes.decode("utf-8", errors="replace"),
            "_content_encoding": content_encoding,
            "_gzip_decode_error": gzip_decode_error,
        }

    _log(
        "Received chat response",
        payload={
            "status_code": response.status_code,
            "elapsed_seconds": round(elapsed, 3),
            "content_encoding": content_encoding,
            "gzip_decode_error": gzip_decode_error,
            "x_provider": response.headers.get("X-Provider"),
            "x_session_id": response.headers.get("X-Session-Id"),
            "x_session_created": response.headers.get("X-Session-Created"),
            "x_session_delta": response.headers.get("X-Session-Delta"),
            "x_session_fallback_reason": response.headers.get("X-Session-Fallback-Reason"),
            "response_body": body,
        },
    )

    _record_request_trace(
        requested_session_id=session_id,
        response=response,
        body=body,
        prompt=prompt,
        elapsed=elapsed,
    )
    return response, body, elapsed


def _assert_ok_with_content(response: Response, body: Dict[str, Any], *, phase: str) -> None:
    assert response.status_code == 200, (
        f"{phase}: expected HTTP 200, got {response.status_code} body={response.text}"
    )
    text = _extract_response_text(body)
    assert text.strip(), f"{phase}: expected non-empty assistant content, body={body}"


def _require_precondition(phase_num: int) -> None:
    if phase_num == 1:
        return
    previous_key = f"phase_{phase_num - 1}_passed"
    if not _PHASE_STATE.get(previous_key):
        pytest.fail(
            f"Phase {phase_num} precondition failed: Phase {phase_num - 1} did not pass. "
            f"Aborting remaining phases as requested."
        )


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def test_phase_1_single_query_local_qwen3() -> None:
    """Phase 1: Single query to plan model, validate qwen3/local provenance."""
    _require_precondition(1)
    _require_local_proxy()

    response, body, _elapsed, session_id = _chat_until_local_qwen3(
        phase="phase_1",
        prompt=(
            "You are performing a diagnostics handshake. Reply with exactly: "
            "PHASE1_OK and one short sentence."
        ),
        max_tokens=80,
    )

    _assert_ok_with_content(response, body, phase="phase_1")
    _assert_local_qwen3(response, body, phase="phase_1")

    effective_session_id = _session_id_from_response(response, session_id)
    _PHASE_STATE["phase_1_session_id"] = effective_session_id
    _PHASE_STATE["phase_1_passed"] = True

    _log(
        "Phase 1 completed",
        payload={
            "requested_session_id": session_id,
            "effective_session_id": effective_session_id,
            "x_provider": _provider_from_response(response),
        },
    )


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def test_phase_2_followup_uses_kv_cache_and_stays_local_qwen3() -> None:
    """Phase 2: Follow-up in same session, validate cache use + qwen3/local."""
    _require_precondition(2)

    session_id = _PHASE_STATE["phase_1_session_id"]
    assert isinstance(session_id, str) and session_id, "phase_2: missing phase_1 session id"

    before_metrics = _admin_metrics()
    before_restore_success = int(before_metrics.get("restore_success_total", 0) or 0)

    response, body, _elapsed = _chat(
        prompt=(
            "Follow-up continuity check. Refer to the prior PHASE1_OK handshake "
            "and answer in 2 short bullet points."
        ),
        session_id=session_id,
        max_tokens=140,
    )

    _assert_ok_with_content(response, body, phase="phase_2")
    _assert_local_qwen3(response, body, phase="phase_2")

    returned_session_id = _session_id_from_response(response, session_id)
    assert returned_session_id == session_id, (
        "phase_2: expected same session id for continuity, "
        f"got returned={returned_session_id} expected={session_id}"
    )

    created_header = (response.headers.get("X-Session-Created") or "").strip().lower()
    assert created_header in {"false", "0", ""}, (
        f"phase_2: expected existing session (X-Session-Created=false), got {created_header!r}"
    )

    restore_signal_in_response = _response_has_restore_signal(response, body)
    delta_header = (response.headers.get("X-Session-Delta") or "").strip().lower() == "true"
    restore_incremented, after_restore_success = _wait_for_restore_success_increment(
        before_restore_success,
        timeout_seconds=12.0,
    )

    _log(
        "Phase 2 cache evidence summary",
        payload={
            "restore_signal_in_response": restore_signal_in_response,
            "x_session_delta_true": delta_header,
            "restore_success_before": before_restore_success,
            "restore_success_after": after_restore_success,
            "restore_success_incremented": restore_incremented,
        },
    )

    assert (
        restore_signal_in_response or delta_header or restore_incremented
    ), (
        "phase_2: expected evidence of KV/session cache use "
        "(restore headers/json, delta header, or restore_success_total increment)"
    )

    _PHASE_STATE["phase_2_passed"] = True
    _log("Phase 2 completed", payload={"session_id": session_id})


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------

def test_phase_3_near_simultaneous_queries_first_local_second_remote() -> None:
    """Phase 3: Near-simultaneous requests; first local qwen3, second remote fallback."""
    _require_precondition(3)

    primary_session_id = _PHASE_STATE["phase_1_session_id"]
    assert isinstance(primary_session_id, str) and primary_session_id, "phase_3: missing primary session id"

    remote_session_id = _new_session_id("plan-live-e2e-remote")

    def _primary_call():
        return _chat(
            prompt=(
                "Generate a detailed numbered troubleshooting checklist with 40 items "
                "for operating a local LLM proxy. Keep each item concise but complete."
            ),
            session_id=primary_session_id,
            max_tokens=900,
            timeout=max(240.0, DEFAULT_TIMEOUT),
        )

    def _secondary_call():
        return _chat(
            prompt="Provide a one-paragraph status summary for phase 3 secondary call.",
            session_id=remote_session_id,
            max_tokens=180,
            timeout=max(120.0, DEFAULT_TIMEOUT),
        )

    _log(
        "Phase 3 launching near-simultaneous calls",
        payload={
            "primary_session_id": primary_session_id,
            "secondary_session_id": remote_session_id,
        },
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        primary_future = executor.submit(_primary_call)
        time.sleep(0.25)
        secondary_future = executor.submit(_secondary_call)
        primary_response, primary_body, _ = primary_future.result()
        secondary_response, secondary_body, _ = secondary_future.result()

    _assert_ok_with_content(primary_response, primary_body, phase="phase_3.primary")
    _assert_local_qwen3(primary_response, primary_body, phase="phase_3.primary")

    _assert_ok_with_content(secondary_response, secondary_body, phase="phase_3.secondary")
    remote_provider = _assert_remote_provider(secondary_response, phase="phase_3.secondary")

    effective_remote_session_id = _session_id_from_response(secondary_response, remote_session_id)
    _PHASE_STATE["phase_3_remote_session_id"] = effective_remote_session_id
    _PHASE_STATE["phase_3_remote_provider"] = remote_provider
    _PHASE_STATE["phase_3_passed"] = True

    _log(
        "Phase 3 completed",
        payload={
            "remote_provider": remote_provider,
            "remote_session_id": effective_remote_session_id,
        },
    )


# ---------------------------------------------------------------------------
# Phase 4
# ---------------------------------------------------------------------------

def test_phase_4_new_session_plus_remote_followup_validate_both() -> None:
    """Phase 4: New session query + simultaneous follow-up to prior remote query."""
    _require_precondition(4)

    remote_session_id = _PHASE_STATE.get("phase_3_remote_session_id")
    assert isinstance(remote_session_id, str) and remote_session_id, "phase_4: missing remote session id from phase 3"

    new_session_id = _new_session_id("plan-live-e2e-new")

    def _new_session_call():
        return _chat(
            prompt="Start a fresh session and respond with PHASE4_NEW_SESSION_OK plus one sentence.",
            session_id=new_session_id,
            max_tokens=140,
            timeout=max(120.0, DEFAULT_TIMEOUT),
        )

    def _remote_followup_call():
        return _chat(
            prompt=(
                "Follow up the prior remote conversation with one concise paragraph "
                "and include token REMOTE_FOLLOWUP_OK."
            ),
            session_id=remote_session_id,
            max_tokens=180,
            timeout=max(120.0, DEFAULT_TIMEOUT),
        )

    _log(
        "Phase 4 launching simultaneous calls",
        payload={
            "new_session_id": new_session_id,
            "remote_followup_session_id": remote_session_id,
            "phase_3_remote_provider": _PHASE_STATE.get("phase_3_remote_provider"),
        },
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        new_future = executor.submit(_new_session_call)
        remote_future = executor.submit(_remote_followup_call)
        new_response, new_body, _ = new_future.result()
        remote_response, remote_body, _ = remote_future.result()

    _assert_ok_with_content(new_response, new_body, phase="phase_4.new_session")
    new_session_provider = _provider_from_response(new_response)
    assert new_session_provider, "phase_4.new_session: expected X-Provider header"

    _assert_ok_with_content(remote_response, remote_body, phase="phase_4.remote_followup")
    remote_provider = _assert_remote_provider(remote_response, phase="phase_4.remote_followup")

    _PHASE_STATE["phase_4_passed"] = True
    _log(
        "Phase 4 completed",
        payload={
            "new_session_provider": new_session_provider,
            "remote_followup_provider": remote_provider,
            "all_phases_passed": True,
        },
    )
