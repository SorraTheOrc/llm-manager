#!/usr/bin/env python3
"""Bounded controlled replay of a repeated prompt (LP-0MWRQVLFY003BK5L AC3).

Sends the **same** prompt twice through the proxy, each from a *fresh*
``X-Session-Id``, and records the two signals that prove the proxy cannot price
llama-server's cross-session prompt cache:

* the proxy's ``dispatch_first_byte_ms`` / ``cache_state`` / ``cached_ratio`` and
  the ``prefill_progress_measured`` line llama-server returned for that exact
  dispatch (matched by the unique session id);
* the llama-server ``prompt eval time`` lines appended during the replay.

The first request is genuinely cold (full prefill). The second uses a different
session, so the proxy's per-(model, session) ratio table has no entry and prices
it ``cold / 0.0`` — yet llama-server restores the identical prefix, so its
``prompt eval`` line reports only a handful of tokens and the first byte arrives
in a fraction of the first request's time. That is the "warm prefix the proxy
did not price" the investigation documents.

Bounded by construction: exactly two requests, small ``--max-tokens``, a
``--timeout`` and no proxy restart. The item's constraint (never add a
synchronous ``/slots`` call to the admission path; do not restart the proxy for
a replay when a warm session exists) is respected — this only speaks the normal
chat-completions protocol.

Usage::

    ./proxy/scripts/replay_cold_labelled_cache_hit.py            # human summary
    ./proxy/scripts/replay_cold_labelled_cache_hit.py --json     # machine readable

Exit codes: 0 replay recorded, 1 the replay could not be captured.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

try:  # imported both as a script (dir on sys.path) and from the tests
    import analyse_cold_labelled_cache_hits as analysis
except ImportError:  # pragma: no cover - defensive
    analysis = None  # type: ignore[assignment]

DEFAULT_PROXY_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "private"  # local-only alias; force_full_prompt sends the whole prompt
DEFAULT_LOG_DIR = analysis.DEFAULT_LOG_DIR if analysis else Path("/var/log/llama-proxy")
DEFAULT_TARGET_TOKENS = 4000

#: One deterministic filler line; repeated so the prompt token count is a
#: function of ``--target-tokens`` and identical across runs.
_FILLER_LINE = (
    "item{n:04d} alpha beta gamma delta epsilon zeta eta theta iota kappa"
)


def build_prompt(target_tokens: int) -> str:
    """Return a deterministic prompt of roughly *target_tokens* tokens.

    Roughly four characters per token, so ``target_tokens * 16`` characters of
    filler line (each line ~65 chars) yields close to the requested size. The
    exact token count is reported from llama-server's own measurement, never
    assumed.
    """
    target_chars = max(1, target_tokens) * 4
    line_len = len(_FILLER_LINE.format(n=0)) + 1
    count = max(1, target_chars // line_len)
    filler = "\n".join(_FILLER_LINE.format(n=i) for i in range(count))
    return (
        "Read the following reference and answer with the single word OK.\n\n"
        f"{filler}\n\nQuestion: reply with the single word OK."
    )


def _chat_body(model: str, prompt: str, max_tokens: int) -> bytes:
    return json.dumps(
        {
            "model": model,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")


def send_once(
    proxy_url: str,
    model: str,
    prompt: str,
    session: str,
    max_tokens: int = 8,
    timeout: float = 180.0,
) -> dict:
    """Send one streaming request; return client-side first-byte timings."""
    request = urllib.request.Request(
        proxy_url.rstrip("/") + "/v1/chat/completions",
        data=_chat_body(model, prompt, max_tokens),
        headers={"Content-Type": "application/json", "X-Session-Id": session},
        method="POST",
    )
    started = time.time()
    first_byte: float | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for chunk in response:
            if first_byte is None and chunk.strip():
                first_byte = time.time() - started
    return {
        "session": session,
        "client_first_byte_s": round(first_byte, 3) if first_byte is not None else None,
        "client_total_s": round(time.time() - started, 3),
    }


def _read_from(path: Path, offset: int) -> tuple[list[str], int]:
    """Return lines appended to *path* since byte *offset* (and the new offset)."""
    if not path.is_file():
        return [], offset
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    return data.decode("utf-8", "replace").splitlines(), offset + len(data)


def _offset(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def _extract_replay_lines(
    proxy_lines: list[str], sessions: list[str]
) -> dict[str, dict]:
    """Pull each replay session's dispatch + measurement from the proxy lines."""
    out: dict[str, dict] = {session: {} for session in sessions}
    for line in proxy_lines:
        dispatch = analysis.parse_dispatch(line)
        if dispatch is not None and dispatch.session in out:
            out[dispatch.session]["dispatch"] = {
                "cache_state": dispatch.cache_state,
                "cached_ratio": dispatch.cached_ratio,
                "estimated_tokens": dispatch.estimated_tokens,
                "dispatch_first_byte_ms": dispatch.first_byte_ms,
                "dispatch_to_first_byte_ms": dispatch.to_first_byte_ms,
            }
            continue
        measurement = analysis.parse_measurement(line)
        if measurement is not None and measurement.session in out:
            out[measurement.session]["measurement"] = {
                "total": measurement.total,
                "cache": measurement.cache,
                "measured_cached_ratio": measurement.measured_cached_ratio,
                "measured_new_tokens": measurement.measured_new_tokens,
                "priced_new_tokens": measurement.priced_new_tokens,
            }
    return out


def _extract_prompt_eval(llama_lines: list[str]) -> list[dict]:
    """Every ``prompt eval time`` line appended during the replay."""
    out: list[dict] = []
    for line in llama_lines:
        entry = analysis.parse_prompt_eval(line)
        if entry is not None:
            out.append(entry)
    return out


def run_replay(
    proxy_url: str = DEFAULT_PROXY_URL,
    model: str = DEFAULT_MODEL,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    log_dir: Path = DEFAULT_LOG_DIR,
    max_tokens: int = 8,
    timeout: float = 180.0,
    retries: int = 3,
    retry_delay: float = 20.0,
) -> dict:
    """Run the two-request replay and return the recorded evidence.

    The proxy answers a local-only model with a *retryable* 503 when another
    prefill is in flight (``local_dispatch_denied``); the two requests are re-
    sent as a pair after a bounded delay so the replay still runs on a busy
    host without disturbing the in-flight work. *retries* bounds the attempts.
    """
    prompt = build_prompt(target_tokens)
    proxy_log = Path(log_dir) / "proxy.log"
    llama_log = Path(log_dir) / "llama-server.log"

    attempts = 0
    while True:
        attempts += 1
        proxy_offset = _offset(proxy_log)
        llama_offset = _offset(llama_log)

        session_a = f"replay-cold-a-{uuid.uuid4().hex[:8]}"
        session_b = f"replay-cold-b-{uuid.uuid4().hex[:8]}"
        try:
            first = send_once(proxy_url, model, prompt, session_a, max_tokens, timeout)
            second = send_once(proxy_url, model, prompt, session_b, max_tokens, timeout)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 503 or attempts > retries:
                raise
            time.sleep(retry_delay)
        except urllib.error.URLError:
            if attempts > retries:
                raise
            time.sleep(retry_delay)

    # Give the proxy/logging a moment to flush the final measured line.
    time.sleep(0.5)
    proxy_lines, _ = _read_from(proxy_log, proxy_offset)
    llama_lines, _ = _read_from(llama_log, llama_offset)

    sessions = [session_a, session_b]
    captured = _extract_replay_lines(proxy_lines, sessions)
    evals = _extract_prompt_eval(llama_lines)

    cold = captured.get(session_a, {})
    warm = captured.get(session_b, {})
    warm_measurement = warm.get("measurement") or {}
    warm_dispatch = warm.get("dispatch") or {}
    verdict = {
        "second_session_priced_cold": warm_dispatch.get("cache_state") == "cold"
        and warm_dispatch.get("cached_ratio") == 0.0,
        "second_session_served_warm": (
            warm_measurement.get("measured_cached_ratio", 0.0) or 0.0
        )
        >= 0.5,
        "second_session_measured_cached_ratio": warm_measurement.get(
            "measured_cached_ratio"
        ),
        "first_byte_speedup": (
            round(first["client_first_byte_s"] / second["client_first_byte_s"], 1)
            if first.get("client_first_byte_s") and second.get("client_first_byte_s")
            else None
        ),
    }
    verdict["demonstrated"] = bool(
        verdict["second_session_priced_cold"]
        and verdict["second_session_served_warm"]
    )

    return {
        "proxy_url": proxy_url,
        "model": model,
        "prompt_chars": len(prompt),
        "requests": [
            {"label": "cold (first session)", **first, **cold},
            {"label": "warm (second, fresh session)", **second, **warm},
        ],
        "llama_prompt_eval_lines": evals,
        "verdict": verdict,
    }


def _render(result: dict) -> str:
    lines = [
        "bounded controlled replay — cold-labelled prompt-cache hit",
        "=" * 58,
        f"proxy / model                 : {result['proxy_url']} / {result['model']}",
        f"prompt size (chars)           : {result['prompt_chars']:,}",
    ]
    for request in result["requests"]:
        lines.append("")
        lines.append(f"[{request['label']}]")
        lines.append(f"  session                     : {request['session']}")
        lines.append(
            f"  first byte (client)         : {request['client_first_byte_s']} s "
            f"(total {request['client_total_s']} s)"
        )
        dispatch = request.get("dispatch") or {}
        measurement = request.get("measurement") or {}
        if dispatch:
            lines.append(
                f"  proxy label                 : cache_state={dispatch['cache_state']} "
                f"cached_ratio={dispatch['cached_ratio']} "
                f"estimated_tokens={dispatch['estimated_tokens']}"
            )
            lines.append(
                f"  proxy dispatch first byte   : {dispatch['dispatch_first_byte_ms']} ms"
            )
        if measurement:
            lines.append(
                f"  llama-server reuse          : cache={measurement['cache']} / "
                f"total={measurement['total']} "
                f"(measured_cached_ratio={measurement['measured_cached_ratio']}, "
                f"new_tokens={measurement['measured_new_tokens']})"
            )
    if result["llama_prompt_eval_lines"]:
        lines.append("")
        lines.append("llama-server prompt eval lines appended during replay")
        lines.append("-" * 58)
        for entry in result["llama_prompt_eval_lines"]:
            lines.append(f"  {entry['line']}")
    lines.append("")
    verdict = result["verdict"]
    lines.append(
        "verdict: second session priced "
        f"{'COLD' if verdict['second_session_priced_cold'] else 'warm'} but served "
        f"{'WARM' if verdict['second_session_served_warm'] else 'cold'} "
        f"(measured_cached_ratio={verdict['second_session_measured_cached_ratio']}, "
        f"first-byte speedup x{verdict['first_byte_speedup']})"
    )
    lines.append(f"demonstrated: {verdict['demonstrated']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--proxy-url", default=DEFAULT_PROXY_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--retries", type=int, default=3,
        help="retry the request pair on the proxy's retryable 503 (capacity)",
    )
    parser.add_argument("--retry-delay", type=float, default=20.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if analysis is None:  # pragma: no cover - only if the sibling script is missing
        print("error: analyse_cold_labelled_cache_hits.py not importable", file=sys.stderr)
        return 1

    try:
        result = run_replay(
            proxy_url=args.proxy_url,
            model=args.model,
            target_tokens=args.target_tokens,
            log_dir=args.log_dir,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            retry_delay=args.retry_delay,
        )
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: replay failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=False))
    else:
        print(_render(result))
    return 0 if result["verdict"].get("second_session_served_warm") else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
