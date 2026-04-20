#!/usr/bin/env python3
"""
Performance benchmark for session-based incremental prompt ingestion.

Measures and compares ingestion time for multi-turn conversations
with and without session reuse.

Usage:
    python benchmark_session_ingestion.py [--url URL] [--model MODEL] [--turns TURNS] [--runs RUNS]

Examples:
    # Default benchmark (3 turns, 2 runs)
    python benchmark_session_ingestion.py

    # Longer benchmark (5 turns, 3 runs)
    python benchmark_session_ingestion.py --turns 5 --runs 3

    # Specify model and URL
    python benchmark_session_ingestion.py --model qwen3 --url http://localhost:8000
"""

import argparse
import json
import sys
import time
from typing import Optional

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required. Install with: pip install httpx")
    sys.exit(1)


def send_chat_request(
    url: str,
    messages: list[dict],
    model: str,
    session_id: Optional[str] = None,
    stream: bool = False,
) -> tuple[dict, dict, float]:
    """Send a chat completion request and measure elapsed time.

    Returns:
        Tuple of (response_json, response_headers, elapsed_seconds)
    """
    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["X-Session-Id"] = session_id

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 50,
        "stream": stream,
    }

    endpoint = f"{url}/v1/chat/completions"
    start = time.monotonic()

    with httpx.Client(timeout=120.0) as client:
        response = client.post(endpoint, json=payload, headers=headers)

    elapsed = time.monotonic() - start
    response_headers = dict(response.headers)

    try:
        response_json = response.json()
    except Exception:
        response_json = {"error": response.text}

    return response_json, response_headers, elapsed


def count_message_tokens(messages: list[dict]) -> int:
    """Heuristic: approximate token count (1 token ~ 4 chars)."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        total += max(1, len(content) // 4)
    return total


def run_benchmark(
    url: str, model: str, turns: int = 3, runs: int = 2
) -> dict:
    """Run the benchmark comparing session vs no-session ingestion.

    Returns a dict with:
      - no_session_times: list of per-turn times for no-session
      - session_times: list of per-turn times for session
      - no_session_total: total time without session
      - session_total: total time with session
      - reduction_pct: latency reduction percentage
      - details: per-turn breakdown
    """
    results = {
        "no_session_times": [],
        "session_times": [],
        "no_session_totals": [],
        "session_totals": [],
        "runs": runs,
        "turns": turns,
        "details": [],
        "metadata": {"url": url, "model": model},
    }

    for run in range(runs):
        run_detail = {"run": run + 1, "no_session": [], "session": []}

        # --- NO SESSION (full history each time) ---
        conversation: list[dict] = [
            {"role": "system", "content": "You are a helpful assistant. Be brief and concise."}
        ]
        no_session_times = []

        for turn in range(1, turns + 1):
            conversation.append({
                "role": "user",
                "content": f"Run {run+1}, turn {turn}: Briefly describe topic {turn}."
            })
            tokens = count_message_tokens(conversation)

            response, headers, elapsed = send_chat_request(
                url, conversation, model, session_id=None
            )

            try:
                assistant_content = response["choices"][0]["message"]["content"]
                conversation.append({"role": "assistant", "content": assistant_content})
            except (KeyError, IndexError):
                conversation.append({"role": "assistant", "content": "Error"})

            no_session_times.append(elapsed)
            run_detail["no_session"].append({
                "turn": turn,
                "elapsed_s": round(elapsed, 3),
                "messages_sent": len(conversation) - 1,  # minus system
                "approx_tokens_sent": tokens,
                "delta": False,
            })

        # --- WITH SESSION ---
        conversation2: list[dict] = [
            {"role": "system", "content": "You are a helpful assistant. Be brief and concise."}
        ]
        session_id = None
        session_times = []

        for turn in range(1, turns + 1):
            conversation2.append({
                "role": "user",
                "content": f"Run {run+1}, turn {turn}: Briefly describe topic {turn}."
            })
            tokens = count_message_tokens(conversation2)

            response, headers, elapsed = send_chat_request(
                url, conversation2, model, session_id=session_id
            )

            if not session_id:
                session_id = headers.get("x-session-id")

            try:
                assistant_content = response["choices"][0]["message"]["content"]
                conversation2.append({"role": "assistant", "content": assistant_content})
            except (KeyError, IndexError):
                conversation2.append({"role": "assistant", "content": "Error"})

            is_delta = headers.get("x-session-delta") == "true"
            session_times.append(elapsed)
            run_detail["session"].append({
                "turn": turn,
                "elapsed_s": round(elapsed, 3),
                "messages_sent": len(conversation2) - 1,
                "approx_tokens_sent": tokens,
                "delta": is_delta,
                "session_id": (session_id or "")[:8] + "...",
            })

        results["no_session_times"].extend(no_session_times)
        results["session_times"].extend(session_times)
        results["no_session_totals"].append(sum(no_session_times))
        results["session_totals"].append(sum(session_times))
        results["details"].append(run_detail)

    # Calculate aggregate
    avg_no = sum(results["no_session_totals"]) / len(results["no_session_totals"])
    avg_with = sum(results["session_totals"]) / len(results["session_totals"])
    results["avg_no_session_total_s"] = round(avg_no, 3)
    results["avg_with_session_total_s"] = round(avg_with, 3)

    if avg_no > 0:
        results["reduction_pct"] = round(((avg_no - avg_with) / avg_no) * 100, 1)
    else:
        results["reduction_pct"] = 0.0

    # Per-turn average
    avg_no_turn = sum(results["no_session_times"]) / len(results["no_session_times"])
    avg_session_turn = sum(results["session_times"]) / len(results["session_times"])
    results["avg_no_session_per_turn_s"] = round(avg_no_turn, 3)
    results["avg_with_session_per_turn_s"] = round(avg_session_turn, 3)

    if avg_no_turn > 0:
        results["per_turn_reduction_pct"] = round(
            ((avg_no_turn - avg_session_turn) / avg_no_turn) * 100, 1
        )
    else:
        results["per_turn_reduction_pct"] = 0.0

    return results


def print_results(results: dict) -> None:
    """Print benchmark results in a human-readable format."""
    print(f"\n{'='*60}")
    print(f"Session Ingestion Benchmark Results")
    print(f"{'='*60}")
    print(f"URL: {results['metadata']['url']}")
    print(f"Model: {results['metadata']['model']}")
    print(f"Turns: {results['turns']}, Runs: {results['runs']}")
    print()

    for detail in results["details"]:
        print(f"  Run {detail['run']}:")
        print(f"    No session:")
        for t in detail["no_session"]:
            print(f"      Turn {t['turn']}: {t['elapsed_s']}s "
                  f"({t['messages_sent']} msgs, ~{t['approx_tokens_sent']} tokens)")
        print(f"    With session:")
        for t in detail["session"]:
            print(f"      Turn {t['turn']}: {t['elapsed_s']}s "
                  f"(delta={t['delta']}, {t['messages_sent']} msgs, "
                  f"~{t['approx_tokens_sent']} tokens)")

    print(f"\n  Aggregate:")
    print(f"    Avg total (no session):  {results['avg_no_session_total_s']}s")
    print(f"    Avg total (with session): {results['avg_with_session_total_s']}s")
    print(f"    Total latency reduction:  {results['reduction_pct']}%")
    print()
    print(f"    Avg per turn (no session):  {results['avg_no_session_per_turn_s']}s")
    print(f"    Avg per turn (with session): {results['avg_with_session_per_turn_s']}s")
    print(f"    Per-turn latency reduction: {results['per_turn_reduction_pct']}%")
    print(f"{'='*60}")

    # Pass/fail check against >= 50% target
    if results["per_turn_reduction_pct"] >= 50:
        print("✓ BENCHMARK PASSED: >= 50% latency reduction achieved")
    else:
        print(f"✗ BENCHMARK NOT MET: {results['per_turn_reduction_pct']}% reduction "
              f"(target: >= 50%)")
        print("  Note: Results vary depending on model size, prompt length, and hardware.")
        print("  The 50% target is most achievable for multi-turn conversations with longer prompts.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark session-based incremental ingestion"
    )
    parser.add_argument(
        "--url", default="http://localhost:8000",
        help="Base URL of the proxy (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--model", default="gemma4",
        help="Model name (default: gemma4)"
    )
    parser.add_argument(
        "--turns", type=int, default=3,
        help="Number of conversation turns (default: 3)"
    )
    parser.add_argument(
        "--runs", type=int, default=2,
        help="Number of benchmark runs (default: 2)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    # Check proxy availability
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{args.url}/health")
            if resp.status_code != 200:
                print(f"WARNING: Proxy health check returned {resp.status_code}")
    except Exception as e:
        print(f"ERROR: Cannot connect to proxy at {args.url}: {e}")
        print("Make sure the proxy is running before running benchmarks.")
        sys.exit(1)

    results = run_benchmark(args.url, args.model, args.turns, args.runs)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results)


if __name__ == "__main__":
    main()