"""Fixtures copied from real lines in /var/log/llama-proxy/proxy.log.

These lines preserve the exact structured INFO format the proxy emits
(``%Y-%m-%d %H:%M:%S,%f - LEVEL - message``). Request payloads are truncated
in the same way the log truncates them (the proxy cuts payloads off; the
analysis must tolerate this).
"""

from __future__ import annotations

# --- Stream started ---------------------------------------------------------

STREAM_STARTED_LOCAL = (
    "2026-08-02 13:58:32,260 - INFO - Stream started: provider=local model=Qwen3 "
    "session=019fc284-dcb8-74ca-9a64-9306b6f9d286 "
    "request=[{'type': 'text', 'text': '/plan SA-0MSAS108O009DYKT'}]"
)

STREAM_STARTED_REMOTE = (
    "2026-08-02 13:57:06,975 - INFO - Stream started: provider=opencode-go "
    "model=deepseek-v4-flash session=019fbfea-40b0-74a5-8cc9-bf3723f2b458 "
    "request=[{'type': 'text', 'text': 'The conversation history before this point was compac..."
)

STREAM_STARTED_DEEPSEEK = (
    "2026-08-02 14:52:41,214 - INFO - Stream started: provider=deepseek "
    "model=deepseek-v4-flash session=019fc2be-eb73-7830-a55e-c8bd5d21c927 "
    "request=[{{'type': 'text', 'text': 'Summarize the current state of the proxy...'}}]"
)

STREAM_STARTED_SESSION_UNKNOWN = (
    "2026-08-02 14:00:05,000 - INFO - Stream started: provider=opencode-go "
    "model=deepseek-v4-flash session=unknown request=[{'type': 'text', 'text': '...'}]"
)

# --- Stream finished --------------------------------------------------------

# tokens before session
STREAM_FINISHED_TOKENS_SESSION = (
    "2026-08-02 14:02:19,920 - INFO - Stream finished: reason=tool_calls "
    "tokens=43550/460/44010 session=019fc27d-3a46-7e5c-871e-57ab32f875f3 "
    "provider=opencode-go model=deepseek-v4-flash "
    "request=[{'type': 'text', 'text': '<skill name=\"implement\" location=\"/home/rgardler/.pi/..."
)

# session before tokens (different field order)
STREAM_FINISHED_SESSION_TOKENS = (
    "2026-08-02 14:03:08,814 - INFO - Stream finished: reason=tool_calls "
    "session=019fc27d-3a46-7e5c-871e-57ab32f875f3 "
    "tokens=52031/56/52087 provider=opencode-go model=deepseek-v4-flash "
    "request=[{'type': 'text', 'text': '<skill name=\"implement\" location=\"/home/rgardler/.pi/..."
)

# tokens, no session (remote-only responses are logged without a session UUID)
STREAM_FINISHED_TOKENS_NO_SESSION = (
    "2026-08-02 14:00:33,338 - INFO - Stream finished: reason=stop "
    "tokens=49364/6444/55808 provider=opencode-go model=deepseek-v4-flash "
    "request=[{'type': 'text', 'text': '<conversation>\\n[Assistant thinking]: The child OSL-0..."
)

# no tokens, with session (token counts are not always logged)
STREAM_FINISHED_NO_TOKENS = (
    "2026-08-02 13:59:04,893 - INFO - Stream finished: reason=tool_calls "
    "session=019fc27d-3a46-7e5c-871e-57ab32f875f3 provider=local model=Qwen3 "
    "request=[{'type': 'text', 'text': '<skill name=\"implement\" location=\"/home/rgardler/.pi/..."
)

STREAM_FINISHED_LOCAL = (
    "2026-08-02 13:59:07,344 - INFO - Stream finished: reason=tool_calls "
    "tokens=2430/120/2550 session=019fc284-dcb8-74ca-9a64-9306b6f9d286 "
    "provider=local model=Qwen3 request=[{'type': 'text', 'text': '/plan SA-0MSAS108O009DYKT'}]"
)

# --- Fallback triggered (never carries a session UUID) ----------------------

FALLBACK_CONCURRENCY = (
    "2026-08-02 13:58:04,301 - INFO - Fallback triggered for model=v1/chat/completions, "
    "from=local-qwen3, to=opencode-go-deepseek, reason=local_concurrency_limit"
)

FALLBACK_WARM_CACHE = (
    "2026-08-02 13:57:06,974 - INFO - Fallback triggered for model=v1/chat/completions, "
    "from=local-qwen3, to=opencode-go-deepseek, reason=warm_cache_bypass"
)

FALLBACK_HTTP_400 = (
    "2026-08-02 14:02:13,126 - INFO - Fallback triggered for model=v1/chat/completions, "
    "from=local-qwen3, to=opencode-go-deepseek, reason=HTTP 400"
)

# --- routing_skip_local (session + reason, arrow before the explanation) ----

ROUTING_SKIP_WARM = (
    "2026-08-02 14:02:20,157 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 "
    "estimated_tokens=40000 cold_threshold=39594 warm_threshold=39594 new_tokens=323 "
    "cached_ratio=0.99 reason=warm_cache_bypass → skipping local, routing to next remote "
    "provider session=019fc27d-3a46-7e5c-871e-57ab32f875f3"
)

ROUTING_SKIP_LARGE_CONTEXT = (
    "2026-08-02 14:30:00,000 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 "
    "estimated_tokens=150000 cold_threshold=39594 warm_threshold=39594 new_tokens=500 "
    "cached_ratio=0.00 reason=large_context_bypass → skipping local, routing to next remote "
    "provider session=aaaaaaaa-1111-2222-3333-444444444444"
)

# --- local_dispatch_denied --------------------------------------------------

DISPATCH_DENIED = (
    "2026-08-02 14:03:06,616 - INFO - local_dispatch_denied session=019fc245 "
    "owner=019fc27d active=4"
)

# --- Lines the parser must ignore -------------------------------------------

ROUTING_CHECK_IGNORED = (
    "2026-08-02 13:57:33,529 - INFO - routing_check provider=local-qwen3 model=Qwen3 "
    "estimated_tokens=12217 cold_threshold=39594 warm_threshold=39594 new_tokens=12217 "
    "cached_ratio=0.00 messages=2 session=019fc28d-051b-75fd-9f88-1c336f6779e0"
)

SESSION_HEADER_IGNORED = (
    "2026-08-02 13:57:33,529 - INFO - Session header resolved: source=session_id "
    "session=019fc28d..."
)

LEASE_RENEWED_IGNORED = "2026-08-02 13:59:04,901 - INFO - lease_renewed session=019fc27d timeout=60s"

REQUEST_ROUTING_IGNORED = "2026-08-02 13:57:06,974 - INFO - Request routing: model=plan → local-qwen3"

WARNING_LINE_IGNORED = (
    "2026-08-02 14:02:11,371 - WARNING - Response truncated: finish_reason=length "
    "session=019fc27d-3a46-7e5c-871e-57ab32f875f3 model=Qwen3"
)

MALFORMED_LINE = "garbage without a timestamp prefix"

# --- Config fixture (proxy/config.yaml fragment, real values) ---------------

CONFIG_FRAGMENT = """\
# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
default_model: code

server:
  # --- Routing thresholds -----------------------------------------------------
  llama_router_mode: true
  max_concurrent_queries: 16
  local_large_context_cold_cache_threshold: 60000
  local_large_context_warm_cache_threshold: 100000
  local_model_ctx_size: 262144
  local_max_concurrent_queries: 1
  local_dispatch_lease_timeout_seconds: 60
  session_single_flight_mode: queue
  session_slot_save_path: /home/rgardler/projects/llm/slot-cache
  # Server restart required to change this value (no hot reload).
  session_slot_pool_size: 6

  # -----------------------------------------------------------------------
  # Slot schedule (LP-0MRXZU90M007WNWT)
  # -----------------------------------------------------------------------
  slot_schedule:
    enabled: true
    drain_minutes: 3
    entries:
      - time: "23:59"
        slots: 8
      - time: "10:00"
        slots: 6

  session_slot_timeout_seconds: 3.0
  session_slot_max_prompt_tokens: 12288
  session_slot_timeout_per_token_seconds: 0.001
  session_slot_max_timeout_seconds: 60
"""

# ---------------------------------------------------------------------------
# Synthetic end-to-end fixture: two sessions across one rotated file boundary
# ---------------------------------------------------------------------------

# Session S1: local-only, starts 13:30 (before window start at 14:00), active in window.
S1 = "11111111-1111-1111-1111-111111111111"
# Session S2: local -> remote fallback, fully inside the window.
S2 = "22222222-2222-2222-2222-222222222222"

E2E_LINES = [
    # Rotated file content (13:29:00 - 14:00:00): S1 starts before the window.
    f"2026-08-02 13:30:00,000 - INFO - Stream started: provider=local model=Qwen3 session={S1} request=[{{'type': 'text', 'text': 'first'}}]",
    f"2026-08-02 13:30:05,000 - INFO - Stream finished: reason=stop tokens=1200/50/1250 session={S1} provider=local model=Qwen3 request=[{{'type': 'text', 'text': 'first'}}]",
    # Live file content (14:00:00+): S1 continues, S2 starts local then falls back.
    f"2026-08-02 14:00:10,000 - INFO - Stream started: provider=local model=Qwen3 session={S1} request=[{{'type': 'text', 'text': 'second'}}]",
    f"2026-08-02 14:00:12,000 - INFO - Stream finished: reason=stop tokens=1300/60/1360 session={S1} provider=local model=Qwen3 request=[{{'type': 'text', 'text': 'second'}}]",
    f"2026-08-02 14:01:00,000 - INFO - Stream started: provider=local model=Qwen3 session={S2} request=[{{'type': 'text', 'text': 'hello'}}]",
    f"2026-08-02 14:01:05,000 - INFO - Stream finished: reason=stop tokens=900/40/940 session={S2} provider=local model=Qwen3 request=[{{'type': 'text', 'text': 'hello'}}]",
    FALLBACK_CONCURRENCY.replace("13:58:04,301", "14:01:06,000"),
    f"2026-08-02 14:01:06,100 - INFO - routing_skip_local provider=local-qwen3 model=Qwen3 "
    f"estimated_tokens=5000 cold_threshold=39594 warm_threshold=39594 new_tokens=50 cached_ratio=0.50 "
    f"reason=local_concurrency_limit → skipping local, routing to next remote provider session={S2}",
    f"2026-08-02 14:01:06,200 - INFO - Stream started: provider=opencode-go model=deepseek-v4-flash session={S2} request=[{{'type': 'text', 'text': 'hello'}}]",
    f"2026-08-02 14:01:09,000 - INFO - Stream finished: reason=stop tokens=950/200/1150 session={S2} provider=opencode-go model=deepseek-v4-flash request=[{{'type': 'text', 'text': 'hello'}}]",
    f"2026-08-02 14:02:00,000 - INFO - local_dispatch_denied session=33333333-3333-3333-3333-333333333333 owner={S2} active=6",
]
