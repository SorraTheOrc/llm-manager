# Qwen Thinking Preservation

Work item: LP-0MT5YLL36000ZYRT — "ensure Qwen is preserving thinking"

## Summary

This document records how the llm-manager stack ensures Qwen models preserve
their thinking (reasoning) output end-to-end, the required pi-agent model
configuration, and the verification steps to confirm thinking content appears
in session transcripts.

## 1. Model configuration (pi agent `models.json`)

Qwen models are consumed through the pi coding agent. The runtime model
configuration lives at `~/.pi/agent/models.json` (merged over the
auto-refreshed store `~/.pi/agent/models-store.json`), which is outside this
repo. To keep the configuration versioned and auditable, the canonical
overrides are tracked here:

- `config/pi-agent/qwen-thinking-overrides.json`

It defines `modelOverrides` that must be present under
`providers.opencode` and `providers.opencode-go` of the live `models.json`:

| Provider | Model(s) | Thinking mechanism |
|----------|----------|--------------------|
| `opencode` (anthropic-messages API) | `qwen3.5-plus`, `qwen3.6-plus` | `thinkingLevelMap` (maps `off`/`high`/`max` for Anthropic-side adaptive thinking) |
| `opencode-go` (openai-completions API) | `qwen3.6-plus`, `qwen3.7-max`, `qwen3.7-plus`, `qwen3.8-max` | `compat.thinkingFormat: "qwen"` (`enable_thinking: true`) + `thinkingLevelMap` |

Notes:

- `thinkingFormat: "qwen"` sends a top-level `enable_thinking` flag
  (DashScope-style), which the opencode-go `zen/go` endpoint accepts. The
  alternative `qwen-chat-template` (`chat_template_kwargs` +
  `preserve_thinking: true`) targets local vLLM/HF-chat-template servers and is
  **not** appropriate for the remote opencode-go endpoint.
- The work item lists `qwen3.8-plus`, but the pi catalog / models-store expose
  **`qwen3.8-max`**; the configured variants cover every Qwen model the stack
  actually exposes.
- `thinkingLevelMap.off` is `null` so a "no thinking" request disables
  thinking rather than defaulting it on.

### Applying / verifying the live config

The tests in `tests/test_qwen_thinking_config.py` validate:

- the canonical overrides file covers every Qwen variant with the correct
  thinking config (AC1, AC3), and
- the live `~/.pi/agent/models.json` matches the canonical overrides when
  present (drift detection; skipped when the file is absent, e.g. CI).

To (re)apply the canonical overrides to the live file, merge the
`providers.*.modelOverrides` keys into `~/.pi/agent/models.json` by hand or
with a small JSON merge — the test suite will confirm parity afterwards.

## 2. Proxy configuration review (AC2)

Reviewed the llm-manager proxy (`proxy/proxy/`) for any logic that strips or
suppresses `reasoning_content` / thinking blocks:

- **No stripping logic exists.** Searches across `server.py`, `session.py`,
  `utils.py`, `session_recorder.py`, `provider.py`, and `proxy_remote.py`
  found no code that deletes `reasoning_content` from streamed or non-streamed
  responses.
- **Thinking is preserved into session history.** `session.py`'s
  `extract_streamed_assistant_message_from_sse` reconstructs assistant
  messages with both `content` and `reasoning_content`, so the persisted
  session transcript keeps the full thinking text.
- **Thinking-only turns** (no final content, no tool call) emit the literal
  placeholder `"Thinking..."` in `content` while the full thinking text stays
  in `reasoning_content` (LP-0MSEHOE7B005DE08) — thinking is never dropped,
  it simply is not replayed verbatim into the content field to avoid context
  bloat.
- **Remote round-trip repair** (LP-0MSGU3JNU0092AFQ): assistant messages
  missing `reasoning_content` get `""` injected before replay to
  thinking-mode providers that reject such messages.

## 3. Testing (AC4)

`proxy/tests/test_qwen_thinking_preserved.py` verifies, with mocked/assembled
Qwen-style SSE streams:

1. A Qwen stream's `reasoning_content` lands in the reconstructed assistant
   message (session transcript assembly) and survives the session-recorder
   round-trip to disk unchanged (`TestQwenStyleThinkingPreservation`,
   `TestSessionRecorderRoundTrip`).
2. Nothing in the proxy deletes `reasoning_content` — the round-trip tests
   would fail if the proxy stripped it at any stage.

### Operator verification (live, end-to-end)

AC4 also calls for confirming thinking content appears in a real pi session
transcript. To verify against a live Qwen model:

1. Ensure `~/.pi/agent/models.json` carries the canonical overrides
   (automatic if `tests/test_qwen_thinking_config.py` passes).
2. Start a pi session with a Qwen model and a thinking level, e.g.
   `/model qwen3.7-max` then set thinking level `high` in the prompt settings.
3. Ask a question that benefits from reasoning; the session transcript should
   include the thinking (reasoning) content.
4. Optionally inspect proxy session recordings
   (default `~/.llm-proxy/session-recordings/`, LP-0MT2TC7FG008BXIU) — the
   recorded `provider_to_client` response must carry `reasoning_content`.

## Acceptance criteria mapping

| AC | Status | Evidence |
|----|--------|----------|
| 1. Qwen entries have `thinkingFormat`/thinking config | Met | canonical overrides + live parity tests |
| 2. Proxy does not strip thinking | Met | code review + round-trip regression tests (`TestSessionRecorderRoundTrip`, `TestQwenStyleThinkingPreservation`) |
| 3. All Qwen variants reviewed consistently | Met | canonical overrides cover all exposed variants (`qwen3.8-max` replaces the non-existent `qwen3.8-plus`) |
| 4. Testing confirms thinking appears in transcript | Met (proxy-level) | `TestQwenStyleThinkingPreservation` etc.; live operator verification steps above |