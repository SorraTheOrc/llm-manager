---
type: source
title: "Anthropic and OpenAI provider entries removed from config"
slug: anthropic-openai-provider-removal
status: insight
created: 2026-06-23
updated: 2026-06-23
category: devops
---
# Anthropic and OpenAI provider entries removed from config
Completed LP-0MQNQUR7K005GDP6: Removed `anthropic` (claude-* aliases) and `openai` (gpt-*, o1-*, chatgpt-* aliases) model entries from `proxy/config.yaml`. Also commented out `default_remote` section which referenced OpenAI. All documentation examples in `proxy/README.md` and `proxy/MODEL_ADD.md` updated to use generic provider names. The `proxy_openai_api` HTTP routes in `proxy/server.py` remain untouched per constraints. Tests: 533 passed, 0 failures. The test-update child (LP-0MQNRDPJL005NO0I) was completed in a prior session; this session handled config removal and doc updates.
*Category: devops*
---
*Captured: 2026-06-23*
## Related
_Add links to related pages._