---
type: source
title: "Observation: Provider removal from config.yaml and docs complete"
slug: obs-2026-06-23-provider-removal-from-config-yaml-and-docs-complete
status: observation
created: 2026-06-23
updated: 2026-06-23
relevance: high
observed_at: 2026-06-23T15:10:20.087Z
tags: ["config", "proxy", "providers", "removal"]
source_context: "Implementing LP-0MQNQUR7K005GDP6: Remove anthropic and openai providers from proxy/config.yaml"
---
# ⭐ Observation: Provider removal from config.yaml and docs complete
Removed anthropic and openai model entries from proxy/config.yaml (LP-0MQNQUR7K005GDP6). The anthropic entry had claude-* aliases; the openai entry had gpt-*, o1-*, chatgpt-* aliases. Also commented out default_remote section. Updated all documentation examples in proxy/README.md and proxy/MODEL_ADD.md to use generic provider-a/provider-b names. All 533 tests pass.
*Relevance: high*

*Context: Implementing LP-0MQNQUR7K005GDP6: Remove anthropic and openai providers from proxy/config.yaml*

*Tags: config proxy providers removal*
---
*Observed: 2026-06-23T15:10:20.087Z*