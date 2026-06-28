---
type: source
title: "Observation: Intake completed: unload_lru eviction observability"
slug: obs-2026-06-24-intake-completed-unload-lru-eviction-observability
status: observation
created: 2026-06-24
updated: 2026-06-24
relevance: medium
observed_at: 2026-06-24T08:24:53.082Z
tags: ["intake", "observability", "router", "eviction"]
source_context: "Intake brief for LP-0MQMC3S3K0013E42"
---
# 🔍 Observation: Intake completed: unload_lru eviction observability
Completed intake for LP-0MQMC3S3K0013E42 (Router preload / models_max & eviction config review). Key findings: (1) LLAMA_MODELS_MAX env var already works end-to-end in start-llama.sh — the AC was removed as already implemented. (2) Detection of unload_lru must be via log parsing since it's an internal llama-server event. (3) Threshold is configurable via env var defaulting to 3 events in 5 minutes. Effort: Small (4.33h expected). Risk: Low.
*Relevance: medium*

*Context: Intake brief for LP-0MQMC3S3K0013E42*

*Tags: intake observability router eviction*
---
*Observed: 2026-06-24T08:24:53.082Z*