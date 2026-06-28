---
type: source
title: "Observation: pool_size>1 alone insufficient to prevent slot thrashing"
slug: obs-2026-06-24-pool-size-1-alone-insufficient-to-prevent-slot-thrashing
status: observation
created: 2026-06-24
updated: 2026-06-24
relevance: medium
observed_at: 2026-06-24T01:31:00.246Z
source_context: "LP-0MQR75QYM001HAUB Slot thrashing reproduction & evidence"
---
# 🔍 Observation: pool_size>1 alone insufficient to prevent slot thrashing
The F1 investigation tested pool_size values from 1 to 8 with 4 concurrent sessions. pool_size=4 (matching session count) dropped steal rate from 98% to 40% — a meaningful improvement but insufficient. Even with pool_size=8 (more slots than sessions), steal rate was 43% due to hash collisions (birthday problem). Conclusion: increasing pool_size is a partial fix; the hybrid reservation+queue approach is required for near-zero thrashing.
*Relevance: medium*

*Context: LP-0MQR75QYM001HAUB Slot thrashing reproduction & evidence*
---
*Observed: 2026-06-24T01:31:00.246Z*