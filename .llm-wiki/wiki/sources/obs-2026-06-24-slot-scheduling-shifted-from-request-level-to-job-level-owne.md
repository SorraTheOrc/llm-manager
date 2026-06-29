---
type: source
title: "Observation: Slot scheduling shifted from request-level to job-level ownership"
slug: obs-2026-06-24-slot-scheduling-shifted-from-request-level-to-job-level-owne
status: observation
created: 2026-06-24
updated: 2026-06-24
relevance: high
observed_at: 2026-06-24T13:44:31.035Z
tags: ["slot-scheduling", "job-ownership", "design-evolution"]
source_context: "Slot management design discussion - shifting from request-level reservation to job-level ownership model"
---
# ⭐ Observation: Slot scheduling shifted from request-level to job-level ownership
User rejected the hybrid reservation + fair queue approach (Approach D in slot-management-design.md) because it focused on fairness per-turn rather than job completion. New direction: treat each multi-step conversation as an atomic Job that exclusively owns its Slot until completion. Core rules: Job admission binds tenant to slot; sticky execution guarantees all steps go to same slot; completion only on explicit end or timeout. Queue holds jobs (not turns), max_depth=4, overflow rejects with 15min Retry-After.
*Relevance: high*

*Context: Slot management design discussion - shifting from request-level reservation to job-level ownership model*

*Tags: slot-scheduling job-ownership design-evolution*
---
*Observed: 2026-06-24T13:44:31.035Z*