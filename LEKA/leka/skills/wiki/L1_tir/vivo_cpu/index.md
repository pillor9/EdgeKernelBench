---
id: vivo_cpu/index
hardware: vivo_cpu (V2403A, SM8650, AArch64 Android CPU)
tags: [index, navigation]
layers: [L1]
evidence_basis: [derived_index]
version_sensitivity: low -- regenerate when a routed page, target, or counter field changes
---

# Vivo CPU: index

Read pages by question rather than applying a fixed tuning ladder.

| question | page | required output |
|---|---|---|
| Which silicon facts are observed, and what does the generic target permit? | `wiki/hardware/vivo_cpu/arch_facts.md` | product/runtime/target separation |
| Which TVM 0.23 CPU constructs are legal? | `opt_patterns.md`, then `pitfalls.md` | syntax-valid, race-free loop plan |
| Which implementation mechanism fits an existing generated PrimFunc? | `reference_families.md` | L1-local family, ownership, compatibility, falsifier |
| What do PMU timing and counters support? | `patterns.md` | evidence-backed hypothesis |
| Repeated variants of one mechanism have stalled | `escape.md` | a different frame/mechanism |

Only L0 may redesign the whole graph, algorithm, or fixed-state plan. L1 may
reschedule existing PrimFuncs and fuse one declared closed 2–3-call region
with its TIR under `leka.l1_fuse_of`; unrelated graph/ABI structure stays frozen.
After an L0 base change, re-read and re-profile the new canonical module.
