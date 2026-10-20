---
id: jetson_gpu/index
hardware: jetson_gpu (Jetson AGX Orin, CUDA sm_87)
tags: [index, navigation]
layers: [L1]
evidence_basis: [derived_index]
version_sensitivity: low -- regenerate when a routed page or digest field changes
---

# Jetson GPU: index

Read pages by question, not as a mandatory optimization sequence.

| question | page | required output |
|---|---|---|
| What does this board expose, and what is only a product maximum? | `wiki/hardware/jetson_gpu/arch_facts.md` | separate product, live runtime, and target-contract facts |
| Which TVM dialect and CUDA constructs are legal? | `opt_patterns.md`, then `pitfalls.md` | syntax-compatible implementation plan |
| Which implementation mechanism fits an existing generated PrimFunc? | `reference_families.md` | L1-local family, preconditions, compatibility, falsifier |
| What does the current profile actually support? | `patterns.md` | evidence-backed hypothesis and expected counter/timing change |
| The current mechanism has stalled | `escape.md` | a different frame/mechanism, not another parameter sibling |

Only L0 may redesign the whole graph, algorithm, or fixed-state plan. L1 may
reschedule existing PrimFuncs and fuse one declared closed 2–3-call region
with its TIR under `leka.l1_fuse_of`; no larger or accumulated graph edit is
allowed. After an L0 base change, re-read the canonical module and timing
breakdown. Use the hardware reference for backend limits and the syntax
pages for compiler constraints.
