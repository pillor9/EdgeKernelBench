---
id: rk3588_gpu/index
hardware: rk3588_gpu (RK3588S, Mali-G610, OpenCL)
tags: [index, navigation]
layers: [L1]
evidence_basis: [derived_index]
version_sensitivity: low -- regenerate when a routed page or profiler capability changes
---

# RK3588 GPU: index

Read pages by question, not as a fixed tuning ladder.

| question | page | required output |
|---|---|---|
| Product claim, live driver fact, or backend limit? | `wiki/hardware/rk3588_gpu/arch_facts.md` | product, runtime and target-contract facts |
| Exact TVM 0.23/OpenCL syntax? | `opt_patterns.md`, then `pitfalls.md` | legal implementation plan with no CUDA/Adreno transfer |
| Which implementation mechanism fits an existing generated PrimFunc? | `reference_families.md` | L1-local family, compatibility, resource risk, falsifier |
| Event timing, counters, or static compiler evidence? | `patterns.md` | correctly typed evidence; no invented NCU signal |
| Same mechanism is no longer learning | `escape.md` | forced frame/mechanism change |

Only L0 may redesign the whole graph, algorithm family, or fixed-state layout.
L1 applies these cards to existing PrimFuncs or one declared closed 2–3-call
local fusion region under `leka.l1_fuse_of`. Preserve external tensor interfaces
and all unrelated graph/ABI structure; larger or accumulated changes go to L0.

OpenCL event time is the per-kernel timing authority. Streamline/gatord or
libGPUCounters can add dynamic Mali counters when deployed, while Mali Offline
Compiler adds static code estimates; these evidence types must remain
separate [arm-performance-studio] [arm-libgpucounters] [arm-malioc].
