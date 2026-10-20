---
id: rk3588_gpu/patterns
hardware: rk3588_gpu (RK3588, Mali-G610, OpenCL)
tags: [symptom-map, digest, bottleneck, profiler]
layers: [L1]
evidence_basis: [backend_contract, vendor_doc, derived_index]
version_sensitivity: high -- driver, counter availability and digest fields may change
---

# RK3588 GPU: profile evidence routing

## Evidence types

| Evidence | Source | Interpretation |
|---|---|---|
| Operation timing | TVM VM/OpenCL profiling | Runtime operation name, duration and call count; static PrimFunc identity requires a validated symbol map. |
| Dynamic counters | Streamline with a compatible `gatord`, or libGPUCounters | Shader, cache and memory activity over the recorded capture scope. [arm-g610-counters] [arm-libgpucounters] |
| Static estimates | Mali Offline Compiler | Compile-time resource and pipeline estimates; separate from measured latency and runtime counters. [arm-malioc] |
| Environment context | Optional `devfreq` or `hwcpipe` collector | Device-wide window with `exact_bundle=false`; not per-candidate evidence. [leka-rk3588-contract] |

Check collector status, capture scope and candidate binding before using any
result. Never divide device-wide counters among kernels by timing share.
Tracepoint availability alone does not establish an instruction trace or
complete kernel coverage.

## Digest routing

The mechanism column routes work by owner. Global graph/algorithm/fixed-state
items are L0 requests. L1 may reschedule existing bodies or jointly fuse one
closed 2–3-call region under `leka.l1_fuse_of`; larger or escaping regions go to
L0. Source-body-only refinements belong to L2.

| `bottleneck_hypothesis` | accepted evidence | candidate mechanism | falsifier |
|---|---|---|---|
| `launch_overhead` | many short OpenCL events and repeated launches | L1: declared closed 2–3-call fusion + TIR; L0: broader fusion/group work/change regions | launch count falls but whole-graph time does not |
| `memory_bandwidth` | static load/store or texture pipeline dominates; later dynamic memory counters can strengthen it | L0: remove a graph pass/fuse; L1: contiguous ownership and reuse-backed local staging inside one PrimFunc | static/dynamic memory signal falls without time improvement |
| `compute` | arithmetic dominates an available static pipeline estimate; later dynamic shader counters can strengthen it | L0: different algorithmic frame; L1: reduce instructions/expose independent work inside one PrimFunc | estimated work falls but event time does not |
| `latency_or_occupancy` | static spill, excessive register use, or a non-arithmetic/non-memory dominant static estimate | shrink private state/workgroup, add independent groups, restructure reduction | spill/resource estimate improves without event-time improvement |
| `unknown` | only timing, contradictory evidence, or low confidence | compare distinct mechanism families; deploy a focused counter/static-code probe | do not turn absence of a counter into a causal claim |

Static and dynamic evidence are complementary: if Mali Offline Compiler is
deployed and validated, it can estimate generated-code pressure before a
device run, while event timing and Mali counters decide whether that pressure
matters on this board.

Devfreq clock and utilization readings describe a device-wide window. The
current optional collectors publish `exact_bundle=false`; use these readings
as environment context, not as causal evidence for a particular PrimFunc.
[leka-rk3588-contract]
