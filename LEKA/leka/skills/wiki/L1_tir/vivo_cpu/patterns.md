---
id: vivo_cpu/patterns
hardware: vivo_cpu (V2403A, SM8650, AArch64 Android CPU)
tags: [symptom-map, digest, bottleneck, profiler]
layers: [L1]
evidence_basis: [backend_contract, derived_index]
version_sensitivity: medium -- counter availability, affinity proof, and thresholds must match the shipped digest
---

# Vivo CPU: profile evidence routing

This backend uses timing plus a portable whole-process simpleperf set:
`cpu-cycles`, `instructions`, `cache-references`, `cache-misses`, and
`branch-misses`. GPU occupancy, NCU, Mali counters, and ROCm traces do not
describe its CPU PrimFuncs. The current digest does not expose backend-stall,
L1D/LLC-specific, or migration counters; do not reason as if it did.

The mechanism column routes work by owner. Global graph/algorithm/fixed-state
items go to L0. L1 may reschedule existing bodies or jointly fuse one closed
2–3-call region under `leka.l1_fuse_of`; larger or escaping regions go to L0.
LLVM source-body details go to L2.

| `bottleneck_hypothesis` | accepted evidence | candidate mechanism | falsifier |
|---|---|---|---|
| `memory_bandwidth` | low IPC plus a high aggregate `cache_misses/cache_references` ratio; cache level remains unknown | L0: fuse/remove passes or static pack; L1: contiguous traversal/cache panels inside one PrimFunc | aggregate miss signal falls without paired latency improvement |
| `instruction_dependency` | low IPC with a low aggregate miss ratio; this is a lead, not a measured stall diagnosis | independent accumulators, shorten dependency chains, tolerance-valid reassociation | IPC changes without time improvement or spilling rises |
| `schedule_granularity` | verified eight-thread policy plus no explicit parallel loop, or poor scaling under otherwise comparable runtime snapshots | partition independent outer work inside the existing PrimFunc | added parallelism fails to improve paired time or raises instability |
| `graph_redundancy` | multiple exact `R.call_tir`/profile rows or materialized intermediates | L1: declared closed 2–3-call fusion + TIR; L0: broader fusion/CSE/tensor-pass removal | call/traffic reduction is absent or whole-graph time does not improve |
| `static_weight_layout` | fixed rank-2-or-higher state plus a transpose/layout use recorded by the trusted task projection | L0: declared one-time relayout/packing through the parameter-plan ABI | preparation is timed/undeclared or entry time does not improve |
| `compute` | relatively high IPC under comparable observed runtime context | L0: reduce high-level work; L1: hoist invariants/vectorize/unroll inside one PrimFunc | instruction count falls but time does not |
| `unknown` | incomplete or low-confidence PMU/runtime evidence | ask L0 about graph families or compare L1-local loop families; state the missing observation | no causal label without an observable effect |

TVM profile rows may report per-operation call/time data. Whole-process
cycles/instructions/cache/branch counters must not be allocated to PrimFuncs by
elapsed-time share. Report graph-level PMU evidence and per-operation timing as
separate artifacts. Runtime CPU-allowed/frequency/thermal snapshots are
context, not proof of stable affinity or equal work distribution.

The frozen generic AArch64 target remains authoritative even if PMU or device
identity reveals newer silicon features [leka-vivo-contract].
