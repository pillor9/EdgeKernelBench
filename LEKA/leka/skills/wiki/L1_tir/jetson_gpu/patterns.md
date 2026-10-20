---
id: jetson_gpu/patterns
hardware: jetson_gpu (Jetson AGX Orin, CUDA sm_87)
tags: [symptom-map, digest, bottleneck, profiler]
layers: [L1]
evidence_basis: [backend_contract, vendor_doc, derived_index]
version_sensitivity: medium -- labels and thresholds must match the shipped perf digest
---

# Jetson GPU: profile evidence routing

The digest is a bounded causal hint. Per-kernel time identifies where to work;
NCU counters suggest why. Neither overrides paired timing or correctness. Use only metrics present in the result;
missing DRAM data does not imply zero bandwidth.

The mechanism column routes work by owner: global graph/algorithm/fixed-state
changes go to L0. L1 may reschedule existing bodies or jointly fuse one closed
2–3-call region under `leka.l1_fuse_of`; larger or escaping regions go to L0.

| `bottleneck_hypothesis` | accepted evidence | candidate mechanism | falsifier |
|---|---|---|---|
| `launch_overhead` | many short launches in the timing summary | L1: declared closed 2–3-call fusion + TIR; L0: broader fusion/batching/pass removal | launches fall but whole-graph time does not |
| `memory_bandwidth` | high L2/DRAM throughput with lower SM throughput | L0: static pack/fusion/remove traffic; L1: contiguous ownership and reuse tiling inside one PrimFunc | traffic signal falls but latency does not |
| `compute` | high SM throughput | L0: different algorithm/dataflow; L1: reduce instructions or expose legal instruction shapes inside one PrimFunc | instruction/work estimate falls without time change |
| `latency_or_occupancy` | low occupancy or long-scoreboard stalls | reduce register/shared footprint, increase independent blocks, shorten latency chains | occupancy/stalls improve without time improvement |
| `instruction_dependency` | both compute and memory pipes low, with no resource-pressure explanation | independent accumulators, reassociation if legal, expose more independent outputs | pipes remain idle or numerical behavior changes |
| `unknown` | absent/low-confidence or contradictory signals | compare structurally different families with cheap measurement; request a focused counter set | any claimed cause lacks a corresponding observable change |

State a hypothesis and a falsifying observation, make a coherent edit, and
validate it with correctness checks and paired timing.

Timing attribution can be compiled-kernel rather than exact static PrimFunc.
Use the validated kernel join keys when present and do not infer one-to-one graph
ownership from a compiler symbol alone.
