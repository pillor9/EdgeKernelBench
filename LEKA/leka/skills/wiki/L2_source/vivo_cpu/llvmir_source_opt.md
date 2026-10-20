---
id: L2_source/vivo_cpu/llvmir_source_opt
hardware: vivo_cpu (heterogeneous AArch64 Android CPU, generic target)
layer: L2 (generated LLVM IR function bodies)
evidence_basis: [compiler_doc, backend_contract]
version_sensitivity: high -- LLVM IR and llc must match the configured toolchain
---

# LLVM IR optimization for generic AArch64

This page authorizes only body-level edits to functions already present in the
read-only generated LLVM module. Global graph, algorithm, fusion and fixed-state
choices belong to L0. Bounded local fusion, loop organization and parallel
structure belong to L1. Candidate-side `FoldConstant`, `FuseOps`, `FuseTIR`, DLight
and MetaSchedule remain outside this layer's authority.

## Toolchain constraints

Use the IR dialect accepted by the configured LLVM build. The driver compiles
with the matching `llc` for `aarch64-linux-android`; a different NDK compiler
version may not accept the emitted IR. Preserve all function signatures,
metadata references, target attributes and aliasing guarantees.
[llvm-language-reference]

## Candidate mechanisms

- Hoist redundant address arithmetic when dominance, aliasing and memory
  behavior make the transformation legal.
- Use independent accumulators to shorten a reduction dependency chain when
  the task's numerical contract permits the final combination. Check whether
  larger live state introduces spills.
- Loop metadata may request vectorization, interleaving or unrolling. It is a
  compiler hint rather than proof of the resulting machine instructions.
  Inspect the actual generated body and preserve existing loop semantics.
  [llvm-language-reference]
- Retain accurate alignment on loads and stores. A base allocation's alignment
  does not establish the same alignment for an arbitrary offset.
- Add floating-point flags only when their full semantic implications satisfy
  the task contract. For example, reassociation permission does not establish
  the absence of NaNs, infinities or signed zero. [llvm-language-reference]

The generic AArch64 target remains authoritative. Optional silicon features
cannot be enabled by changing source attributes. The `vivo_cpu_8threads_affinity_mode1_v1`
policy verifies an eight-thread TVM pool, without promising equal core capacity
or stable affinity. Mode 1 names `kBig` in local TVM source; this is not proof
of target ABI mapping, big-only execution, or actual core utilization. Thread
policy belongs to the evaluator and is fixed for a cohort; a candidate cannot
change it. Keep the current source when there is no supported improvement.
[leka-vivo-contract]

[llvm-language-reference]: https://llvm.org/docs/LangRef.html
[leka-vivo-contract]: ../../sources/vivo_cpu/backend_contract.json
