---
id: vivo_cpu/arch_facts
hardware: vivo_cpu (Snapdragon heterogeneous AArch64, NEON)
tags: [facts, memory, execution-width]
layers: []
evidence_basis: [vendor_doc, backend_contract]
version_sensitivity: mixed -- verify runtime state and compiler compatibility for each deployment
---

# Vivo CPU: architecture facts

Use product documentation for the processor family, current runtime queries
for the attached handset, and the configured LLVM target for code-generation
legality. Optional physical ISA features do not authorize a target change.
[qualcomm-sm8650] [leka-vivo-contract]

## Product and architecture

- Snapdragon 8 Gen 3 has one Prime, five Performance and two Efficiency CPU
  cores. These groups do not establish execution affinity or live frequency.
  [qualcomm-sm8650]
- Arm Neon/Advanced SIMD uses 128-bit vector registers. Four f32 lanes describe
  an instruction width, not an application speedup. [arm-neon]
- Cache capacity, line size and sharing must be queried when available. A
  processor-family label alone is insufficient to establish cache residency.

## LEKA target and runtime contract

- The compiler target is `llvm -keys=arm_cpu,cpu -mcpu=generic -mtriple=aarch64-linux-android -num-cores=8`.
  The generic target is shared across the heterogeneous cores.
  [leka-vivo-contract]
- The runtime policy is `vivo_cpu_8threads_affinity_mode1_v1`: immediately before execution the
  RPC worker calls `runtime.config_threadpool(1, 8)` and rejects the run unless
  `runtime.NumThreads()` reports exactly 8. This establishes pool size, not
  stable affinity, equal core capacity or fixed frequency.
  [leka-vivo-contract]
- Mode `1` names `kBig` in the inspected local TVM source. The explicit request
  for eight threads does not establish big-only execution or utilization of
  eight specific CPUs. The worker does not verify the target binary's ABI or
  per-thread placement. This mode change does not establish a cause or cure
  for the observed dependence of latency on call spacing.
- Historical `vivo_cpu_8allcores` jobs used unnamed mode `0`. Saved jobs missing
  `runtime_affinity_mode` retain mode `0`; an explicit integer `0` is also
  accepted for legacy reproduction. Actual requests receive a mode-specific
  label and preserve the supplied configuration label separately. New and old
  cohorts must be distinguished. Candidates cannot change evaluator policy.
- Candidates must pass the configured TVM parser, validator, LLVM build and
  device checks. Dot-product, i8mm, BF16, SVE or core-specific support observed
  on the hardware does not widen the generic compiler contract.
  [leka-vivo-contract]
- Treat allowed CPUs, online cores, worker placement, frequency and thermal
  readings as runtime state. A snapshot is context for a measurement, not a
  guarantee that placement and clocks were constant throughout execution.

[qualcomm-sm8650]: https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-gen-3-mobile-platform
[arm-neon]: https://developer.arm.com/Architectures/Neon
[leka-vivo-contract]: ../../sources/vivo_cpu/backend_contract.json
