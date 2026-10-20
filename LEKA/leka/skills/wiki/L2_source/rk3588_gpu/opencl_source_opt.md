---
id: L2_source/rk3588_gpu/opencl_source_opt
hardware: rk3588_gpu (Mali-G610 MC4, Valhall)
layer: L2 (generated OpenCL function bodies)
evidence_basis: [vendor_doc, backend_contract]
version_sensitivity: high -- the device OpenCL compiler performs final code generation
---

# OpenCL source optimization on Mali-G610

This page authorizes only body-level edits to kernels already present in the
read-only generated OpenCL unit. Global graph, algorithm, fusion and fixed-state
choices belong to L0. Bounded local fusion, NDRange and TIR structure belong to
L1. Candidate-side `FoldConstant`, `FuseOps`, `FuseTIR`, DLight and MetaSchedule
remain outside this layer's authority.

## Execution constraints

Read the device's OpenCL C version, extensions and resource limits. Do not
assume CUDA warp operations, Adreno image paths or OpenCL subgroup support.
Preserve signatures, launch geometry, memory address spaces and synchronization.
[leka-rk3588-contract]

## Candidate mechanisms

- Use contiguous loads and stores. Explicit vector forms such as `float2`,
  `float4`, `vload4` and `vstore4` require valid bounds, alignment and tails;
  compare the generated result with scalar code. [arm-opencl-guide]
- Hoist invariant address expressions and remove redundant conversions only
  when range analysis proves that overflow and signedness behavior are
  unchanged.
- Unroll short compile-time loops when the register and instruction costs are
  acceptable. Evaluate code growth and spills with compiler evidence where
  available, followed by device timing.
- Consider independent accumulators or fused arithmetic when the numerical
  contract permits them; preserve any required reduction ordering.
- Preserve uniform participation in workgroup barriers. Local-memory staging
  is useful only when its reuse justifies copies and synchronization. Changes
  to the staging structure belong to L1. [arm-opencl-guide]

Use Mali Offline Compiler output as static resource estimates, separate from
runtime measurements. Confirm improvements using the evaluator's paired
latency and dispersion fields. Missing counters do not establish a bottleneck.
[arm-malioc]

[leka-rk3588-contract]: ../../sources/rk3588_gpu/backend_contract.json
[arm-opencl-guide]: https://developer.arm.com/-/media/developer/Graphics%20and%20Multimedia/Developer%20Guides%20-%20PDFs/Arm%20Guide%20to%20OpenCL%20Programming.pdf
[arm-malioc]: https://developer.arm.com/tools-and-software/mali-offline-compiler
