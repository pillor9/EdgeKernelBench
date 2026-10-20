---
id: L2_source/jetson_gpu/cuda_source_opt
hardware: jetson_gpu (Orin, Ampere, compute capability 8.7)
layer: L2 (generated CUDA function bodies)
evidence_basis: [vendor_doc, backend_contract]
version_sensitivity: high -- compile with the configured CUDA toolchain
---

# CUDA source optimization on Orin

This page authorizes only body-level edits to kernels already present in the
read-only generated CUDA unit. Global graph, algorithm, fusion and fixed-state
choices belong to L0. Bounded local fusion, launch geometry and TIR structure
belong to L1. Candidate-side `FoldConstant`, `FuseOps`, `FuseTIR`, DLight and
MetaSchedule remain outside this layer's authority.

## Resource constraints

Respect the current launch geometry, function signatures and backend limits.
The 49,152-byte shared-memory planning ceiling requires no assumption about a
larger CUDA opt-in allocation. Missing profiler metrics remain unknown.
[leka-jetson-contract] [nvidia-cuda-cc]

## Candidate mechanisms

- Unroll short inner loops with compile-time bounds, checking register use
  and instruction growth. More unrolling can increase resource pressure.
- Use independent accumulators to shorten dependency chains when the task's
  numerical tolerance permits the final combination.
- Hoist repeated address arithmetic and reuse already loaded values where
  aliasing and control-flow semantics permit it.
- Consider vector loads and stores only when the full access, including
  offsets and tails, satisfies alignment and bounds requirements.
- Ampere supports asynchronous global-to-shared copies. Use them only within
  an existing compatible staging structure, preserving all waits and barriers
  needed before consumption. Changes to storage or launch structure go to L1.
  [nvidia-ampere-tuning]

Choose a mechanism from the generated body and collected evidence, then check
correctness and paired timing. Do not alter signatures, launch geometry,
host-side compilation settings or task tolerances. Approximate arithmetic is
subject to the same independent correctness checks as any other candidate.
[nvidia-cuda-best-practices]

[leka-jetson-contract]: ../../sources/jetson_gpu/backend_contract.json
[nvidia-cuda-cc]: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
[nvidia-ampere-tuning]: https://docs.nvidia.com/cuda/archive/12.6.0/ampere-tuning-guide/index.html
[nvidia-cuda-best-practices]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
