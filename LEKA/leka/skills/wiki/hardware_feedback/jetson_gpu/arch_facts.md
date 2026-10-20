---
id: jetson_gpu/arch_facts
hardware: jetson_gpu (Orin integrated Ampere, CUDA)
tags: [facts, memory, execution-width]
layers: []
evidence_basis: [vendor_doc, backend_contract]
version_sensitivity: mixed -- verify compiler and runtime limits for each deployment
---

# Jetson GPU: architecture facts

Use the active backend contract for code-generation legality and a current
runtime query for device properties. Product-family maxima do not identify the
attached module or its available resources. [leka-jetson-contract]

## Product and architecture

- Jetson AGX Orin uses an integrated NVIDIA Ampere GPU. The full configuration
  provides 16 streaming multiprocessors, 2048 CUDA cores and 64 Tensor Cores;
  other module configurations may differ. [nvidia-orin-tech-brief]
- The full configuration has a published LPDDR5 peak bandwidth of 204.8 GB/s.
  This is a product specification, not sustained application bandwidth.
  [nvidia-orin-reviewer-guide]
- Compute capability 8.7 uses 32-thread warps, permits up to 1024 threads per
  block, and provides 64K 32-bit registers per block. Its larger shared-memory
  allocations require an explicit CUDA opt-in path. [nvidia-cuda-cc]

## LEKA target and planning limits

- The target is `cuda -arch=sm_87`. Plan for at most 1024 threads, 65,536
  registers and 49,152 bytes of shared memory per block. These conservative
  constraints must also satisfy the actual compiler and device limits.
  [leka-jetson-contract]
- The 49,152-byte shared-memory planning ceiling does not describe the
  hardware's physical opt-in maximum. A candidate cannot assume the host
  build enables a larger allocation. [leka-jetson-contract] [nvidia-cuda-cc]
- Query device identity, available memory, resource limits, CUDA versions and
  power state for the current run. Do not infer them from a module name.
- Only counters present in a successful profiler result may support a
  bottleneck hypothesis. Missing DRAM metrics do not imply zero traffic.

[nvidia-orin-tech-brief]: https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf
[nvidia-orin-reviewer-guide]: https://developer.nvidia.com/sites/default/files/akamai/Jetson_AGX_Orin_Developer_Kit_RG_0.pdf
[nvidia-cuda-cc]: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
[leka-jetson-contract]: ../../sources/jetson_gpu/backend_contract.json
