---
id: L1_tir/jetson_gpu/reference_gemm_structures
hardware: jetson_gpu (Orin, Ampere sm_87)
layer: L1 (TIR tiling, threading and memory-scope decisions)
evidence_basis: [vendor_doc, backend_contract]
---

# Reference GEMM structures on CUDA

Apply these mechanisms to an already generated GEMM PrimFunc or its declared
closed 2–3-call local fusion under `leka.l1_fuse_of`. A compatible epilogue may
be combined with the TIR implementation. Larger graph, algorithm-family,
fixed-state and external-interface changes belong to L0.

CUTLASS describes nested tiling across threadblocks, warps and per-thread
registers. A threadblock owns an output tile; shared operand tiles support
reuse across warps; each thread accumulates an output patch in registers.
Software pipelining can overlap transfers and computation when the storage
and synchronization structure supports it. [nvidia-cutlass-gemm]

In TIR, express these choices through spatial and reduction splits,
`blockIdx`/`threadIdx` bindings, shared buffers and local accumulators. Choose
shapes from the task extents and resource limits. A larger tile may improve
reuse while reducing occupancy or leaving too little independent work.
[nvidia-cutlass-gemm]

Plan within the 49,152-byte per-block shared-memory ceiling and the active
backend contract. Introducing or resizing buffers belongs to L1; source-level
instruction forms belong to L2. Hardware support for Tensor Cores or async
copy does not override the candidate validator or authorize extra intrinsics.
[leka-jetson-contract]

[nvidia-cutlass-gemm]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
[leka-jetson-contract]: ../../sources/jetson_gpu/backend_contract.json
