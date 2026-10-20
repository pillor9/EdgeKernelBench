---
id: L1_tir/rk3588_gpu/reference_gemm_structures
hardware: rk3588_gpu (Mali-G610 MC4, Valhall)
layer: L1 (TIR tiling, threading and memory-scope decisions)
evidence_basis: [vendor_doc, tvm_source]
---

# Reference GEMM structures on Mali

Apply these mechanisms to an already generated GEMM PrimFunc or its declared
closed 2–3-call local fusion under `leka.l1_fuse_of`. A compatible epilogue may
be combined with the TIR implementation. Larger graph, algorithm-family,
fixed-state and external-interface changes belong to L0.

- **Register blocking.** Assign each work-item a small output patch and reuse
  operands across independent accumulators. Derive patch dimensions from the
  dtype, shape, compiler output and register pressure.
- **Contiguous accesses.** Map neighboring work-items to neighboring elements.
  Use vector-width accesses only when alignment, extent and tails are valid.
  [arm-opencl-guide]
- **Workgroup geometry.** Compare shapes using the current device and compiled
  kernel limits. Equal workgroup sizes can have different access patterns and
  edge utilization; measure their effect on the complete candidate.
- **Storage reuse.** Compare direct reads with cooperative staging where the
  existing dataflow permits it. Mali local storage does not have the same
  implementation as CUDA shared SRAM, so account for copy and barrier costs.
  [arm-opencl-guide]
- **Reduction structure.** Choose reduction tiling and independent partials
  from the actual extents. The pinned TVM OpenCL matmul rule provides a
  mechanism reference; its constants are not a tuned configuration for the
  current task. [tvm-dlight-opencl-matmul]

Layout changes and static parameter packing remain L0 decisions. A useful
proposal states the expected reuse, resource cost and timing observation that
would reject the hypothesis. Promotion requires target compilation,
independent correctness checks and paired device timing.

[arm-opencl-guide]: https://developer.arm.com/-/media/developer/Graphics%20and%20Multimedia/Developer%20Guides%20-%20PDFs/Arm%20Guide%20to%20OpenCL%20Programming.pdf
[tvm-dlight-opencl-matmul]: https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/dlight/gpu/matmul.py
