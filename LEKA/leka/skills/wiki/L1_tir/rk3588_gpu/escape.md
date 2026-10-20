---
id: rk3588_gpu/escape
hardware: rk3588_gpu (RK3588S, Mali-G610, OpenCL)
tags: [escape, frame-change, graph, dataflow]
layers: [L1]
evidence_basis: [vendor_doc, tvm_source, inferred]
version_sensitivity: high -- advanced OpenCL dataflows depend on TVM and the commercial Mali driver
---

# RK3588 GPU: reconsidering an optimization strategy

When repeated variations of one mechanism stop improving, consider a different
implementation strategy. Another workgroup, tile, vector width, or unroll value
tests the same mechanism. Preserve the accepted candidate and search history
while reconsidering the hypothesis. This is search guidance for the agent;
the controller does not automatically detect or reject such repetitions.

Eligible L1 frame changes may reschedule an existing generated PrimFunc or
jointly fuse one closed 2–3-call canonical region under `leka.l1_fuse_of`.
Examples of body-local changes:

- work-item-owned versus workgroup-owned reduction;
- local-panel tiling versus private-register streaming when local copies and
  barriers fail to repay reuse;
- square product tiling versus scarce-axis or batched-small dataflow;
- ordinary staging versus software-pipelined/double-buffered staging only when
  emitted OpenCL and local-memory/barrier use validate.

Whole-graph fusion/split/recomputation, a different convolution algorithm
family, static parameter layout, or changes beyond the bounded local-fusion
contract are L0-only. Send those to L0; the local exception preserves external
interfaces, private intermediates, and all unrelated graph/ABI structure.

The packet must name old/new frames, semantic invariant, expected OpenCL event
and optional counter/static-code change, resource risk, and rollback. A CUDA
or Adreno schedule requires an explicit port and validation before use on Mali
[leka-rk3588-contract].
