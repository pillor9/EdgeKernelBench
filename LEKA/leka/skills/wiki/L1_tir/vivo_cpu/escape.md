---
id: vivo_cpu/escape
hardware: vivo_cpu (V2403A, SM8650, AArch64 Android CPU)
tags: [escape, frame-change, graph, dataflow]
layers: [L1]
evidence_basis: [tvm_source, inferred]
version_sensitivity: high -- advanced loop/dataflow changes depend on LLVM lowering and runtime policy
---

# Vivo CPU: reconsidering an optimization strategy

When repeated variations of one mechanism stop improving, consider a different
implementation strategy. Another cache tile, vector width, unroll factor, or
thread count tests the same mechanism. Keep the accepted implementation and
search history while reconsidering the hypothesis. This is search guidance for
the agent; the controller does not automatically reject repeated mechanisms.

Eligible L1 frame changes may reschedule an existing generated PrimFunc or
jointly fuse one closed 2–3-call canonical region under `leka.l1_fuse_of`.
Examples of body-local changes:

- output-owned reduction versus thread partials and explicit combine;
- cache-panel product versus scarce-axis/batched-small dataflow;
- ordinary loop nest versus software-pipelined independent accumulators;
- uniform parallel chunks versus topology-aware chunks chosen only from the
  configured runtime-policy contract.

Whole-graph fusion/split, a different convolution algorithm family, static
parameter layout, or changes beyond the bounded local-fusion contract are
L0-only. Local fusion preserves external interfaces, private intermediates,
and unrelated graph/ABI structure; send larger changes to L0.

State old/new frames, semantic and numerical invariants, expected PMU/timing
change, affinity/cache risk, and rollback. The generic AArch64 target remains
absolute: observed dot-product, i8mm, BF16, or core-specific features cannot be
enabled by an escape [leka-vivo-contract].
