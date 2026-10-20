---
id: jetson_gpu/escape
hardware: jetson_gpu (Jetson AGX Orin, CUDA sm_87)
tags: [escape, frame-change, graph, dataflow]
layers: [L1]
evidence_basis: [vendor_doc, tvm_source, inferred]
version_sensitivity: high -- advanced dataflows are compiler and resource sensitive
---

# Jetson GPU: reconsidering an optimization strategy

When repeated variations of one mechanism stop improving, consider a different
implementation strategy. Another tile, vector width, unroll factor, or block
size tests the same mechanism. Preserve the accepted candidate and search
history while reconsidering the hypothesis. This is search guidance for the
agent; the controller does not automatically detect or reject such repetitions.

Eligible L1 frame changes may reschedule an existing generated PrimFunc or
jointly fuse one closed 2–3-call canonical region under `leka.l1_fuse_of`.
Examples of body-local changes:

- output-parallel reduction to block/multistage reduction, or the reverse;
- square output tiling to scarce-axis streaming/batched products;
- ordinary staging to double-buffer/software-pipelined staging when generated
  synchronization and resource use remain legal;

Whole-graph fusion/split/recomputation, a new convolution algorithm family,
static parameter layout, or changes beyond the bounded local-fusion contract
are L0-only escapes. Send those to L0. Local fusion must preserve external
interfaces, keep intermediates private, and retain all unrelated graph/ABIs.

State the previous and proposed strategies, semantic invariant, expected
profile change, resource risk, and rollback parent. A renamed parameter sweep
does not test a different mechanism. The live CUDA contract still governs
candidate legality; product resource maxima do not authorize resources absent
from `BACKEND_FACTS_JSON` [leka-jetson-contract].
