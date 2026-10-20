---
id: L0_relax/jetson_gpu/graph_backend
hardware: jetson_gpu (Jetson AGX Orin, CUDA sm_87)
layer: L0 (Relax graph and tolerance-valid algorithm rewrites)
evidence_basis: [vendor_doc, device_contract, tvm_source]
version_sensitivity: medium -- revalidate after target/runtime or Relax pass changes
---

# Relax graph planning for Jetson GPU

Read `wiki/hardware/jetson_gpu/arch_facts.md` for the actual target contract.
Those facts constrain the graph's eventual execution, but CUDA scheduling
syntax remains L1/L2 vocabulary.

- Explicitly choose and encode long fused producer/consumer regions in Relax.
  On this CUDA backend, unnecessary materialized elementwise intermediates add
  both unified-memory traffic and launches; confirm the effect in the trusted
  kernel breakdown rather than assuming every syntactic chain fuses.
- Tolerance-valid reassociation of matrix products or shared projections is valuable
  when it reduces total arithmetic or avoids repeated reads. Preserve the
  frozen positional ABI, output tree, dtype behavior, and PyTorch semantics.
- Fixed-state transpose/packing/layout may run outside the timed entry only
  through `PARAMETER_PLAN_SCHEMA.json`; otherwise it remains timed Relax work.
  The fixed pass pipeline never invents free preparation.
- Tensor-core or `cp.async` feasibility is not a graph-level fact. L0 may
  expose compatible shapes/layouts, then use T1 and hardware evidence to hand
  the mechanism to L1/L2 without embedding TIR or CUDA.
- When many short kernels dominate, first test fusion/boundary changes. When a
  single kernel dominates, avoid gratuitous graph churn and let lower layers
  own tile, thread, memory-scope, and instruction decisions; L0 still owns any
  justified tolerance-valid high-level algorithm change.
