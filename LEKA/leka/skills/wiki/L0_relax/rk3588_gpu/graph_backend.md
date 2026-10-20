---
id: L0_relax/rk3588_gpu/graph_backend
hardware: rk3588_gpu (Mali-G610 Valhall, OpenCL)
layer: L0 (Relax graph and tolerance-valid algorithm rewrites)
evidence_basis: [vendor_doc, device_contract, tvm_source]
version_sensitivity: medium -- revalidate after Mali driver/target or Relax pass changes
---

# Relax graph planning for RK3588 GPU

Read `wiki/hardware/rk3588_gpu/arch_facts.md` before choosing a graph family.
The G610 uses unified memory and its OpenCL local address space is not
CUDA-style dedicated shared SRAM; L0 should optimize graph traffic and kernel
boundaries without inventing lower-layer storage assumptions.

- Explicitly fuse tolerance-valid elementwise tails and shared preprocessing when that removes a
  materialized tensor or launch. Verify the final lowered breakdown because
  OpenCL code generation may split a graph differently from CUDA.
- Favor tolerance-valid graph rewrites that reduce full-tensor traffic or
  repeated reductions. Preserve accumulation/dtype semantics required by the
  frozen PyTorch projection and tolerances.
- Fixed-parameter transpose/packing may run outside the timed entry only through
  `PARAMETER_PLAN_SCHEMA.json`; every dynamic or undeclared transform is timed.
- Layout planning should expose contiguous vectorizable axes, but workgroup
  size, warp/subgroup assumptions, local-memory staging, and OpenCL C syntax
  belong to L1/L2 and must not appear in a Relax candidate.
- If the profile becomes one dominant kernel, avoid repeatedly reshaping an
  already sound graph merely to chase a lower-layer issue; deliver the graph
  constraint and evidence to TIR instead.
