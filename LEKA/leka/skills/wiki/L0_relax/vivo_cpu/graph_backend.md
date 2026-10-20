---
id: L0_relax/vivo_cpu/graph_backend
hardware: vivo_cpu (Snapdragon 8 Gen 3 heterogeneous AArch64)
layer: L0 (Relax graph and tolerance-valid algorithm rewrites)
evidence_basis: [vendor_doc, device_contract, tvm_source]
version_sensitivity: medium -- revalidate after runtime policy/target or Relax pass changes
---

# Relax graph planning for Vivo CPU

Read the mounted `wiki/hardware/vivo_cpu/arch_facts.md` fact volume for the frozen generic AArch64
target and heterogeneous-core boundary. Physical optional ISA features do not
authorize L0 to assume a wider compiler target.

- CPU VM and packed-call overhead can make chains of small operators costly.
  Explicitly encode tolerance-valid fusion and common-subexpression elimination when they remove
  dispatches and intermediate memory, then confirm with the trusted breakdown.
- Tolerance-valid algorithmic rewrites that cut complete tensor passes or repeated
  reductions often matter more than exposing GPU-shaped parallel fragments.
- Keep enough outer independent work for L1 to parallelize. A graph rewrite
  that creates many tiny kernels or one serial dependency chain can erase the
  arithmetic saving on this heterogeneous CPU.
- Fixed-state packing/transposition/layout may run outside the timed entry only
  through `PARAMETER_PLAN_SCHEMA.json`; dynamic or undeclared preparation is
  timed, and the pass pipeline never invents persistent state.
- Core affinity, thread count, cache blocking, vector width, LLVM intrinsics,
  and instruction selection are lower-layer/runtime-policy vocabulary. L0 may
  expose shape/layout opportunity but must not hard-code those mechanisms.
