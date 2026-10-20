---
id: L0_relax/common/graph_opt
hardware: all backends (backend placement notes inline)
layer: L0 (Relax graph rewrites)
evidence_basis: [tvm_source]
note: correctness authority is the task semantics and declared tolerance; every rewrite must pass it
---

# Graph-level optimization patterns (Relax)

## Graph transformations
- **Kernel-boundary control**: L0 explicitly chooses the intended fused/split
  regions with private Relax helpers carrying
  `R.func_attr({"Primitive": True})`, called statically from the module. Many
  tiny ops -> many launches (launch_overhead verdict on GPU digests;
  vm_overhead rows in the breakdown). T01 does not run `FuseOps`;
  `FuseTIR` only realizes L0's marked helper and never plans a new boundary.
  Unsupported marked combinations fail at T0 instead of being repartitioned.
  L1 additionally may fuse one closed 2–3-call canonical region jointly with
  TIR under its explicit `leka.l1_fuse_of` contract. This does not change L0/T01,
  confer whole-graph authority on L1, or make T01 plan fusion automatically.
- **Algorithm-family substitution** (tolerance-valid): associativity
  reordering that shrinks the dominant dimension of a chain of matmuls;
  factoring shared subexpressions into one computed value; online/streaming
  reduction forms that remove an intermediate tensor.
- **Layout planning under the current ABI**: change activation layouts and
  graph structure when the saved traffic exceeds the conversion cost. For
  declared fixed state, `PARAMETER_PLAN_SCHEMA.json` is the only legal
  evaluator-owned outside-timing preparation path; dynamic-input-dependent
  transforms remain timed.
- **Constant folding**: L0 chooses the simplification and submits its already
  folded Relax form. T01 intentionally does not run `FoldConstant`, so do not
  delegate identities or constant subgraphs to the framework.

## Layer boundaries
- Thread mapping, tiling, and vectorization belong to L1. Describe the graph
  transformation and any scheduling opportunities it creates.
- No hand-written TIR, `R.call_tir*`, or combined Relax+TIR candidate. The L0
  agent may rebuild the whole Relax graph, but the layer boundary remains pure
  Relax and the trusted `main` ABI remains frozen.
- Splitting one reduction (for example split-K) trades a shorter dependency
  chain for extra partial storage, launches and a combine. Treat it as a
  hypothesis whose value depends on the live graph and device evidence; validate the
  hypothesis with the current task.
- A graph identical to the current base is not a candidate (the driver rejects
  no-change promotions).

## Backend placement notes
- jetson_gpu / rk3588_gpu: test fusion when dispatch and intermediate storage
  are significant. Use L1/L2 to address a dominant kernel's schedule or source.
- vivo_cpu: inspect per-operation timing before deciding whether dispatch,
  materialized intermediates or a hot PrimFunc is the dominant opportunity.
