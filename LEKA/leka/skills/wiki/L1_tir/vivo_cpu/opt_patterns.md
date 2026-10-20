---
id: vivo_cpu/opt_patterns
hardware: vivo_cpu (Snapdragon big.LITTLE AArch64, NEON)
tags: [fold, launches, access, tiling, reduction]
layers: [L1]
evidence_basis: [tvm_source, backend_contract]
version_sensitivity: high -- exact TVM 0.23.0 syntax, generic AArch64 target, and runtime policy
---

# vivo CPU: TVM 0.23 syntax and mechanism contract

Ownership boundary: L0/T01 remain immutable canonical references. L1 may
reschedule existing PrimFuncs and jointly fuse one closed region of 2–3
consecutive pure single-output calls under the `leka.l1_fuse_of` contract.
All unrelated graph structure and nonregion ABIs stay frozen; broader graph,
algorithm, or fixed-state changes go to L0. The fixed
`LegalizeOps/FuseTIR(explicit Primitive helpers only)/DeadCodeElimination`
chain and the separate executable L1 seed are translation/seed references,
never candidate APIs or graph/fusion policy.

## Version lock and model-visible syntax

- Syntax is locked to Apache TVM `0.23.0`, commit
  `75ce1b4d46053b28fa926a6e29aa715b2463dc92`. Unversioned TVM pages may
  document a later IR dialect. Candidate syntax and compilation must match
  the configured TVM toolchain.
  [tvm-023-release] [tvm-compatibility]
- Reference snippets may omit imports for brevity. Every complete candidate
  file must include `import tvm` and its required legacy aliases from
  `tvm.script` (`ir as I`, `tir as T`, and `relax as R`). The worker does not
  inject these imports. Do not use the newer `tirx` dialect.
  [tvm-023-tir-parser]
- Use `@I.ir_module`, `@T.prim_func(private=True)`, `with T.block(...)`,
  `T.alloc_buffer`, `T.axis.*`, and `@R.function`. Relax invokes TIR with
  `R.call_tir(..., out_sinfo=...)`; lists denote multiple outputs,
  `R.Tensor((), dtype=...)` denotes a scalar tensor, and
  `tir_vars=R.shape([...])` carries unpacked scalar TIR arguments after output
  buffers. [tvm-023-tir-builder] [tvm-023-call-tir]
  [tvm-023-call-tir-rewrite]
- Construct Relax primitive values with `R.prim_value(...)`. Use `T.And` and
  `T.Or` for compound predicates; `T.Cast`, `T.cast`, `T.Select`, and
  `T.if_then_else` belong to the pinned surface. [tvm-023-relax-builder]
  [tvm-023-tir-builder]

## Candidate-legal LLVM/AArch64 mechanisms

- CPU PrimFuncs use ordinary loops: `range`/`T.serial`, `T.parallel`,
  `T.vectorized`, `T.unroll`, and `T.grid`. There are no GPU launch axes,
  shared-memory scopes, or GPU barriers in this backend. [tvm-023-tir-builder]
  [leka-vivo-contract]
- Parallelize only an independent outer domain whose iterations do not write
  overlapping output state. Select the runtime policy exactly from the injected
  capability card; do not infer a homogeneous core count or scheduling policy
  from the SoC name. [leka-vivo-contract]
- Vectorize a provably contiguous innermost domain, with a factor compatible
  with dtype, alignment, divisibility, and the frozen AArch64 ISA. Keep an
  explicit tail when divisibility is not guaranteed. Let LLVM lower the legal
  vector loop; do not bypass the target with raw ISA calls.
  [tvm-023-tir-builder] [leka-vivo-contract]
- For matrix-product-shaped or convolution-shaped work, use cache blocking and
  a small local accumulator tile only when the shape exposes reuse across both
  output dimensions. Treat block sizes, accumulator shapes, vector factors, and
  unroll depths as shape- and resource-dependent search variables rather than
  constants. [tvm-dlight-cpu-gemv] [leka-vivo-contract]
- For GEMV-shaped inner reductions, the only CPU DLight rule in this release is
  useful as a mechanism reference: it recognizes a restricted normalized GEMV,
  allows an injective epilogue, tiles spatial/reduction domains, vectorizes,
  parallelizes an outer spatial loop, and annotates unrolling. Its numeric
  defaults are not measurements for this heterogeneous phone and candidate-side
  DLight/Schedule use remains forbidden. [tvm-dlight-cpu-gemv]
- For reductions, keep per-thread accumulators private, preserve a defensible
  accumulation order, and combine once per output. Never apply `T.parallel`
  directly to a reduction axis unless the implementation creates disjoint
  partial outputs and performs a correct explicit combine.
  [tvm-023-tir-builder]
- Static parameter relayout and global fusion are L0-only choices. L1 may
  co-design one declared closed 2–3-call fusion and TIR schedule; fixed-state
  plans, unrelated graph structure, and nonregion ABIs stay frozen.
  [leka-vivo-contract]

## Reference-only ARM intrinsics

- TVM 0.23 registers NEON int8 and dot-product tensor intrinsics, and also
  contains SME intrinsics. Those registrations describe compiler capability,
  not this candidate's target capability. [tvm-023-arm-intrin]
- The official MetaSchedule tests request NEON with `-mattr=+neon` and dot
  product with `-mattr=+neon,+v8.2a,+dotprod`. The frozen vivo target is
  `mcpu=generic` and does not grant optional extension use, so SDOT, UDOT, HDOT,
  i8mm, BF16, SVE, and SME remain forbidden unless a future target contract
  explicitly enables and validates them. [tvm-023-arm-target-tests]
  [leka-vivo-contract]

## Applicability and promotion

- Tag every proposal with backend `llvm`, Android AArch64 triple, exact TVM
  commit, dtype, shape predicate, contiguous/vectorized axis, parallel domain,
  runtime-policy ID, and scratch footprint. [leka-vivo-contract]
- Parser round-trip proves syntax only. Promotion requires frozen-target
  compilation, independent correctness, a fresh runtime-policy/affinity proof,
  and timing on the physical phone. [tvm-023-roundtrip]
  [leka-vivo-contract]

## Source IDs

- [tvm-023-release]: [Apache TVM v0.23.0, commit `75ce1b4`](https://github.com/apache/tvm/releases/tag/v0.23.0)
- [tvm-023-tir-parser]: [Pinned TIR parser entry](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/parser/tir/entry.py)
- [tvm-023-tir-builder]: [Pinned TIR script builder](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/ir_builder/tir/ir.py)
- [tvm-023-relax-builder]: [Pinned Relax script builder](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/ir_builder/relax/ir.py)
- [tvm-023-call-tir]: [Pinned `R.call_tir` signature](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/relax/op/base.py)
- [tvm-023-call-tir-rewrite]: [Pinned call-TIR ABI rewrite](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/src/relax/transform/call_tir_rewrite.cc)
- [tvm-023-arm-intrin]: [Pinned ARM CPU tensor intrinsics](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/tir/tensor_intrin/arm_cpu.py)
- [tvm-023-arm-target-tests]: [Pinned ARM intrinsic target tests](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/tests/python/meta_schedule/test_meta_schedule_post_order_apply.py)
- [tvm-023-roundtrip]: [Pinned TVMScript round-trip tests](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/tests/python/tvmscript/test_tvmscript_roundtrip.py)
- [tvm-dlight-cpu-gemv]: [Pinned CPU GEMV rule](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/dlight/cpu/gemv.py)
- [leka-vivo-contract]: [LEKA CPU backend contract](../../sources/vivo_cpu/backend_contract.json)

Evidence key: `vivo_cpu/opt_patterns`.
