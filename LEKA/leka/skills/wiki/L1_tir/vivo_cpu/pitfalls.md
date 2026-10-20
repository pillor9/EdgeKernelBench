---
id: vivo_cpu/pitfalls
hardware: vivo_cpu (Snapdragon big.LITTLE AArch64, NEON)
tags: [compile-error, lowering, abi, numerics]
layers: [L1]
evidence_basis: [tvm_source, backend_contract]
version_sensitivity: high -- exact TVM 0.23.0 parser, generic AArch64 ISA, and runtime-policy failures
---

# vivo CPU: pinned failures and forbidden migrations

Relax syntax below is diagnostic ABI context only unless used for the explicit
`leka.l1_fuse_of` bounded local-fusion contract: one closed 2–3-call canonical
chain and its fused TIR. Every other Relax function/call remains read-only.
The local exception does not authorize an algorithm, layout, or whole-graph edit.

## Version and API failures

- The compiler is TVM `0.23.0` at
  `75ce1b4d46053b28fa926a6e29aa715b2463dc92`. Use legacy
  `tvm.script.tir as T`, `T.block`, `T.alloc_buffer`, and
  `R.call_tir(..., out_sinfo=...)`. The unversioned site's `tirx`, `T.sblock`,
  `T.sblock_alloc_buffer`, `s_tir=True`, and `out_ty=` fail this parser. Wiki
  fragments omit imports intentionally; do not infer the newer namespace.
  [tvm-023-release] [tvm-023-tir-parser] [tvm-023-call-tir]
- Use `R.prim_value(...)`, not `R.PrimValue(...)`. `T.all` and `T.any` are
  absent; combine predicates with `T.And` and `T.Or`.
  [tvm-023-relax-builder] [tvm-023-tir-builder]
- Keep every `call_tir` callee in the same IRModule. Tensor inputs precede
  output buffers, `tir_vars` scalars follow outputs, a list of `out_sinfo`
  entries returns a tuple, and `R.Tensor((), dtype=...)` represents a zero-rank
  tensor. [tvm-023-call-tir] [tvm-023-call-tir-rewrite]

## LLVM/AArch64 contract failures

- Do not use `T.thread_binding`, GPU storage scopes, or
  `T.tvm_storage_sync` in a CPU PrimFunc. The candidate-legal loop surface is
  `range`/`T.serial`, `T.parallel`, `T.vectorized`, `T.unroll`, and `T.grid`.
  [tvm-023-tir-builder] [leka-vivo-contract]
- `T.parallel` is legal only for independent iterations. Parallelizing a
  reduction axis or two iterations that can update the same output introduces a
  race even when a small correctness sample happens to pass.
  [tvm-023-tir-builder]
- A loop body alone is not an allocation frame for `T.alloc_buffer` in this
  pinned parser. For scratch owned by each parallel tile, put a named
  `with T.block("tile"):` inside the owning loop, then allocate, initialize,
  and use the scratch in that block. Do not fix a missing-frame error by
  sharing one scratch buffer across parallel iterations. The error signature
  `tvm.tir.frame_missing` identifies this syntax issue; it says nothing about
  numerical correctness or performance. [tvm-023-tir-builder]
- Use exactly a runtime-policy ID exposed by the injected card when one is
  required. Do not invent a thread count, assume homogeneous cores, or encode
  affinity in candidate code. The runtime must prove the selected policy fresh
  for the run. [leka-vivo-contract]
- The frozen target is conservative generic AArch64. Physical observation of
  optional instructions is not permission to emit them. Do not hand-author raw
  LLVM/ISA intrinsics or assume SDOT, UDOT, HDOT, i8mm, BF16, FP16 arithmetic,
  SVE, SVE2, or SME without an explicit future target feature contract.
  [tvm-023-arm-intrin] [tvm-023-arm-target-tests]
  [leka-vivo-contract]
- `T.vectorized` on a strided/gathered axis is not a vectorization contract.
  Prove unit stride, alignment, dtype compatibility, and tail handling; then
  inspect frozen-target lowering and physical-device behavior. Do not copy a
  vector factor or unroll depth from DLight. [tvm-dlight-cpu-gemv]
  [leka-vivo-contract]
- Candidate-side `tir.Schedule`, DLight, MetaSchedule, and tensorization remain
  forbidden. Upstream CPU rules and ARM intrinsics are mechanism references,
  not APIs that the candidate may invoke. [tvm-dlight-cpu-gemv]
  [tvm-023-arm-intrin]

## Semantic and evidence failures

- Reassociating a long reduction, narrowing an accumulator, fusing an operation
  with different rounding, omitting a tail, or replacing infinity with a finite
  sentinel may change results. Preserve the declared dtype and output tree and
  let the independent correctness path judge any deliberate reformulation.
  [leka-vivo-contract]
- Parser round-trip, LLVM IR generation, or an upstream schedule does not prove
  speed. Promotion requires frozen-target compilation, independent correctness,
  fresh runtime-policy/affinity evidence, and physical-phone timing.
  [tvm-023-roundtrip] [leka-vivo-contract]

## Source IDs

- [tvm-023-release]: [Apache TVM v0.23.0](https://github.com/apache/tvm/releases/tag/v0.23.0)
- [tvm-023-tir-parser]: [Pinned TIR parser entry](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/parser/tir/entry.py)
- [tvm-023-tir-builder]: [Pinned TIR script builder](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/ir_builder/tir/ir.py)
- [tvm-023-relax-builder]: [Pinned Relax script builder](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/ir_builder/relax/ir.py)
- [tvm-023-call-tir]: [Pinned `R.call_tir` signature](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/relax/op/base.py)
- [tvm-023-call-tir-rewrite]: [Pinned call-TIR ABI rewrite](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/src/relax/transform/call_tir_rewrite.cc)
- [tvm-023-arm-intrin]: [Pinned ARM CPU tensor intrinsics](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/tir/tensor_intrin/arm_cpu.py)
- [tvm-023-arm-target-tests]: [Pinned ARM target-feature tests](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/tests/python/meta_schedule/test_meta_schedule_post_order_apply.py)
- [tvm-023-roundtrip]: [Pinned TVMScript round-trip tests](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/tests/python/tvmscript/test_tvmscript_roundtrip.py)
- [tvm-dlight-cpu-gemv]: [Pinned CPU GEMV rule](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/dlight/cpu/gemv.py)
- [leka-vivo-contract]: [LEKA CPU backend contract](../../sources/vivo_cpu/backend_contract.json)

Evidence key: `vivo_cpu/pitfalls`.
