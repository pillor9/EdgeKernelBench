---
id: rk3588_gpu/opt_patterns
hardware: rk3588_gpu (Mali-G610 Valhall, OpenCL)
tags: [fold, launches, access, tiling, reduction]
layers: [L1]
evidence_basis: [tvm_source, backend_contract]
version_sensitivity: high -- exact TVM 0.23.0 syntax, target classification, and OpenCL lowering
---

# RK3588 GPU: TVM 0.23 syntax and mechanism contract

Ownership boundary: L0/T01 remain immutable canonical references. L1 may
reschedule existing PrimFuncs and jointly fuse one closed region of 2–3
consecutive pure single-output calls under the `leka.l1_fuse_of` contract.
All unrelated graph structure and nonregion ABIs stay frozen; broader graph,
algorithm, or fixed-state changes go to L0. The fixed
`LegalizeOps/FuseTIR(explicit Primitive helpers only)/DeadCodeElimination`
chain and the separate DLight executable seed are translation/seed references,
never candidate APIs or graph/fusion policy.

## Version lock and model-visible syntax

- Syntax is locked to Apache TVM `0.23.0`, commit
  `75ce1b4d46053b28fa926a6e29aa715b2463dc92`. Never mix this page with the
  unversioned TVM site, which may describe a later IR dialect. Candidate
  compilation must match the configured TVM toolchain.
  [tvm-023-release] [tvm-compatibility]
- Reference snippets may omit imports for brevity. Every complete candidate
  file must include `import tvm` and its required legacy aliases from
  `tvm.script` (`ir as I`, `tir as T`, and `relax as R`). The worker does not
  inject these imports. Do not use the newer `tirx` dialect.
  [tvm-023-tir-parser]
- Use `@I.ir_module`, `@T.prim_func(private=True)`, `with T.block(...)`,
  `T.alloc_buffer`, `T.axis.*`, and `@R.function`. Call a PrimFunc with
  `R.call_tir(..., out_sinfo=...)`; use a list for multiple outputs,
  `R.Tensor((), dtype=...)` for a scalar tensor, and
  `tir_vars=R.shape([...])` for unpacked scalar TIR arguments.
  [tvm-023-tir-builder] [tvm-023-call-tir]
  [tvm-023-call-tir-rewrite]
- Use `R.prim_value(...)` for Relax primitive scalars and `T.And`/`T.Or` for
  compound TIR predicates. [tvm-023-relax-builder]
  [tvm-023-tir-builder]

## Candidate-legal Mali OpenCL mechanisms

- The frozen target is OpenCL with `mali` target keys and a Linux AArch64 host.
  Map workgroups with `blockIdx.*`, work-items with `threadIdx.*`, per-work-item
  scratch with `scope="local"`, and workgroup-shared scratch with
  `scope="shared"`. TVM lowers these scopes to the OpenCL address spaces; all
  extents and storage must remain within the injected card.
  [tvm-023-opencl-codegen] [leka-rk3588-contract]
- For GEMV-shaped inner reductions, the pinned DLight source has an explicit
  Mali branch: it keeps warp shuffle disabled, avoids staging the vector in
  shared memory, partitions the reduction across work-items, and vectorizes
  only a legal divisor of the contiguous extent. These are reusable mechanism
  choices; its fixed factors are upstream defaults, not RK3588 measurements.
  [tvm-dlight-mali-gemv] [tvm-dlight-mali-low-batch-gemv]
- For matrix-product-shaped PrimFuncs, this target does not select TVM's
  Adreno-specific configuration. The generic OpenCL rule demonstrates tiled
  output accumulation, optional cooperative shared reads, local write-back,
  padding, and aligned vector access. Derive all factors from the actual shape,
  local-memory footprint, register pressure, and measured launch topology.
  [tvm-dlight-opencl-matmul] [leka-rk3588-contract]
- For a two-axis transpose, the transferable mechanism is a padded shared tile
  that makes both global reads and writes contiguous. For reductions, retain
  per-work-item partials and use the minimum synchronization structure required
  by the exact reduction and epilogue. Upstream OpenCL tile/thread constants are
  references, not device-optimal values. [tvm-dlight-gpu-transpose]
  [tvm-dlight-gpu-reduction]
- Producer-consumer work may be scheduled together inside an existing PrimFunc
  or one declared closed 2–3-call fusion under `leka.l1_fuse_of`. Global fusion
  and static-parameter relayout remain L0-only; keep external interfaces and
  unrelated graph/ABIs unchanged. [leka-rk3588-contract]

## Backend discriminator: Mali is not Adreno

- In TVM 0.23's matmul rule, the Adreno branch is selected when the OpenCL host
  contains `android` or target attributes contain `adreno`. The RK3588 target's
  host is `aarch64-linux-gnu`, so it remains on the Mali/generic OpenCL path.
  Keep that host contract unchanged. [tvm-dlight-opencl-matmul]
  [leka-rk3588-contract]
- Adreno DLight convolution/layout rules and the Relax CLML pipeline are
  incompatible reference material for this backend. Do not copy their image,
  texture, layout, extension, or tile assumptions into Mali code.
  [tvm-dlight-adreno] [tvm-023-adreno-relax]

## Applicability and promotion

- Tag every mechanism with `opencl`, `mali`, Linux host triple, exact TVM
  commit, dtype, fixed/symbolic shape predicate, alignment, workgroup shape,
  local-memory use, and required extensions. An omitted extension is not an
  implicit capability. [leka-rk3588-contract]
- Parser round-trip proves syntax only. Promote a mechanism only after exact
  OpenCL target compilation, independent correctness, and measurement on the
  physical board. [tvm-023-roundtrip] [leka-rk3588-contract]

## Source IDs

- [tvm-023-release]: [Apache TVM v0.23.0, commit `75ce1b4`](https://github.com/apache/tvm/releases/tag/v0.23.0)
- [tvm-023-tir-parser]: [Pinned TIR parser entry](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/parser/tir/entry.py)
- [tvm-023-tir-builder]: [Pinned TIR script builder](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/ir_builder/tir/ir.py)
- [tvm-023-relax-builder]: [Pinned Relax script builder](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/ir_builder/relax/ir.py)
- [tvm-023-call-tir]: [Pinned `R.call_tir` signature](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/relax/op/base.py)
- [tvm-023-call-tir-rewrite]: [Pinned call-TIR ABI rewrite](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/src/relax/transform/call_tir_rewrite.cc)
- [tvm-023-opencl-codegen]: [Pinned OpenCL address-space lowering](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/src/target/source/codegen_opencl.cc)
- [tvm-023-roundtrip]: [Pinned TVMScript round-trip tests](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/tests/python/tvmscript/test_tvmscript_roundtrip.py)
- [tvm-dlight-mali-gemv]: [Pinned GPU GEMV rule and Mali branch](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/dlight/gpu/gemv.py)
- [tvm-dlight-mali-low-batch-gemv]: [Pinned low-batch GEMV Mali branch](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/dlight/gpu/low_batch_gemv.py)
- [tvm-dlight-opencl-matmul]: [Pinned GPU matmul target selection](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/dlight/gpu/matmul.py)
- [tvm-dlight-gpu-transpose]: [Pinned GPU transpose rule](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/dlight/gpu/transpose.py)
- [tvm-dlight-gpu-reduction]: [Pinned GPU reduction rule](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/dlight/gpu/general_reduction.py)
- [tvm-dlight-adreno]: [Pinned Adreno-only DLight rules](https://github.com/apache/tvm/tree/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/dlight/adreno)
- [tvm-023-adreno-relax]: [Pinned Adreno Relax backend](https://github.com/apache/tvm/tree/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/relax/backend/adreno)
- [leka-rk3588-contract]: [LEKA RK3588 backend contract](../../sources/rk3588_gpu/backend_contract.json)

Evidence key: `rk3588_gpu/opt_patterns`.
