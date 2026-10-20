---
id: rk3588_gpu/pitfalls
hardware: rk3588_gpu (Mali-G610 Valhall, OpenCL)
tags: [compile-error, lowering, abi, numerics]
layers: [L1]
evidence_basis: [tvm_source, backend_contract]
version_sensitivity: high -- exact TVM 0.23.0 parser, OpenCL target selection, and lowering failures
---

# RK3588 GPU: pinned failures and forbidden migrations

Relax syntax below is diagnostic ABI context only unless used for the explicit
`leka.l1_fuse_of` bounded local-fusion contract: one closed 2–3-call canonical
chain and its fused TIR. Every other Relax function/call remains read-only.
The local exception does not authorize an algorithm, layout, or whole-graph edit.

## Version and API failures

- The compiler is TVM `0.23.0` at
  `75ce1b4d46053b28fa926a6e29aa715b2463dc92`. Use legacy
  `tvm.script.tir as T`, `T.block`, `T.alloc_buffer`, and
  `R.call_tir(..., out_sinfo=...)`. The unversioned site's `tirx`, `T.sblock`,
  `T.sblock_alloc_buffer`, `s_tir=True`, and `out_ty=` belong to a different
  dialect and fail here. Wiki fragments omit imports intentionally; do not fill
  the omission with the newer namespace. [tvm-023-release]
  [tvm-023-tir-parser] [tvm-023-call-tir]
- Use `R.prim_value(...)`, not `R.PrimValue(...)`. `T.all` and `T.any` are
  absent; use `T.And` and `T.Or`. [tvm-023-relax-builder]
  [tvm-023-tir-builder]
- A `call_tir` callee lives in the same IRModule. Input buffers precede output
  buffers, and `tir_vars` scalar arguments follow the outputs. Multiple
  `out_sinfo` entries produce a tuple; `R.Tensor((), dtype=...)` is a zero-rank
  tensor. [tvm-023-call-tir] [tvm-023-call-tir-rewrite]

## OpenCL and target-classification failures

- TVM's script storage name for per-work-item scratch is `scope="local"`; its
  name for workgroup-shared scratch is `scope="shared"`. Do not write OpenCL C
  address-space keywords as TVMScript storage scopes. [tvm-023-opencl-codegen]
  [tvm-023-tir-builder]
- Device buffer access outside `blockIdx.*`/`threadIdx.*` launch loops is
  rejected by `VerifyMemory`. Work-item extents and their product must satisfy
  the injected target; parser success does not prove launch legality.
  [tvm-023-verify-memory] [leka-rk3588-contract]
- Keep the RK3588 host as `aarch64-linux-gnu`. In the pinned matmul rule, merely
  changing an OpenCL target's host to Android causes the Adreno condition to
  match before the Mali/default path. Such a target change is not an
  optimization: it silently selects the wrong backend mechanism.
  [tvm-dlight-opencl-matmul] [leka-rk3588-contract]
- Never import Adreno image/texture layouts, CLML partitions, Qualcomm
  extensions, or Adreno tile constants into Mali code. The device name, target
  keys, host triple, runtime extensions, and lowering pipeline are different.
  [tvm-dlight-adreno] [tvm-023-adreno-relax]
- Never import CUDA warp shuffle, WMMA, `sm_*`, CUDA vector types, or warp-size
  assumptions. A target field such as `thread_warp_size=1` is a compiler fact,
  not proof that the physical Mali execution group has scalar width.
  [leka-rk3588-contract]
- Shared memory is not automatically faster than direct global reads. Use it
  only with demonstrated intra-workgroup reuse, include every buffer and
  alignment pad in the footprint, and keep barriers workgroup-uniform.
  [tvm-dlight-mali-gemv] [tvm-023-opencl-codegen]
- Do not hard-code a vector width, workgroup shape, unroll depth, or local tile
  merely because it appears in DLight. The upstream constants are rule defaults
  and must be filtered by dtype, alignment, divisibility, resource limits, and
  measurement on this board. [tvm-dlight-mali-gemv]
  [tvm-dlight-opencl-matmul] [leka-rk3588-contract]

## Semantic and evidence failures

- Narrowing an accumulator, omitting a tail, changing reduction order, or
  replacing an IEEE infinity with a finite sentinel can change results. Keep
  the declared dtype and exact output tree until independent correctness shows
  the alternative is valid. [leka-rk3588-contract]
- Parser round-trip, OpenCL source generation, or an upstream schedule does not
  establish a speedup. Promotion requires exact target compilation, independent
  correctness, and physical-board timing. [tvm-023-roundtrip]
  [leka-rk3588-contract]

## Source IDs

- [tvm-023-release]: [Apache TVM v0.23.0](https://github.com/apache/tvm/releases/tag/v0.23.0)
- [tvm-023-tir-parser]: [Pinned TIR parser entry](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/parser/tir/entry.py)
- [tvm-023-tir-builder]: [Pinned TIR script builder](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/ir_builder/tir/ir.py)
- [tvm-023-relax-builder]: [Pinned Relax script builder](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/ir_builder/relax/ir.py)
- [tvm-023-call-tir]: [Pinned `R.call_tir` signature](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/relax/op/base.py)
- [tvm-023-call-tir-rewrite]: [Pinned call-TIR ABI rewrite](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/src/relax/transform/call_tir_rewrite.cc)
- [tvm-023-opencl-codegen]: [Pinned OpenCL address-space lowering](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/src/target/source/codegen_opencl.cc)
- [tvm-023-verify-memory]: [Pinned GPU memory verifier](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/src/tir/analysis/verify_memory.cc)
- [tvm-023-roundtrip]: [Pinned TVMScript round-trip tests](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/tests/python/tvmscript/test_tvmscript_roundtrip.py)
- [tvm-dlight-mali-gemv]: [Pinned GPU GEMV Mali branch](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/dlight/gpu/gemv.py)
- [tvm-dlight-opencl-matmul]: [Pinned GPU matmul target selection](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/dlight/gpu/matmul.py)
- [tvm-dlight-adreno]: [Pinned Adreno-only DLight rules](https://github.com/apache/tvm/tree/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/dlight/adreno)
- [tvm-023-adreno-relax]: [Pinned Adreno Relax backend](https://github.com/apache/tvm/tree/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/relax/backend/adreno)
- [leka-rk3588-contract]: [LEKA RK3588 backend contract](../../sources/rk3588_gpu/backend_contract.json)

Evidence key: `rk3588_gpu/pitfalls`.
