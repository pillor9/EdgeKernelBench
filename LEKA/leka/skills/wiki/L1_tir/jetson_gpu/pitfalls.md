---
id: jetson_gpu/pitfalls
hardware: jetson_gpu (Orin integrated Ampere, CUDA)
tags: [compile-error, lowering, abi, numerics]
layers: [L1]
evidence_basis: [tvm_source, backend_contract]
version_sensitivity: high -- exact TVM 0.23.0 parser and CUDA codegen failures
---

# Jetson GPU: pinned failures and forbidden migrations

Relax syntax below is diagnostic ABI context only unless used for the explicit
`leka.l1_fuse_of` bounded local-fusion contract: one closed 2–3-call canonical
chain and its fused TIR. Every other Relax function/call remains read-only.
The local exception does not authorize an algorithm, layout, or whole-graph edit.

## Version and API failures

- Do not ingest the unversioned TVM documentation into this compiler contract.
  In the pinned `0.23.0` build, import `tvm.script.tir as T`, use `T.block`,
  `T.alloc_buffer`, and `R.call_tir(..., out_sinfo=...)`. The newer spellings
  `tirx`, `T.sblock`, `T.sblock_alloc_buffer`, `@T.prim_func(s_tir=True)`, and
  `out_ty=` fail this parser. Project Wiki fragments intentionally omit imports;
  that omission is not permission to infer the newer namespace.
  [tvm-023-release] [tvm-023-tir-parser] [tvm-023-call-tir]
- `R.PrimValue(...)` is not a TVMScript constructor in this release; use
  `R.prim_value(...)`. `T.all` and `T.any` are absent; use nested `T.And` and
  `T.Or`. These are API failures, not alternative argument conventions.
  [tvm-023-relax-builder] [tvm-023-tir-builder]
- A `call_tir` callee must be a GlobalVar in the same IRModule. Its tensor input
  buffers precede output buffers, and any unpacked `tir_vars` scalar arguments
  follow the outputs. A list of `out_sinfo` entries creates a tuple result; a
  zero-rank output is `R.Tensor((), dtype=...)`, not a Python scalar.
  [tvm-023-call-tir] [tvm-023-call-tir-rewrite]

## CUDA lowering failures

- A device buffer access outside a GPU thread environment is rejected by the
  pinned `VerifyMemory` pass. Bind the launch loops before the first such
  access; a successful syntax parse does not establish device legality.
  [tvm-023-verify-memory]
- CUDA codegen rejects function-local allocation in `scope="global"`; global
  tensors must enter as PrimFunc buffers. Use `local` for per-thread scratch and
  `shared` only for block-cooperative scratch. `T.alloc_buffer` must be created
  in a PrimFunc or block frame. [tvm-023-cuda-codegen]
  [tvm-023-tir-builder]
- A shared-memory barrier beneath thread-divergent control is invalid because
  not every thread reaches the same synchronization point. Predicate loads and
  stores while keeping the barrier itself block-uniform.
  [tvm-023-tir-builder]
- Never copy a fixed block shape, vector width, unroll factor, or shared tile
  from an upstream DLight rule. Those constants are schedule-rule defaults, not
  measurements for this candidate; they may violate the live resource card or
  lose on a different shape. [tvm-dlight-cuda-matmul]
  [leka-jetson-contract]
- Do not transfer OpenCL address-space names, Mali execution-group assumptions,
  Adreno image/texture paths, or CPU `T.parallel` loops into a CUDA PrimFunc.
  CUDA legality is defined by the exact CUDA target and the pinned lowering
  pipeline. [leka-jetson-contract]
- Do not hand-author WMMA merely because `sm_87` supports tensor cores. The
  model-visible backend facts currently expose no registered tensor intrinsic,
  and candidate-side Schedule/DLight/MetaSchedule use remains forbidden.
  [tvm-023-cuda-intrin] [leka-jetson-contract]

## Semantic and evidence failures

- Changing accumulation order, narrowing the accumulator, replacing an IEEE
  infinity with a finite sentinel, or omitting a tail is a semantic change until
  independent correctness proves otherwise. A plausible CUDA optimization is
  not evidence of equivalence. [leka-jetson-contract]
- Parser success, host compilation, an upstream rule, or a profiler hypothesis
  is not a performance result. Only the exact target compile, independent
  correctness path, and physical-device measurement can promote a mechanism.
  [tvm-023-roundtrip] [leka-jetson-contract]

## Source IDs

- [tvm-023-release]: [Apache TVM v0.23.0](https://github.com/apache/tvm/releases/tag/v0.23.0)
- [tvm-023-tir-parser]: [Pinned TIR parser entry](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/parser/tir/entry.py)
- [tvm-023-tir-builder]: [Pinned TIR script builder](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/ir_builder/tir/ir.py)
- [tvm-023-relax-builder]: [Pinned Relax script builder](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/ir_builder/relax/ir.py)
- [tvm-023-call-tir]: [Pinned `R.call_tir` signature](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/relax/op/base.py)
- [tvm-023-call-tir-rewrite]: [Pinned call-TIR ABI rewrite](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/src/relax/transform/call_tir_rewrite.cc)
- [tvm-023-verify-memory]: [Pinned GPU memory verifier](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/src/tir/analysis/verify_memory.cc)
- [tvm-023-cuda-codegen]: [Pinned CUDA allocation lowering](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/src/target/source/codegen_cuda.cc)
- [tvm-023-cuda-intrin]: [Pinned CUDA tensor intrinsics](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/tir/tensor_intrin/cuda.py)
- [tvm-023-roundtrip]: [Pinned TVMScript round-trip tests](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/tests/python/tvmscript/test_tvmscript_roundtrip.py)
- [tvm-dlight-cuda-matmul]: [Pinned GPU matmul rule](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/dlight/gpu/matmul.py)
- [leka-jetson-contract]: [LEKA Jetson backend contract](../../sources/jetson_gpu/backend_contract.json)

Evidence key: `jetson_gpu/pitfalls`.
