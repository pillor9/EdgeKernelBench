---
id: jetson_gpu/opt_patterns
hardware: jetson_gpu (Orin integrated Ampere, CUDA)
tags: [fold, launches, access, tiling, reduction]
layers: [L1]
evidence_basis: [tvm_source, backend_contract]
version_sensitivity: high -- exact TVM 0.23.0 syntax and CUDA lowering contract
---

# Jetson GPU: TVM 0.23 syntax and mechanism contract

Ownership boundary: L0/T01 remain immutable canonical references. L1 may
reschedule existing PrimFuncs and jointly fuse one closed region of 2–3
consecutive pure single-output calls under the `leka.l1_fuse_of` contract.
All unrelated graph structure and nonregion ABIs stay frozen; broader graph,
algorithm, or fixed-state changes go to L0. The fixed
`LegalizeOps/FuseTIR(explicit Primitive helpers only)/DeadCodeElimination`
chain and the separate DLight executable seed are translation/seed references,
never candidate APIs or graph/fusion policy.

## Version lock and model-visible syntax

- The only syntax authority for this page is Apache TVM `0.23.0` at commit
  `75ce1b4d46053b28fa926a6e29aa715b2463dc92`. Unversioned TVM web pages may
  describe a newer dialect and are not admissible evidence for a candidate.
  Candidate syntax and compilation must match the configured TVM toolchain.
  [tvm-023-release] [tvm-compatibility]
- Reference snippets may omit imports for brevity. Every complete candidate
  file must include `import tvm` and its required legacy aliases from
  `tvm.script` (`ir as I`, `tir as T`, and `relax as R`). The worker does not
  inject these imports. Do not use the newer `tirx` dialect.
  [tvm-023-tir-parser]
- A mixed module uses `@I.ir_module`, `@T.prim_func(private=True)`,
  `with T.block("name")`, and `@R.function`. Allocate scratch with
  `T.alloc_buffer`, and bind axes with `T.axis.spatial`, `T.axis.reduce`, or
  `T.axis.remap`. [tvm-023-tir-tutorial] [tvm-023-tir-builder]
- Relax calls TIR with
  `R.call_tir(callee, args, out_sinfo=R.Tensor(...))`. A list of tensor struct
  infos denotes multiple outputs; `R.Tensor((), dtype=...)` denotes a scalar
  tensor. Shape/scalar TIR arguments use `tir_vars=R.shape([...])` and appear
  after output buffers in the PrimFunc ABI. [tvm-023-call-tir]
  [tvm-023-call-tir-rewrite]
- Construct a Relax primitive scalar with `R.prim_value(...)`. Build compound
  TIR predicates with `T.And`/`T.Or`; `T.Cast(dtype, value)`,
  `T.cast(value, dtype)`, `T.Select`, and `T.if_then_else` are in the pinned
  surface. [tvm-023-relax-builder] [tvm-023-tir-builder]

## Candidate-legal CUDA mechanisms

- Map independent output tiles to `blockIdx.*`, CUDA lanes to `threadIdx.*`,
  per-thread accumulators to `scope="local"`, and genuinely reused cooperative
  tiles to `scope="shared"`. Every `T.tvm_storage_sync("shared")` must be
  reached uniformly by the whole block. Exact extents must remain within the
  injected backend card; the Wiki does not replace those live limits.
  [tvm-023-tir-builder] [leka-jetson-contract]
- For a matrix-product-shaped PrimFunc, the reusable upstream mechanism is:
  normalize the spatial/reduction axes, tile both output axes and the reduction
  axis, accumulate an output micro-tile in registers, cooperatively fetch
  contiguous operand tiles, and vectorize only an aligned divisible inner
  access. Treat tile shapes, vector widths, unroll depths, and shared-memory
  layouts as search variables constrained by the current shape and resource
  report. [tvm-dlight-cuda-matmul]
- TVM's CUDA tensorization branch is a reference-only mechanism. It is selected
  upstream only for CUDA architectures at least `sm_70`, sufficiently large
  normalized matrix axes, and supported input/output dtype pairs. The current
  backend card exposes no registered tensor intrinsic, and the candidate
  contract forbids `tir.Schedule`, DLight, and MetaSchedule, so do not emit WMMA
  or tensorization calls merely because the physical target is `sm_87`.
  [tvm-dlight-cuda-matmul] [tvm-023-cuda-intrin]
  [leka-jetson-contract]
- For reductions, keep independent partials in registers, reduce within a
  block, and materialize cross-thread state only when reuse requires shared
  memory. An injective epilogue may be scheduled together only when it is
  already inside the same canonical PrimFunc. Choose the reduction
  decomposition from the actual extent rather than copying an upstream thread
  count. [tvm-dlight-gpu-reduction]
- For a two-axis transpose, the safe transferable mechanism is a padded shared
  tile with coalesced reads and writes. The upstream CUDA tile dimensions are
  evidence that this mechanism exists, not a device-optimal constant for every
  dtype or shape. [tvm-dlight-gpu-transpose]
- Static parameter relayout and global fusion are L0-only choices. L1 may
  co-design one declared closed 2–3-call fusion and TIR schedule, but cannot
  change fixed-state plans, unrelated graph structure, or nonregion ABIs.
  [leka-jetson-contract]

## Applicability and promotion

- Tag each proposal with backend `cuda`, architecture `sm_87`, exact TVM
  commit, dtype tuple, symbolic/fixed shape predicate, alignment predicate,
  storage use, and expected launch topology. A mechanism with a false shape or
  dtype predicate is ineligible rather than merely low confidence.
  [leka-jetson-contract]
- `tvm.script.from_source` parse plus structural round-trip establishes only
  syntax. Promotion still requires target compilation, independent correctness,
  and physical-device timing/profile evidence for the exact candidate. An
  upstream DLight rule by itself is never a speed claim. [tvm-023-roundtrip]
  [leka-jetson-contract]

## Source IDs

- [tvm-023-release]: [Apache TVM v0.23.0, commit `75ce1b4`](https://github.com/apache/tvm/releases/tag/v0.23.0)
- [tvm-023-tir-parser]: [TVM 0.23 TIR parser entry](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/parser/tir/entry.py)
- [tvm-023-tir-builder]: [TVM 0.23 TIR script builder surface](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/ir_builder/tir/ir.py)
- [tvm-023-relax-builder]: [TVM 0.23 Relax script builder surface](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/script/ir_builder/relax/ir.py)
- [tvm-023-tir-tutorial]: [Pinned TensorIR creation tutorial](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/docs/deep_dive/tensor_ir/tutorials/tir_creation.py)
- [tvm-023-call-tir]: [Pinned `R.call_tir` signature](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/relax/op/base.py)
- [tvm-023-call-tir-rewrite]: [Pinned call-TIR output/scalar ABI rewrite](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/src/relax/transform/call_tir_rewrite.cc)
- [tvm-023-cuda-intrin]: [Pinned CUDA tensor intrinsics](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/tir/tensor_intrin/cuda.py)
- [tvm-023-roundtrip]: [Pinned TVMScript round-trip tests](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/tests/python/tvmscript/test_tvmscript_roundtrip.py)
- [tvm-dlight-cuda-matmul]: [Pinned GPU matmul rule](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/dlight/gpu/matmul.py)
- [tvm-dlight-gpu-reduction]: [Pinned GPU reduction rules](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/dlight/gpu/general_reduction.py)
- [tvm-dlight-gpu-transpose]: [Pinned GPU transpose rule](https://github.com/apache/tvm/blob/75ce1b4d46053b28fa926a6e29aa715b2463dc92/python/tvm/dlight/gpu/transpose.py)
- [leka-jetson-contract]: [LEKA Jetson backend contract](../../sources/jetson_gpu/backend_contract.json)

Evidence key: `jetson_gpu/opt_patterns`.
